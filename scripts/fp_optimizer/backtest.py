"""Walk-forward backtest with rolling re-estimation, rebalancing, and costs.

Extends the lightweight single-split comparison in ``verify.py`` to a full
(but still laptop-sized) historical simulation:

    |---- train_window ----|
                           ^ rebalance day t: re-estimate (mu, B, F, D) from
                             ONLY the trailing window, re-solve the QP,
                             trade to the target weights, pay bps costs
                              |---- hold (weights drift with returns) ----|
                                                                   ^ next rebalance

Information timing (no lookahead): the decision on day t uses returns up to
day t-1 inclusive; day t's return is then applied to the NEW weights.

All variants (optimized / unregularized / equal-weight) share the same
schedule, the same estimation windows, the same mu estimator (historical
mean -- noisy, but identical across variants so the comparison is fair),
and the same cost model. The benchmark is buy-and-hold of ``w_b`` with
natural drift and no costs (an index does not pay your trading costs).

Declared failure rule: if a QP solve fails at a rebalance date, the portfolio
HOLDS its current (drifted) weights and the failure is counted and reported
-- never silently patched.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import covariance as cov
from .config import OptimizerConfig
from .data import PortfolioInputs, build_inputs_from_returns, equal_weight
from .optimizer import solve_factor_portfolio


# ----------------------------------------------------------------------
# Configuration / results
# ----------------------------------------------------------------------
@dataclass
class BacktestConfig:
    train_window: int = 504        # trading days used per re-estimation
    rebalance_freq: int = 63       # trading days between rebalances (~quarterly)
    cost_bps: float = 10.0         # one-way cost, bps of traded notional
    min_train_obs: int = 252       # refuse to trade on thinner windows
    periods_per_year: int = 252
    rf: float = 0.0                # risk-free rate for Sharpe (documented: 0)
    subperiod_years: int = 3       # calendar-chunk length for sub-period table

    def validate(self) -> None:
        if self.train_window < self.min_train_obs:
            raise ValueError(
                f"train_window ({self.train_window}) < min_train_obs "
                f"({self.min_train_obs})")
        if self.rebalance_freq < 1:
            raise ValueError(f"rebalance_freq must be >= 1, got {self.rebalance_freq}")
        if self.cost_bps < 0:
            raise ValueError(f"cost_bps must be >= 0, got {self.cost_bps}")
        if int(self.subperiod_years) < 1:
            raise ValueError(f"subperiod_years must be >= 1, got {self.subperiod_years}")


@dataclass
class VariantResult:
    name: str
    equity: pd.Series
    metrics: Dict
    weights_history: pd.DataFrame       # rebalance dates x tickers
    turnover_log: List[Dict]
    n_rebalances: int
    n_solver_failures: int
    note: str = ""


@dataclass
class BacktestReport:
    data_label: str
    provenance: Dict
    bt_config: Dict
    summary: pd.DataFrame               # one row per variant + benchmark
    equity_curves: pd.DataFrame         # dates x variants
    variants: Dict[str, VariantResult]
    benchmark_equity: pd.Series
    subperiods: Optional[pd.DataFrame] = None   # fixed-rule sub-period table


# ----------------------------------------------------------------------
# Mechanics
# ----------------------------------------------------------------------
def max_drawdown(wealth) -> float:
    """Largest percentage fall from a running peak (course T2 definition)."""
    wealth = np.asarray(wealth, dtype=float)
    peak = np.maximum.accumulate(wealth)
    dd = wealth / peak - 1.0
    return float(dd.min())


def _drift_weights(w: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Buy-and-hold weight drift over one period: w_i(1+r_i) / sum_j w_j(1+r_j)."""
    grown = w * (1.0 + r)
    s = float(grown.sum())
    if s <= 1e-12:            # portfolio wiped out; freeze (equity shows the loss)
        return grown
    return grown / s


def performance_metrics(equity: pd.Series, bench_equity: pd.Series,
                        periods_per_year: int = 252, rf: float = 0.0) -> Dict:
    """Realized performance from a wealth path (both start at 100)."""
    rets = equity.pct_change().dropna()
    bench_rets = bench_equity.pct_change().dropna()
    n = len(equity)
    years = max(n / periods_per_year, 1e-12)
    total = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)
    vol = float(rets.std(ddof=1) * np.sqrt(periods_per_year)) if len(rets) else float("nan")
    sharpe = (float(rets.mean()) * periods_per_year - rf) / vol if vol and vol > 0 else float("nan")
    mdd = max_drawdown(equity.values)
    aligned = pd.concat([rets, bench_rets], axis=1, join="inner")
    active = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    te = float(active.std(ddof=1) * np.sqrt(periods_per_year)) if len(active) else float("nan")
    active_geo = float(equity.iloc[-1] / bench_equity.iloc[-1] - 1.0)
    return {
        "total_return": total,
        "cagr": cagr,
        "ann_vol": vol,
        "sharpe_rf0": sharpe,
        "max_drawdown": mdd,
        "tracking_error_vs_bench": te,
        "active_return_geo_vs_bench": active_geo,
    }


# ----------------------------------------------------------------------
# Walk-forward engine
# ----------------------------------------------------------------------
def _run_variant(name: str,
                 asset_returns: pd.DataFrame,
                 factor_returns: pd.DataFrame,
                 w_b: pd.Series,
                 cfg_opt: OptimizerConfig,
                 bt: BacktestConfig,
                 mode: str) -> VariantResult:
    """mode: 'qp_shrunk' | 'qp_unregularized' | 'equal_weight' | 'benchmark'."""
    dates = asset_returns.index
    T = len(dates)
    tickers = list(asset_returns.columns)
    n = len(tickers)
    w_b_vec = w_b.reindex(tickers).to_numpy(float)

    w = w_b_vec.copy()          # every variant starts AT the benchmark
    equity = 100.0
    equity_path = np.empty(T)
    weights_rows: Dict = {}
    turnover_log: List[Dict] = []
    n_reb = 0
    n_fail = 0
    notes: List[str] = []
    cost_rate = bt.cost_bps / 1e4

    for t in range(T):
        r_t = asset_returns.iloc[t].to_numpy(float)

        # --- rebalance decision at the start of day t (info up to t-1) ---
        is_reb_day = (
            mode != "benchmark"
            and t >= bt.train_window
            and (t - bt.train_window) % bt.rebalance_freq == 0
        )
        if is_reb_day:
            n_reb += 1
            w_target = None
            if mode == "equal_weight":
                w_target = equal_weight(n)
            else:
                train_ar = asset_returns.iloc[t - bt.train_window:t]
                train_fr = factor_returns.iloc[t - bt.train_window:t]
                try:
                    pi_t, _ = build_inputs_from_returns(
                        train_ar, train_fr,
                        benchmark_weights=w_b,
                        min_obs=bt.min_train_obs,
                        periods_per_year=bt.periods_per_year)
                    if mode == "qp_shrunk":
                        cfg_t = dataclasses.replace(cfg_opt, previous_weights=list(w))
                        res = solve_factor_portfolio(pi_t, cfg_t)
                    elif mode == "qp_unregularized":
                        Sigma_s = cov.sample_covariance(train_ar, bt.periods_per_year)
                        cfg_t = dataclasses.replace(
                            cfg_opt, shrinkage_intensity=0.0, ridge_epsilon=0.0,
                            previous_weights=list(w))
                        res = solve_factor_portfolio(
                            pi_t, cfg_t, sigma_override=Sigma_s,
                            sigma_label="rolling raw sample covariance (unregularized)")
                    else:
                        raise ValueError(f"unknown mode {mode!r}")
                    if res.success and res.weights is not None:
                        w_target = np.asarray(res.weights, float)
                    else:
                        n_fail += 1
                        notes.append(f"{dates[t].date()}: solve {res.status}, held weights")
                except Exception as exc:   # declared rule: hold, count, report
                    n_fail += 1
                    notes.append(f"{dates[t].date()}: {type(exc).__name__}: {exc}")

            if w_target is not None:
                turnover = float(np.abs(w_target - w).sum())
                cost_drag = cost_rate * turnover
                equity *= (1.0 - cost_drag)
                w = w_target
                weights_rows[str(dates[t].date())] = w.copy()
                turnover_log.append({"date": str(dates[t].date()),
                                     "turnover": turnover,
                                     "cost_drag": cost_drag})

        # --- apply day t's return to the (possibly new) weights ---
        equity *= float(1.0 + w @ r_t)
        w = _drift_weights(w, r_t)
        equity_path[t] = equity

    eq = pd.Series(equity_path, index=dates, name=name)
    weights_history = pd.DataFrame(weights_rows, index=tickers).T \
        if weights_rows else pd.DataFrame(index=[], columns=tickers)
    wts = weights_history
    if not wts.empty:
        wts.index.name = "rebalance_date"

    avg_to = (float(np.mean([r["turnover"] for r in turnover_log]))
              if turnover_log else float("nan"))
    total_cost = float(np.sum([r["cost_drag"] for r in turnover_log])) if turnover_log else 0.0
    return (VariantResult(
        name=name, equity=eq,
        metrics={},  # filled by run_backtest once the benchmark path exists
        weights_history=wts,
        turnover_log=turnover_log,
        n_rebalances=n_reb, n_solver_failures=n_fail,
        note="; ".join(notes[:5]) + (" ..." if len(notes) > 5 else "")),
        avg_to, total_cost)


def run_backtest(asset_returns: pd.DataFrame,
                 factor_returns: pd.DataFrame,
                 benchmark_weights: pd.Series,
                 cfg_opt: Optional[OptimizerConfig] = None,
                 bt: Optional[BacktestConfig] = None,
                 *,
                 data_label: str = "",
                 provenance: Optional[Dict] = None) -> BacktestReport:
    """Run all variants on one common dataset and assemble the report."""
    bt = bt if bt is not None else BacktestConfig()
    bt.validate()
    cfg_opt = cfg_opt if cfg_opt is not None else OptimizerConfig()
    if len(asset_returns) <= bt.train_window + bt.rebalance_freq:
        raise ValueError(
            f"need more than train_window + rebalance_freq observations "
            f"({bt.train_window + bt.rebalance_freq}); got {len(asset_returns)}")

    common = asset_returns.index.intersection(factor_returns.index)
    asset_returns = asset_returns.loc[common]
    factor_returns = factor_returns.loc[common]
    w_b = benchmark_weights.reindex(asset_returns.columns)
    if w_b.isna().any():
        raise ValueError("benchmark_weights missing tickers present in asset_returns")

    modes = [("benchmark_buy_hold", "benchmark"),
             ("optimized_qp_shrunk", "qp_shrunk"),
             ("unregularized_sample_cov", "qp_unregularized"),
             ("equal_weight_1_over_N", "equal_weight")]

    results: Dict[str, Tuple[VariantResult, float, float]] = {}
    for name, mode in modes:
        vr, avg_to, total_cost = _run_variant(
            name, asset_returns, factor_returns, w_b, cfg_opt, bt, mode)
        results[name] = (vr, avg_to, total_cost)

    bench_eq = results["benchmark_buy_hold"][0].equity
    rows = []
    curves = {}
    for name, _ in modes:
        vr, avg_to, total_cost = results[name]
        vr.metrics = performance_metrics(vr.equity, bench_eq,
                                         bt.periods_per_year, bt.rf)
        curves[name] = vr.equity
        m = vr.metrics
        rows.append({
            "variant": name,
            "total_return": round(m["total_return"], 4),
            "cagr": round(m["cagr"], 4),
            "ann_vol": round(m["ann_vol"], 4),
            "sharpe_rf0": round(m["sharpe_rf0"], 3) if np.isfinite(m["sharpe_rf0"]) else None,
            "max_drawdown": round(m["max_drawdown"], 4),
            "tracking_error_vs_bench": round(m["tracking_error_vs_bench"], 4),
            "active_return_geo": round(m["active_return_geo_vs_bench"], 4),
            "avg_turnover_per_rebal": round(avg_to, 4) if np.isfinite(avg_to) else None,
            "total_cost_drag": round(total_cost, 5),
            "n_rebalances": vr.n_rebalances,
            "n_solver_failures": vr.n_solver_failures,
        })
    summary = pd.DataFrame(rows)
    equity_curves = pd.DataFrame(curves)

    # out-of-sample span starts at the first rebalance date (index position
    # train_window); warm-up days are identical across variants by design
    test_start = equity_curves.index[min(bt.train_window, len(equity_curves) - 1)]
    subperiods = subperiod_breakdown(
        equity_curves, test_start, years=bt.subperiod_years,
        periods_per_year=bt.periods_per_year)

    report = BacktestReport(
        data_label=data_label or "unlabeled",
        provenance=provenance or {},
        bt_config=dataclasses.asdict(bt),
        summary=summary,
        equity_curves=equity_curves,
        variants={name: results[name][0] for name, _ in modes},
        benchmark_equity=bench_eq,
        subperiods=subperiods,
    )
    return report


# ----------------------------------------------------------------------
# Sub-period breakdown (fixed calendar-chunk rule)
# ----------------------------------------------------------------------
_VARIANT_SHORT = {
    "benchmark_buy_hold": "bench",
    "optimized_qp_shrunk": "qp",
    "unregularized_sample_cov": "unreg",
    "equal_weight_1_over_N": "ew",
}


def subperiod_breakdown(equity_curves: pd.DataFrame, test_start,
                        years: int = 3,
                        periods_per_year: int = 252) -> pd.DataFrame:
    """Split the OUT-OF-SAMPLE span (first rebalance onward) into consecutive
    calendar-year chunks of ``years`` years and summarize each variant.

    The chunking rule is fixed in advance by ``years`` alone: chunks are
    calendar-year blocks starting at the first rebalance year; only the final
    chunk may be partial. Boundaries are NEVER chosen after seeing results.
    Warm-up days (all variants identical, no trades yet) are excluded.
    """
    idx = equity_curves.index
    oos = equity_curves[idx >= test_start]
    if len(oos) < 2:
        return pd.DataFrame()
    end = oos.index.max()
    y0 = int(oos.index.min().year)

    def _stats(values):
        s = pd.Series(np.asarray(values, dtype=float))
        ret = float(s.iloc[-1] / s.iloc[0] - 1.0)
        r = s.pct_change().dropna()
        vol = float(r.std(ddof=1) * np.sqrt(periods_per_year)) if len(r) > 1 else float("nan")
        dd = float((s / s.cummax() - 1.0).min())
        return ret, vol, dd

    rows: List[Dict] = []
    while True:
        chunk_end = min(end, pd.Timestamp(year=y0 + years - 1, month=12, day=31))
        lo = pd.Timestamp(year=y0, month=1, day=1)
        seg = oos[(oos.index >= lo) & (oos.index <= chunk_end)]
        if len(seg) >= 2:
            row: Dict = {
                "subperiod": f"{y0}-{chunk_end.year}" if chunk_end.year != y0 else str(y0),
                "start": str(seg.index.min().date()),
                "end": str(seg.index.max().date()),
                "n_days": int(len(seg)),
            }
            for col in oos.columns:
                ret, vol, dd = _stats(seg[col].values)
                short = _VARIANT_SHORT.get(col, col)
                row[f"ret_{short}"] = round(ret, 4)
                row[f"vol_{short}"] = round(vol, 4)
                row[f"maxdd_{short}"] = round(dd, 4)
            if "ret_qp" in row and "ret_ew" in row:
                row["qp_vs_ew_active"] = round(
                    (1.0 + row["ret_qp"]) / (1.0 + row["ret_ew"]) - 1.0, 4)
            rows.append(row)
        if chunk_end >= end:
            break
        y0 += years
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Rendering / plotting / persistence
# ----------------------------------------------------------------------
def _df_to_md(df: pd.DataFrame) -> str:
    """Hand-rolled markdown table (avoids the optional 'tabulate' dependency)."""
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = ["" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)
                 for v in row.tolist()]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def render_backtest_markdown(report: BacktestReport) -> str:
    L: List[str] = []
    L.append("# Walk-Forward Backtest Report")
    L.append("")
    L.append(f"- Data: **{report.data_label}**")
    span = f"{report.equity_curves.index[0]} .. {report.equity_curves.index[-1]}"
    L.append(f"- Window: {span}  ·  {len(report.equity_curves)} trading days")
    bc = report.bt_config
    L.append(f"- Train window: {bc['train_window']} days  ·  rebalance every "
             f"{bc['rebalance_freq']} days  ·  cost: {bc['cost_bps']:g} bps one-way "
             f"·  Sharpe rf = {bc['rf']:g}")
    L.append("- Information timing: rebalance on day t uses data up to t-1; "
             "no lookahead. All variants start at the benchmark weights and "
             "share estimation windows, mu estimator (historical mean), "
             "schedule, and costs. Benchmark = buy-and-hold of w_b, no costs.")
    if report.provenance:
        L.append(f"- Provenance: `{report.provenance}`")
    L.append("")
    L.append("## Summary (realized, net of costs)")
    L.append("")
    L.append(_df_to_md(report.summary))
    L.append("")
    if report.subperiods is not None and not report.subperiods.empty:
        sp_cols = [c for c in ("subperiod", "n_days",
                               "ret_bench", "maxdd_bench",
                               "ret_qp", "maxdd_qp",
                               "ret_ew", "maxdd_ew",
                               "qp_vs_ew_active")
                   if c in report.subperiods.columns]
        friendly = {"subperiod": "sub-period", "n_days": "days",
                    "ret_bench": "bench ret", "maxdd_bench": "bench maxDD",
                    "ret_qp": "QP ret", "maxdd_qp": "QP maxDD",
                    "ret_ew": "1/N ret", "maxdd_ew": "1/N maxDD",
                    "qp_vs_ew_active": "QP vs 1/N"}
        L.append(f"## Sub-period breakdown (out-of-sample span, fixed "
                 f"{report.bt_config.get('subperiod_years', 3)}-calendar-year chunks)")
        L.append("")
        L.append(_df_to_md(report.subperiods[sp_cols].rename(columns=friendly)))
        L.append("")
        L.append("*Chunk rule: consecutive calendar-year blocks starting at the"
                 " first-rebalance year; only the final chunk may be partial."
                 " Boundaries are fixed by the rule, never chosen after seeing"
                 " results. Warm-up days (no trades yet) are excluded.*")
        L.append("")
    for name, vr in report.variants.items():
        if name == "benchmark_buy_hold":
            continue
        L.append(f"### {name}")
        L.append(f"- rebalances: {vr.n_rebalances}  ·  solver/estimation failures "
                 f"(held weights): {vr.n_solver_failures}")
        if vr.note:
            L.append(f"- failure notes: {vr.note}")
        L.append("")
    L.append("---")
    L.append("*Descriptive historical simulation, laptop-sized, rules fixed "
             "before running. Underperformance is a valid result. Not "
             "investment advice.*")
    return "\n".join(L)


def maybe_plot(report: BacktestReport, out_png) -> Optional[str]:
    """Equity curves + drawdown chart; returns the path or None if matplotlib missing."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    curves = report.equity_curves
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1.0]})
    for col in curves.columns:
        style = "--" if col == "benchmark_buy_hold" else "-"
        axes[0].plot(curves.index, curves[col], style, linewidth=1.3, label=col)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("wealth (log scale, start = 100)")
    axes[0].set_title(f"Walk-forward backtest — {report.data_label}")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    for col in curves.columns:
        v = curves[col].values
        dd = v / np.maximum.accumulate(v) - 1.0
        axes[1].plot(curves.index, dd * 100.0, style if col == "benchmark_buy_hold" else "-",
                     linewidth=1.0, label=col)
    axes[1].set_ylabel("drawdown (%)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return str(out_png)


def save_backtest(report: BacktestReport, output_dir, *, plot: bool = True) -> Dict[str, str]:
    """Persist summary/equity/turnover/report (+ optional PNG)."""
    from pathlib import Path
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}

    p = out / "backtest_summary.csv"
    report.summary.to_csv(p, index=False)
    paths["summary_csv"] = str(p)

    p = out / "equity_curves.csv"
    report.equity_curves.to_csv(p)
    paths["equity_csv"] = str(p)

    p = out / "backtest_report.md"
    p.write_text(render_backtest_markdown(report), encoding="utf-8")
    paths["report_md"] = str(p)

    if report.subperiods is not None and not report.subperiods.empty:
        p = out / "subperiod_breakdown.csv"
        report.subperiods.to_csv(p, index=False)
        paths["subperiod_csv"] = str(p)

    for name, vr in report.variants.items():
        if vr.turnover_log:
            p = out / f"turnover_{name}.csv"
            pd.DataFrame(vr.turnover_log).to_csv(p, index=False)
            paths[f"turnover_{name}"] = str(p)
        if not vr.weights_history.empty:
            p = out / f"weights_history_{name}.csv"
            vr.weights_history.to_csv(p)
            paths[f"weights_history_{name}"] = str(p)

    if plot:
        png = maybe_plot(report, out / "backtest.png")
        if png:
            paths["chart_png"] = png
    return paths
