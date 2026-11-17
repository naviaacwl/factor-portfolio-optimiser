"""Independent verification suite (course guide, Appendix C).

Checks implemented here, each returning a :class:`CheckRecord` so results can
be tabulated as a compact evaluation record (case / method / expected /
actual / pass-fail / note):

1. ``verify_weights_file``        file-level constraint re-verification:
                                  re-reads the SAVED weights.csv and the RAW
                                  input CSVs and recomputes sum-to-one, box
                                  bounds and active tilts with plain numpy.
                                  Never reuses optimizer internals, so an
                                  optimizer bug cannot validate itself.
2. ``check_small_case_bruteforce``hand-checkable 3-asset / 1-factor instance
                                  solved by exhaustive grid search on the
                                  simplex and compared with the QP.
3. ``check_second_solver``        the same QP re-solved with scipy SLSQP (a
                                  different algorithm and code path) and
                                  compared with CVXPY.
4. ``check_regularization``       shrinkage/ridge properties on a deliberately
                                  near-collinear instance (symmetry, PSD,
                                  condition-number improvement, delta=0 is
                                  identity).
5. ``run_baseline_comparison``    lightweight historical comparison on a
                                  design/test split, rules fixed before the
                                  test window: optimized QP vs Equal-Weight
                                  (1/N) vs unregularized sample covariance.
6. ``check_infeasible_edge_case`` three guaranteed-infeasible constraint sets
                                  must be reported gracefully (status
                                  'infeasible', verdict INFEASIBLE), never crash.
7. ``check_invalid_inputs``       malformed inputs must raise
                                  InputValidationError with actionable messages.

Independence note (per Appendix C): a self-check may repeat the original
error, so checks 1-3 use *different methods* (file re-read + plain numpy,
exhaustive enumeration, a different solver family), not just a rerun of the
same code.
"""
from __future__ import annotations

import copy
import dataclasses
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import covariance as cov
from .compliance import build_compliance_report, save_compliance_report
from .config import ConfigError, OptimizerConfig
from .data import (InputValidationError, PortfolioInputs, build_inputs_from_returns,
                   equal_weight, load_inputs_from_dir, synthetic_inputs,
                   validate_inputs)
from .optimizer import OptimizationResult, objective_value_np, solve_factor_portfolio

# Solver-realistic tolerances. Equality (sum-to-one) is typically satisfied to
# ~1e-8; inequality tilts near a binding bound depend on the solver's eps.
# Tighten by configuring solver="CLARABEL" or solver_kwargs (e.g. OSQP
# eps_abs/eps_rel) and then reducing these tolerances.
DEFAULT_WEIGHT_TOL = 1e-6
DEFAULT_TILT_TOL = 1e-5


@dataclass
class CheckRecord:
    """One row of the compact evaluation record (Appendix C format)."""

    case: str
    method: str
    expected: str
    actual: str
    passed: bool
    note: str = ""


# ----------------------------------------------------------------------
# 1. File-level independent constraint verification
# ----------------------------------------------------------------------
def verify_weights_file(weights_csv, inputs_dir, config_path=None,
                        *, weight_tol: float = DEFAULT_WEIGHT_TOL,
                        tilt_tol: float = DEFAULT_TILT_TOL) -> List[CheckRecord]:
    """Recompute constraint compliance straight from files (independent path).

    Reads the saved ``weights.csv`` and the RAW input CSVs, aligns them, and
    re-derives sum-to-one, box bounds, and every active tilt with plain
    numpy/pandas. This is the "independent constraint verification" evidence:
    weights summing to 1.0 and active factor tilts within bounds.
    """
    records: List[CheckRecord] = []
    weights_csv = Path(weights_csv)
    if not weights_csv.exists():
        return [CheckRecord(
            "file-level constraint verification", f"re-read {weights_csv.name}",
            "file exists", f"MISSING: {weights_csv}", False,
            note="run the optimization first (run_optimization.py)")]

    wdf = pd.read_csv(weights_csv)
    if not {"ticker", "weight"} <= set(wdf.columns):
        return [CheckRecord(
            "file-level constraint verification", f"re-read {weights_csv.name}",
            "columns ticker,weight", f"found {list(wdf.columns)}", False)]

    pi = load_inputs_from_dir(inputs_dir)
    tickers = [str(t) for t in pi.tickers]
    w_tickers = wdf["ticker"].astype(str).tolist()
    if set(w_tickers) != set(tickers) or len(set(w_tickers)) != len(w_tickers):
        records.append(CheckRecord(
            "ticker alignment", "set comparison", "identical ticker sets",
            f"weights:{sorted(set(w_tickers) - set(tickers))} extra / "
            f"{sorted(set(tickers) - set(w_tickers))} missing", False))
        return records
    records.append(CheckRecord(
        "ticker alignment", "set comparison", "identical ticker sets",
        f"{len(tickers)} tickers match", True))

    w = wdf.set_index(wdf["ticker"].astype(str)).loc[tickers, "weight"].to_numpy(float)

    cfg = (OptimizerConfig.from_json(config_path)
           if config_path and Path(config_path).exists() else OptimizerConfig())

    # (a) weights sum to 1.0
    s = float(w.sum())
    records.append(CheckRecord(
        "weights sum to 1.0", f"recomputed from {weights_csv.name}",
        f"1.0 within {weight_tol:g}", f"{s:.10f} (dev {abs(s - 1.0):.3e})",
        abs(s - 1.0) <= weight_tol))

    # (b) box constraints
    w_min = cfg.effective_min_weight()
    viol_lo = float(np.max(np.clip(w_min - w, 0.0, None)))
    viol_hi = float(np.max(np.clip(w - cfg.max_weight, 0.0, None)))
    records.append(CheckRecord(
        "position box constraints", "recomputed from weights.csv",
        f"all weights in [{w_min:g}, {cfg.max_weight:g}]",
        f"max lower violation {viol_lo:.3e}, max upper violation {viol_hi:.3e}",
        viol_lo <= weight_tol and viol_hi <= weight_tol))

    # (c) active factor tilts within bounds -- per factor
    tilt = pi.B.T @ (w - pi.w_b)
    lo, hi = cfg.resolve_tilt_bounds(pi.factor_names)
    for i, fname in enumerate(pi.factor_names):
        t = float(tilt[i])
        viol = float(max(0.0, float(lo[i]) - t, t - float(hi[i])))
        records.append(CheckRecord(
            f"active tilt within bounds: {fname}",
            "recomputed as B'(w - w_b) from raw CSVs",
            f"[{lo[i]:+.4f}, {hi[i]:+.4f}]", f"{t:+.6f} (violation {viol:.3e})",
            viol <= tilt_tol))
    return records


# ----------------------------------------------------------------------
# 2. Hand-checkable small instance vs brute force
# ----------------------------------------------------------------------
def _tiny_instance() -> Tuple[PortfolioInputs, OptimizerConfig]:
    """3 assets, 1 factor, round numbers -- hand-checkable by a human.

    Sigma = B F B' + D with B = [1.0, 0.5, 1.5]', F = [0.04],
    D = diag(0.01, 0.02, 0.015):

        Sigma = [[0.050, 0.020, 0.060],
                 [0.020, 0.030, 0.030],
                 [0.060, 0.030, 0.105]]

    delta = 0 and eps = 0, so the QP uses exactly this hand-computable Sigma.
    """
    pi = PortfolioInputs(
        tickers=np.array(["A", "B", "C"]),
        factor_names=np.array(["F1"]),
        mu=np.array([0.08, 0.06, 0.10]),
        B=np.array([[1.0], [0.5], [1.5]]),
        F=np.array([[0.04]]),
        residual_var=np.array([0.01, 0.02, 0.015]),
        w_b=np.array([0.5, 0.3, 0.2]),
        metadata={"mode": "tiny hand-checkable instance"},
    )
    cfg = OptimizerConfig(
        risk_aversion=1.0, concentration_penalty=0.05,
        shrinkage_intensity=0.0, ridge_epsilon=0.0,
        long_only=True, max_weight=0.60,
        tilt_bounds={"F1": [-0.10, 0.10]},
    )
    return pi, cfg


def check_small_case_bruteforce(step: float = 0.01) -> List[CheckRecord]:
    """Solve the tiny instance by exhaustive simplex grid search and compare.

    The grid optimum is an upper bound on the true optimum; near the optimum
    the quadratic objective is flat, so the expected objective gap is
    O(lambda_max * step^2) -- tiny for step = 0.01. Weight agreement is
    limited by grid resolution (tolerance 3*step, documented).
    """
    records: List[CheckRecord] = []
    pi, cfg = _tiny_instance()
    result = solve_factor_portfolio(pi, cfg)
    records.append(CheckRecord(
        "tiny instance: QP solves", "CVXPY QP",
        "status optimal", f"status={result.status}", result.success,
        note=result.message))
    if not result.success:
        return records

    Sigma = result.sigma
    n = pi.n_assets
    w_min = cfg.effective_min_weight()
    lo, hi = cfg.resolve_tilt_bounds(pi.factor_names)
    b = pi.B[:, 0]

    m = int(round(1.0 / step))
    best_obj, best_w = np.inf, None
    for i in range(m + 1):
        for j in range(m + 1 - i):
            w = np.array([i * step, j * step, 1.0 - (i + j) * step])
            if np.any(w < w_min - 1e-12) or np.any(w > cfg.max_weight + 1e-12):
                continue
            t = float(b @ (w - pi.w_b))
            if t < lo[0] - 1e-12 or t > hi[0] + 1e-12:
                continue
            obj = objective_value_np(w, pi.mu, pi.w_b, Sigma,
                                     cfg.risk_aversion, cfg.concentration_penalty)
            if obj < best_obj:
                best_obj, best_w = obj, w
    if best_w is None:
        records.append(CheckRecord(
            "tiny instance: grid search", "brute force over simplex",
            "at least one feasible grid point", "none found", False,
            note="instance or step misconfigured"))
        return records

    qp_obj = float(result.objective_value)
    gap = best_obj - qp_obj
    records.append(CheckRecord(
        "tiny instance: objective vs brute force",
        f"grid search step={step:g} ({(m + 1) * (m + 2) // 2} simplex points)",
        "QP objective <= grid objective, gap < 1e-3",
        f"qp={qp_obj:.8f}, grid={best_obj:.8f}, gap={gap:.3e}",
        (-1e-9 <= gap < 1e-3),
        note="gap bounded by grid resolution; QP optimum is global (convexity)"))

    wdiff = float(np.max(np.abs(result.weights - best_w)))
    records.append(CheckRecord(
        "tiny instance: weights vs brute force", "grid search argmin",
        f"max |w_qp - w_grid| <= {3 * step:g} (3 x step)",
        f"{wdiff:.4f} (qp={np.round(result.weights, 4).tolist()}, "
        f"grid={np.round(best_w, 4).tolist()})",
        wdiff <= 3 * step + 1e-9))
    return records


# ----------------------------------------------------------------------
# 3. Second-solver cross-check (SLSQP)
# ----------------------------------------------------------------------
def check_second_solver(pi: PortfolioInputs, cfg: OptimizerConfig,
                        *, weight_tol: float = 2e-3,
                        obj_abs_tol: float = 1e-7,
                        obj_rel_tol: float = 1e-6) -> List[CheckRecord]:
    """Re-solve the identical QP with scipy SLSQP and compare.

    Different algorithm (SQP with numerical QP substeps vs conic/ADMM QP) and
    different code path => a genuine second implementation. With
    concentration_penalty > 0 the objective is strictly convex, so both
    solvers must converge to the SAME unique optimum.
    """
    records: List[CheckRecord] = []
    if cfg.turnover_cap is not None:
        return [CheckRecord(
            "second-solver cross-check", "scipy SLSQP", "comparable solution",
            "skipped", True,
            note="turnover_cap active; SLSQP cross-check not configured for "
                 "the nonsmooth 1-norm constraint")]
    try:
        from scipy.optimize import minimize
    except ImportError:
        return [CheckRecord(
            "second-solver cross-check", "scipy SLSQP", "comparable solution",
            "skipped (scipy missing)", False, note="pip install scipy")]

    result = solve_factor_portfolio(pi, cfg)
    if not result.success:
        return [CheckRecord(
            "second-solver cross-check", "scipy SLSQP", "QP solves first",
            f"QP status={result.status}", False)]

    Sigma = result.sigma
    n = pi.n_assets
    gamma, kappa = cfg.risk_aversion, cfg.concentration_penalty
    w_b, mu, Bt = pi.w_b, pi.mu, pi.B.T
    lo, hi = cfg.resolve_tilt_bounds(pi.factor_names)

    def f(w):
        a = w - w_b
        return 0.5 * a @ Sigma @ a - gamma * (mu @ a) + kappa * (w @ w)

    def grad(w):
        a = w - w_b
        return Sigma @ a - gamma * mu + 2.0 * kappa * w

    cons = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0,
         "jac": lambda w: np.ones(n)},
        {"type": "ineq", "fun": lambda w: Bt @ (w - w_b) - lo,
         "jac": lambda w: Bt},
        {"type": "ineq", "fun": lambda w: hi - Bt @ (w - w_b),
         "jac": lambda w: -Bt},
    ]
    bounds = [(cfg.effective_min_weight(), cfg.max_weight)] * n
    x0 = np.clip(pi.w_b, cfg.effective_min_weight(), cfg.max_weight)
    sol = minimize(f, x0, jac=grad, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-12})

    records.append(CheckRecord(
        "second solver: SLSQP converges", "scipy.optimize.minimize(SLSQP)",
        "success", f"success={bool(sol.success)} ({sol.message})",
        bool(sol.success)))

    qp_obj = float(result.objective_value)
    sl_obj = float(sol.fun)
    obj_tol = obj_abs_tol + obj_rel_tol * abs(qp_obj)
    records.append(CheckRecord(
        "second solver: objective agreement", "SLSQP vs CVXPY objective",
        f"|diff| <= {obj_tol:.2e}",
        f"cvxpy={qp_obj:.10f}, slsqp={sl_obj:.10f}, diff={abs(sl_obj - qp_obj):.3e}",
        abs(sl_obj - qp_obj) <= obj_tol))

    wdiff = float(np.max(np.abs(sol.x - result.weights)))
    records.append(CheckRecord(
        "second solver: weight agreement", "max abs weight difference",
        f"<= {weight_tol:g} (unique optimum, strict convexity)",
        f"{wdiff:.3e}",
        wdiff <= weight_tol))
    return records


# ----------------------------------------------------------------------
# 4. Regularization properties on a collinear instance
# ----------------------------------------------------------------------
def check_regularization() -> List[CheckRecord]:
    """Shrinkage + ridge must preserve/improve PSD conditioning; delta=0 = identity."""
    records: List[CheckRecord] = []
    pi, _ = synthetic_inputs(n_assets=12, n_factors=2, seed=3,
                             collinear_corr=0.999)
    S_raw = cov.factor_model_covariance(pi.B, pi.F, pi.residual_var)
    cond_raw = cov.condition_number(S_raw)

    s0 = cov.shrink_covariance(S_raw, 0.0)
    records.append(CheckRecord(
        "regularization: delta=0 is identity", "shrink_covariance(S, 0)",
        "unchanged matrix", f"max abs diff {float(np.max(np.abs(s0 - S_raw))):.3e}",
        bool(np.allclose(s0, S_raw))))

    s1 = cov.shrink_covariance(S_raw, 0.3, target="diagonal")
    sym_ok = bool(np.allclose(s1, s1.T))
    min_eig = cov.min_eigenvalue(s1)
    cond_s1 = cov.condition_number(s1)
    records.append(CheckRecord(
        "regularization: shrunk Sigma symmetric PSD",
        "shrink_covariance(S, 0.3, 'diagonal') on corr=0.999 instance",
        "symmetric, min eigenvalue >= -1e-12",
        f"symmetric={sym_ok}, min eig={min_eig:.3e}",
        sym_ok and min_eig >= -1e-12))
    records.append(CheckRecord(
        "regularization: condition number improves",
        f"cond(raw) vs cond(shrunk) on near-collinear instance",
        f"cond(shrunk) < cond(raw) = {cond_raw:.3e}",
        f"cond(shrunk) = {cond_s1:.3e}",
        cond_s1 < cond_raw,
        note="descriptive expectation for the deliberately collinear instance"))

    r1 = cov.ridge_covariance(s1, 1e-8)
    records.append(CheckRecord(
        "regularization: ridge keeps PSD",
        "ridge_covariance(shrunk, 1e-8)",
        "min eigenvalue >= 0", f"min eig={cov.min_eigenvalue(r1):.3e}",
        cov.min_eigenvalue(r1) >= 0.0))

    s2 = cov.shrink_covariance(S_raw, 0.5, target="scaled_identity")
    records.append(CheckRecord(
        "regularization: scaled-identity target PSD",
        "shrink_covariance(S, 0.5, 'scaled_identity')",
        "min eigenvalue >= -1e-12", f"min eig={cov.min_eigenvalue(s2):.3e}",
        cov.min_eigenvalue(s2) >= -1e-12))
    return records


# ----------------------------------------------------------------------
# 5. Baseline comparison (Equal-Weight 1/N and unregularized sample cov)
# ----------------------------------------------------------------------
def _max_drawdown(wealth: np.ndarray) -> float:
    """Largest percentage fall from a running peak (course T2 definition)."""
    wealth = np.asarray(wealth, dtype=float)
    peak = np.maximum.accumulate(wealth)
    dd = wealth / peak - 1.0
    return float(dd.min())


@dataclass
class BaselineComparison:
    table: pd.DataFrame
    records: List[CheckRecord]
    provenance: Dict


def make_synthetic_returns(pi: PortfolioInputs, n_days: int = 1500,
                           seed: int = 99, start: str = "2018-01-02",
                           periods_per_year: int = 252
                           ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate daily returns consistent with pi's factor model.

    Used so the full verification suite (including the baseline comparison)
    runs OFFLINE and reproducibly. Clearly labeled synthetic: results
    demonstrate the pipeline, not a claim about real markets.
    """
    rng = np.random.default_rng(seed)
    n, k = pi.n_assets, pi.n_factors
    F_daily = pi.F / periods_per_year
    sd_daily = np.sqrt(pi.residual_var / periods_per_year)
    dates = pd.bdate_range(start=start, periods=n_days)
    # Generator.multivariate_normal (method='svd') tolerates semidefinite cov
    f = rng.multivariate_normal(np.zeros(k), F_daily, size=n_days)
    eps = rng.normal(0.0, 1.0, size=(n_days, n)) * sd_daily
    r = f @ pi.B.T + eps
    asset_returns = pd.DataFrame(r, index=dates,
                                 columns=[str(t) for t in pi.tickers])
    factor_returns = pd.DataFrame(f, index=dates,
                                  columns=[str(x) for x in pi.factor_names])
    return asset_returns, factor_returns


def run_baseline_comparison(asset_returns: pd.DataFrame,
                            factor_returns: pd.DataFrame,
                            design_end,
                            *,
                            cfg: Optional[OptimizerConfig] = None,
                            periods_per_year: int = 252) -> BaselineComparison:
    """Design/test comparison with rules fixed BEFORE seeing test data.

    Design window (<= design_end): estimate mu, B, F, D by OLS (equal-weight
    benchmark). Three portfolios, each allocated ONCE at design end and held
    (buy-and-hold) over the test window (> design_end):

      optimized_qp_shrunk        project method: factor-model covariance with
                                 shrinkage + ridge, tilt bounds enforced
      equal_weight_1_over_N      naive reference heuristic (no optimization)
      unregularized_sample_cov   SAME objective and constraints, but Sigma is
                                 the raw sample covariance of asset returns
                                 (delta = 0, ridge = 0) -- isolates the effect
                                 of regularization

    Test metrics vs the benchmark: cumulative return, annualized vol, max
    drawdown (T2 running-peak definition), tracking error (std of daily
    active returns, annualized), geometric active return. Descriptive
    outcomes (including underperformance) are valid results per Appendix C.
    """
    records: List[CheckRecord] = []
    cfg = cfg if cfg is not None else OptimizerConfig()
    mask = asset_returns.index <= design_end
    fmask = factor_returns.index <= design_end
    ar_d, ar_t = asset_returns[mask], asset_returns[~mask]
    fr_d = factor_returns[fmask]
    if len(ar_d) < 30 or len(ar_t) < 20:
        raise ValueError(
            f"design/test split too small: {len(ar_d)} design rows, "
            f"{len(ar_t)} test rows (need >= 30 / >= 20)")

    pi_d, prov = build_inputs_from_returns(
        ar_d, fr_d, periods_per_year=periods_per_year)
    n = pi_d.n_assets

    res_opt = solve_factor_portfolio(pi_d, cfg)
    Sigma_sample = cov.sample_covariance(ar_d, periods_per_year=periods_per_year)
    cfg_unreg = dataclasses.replace(
        cfg, shrinkage_intensity=0.0, ridge_epsilon=0.0)
    res_unreg = solve_factor_portfolio(
        pi_d, cfg_unreg, sigma_override=Sigma_sample,
        sigma_label="raw sample covariance of asset returns (unregularized)")

    variants = [
        ("optimized_qp_shrunk", res_opt),
        ("unregularized_sample_cov", res_unreg),
    ]
    rows = []
    lo, hi = cfg.resolve_tilt_bounds(pi_d.factor_names)
    R_test = ar_t.to_numpy(float)
    w_b = pi_d.w_b
    bench_rets = R_test @ w_b
    bench_wealth = 100.0 * np.cumprod(1.0 + bench_rets)

    def _metrics_row(name, w, sigma_cond: Optional[float], design_status: str):
        pr = R_test @ w
        wealth = 100.0 * np.cumprod(1.0 + pr)
        active = pr - bench_rets
        tilt = pi_d.B.T @ (w - w_b)
        tilt_ok = bool(np.all(tilt >= lo - DEFAULT_TILT_TOL)
                       and np.all(tilt <= hi + DEFAULT_TILT_TOL))
        return {
            "variant": name,
            "design_status": design_status,
            "sigma_condition_number": sigma_cond,
            "design_tilts_within_bounds": tilt_ok,
            "test_cum_return": float(wealth[-1] / 100.0 - 1.0),
            "test_ann_vol": float(np.std(pr, ddof=1) * np.sqrt(periods_per_year)),
            "test_max_drawdown": _max_drawdown(wealth),
            "test_tracking_error": float(np.std(active, ddof=1)
                                         * np.sqrt(periods_per_year)),
            "test_active_return_geo": float(wealth[-1] / bench_wealth[-1] - 1.0),
            "benchmark_test_cum_return": float(bench_wealth[-1] / 100.0 - 1.0),
        }, tilt_ok

    built_ok = True
    row, _ = _metrics_row("benchmark_design", w_b, None, "reference")
    rows.append(row)
    tilt_ok_opt = True
    for name, res in variants:
        if res.success and res.weights is not None:
            row, tilt_ok = _metrics_row(name, res.weights,
                                        res.condition_number_regularized,
                                        res.status)
            rows.append(row)
            if name == "optimized_qp_shrunk":
                tilt_ok_opt = tilt_ok
                if not tilt_ok:
                    built_ok = False
        else:
            rows.append({"variant": name, "design_status": res.status,
                         "sigma_condition_number": None,
                         "design_tilts_within_bounds": False})
            built_ok = False
    row, _ = _metrics_row("equal_weight_1_over_N", equal_weight(n), None,
                          "reference (no solve)")
    rows.append(row)

    table = pd.DataFrame(rows)
    records.append(CheckRecord(
        "baselines: all variants constructed and solved",
        "design-window estimation + QP solves",
        "both QP variants optimal; optimized tilts within bounds",
        f"optimized={res_opt.status}, unregularized={res_unreg.status}, "
        f"tilts_ok={tilt_ok_opt}, table rows={len(table)}",
        bool(built_ok and res_opt.success and res_unreg.success)))

    finite_cols = ["test_cum_return", "test_ann_vol", "test_max_drawdown",
                   "test_tracking_error", "test_active_return_geo"]
    all_finite = True
    for _, r in table.iterrows():
        for c in finite_cols:
            v = r.get(c)
            if v is None or pd.isna(v) or not np.isfinite(float(v)):
                all_finite = False
    records.append(CheckRecord(
        "baselines: realized metrics finite", "test-window evaluation",
        "all reported metrics finite", f"all_finite={all_finite}",
        all_finite))

    # descriptive comparison rows (Appendix C: negative results are valid)
    qps = table[table["variant"].isin(
        ["optimized_qp_shrunk", "unregularized_sample_cov",
         "equal_weight_1_over_N"])].dropna(subset=["test_tracking_error"])
    if len(qps):
        best_te = qps.loc[qps["test_tracking_error"].idxmin()]
        best_ar = qps.loc[qps["test_active_return_geo"].idxmax()]
        records.append(CheckRecord(
            "baselines: lowest realized tracking error (descriptive)",
            "test-window comparison", "reported, no assertion",
            f"{best_te['variant']} (TE={best_te['test_tracking_error']:.4f})",
            True, note="descriptive outcome; underperformance is a valid result"))
        records.append(CheckRecord(
            "baselines: highest realized active return (descriptive)",
            "test-window comparison", "reported, no assertion",
            f"{best_ar['variant']} (geo active={best_ar['test_active_return_geo']:+.4f})",
            True, note="descriptive outcome; underperformance is a valid result"))

    cond_opt = res_opt.condition_number_regularized
    cond_unreg = res_unreg.condition_number_regularized
    records.append(CheckRecord(
        "baselines: covariance conditioning (descriptive)",
        "cond(Sigma) of each variant", "reported, no assertion",
        f"shrunk factor-model={cond_opt:.3e}, raw sample={cond_unreg:.3e}",
        True, note="regularization is expected to lower the condition number; "
                   "actual values reported for audit"))

    provenance = {
        "design_window": [str(ar_d.index.min()), str(ar_d.index.max())],
        "test_window": [str(ar_t.index.min()), str(ar_t.index.max())],
        "design_end": str(design_end),
        "n_design_obs": int(len(ar_d)), "n_test_obs": int(len(ar_t)),
        "estimation": prov,
        "rule": "weights fixed once at design end; buy-and-hold over test; "
                "no test-window information used in design",
        "note": "the harness benchmark defaults to equal weight; if a "
                "user-supplied benchmark equals 1/N, the benchmark_design and "
                "equal_weight_1_over_N rows coincide by construction",
    }
    return BaselineComparison(table=table, records=records, provenance=provenance)


# ----------------------------------------------------------------------
# 6. Infeasible-constraint edge case
# ----------------------------------------------------------------------
def check_infeasible_edge_case(pi: PortfolioInputs,
                               base_cfg: Optional[OptimizerConfig] = None
                               ) -> List[CheckRecord]:
    """Three guaranteed-infeasible constraint sets must fail GRACEFULLY.

    A: contradictory tilt bounds (lo > hi)
    B: unreachable tilt requirement (lower bound provably above the maximum
       attainable active exposure given sum-to-one and the box)
    C: box that cannot sum to 1 (min_weight * n > 1)

    Expected behavior in all cases: no exception, result.status == 'infeasible',
    result.success is False, compliance verdict == 'INFEASIBLE'.
    """
    records: List[CheckRecord] = []
    base = base_cfg if base_cfg is not None else OptimizerConfig()
    first_factor = str(pi.factor_names[0])

    # A: lo > hi
    cfg_a = dataclasses.replace(
        base, tilt_bounds={**(base.tilt_bounds or {}), first_factor: [0.2, -0.2]})
    # B: unreachable tilt. With 1'w = 1'w_b = 1 and box bounds, the L1 norm of
    #    active weights is at most 2 (long-only), so |B_j' a| <= 2*max|B_j|.
    max_load = float(np.max(np.abs(pi.B[:, 0])))
    unreachable = 2.0 * max_load + 1.0
    cfg_b = dataclasses.replace(
        base,
        tilt_bounds={**(base.tilt_bounds or {}),
                     first_factor: [unreachable, unreachable + 1.0]})
    # C: box infeasible with sum-to-one (only when n * min_weight > 1)
    cfg_c = None
    if pi.n_assets * 0.5 > 1.0:
        cfg_c = dataclasses.replace(base, long_only=True,
                                    min_weight=0.5, max_weight=0.6)

    for label, cfg, detail in [
            ("A: contradictory tilt bounds (lo > hi)", cfg_a, "[0.2, -0.2]"),
            (f"B: unreachable tilt (>= {unreachable:.3f})", cfg_b,
             f"max attainable |tilt| <= {2.0 * max_load:.3f}"),
            ("C: box cannot sum to 1 (min_weight=0.5)", cfg_c, "n*0.5 > 1")]:
        if cfg is None:
            records.append(CheckRecord(
                f"infeasible edge case {label}", "CVXPY QP",
                "n/a", "skipped (universe too small for this case)", True,
                note="case requires n_assets >= 3"))
            continue
        try:
            res = solve_factor_portfolio(pi, cfg)
            rep = build_compliance_report(res, pi, cfg)
            ok = (res.status == "infeasible" and not res.success
                  and rep.verdict == "INFEASIBLE" and res.weights is None)
            records.append(CheckRecord(
                f"infeasible edge case {label}", "CVXPY QP + compliance report",
                "status=infeasible, verdict=INFEASIBLE, no crash, no weights",
                f"status={res.status}, verdict={rep.verdict}, detail={detail}",
                ok, note=res.message[:160]))
        except Exception as exc:  # must never happen: graceful failure required
            records.append(CheckRecord(
                f"infeasible edge case {label}", "CVXPY QP + compliance report",
                "graceful infeasible report", f"RAISED {type(exc).__name__}: {exc}",
                False))
    return records


# ----------------------------------------------------------------------
# 7. Invalid-input handling
# ----------------------------------------------------------------------
def check_invalid_inputs(pi: PortfolioInputs) -> List[CheckRecord]:
    """Malformed inputs must raise InputValidationError, never pass silently."""
    records: List[CheckRecord] = []

    def expect_invalid(label: str, mutate) -> None:
        bad = copy.deepcopy(pi)
        mutate(bad)
        try:
            validate_inputs(bad)
            records.append(CheckRecord(
                f"invalid input rejected: {label}", "validate_inputs",
                "InputValidationError", "NO ERROR RAISED (accepted bad input!)",
                False))
        except InputValidationError as exc:
            records.append(CheckRecord(
                f"invalid input rejected: {label}", "validate_inputs",
                "InputValidationError", f"raised: {str(exc)[:110]}", True))
        except Exception as exc:
            records.append(CheckRecord(
                f"invalid input rejected: {label}", "validate_inputs",
                "InputValidationError", f"wrong exception {type(exc).__name__}: {exc}",
                False))

    def m_benchmark_sum(bad): bad.w_b = bad.w_b * 0.9
    def m_nonpsd(bad): bad.F = bad.F.copy(); bad.F[0, 0] = -abs(bad.F[0, 0]) - 1.0
    def m_asym(bad): bad.F = bad.F.copy(); bad.F[0, -1] += 0.5
    def m_nan(bad): bad.mu = bad.mu.copy(); bad.mu[0] = np.nan
    def m_shape(bad): bad.B = bad.B[:-1]
    def m_neg_resid(bad): bad.residual_var = bad.residual_var.copy(); bad.residual_var[0] = -0.01

    expect_invalid("benchmark weights sum to 0.9", m_benchmark_sum)
    expect_invalid("factor covariance not PSD", m_nonpsd)
    expect_invalid("factor covariance asymmetric", m_asym)
    expect_invalid("expected return is NaN", m_nan)
    expect_invalid("loadings shape mismatch", m_shape)
    expect_invalid("negative residual variance", m_neg_resid)

    # file-level: missing directory must fail loudly
    with tempfile.TemporaryDirectory() as td:
        try:
            load_inputs_from_dir(Path(td) / "does_not_exist")
            records.append(CheckRecord(
                "invalid input rejected: missing input dir", "load_inputs_from_dir",
                "InputValidationError", "NO ERROR RAISED", False))
        except InputValidationError as exc:
            records.append(CheckRecord(
                "invalid input rejected: missing input dir", "load_inputs_from_dir",
                "InputValidationError", f"raised: {str(exc)[:110]}", True))
    return records


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------
def _load_wide_prices(path) -> pd.DataFrame:
    """Load a wide-format daily price CSV (a 'date' column or DatetimeIndex)."""
    df = pd.read_csv(path)
    date_col = None
    for cand in ("date", "Date", "DATE"):
        if cand in df.columns:
            date_col = cand
            break
    if date_col is None:
        date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="any")
    if df.empty:
        raise ValueError(f"no usable rows in {path}")
    return df


def records_to_markdown(records: List[CheckRecord]) -> str:
    lines = ["| # | Case | Method | Expected | Actual | Pass/Fail | Note |",
             "|---|------|--------|----------|--------|-----------|------|"]
    for i, r in enumerate(records, 1):
        note = (r.note or "").replace("|", "/").replace("\n", " ")
        lines.append(
            f"| {i} | {r.case} | {r.method} | {r.expected} | {r.actual} | "
            f"{'PASS' if r.passed else '**FAIL**'} | {note} |")
    return "\n".join(lines)


def run_all(project_root=None, *,
            inputs_dir=None, config_path=None, run_dir=None,
            baseline_mode: str = "synthetic",
            prices_csv=None, factors_csv=None, design_end=None,
            tiny_step: float = 0.01,
            periods_per_year: int = 252) -> Tuple[List[CheckRecord], pd.DataFrame, Dict]:
    """Run the complete verification suite and persist the evaluation record.

    baseline_mode:
      "synthetic" (default): returns simulated from the example factor model
                             (seeded, offline, clearly labeled synthetic)
      "market":              requires --prices/--factors wide CSVs of daily
                             adjusted closes; design/test split at design_end
                             (default: 60% point of the common window)
    """
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
    inputs_dir = Path(inputs_dir) if inputs_dir else root / "examples" / "basic"
    run_dir = Path(run_dir) if run_dir else root / "outputs" / "verification"
    run_dir.mkdir(parents=True, exist_ok=True)

    records: List[CheckRecord] = []

    # --- load + main solve + compliance artifacts -------------------------
    pi = load_inputs_from_dir(inputs_dir)
    cfg_path = Path(config_path) if config_path else inputs_dir / "config.json"
    cfg = (OptimizerConfig.from_json(cfg_path) if cfg_path.exists()
           else OptimizerConfig())
    records.append(CheckRecord(
        "inputs load + validate", "load_inputs_from_dir",
        "no error", f"n_assets={pi.n_assets}, n_factors={pi.n_factors}, "
                    f"config={'file' if cfg_path.exists() else 'defaults'}",
        True))

    result = solve_factor_portfolio(pi, cfg)
    report = build_compliance_report(result, pi, cfg)
    paths = save_compliance_report(report, result, pi, cfg, run_dir)
    records.append(CheckRecord(
        "main optimization", f"CVXPY QP ({result.solver_name})",
        "status=optimal, verdict=PASS",
        f"status={result.status}, verdict={report.verdict}",
        bool(result.success and report.verdict == "PASS"),
        note=result.message[:160]))

    # --- 1. independent file-level constraint verification -----------------
    weights_csv = paths.get("weights_csv")
    if weights_csv is not None:
        records += verify_weights_file(
            weights_csv, inputs_dir, cfg_path if cfg_path.exists() else None)

    # --- 2/3. brute force + second solver ----------------------------------
    records += check_small_case_bruteforce(step=tiny_step)
    records += check_second_solver(pi, cfg)

    # --- 4. regularization properties ---------------------------------------
    records += check_regularization()

    # --- 5. baseline comparison ---------------------------------------------
    if baseline_mode == "market" and prices_csv and factors_csv:
        ap = _load_wide_prices(prices_csv)
        fp = _load_wide_prices(factors_csv)
        common = ap.index.intersection(fp.index)
        ap, fp = ap.loc[common], fp.loc[common]
        asset_rets = ap.pct_change().dropna(how="any")
        factor_rets = fp.pct_change().dropna(how="any")
        common = asset_rets.index.intersection(factor_rets.index)
        asset_rets, factor_rets = asset_rets.loc[common], factor_rets.loc[common]
        if design_end is None:
            design_end = common[int(0.6 * len(common)) - 1]
        else:
            design_end = pd.Timestamp(design_end)
        mode_label = "market"
    else:
        asset_rets, factor_rets = make_synthetic_returns(
            pi, seed=99, periods_per_year=periods_per_year)
        design_end = asset_rets.index[int(0.6 * len(asset_rets.index)) - 1]
        mode_label = "synthetic (seeded simulation from the example factor model)"
    comp = run_baseline_comparison(asset_rets, factor_rets, design_end,
                                   cfg=cfg, periods_per_year=periods_per_year)
    comp.table.to_csv(run_dir / "baseline_comparison.csv", index=False)
    records += comp.records

    # --- 6/7. edge cases ------------------------------------------------------
    records += check_infeasible_edge_case(pi, cfg)
    records += check_invalid_inputs(pi)

    # --- persist the compact evaluation record --------------------------------
    df = pd.DataFrame([asdict(r) for r in records])
    df.to_csv(run_dir / "evaluation_record.csv", index=False)
    n_pass = int(df["passed"].sum())
    n_total = len(df)
    header = [
        "# Evaluation Record (independent verification suite)",
        "",
        f"- Inputs: `{inputs_dir}`  ·  Config: `{cfg_path if cfg_path.exists() else 'defaults'}`",
        f"- Baseline data mode: {mode_label}",
        f"- Design/test split at: {design_end}",
        f"- Result: **{n_pass}/{n_total} checks passed**",
        f"- Baseline comparison provenance: {comp.provenance}",
        "",
    ]
    (run_dir / "evaluation_record.md").write_text(
        "\n".join(header) + records_to_markdown(records), encoding="utf-8")

    print(records_to_markdown(records).replace("|", " | "))
    print(f"\nSUMMARY: {n_pass}/{n_total} checks passed; "
          f"artifacts saved under {run_dir}")
    return records, df, {"run_dir": run_dir, "n_pass": n_pass,
                         "n_total": n_total,
                         "baseline_provenance": comp.provenance}
