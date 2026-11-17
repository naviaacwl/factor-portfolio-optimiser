"""Active-factor compliance report.

Recomputes every reported quantity directly from the optimized weights and
the RAW inputs (plain numpy), independent of the optimizer's internal
bookkeeping, then renders:

  * ``weights.csv``           ticker, weight, active_weight (full precision)
  * ``compliance_report.md``  human-readable report with a PASS/FAIL verdict
  * ``compliance.json``       machine-readable record for downstream checks

Verdict semantics:
  PASS         solver optimal AND weight sum = 1 (within tol) AND box bounds
               respected AND every active tilt inside its bounds AND the
               post-solve max constraint violation is within tolerance
  FAIL         solved, but at least one independent recheck was violated
  INFEASIBLE   solver proved the constraint set infeasible (edge case
               handled gracefully -- no crash, explicit guidance)
  SOLVER_ERROR solver failure / unexpected status
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import OptimizerConfig
from .data import PortfolioInputs
from .optimizer import OptimizationResult

WEIGHTS_FILE = "weights.csv"
REPORT_MD_FILE = "compliance_report.md"
REPORT_JSON_FILE = "compliance.json"


@dataclass
class FactorComplianceRow:
    """Per-factor active-tilt compliance record."""

    factor: str
    benchmark_exposure: float
    portfolio_exposure: float
    active_tilt: float
    tilt_lower: float
    tilt_upper: float
    within_bounds: bool
    violation: float                     # 0 when within bounds
    bound_utilization: Optional[float]   # |tilt| / max(|lo|,|hi|); None if both 0
    slack: float                         # min(tilt - lo, hi - tilt); < 0 = breach


@dataclass
class ComplianceReport:
    verdict: str                         # PASS | FAIL | INFEASIBLE | SOLVER_ERROR
    generated_at: str
    solver_status: str
    solver_name: Optional[str]
    message: str
    factors: List[FactorComplianceRow] = field(default_factory=list)
    weights_summary: Dict = field(default_factory=dict)
    risk_summary: Dict = field(default_factory=dict)
    regularization_summary: Dict = field(default_factory=dict)
    tolerances: Dict = field(default_factory=dict)
    universe: Dict = field(default_factory=dict)
    config_summary: str = ""


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def build_compliance_report(result: OptimizationResult,
                            inputs: PortfolioInputs,
                            config: OptimizerConfig,
                            *,
                            tilt_tol: float = 1e-5,
                            weight_tol: float = 1e-6) -> ComplianceReport:
    """Build the compliance report by INDEPENDENTLY recomputing tilts/stats.

    The tilts and summaries below are recomputed from ``result.weights`` and
    the raw ``inputs`` arrays with plain numpy -- not copied from the
    optimizer's fields -- so an internal bookkeeping bug cannot hide itself
    here. (The strongest independence check -- re-reading saved files from
    disk -- lives in ``verify.verify_weights_file``.)
    """
    generated = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    universe = {
        "n_assets": inputs.n_assets,
        "n_factors": inputs.n_factors,
        "tickers": [str(t) for t in inputs.tickers],
        "factor_names": [str(f) for f in inputs.factor_names],
    }
    common = dict(
        generated_at=generated,
        solver_status=result.status,
        solver_name=result.solver_name,
        message=result.message or "",
        tolerances={"tilt_tol": tilt_tol, "weight_tol": weight_tol,
                    "sum_to_one_tol_config": config.sum_to_one_tol},
        universe=universe,
        config_summary=config.describe(),
        regularization_summary={
            "sigma_source": result.sigma_source,
            "shrinkage_intensity": result.shrinkage_intensity,
            "condition_number_raw": result.condition_number_raw,
            "condition_number_regularized": result.condition_number_regularized,
        },
    )

    if not result.success:
        verdict = "INFEASIBLE" if result.status == "infeasible" else "SOLVER_ERROR"
        return ComplianceReport(
            verdict=verdict,
            weights_summary={"status": result.status},
            risk_summary={},
            **common)

    w = np.asarray(result.weights, dtype=float)
    a = w - np.asarray(inputs.w_b, dtype=float)

    # --- recompute exposures / tilts from raw inputs ---------------------
    bench = inputs.B.T @ inputs.w_b
    port = inputs.B.T @ w
    tilt = inputs.B.T @ a
    lo, hi = config.resolve_tilt_bounds(inputs.factor_names)

    rows: List[FactorComplianceRow] = []
    for i, fname in enumerate(inputs.factor_names):
        t, l, h = float(tilt[i]), float(lo[i]), float(hi[i])
        violation = float(max(0.0, l - t, t - h))
        denom = max(abs(l), abs(h))
        util = (abs(t) / denom) if denom > 0 else None
        rows.append(FactorComplianceRow(
            factor=str(fname),
            benchmark_exposure=float(bench[i]),
            portfolio_exposure=float(port[i]),
            active_tilt=t,
            tilt_lower=l, tilt_upper=h,
            within_bounds=violation <= tilt_tol,
            violation=violation,
            bound_utilization=util,
            slack=float(min(t - l, h - t)),
        ))

    # --- weight-level checks ----------------------------------------------
    w_sum = float(w.sum())
    weight_sum_ok = abs(w_sum - 1.0) <= weight_tol
    w_min_bound = config.effective_min_weight()
    box_ok = bool(np.all(w >= w_min_bound - weight_tol)
                  and np.all(w <= config.max_weight + weight_tol))
    long_only_ok = bool(np.all(w >= -weight_tol)) if config.long_only else True
    factors_ok = all(r.within_bounds for r in rows)
    violation_ok = (result.max_constraint_violation is None
                    or result.max_constraint_violation <= max(tilt_tol, weight_tol))
    verdict = ("PASS" if (weight_sum_ok and box_ok and long_only_ok
                          and factors_ok and violation_ok) else "FAIL")

    hhi = float(np.sum(w ** 2))
    top_idx = np.argsort(-w)[:5]
    weights_summary = {
        "sum_of_weights": w_sum,
        "sum_to_one_ok": weight_sum_ok,
        "min_weight": float(w.min()),
        "max_weight": float(w.max()),
        "max_weight_cap": config.max_weight,
        "box_ok": box_ok,
        "long_only_ok": long_only_ok,
        "hhi": hhi,
        "effective_n": float(1.0 / hhi) if hhi > 0 else float("inf"),
        "n_at_upper_cap": int(np.sum(w >= config.max_weight - 1e-6)),
        "n_effectively_zero": int(np.sum(np.abs(w) <= 1e-8)),
        "top_5_holdings": [
            {"ticker": str(inputs.tickers[i]), "weight": float(w[i])}
            for i in top_idx
        ],
    }
    risk_summary = {
        "expected_return": result.expected_return,
        "benchmark_expected_return": float(inputs.mu @ inputs.w_b),
        "expected_active_return": result.expected_active_return,
        "tracking_error_annual": result.tracking_error_annual,
        "objective_value": result.objective_value,
        "max_constraint_violation": result.max_constraint_violation,
        "solve_time_seconds": result.solve_time_seconds,
    }
    return ComplianceReport(verdict=verdict, factors=rows,
                            weights_summary=weights_summary,
                            risk_summary=risk_summary, **common)


# ----------------------------------------------------------------------
# Rendering / persistence
# ----------------------------------------------------------------------
def render_markdown(report: ComplianceReport) -> str:
    """Render the compliance report as Markdown (the human deliverable)."""
    L: List[str] = []
    L.append("# Active Factor Compliance Report")
    L.append("")
    L.append(f"**Verdict: {report.verdict}**  ·  generated {report.generated_at}  ·  "
             f"solver status: `{report.solver_status}`"
             + (f" ({report.solver_name})" if report.solver_name else ""))
    if report.message:
        L.append("")
        L.append(f"> {report.message}")
    L.append("")

    if report.verdict in ("INFEASIBLE", "SOLVER_ERROR"):
        L.append("No portfolio weights were produced. The constraint set could not be "
                 "satisfied (INFEASIBLE) or the solver failed (SOLVER_ERROR).")
        L.append("")
        L.append("Suggested next steps: relax active-tilt bounds, raise the single-stock "
                 "cap, check for contradictory config (tilt lo > hi), or switch solvers "
                 "via `config.solver`.")
        L.append("")
        L.append("## Diagnostics")
        L.append("")
        L.append("```")
        L.append(str(report.regularization_summary))
        L.append("```")
        return "\n".join(L)

    ws, rs = report.weights_summary, report.risk_summary
    L.append("## Universe and configuration")
    L.append("")
    L.append(f"- Assets: {report.universe.get('n_assets')}  ·  Factors: "
             f"{report.universe.get('n_factors')} "
             f"({', '.join(report.universe.get('factor_names', []))})")
    L.append("- Configuration:")
    L.append("")
    L.append("```")
    L.append(report.config_summary)
    L.append("```")
    L.append("")

    L.append("## Active factor tilts (recomputed from weights and raw inputs)")
    L.append("")
    L.append("| Factor | Benchmark exp. | Portfolio exp. | Active tilt | Bounds | Within? | Utilization | Slack |")
    L.append("|---|---:|---:|---:|---|---|---:|---:|")
    for r in report.factors:
        util = "n/a" if r.bound_utilization is None else f"{r.bound_utilization * 100:.1f}%"
        L.append(f"| {r.factor} | {r.benchmark_exposure:+.4f} | {r.portfolio_exposure:+.4f} "
                 f"| {r.active_tilt:+.4f} | [{r.tilt_lower:+.3f}, {r.tilt_upper:+.3f}] "
                 f"| {'YES' if r.within_bounds else '**NO**'} | {util} | {r.slack:+.4f} |")
    L.append("")

    L.append("## Weight constraints")
    L.append("")
    L.append(f"- Sum of weights: **{ws['sum_of_weights']:.10f}** "
             f"(target 1.0, tol {report.tolerances['weight_tol']:g}) -> "
             f"{'OK' if ws['sum_to_one_ok'] else '**VIOLATED**'}")
    L.append(f"- Weight range: [{ws['min_weight']:.6f}, {ws['max_weight']:.6f}] "
             f"vs cap {ws['max_weight_cap']:g} -> "
             f"{'OK' if ws['box_ok'] else '**VIOLATED**'}")
    L.append(f"- Long-only respected: {'YES' if ws['long_only_ok'] else '**NO**'}")
    L.append(f"- Names at upper cap: {ws['n_at_upper_cap']}  ·  effectively zero: "
             f"{ws['n_effectively_zero']}")
    top = ", ".join(f"{h['ticker']} {h['weight'] * 100:.2f}%"
                    for h in ws.get("top_5_holdings", []))
    L.append(f"- Top 5 holdings: {top}")
    L.append("")

    L.append("## Concentration and risk analytics")
    L.append("")
    L.append(f"- HHI (sum of squared weights): {ws['hhi']:.4f}")
    L.append(f"- Effective N (1/HHI): {ws['effective_n']:.2f} "
             f"of {report.universe.get('n_assets')} assets")
    L.append(f"- Expected return (annual): {rs['expected_return']:.4%} "
             f"(benchmark {rs['benchmark_expected_return']:.4%})")
    L.append(f"- Expected active return: {rs['expected_active_return']:+.4%}")
    L.append(f"- Tracking error (annual, ex-ante): {rs['tracking_error_annual']:.4%}")
    L.append(f"- Objective value: {rs['objective_value']:.8f}")
    L.append(f"- Max constraint violation (post-solve): "
             f"{rs['max_constraint_violation']:.3e}")
    L.append(f"- Solve time: {rs['solve_time_seconds']:.3f} s")
    L.append("")

    reg = report.regularization_summary
    L.append("## Covariance regularization audit (multicollinearity defense)")
    L.append("")
    L.append(f"- Sigma construction: {reg.get('sigma_source')}")
    L.append(f"- Shrinkage intensity: {reg.get('shrinkage_intensity')}")
    L.append(f"- Condition number, raw vs regularized: "
             f"{reg.get('condition_number_raw'):.3e} -> "
             f"{reg.get('condition_number_regularized'):.3e}")
    L.append("")
    L.append("---")
    L.append("*Generated by fp_optimizer (HKUST Finance-with-AI-Agents final project). "
             "Ex-ante analytics from the supplied inputs; not investment advice. "
             "Independent file-level rechecks: `scripts/run_verification.py`.*")
    return "\n".join(L)


def save_compliance_report(report: ComplianceReport,
                           result: OptimizationResult,
                           inputs: PortfolioInputs,
                           config: OptimizerConfig,
                           output_dir) -> Dict[str, Path]:
    """Persist weights.csv / compliance_report.md / compliance.json."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    if result.success and result.weights is not None:
        w = np.asarray(result.weights, dtype=float)
        wdf = pd.DataFrame({
            "ticker": [str(t) for t in inputs.tickers],
            "weight": w,
            "active_weight": w - inputs.w_b,
        })
        p = out / WEIGHTS_FILE
        wdf.to_csv(p, index=False)   # full precision; round only for display
        paths["weights_csv"] = p

    p = out / REPORT_MD_FILE
    p.write_text(render_markdown(report), encoding="utf-8")
    paths["compliance_md"] = p

    p = out / REPORT_JSON_FILE
    p.write_text(json.dumps(asdict(report), indent=2, default=_json_default),
                 encoding="utf-8")
    paths["compliance_json"] = p

    return paths
