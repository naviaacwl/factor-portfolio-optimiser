"""Convex QP core (CVXPY): benchmark-relative active-risk optimization.

Problem, with active weights ``a = w - w_b`` and regularized covariance
``Sigma`` (factor model + shrinkage + ridge, or an override for baselines):

    minimize    0.5 * a' Sigma a  -  gamma * mu' a  +  kappa * w' w
    subject to  1' w = 1                              (fully invested)
                w_min <= w <= w_max                   (box / single-stock cap)
                lo <= B' (w - w_b) <= hi              (active factor tilts)
                ||w - w_prev||_1 <= turnover_cap      (optional)

Convexity argument (the "mathematical optimality" claim):
  * ``Sigma`` is PSD by construction (factor model of a PSD F plus a
    nonnegative diagonal, then shrinkage toward a PSD target and a
    nonnegative ridge preserve PSD), so ``a' Sigma a`` is convex in w;
  * ``mu' a`` is affine; ``w' w`` is convex; gamma, kappa >= 0
    => the objective is a convex quadratic;
  * all constraints are affine (the optional turnover constraint is a
    1-norm, also convex).
  Therefore any local optimum is a global optimum, and with kappa > 0 the
  objective is strictly convex and the optimum is unique.

Infeasibility is *reported*, not raised: the result carries
``status='infeasible'`` so downstream code (compliance report, verification
suite) can exercise the edge case gracefully.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import covariance as cov
from .config import OptimizerConfig
from .data import PortfolioInputs

try:
    import cvxpy as cp
except ImportError:  # pragma: no cover - environment guard
    cp = None


class SolverNotAvailableError(RuntimeError):
    """Raised when cvxpy is not installed."""


@dataclass
class OptimizationResult:
    """Full record of one solve: solution, risk analytics, and diagnostics."""

    status: str                       # optimal | infeasible | unbounded | solver_error
    success: bool
    message: str = ""
    weights: Optional[np.ndarray] = None
    active_weights: Optional[np.ndarray] = None
    objective_value: Optional[float] = None
    sigma: Optional[np.ndarray] = None
    sigma_source: str = ""
    expected_return: Optional[float] = None
    expected_active_return: Optional[float] = None
    tracking_error_annual: Optional[float] = None
    portfolio_exposures: Optional[np.ndarray] = None
    active_exposures: Optional[np.ndarray] = None
    condition_number_raw: Optional[float] = None
    condition_number_regularized: Optional[float] = None
    shrinkage_intensity: Optional[float] = None
    max_constraint_violation: Optional[float] = None
    solve_time_seconds: Optional[float] = None
    solver_name: Optional[str] = None
    config: Optional[OptimizerConfig] = None


def objective_value_np(w, mu, w_b, Sigma, gamma: float, kappa: float) -> float:
    """Plain-numpy evaluation of the QP objective (used by cross-checks)."""
    w = np.asarray(w, dtype=float)
    a = w - np.asarray(w_b, dtype=float)
    return float(0.5 * a @ Sigma @ a - gamma * (np.asarray(mu, dtype=float) @ a)
                 + kappa * (w @ w))


def _max_constraint_violation(w, inputs: PortfolioInputs,
                              config: OptimizerConfig,
                              tilt_lo: np.ndarray, tilt_hi: np.ndarray) -> float:
    """Worst violation across all constraints (0 = feasible)."""
    a = w - inputs.w_b
    tilt = inputs.B.T @ a
    w_min = config.effective_min_weight()
    viols = [
        abs(float(w.sum()) - 1.0),
        float(np.max(np.clip(w_min - w, 0.0, None))),
        float(np.max(np.clip(w - config.max_weight, 0.0, None))),
        float(np.max(np.clip(tilt_lo - tilt, 0.0, None))) if len(tilt) else 0.0,
        float(np.max(np.clip(tilt - tilt_hi, 0.0, None))) if len(tilt) else 0.0,
    ]
    if config.turnover_cap is not None:
        w_prev = np.asarray(config.previous_weights, dtype=float)
        viols.append(max(0.0, float(np.abs(w - w_prev).sum()) - config.turnover_cap))
    return float(max(viols))


def solve_factor_portfolio(inputs: PortfolioInputs,
                           config: OptimizerConfig,
                           *,
                           sigma_override: Optional[np.ndarray] = None,
                           sigma_label: str = "") -> OptimizationResult:
    """Solve the benchmark-relative factor QP.

    Parameters
    ----------
    inputs          validated PortfolioInputs
    config          OptimizerConfig (objective weights, regularization, bounds)
    sigma_override  optional (n x n) covariance used INSTEAD of the regularized
                    factor-model covariance -- this is how the verification
                    suite builds the *unregularized sample covariance* baseline
                    under identical constraints and objective.
    sigma_label     provenance string recorded in the result for the override.
    """
    if cp is None:
        raise SolverNotAvailableError(
            "cvxpy is not installed. Run: pip install -r requirements.txt")

    n = inputs.n_assets
    w_b = np.asarray(inputs.w_b, dtype=float)
    mu = np.asarray(inputs.mu, dtype=float)

    # --- covariance assembly + regularization ---------------------------
    if sigma_override is not None:
        S = np.asarray(sigma_override, dtype=float)
        if S.shape != (n, n):
            raise ValueError(
                f"sigma_override has shape {S.shape}, expected ({n}, {n})")
        Sigma = 0.5 * (S + S.T)
        cond_raw = cov.condition_number(Sigma)
        cond_reg = cond_raw
        source = sigma_label or "override (as supplied; symmetrized only)"
        delta_used = 0.0
    else:
        Sigma_raw = cov.factor_model_covariance(inputs.B, inputs.F,
                                                inputs.residual_var)
        cond_raw = cov.condition_number(Sigma_raw)
        Sigma_shrunk = cov.shrink_covariance(
            Sigma_raw, config.shrinkage_intensity, config.shrinkage_target)
        Sigma = cov.ridge_covariance(Sigma_shrunk, config.ridge_epsilon)
        cond_reg = cov.condition_number(Sigma)
        source = (f"factor model B F B' + D; shrinkage "
                  f"{config.shrinkage_intensity:g} -> {config.shrinkage_target}; "
                  f"ridge {config.ridge_epsilon:g}")
        delta_used = config.shrinkage_intensity

    w_min = config.effective_min_weight()
    tilt_lo, tilt_hi = config.resolve_tilt_bounds(inputs.factor_names)
    Bt = np.asarray(inputs.B, dtype=float).T

    # --- CVXPY model -----------------------------------------------------
    w = cp.Variable(n)
    a = w - w_b
    tilt = Bt @ a
    objective = (0.5 * cp.quad_form(a, cp.psd_wrap(Sigma))
                 - config.risk_aversion * (mu @ a)
                 + config.concentration_penalty * cp.sum_squares(w))
    constraints = [
        cp.sum(w) == 1,
        w >= w_min,
        w <= config.max_weight,
        tilt >= tilt_lo,
        tilt <= tilt_hi,
    ]
    if config.turnover_cap is not None:
        w_prev = np.asarray(config.previous_weights, dtype=float)
        if w_prev.shape != (n,):
            raise ValueError(
                f"previous_weights has shape {w_prev.shape}, expected ({n},)")
        constraints.append(cp.norm(w - w_prev, 1) <= config.turnover_cap)

    problem = cp.Problem(cp.Minimize(objective), constraints)
    t0 = time.perf_counter()
    try:
        if config.solver:
            problem.solve(solver=config.solver, **config.solver_kwargs)
        else:
            problem.solve(**config.solver_kwargs)
    except cp.SolverError as exc:
        return OptimizationResult(
            status="solver_error", success=False,
            message=(f"solver failed: {exc}. Try config.solver='CLARABEL' or "
                     "'OSQP', or check input scaling."),
            sigma=Sigma, sigma_source=source,
            condition_number_raw=cond_raw, condition_number_regularized=cond_reg,
            shrinkage_intensity=delta_used,
            solve_time_seconds=time.perf_counter() - t0, config=config)
    elapsed = time.perf_counter() - t0
    solver_used = None
    if problem.solver_stats is not None:
        solver_used = problem.solver_stats.solver_name

    diagnostics = dict(sigma=Sigma, sigma_source=source,
                       condition_number_raw=cond_raw,
                       condition_number_regularized=cond_reg,
                       shrinkage_intensity=delta_used,
                       solve_time_seconds=elapsed, solver_name=solver_used,
                       config=config)

    status = problem.status
    if status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        if w.value is None:
            return OptimizationResult(
                status="solver_error", success=False,
                message="solver reported optimal but returned no weights",
                **diagnostics)
        w_val = np.asarray(w.value, dtype=float).ravel()
        active = w_val - w_b
        te = float(np.sqrt(max(float(active @ Sigma @ active), 0.0)))
        exposures = Bt @ w_val
        violation = _max_constraint_violation(w_val, inputs, config,
                                              tilt_lo, tilt_hi)
        msg = ""
        if status == cp.OPTIMAL_INACCURATE:
            msg = ("solver returned OPTIMAL_INACCURATE; treat numbers with "
                   "caution, consider another solver or tighter tolerances")
        return OptimizationResult(
            status="optimal", success=True, message=msg,
            weights=w_val, active_weights=active,
            objective_value=objective_value_np(
                w_val, mu, w_b, Sigma,
                config.risk_aversion, config.concentration_penalty),
            expected_return=float(mu @ w_val),
            expected_active_return=float(mu @ active),
            tracking_error_annual=te,
            portfolio_exposures=exposures,
            active_exposures=exposures - Bt @ w_b,
            max_constraint_violation=violation,
            **diagnostics)

    if status in (cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE):
        return OptimizationResult(
            status="infeasible", success=False,
            message=(f"constraints cannot be satisfied simultaneously (solver "
                     f"status: {status}). Relax tilt bounds, position caps, or "
                     "review contradictory config (e.g. tilt lo > hi, box that "
                     "cannot sum to 1)."),
            **diagnostics)

    if status in (cp.UNBOUNDED, cp.UNBOUNDED_INACCURATE):
        return OptimizationResult(
            status="unbounded", success=False,
            message=(f"problem reported unbounded (status: {status}); with a "
                     "PSD Sigma and box constraints this should not happen -- "
                     "check Sigma regularization settings."),
            **diagnostics)

    return OptimizationResult(
        status="solver_error", success=False,
        message=f"unexpected solver status: {status}", **diagnostics)
