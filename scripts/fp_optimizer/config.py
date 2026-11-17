"""Configuration for the factor-portfolio QP optimizer.

All tunable knobs live in one dataclass so a run is fully reproducible from
its config file.

Units convention (project-wide):
  * expected returns ``mu``        : ANNUAL, decimals (0.08 = 8%)
  * factor covariance ``F``        : ANNUAL, decimals of variance
  * residual variances             : ANNUAL, decimals of variance
  * weights                        : fractions of portfolio value (1.0 = 100%)
  * tilt bounds                    : same units as factor loadings x weights

Design note: constraint *feasibility* is intentionally NOT pre-judged here.
Contradictory bounds (e.g. tilt lo > hi, or a box that cannot sum to 1) are
passed through to the solver so the infeasible path is exercised and reported
gracefully -- see ``fp_optimizer.verify.check_infeasible_edge_case``.
Only clearly invalid *scalars* (negative penalties, max_weight out of range,
unknown keys) raise :class:`ConfigError`.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

VALID_SHRINKAGE_TARGETS = ("diagonal", "scaled_identity")


class ConfigError(ValueError):
    """Raised when a configuration value is out of its valid range."""


@dataclass
class OptimizerConfig:
    """Knobs for the objective, covariance regularization, and constraints.

    Objective, with active weights ``a = w - w_b``:

        minimize  0.5 * a' Sigma a  -  risk_aversion * mu' a
                        +  concentration_penalty * w' w

    * ``0.5 * a' Sigma a``  -> tracking-error variance (active risk)
    * ``- risk_aversion * mu' a`` -> expected active return (reward)
    * ``concentration_penalty * w' w`` -> single-stock concentration penalty
      (w'w is the Herfindahl-Hirschman Index of the portfolio; penalizing it
      pushes toward diversified, "effective-N"-friendly solutions)
    """

    # --- objective weights ---
    risk_aversion: float = 1.0
    concentration_penalty: float = 0.05

    # --- covariance regularization (multicollinearity defense) ---
    # Sigma_used = (1-delta)*Sigma + delta*Target  +  eps*mean(diag)*I
    shrinkage_intensity: float = 0.10
    shrinkage_target: str = "diagonal"
    ridge_epsilon: float = 1e-8

    # --- position constraints ---
    long_only: bool = True
    min_weight: Optional[float] = None   # default: 0.0 if long_only else -1.0
    max_weight: float = 0.30             # single-stock cap

    # --- active factor tilt bounds ---
    # factor_name -> [lower_bound, upper_bound] on B'(w - w_b)
    tilt_bounds: Optional[Dict[str, List[float]]] = None
    default_tilt_bound: float = 0.10     # symmetric bound for unlisted factors

    # --- optional turnover cap: sum_i |w_i - w_prev_i| <= cap ---
    turnover_cap: Optional[float] = None
    previous_weights: Optional[List[float]] = None

    # --- numerics / reporting ---
    sum_to_one_tol: float = 1e-6
    solver: Optional[str] = None         # None -> cvxpy default; e.g. "CLARABEL", "OSQP"
    solver_kwargs: Dict[str, Any] = field(default_factory=dict)
    periods_per_year: int = 252

    def __post_init__(self) -> None:
        self.validate()

    # ------------------------------------------------------------------
    def validate(self) -> None:
        """Validate scalar ranges and types. Does NOT judge feasibility."""
        if self.risk_aversion < 0:
            raise ConfigError(f"risk_aversion must be >= 0, got {self.risk_aversion}")
        if self.concentration_penalty < 0:
            raise ConfigError(
                f"concentration_penalty must be >= 0, got {self.concentration_penalty}")
        if not (0.0 <= self.shrinkage_intensity <= 1.0):
            raise ConfigError(
                f"shrinkage_intensity must be in [0, 1], got {self.shrinkage_intensity}")
        if self.shrinkage_target not in VALID_SHRINKAGE_TARGETS:
            raise ConfigError(
                f"shrinkage_target must be one of {VALID_SHRINKAGE_TARGETS}, "
                f"got {self.shrinkage_target!r}")
        if self.ridge_epsilon < 0:
            raise ConfigError(f"ridge_epsilon must be >= 0, got {self.ridge_epsilon}")
        if not (0.0 < self.max_weight <= 1.0):
            raise ConfigError(f"max_weight must be in (0, 1], got {self.max_weight}")
        if self.min_weight is not None and self.min_weight > self.max_weight:
            # Still allowed through to the solver (it will report infeasible),
            # but flag it loudly since it is almost always a typo:
            raise ConfigError(
                f"min_weight ({self.min_weight}) exceeds max_weight ({self.max_weight}); "
                "this makes every problem infeasible by construction")
        if self.default_tilt_bound < 0:
            raise ConfigError(
                f"default_tilt_bound must be >= 0, got {self.default_tilt_bound}")
        for name, bounds in (self.tilt_bounds or {}).items():
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise ConfigError(
                    f"tilt_bounds[{name!r}] must be a [lo, hi] pair, got {bounds!r}")
            if not all(isinstance(x, (int, float)) and np.isfinite(x) for x in bounds):
                raise ConfigError(
                    f"tilt_bounds[{name!r}] must contain finite numbers, got {bounds!r}")
        if self.turnover_cap is not None:
            if self.turnover_cap < 0:
                raise ConfigError(f"turnover_cap must be >= 0, got {self.turnover_cap}")
            if self.previous_weights is None:
                raise ConfigError("turnover_cap requires previous_weights to be set")
        if self.sum_to_one_tol <= 0:
            raise ConfigError(f"sum_to_one_tol must be > 0, got {self.sum_to_one_tol}")
        if self.solver is not None and not isinstance(self.solver, str):
            raise ConfigError(f"solver must be a string or None, got {self.solver!r}")
        if self.periods_per_year <= 0:
            raise ConfigError(
                f"periods_per_year must be > 0, got {self.periods_per_year}")

    # ------------------------------------------------------------------
    def effective_min_weight(self) -> float:
        """Lower box bound actually used by the optimizer."""
        if self.min_weight is not None:
            return float(self.min_weight)
        return 0.0 if self.long_only else -1.0

    def resolve_tilt_bounds(
            self, factor_names) -> Tuple[np.ndarray, np.ndarray]:
        """Return (lo, hi) arrays aligned with ``factor_names``.

        Factors absent from ``tilt_bounds`` get the symmetric
        ``+/- default_tilt_bound``.
        """
        names = [str(f) for f in factor_names]
        lo = np.empty(len(names), dtype=float)
        hi = np.empty(len(names), dtype=float)
        for i, fname in enumerate(names):
            if self.tilt_bounds and fname in self.tilt_bounds:
                b = self.tilt_bounds[fname]
                lo[i], hi[i] = float(b[0]), float(b[1])
            else:
                lo[i] = -float(self.default_tilt_bound)
                hi[i] = float(self.default_tilt_bound)
        return lo, hi

    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizerConfig":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ConfigError(
                f"unknown config keys (typo?): {unknown}; valid keys: {sorted(known)}")
        return cls(**data)

    @classmethod
    def from_json(cls, path) -> "OptimizerConfig":
        p = Path(path)
        if not p.exists():
            raise ConfigError(f"config file not found: {p}")
        with open(p, "r", encoding="utf-8") as fh:
            try:
                data = json.load(fh)
            except json.JSONDecodeError as exc:
                raise ConfigError(f"config file {p} is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ConfigError(f"config file {p} must contain a JSON object")
        return cls.from_dict(data)

    def to_json(self, path, indent: int = 2) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(dataclasses.asdict(self), fh, indent=indent)
        return p

    # ------------------------------------------------------------------
    def describe(self) -> str:
        """Human-readable summary for logs and reports."""
        lines = [
            f"risk_aversion (gamma)      = {self.risk_aversion:g}",
            f"concentration_penalty (k)  = {self.concentration_penalty:g}",
            f"shrinkage_intensity (delta)= {self.shrinkage_intensity:g} "
            f"-> {self.shrinkage_target}",
            f"ridge_epsilon              = {self.ridge_epsilon:g}",
            f"long_only                  = {self.long_only}",
            f"weight box                 = [{self.effective_min_weight():g}, "
            f"{self.max_weight:g}]",
            f"default_tilt_bound         = +/-{self.default_tilt_bound:g}",
            f"tilt_bounds                = {self.tilt_bounds}",
            f"turnover_cap               = {self.turnover_cap}",
            f"solver                     = {self.solver or 'cvxpy default'}",
        ]
        return "\n".join(lines)
