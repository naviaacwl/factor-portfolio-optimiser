"""Covariance assembly and regularization (the multicollinearity defense).

Why this module exists
----------------------
Manual covariance estimation from historical returns is error-prone and, with
correlated factors or many assets, the sample covariance becomes
ill-conditioned (near-singular). An ill-conditioned Sigma makes the QP
solution unstable: tiny input changes flip weights wildly, and numerical
solvers can fail or return garbage. The defenses here, applied in order:

1. **Factor-model structure**   ``Sigma = B F B' + D``
   Imposes a rank-K + diagonal structure instead of estimating all n(n+1)/2
   entries directly (K << n), which is the classical remedy for both
   estimation noise and collinearity among assets.
2. **Linear shrinkage**         ``(1-delta) * S + delta * T``
   with ``T = diag(S)`` or ``(trace(S)/n) * I`` (Ledoit-Wolf style). Pulls
   the estimate toward a well-conditioned target; ``delta = 0`` recovers the
   unregularized baseline used in the comparison experiments.
3. **Relative ridge**           ``S + eps * mean(diag(S)) * I``
   A tiny PSD safety net so the quadratic form is always convex for the
   solver even if upstream rounding produced a slightly negative eigenvalue.

Condition numbers before/after regularization are exposed (and reported in
the compliance report) so the effect is auditable, not hidden.
"""
from __future__ import annotations

import numpy as np


def _symmetrize(S: np.ndarray) -> np.ndarray:
    return 0.5 * (S + S.T)


def _as_square(S, name: str) -> np.ndarray:
    S = np.asarray(S, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape {S.shape}")
    return S


def factor_model_covariance(B, F, residual_var) -> np.ndarray:
    """``Sigma = B F B' + diag(residual_var)`` (annual units, symmetric)."""
    B = np.asarray(B, dtype=float)
    F = _as_square(F, "factor covariance F")
    d = np.asarray(residual_var, dtype=float).ravel()
    if B.ndim != 2:
        raise ValueError(f"factor loadings B must be 2-D (n x k), got shape {B.shape}")
    n, k = B.shape
    if F.shape != (k, k):
        raise ValueError(
            f"F has shape {F.shape} but B implies k = {k} factors")
    if d.shape != (n,):
        raise ValueError(
            f"residual_var has shape {d.shape}, expected ({n},) to match B rows")
    if np.any(d < 0):
        raise ValueError("residual_var contains negative values")
    return _symmetrize(B @ F @ B.T + np.diag(d))


def shrink_covariance(S, delta: float, target: str = "diagonal") -> np.ndarray:
    """Linear shrinkage ``(1-delta)*S + delta*T``.

    target = "diagonal"        : T = diag(S)          (kills correlations)
    target = "scaled_identity" : T = (trace(S)/n) * I (constant-variance model)

    ``delta = 0`` returns S unchanged (the unregularized baseline);
    ``delta = 1`` returns the pure target.
    """
    S = _symmetrize(_as_square(S, "covariance S"))
    if not (0.0 <= float(delta) <= 1.0):
        raise ValueError(f"shrinkage intensity delta must be in [0, 1], got {delta}")
    n = S.shape[0]
    if target == "diagonal":
        T = np.diag(np.diag(S))
    elif target == "scaled_identity":
        T = (np.trace(S) / n) * np.eye(n)
    else:
        raise ValueError(
            f"unknown shrinkage target {target!r}; use 'diagonal' or 'scaled_identity'")
    return _symmetrize((1.0 - float(delta)) * S + float(delta) * T)


def ridge_covariance(S, epsilon: float) -> np.ndarray:
    """Add a *relative* ridge: ``S + eps * mean(diag(S)) * I``.

    Relative scaling keeps the ridge meaningful across unit conventions
    (daily vs annual variances). ``epsilon = 0`` returns a copy of S.
    """
    S = _symmetrize(_as_square(S, "covariance S"))
    eps = float(epsilon)
    if eps < 0:
        raise ValueError(f"ridge epsilon must be >= 0, got {epsilon}")
    if eps == 0.0:
        return S.copy()
    n = S.shape[0]
    trace = float(np.trace(S))
    scale = trace / n if trace > 0 else 1.0
    return _symmetrize(S + eps * scale * np.eye(n))


def nearest_psd(S, eigen_floor: float = 0.0) -> np.ndarray:
    """Clip negative eigenvalues to ``eigen_floor`` (spectral decomposition).

    Provided as an explicit, *documented* repair for user-supplied
    covariances that fail the PSD check. The pipeline never calls this
    silently: input validation raises instead, and using this function is
    recorded in run metadata.
    """
    S = _symmetrize(_as_square(S, "covariance S"))
    vals, vecs = np.linalg.eigh(S)
    vals = np.clip(vals, float(eigen_floor), None)
    return _symmetrize(vecs @ np.diag(vals) @ vecs.T)


def condition_number(S) -> float:
    """lambda_max / lambda_min (symmetric eigendecomposition). ``inf`` if singular."""
    S = _symmetrize(_as_square(S, "covariance S"))
    vals = np.linalg.eigvalsh(S)
    vmax, vmin = float(vals[-1]), float(vals[0])
    if vmax <= 0.0 or vmin <= 0.0:
        return float("inf")
    return vmax / vmin


def min_eigenvalue(S) -> float:
    """Smallest eigenvalue; negative values indicate an indefinite matrix."""
    S = _symmetrize(_as_square(S, "covariance S"))
    return float(np.linalg.eigvalsh(S)[0])


def sample_covariance(returns, periods_per_year: int = 252) -> np.ndarray:
    """Unbiased sample covariance of a (T x n) return frame, annualized.

    Accepts a pandas DataFrame (rows = dates, cols = assets) or 2-D array.
    This is the *unregularized* estimator used as a comparison baseline.
    """
    X = returns.values if hasattr(returns, "values") else np.asarray(returns)
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"returns must be 2-D (T x n), got shape {X.shape}")
    if X.shape[0] < 2:
        raise ValueError("need at least 2 observations for a sample covariance")
    S = np.atleast_2d(np.cov(X, rowvar=False, ddof=1))
    return _symmetrize(S * float(periods_per_year))
