"""Portfolio input container, file loading with validation, and builders
that estimate factor-model inputs from historical return data.

Input pack layout (a directory, see ``examples/basic/``):

    expected_returns.csv    ticker,mu                      (mu: ANNUAL decimal)
    factor_loadings.csv     ticker,<factor_1>,...,<factor_K>
    factor_covariance.csv   factor,<factor_1>,...,<factor_K>   (K x K, ANNUAL)
    benchmark_weights.csv   ticker,weight                  (must sum to 1.0)
    residual_variance.csv   ticker,residual_variance       (ANNUAL, >= 0)
    config.json             optional OptimizerConfig overrides

All frames are aligned on the ticker order of ``expected_returns.csv`` and the
factor order of ``factor_loadings.csv`` columns. Validation is strict and
fails loudly with the offending file/field named -- per the course guide:
"do not silently invent or drop data; apply a declared rule or stop with an
explanation."
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

EXPECTED_RETURNS_FILE = "expected_returns.csv"
FACTOR_LOADINGS_FILE = "factor_loadings.csv"
FACTOR_COVARIANCE_FILE = "factor_covariance.csv"
BENCHMARK_WEIGHTS_FILE = "benchmark_weights.csv"
RESIDUAL_VARIANCE_FILE = "residual_variance.csv"
CONFIG_FILE = "config.json"


class InputValidationError(ValueError):
    """Raised when supplied inputs fail validation.

    Messages always name the offending file and field so the failure is
    actionable (e.g. 'benchmark_weights.csv: weights sum to 0.9, expected 1.0').
    """


@dataclass
class PortfolioInputs:
    """Everything the QP needs, in aligned numpy arrays.

    Attributes
    ----------
    tickers       (n,) asset identifiers, aligned with mu / B rows / w_b
    factor_names  (k,) factor identifiers, aligned with B columns / F rows
    mu            (n,) annual expected returns, decimals
    B             (n,k) factor loadings (exposure matrix)
    F             (k,k) annual factor covariance (symmetric PSD)
    residual_var  (n,) annual idiosyncratic variances (>= 0)
    w_b           (n,) benchmark weights, summing to 1.0
    metadata      provenance (source dir / mode / retrieval time / ...)
    """

    tickers: np.ndarray
    factor_names: np.ndarray
    mu: np.ndarray
    B: np.ndarray
    F: np.ndarray
    residual_var: np.ndarray
    w_b: np.ndarray
    metadata: Dict = field(default_factory=dict)

    @property
    def n_assets(self) -> int:
        return int(self.mu.shape[0])

    @property
    def n_factors(self) -> int:
        return int(self.F.shape[0])

    def benchmark_exposures(self) -> np.ndarray:
        """B' w_b  -- the benchmark's own factor exposures."""
        return self.B.T @ self.w_b

    def portfolio_exposures(self, w) -> np.ndarray:
        return self.B.T @ np.asarray(w, dtype=float)

    def active_exposures(self, w) -> np.ndarray:
        """B' (w - w_b) -- the active factor tilts."""
        return self.portfolio_exposures(w) - self.benchmark_exposures()


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------
def validate_inputs(pi: PortfolioInputs, *,
                    benchmark_sum_tol: float = 1e-6,
                    psd_tol: float = 1e-8) -> None:
    """Strict structural + financial validation. Raises InputValidationError."""
    n, k = pi.n_assets, pi.n_factors
    if n < 1:
        raise InputValidationError("empty universe (n_assets = 0)")
    if k < 1:
        raise InputValidationError("no factors (n_factors = 0)")

    expected_shapes = {
        "mu": (n,), "B": (n, k), "F": (k, k),
        "residual_var": (n,), "w_b": (n,),
        "tickers": (n,), "factor_names": (k,),
    }
    for name, shape in expected_shapes.items():
        arr = getattr(pi, name)
        if np.asarray(arr).shape != shape:
            raise InputValidationError(
                f"{name}: shape {np.asarray(arr).shape} does not match expected {shape}")

    for name, arr in (("mu (expected_returns)", pi.mu),
                      ("B (factor_loadings)", pi.B),
                      ("F (factor_covariance)", pi.F),
                      ("residual_variance", pi.residual_var),
                      ("w_b (benchmark_weights)", pi.w_b)):
        if not np.all(np.isfinite(np.asarray(arr, dtype=float))):
            raise InputValidationError(f"{name}: contains NaN or infinite values")

    if len(set(pi.tickers.tolist())) != n:
        raise InputValidationError("tickers contain duplicates")

    if not np.allclose(pi.F, pi.F.T, atol=1e-10):
        asym = float(np.max(np.abs(pi.F - pi.F.T)))
        raise InputValidationError(
            f"F (factor_covariance) is not symmetric (max asymmetry {asym:.3e}); "
            "fix the source file or symmetrize explicitly")
    f_eig = np.linalg.eigvalsh(pi.F)
    scale = max(float(np.abs(pi.F).max()), 1e-12)
    if float(f_eig[0]) < -psd_tol * scale:
        raise InputValidationError(
            f"F (factor_covariance) is not positive semi-definite "
            f"(min eigenvalue {float(f_eig[0]):.3e}); fix the estimate "
            "(or apply covariance.nearest_psd explicitly and document it)")
    if np.any(pi.residual_var < 0):
        raise InputValidationError("residual_variance contains negative values")

    w_sum = float(pi.w_b.sum())
    if abs(w_sum - 1.0) > benchmark_sum_tol:
        raise InputValidationError(
            f"benchmark_weights sum to {w_sum:.10f}, expected 1.0 within "
            f"{benchmark_sum_tol:g}")
    if np.any(pi.w_b < -1e-9):
        raise InputValidationError(
            "benchmark_weights contain negative weights (short benchmark "
            "positions are out of scope for this tool)")


# ----------------------------------------------------------------------
# File loading
# ----------------------------------------------------------------------
def _read_csv_checked(path: Path, required_cols: List[str]) -> pd.DataFrame:
    if not path.exists():
        raise InputValidationError(f"missing required input file: {path}")
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # parser errors -> actionable message
        raise InputValidationError(f"{path.name}: unreadable CSV ({exc})") from exc
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise InputValidationError(
            f"{path.name}: missing column(s) {missing}; found {list(df.columns)}")
    return df


def _align_by_ticker(df: pd.DataFrame, tickers: List[str], fname: str,
                     value_col: Optional[str] = None):
    """Reindex ``df`` to ``tickers``; raise on set mismatch or duplicates."""
    frame_tickers = df["ticker"].astype(str).tolist()
    if len(set(frame_tickers)) != len(frame_tickers):
        raise InputValidationError(f"{fname}: duplicate tickers")
    missing = sorted(set(tickers) - set(frame_tickers))
    extra = sorted(set(frame_tickers) - set(tickers))
    if missing or extra:
        raise InputValidationError(
            f"{fname}: ticker set does not match {EXPECTED_RETURNS_FILE} "
            f"(missing: {missing}, extra: {extra})")
    out = df.set_index("ticker")
    out.index = out.index.astype(str)
    out = out.loc[tickers]
    return out[value_col] if value_col else out


def load_inputs_from_dir(input_dir, *,
                         benchmark_sum_tol: float = 1e-6,
                         require_residual_variance: bool = True) -> PortfolioInputs:
    """Load and validate a full input pack directory."""
    d = Path(input_dir)
    if not d.is_dir():
        raise InputValidationError(f"input directory not found: {d}")

    er = _read_csv_checked(d / EXPECTED_RETURNS_FILE, ["ticker", "mu"])
    fl = _read_csv_checked(d / FACTOR_LOADINGS_FILE, ["ticker"])
    fc = _read_csv_checked(d / FACTOR_COVARIANCE_FILE, ["factor"])
    bw = _read_csv_checked(d / BENCHMARK_WEIGHTS_FILE, ["ticker", "weight"])

    tickers = er["ticker"].astype(str).tolist()
    if len(set(tickers)) != len(tickers):
        raise InputValidationError(f"{EXPECTED_RETURNS_FILE}: duplicate tickers")

    factor_names = [str(c) for c in fl.columns if c != "ticker"]
    if not factor_names:
        raise InputValidationError(
            f"{FACTOR_LOADINGS_FILE}: no factor columns besides 'ticker'")

    mu = er["mu"].to_numpy(dtype=float)
    fl_aligned = _align_by_ticker(fl, tickers, FACTOR_LOADINGS_FILE)
    B = fl_aligned[factor_names].to_numpy(dtype=float)
    w_b = _align_by_ticker(bw, tickers, BENCHMARK_WEIGHTS_FILE, "weight").to_numpy(dtype=float)

    # factor covariance: row and column factor sets must match the loadings
    # NOTE: set_index("factor") by NAME (passing a Series would keep the column)
    fc = fc.set_index("factor")
    fc.index = fc.index.astype(str)
    fc.columns = [str(c) for c in fc.columns]
    fc_cols = list(fc.columns)
    if set(fc.index) != set(factor_names) or set(fc_cols) != set(factor_names):
        raise InputValidationError(
            f"{FACTOR_COVARIANCE_FILE}: factor names {sorted(set(fc.index) | set(fc_cols))} "
            f"do not match {FACTOR_LOADINGS_FILE} factors {sorted(factor_names)}")
    F = fc.loc[factor_names, factor_names].to_numpy(dtype=float)

    rv_path = d / RESIDUAL_VARIANCE_FILE
    if rv_path.exists():
        rv = _read_csv_checked(rv_path, ["ticker", "residual_variance"])
        residual_var = _align_by_ticker(
            rv, tickers, RESIDUAL_VARIANCE_FILE, "residual_variance").to_numpy(dtype=float)
    elif require_residual_variance:
        raise InputValidationError(
            f"{RESIDUAL_VARIANCE_FILE} not found in {d}. Provide per-asset "
            "idiosyncratic variances (recommended), or pass "
            "require_residual_variance=False to model pure factor risk.")
    else:
        residual_var = np.zeros(len(tickers))

    pi = PortfolioInputs(
        tickers=np.array(tickers),
        factor_names=np.array(factor_names),
        mu=mu, B=B, F=F, residual_var=residual_var, w_b=w_b,
        metadata={
            "source_dir": str(d.resolve()),
            "loaded_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "residual_variance_provided": rv_path.exists(),
        },
    )
    validate_inputs(pi, benchmark_sum_tol=benchmark_sum_tol)
    return pi


# ----------------------------------------------------------------------
# Builders from return data (real or simulated)
# ----------------------------------------------------------------------
def equal_weight(n: int) -> np.ndarray:
    """The 1/N reference portfolio."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    return np.full(n, 1.0 / n)


def build_inputs_from_returns(asset_returns: pd.DataFrame,
                              factor_returns: pd.DataFrame,
                              *,
                              benchmark_weights=None,
                              mu=None,
                              periods_per_year: int = 252,
                              min_obs: Optional[int] = None
                              ) -> Tuple[PortfolioInputs, Dict]:
    """Estimate factor-model inputs from aligned return series (OLS).

    Model per asset i:   r_it = alpha_i + B_i' f_t + eps_it

    * ``B``            : OLS slopes of asset returns on factor returns
    * ``residual_var`` : var(eps_i) * periods_per_year  (ddof = K + 1)
    * ``F``            : unbiased sample covariance of factor returns,
                         annualized (NO shrinkage here -- regularization is
                         the optimizer's documented, configurable step)
    * ``mu``           : user override if given; otherwise historical mean *
                         periods_per_year. WARNING: sample means are noisy
                         expected-return estimates; all comparison variants
                         use the same mu so the comparison stays fair
                         (see report.md, Limitations).
    * ``w_b``          : user-supplied (must sum to 1) or equal weight (1/N).

    Returns (PortfolioInputs, provenance dict).
    """
    if not isinstance(asset_returns, pd.DataFrame):
        raise TypeError("asset_returns must be a pandas DataFrame (dates x tickers)")
    if not isinstance(factor_returns, pd.DataFrame):
        raise TypeError("factor_returns must be a pandas DataFrame (dates x factors)")
    if asset_returns.shape[1] < 1 or factor_returns.shape[1] < 1:
        raise InputValidationError("asset_returns and factor_returns must have >= 1 column")

    common = asset_returns.index.intersection(factor_returns.index)
    ar = asset_returns.loc[common].dropna(how="any")
    fr = factor_returns.loc[ar.index].dropna(how="any")
    ar = ar.loc[fr.index]

    T, n = ar.shape
    k = fr.shape[1]
    need = min_obs if min_obs is not None else max(k + 2, 30)
    if T < need:
        raise InputValidationError(
            f"only {T} aligned complete observations; need at least {need} "
            "(check date overlap and NaN handling of the supplied series)")

    X = np.column_stack([np.ones(T), fr.to_numpy(dtype=float)])
    R = ar.to_numpy(dtype=float)
    coef, *_ = np.linalg.lstsq(X, R, rcond=None)          # (k+1, n)
    B = coef[1:].T                                         # (n, k)
    resid = R - X @ coef                                   # (T, n)
    residual_var = np.var(resid, axis=0, ddof=k + 1) * periods_per_year
    F = np.atleast_2d(
        np.cov(fr.to_numpy(dtype=float), rowvar=False, ddof=1) * periods_per_year)

    if mu is None:
        mu_vec = ar.mean(axis=0).to_numpy(dtype=float) * periods_per_year
        mu_source = "historical mean (noisy in-sample estimate)"
    else:
        mu_s = pd.Series(mu)
        if set(map(str, mu_s.index)) != set(map(str, ar.columns)):
            raise InputValidationError(
                "mu override tickers do not match asset_returns columns")
        mu_vec = mu_s.reindex(ar.columns).to_numpy(dtype=float)
        mu_source = "user override"

    if benchmark_weights is None:
        w_b = equal_weight(n)
        bm_source = "equal weight (1/N)"
    else:
        bw = pd.Series(benchmark_weights)
        bw = bw.reindex(ar.columns)
        if bw.isna().any():
            raise InputValidationError(
                "benchmark_weights are missing tickers present in asset_returns")
        w_b = bw.to_numpy(dtype=float)
        s = float(w_b.sum())
        if abs(s - 1.0) > 1e-6:
            raise InputValidationError(
                f"benchmark_weights sum to {s:.10f}, expected 1.0 (values are "
                "never silently renormalized)")
        bm_source = "user supplied"

    pi = PortfolioInputs(
        tickers=np.array([str(c) for c in ar.columns]),
        factor_names=np.array([str(c) for c in fr.columns]),
        mu=mu_vec, B=B, F=F, residual_var=residual_var, w_b=w_b,
        metadata={"mode": "estimated_from_returns"},
    )
    provenance = {
        "n_obs": int(T),
        "first_date": str(ar.index.min()),
        "last_date": str(ar.index.max()),
        "periods_per_year": periods_per_year,
        "mu_source": mu_source,
        "benchmark_source": bm_source,
        "estimation": "OLS factor model; F = sample factor covariance; "
                      "residual_var with ddof = K+1",
    }
    validate_inputs(pi)
    return pi, provenance


def fetch_yfinance_inputs(tickers, factor_tickers, start, end, *,
                          benchmark_weights=None, mu=None,
                          auto_adjust: bool = True,
                          periods_per_year: int = 252
                          ) -> Tuple[PortfolioInputs, Dict]:
    """Real-data mode: download adjusted daily closes from Yahoo Finance.

    ``factor_tickers`` are ETF proxies used as observable factor returns
    (e.g. SPY for market, USMV for low-vol, VLUE for value). Records the
    provenance required by the course guide (Appendix B): tickers, window,
    retrieval timestamp, price-adjustment setting, rows used.

    Raises RuntimeError with an actionable message when yfinance is missing
    or the download is empty -- never fabricates data.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "yfinance is required for real-data mode: pip install yfinance. "
            "Alternatively use the synthetic pack under examples/basic or "
            "`prepare_inputs.py synthetic`.") from exc

    tickers = [str(t) for t in tickers]
    factor_tickers = [str(t) for t in factor_tickers]
    all_tickers = list(dict.fromkeys(tickers + factor_tickers))

    data = yf.download(all_tickers, start=start, end=end,
                       auto_adjust=auto_adjust, progress=False)
    if isinstance(data.columns, pd.MultiIndex) and "Close" in data.columns.get_level_values(0):
        prices = data["Close"]
    elif "Close" in getattr(data, "columns", []):
        prices = data[["Close"]].rename(columns={"Close": all_tickers[0]})
    else:
        raise RuntimeError(
            f"unexpected yfinance payload (columns={getattr(data, 'columns', None)}); "
            "check tickers/network or use synthetic mode")
    if prices is None or prices.empty:
        raise RuntimeError(
            f"yfinance returned no data for {all_tickers} over {start}..{end}; "
            "check tickers/window/network, or use synthetic mode")

    prices = prices.dropna(how="any")
    if len(prices) < 60:
        raise RuntimeError(
            f"only {len(prices)} complete daily rows after dropping NaN; need >= 60 "
            "for a stable factor-model estimate")
    rets = prices.pct_change().dropna(how="any")
    asset_returns = rets[tickers]
    factor_returns = rets[factor_tickers]

    pi, prov = build_inputs_from_returns(
        asset_returns, factor_returns,
        benchmark_weights=benchmark_weights, mu=mu,
        periods_per_year=periods_per_year)
    prov.update({
        "mode": "yfinance",
        "retrieved_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "tickers": tickers,
        "factor_tickers": factor_tickers,
        "start": str(start),
        "end": str(end),
        "auto_adjust": bool(auto_adjust),
        "n_rows_used": int(len(asset_returns)),
        "price_basis": "provider-adjusted daily closes; simple returns",
    })
    pi.metadata.update(prov)
    return pi, prov


def synthetic_inputs(n_assets: int = 24, n_factors: int = 3, seed: int = 7,
                     collinear_corr: float = 0.95,
                     periods_per_year: int = 252
                     ) -> Tuple[PortfolioInputs, Dict]:
    """Seeded SYNTHETIC pack with a deliberately near-collinear factor pair.

    Factor 2's loadings are ~0.62 x factor 1's loadings plus noise and F
    embeds correlation ``collinear_corr`` between factors 1 and 2, so the
    value of shrinkage regularization is demonstrable. Data are synthetic:
    no claim about real assets is made or implied.
    """
    if n_assets < 2:
        raise ValueError(f"n_assets must be >= 2, got {n_assets}")
    if n_factors < 1:
        raise ValueError(f"n_factors must be >= 1, got {n_factors}")
    rng = np.random.default_rng(seed)

    vols = np.abs(rng.normal(0.10, 0.04, size=n_factors)) + 0.02
    R = np.eye(n_factors)
    if n_factors >= 2:
        R[0, 1] = R[1, 0] = float(collinear_corr)
    # guarantee PSD: clip eigenvalues, then renormalize to a correlation matrix
    vals, vecs = np.linalg.eigh(R)
    vals = np.clip(vals, 1e-8, None)
    R = vecs @ np.diag(vals) @ vecs.T
    d = np.sqrt(np.diag(R))
    R = R / np.outer(d, d)
    # vols are annual standard deviations, so F is already in annual units
    F = np.outer(vols, vols) * R

    b1 = rng.normal(1.0, 0.25, size=n_assets)
    B = np.empty((n_assets, n_factors))
    B[:, 0] = b1
    if n_factors >= 2:
        B[:, 1] = 0.62 * b1 + rng.normal(0.0, 0.08, size=n_assets)
    for j in range(2, n_factors):
        B[:, j] = rng.normal(0.0, 0.5, size=n_assets)

    mu = rng.uniform(0.03, 0.12, size=n_assets)
    residual_var = rng.uniform(0.02, 0.09, size=n_assets)
    w_raw = rng.uniform(0.5, 1.5, size=n_assets)
    w_b = w_raw / w_raw.sum()

    tickers = np.array([f"S{i + 1:02d}" for i in range(n_assets)])
    factor_names = np.array([f"F{i + 1}" for i in range(n_factors)])
    pi = PortfolioInputs(
        tickers=tickers, factor_names=factor_names,
        mu=mu, B=B, F=F, residual_var=residual_var, w_b=w_b,
        metadata={"mode": "synthetic", "seed": seed, "synthetic": True},
    )
    provenance = {
        "mode": "synthetic", "seed": int(seed), "n_assets": int(n_assets),
        "n_factors": int(n_factors), "collinear_corr": float(collinear_corr),
        "periods_per_year": int(periods_per_year),
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "note": "SYNTHETIC data for offline demonstration; not real assets",
    }
    validate_inputs(pi)
    return pi, provenance
