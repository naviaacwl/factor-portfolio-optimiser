# Data dictionary — examples/basic

**Data kind: SYNTHETIC.** These eight assets (`S01`–`S08`), three factors and
all numbers in this folder are hand-constructed fiction for offline
demonstration and testing. They are not real securities, not derived from
market data, and nothing here is investment advice.

## Units convention (project-wide)

| Field | Units |
|---|---|
| `mu` (expected returns) | ANNUAL, decimals (0.10 = 10% per year) |
| `factor_covariance` (F) | ANNUAL variances/covariances, decimals |
| `residual_variance` | ANNUAL idiosyncratic variance, decimals |
| `weight`, `active_weight` | fractions of portfolio value (sum to 1.0) |
| tilt bounds (config.json) | same units as `B' (w − w_b)` |

## Files

| File | Contents |
|---|---|
| `expected_returns.csv` | `ticker,mu` — annual expected returns |
| `factor_loadings.csv` | `ticker,MKT,F2,F3` — exposure matrix B (8×3) |
| `factor_covariance.csv` | 3×3 symmetric PSD matrix F (rows/cols = factors) |
| `benchmark_weights.csv` | benchmark index weights, sum = 1.00 exactly |
| `residual_variance.csv` | per-asset idiosyncratic variances (diagonal D) |
| `config.json` | optimizer configuration (objective, regularization, bounds) |

## Factor design (deliberate multicollinearity)

* **MKT** — market-like factor, annual vol 16% (variance 0.0256).
* **F2** — a synthetic factor deliberately **near-collinear with MKT**:
  loadings ≈ 0.62 × MKT loadings, and corr(MKT, F2) = 0.92. This is the
  pathological structure that makes raw covariance estimation unstable and
  gives the shrinkage regularization something real to fix.
* **F3** — an independent style-like factor, annual vol 4% (variance 0.0016),
  corr(MKT, F3) = 0.10, corr(F2, F3) = 0.15.

F was constructed as `D_vol^{1/2} · R · D_vol^{1/2}` with vols
(0.16, 0.05, 0.04) and correlation matrix

```
R = [[1.00, 0.92, 0.10],
     [0.92, 1.00, 0.15],
     [0.10, 0.15, 1.00]]
```

which is positive definite (leading minors 1 > 0, 0.1536 > 0, det ≈ 0.1487 > 0),
so F is PSD by construction.

## Benchmark

A fictitious cap-weight-like index: 0.25 + 0.20 + 0.15 + 0.12 + 0.10 + 0.08 +
0.06 + 0.04 = **1.00**. The largest weight (0.25) is below the single-stock
cap (0.30), so the benchmark itself is feasible for the optimizer.

## How to regenerate / replace

* Larger synthetic packs: `python scripts/prepare_inputs.py synthetic --outdir examples/synthetic30 --n-assets 30 --seed 11`
* Real data: `python scripts/prepare_inputs.py yfinance --outdir examples/real --tickers ... --factors ... --start ... --end ...`
  (records retrieval time, window and adjustment setting in `provenance.json`).
