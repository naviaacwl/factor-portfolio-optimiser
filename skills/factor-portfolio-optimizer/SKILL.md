---
name: factor-portfolio-optimizer
description: Benchmark-relative factor portfolio construction via convex Quadratic Programming. Takes asset expected returns, factor loadings, factor covariance, and benchmark weights; produces optimized portfolio weights plus an active-factor compliance report (weights sum to 1.0, single-stock caps, active factor tilts within bounds), with shrinkage regularization against multicollinearity and an independent verification suite (file-level constraint rechecks, brute-force and SLSQP cross-checks, equal-weight and unregularized-covariance baselines, infeasible-constraint edge case). Use when the user asks to build or tilt a factor-constrained portfolio, minimize tracking error against a benchmark, audit active factor exposures, or compare regularized versus raw covariance estimates.
---

# Factor Portfolio Optimizer (Skill)

Reusable procedure for constructing a benchmark-relative equity portfolio
under active factor-tilt and concentration constraints, solving a convex QP,
and emitting a machine-checked compliance report. Built for the HKUST
*Finance with AI Agents* final project; part of the
[openJiuwen / JiuwenSwarm](https://github.com/openJiuwen-ai/jiuwenswarm)
ecosystem.

## Scope

* Single-period, benchmark-relative optimization (long-only by default;
  configurable box bounds and optional turnover cap).
* Objective trades off tracking error (active risk), expected active return,
  and single-stock concentration (HHI penalty).
* Hard constraints: fully invested (Σw = 1), position box, active factor
  tilts `lo ≤ B'(w − w_b) ≤ hi`.
* Multicollinearity defense: factor-model covariance `B F B' + D`, linear
  shrinkage (Ledoit-Wolf style), relative ridge; conditioning audited.
* Walk-forward backtest layer (`backtest.py` / `run_backtest.py`): rolling
  re-estimation, periodic rebalancing with bps transaction costs, buy-and-hold
  drift between rebalances, no lookahead; variants vs 1/N, unregularized
  covariance, and a drifting buy-and-hold benchmark.
* NOT in scope: transaction-cost *optimization* (costs are a fixed-bps model),
  live trading, investment advice.

## Inputs (the user must supply)

1. A Python 3.9+ runtime with the dependencies installed:
   `pip install -r requirements.txt` (numpy, pandas, scipy, cvxpy;
   yfinance optional for real-data mode).
2. An input-pack directory with the documented CSV schema
   (`expected_returns.csv`, `factor_loadings.csv`, `factor_covariance.csv`,
   `benchmark_weights.csv`, `residual_variance.csv`, optional `config.json`)
   — see `examples/basic/` for a lawful synthetic example and its data
   dictionary. Units: annual decimals; benchmark weights must sum to 1.0.
3. OR raw data for estimation: daily adjusted-close price CSVs (assets and
   factor-proxy ETFs), or network access for `prepare_inputs.py yfinance`
   mode (provenance is recorded automatically).

## Outputs

Under the chosen output folder:

* `weights.csv` — optimized weights and active weights, full precision.
* `compliance_report.md` — verdict PASS/FAIL/INFEASIBLE, per-factor tilt
  table (benchmark vs portfolio exposure, bound utilization, slack),
  weight-sum/box checks, concentration (HHI, effective N), tracking error,
  covariance-conditioning audit.
* `compliance.json` — machine-readable record.
* From verification runs: `evaluation_record.csv/.md` (compact evaluation
  record) and `baseline_comparison.csv`.
* From backtest runs: `backtest_summary.csv`, `equity_curves.csv`,
  `backtest_report.md`, `subperiod_breakdown.csv` (fixed calendar-chunk rule,
  `--subperiod-years`), per-variant turnover/weights-history CSVs, optional
  `backtest.png` chart.

## Procedure

Run from the repository root (all script paths are relative to it; the skill
folder lives at `skills/factor-portfolio-optimizer/`).

1. **Establish inputs.** Confirm the input-pack directory (or fetch/build
   one). State explicitly whether data are real or synthetic; record source,
   window and retrieval time (provenance.json for generated packs). Validate
   before analyzing — the loader raises `InputValidationError` naming the
   offending file/field; do not silently patch data.
   * Optional pack builders:
     `python scripts/prepare_inputs.py synthetic --outdir examples/synthetic30 --n-assets 30 --seed 11`
     `python scripts/prepare_inputs.py yfinance --outdir examples/real --tickers SPY,QQQ,IWM,EFA,TLT,GLD --factors SPY,USMV,VLUE --start 2018-01-01 --end 2024-12-31`
2. **Optimize.**
   `python scripts/run_optimization.py --inputs <pack-dir> --output <out-dir> --print-report`
   Inspect the exit code (0 = PASS, 1 = FAIL, 2 = INFEASIBLE/SOLVER_ERROR,
   3 = input error), the verdict, and the tilt table. Open the saved files —
   a chat message saying "done" is not execution evidence.
3. **Verify independently.**
   `python scripts/run_verification.py --inputs <pack-dir>`
   (add `--baseline-mode market --prices <csv> --factors <csv> --design-end <date>`
   for a real-data historical comparison; default is a seeded synthetic
   simulation, clearly labeled). Review `evaluation_record.md`: every check
   must pass or the discrepancy must be explained.
4. **Backtest (optional but recommended).**
   `python scripts/run_backtest.py --inputs <pack-dir> --output <bt-dir> --plot`
   (synthetic mode), or for real data:
   `python scripts/fetch_market_data.py --tickers ... --factors ... --start ... --end ...`
   then `python scripts/run_backtest.py --mode market --prices <csv>
   --factors <csv> --benchmark-csv <csv> --config <pack-dir>/config.json`.
   Fixed rules: 504-day rolling estimation window, 63-day rebalancing, 10 bp
   one-way costs, day-t decisions use data up to t−1 only; a failed solve
   holds current weights and is counted — never silently patched.
5. **Report.** Summarize to the user: verdict, binding constraints, active
   tilts vs bounds, tracking error, effective N, conditioning improvement,
   backtest outcomes (including underperformance, honestly), and the file
   paths of all artifacts (evidence, not claims).

## Checks (built into the procedure)

* Weights sum to 1.0 within 1e-6 — recomputed FROM THE SAVED FILES with
  plain numpy, independent of the optimizer.
* Every active tilt within its configured bound — recomputed from raw CSVs.
* Box/long-only constraints respected; max constraint violation reported.
* Optimality cross-checks: hand-computable 3-asset instance vs exhaustive
  grid search; identical QP re-solved with scipy SLSQP (different algorithm
  family) — weights/objective must agree within documented tolerances.
* Regularization: shrinkage preserves symmetry/PSD and lowers the condition
  number on a near-collinear instance; δ=0 is exactly the raw matrix.
* Infeasible constraint sets: three guaranteed-infeasible configurations
  must return status `infeasible` and verdict `INFEASIBLE` — never crash,
  never silently relax.
* Invalid inputs (bad sums, non-PSD F, NaNs, shape mismatches): must raise
  `InputValidationError` with an actionable message.

## Failure handling

* **`SolverNotAvailableError` / cvxpy missing** → `pip install -r requirements.txt`.
* **Solver error or OPTIMAL_INACCURATE** → set `"solver": "CLARABEL"` (or
  `"OSQP"` with tighter `solver_kwargs` eps) in config.json and rerun; check
  input scaling (mixing daily and annual units is the usual cause).
* **INFEASIBLE** → report which constraint families conflict (tilt bounds vs
  position caps vs sum-to-one). Suggest specific relaxations to the user;
  do NOT relax silently.
* **InputValidationError** → the message names the file/field; fix the source
  data. For non-PSD user covariances, `covariance.nearest_psd` exists as an
  explicit, documented repair — record it if used.
* **yfinance download fails** → fall back to a course CSV or the synthetic
  pack; record the fallback honestly.

## Dependencies

numpy >= 1.24, pandas >= 2.0, scipy >= 1.10, cvxpy >= 1.4; optional
yfinance >= 0.2.40 (real-data mode) and matplotlib >= 3.7 (backtest charts).
CPU-only, laptop-sized runs. Verified end-to-end on Python 3.9.6 with
numpy 2.0.2 / pandas 2.3.3 / scipy 1.13.1 / cvxpy 1.7.5 (OSQP).

## Limitations

Estimates (mu, F, D) carry estimation error; historical-mean mu is a noisy
default. The design-window baseline comparison has in-sample bias affecting
all optimized variants equally. Verification independence is method-level
(file re-read, enumeration, second solver), not a second human reviewer.
Synthetic examples demonstrate the pipeline, not market claims.

## Attribution & license

HKUST IEDA *Finance with AI Agents* final project; AI-assisted via WorkSwarm,
author-reviewed. MIT license (see repository `LICENSE`). openJiuwen /
JiuwenSwarm: https://github.com/openJiuwen-ai/jiuwenswarm
