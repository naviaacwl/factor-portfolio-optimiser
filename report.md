# Factor Portfolio Optimizer with Active-Risk Control

**Finance with AI Agents — Final Project Report**
Author: [your name] · Repository: [URL] · Commit: [URL]
Status of this document: complete, including **actual execution results** —
§5–§7 were filled from WorkSwarm agent runs on 2026-11-16/17 (optimization on
synthetic + real-data packs, verification suite 34/34 checks passed, and
walk-forward backtests on both datasets; the real-data window is
2015→2026-11-16 with a fixed-rule sub-period breakdown). Remaining
author-side items before submission: name/repository/commit identifiers, PDF
conversion, screenshots from the agent session logs, and the
fresh-conversation Skill-reuse evidence (§8).

---

## 1. Scope statement

> For **a quantitative portfolio analyst**, my project takes **asset expected
> returns, factor loadings, factor covariance, and benchmark weights**,
> produces **optimized portfolio weights and an active factor compliance
> report**, and is useful because **manual factor tilting and covariance
> estimation are error-prone, susceptible to multicollinearity, and struggle
> to balance trade-offs between tracking error, factor exposures, and
> single-stock concentration; automating this via convex Quadratic Programming
> ensures mathematical optimality and strict risk control**. I will check it
> using **independent constraint verification (weights summing to 1.0, active
> factor tilts within bounds) and a lightweight historical comparison against
> an Equal-Weight (1/N) and unregularized sample covariance benchmark,
> alongside an infeasible-constraint edge-case test**.

This matches the course "quick scope test" (instructions, Appendix A) and the
Portfolio-and-risk direction: *construct a constrained portfolio and explain
trade-offs; check constraint feasibility, a simple benchmark, risk
definitions, and a later-period test.*

## 2. The finance problem

**Who is the user?** A quantitative portfolio analyst (or a student/fund
junior doing benchmark-relative mandates) who must build an equity portfolio
that tracks a benchmark while expressing controlled active views on factors.

**What is the task?** Given a universe of assets with expected returns `mu`,
a factor exposure matrix `B`, a factor covariance `F` (plus idiosyncratic
variances `D`), and benchmark weights `w_b`, choose portfolio weights `w`
that minimize active risk subject to hard compliance constraints, and prove
compliance with a machine-checked report.

**Why is the current approach hard?** Done manually (spreadsheets / ad-hoc
scripts), the workflow is:

1. **Error-prone.** Hand-tilting weights to hit factor targets while keeping
   Σw = 1, position caps, and tilt bounds is a juggling act; a single stale
   cell breaks compliance silently.
2. **Susceptible to multicollinearity.** Sample covariance estimates from
   correlated factors/assets are near-singular; naive solvers produce extreme,
   unstable weights that flip on tiny input changes.
3. **Trade-offs are opaque.** Tracking error vs factor exposure vs
   single-stock concentration cannot be balanced by intuition — there is no
   guarantee a hand-built portfolio is even feasible, let alone optimal.

**What does the project deliver?**

* **Inputs:** five small CSV files (an "input pack", see
  `examples/basic/data_dictionary.md`): expected returns, factor loadings,
  factor covariance, benchmark weights, residual variances (+ optional
  `config.json`).
* **Output:** `weights.csv` (optimized weights and active weights),
  `compliance_report.md` (per-factor tilt table with bounds, utilization and
  slack; weight-sum/box checks; concentration analytics; covariance
  conditioning audit; PASS/FAIL/INFEASIBLE verdict), `compliance.json`.
* **Smallest useful version:** the bundled 8-asset, 3-factor synthetic pack,
  one command:
  `python scripts/run_optimization.py --inputs examples/basic --output outputs/run1`

## 3. Data

| Pack | Kind | Source / provenance |
|---|---|---|
| `examples/basic/` | **Synthetic** (hand-constructed) | 8 fictitious assets S01–S08; factors MKT, F2 (deliberately near-collinear with MKT: loadings ≈ 0.62×MKT, corr 0.92), F3 (independent). F built as D^½·R·D^½ with a provably PD correlation matrix (leading minors 1, 0.1536, det ≈ 0.1487 > 0). Benchmark sums to exactly 1.00. See its `data_dictionary.md`. |
| Generated synthetic | Synthetic, seeded | `scripts/prepare_inputs.py synthetic` (reproducible via seed; provenance.json records generator, seed, timestamp) |
| Real-data mode | Real (optional) | `scripts/prepare_inputs.py yfinance` — public Yahoo Finance adjusted closes; provenance.json records tickers, window, retrieval timestamp, `auto_adjust` setting, rows used, per the course data rules (Appendix B). |
| `examples/real/` + `data/market/` | **Real (yfinance)** — executed 2026-11-16, extended 2026-11-17 | 12 US large-caps (AAPL, MSFT, GOOGL, AMZN, NVDA, JPM, JNJ, XOM, PG, KO, HD, MRK); factor proxies SPY (market), QQQ (growth), IWM (size); window 2015-01-01→2026-11-16, 2,985 common trading days used; `auto_adjust=True` (field Close); retrieved 2026-11-17T17:00:39+08:00; recorded in `provenance.json`. End-date rule: **latest available data at run time** — the original through-2024 window was extended after review with no parameter re-tuning (disclosed in §7.1). Benchmark: equal weight over the 12 assets. **Selection-bias disclosure:** tickers were chosen with hindsight of 2015–2026 mega-cap winners; this pack demonstrates mechanics, not investable alpha. |

All units are annual decimals; the loader validates finiteness, symmetry and
PSD-ness of F, non-negative residual variances, and benchmark weights summing
to 1.0 — and stops with the offending file/field named rather than silently
patching data.

## 4. Method

### 4.1 The convex QP

With active weights `a = w − w_b`, regularized covariance `Σ`, expected
returns `μ`, and loadings `B`:

```
minimize    ½ aᵀΣa  −  γ μᵀa  +  κ wᵀw
subject to  1ᵀw = 1                          (fully invested)
            w_min ≤ w ≤ w_max                (long-only + single-stock cap)
            lo ≤ Bᵀ(w − w_b) ≤ hi            (active factor tilt bounds)
            ‖w − w_prev‖₁ ≤ T                (optional turnover cap)
```

* `½ aᵀΣa` — tracking-error variance (the tracking-error trade-off);
* `−γ μᵀa` — expected active return (γ = `risk_aversion`);
* `κ wᵀw` — single-stock concentration penalty (wᵀw is the portfolio HHI;
  penalizing it raises "effective N" = 1/HHI);
* tilt bounds are **hard constraints** — strict risk control, reported per
  factor in the compliance report.

**Convexity (why "mathematically optimal" is a fair claim):** Σ is PSD by
construction (PSD factor covariance F ⇒ B F Bᵀ PSD; plus nonnegative diagonal
D; shrinkage toward a PSD target and a nonnegative ridge preserve PSD), and
γ, κ ≥ 0, so the objective is a convex quadratic; all constraints are affine
(the turnover cap is a convex 1-norm). Any local optimum is therefore a global
optimum; with κ > 0 the objective is strictly convex and the optimum is
unique. Solved with CVXPY (default QP solver; CLARABEL/OSQP selectable).

### 4.2 Multicollinearity defense

1. **Factor-model structure:** `Σ = B F Bᵀ + D` — estimates K(K+1)/2 + n
   parameters instead of n(n+1)/2.
2. **Linear shrinkage (Ledoit-Wolf style):** `Σ_δ = (1−δ)Σ + δ·T`, with
   T = diag(Σ) or scaled identity; δ = `shrinkage_intensity` (0 recovers the
   unregularized baseline used in the comparison).
3. **Relative ridge:** `Σ_δ + ε·mean(diag)·I` — numerical PSD safety.
4. **Audit, don't hide:** the compliance report prints condition numbers
   before/after regularization, so the effect is measurable.

### 4.3 The active factor compliance report

For every factor k: benchmark exposure `(Bᵀw_b)_k`, portfolio exposure
`(Bᵀw)_k`, active tilt `(Bᵀ(w−w_b))_k`, bounds, within-bounds flag,
violation magnitude, bound utilization and slack. Plus portfolio-level
checks: sum of weights vs 1.0, box compliance, long-only flag, HHI /
effective N, names at cap, expected return / active return / ex-ante
tracking error, max post-solve constraint violation, solver status and
timing. Overall verdict: **PASS / FAIL / INFEASIBLE / SOLVER_ERROR**. All
quantities are recomputed from the weights and raw inputs with plain numpy
(not copied from solver internals).

## 5. Actual WorkSwarm use

Design and implementation were produced in WorkSwarm (JiuwenSwarm agent)
sessions: the agent read the four course PDFs (final project instructions,
T1–T3 guides) with its PDF tool, extracted the deliverable requirements,
designed the QP formulation and verification plan against those requirements,
and generated the complete package (core modules, entry scripts, example
pack, Skill, README, this report skeleton).

Execution evidence (all runs performed 2026-11-16/17 in WorkSwarm agent
sessions on macOS, Python 3.9.6 venv — numpy 2.0.2, pandas 2.3.3, scipy
1.13.1, cvxpy 1.7.5/OSQP; screenshots to be captured by the author from these
session logs):

1. **Optimization runs.**
   `python scripts/run_optimization.py --inputs examples/basic --output outputs/run1 --print-report`
   → solver status `optimal` (CVXPY/OSQP), verdict **PASS**, Σw = 1.0000000000,
   F3 tilt binding at +0.0800; artifacts `outputs/run1/{weights.csv,
   compliance_report.md, compliance.json}`.
   `python scripts/run_optimization.py --inputs examples/real --output outputs/run_real`
   → verdict **PASS**, 5 names at the 15% cap, QQQ tilt binding at +0.100
   (extended 2015→2026-11 window, re-run 2026-11-17); artifacts under
   `outputs/run_real/`.
2. **Verification run.** `python scripts/run_verification.py --inputs examples/basic`
   → **34/34 checks passed**; artifacts under `outputs/verification/`
   (`evaluation_record.md/.csv`, `baseline_comparison.csv`).
3. **Correction/debug episodes with the agent** (actual failures found by
   executing, fixed, then re-run to green — real tool output):
   * First run of `run_optimization.py` aborted with
     `InputValidationError: factor_covariance.csv: factor names ['F2','F3','MKT','factor'] do not match ...`
     — root cause: passing a *Series* to pandas `set_index` keeps the source
     column; fixed in `data.py` (set index by column name), re-run → PASS.
   * `run_verification.py` raised `ValueError: The truth value of an array with
     more than one element is ambiguous` — traceback localized it to
     `verify.py:145` (`t - hi` instead of `t - hi[i]`); fixed, re-run.
   * numpy `Generator.multivariate_normal` rejects `allow_singular` (legacy-API
     parameter); removed — the default SVD method already tolerates
     semidefinite covariances; re-run.
   * Backtest markdown rendering hit `ImportError: Missing optional dependency
     'tabulate'` (pandas `to_markdown`); replaced with a hand-rolled renderer
     to keep the hard dependency list at four packages.
4. **Fresh-conversation reuse run of the Skill on a NEW case:** still to be
   performed by the author in a fresh session (see §8) — labeled honestly as
   pending.

## 6. Result

Actual results from the 2026-11-16 agent runs (full tables in
`outputs/run1/compliance_report.md` and `outputs/run_real/compliance_report.md`).

### 6.1 Synthetic example pack (`examples/basic`, 8 assets, factors MKT/F2/F3)

* Solver: CVXPY/**OSQP**, status `optimal`, 0.008 s → **verdict PASS**.
* Weights: sum = 1.0000000000 (dev 2.2e-16); range [−0.000000, 0.273258] vs cap
  0.30; 0 names at cap, 2 effectively zero; HHI 0.2080; **effective N 4.81/8**;
  top holdings S06 27.3%, S03 24.1%, S01 20.6%.
* Active tilts (recomputed independently from the raw CSVs):

| Factor | Bench exp. | Port exp. | Tilt | Bounds | Within? | Utilization |
|---|---:|---:|---:|---|---|---:|
| MKT | +1.0165 | +1.0058 | −0.0107 | ±0.100 | YES | 10.7% |
| F2 | +0.6206 | +0.6072 | −0.0134 | ±0.050 | YES | 26.9% |
| F3 | +0.0750 | +0.1550 | **+0.0800** | ±0.080 | YES | **100.0% (binding)** |

* Risk analytics: expected return 10.46% vs benchmark 9.27% → expected active
  **+1.19%**; ex-ante **tracking error 6.09%**; objective 3.44e-4; max
  post-solve constraint violation 1.1e-16.
* Conditioning audit: condition number 9.60 → 8.33 (the small pack is
  well-conditioned by construction; the collinearity stress test is §7 check #4).

### 6.2 Real-data pack (`examples/real`, 12 US large-caps, factors SPY/QQQ/IWM, 2015→2026-11)

* Solver: CVXPY/**OSQP**, status `optimal` → **verdict PASS**; max post-solve
  constraint violation 1.1e-16.
* Weights: sum = 1.0000000000; range [−0.000000, 0.150000]; **5 names at the
  15% cap** (AAPL, MSFT, NVDA, JPM, XOM); 2 driven to ≈0; HHI 0.1299;
  **effective N 7.70/12**.
* Active tilts (recomputed independently from the raw CSVs):

| Factor | Bench exp. | Port exp. | Tilt | Bounds | Within? | Utilization |
|---|---:|---:|---:|---|---|---:|
| SPY | +1.0310 | +0.9795 | −0.0515 | ±0.100 | YES | 51.5% |
| QQQ | +0.0424 | +0.1424 | **+0.1000** | ±0.100 | YES | **100.0% (binding)** |
| IWM | −0.1305 | −0.0649 | +0.0656 | ±0.100 | YES | 65.6% |

  (In the original through-2024 run IWM also bound at +0.100; with the window
  extended, μ̂ moved and IWM came off its bound while QQQ stayed pinned —
  expected behavior when inputs change, disclosed rather than hidden.)
* Risk analytics: with μ̂ = full-window historical means (noisy, in-sample —
  reported as inputs, not forecasts): expected return 26.09% vs benchmark
  22.00% → expected active +4.09%; ex-ante **tracking error 5.07%**;
  objective −0.0331.
* Conditioning audit: condition number 22.24 → 19.49.

## 7. Evaluation and verification (design fixed before execution)

Per instructions Appendix C ("Calculator/optimizer: hand-checkable/reference
cases and a simple heuristic; check constraints and invalid input";
"Portfolio/strategy: equal-weight reference on the same data; fix rules
before a later-period test"). Implemented in `scripts/fp_optimizer/verify.py`,
one command: `python scripts/run_verification.py`.

| # | Check | Method (why it is *independent*) | Expected |
|---|---|---|---|
| 1 | Constraint verification | Re-read SAVED `weights.csv` + RAW input CSVs; recompute Σw, box, every tilt with plain numpy — no optimizer internals reused | sum = 1.0 (1e-6); all tilts within bounds (1e-5) |
| 2 | Hand-checkable small instance | 3-asset/1-factor problem with round numbers (Σ hand-computable, δ=ε=0); solved by exhaustive simplex grid search | QP objective ≤ grid objective, gap < 1e-3; weights agree within 3×step |
| 3 | Second implementation | Identical QP re-solved with scipy **SLSQP** (different algorithm family and code path); strict convexity ⇒ unique optimum both must reach | objectives agree ≤ 1e-7(+rel), weights ≤ 2e-3 |
| 4 | Regularization properties | Near-collinear synthetic instance (corr 0.999): δ=0 is identity; shrunk Σ symmetric, PSD; ridge keeps min-eig ≥ 0 | condition number improves; PSD preserved |
| 5 | Lightweight historical comparison | Design/test split, rules fixed before test: all inputs estimated on design window only; three portfolios allocated once, buy-and-hold over test: **optimized QP (shrunk)** vs **Equal-Weight 1/N** vs **unregularized sample covariance** (same objective/constraints, raw Σ). Metrics: realized cum return, ann. vol, max drawdown (running-peak definition), tracking error, geometric active return vs benchmark | All variants solve; metrics finite; outcomes reported descriptively — underperformance is a valid, analyze-don't-hide result |
| 6 | Infeasible-constraint edge case | Three guaranteed-infeasible sets: (A) tilt lo > hi; (B) tilt lower bound provably above max attainable (2·max&#124;B&#124;+1); (C) box with n·min_weight > 1 | status `infeasible`, verdict `INFEASIBLE`, no crash, no silent relaxation |
| 7 | Invalid inputs | Benchmark sum 0.9; non-PSD F; asymmetric F; NaN mu; shape mismatch; negative residual variance; missing directory | `InputValidationError` with file/field named — never silent pass |

Baselines default to a seeded synthetic simulation from the example factor
model (offline, reproducible, clearly labeled); real price CSVs plug in via
`--baseline-mode market --prices ... --factors ... --design-end ...`.

**Compact evaluation record** (Appendix C format) — auto-written to
`outputs/verification/evaluation_record.md` as *case / method / expected /
actual / pass-fail / note*.

**Result (2026-11-16 agent run, `examples/basic`): `SUMMARY: 34/34 checks
passed`.** Highlights from `evaluation_record.md`:

* Saved-file rechecks: Σw = 1.0000000000 (dev 2.2e-16); all three tilts within
  bounds with zero violation; box respected (max lower violation 5.8e-23).
* Brute force (exhaustive simplex grid, step 0.01, 5,151 points): QP objective
  0.01348593 ≤ grid best 0.01349375, gap 7.8e-6 (bounded by grid resolution);
  weights agree within 0.0096.
* Second solver (scipy SLSQP — different algorithm family): objective agrees to
  **3.1e-15**; max weight difference **1.6e-7** — consistent with the unique
  optimum that strict convexity guarantees.
* Regularization on a corr-0.999 near-collinear instance: δ=0 is exactly the
  identity; condition number **21.84 → 12.04** after shrinkage; PSD preserved
  (min eig 4.06e-2); ridge keeps min eig ≥ 0.
* Infeasible edge cases (3/3): status `infeasible`, verdict `INFEASIBLE`,
  graceful report, no crash, no silent relaxation.
* Invalid inputs (7/7): `InputValidationError` naming the offending file/field.

Lightweight historical comparison (`baseline_comparison.csv`; seeded synthetic
simulation from the example factor model, 900-day design / 600-day test, rules
fixed before the test window, one allocation then buy-and-hold):

| Variant | Design status | σ cond # | Test cum ret | Test ann vol | Test max DD | Test TE | Active (geo) |
|---|---|---:|---:|---:|---:|---:|---:|
| benchmark = equal-weight 1/N | reference | — | −12.33% | 20.69% | −43.00% | 0 | 0 |
| **optimized QP (shrunk)** | optimal | 8.20 | −13.59% | 23.77% | −43.36% | 6.92% | **−1.45%** |
| unregularized sample Σ | optimal | 9.46 | −13.67% | 23.78% | −43.25% | 6.96% | −1.54% |

**Honest interpretation.** The optimized QP *underperformed* the 1/N reference
by 1.45% (geometric) over the test window while carrying 6.9% tracking error:
with μ̂ estimated as historical means, the γμᵀa term tilts toward whatever did
well in the design window, and that tilt did not persist out-of-sample —
textbook estimation-error chasing, reported rather than hidden (instructions
Appendix C explicitly allows underperformance as a valid result). The
unregularized variant behaved almost identically (−1.54%): on this benign
8-asset instance the sample covariance is well conditioned (9.5 vs 8.2), so
shrinkage changes little — its measurable benefit appears in the stressed
near-collinear case above (21.8 → 12.0), not here.

### 7.1 Walk-forward backtest (`backtest.py` / `run_backtest.py`)

Rules (fixed before running): 504-day rolling estimation window; rebalance
every 63 trading days (≈quarterly); μ̂ (historical mean), B̂, F̂, D̂ re-estimated
by OLS on the trailing window only; rebalance on day t uses data up to t−1
(no lookahead); between rebalances weights drift buy-and-hold; 10 bp one-way
cost on traded notional at each rebalance; all variants start at the benchmark
weights; a failed solve holds current weights and is counted (0 occurred);
Sharpe uses rf = 0. Sub-period tables use a **fixed calendar-chunk rule**
implemented in code (`--subperiod-years`, default 3): consecutive
calendar-year blocks from the first-rebalance year, last chunk possibly
partial — boundaries are never chosen after seeing results.

**Synthetic data** (2,000 days simulated from the example pack, seed 99, 24
rebalances; artifacts in `outputs/backtest/`):

| Variant | Total ret | CAGR | Ann vol | Sharpe | Max DD | TE vs bench | Avg turnover | Cost drag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| benchmark buy-and-hold | +60.3% | 6.13% | 21.44% | 0.385 | −43.5% | — | — | 0 |
| **optimized QP (shrunk)** | +30.2% | 3.38% | 21.44% | 0.262 | −47.3% | 5.85% | 40.5% | 0.97% |
| unregularized sample Σ | +30.1% | 3.37% | 21.44% | 0.262 | −47.4% | 5.84% | 40.5% | 0.97% |
| equal-weight 1/N (rebalanced) | +38.4% | 4.18% | 20.53% | 0.302 | −45.0% | 5.62% | 10.6% | 0.25% |

**Real data** (yfinance 12 large-caps, factors SPY/QQQ/IWM, daily adjusted
closes 2015-01-01→2026-11-16, 2,985 rows; backtest spans 2017-01→2026-11
after the 504-day warm-up; **40 rebalances; 0 solver failures**; artifacts in
`outputs/backtest_market/`):

| Variant | Total ret | CAGR | Ann vol | Sharpe | Max DD | TE vs bench | Avg turnover | Cost drag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| benchmark buy-and-hold (drifts) | +3,346% | 34.83% | 31.65% | 1.103 | −48.97% | — | — | 0 |
| **optimized QP (shrunk)** | +1,044% | 22.85% | 18.95% | 1.181 | **−28.43%** | 19.09% | 38.6% | 1.54% |
| unregularized sample Σ | +1,045% | 22.85% | 18.94% | 1.182 | −28.42% | 19.10% | 38.7% | 1.55% |
| equal-weight 1/N (rebalanced) | +1,202% | 24.20% | 17.50% | **1.327** | −29.70% | 21.31% | 9.7% | 0.39% |

Sub-period breakdown (fixed 3-calendar-year chunk rule; also saved as
`subperiod_breakdown.csv`):

| Sub-period | Bench ret | Bench maxDD | QP ret | QP maxDD | 1/N ret | 1/N maxDD | QP vs 1/N |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2017–2019 | +98.5% | −34.4% | +84.9% | −24.6% | +87.3% | −19.3% | −1.3% |
| 2020–2022 | +65.8% | −49.0% | +52.1% | −28.4% | +56.8% | −29.7% | −3.0% |
| 2023–2025 | +593.4% | −33.5% | +102.3% | −19.0% | +117.9% | −15.4% | −7.2% |
| 2026 (partial) | **−12.2%** | −29.7% | **+18.7%** | **−5.8%** | +19.9% | −6.2% | −1.1% |

**Honest interpretation (real data).**

1. *Selection bias first:* the 12 tickers were chosen with hindsight of the
   2015–2026 mega-cap winners (NVDA, AAPL, MSFT…), so the benchmark's 34.8%
   CAGR is an artifact of the universe, not a strategy result. This backtest
   demonstrates mechanics and risk control, not investable alpha.
2. *Drawdown control replicates in every sub-period, including the unseen
   2025–26 data:* drifting-benchmark max drawdowns of −34%, −49%, −33%, −30%
   vs QP −25%, −28%, −19%, −6%. The strongest single piece of evidence is
   2026: when old-winner momentum collapsed, the drifting benchmark lost
   12.2% (max DD −29.7%) while the constrained QP returned **+18.7%** with a
   −5.8% drawdown — the compliance machinery earning its keep out-of-sample.
3. *The drag vs rebalanced 1/N is real and regime-dependent:* it widened from
   −1.3% (2017–19) to −7.2% (2023–25), then converged to −1.1% (2026). The
   mechanism is visible in the saved weights history: with μ̂ = trailing
   historical means, the QP held NVDA pinned at the 15% cap at **every**
   rebalance from 2024-01 through 2026-01, while the 2025–26 rotation paid
   JNJ (+105%), KO and XOM instead. Full-period Sharpe: QP 1.181 vs 1/N
   1.327, at ~4× the turnover costs. Improving μ̂ (views, mean-shrinkage, or
   γ=0 pure risk-based allocation) is the natural next step — a finding, not
   a failure, and exactly what Appendix C asks to surface.
4. *Shrinkage vs unregularized is a wash* (identical to three significant
   figures): 12 assets × 504-day windows keep the rolling sample covariance
   well conditioned (cond 22.2 → 19.5). The regularization defense matters
   for stressed/collinear inputs (§7 check #4: 21.8 → 12.0), which these
   universes do not contain.
5. *Sequencing disclosure:* the real-data window originally ended 2024-12-31
   (a round decade inherited from a docstring example, not an analytical
   choice). It was extended to the latest available date (2026-11-16) **with
   no parameter re-tuning** — γ, κ, the 15% cap and ±0.10 tilt bands are
   unchanged judgments made before either run. The extension added data that
   works *against* the QP's relative-return story (item 3), so it was not a
   favorable-window search; the through-2024 results are superseded, not
   cherry-picked against.

## 8. Reusable contribution

* **Form:** regular Skill — `skills/factor-portfolio-optimizer/SKILL.md`
  (YAML name/description; scope, inputs/outputs, procedure, built-in checks,
  failure handling, dependencies, limitations, attribution) wrapping the
  generalized scripts. Configurable: input pack, universe, factors, bounds,
  output folder — no cached answers, no secrets, no personal absolute paths.
* **Generalization already applied:** ticker/factor-agnostic (names come from
  the CSV headers), period-agnostic (annualization configurable), output
  folders are CLI arguments, real-data mode records its own provenance.
* **Publication plan:** public GitHub repository with this README link to
  https://github.com/openJiuwen-ai/jiuwenswarm; submit repository URL + exact
  commit to the course. Hub submission only if selected.
* **Fresh-run evidence [FILL AFTER RUN]:** in a fresh conversation/clean
  setup, load the Skill, build a NEW case (different tickers via
  `prepare_inputs.py yfinance` or a different synthetic seed) into a NEW
  output folder, execute the essential workflow, and verify — label peer
  reuse vs self-reproduction honestly. A load-and-plan check alone does not
  count as execution.

## 9. Limitations

1. **Estimation error.** `mu`, `B`, `F`, `D` are estimates; the historical-mean
   default for `mu` is deliberately noisy. Shrinkage mitigates — it does not
   eliminate — covariance estimation error.
2. **In-sample bias.** The design-window comparison favors all optimized
   variants in-sample; the honest signal is the test-window (later-period)
   result, which can legitimately show underperformance vs 1/N.
3. **Single-period core; fixed-bps cost model.** The optimizer is
   single-period; `backtest.py` adds rolling re-estimation, periodic
   rebalancing and bps costs, but costs are a flat 10 bp one-way model (no
   slippage/market-impact curves), benchmark weights are static, and μ̂ stays
   the historical mean unless overridden.
4. **Factor proxies.** In real-data mode, observable ETF proxies stand in for
   latent factors; the OLS "loadings" are regression betas, not fundamental
   exposures.
5. **Verification independence is method-level** (file re-read, exhaustive
   enumeration, second solver family) — stronger than rerunning the same code,
   weaker than an independent human reviewer. Same-agent self-checks are
   labeled as such.
6. **Not investment advice.** Synthetic examples are fictions; real-data runs
   use public historical data with recorded provenance.
7. **Universe selection bias in the real-data demo.** The 12 large-caps were
   chosen with hindsight of 2015–2026 winners; benchmark and variant returns
   inherit that bias. The run demonstrates mechanics and risk control, not
   investable alpha.

## 10. Attribution and AI assistance

* Course materials: HKUST IEDA *Finance with AI Agents* — final project
  instructions and Tutorial 1–3 student guides (structure, evaluation and
  packaging conventions follow them).
* Substantial AI assistance: the JiuwenSwarm/WorkSwarm agent generated the
  code and drafted this report from the project specification; the author
  reviewed the formulation, design choices and claims, and remains
  responsible for them.
* Software: numpy/pandas/scipy (BSD), cvxpy (Apache-2.0), yfinance
  (Apache-2.0, optional). Project code MIT-licensed (`LICENSE`).
* Data: synthetic (hand-constructed) or public Yahoo Finance data in
  real-data mode, subject to Yahoo's terms; no redistribution of paid data.

## Appendix A — Six-part working prompt for the execution run

(Per instructions Appendix B; send as one message in a fresh WorkSwarm
conversation in the project folder.)

```
GOAL
Run and verify the factor-portfolio optimizer for a quantitative portfolio
analyst: produce optimized weights and an active-factor compliance report
for the bundled example, then execute the full independent verification
suite. One agent; no team needed.

INPUTS
Work only in this repository. Read README.md, examples/basic/ (five CSVs +
config.json + data_dictionary.md). Data are SYNTHETIC, annual units.
If any file is missing or validation fails, stop and explain; do not
invent or patch data.

METHOD
Use the installed Python 3 runtime:
  1. pip install -r requirements.txt   (if needed)
  2. python scripts/run_optimization.py --inputs examples/basic --output outputs/run1 --print-report
  3. python scripts/run_verification.py --inputs examples/basic
  4. python scripts/run_backtest.py --inputs examples/basic --output outputs/backtest --plot
Inspect actual tool output and open the saved files; a chat message saying
"done" is not evidence. Do not modify the optimizer to make checks pass —
report failures instead.

OUTPUT
Save artifacts under outputs/run1/ and outputs/verification/ (weights.csv,
compliance_report.md, compliance.json, evaluation_record.md,
baseline_comparison.csv). Then fill report.md sections 5-7 with the ACTUAL
results: verdict, tilt table, evaluation-record summary line, baseline
comparison table, and one honest interpretation paragraph.

BOUNDARIES
Leave inputs, scripts and examples unchanged except filling the marked
[FILL AFTER RUN] sections of report.md. No installs beyond
requirements.txt, no credentials, no Git pushes, no publishing. Ask before
replacing any existing output file.

CHECKS
Expected: solver status optimal; verdict PASS; weights sum to 1.0 within
1e-6; every active tilt inside its config bound; all verification checks
pass (including infeasible edge case reported gracefully and invalid inputs
rejected). Show the actual run output and the confirmed artifact paths
together in a short completion summary suitable for a screenshot. Report
any unresolved issue honestly; never invent a passed check.
```

## Appendix B — Submission checklist mapping (instructions §4)

| Required item | Where it lives |
|---|---|
| Report PDF (≥ 3 pages) | this `report.md` → convert to PDF (§5–§7 already filled with the 2026-11-16 run results; §8 fresh-run evidence remains author-side) |
| Source & Repository | repo URL + exact commit URL + source ZIP of `final-project/` |
| Demonstration (5 min) | run §Appendix A prompt live/recorded: task → core output → one check → Skill |
| Execution evidence | 2–4 screenshots: optimization run, verification record, agent debug moment, fresh-case Skill reuse |
| Community record | README "Community contribution record" section + public GitHub + openJiuwen link + fresh-run evidence |
