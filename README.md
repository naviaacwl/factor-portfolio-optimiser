
# Factor Portfolio Optimiser - IEDA 4000H Final Project

Benchmark-relative factor portfolio construction via **convex Quadratic
Programming**: takes asset expected returns, factor loadings, factor
covariance, and benchmark weights; produces **optimised portfolio weights**

---

Built with WorkSwarm (JiuwenSwarm agent) for the HKUST IEDA 4000H course. Part of the [openJiuwen / JiuwenSwarm](https://github.com/openJiuwen-ai/jiuwenswarm) ecosystem.

---

**Input files/fields, units, sources, and data-access instructions:**
An input-pack directory (see `examples/basic/` and its `data_dictionary.md`):

| File | Fields | Units |
|---|---|---|
| `expected_returns.csv` | `ticker,mu` | annual decimals (0.10 = 10%) |
| `factor_loadings.csv` | `ticker,<factor cols>` | exposures (B) |
| `factor_covariance.csv` | `factor,<factor cols>` | annual, symmetric PSD (F) |
| `benchmark_weights.csv` | `ticker,weight` | fractions, sum = 1.0 |
| `residual_variance.csv` | `ticker,residual_variance` | annual (D) |
| `config.json` | optimizer configuration | optional |

The included example pack is **synthetic** (hand-constructed, with a
deliberately near-collinear factor pair). Real-data mode:
`python scripts/prepare_inputs.py yfinance ...` downloads public Yahoo
Finance data and records provenance (tickers, window, retrieval time,
adjustment setting) in `provenance.json`.

---

**Bundled real-data snapshot (no network needed).** Two kinds of real data
ship in the repo so every reported number reproduces offline:

| Path | Contents |
|---|---|
| `data/market/` | Raw snapshot: `prices.csv`, `factors.csv`, `benchmark_weights.csv`, `provenance.json` — Yahoo Finance adjusted daily closes, 12 US large-caps + SPY/QQQ/IWM, **2015-01-02 → 2026-11-16**, `auto_adjust=True`, retrieved **2026-11-17T16:21:02+08:00** |
| `examples/real/` | Derived input pack estimated from that snapshot (OLS loadings, factor covariance, residual variances, historical-mean μ) |

Because the snapshot is committed, `run_backtest.py --mode market` runs with no
internet access. Adjusted closes are revised whenever a later dividend or split
occurs, so re-fetching will shift prices slightly — **quote the retrieval date
when reporting results**.

**How to start:**

```bash
pip install -r requirements.txt

# 1) optimize the bundled synthetic example
python scripts/run_optimization.py --inputs examples/basic --output outputs/run1 --print-report

# 2) run the full independent verification suite
python scripts/run_verification.py --inputs examples/basic

# 3) walk-forward backtest (rolling re-estimation, quarterly rebalancing, 10bp costs)
python scripts/run_backtest.py --inputs examples/basic --output outputs/backtest --plot

# 4) real-data mode (yfinance, optional): fetch prices → build pack → optimize → backtest
python scripts/fetch_market_data.py \
    --tickers AAPL,MSFT,GOOGL,AMZN,NVDA,JPM,JNJ,XOM,PG,KO,HD,MRK \
    --factors SPY,QQQ,IWM --start 2015-01-01 --end 2026-11-17 --outdir data/market
python scripts/prepare_inputs.py yfinance --outdir examples/real \
    --tickers AAPL,MSFT,GOOGL,AMZN,NVDA,JPM,JNJ,XOM,PG,KO,HD,MRK \
    --factors SPY,QQQ,IWM --start 2015-01-01 --end 2026-11-17 \
    --max-weight 0.15 --default-tilt-bound 0.10
python scripts/run_optimization.py --inputs examples/real --output outputs/run_real --print-report
python scripts/run_backtest.py --mode market --prices data/market/prices.csv \
    --factors data/market/factors.csv --benchmark-csv data/market/benchmark_weights.csv \
    --config examples/real/config.json --output outputs/backtest_market --plot

# 5) OPTIONAL — refresh the snapshot and rebuild the pack from scratch (network)
python scripts/fetch_market_data.py \
    --tickers AAPL,MSFT,GOOGL,AMZN,NVDA,JPM,JNJ,XOM,PG,KO,HD,MRK \
    --factors SPY,QQQ,IWM --start 2015-01-01 --end 2026-11-17 --outdir data/market
python scripts/prepare_inputs.py yfinance --outdir examples/real \
    --tickers AAPL,MSFT,GOOGL,AMZN,NVDA,JPM,JNJ,XOM,PG,KO,HD,MRK \
    --factors SPY,QQQ,IWM --start 2015-01-01 --end 2026-11-17 \
    --max-weight 0.15 --default-tilt-bound 0.10
python scripts/run_optimization.py --inputs examples/real --output outputs/run_real --print-report
python scripts/run_backtest.py --mode market --prices data/market/prices.csv \
    --factors data/market/factors.csv --benchmark-csv data/market/benchmark_weights.csv \
    --config examples/real/config.json --output outputs/backtest_market --plot
```

**One small example and its expected checks:**
`examples/basic` — 8 synthetic assets, 3 factors (MKT, F2≈collinear with MKT,
F3 independent). Expected after running:

- solver status `optimal`, compliance verdict **PASS**
- weights sum to 1.0 within 1e-6; every weight within [0, 0.30]
- every active tilt `B'(w − w_b)` inside its `config.json` bound
- verification suite: all checks pass, incl. brute-force and SLSQP
  cross-checks, the infeasible edge case (reported gracefully), and
  invalid-input rejection; artifacts in `outputs/verification/`

**Output files and what they mean:**

| File | Meaning |
|---|---|
| `weights.csv` | `ticker,weight,active_weight` (full precision) |
| `compliance_report.md` | human-readable verdict: per-factor tilt table, weight/concentration analytics, covariance-conditioning audit |
| `compliance.json` | same record, machine-readable |
| `evaluation_record.csv/.md` | compact evaluation record (case/expected/actual/pass-fail) |
| `baseline_comparison.csv` | optimized QP vs Equal-Weight (1/N) vs unregularized sample covariance on a design/test split |
| `backtest_summary.csv`, `equity_curves.csv`, `backtest_report.md`, `subperiod_breakdown.csv`, `backtest.png` | walk-forward backtest (rolling re-estimation + rebalancing + bps costs) across the four variants, plus a fixed-rule calendar-chunk sub-period table |
| `data/market/` | real-data mode: raw adjusted-close CSVs (assets + factor proxies), equal-weight benchmark CSV, `provenance.json` (retrieval timestamp, window, auto_adjust) |

**How to verify/reproduce the important result:**
`python scripts/run_verification.py` re-reads the SAVED weights and RAW input
CSVs and recomputes every constraint with plain numpy (independent of the
optimizer's internals); cross-checks a hand-computable 3-asset instance
against exhaustive grid search and against scipy SLSQP; and writes the
evaluation record. See `report.md` §Evaluation.

**Known failures and limitations:**

- Expected returns and covariances are estimates; garbage in, garbage out.
  Historical-mean `mu` is a deliberately noisy default (documented).
- Single-period, buy-and-hold semantics in the baseline comparison; no
  transaction costs beyond the optional turnover cap.
- In-sample design-window estimation biases all optimized variants; the
  equal-weight reference is immune to it — this is why 1/N is the honest
  baseline.
- Infeasible constraint sets are reported (verdict `INFEASIBLE`), never
  silently relaxed.
- Not investment advice.

---

## Repository layout

```
final-project/
├── README.md                  # this file (Appendix F template)
├── report.md                  # project report (→ report.pdf for submission)
├── requirements.txt
├── LICENSE
├── scripts/
│   ├── fp_optimizer/          # core package
│   │   ├── config.py          # OptimizerConfig (objective, regularization, bounds)
│   │   ├── data.py            # input packs, validation, OLS/yfinance/synthetic builders
│   │   ├── covariance.py      # factor-model Sigma, shrinkage, ridge, conditioning
│   │   ├── optimizer.py       # CVXPY convex QP
│   │   ├── compliance.py      # active-factor compliance report
│   │   ├── backtest.py        # walk-forward backtest (costs, drift, variants)
│   │   └── verify.py          # independent verification suite
│   ├── run_optimization.py    # entry point: inputs → weights + report
│   ├── prepare_inputs.py      # entry point: build synthetic / yfinance packs
│   ├── fetch_market_data.py   # entry point: save raw yfinance price CSVs + provenance
│   ├── run_backtest.py        # entry point: walk-forward backtest
│   └── run_verification.py    # entry point: full verification suite
├── examples/basic/            # lawful small synthetic example + data dictionary
├── examples/real/             # real-data pack (yfinance; generated, has provenance.json)
├── data/market/               # raw downloaded price CSVs (generated by fetch script)
├── skills/factor-portfolio-optimizer/   # reusable Skill package
└── outputs/                   # run artifacts (git-ignored; commit selected results)
```
