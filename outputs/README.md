# outputs/

Run artifacts land here. The **selected results referenced in `report.md` are
committed on purpose** so a reviewer can check the claims without rerunning
anything; bulk regenerable files (daily equity curves, per-rebalance turnover
logs) are git-ignored — see `.gitignore`.

```
outputs/
├── run1/                      # run_optimization.py --inputs examples/basic
│   ├── weights.csv            # weights + active weights, full precision
│   ├── compliance_report.md   # human-readable compliance report
│   └── compliance.json        # machine-readable record
├── run_real/                  # run_optimization.py --inputs examples/real
│   └── (same three files, real market data)
├── verification/              # run_verification.py
│   ├── evaluation_record.csv / .md   # compact evaluation record (34 checks)
│   └── baseline_comparison.csv       # optimized vs 1/N vs unregularized
├── backtest/                  # run_backtest.py (synthetic returns)
│   ├── backtest_summary.csv
│   ├── subperiod_breakdown.csv
│   └── backtest_report.md
└── backtest_market/           # run_backtest.py --mode market (real prices)
    ├── backtest_summary.csv
    ├── subperiod_breakdown.csv
    ├── backtest_report.md
    ├── backtest.png           # equity curves (log) + drawdown panel
    └── weights_history_optimized_qp_shrunk.csv
```

To regenerate everything from scratch:

```bash
python scripts/run_optimization.py --inputs examples/basic --output outputs/run1 --print-report
python scripts/run_optimization.py --inputs examples/real  --output outputs/run_real --print-report
python scripts/run_verification.py --inputs examples/basic
python scripts/run_backtest.py     --inputs examples/basic --output outputs/backtest --plot
# real-data backtest needs data/market/ first (see README step 4)
python scripts/run_backtest.py --mode market --prices data/market/prices.csv \
    --factors data/market/factors.csv \
    --benchmark-csv data/market/benchmark_weights.csv \
    --config examples/real/config.json --output outputs/backtest_market --plot
```
