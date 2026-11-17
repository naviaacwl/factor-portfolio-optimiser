# Walk-Forward Backtest Report

- Data: **SYNTHETIC returns simulated (seed=99) from the factor model in examples/basic**
- Window: 2018-01-02 00:00:00 .. 2025-09-01 00:00:00  ·  2000 trading days
- Train window: 504 days  ·  rebalance every 63 days  ·  cost: 10 bps one-way ·  Sharpe rf = 0
- Information timing: rebalance on day t uses data up to t-1; no lookahead. All variants start at the benchmark weights and share estimation windows, mu estimator (historical mean), schedule, and costs. Benchmark = buy-and-hold of w_b, no costs.
- Provenance: `{'mode': 'synthetic', 'seed': 99, 'n_days': 2000, 'source_pack': 'examples/basic', 'note': "seeded simulation from the pack's B/F/D; NOT real market data"}`

## Summary (realized, net of costs)

| variant | total_return | cagr | ann_vol | sharpe_rf0 | max_drawdown | tracking_error_vs_bench | active_return_geo | avg_turnover_per_rebal | total_cost_drag | n_rebalances | n_solver_failures |
|---|---|---|---|---|---|---|---|---|---|---|---|
| benchmark_buy_hold | 0.6031 | 0.0613 | 0.2144 | 0.385 | -0.4354 | 0.0 | 0.0 |  | 0.0 | 0 | 0 |
| optimized_qp_shrunk | 0.302 | 0.0338 | 0.2144 | 0.262 | -0.4731 | 0.0585 | -0.1878 | 0.4052 | 0.00973 | 24 | 0 |
| unregularized_sample_cov | 0.3012 | 0.0337 | 0.2144 | 0.262 | -0.4735 | 0.0584 | -0.1883 | 0.405 | 0.00972 | 24 | 0 |
| equal_weight_1_over_N | 0.384 | 0.0418 | 0.2053 | 0.302 | -0.4502 | 0.0562 | -0.1367 | 0.1057 | 0.00254 | 24 | 0 |

## Sub-period breakdown (out-of-sample span, fixed 3-calendar-year chunks)

| sub-period | days | bench ret | bench maxDD | QP ret | QP maxDD | 1/N ret | 1/N maxDD | QP vs 1/N |
|---|---|---|---|---|---|---|---|---|
| 2019-2021 | 540 | 0.1223 | -0.3214 | 0.155 | -0.3101 | 0.014 | -0.3622 | 0.1391 |
| 2022-2024 | 782 | 0.333 | -0.2457 | 0.214 | -0.2988 | 0.3889 | -0.2521 | -0.1259 |
| 2025 | 174 | 0.2134 | -0.1627 | 0.0637 | -0.1921 | 0.1322 | -0.1675 | -0.0605 |

*Chunk rule: consecutive calendar-year blocks starting at the first-rebalance year; only the final chunk may be partial. Boundaries are fixed by the rule, never chosen after seeing results. Warm-up days (no trades yet) are excluded.*

### optimized_qp_shrunk
- rebalances: 24  ·  solver/estimation failures (held weights): 0

### unregularized_sample_cov
- rebalances: 24  ·  solver/estimation failures (held weights): 0

### equal_weight_1_over_N
- rebalances: 24  ·  solver/estimation failures (held weights): 0

---
*Descriptive historical simulation, laptop-sized, rules fixed before running. Underperformance is a valid result. Not investment advice.*