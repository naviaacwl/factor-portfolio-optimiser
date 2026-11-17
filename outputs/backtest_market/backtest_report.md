# Walk-Forward Backtest Report

- Data: **MARKET data: data/market/prices.csv / data/market/factors.csv (daily adjusted closes, simple returns)**
- Window: 2015-01-05 00:00:00 .. 2026-11-16 00:00:00  ·  2985 trading days
- Train window: 504 days  ·  rebalance every 63 days  ·  cost: 10 bps one-way ·  Sharpe rf = 0
- Information timing: rebalance on day t uses data up to t-1; no lookahead. All variants start at the benchmark weights and share estimation windows, mu estimator (historical mean), schedule, and costs. Benchmark = buy-and-hold of w_b, no costs.
- Provenance: `{'mode': 'market', 'prices_csv': 'data/market/prices.csv', 'factors_csv': 'data/market/factors.csv', 'benchmark': 'data/market/benchmark_weights.csv', 'first_date': '2015-01-05 00:00:00', 'last_date': '2026-11-16 00:00:00', 'n_obs': 2985}`

## Summary (realized, net of costs)

| variant | total_return | cagr | ann_vol | sharpe_rf0 | max_drawdown | tracking_error_vs_bench | active_return_geo | avg_turnover_per_rebal | total_cost_drag | n_rebalances | n_solver_failures |
|---|---|---|---|---|---|---|---|---|---|---|---|
| benchmark_buy_hold | 33.4572 | 0.3483 | 0.3165 | 1.103 | -0.4897 | 0.0 | 0.0 |  | 0.0 | 0 | 0 |
| optimized_qp_shrunk | 10.4414 | 0.2285 | 0.1895 | 1.181 | -0.2843 | 0.1909 | -0.668 | 0.3858 | 0.01543 | 40 | 0 |
| unregularized_sample_cov | 10.449 | 0.2285 | 0.1894 | 1.182 | -0.2842 | 0.191 | -0.6677 | 0.3871 | 0.01548 | 40 | 0 |
| equal_weight_1_over_N | 12.0248 | 0.242 | 0.175 | 1.327 | -0.297 | 0.2131 | -0.622 | 0.0965 | 0.00386 | 40 | 0 |

## Sub-period breakdown (out-of-sample span, fixed 3-calendar-year chunks)

| sub-period | days | bench ret | bench maxDD | QP ret | QP maxDD | 1/N ret | 1/N maxDD | QP vs 1/N |
|---|---|---|---|---|---|---|---|---|
| 2017-2019 | 753 | 0.9845 | -0.3435 | 0.8494 | -0.2459 | 0.8732 | -0.1926 | -0.0127 |
| 2020-2022 | 756 | 0.6584 | -0.4897 | 0.5213 | -0.2843 | 0.5675 | -0.297 | -0.0295 |
| 2023-2025 | 752 | 5.934 | -0.3347 | 1.0233 | -0.1897 | 1.1793 | -0.1535 | -0.0716 |
| 2026 | 220 | -0.1215 | -0.2967 | 0.1865 | -0.0579 | 0.1992 | -0.0618 | -0.0106 |

*Chunk rule: consecutive calendar-year blocks starting at the first-rebalance year; only the final chunk may be partial. Boundaries are fixed by the rule, never chosen after seeing results. Warm-up days (no trades yet) are excluded.*

### optimized_qp_shrunk
- rebalances: 40  ·  solver/estimation failures (held weights): 0

### unregularized_sample_cov
- rebalances: 40  ·  solver/estimation failures (held weights): 0

### equal_weight_1_over_N
- rebalances: 40  ·  solver/estimation failures (held weights): 0

---
*Descriptive historical simulation, laptop-sized, rules fixed before running. Underperformance is a valid result. Not investment advice.*