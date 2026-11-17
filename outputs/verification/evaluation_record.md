# Evaluation Record (independent verification suite)

- Inputs: `examples/basic`  ·  Config: `examples/basic/config.json`
- Baseline data mode: synthetic (seeded simulation from the example factor model)
- Design/test split at: 2021-06-14 00:00:00
- Result: **34/34 checks passed**
- Baseline comparison provenance: {'design_window': ['2018-01-02 00:00:00', '2021-06-14 00:00:00'], 'test_window': ['2021-06-15 00:00:00', '2023-10-02 00:00:00'], 'design_end': '2021-06-14 00:00:00', 'n_design_obs': 900, 'n_test_obs': 600, 'estimation': {'n_obs': 900, 'first_date': '2018-01-02 00:00:00', 'last_date': '2021-06-14 00:00:00', 'periods_per_year': 252, 'mu_source': 'historical mean (noisy in-sample estimate)', 'benchmark_source': 'equal weight (1/N)', 'estimation': 'OLS factor model; F = sample factor covariance; residual_var with ddof = K+1'}, 'rule': 'weights fixed once at design end; buy-and-hold over test; no test-window information used in design', 'note': 'the harness benchmark defaults to equal weight; if a user-supplied benchmark equals 1/N, the benchmark_design and equal_weight_1_over_N rows coincide by construction'}
| # | Case | Method | Expected | Actual | Pass/Fail | Note |
|---|------|--------|----------|--------|-----------|------|
| 1 | inputs load + validate | load_inputs_from_dir | no error | n_assets=8, n_factors=3, config=file | PASS |  |
| 2 | main optimization | CVXPY QP (OSQP) | status=optimal, verdict=PASS | status=optimal, verdict=PASS | PASS |  |
| 3 | ticker alignment | set comparison | identical ticker sets | 8 tickers match | PASS |  |
| 4 | weights sum to 1.0 | recomputed from weights.csv | 1.0 within 1e-06 | 1.0000000000 (dev 2.220e-16) | PASS |  |
| 5 | position box constraints | recomputed from weights.csv | all weights in [0, 0.3] | max lower violation 5.767e-23, max upper violation 0.000e+00 | PASS |  |
| 6 | active tilt within bounds: MKT | recomputed as B'(w - w_b) from raw CSVs | [-0.1000, +0.1000] | -0.010710 (violation 0.000e+00) | PASS |  |
| 7 | active tilt within bounds: F2 | recomputed as B'(w - w_b) from raw CSVs | [-0.0500, +0.0500] | -0.013431 (violation 0.000e+00) | PASS |  |
| 8 | active tilt within bounds: F3 | recomputed as B'(w - w_b) from raw CSVs | [-0.0800, +0.0800] | +0.080000 (violation 0.000e+00) | PASS |  |
| 9 | tiny instance: QP solves | CVXPY QP | status optimal | status=optimal | PASS |  |
| 10 | tiny instance: objective vs brute force | grid search step=0.01 (5151 simplex points) | QP objective <= grid objective, gap < 1e-3 | qp=0.01348593, grid=0.01349375, gap=7.824e-06 | PASS | gap bounded by grid resolution; QP optimum is global (convexity) |
| 11 | tiny instance: weights vs brute force | grid search argmin | max |w_qp - w_grid| <= 0.03 (3 x step) | 0.0096 (qp=[0.3504, 0.2748, 0.3748], grid=[0.36, 0.27, 0.37]) | PASS |  |
| 12 | second solver: SLSQP converges | scipy.optimize.minimize(SLSQP) | success | success=True (Optimization terminated successfully) | PASS |  |
| 13 | second solver: objective agreement | SLSQP vs CVXPY objective | |diff| <= 1.00e-07 | cvxpy=0.0003439510, slsqp=0.0003439510, diff=3.103e-15 | PASS |  |
| 14 | second solver: weight agreement | max abs weight difference | <= 0.002 (unique optimum, strict convexity) | 1.610e-07 | PASS |  |
| 15 | regularization: delta=0 is identity | shrink_covariance(S, 0) | unchanged matrix | max abs diff 0.000e+00 | PASS |  |
| 16 | regularization: shrunk Sigma symmetric PSD | shrink_covariance(S, 0.3, 'diagonal') on corr=0.999 instance | symmetric, min eigenvalue >= -1e-12 | symmetric=True, min eig=4.056e-02 | PASS |  |
| 17 | regularization: condition number improves | cond(raw) vs cond(shrunk) on near-collinear instance | cond(shrunk) < cond(raw) = 2.184e+01 | cond(shrunk) = 1.204e+01 | PASS | descriptive expectation for the deliberately collinear instance |
| 18 | regularization: ridge keeps PSD | ridge_covariance(shrunk, 1e-8) | min eigenvalue >= 0 | min eig=4.056e-02 | PASS |  |
| 19 | regularization: scaled-identity target PSD | shrink_covariance(S, 0.5, 'scaled_identity') | min eigenvalue >= -1e-12 | min eig=6.972e-02 | PASS |  |
| 20 | baselines: all variants constructed and solved | design-window estimation + QP solves | both QP variants optimal; optimized tilts within bounds | optimized=optimal, unregularized=optimal, tilts_ok=True, table rows=4 | PASS |  |
| 21 | baselines: realized metrics finite | test-window evaluation | all reported metrics finite | all_finite=True | PASS |  |
| 22 | baselines: lowest realized tracking error (descriptive) | test-window comparison | reported, no assertion | equal_weight_1_over_N (TE=0.0000) | PASS | descriptive outcome; underperformance is a valid result |
| 23 | baselines: highest realized active return (descriptive) | test-window comparison | reported, no assertion | equal_weight_1_over_N (geo active=+0.0000) | PASS | descriptive outcome; underperformance is a valid result |
| 24 | baselines: covariance conditioning (descriptive) | cond(Sigma) of each variant | reported, no assertion | shrunk factor-model=8.200e+00, raw sample=9.456e+00 | PASS | regularization is expected to lower the condition number; actual values reported for audit |
| 25 | infeasible edge case A: contradictory tilt bounds (lo > hi) | CVXPY QP + compliance report | status=infeasible, verdict=INFEASIBLE, no crash, no weights | status=infeasible, verdict=INFEASIBLE, detail=[0.2, -0.2] | PASS | constraints cannot be satisfied simultaneously (solver status: infeasible). Relax tilt bounds, position caps, or review contradictory config (e.g. tilt lo > hi, |
| 26 | infeasible edge case B: unreachable tilt (>= 3.600) | CVXPY QP + compliance report | status=infeasible, verdict=INFEASIBLE, no crash, no weights | status=infeasible, verdict=INFEASIBLE, detail=max attainable |tilt| <= 2.600 | PASS | constraints cannot be satisfied simultaneously (solver status: infeasible). Relax tilt bounds, position caps, or review contradictory config (e.g. tilt lo > hi, |
| 27 | infeasible edge case C: box cannot sum to 1 (min_weight=0.5) | CVXPY QP + compliance report | status=infeasible, verdict=INFEASIBLE, no crash, no weights | status=infeasible, verdict=INFEASIBLE, detail=n*0.5 > 1 | PASS | constraints cannot be satisfied simultaneously (solver status: infeasible). Relax tilt bounds, position caps, or review contradictory config (e.g. tilt lo > hi, |
| 28 | invalid input rejected: benchmark weights sum to 0.9 | validate_inputs | InputValidationError | raised: benchmark_weights sum to 0.9000000000, expected 1.0 within 1e-06 | PASS |  |
| 29 | invalid input rejected: factor covariance not PSD | validate_inputs | InputValidationError | raised: F (factor_covariance) is not positive semi-definite (min eigenvalue -1.026e+00); fix the estimate (or apply co | PASS |  |
| 30 | invalid input rejected: factor covariance asymmetric | validate_inputs | InputValidationError | raised: F (factor_covariance) is not symmetric (max asymmetry 5.000e-01); fix the source file or symmetrize explicitly | PASS |  |
| 31 | invalid input rejected: expected return is NaN | validate_inputs | InputValidationError | raised: mu (expected_returns): contains NaN or infinite values | PASS |  |
| 32 | invalid input rejected: loadings shape mismatch | validate_inputs | InputValidationError | raised: B: shape (7, 3) does not match expected (8, 3) | PASS |  |
| 33 | invalid input rejected: negative residual variance | validate_inputs | InputValidationError | raised: residual_variance contains negative values | PASS |  |
| 34 | invalid input rejected: missing input dir | load_inputs_from_dir | InputValidationError | raised: input directory not found: /var/folders/wy/np9j7h4x2j3484vc6rfj41hw0000gn/T/tmppj2p1hu4/does_not_exist | PASS |  |