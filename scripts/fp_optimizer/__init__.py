"""fp_optimizer -- benchmark-relative factor portfolio optimization via convex QP.

HKUST "Finance with AI Agents" final project core package.

For a quantitative portfolio analyst: takes asset expected returns, factor
loadings, factor covariance, and benchmark weights; produces optimized
portfolio weights and an active-factor compliance report.

Modules
-------
config      OptimizerConfig: objective weights, regularization, constraints
data        input pack loading/validation; builders from return data (OLS
            factor model, yfinance fetch, seeded synthetic)
covariance  factor-model Sigma assembly; Ledoit-Wolf-style shrinkage; ridge;
            condition-number diagnostics (multicollinearity defense)
optimizer   CVXPY convex QP: tracking error vs factor tilts vs concentration
compliance  active-factor compliance report (Markdown / JSON / CSV)
backtest    walk-forward backtest: rolling re-estimation, periodic rebalancing,
            transaction costs, buy-and-hold drift; variants vs 1/N and
            unregularized sample covariance
verify      independent verification suite (file-level constraint rechecks,
            brute-force and SLSQP cross-checks, 1/N and unregularized-sample-
            covariance baselines, infeasible edge case, invalid inputs)

See the project README.md for usage and report.md for the full write-up.
"""
from .backtest import (BacktestConfig, BacktestReport, VariantResult,
                       max_drawdown, performance_metrics,
                       render_backtest_markdown, run_backtest, save_backtest,
                       subperiod_breakdown)
from .compliance import (ComplianceReport, FactorComplianceRow,
                         build_compliance_report, render_markdown,
                         save_compliance_report)
from .config import ConfigError, OptimizerConfig
from .covariance import (condition_number, factor_model_covariance,
                         min_eigenvalue, nearest_psd, ridge_covariance,
                         sample_covariance, shrink_covariance)
from .data import (InputValidationError, PortfolioInputs,
                   build_inputs_from_returns, equal_weight,
                   fetch_yfinance_inputs, load_inputs_from_dir,
                   synthetic_inputs, validate_inputs)
from .optimizer import (OptimizationResult, SolverNotAvailableError,
                        objective_value_np, solve_factor_portfolio)

__version__ = "0.1.0"

__all__ = [
    "BacktestConfig", "BacktestReport", "VariantResult", "max_drawdown",
    "performance_metrics", "render_backtest_markdown", "run_backtest",
    "save_backtest",
    "ComplianceReport", "FactorComplianceRow", "build_compliance_report",
    "render_markdown", "save_compliance_report",
    "ConfigError", "OptimizerConfig",
    "condition_number", "factor_model_covariance", "min_eigenvalue",
    "nearest_psd", "ridge_covariance", "sample_covariance", "shrink_covariance",
    "InputValidationError", "PortfolioInputs", "build_inputs_from_returns",
    "equal_weight", "fetch_yfinance_inputs", "load_inputs_from_dir",
    "synthetic_inputs", "validate_inputs",
    "OptimizationResult", "SolverNotAvailableError", "objective_value_np",
    "solve_factor_portfolio",
    "__version__",
]
