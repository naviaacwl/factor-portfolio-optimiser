#!/usr/bin/env python3
"""Run the full independent verification suite.

What it does (see fp_optimizer/verify.py for details):
  1. loads the input pack, solves the QP, saves weights + compliance report
  2. re-verifies constraints FROM THE SAVED FILES with plain numpy
     (weights sum to 1.0; every active tilt within bounds; box constraints)
  3. cross-checks a tiny hand-computable instance against brute-force grid
     search and against a second solver (scipy SLSQP)
  4. checks shrinkage/ridge regularization properties on a near-collinear
     instance (PSD preserved, condition number improved)
  5. runs the lightweight historical comparison on a design/test split:
     optimized QP vs Equal-Weight (1/N) vs unregularized sample covariance
     (synthetic seeded returns by default; real price CSVs via --prices)
  6. exercises three guaranteed-infeasible constraint sets (must be reported
     gracefully, never crash)
  7. exercises invalid inputs (must raise InputValidationError)

Artifacts written to --run-dir (default outputs/verification):
  weights.csv, compliance_report.md, compliance.json,
  evaluation_record.csv, evaluation_record.md, baseline_comparison.csv

Usage (from the project root):
    python scripts/run_verification.py
    python scripts/run_verification.py --inputs examples/basic
    python scripts/run_verification.py --baseline-mode market \
        --prices data/prices.csv --factors data/factor_prices.csv \
        --design-end 2022-12-30

Exit codes: 0 = all checks passed, 1 = at least one check failed,
            3 = input/environment error.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fp_optimizer.config import ConfigError
from fp_optimizer.data import InputValidationError
from fp_optimizer.optimizer import SolverNotAvailableError
from fp_optimizer.verify import run_all


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Independent verification suite for the factor-portfolio QP.")
    parser.add_argument("--inputs", default=None,
                        help="input pack directory (default: examples/basic)")
    parser.add_argument("--config", default=None,
                        help="config.json path (default: <inputs>/config.json)")
    parser.add_argument("--run-dir", default=None,
                        help="artifact directory (default: outputs/verification)")
    parser.add_argument("--baseline-mode", choices=["synthetic", "market"],
                        default="synthetic",
                        help="synthetic = seeded offline simulation (default); "
                             "market = real price CSVs via --prices/--factors")
    parser.add_argument("--prices", default=None,
                        help="market mode: wide CSV of daily adjusted closes "
                             "(a 'date' column + one column per asset)")
    parser.add_argument("--factors", default=None,
                        help="market mode: wide CSV of daily adjusted closes "
                             "for factor-proxy ETFs")
    parser.add_argument("--design-end", default=None,
                        help="market mode: last design date (YYYY-MM-DD); "
                             "default: 60%% point of the common window")
    parser.add_argument("--tiny-step", type=float, default=0.01,
                        help="grid step for the brute-force tiny case")
    args = parser.parse_args(argv)

    try:
        records, df, summary = run_all(
            inputs_dir=args.inputs, config_path=args.config,
            run_dir=args.run_dir, baseline_mode=args.baseline_mode,
            prices_csv=args.prices, factors_csv=args.factors,
            design_end=args.design_end, tiny_step=args.tiny_step)
    except (InputValidationError, ConfigError, SolverNotAvailableError,
            RuntimeError, ValueError) as exc:
        print(f"[error] {type(exc).__name__}: {exc}")
        return 3

    failed = df[~df["passed"]]
    if len(failed):
        print("\nFAILED CHECKS:")
        for _, r in failed.iterrows():
            print(f"  - {r['case']}: expected {r['expected']!r}, "
                  f"got {r['actual']!r} ({r['note']})")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
