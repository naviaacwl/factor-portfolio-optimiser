#!/usr/bin/env python3
"""Main entry point: inputs -> optimized weights + active-factor compliance report.

Usage (from the project root):
    python scripts/run_optimization.py --inputs examples/basic --output outputs/run1
    python scripts/run_optimization.py --inputs examples/basic \
        --config my_config.json --output outputs/run2 --print-report

Exit codes:
    0 = compliance verdict PASS
    1 = compliance verdict FAIL (solved, but an independent recheck violated)
    2 = INFEASIBLE or SOLVER_ERROR (reported gracefully, artifacts still saved)
    3 = input/config error (validation failed before solving)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# make `fp_optimizer` importable regardless of the working directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fp_optimizer.compliance import (build_compliance_report, render_markdown,
                                     save_compliance_report)
from fp_optimizer.config import ConfigError, OptimizerConfig
from fp_optimizer.data import InputValidationError, load_inputs_from_dir
from fp_optimizer.optimizer import SolverNotAvailableError, solve_factor_portfolio


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark-relative factor portfolio optimization (convex QP).")
    parser.add_argument("--inputs", required=True,
                        help="input pack directory (see examples/basic)")
    parser.add_argument("--config", default=None,
                        help="config.json path (default: <inputs>/config.json, "
                             "else built-in defaults)")
    parser.add_argument("--output", default="outputs/run1",
                        help="output directory for weights.csv / compliance report")
    parser.add_argument("--print-report", action="store_true",
                        help="print the full Markdown compliance report to stdout")
    args = parser.parse_args(argv)

    inputs_dir = Path(args.inputs)
    output_dir = Path(args.output)
    config_path = Path(args.config) if args.config else inputs_dir / "config.json"

    try:
        if config_path.exists():
            cfg = OptimizerConfig.from_json(config_path)
            print(f"[config] loaded {config_path}")
        else:
            cfg = OptimizerConfig()
            print(f"[config] none found at {config_path}; using built-in defaults")
        pi = load_inputs_from_dir(inputs_dir)
    except (ConfigError, InputValidationError) as exc:
        print(f"[input/config error] {exc}")
        return 3
    except SolverNotAvailableError as exc:
        print(f"[environment error] {exc}")
        return 3

    print(f"[inputs] {pi.n_assets} assets, {pi.n_factors} factors "
          f"({', '.join(str(f) for f in pi.factor_names)}), "
          f"source: {inputs_dir}")
    print(cfg.describe())

    result = solve_factor_portfolio(pi, cfg)
    report = build_compliance_report(result, pi, cfg)
    paths = save_compliance_report(report, result, pi, cfg, output_dir)

    if args.print_report:
        print()
        print(render_markdown(report))
    else:
        print(f"\n[result] solver status : {result.status}"
              + (f" ({result.solver_name})" if result.solver_name else ""))
        print(f"[result] verdict       : {report.verdict}")
        if result.success:
            rs = report.risk_summary
            ws = report.weights_summary
            print(f"[result] tracking error: {rs['tracking_error_annual']:.4%} (annual, ex-ante)")
            print(f"[result] active return : {rs['expected_active_return']:+.4%} (expected)")
            print(f"[result] weight sum    : {ws['sum_of_weights']:.10f}")
            print(f"[result] max weight    : {ws['max_weight']:.4f} (cap {ws['max_weight_cap']:g})")
            print(f"[result] effective N   : {ws['effective_n']:.2f}")
            for r in report.factors:
                flag = "OK " if r.within_bounds else "OUT"
                print(f"[tilt] {flag} {r.factor:>6}: {r.active_tilt:+.4f} in "
                      f"[{r.tilt_lower:+.3f}, {r.tilt_upper:+.3f}]")
        else:
            print(f"[result] {result.message}")

    print("\n[saved]")
    for k, v in paths.items():
        print(f"  {k}: {v}")

    if report.verdict == "PASS":
        return 0
    if report.verdict == "FAIL":
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
