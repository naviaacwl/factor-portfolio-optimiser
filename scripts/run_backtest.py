#!/usr/bin/env python3
"""Walk-forward backtest entry point.

Variants (all on one common dataset, same schedule/costs, no lookahead):
  benchmark_buy_hold          buy-and-hold of the benchmark weights (no costs)
  optimized_qp_shrunk         rolling re-estimation + QP with shrinkage (project method)
  unregularized_sample_cov    same, but Sigma = raw rolling sample covariance
  equal_weight_1_over_N       rebalanced 1/N heuristic (pays the same costs)

Modes:
  synthetic (default) : returns simulated (seeded) from an input pack's factor
                        model -> fully offline and reproducible; clearly labeled
  market              : daily adjusted-close CSVs (assets + factor proxies),
                        e.g. downloaded via yfinance

Usage (from the project root):
  python scripts/run_backtest.py --inputs examples/basic --output outputs/backtest --plot
  python scripts/run_backtest.py --mode market --prices data/prices.csv \
      --factors data/factor_prices.csv --benchmark-csv data/bench.csv \
      --config data/config.json --output outputs/backtest_market --plot
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fp_optimizer.backtest import (BacktestConfig, render_backtest_markdown,
                                   run_backtest, save_backtest)
from fp_optimizer.config import ConfigError, OptimizerConfig
from fp_optimizer.data import (InputValidationError, load_inputs_from_dir)
from fp_optimizer.verify import _load_wide_prices, make_synthetic_returns


def _series_from_csv(path, col: str) -> pd.Series:
    df = pd.read_csv(path)
    if "ticker" not in df.columns or col not in df.columns:
        raise SystemExit(f"[error] {path} must have columns: ticker,{col}")
    return df.set_index(df["ticker"].astype(str))[col]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Walk-forward backtest of the factor-portfolio QP.")
    parser.add_argument("--mode", choices=["synthetic", "market"],
                        default="synthetic")
    parser.add_argument("--inputs", default="examples/basic",
                        help="input pack (synthetic mode: factor model source; "
                             "also supplies config.json/benchmark when present)")
    parser.add_argument("--config", default=None,
                        help="optimizer config.json (default: <inputs>/config.json)")
    # market mode
    parser.add_argument("--prices", default=None,
                        help="market mode: wide CSV of daily adjusted closes (assets)")
    parser.add_argument("--factors", default=None,
                        help="market mode: wide CSV of daily adjusted closes (factor proxies)")
    parser.add_argument("--benchmark-csv", default=None,
                        help="market mode: benchmark weights CSV (ticker,weight); "
                             "default equal weight")
    # synthetic mode
    parser.add_argument("--n-days", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=99)
    # backtest knobs
    parser.add_argument("--train-window", type=int, default=504)
    parser.add_argument("--rebalance-freq", type=int, default=63)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--subperiod-years", type=int, default=3,
                        help="calendar-chunk length for the sub-period table")
    parser.add_argument("--output", default="outputs/backtest")
    parser.add_argument("--plot", action="store_true",
                        help="also write backtest.png (requires matplotlib)")
    args = parser.parse_args(argv)

    try:
        inputs_dir = Path(args.inputs)
        cfg_path = Path(args.config) if args.config else inputs_dir / "config.json"
        cfg = (OptimizerConfig.from_json(cfg_path) if cfg_path.exists()
               else OptimizerConfig())

        if args.mode == "synthetic":
            pi = load_inputs_from_dir(inputs_dir)
            asset_rets, factor_rets = make_synthetic_returns(
                pi, n_days=args.n_days, seed=args.seed)
            benchmark = pd.Series(pi.w_b, index=[str(t) for t in pi.tickers])
            data_label = (f"SYNTHETIC returns simulated (seed={args.seed}) from the "
                          f"factor model in {inputs_dir}")
            provenance = {
                "mode": "synthetic", "seed": args.seed,
                "n_days": args.n_days,
                "source_pack": str(inputs_dir),
                "note": "seeded simulation from the pack's B/F/D; NOT real market data",
            }
        else:
            if not (args.prices and args.factors):
                raise SystemExit("[error] market mode requires --prices and --factors CSVs")
            ap = _load_wide_prices(args.prices)
            fp = _load_wide_prices(args.factors)
            asset_rets = ap.pct_change().dropna(how="any")
            factor_rets = fp.pct_change().dropna(how="any")
            common = asset_rets.index.intersection(factor_rets.index)
            asset_rets, factor_rets = asset_rets.loc[common], factor_rets.loc[common]
            if args.benchmark_csv:
                benchmark = _series_from_csv(args.benchmark_csv, "weight")
            else:
                from fp_optimizer.data import equal_weight
                import numpy as np
                benchmark = pd.Series(
                    equal_weight(asset_rets.shape[1]), index=list(asset_rets.columns))
            data_label = (f"MARKET data: {args.prices} / {args.factors} "
                          f"(daily adjusted closes, simple returns)")
            provenance = {
                "mode": "market",
                "prices_csv": str(args.prices), "factors_csv": str(args.factors),
                "benchmark": str(args.benchmark_csv) if args.benchmark_csv else "equal weight",
                "first_date": str(common.min()), "last_date": str(common.max()),
                "n_obs": int(len(common)),
            }

        bt = BacktestConfig(train_window=args.train_window,
                            rebalance_freq=args.rebalance_freq,
                            cost_bps=args.cost_bps,
                            subperiod_years=args.subperiod_years)
        report = run_backtest(asset_rets, factor_rets, benchmark,
                              cfg_opt=cfg, bt=bt,
                              data_label=data_label, provenance=provenance)
        paths = save_backtest(report, args.output, plot=args.plot)
    except (InputValidationError, ConfigError, ValueError, RuntimeError) as exc:
        print(f"[error] {type(exc).__name__}: {exc}")
        return 3

    print(render_backtest_markdown(report))
    print("\n[saved]")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
