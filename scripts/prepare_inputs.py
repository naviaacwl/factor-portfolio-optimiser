#!/usr/bin/env python3
"""Build input packs for the factor-portfolio optimizer.

Modes
-----
synthetic : seeded offline pack (no network). Larger universe with a
            near-collinear factor pair, useful for regularization experiments.
yfinance  : download adjusted daily closes for assets and factor-proxy ETFs,
            estimate the factor model by OLS, and record provenance (tickers,
            window, retrieval timestamp, auto_adjust setting) as the course
            guide requires. Validate the download; never fabricate data.

Usage (from the project root):
    python scripts/prepare_inputs.py synthetic \
        --outdir examples/synthetic30 --n-assets 30 --n-factors 3 --seed 11

    python scripts/prepare_inputs.py yfinance \
        --outdir examples/real --tickers SPY,QQQ,IWM,EFA,TLT,GLD,VNQ,DBC \
        --factors SPY,USMV,VLUE --start 2018-01-01 --end 2024-12-31 \
        [--benchmark-csv path/to/benchmark_weights.csv] [--mu-csv path/to/mu.csv]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fp_optimizer import __version__
from fp_optimizer.config import OptimizerConfig
from fp_optimizer.data import (InputValidationError, PortfolioInputs,
                               fetch_yfinance_inputs, synthetic_inputs)


def write_input_pack(pi: PortfolioInputs, cfg: OptimizerConfig, outdir,
                     provenance: dict, data_kind: str) -> Path:
    """Persist a PortfolioInputs + config as a standard input-pack directory."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    tickers = [str(t) for t in pi.tickers]
    factors = [str(f) for f in pi.factor_names]

    pd.DataFrame({"ticker": tickers, "mu": pi.mu}) \
        .to_csv(out / "expected_returns.csv", index=False)

    loadings = pd.DataFrame(pi.B, columns=factors)
    loadings.insert(0, "ticker", tickers)
    loadings.to_csv(out / "factor_loadings.csv", index=False)

    fcov = pd.DataFrame(pi.F, index=factors, columns=factors)
    fcov.index.name = "factor"
    fcov.to_csv(out / "factor_covariance.csv")

    pd.DataFrame({"ticker": tickers, "weight": pi.w_b}) \
        .to_csv(out / "benchmark_weights.csv", index=False)

    pd.DataFrame({"ticker": tickers, "residual_variance": pi.residual_var}) \
        .to_csv(out / "residual_variance.csv", index=False)

    cfg.to_json(out / "config.json")

    provenance = dict(provenance)
    provenance.update({
        "data_kind": data_kind,  # "synthetic" or "real (yfinance)"
        "generated_by": f"prepare_inputs.py (fp_optimizer {__version__})",
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "units": "mu/F/residual_variance are ANNUAL decimals; weights sum to 1.0",
    })
    (out / "provenance.json").write_text(
        json.dumps(provenance, indent=2, default=str), encoding="utf-8")

    (out / "data_dictionary.md").write_text(
        "\n".join([
            f"# Data dictionary -- {out.name}",
            "",
            f"- Data kind: **{provenance['data_kind']}**",
            f"- Generated: {provenance['generated_at']} by {provenance['generated_by']}",
            f"- Units: expected returns, factor covariance and residual variances "
            f"are ANNUAL decimals; weights are fractions summing to 1.0.",
            f"- Factors: {', '.join(factors)}",
            f"- Assets ({len(tickers)}): {', '.join(tickers[:12])}"
            + (" ..." if len(tickers) > 12 else ""),
            "",
            "```json",
            json.dumps(provenance, indent=2, default=str),
            "```",
            "",
        ]), encoding="utf-8")
    return out


def _series_from_csv(path, name: str) -> pd.Series:
    df = pd.read_csv(path)
    if "ticker" not in df.columns or name not in df.columns:
        raise SystemExit(f"[error] {path} must have columns: ticker,{name}")
    return df.set_index(df["ticker"].astype(str))[name]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="mode", required=True)

    p_syn = sub.add_parser("synthetic", help="seeded offline synthetic pack")
    p_syn.add_argument("--outdir", required=True)
    p_syn.add_argument("--n-assets", type=int, default=30)
    p_syn.add_argument("--n-factors", type=int, default=3)
    p_syn.add_argument("--seed", type=int, default=11)
    p_syn.add_argument("--collinear-corr", type=float, default=0.95)
    p_syn.add_argument("--max-weight", type=float, default=0.15,
                       help="single-stock cap in the generated config")

    p_yf = sub.add_parser("yfinance", help="real-data pack via Yahoo Finance")
    p_yf.add_argument("--outdir", required=True)
    p_yf.add_argument("--tickers", required=True,
                      help="comma-separated asset tickers")
    p_yf.add_argument("--factors", required=True,
                      help="comma-separated factor-proxy ETF tickers")
    p_yf.add_argument("--start", required=True)
    p_yf.add_argument("--end", required=True)
    p_yf.add_argument("--benchmark-csv", default=None,
                      help="optional CSV (ticker,weight) summing to 1; "
                           "default benchmark is equal weight")
    p_yf.add_argument("--mu-csv", default=None,
                      help="optional CSV (ticker,mu) overriding historical-mean mu")
    p_yf.add_argument("--max-weight", type=float, default=0.30)
    p_yf.add_argument("--default-tilt-bound", type=float, default=0.10)
    p_yf.add_argument("--no-auto-adjust", action="store_true",
                      help="download unadjusted closes (NOT recommended)")

    args = parser.parse_args(argv)

    try:
        if args.mode == "synthetic":
            pi, prov = synthetic_inputs(
                n_assets=args.n_assets, n_factors=args.n_factors,
                seed=args.seed, collinear_corr=args.collinear_corr)
            cfg = OptimizerConfig(max_weight=args.max_weight,
                                  default_tilt_bound=0.10)
            out = write_input_pack(pi, cfg, args.outdir, prov, "synthetic")
            print(f"[ok] synthetic input pack written to {out}")
            print(f"     seed={args.seed} (reruns reproduce identical inputs)")
            return 0

        # yfinance mode
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
        factors = [t.strip() for t in args.factors.split(",") if t.strip()]
        benchmark = (_series_from_csv(args.benchmark_csv, "weight")
                     if args.benchmark_csv else None)
        mu = _series_from_csv(args.mu_csv, "mu") if args.mu_csv else None
        pi, prov = fetch_yfinance_inputs(
            tickers, factors, args.start, args.end,
            benchmark_weights=benchmark, mu=mu,
            auto_adjust=not args.no_auto_adjust)
        cfg = OptimizerConfig(max_weight=args.max_weight,
                              default_tilt_bound=args.default_tilt_bound)
        out = write_input_pack(pi, cfg, args.outdir, prov, "real (yfinance)")
        print(f"[ok] yfinance input pack written to {out}")
        print(f"     window {args.start}..{args.end}, rows used "
              f"{prov.get('n_rows_used')}, retrieved {prov.get('retrieved_at')}")
        print("     NOTE: mu defaults to historical means (noisy estimates); "
              "override with --mu-csv for a views-based run.")
        return 0
    except (InputValidationError, RuntimeError, ValueError) as exc:
        print(f"[error] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
