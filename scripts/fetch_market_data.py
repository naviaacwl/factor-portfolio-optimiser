#!/usr/bin/env python3
"""Download daily adjusted closes via yfinance and save wide CSVs for the backtest.

Outputs (under --outdir, default data/market):
    prices.csv             Date + one column per asset ticker (adjusted closes)
    factors.csv            Date + one column per factor-proxy ETF ticker
    benchmark_weights.csv  ticker,weight (equal weight over the asset universe)
    provenance.json        tickers, window, retrieval timestamp, auto_adjust, rows

Never fabricates data: fails loudly on empty/partial downloads. Rows where any
ticker is NaN (e.g. holidays mismatch) are dropped; the loader re-checks.

Usage:
  python scripts/fetch_market_data.py \
      --tickers AAPL,MSFT,GOOGL,AMZN,NVDA,JPM,JNJ,XOM,PG,KO,HD,MRK \
      --factors SPY,QQQ,IWM --start 2015-01-01 --end 2024-12-31
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import pandas as pd


def _download(tickers, start, end, auto_adjust: bool = True) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise SystemExit(
            "[error] yfinance required: pip install yfinance "
            "(or use synthetic mode)") from exc
    data = yf.download(tickers, start=start, end=end,
                       auto_adjust=auto_adjust, progress=False)
    if (isinstance(data.columns, pd.MultiIndex)
            and "Close" in data.columns.get_level_values(0)):
        prices = data["Close"]
    else:
        raise SystemExit("[error] unexpected yfinance payload; check tickers/network")
    if prices is None or prices.empty:
        raise SystemExit(
            "[error] yfinance returned no data; check tickers/window/network")
    missing = [t for t in tickers
               if t not in prices.columns or prices[t].isna().all()]
    if missing:
        raise SystemExit(f"[error] no usable data for tickers: {missing}")
    prices = prices[tickers].dropna(how="any").sort_index()
    prices.index.name = "Date"
    return prices


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tickers", required=True,
                    help="comma-separated asset tickers")
    ap.add_argument("--factors", required=True,
                    help="comma-separated factor-proxy ETF tickers")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--outdir", default="data/market")
    ap.add_argument("--no-auto-adjust", action="store_true",
                    help="download unadjusted closes (NOT recommended)")
    args = ap.parse_args(argv)

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    factors = [t.strip().upper() for t in args.factors.split(",") if t.strip()]
    aa = not args.no_auto_adjust

    prices = _download(tickers, args.start, args.end, aa)
    fac = _download(factors, args.start, args.end, aa)
    common = prices.index.intersection(fac.index)
    if len(common) < 300:
        raise SystemExit(f"[error] only {len(common)} common trading days; "
                         "widen the window or check the tickers")
    prices, fac = prices.loc[common], fac.loc[common]

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    prices.to_csv(out / "prices.csv")
    fac.to_csv(out / "factors.csv")
    pd.DataFrame({"ticker": tickers,
                  "weight": [1.0 / len(tickers)] * len(tickers)}
                 ).to_csv(out / "benchmark_weights.csv", index=False)
    prov = {
        "source": "Yahoo Finance via yfinance",
        "retrieved_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "tickers": tickers,
        "factors": factors,
        "start": args.start, "end": args.end,
        "auto_adjust": aa, "field": "Close",
        "n_rows": int(len(common)),
        "first_date": str(common.min().date()),
        "last_date": str(common.max().date()),
        "benchmark": "equal weight over the asset universe (static, drifting)",
    }
    (out / "provenance.json").write_text(json.dumps(prov, indent=2),
                                         encoding="utf-8")
    print(f"[ok] wrote {out}/prices.csv ({len(tickers)} assets), "
          f"factors.csv ({len(factors)} factors), benchmark_weights.csv, "
          f"provenance.json")
    print(f"     {len(common)} common trading days: "
          f"{prov['first_date']} .. {prov['last_date']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
