#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Download SPY daily OHLCV to data/SPY.csv (requires yfinance)."""

from pathlib import Path

import pandas as pd

try:
    import yfinance as yf
except ImportError as e:
    raise SystemExit("Install yfinance: pip install yfinance") from e


def main():
    root = Path(__file__).resolve().parent.parent
    out = root / "data" / "SPY.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    #to fix the slice issue I used `Ticker.history(period="max")`
    # `download()` without an explicit window can return only a short recent slice in some
    # environments; `Ticker.history(period="max")` reliably pulls full daily history.
    ticker = yf.Ticker("SPY")
    df = ticker.history(period="max", auto_adjust=False, interval="1d")
    if df.empty:
        raise SystemExit("No data returned for SPY.")

    df = df.rename_axis("Date").reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    want = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in want if c not in df.columns]
    if missing:
        raise SystemExit(f"Unexpected columns; missing {missing}. Got: {list(df.columns)}")

    df = df[want].copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
