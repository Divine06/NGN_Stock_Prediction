"""Download daily price history from the Kobo Terminal API into data/history/.

Needs an API key on the Starter tier or above (the free Personal tier does not
include price history). Set it in the environment, never in frontend code:

    set KOBO_API_KEY=your_key          (Windows)   export KOBO_API_KEY=your_key   (macOS/Linux)
    python -m backend.ml.kobo_backfill --tickers DANGCEM,MTNN --from 2016-01-01

Kobo documents the endpoint (GET /api/ngxdata/prices/:symbol with from/to) but
not the exact history response fields, so the raw JSON is saved next to each
CSV. Open one and check the parsed columns before training.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date

import pandas as pd

from . import config

BASE_URL = "https://koboterminal.com/api/ngxdata/prices/{symbol}"
FIELD_MAP = {
    "date": ["date", "trade_date", "tradeDate", "day", "timestamp", "datetime"],
    "open": ["open", "open_price", "openPrice"],
    "high": ["high", "high_price", "highPrice"],
    "low": ["low", "low_price", "lowPrice"],
    "close": ["close", "close_price", "closePrice", "closing_price", "price", "current_price"],
    "volume": ["volume", "traded_volume", "volumeTraded"],
}


def _records(payload) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "prices", "history", "results", "items"):
            if key in payload:
                found = _records(payload[key])
                if found:
                    return found
    return []


def parse_history(payload) -> pd.DataFrame:
    recs = [r for r in _records(payload) if isinstance(r, dict)]
    if not recs:
        raise ValueError("no list of price records found in the response")
    rows = []
    for r in recs:
        row = {}
        for field, keys in FIELD_MAP.items():
            row[field] = next((r[k] for k in keys if k in r and r[k] is not None), None)
        rows.append(row)
    df = pd.DataFrame(rows)
    if df["date"].isna().all() or df["close"].isna().all():
        raise ValueError(f"could not find date/close fields; record keys are {sorted(recs[0])}")
    return df.dropna(subset=["date", "close"])


def fetch(symbol: str, start: str, end: str, api_key: str) -> dict | list:
    import requests
    resp = requests.get(BASE_URL.format(symbol=symbol), params={"from": start, "to": end},
                        headers={"X-API-Key": api_key, "Accept": "application/json"}, timeout=30)
    if resp.status_code == 429:
        raise RuntimeError("rate limited (429): wait a minute and retry")
    resp.raise_for_status()
    return resp.json()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Backfill NGX daily history from Kobo Terminal")
    ap.add_argument("--tickers", required=True)
    ap.add_argument("--from", dest="start", default="2016-01-01")
    ap.add_argument("--to", dest="end", default=date.today().isoformat())
    args = ap.parse_args(argv)

    key = os.getenv("KOBO_API_KEY")
    if not key:
        print("Set KOBO_API_KEY first (Starter tier or above is needed for price history).")
        return 1
    out_dir = config.HISTORY_DIR
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    for i, t in enumerate(s.strip().upper() for s in args.tickers.split(",")):
        if i:
            time.sleep(3.5)  # stay under 20 requests/minute
        try:
            payload = fetch(t, args.start, args.end, key)
            (out_dir / "raw" / f"{t}.json").write_text(json.dumps(payload, indent=1))
            df = parse_history(payload)
            df.to_csv(out_dir / f"{t}.csv", index=False)
            print(f"{t}: {len(df)} rows, {df['date'].min()} -> {df['date'].max()}")
        except Exception as exc:
            print(f"{t}: FAILED - {type(exc).__name__}: {exc}")
    print(f"Saved to {out_dir}. Check one file, then run: python -m backend.ml.train")
    return 0


if __name__ == "__main__":
    sys.exit(main())
