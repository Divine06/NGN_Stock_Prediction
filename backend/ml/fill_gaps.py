"""Fill holes in data/raw/{TICKER}.csv with Investo (your existing historical_updater.py),
or rebuild a ticker whose history came from the wrong exchange.

Run from the project root, with INVESTO_API_KEY in your .env:

    python -m backend.ml.fill_gaps                 # report + fill gaps for all ten tickers
    python -m backend.ml.fill_gaps --dry-run       # only list the gaps
    python -m backend.ml.fill_gaps --rebuild SEPLAT --from 2014-04-09

--rebuild backs up raw/{TICKER}.csv and ngx.db first, then replaces that ticker's
history with a fresh download, so London-pence prices cannot survive the merge.
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from contextlib import closing
from datetime import date, timedelta

import pandas as pd

from . import config
from .history import discover_tickers, read_price_csv
from .validate import validate_ticker

MIN_GAP_DAYS = 6       # a normal week + holiday never leaves more than 5 calendar days between sessions
LOOKBACK_DAYS = 730


def find_gaps(ticker: str, today: date | None = None) -> list[tuple[str, str]]:
    """(start, end) date ranges to request: interior holes in the last two years plus the tail to today."""
    today = today or date.today()
    path = config.RAW_DIR / f"{ticker}.csv"
    if not path.exists():
        return [("2014-01-01", today.isoformat())]
    df = read_price_csv(path)
    idx = df.index[df.index >= pd.Timestamp(today) - pd.Timedelta(days=LOOKBACK_DAYS)]
    ranges = []
    for prev, cur in zip(idx[:-1], idx[1:]):
        if (cur - prev).days >= MIN_GAP_DAYS:
            ranges.append(((prev + timedelta(days=1)).date().isoformat(),
                           (cur - timedelta(days=1)).date().isoformat()))
    last = df.index[-1].date()
    if (today - last).days >= 1:
        ranges.append(((last + timedelta(days=1)).isoformat(), today.isoformat()))
    return ranges


def _updater():
    try:
        from backend.data import historical_updater as hu
    except RuntimeError as exc:  # raised by historical_updater when INVESTO_API_KEY is missing
        sys.exit(f"{exc}\nAdd INVESTO_API_KEY=... to the .env file in the project root.")
    return hu


def fill(ticker: str, ranges: list[tuple[str, str]]) -> int:
    hu = _updater()
    rows = []
    for start, end in ranges:
        rows.extend(hu.fetch_history(ticker, start, end))
    if not rows:
        print(f"{ticker}: Investo returned no rows for {len(ranges)} range(s)")
        return 0
    hu.update_raw_csv(ticker, rows)
    hu.update_database(ticker)
    return len(rows)


def rebuild(ticker: str, start: str) -> None:
    hu = _updater()
    stamp = date.today().strftime("%Y%m%d")
    raw = config.RAW_DIR / f"{ticker}.csv"
    backups = []
    if raw.exists():
        shutil.copy2(raw, raw.with_name(f"{ticker}.csv.bak-{stamp}"))
        backups.append(f"{ticker}.csv.bak-{stamp}")
    db = config.DATA_DIR / "ngx.db"
    if db.exists():
        shutil.copy2(db, db.with_name(f"ngx.db.bak-{stamp}"))
        backups.append(f"ngx.db.bak-{stamp}")
    rows = hu.fetch_history(ticker, start, date.today().isoformat())
    if not rows:
        sys.exit(f"{ticker}: Investo returned no data; the old file was left in place.")
    fresh = hu.normalise_history(rows)
    fresh.to_csv(raw, index=False)
    if db.exists():
        with closing(sqlite3.connect(db)) as con:
            cid = con.execute("SELECT company_id FROM Company WHERE ticker = ?", (ticker,)).fetchone()
            if cid:
                con.execute("DELETE FROM StockPrice WHERE company_id = ?", (cid[0],))
                con.commit()
        hu.update_database(ticker)
    print(f"{ticker}: rebuilt with {len(fresh)} rows ({fresh['timestamp'].min()} to {fresh['timestamp'].max()}). "
          f"Backups: {', '.join(backups) or 'none (no previous files)'}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fill NGX history gaps from Investo")
    ap.add_argument("--tickers")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rebuild", metavar="TICKER", help="replace one ticker's history completely")
    ap.add_argument("--from", dest="start", default="2014-01-01", help="start date for --rebuild")
    args = ap.parse_args(argv)

    if args.rebuild:
        rebuild(args.rebuild.upper(), args.start)
        print(validate_ticker(args.rebuild.upper())["checks"])
        return 0

    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else discover_tickers()
    for t in tickers:
        ranges = find_gaps(t)
        days = sum((date.fromisoformat(b) - date.fromisoformat(a)).days + 1 for a, b in ranges)
        print(f"{t}: {len(ranges)} range(s), {days} calendar days missing"
              + (f" -> {ranges[-3:]}" if ranges else ""))
        if ranges and not args.dry_run:
            n = fill(t, ranges)
            print(f"{t}: {n} rows received")
    if not args.dry_run:
        print("\nNow retrain: python -m backend.ml.train")
    return 0


if __name__ == "__main__":
    sys.exit(main())
