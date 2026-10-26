"""Data-quality checks run before training and before every prediction.

    python -m backend.ml.validate        (from the project root)

Checks per ticker:
  1. scale     - last historical close vs the latest live Kobo price. More than 5x apart
                 means the history is in a different currency/exchange (e.g. London pence).
  2. band      - NGX limits a stock to +/-10% per session. Many larger day-to-day moves in
                 the history mean it is not NGX data (or is corrupted).
  3. gaps      - holes longer than MAX_GAP_DAYS in the last two years.
  4. freshness - how old the latest daily close is.

Status: OK, WARNING (usable, but read the detail) or INVALID (excluded from
training and prediction until the history is replaced).
"""
from __future__ import annotations

import sqlite3
import sys
from contextlib import closing
from datetime import datetime

import numpy as np
import pandas as pd

from . import config
from .history import discover_tickers, load_history

RANK = {"OK": 0, "WARNING": 1, "INVALID": 2}


def latest_live_price(ticker: str) -> tuple[float | None, str | None]:
    if not config.DB_PATH.exists():
        return None, None
    try:
        with closing(sqlite3.connect(config.DB_PATH)) as con:
            row = con.execute(f"SELECT price, observed_at FROM {config.OBS_TABLE} WHERE UPPER(ticker)=? "
                              "ORDER BY observed_at DESC LIMIT 1", (ticker.upper(),)).fetchone()
    except sqlite3.Error:
        return None, None
    return (float(row[0]), str(row[1])) if row and row[0] else (None, None)


def validate_ticker(ticker: str, live_price: float | None = None, now: datetime | None = None) -> dict:
    ticker = ticker.upper()
    now = now or datetime.now(config.LAGOS)
    checks = []
    hist = load_history(ticker, include_live=False)
    if hist.empty:
        return {"ticker": ticker, "status": "INVALID",
                "checks": [{"check": "history", "status": "INVALID",
                            "detail": f"No historical CSV found in {config.RAW_DIR}"}]}

    close = hist["close"].astype(float)
    last_date, last_close = hist.index[-1], float(close.iloc[-1])

    # 1. scale vs live price
    live_at = None
    if live_price is None:
        live_price, live_at = latest_live_price(ticker)
    if live_price:
        ratio = live_price / last_close
        if ratio > config.SCALE_MISMATCH_RATIO or ratio < 1 / config.SCALE_MISMATCH_RATIO:
            checks.append({"check": "scale", "status": "INVALID", "detail": (
                f"Live NGX price NGN {live_price:,.2f} is {ratio:.1f}x the last historical close "
                f"{last_close:,.2f} ({last_date.date()}). The history is probably from another exchange or "
                "currency (e.g. London pence). Replace it with the Lagos (NGN) history: "
                f"python -m backend.ml.import_csv {ticker} <file.csv> --replace")})
        else:
            checks.append({"check": "scale", "status": "OK",
                           "detail": f"live/historical price ratio {ratio:.2f}"})
    else:
        checks.append({"check": "scale", "status": "OK",
                       "detail": "no live price stored yet; scale not checked"})

    # 2. NGX daily price band on consecutive trading days
    gap = hist.index.to_series().diff().dt.days
    consecutive = (gap <= 4).to_numpy()
    moves = close.pct_change().abs().to_numpy()
    breaches = int(np.sum((moves > config.DISCONTINUITY_MOVE) & consecutive))
    share = breaches / max(1, int(consecutive.sum()))
    status = "OK"
    if share > config.BAND_BREACH_INVALID and breaches >= 10:
        status = "INVALID"
    elif share > config.BAND_BREACH_WARN:
        status = "WARNING"
    checks.append({"check": "band", "status": status, "detail": (
        f"{breaches} day-to-day moves beyond the NGX +/-10% band ({share * 100:.2f}% of sessions)"
        + ("; NGX data should have almost none, so this is likely not NGX prices" if status != "OK" else ""))})

    # 3. gaps in the last two years
    recent = hist[hist.index >= last_date - pd.Timedelta(days=730)]
    rgap = recent.index.to_series().diff().dt.days
    holes = [(recent.index[i - 1].date().isoformat(), recent.index[i].date().isoformat(), int(rgap.iloc[i]))
             for i in range(1, len(recent)) if rgap.iloc[i] > config.MAX_GAP_DAYS]
    checks.append({"check": "gaps", "status": "WARNING" if holes else "OK",
                   "detail": (f"{len(holes)} gap(s) over {config.MAX_GAP_DAYS} days in the last two years: "
                              + ", ".join(f"{a} to {b} ({d}d)" for a, b, d in holes[-6:])
                              + ". Fill them: python -m backend.ml.import_csv (free download, see guide)")
                   if holes else "no long gaps",
                   "gaps": holes})

    # 4. freshness of the historical file (live closes extend it, see coverage)
    age = (pd.Timestamp(now.date()) - last_date).days
    checks.append({"check": "freshness", "status": "WARNING" if age > 30 else "OK",
                   "detail": f"historical CSV ends {last_date.date()} ({age} days ago)"})

    overall = max((c["status"] for c in checks), key=RANK.get)
    return {"ticker": ticker, "status": overall, "rows": int(len(hist)),
            "first_date": hist.index[0].date().isoformat(), "last_date": last_date.date().isoformat(),
            "last_close": last_close, "live_price": live_price, "live_observed_at": live_at, "checks": checks}


def validate_all(tickers: list[str] | None = None) -> list[dict]:
    return [validate_ticker(t) for t in (tickers or discover_tickers())]


def main(argv=None) -> int:
    reports = validate_all()
    for r in reports:
        print(f"\n{r['ticker']:<11} {r['status']}")
        for c in r["checks"]:
            if c["status"] != "OK" or c["check"] in ("scale", "freshness"):
                print(f"   [{c['status']:<7}] {c['check']}: {c['detail']}")
    bad = [r["ticker"] for r in reports if r["status"] == "INVALID"]
    print(f"\nINVALID (excluded until fixed): {', '.join(bad) or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
