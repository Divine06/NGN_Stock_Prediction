"""Predictions table in the same SQLite database as the live observations.

Each run stores one row per ticker per as_of_date (re-running the same day
overwrites it). When the next day's close arrives, the realised return is
filled in, so the system keeps a live, out-of-sample track record.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker           TEXT NOT NULL,
    as_of_date       TEXT NOT NULL,      -- date of the last daily close used
    target_date      TEXT,               -- next weekday (holidays not excluded)
    last_close       REAL,
    ensemble_return  REAL,               -- predicted next-day log return
    ensemble_price   REAL,
    model_outputs    TEXT,               -- JSON {model: {return, price, path}}
    forecast_path    TEXT,               -- JSON [{h, date, price, low, high, ...}] for days 1..HORIZON
    best_model       TEXT,
    confidence       TEXT,
    signal           TEXT NOT NULL,      -- BUY / HOLD / SELL / INSUFFICIENT_DATA
    score            REAL,
    sentiment_score  REAL,
    n_headlines      INTEGER,
    evidence         TEXT,               -- JSON list of reasons
    created_at       TEXT NOT NULL,
    actual_close     REAL,               -- filled in when the next close is known
    actual_return    REAL,
    direction_hit    INTEGER,            -- 1/0, NULL if either move was zero
    UNIQUE(ticker, as_of_date)
);
CREATE INDEX IF NOT EXISTS idx_predictions_ticker_date ON predictions(ticker, as_of_date);
"""


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = Path(db_path or config.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    cols = {r[1] for r in con.execute("PRAGMA table_info(predictions)")}
    if "forecast_path" not in cols:  # databases created before the 5-day forecast
        con.execute("ALTER TABLE predictions ADD COLUMN forecast_path TEXT")
        con.commit()
    return con


@contextmanager
def _db(db_path: Path | None = None):
    con = _connect(db_path)
    try:
        with con:  # commit on success, roll back on error
            yield con
    finally:
        con.close()


def save_prediction(rec: dict, db_path: Path | None = None) -> None:
    if not rec.get("as_of_date"):
        return  # nothing to anchor the row to (no data at all)
    sent = rec.get("sentiment") or {}
    row = {
        "ticker": rec["ticker"],
        "as_of_date": rec["as_of_date"],
        "target_date": rec.get("target_date"),
        "last_close": rec.get("last_close"),
        "ensemble_return": rec.get("ensemble_return"),
        "ensemble_price": rec.get("ensemble_price"),
        "model_outputs": json.dumps(rec.get("models", {})),
        "forecast_path": json.dumps(rec.get("forecast_path") or []),
        "best_model": rec.get("best_model"),
        "confidence": rec.get("confidence"),
        "signal": rec["signal"],
        "score": rec.get("score"),
        "sentiment_score": sent.get("score"),
        "n_headlines": sent.get("n_headlines"),
        "evidence": json.dumps(rec.get("evidence", [])),
        "created_at": datetime.now(config.LAGOS).isoformat(timespec="seconds"),
    }
    cols = ", ".join(row)
    marks = ", ".join(f":{k}" for k in row)
    updates = ", ".join(f"{k}=excluded.{k}" for k in row if k not in ("ticker", "as_of_date"))
    with _db(db_path) as con:
        con.execute(f"INSERT INTO predictions ({cols}) VALUES ({marks}) "
                    f"ON CONFLICT(ticker, as_of_date) DO UPDATE SET {updates}", row)


def _row_to_dict(r: sqlite3.Row) -> dict:
    d = dict(r)
    for key in ("model_outputs", "evidence", "forecast_path"):
        if d.get(key):
            d[key] = json.loads(d[key])
    return d


def latest_predictions(db_path: Path | None = None) -> list[dict]:
    with _db(db_path) as con:
        rows = con.execute(
            "SELECT p.* FROM predictions p JOIN (SELECT ticker, MAX(as_of_date) AS d FROM predictions "
            "GROUP BY ticker) last ON p.ticker = last.ticker AND p.as_of_date = last.d ORDER BY p.ticker"
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def ticker_predictions(ticker: str, limit: int = 30, db_path: Path | None = None) -> list[dict]:
    with _db(db_path) as con:
        rows = con.execute("SELECT * FROM predictions WHERE ticker = ? ORDER BY as_of_date DESC LIMIT ?",
                           (ticker.upper(), limit)).fetchall()
    return [_row_to_dict(r) for r in rows]


def update_realised(ticker: str, daily_close: "dict[str, float]", db_path: Path | None = None) -> int:
    """daily_close maps 'YYYY-MM-DD' -> close. Fills outcomes for past predictions."""
    if not daily_close:
        return 0
    dates = sorted(daily_close)
    updated = 0
    with _db(db_path) as con:
        pending = con.execute("SELECT id, as_of_date, last_close, ensemble_return FROM predictions "
                              "WHERE ticker = ? AND actual_close IS NULL", (ticker.upper(),)).fetchall()
        for p in pending:
            later = [d for d in dates if d > p["as_of_date"]]
            if not later or not p["last_close"]:
                continue
            actual = float(daily_close[later[0]])
            ret = float(np.log(actual / p["last_close"]))
            pred = p["ensemble_return"]
            hit = None
            if pred is not None and abs(ret) > 1e-9 and abs(pred) > 1e-12:
                hit = int(np.sign(ret) == np.sign(pred))
            con.execute("UPDATE predictions SET actual_close=?, actual_return=?, direction_hit=? WHERE id=?",
                        (actual, ret, hit, p["id"]))
            updated += 1
    return updated


def live_accuracy(db_path: Path | None = None) -> dict:
    with _db(db_path) as con:
        rows = con.execute(
            "SELECT ticker, COUNT(*) AS n, AVG(direction_hit) AS dir_acc, "
            "AVG(ABS(ensemble_return - actual_return)) AS mae, "
            "SUM(CASE WHEN signal='BUY' AND actual_return > 0 THEN 1 ELSE 0 END) AS buy_hits, "
            "SUM(CASE WHEN signal='BUY' THEN 1 ELSE 0 END) AS buys, "
            "SUM(CASE WHEN signal='SELL' AND actual_return < 0 THEN 1 ELSE 0 END) AS sell_hits, "
            "SUM(CASE WHEN signal='SELL' THEN 1 ELSE 0 END) AS sells "
            "FROM predictions WHERE actual_return IS NOT NULL GROUP BY ticker ORDER BY ticker").fetchall()
    per = [dict(r) for r in rows]
    n = sum(r["n"] for r in per)
    return {
        "evaluated_predictions": n,
        "per_ticker": per,
        "note": "Live out-of-sample record: each prediction is scored against the next close collected.",
    }
