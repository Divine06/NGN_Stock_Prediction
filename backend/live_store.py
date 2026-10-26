"""SQLite storage for observations collected from the live Kobo feed.

This module does not read or write historical CSV files.
"""
from pathlib import Path
import sqlite3
from datetime import datetime, timezone

DB_PATH = Path("backend/data/ngx_live.db")


def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=20)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS price_observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                price REAL NOT NULL,
                change_pct REAL,
                volume REAL,
                source TEXT NOT NULL DEFAULT 'Kobo Terminal',
                UNIQUE(ticker, observed_at)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_observations_ticker_time
            ON price_observations(ticker, observed_at)
        """)
        conn.commit()


def save_observation(ticker, price, change_pct=None, volume=None,
                     source="Kobo Terminal", observed_at=None):
    if price is None or float(price) <= 0:
        return False
    observed_at = observed_at or datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute("""
            INSERT OR IGNORE INTO price_observations
            (ticker, observed_at, price, change_pct, volume, source)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            str(ticker).upper(), observed_at, float(price),
            None if change_pct is None else float(change_pct),
            None if volume is None else float(volume),
            source
        ))
        conn.commit()
    return True



def get_history(ticker, limit=5000, since=None):
    import csv

    ticker = str(ticker).upper().strip()
    rows_by_date = {}

    # 1. Load historical daily prices from the feature CSV.
    project_root = Path(__file__).resolve().parent.parent
    csv_path = (
    Path(__file__).resolve().parent
    / "data"
    / "features"
    / f"{str(ticker).upper()}_features.csv"
        )
    print("LIVE STORE MODULE:", Path(__file__).resolve())
    print("HISTORICAL CSV PATH:", csv_path.resolve())
    print("CSV EXISTS:", csv_path.exists())

    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)

            for row in reader:
                date = (row.get("timestamp") or "").strip()
                close = row.get("close")

                if not date or close is None:
                    continue

                try:
                    price = float(close)
                    if price <= 0:
                        continue
                except (ValueError, TypeError):
                    continue

                day = date[:10]

                if since and day < since[:10]:
                    continue

                rows_by_date[day] = {
                    "ticker": ticker,
                    "observed_at": date,
                    "close": price,
                    "change_pct": None,
                    "volume": _optional_float(row.get("volume")),
                    "source": "Historical CSV",
                }

    # 2. Merge SQLite observations. Prefer the newest live observation
    #    for a date that also exists in the historical CSV.
    query = """
        SELECT ticker, observed_at, price AS close,
               change_pct, volume, source
        FROM price_observations
        WHERE ticker = ?
    """
    params = [ticker]

    if since:
        query += " AND observed_at >= ?"
        params.append(since)

    query += " ORDER BY observed_at ASC"

    with _connect() as conn:
        live_rows = conn.execute(query, params).fetchall()

    for item in live_rows:
        row = dict(item)
        day = row["observed_at"][:10]

        if since and day < since[:10]:
            continue

        previous = rows_by_date.get(day)

        # Keep the latest observation when several live records
        # exist on the same day.
        if previous is None or previous.get("source") != "Kobo Terminal":
            rows_by_date[day] = row
        else:
            if row["observed_at"] > previous["observed_at"]:
                rows_by_date[day] = row

    # 3. Return chronological daily prices.
    history = sorted(
        rows_by_date.values(),
        key=lambda row: row["observed_at"],
    )

    if limit is not None:
        history = history[-max(0, int(limit)):]

    return history


def _optional_float(value):
    try:
        number = float(value)
        return number if number == number else None
    except (TypeError, ValueError):
        return None



def latest_observations():
    with _connect() as conn:
        rows = conn.execute("""
            SELECT p.ticker, p.observed_at, p.price AS close,
                   p.change_pct, p.volume, p.source
            FROM price_observations p
            JOIN (
                SELECT ticker, MAX(observed_at) AS latest
                FROM price_observations GROUP BY ticker
            ) latest
            ON latest.ticker = p.ticker AND latest.latest = p.observed_at
            ORDER BY p.ticker
        """).fetchall()
    return [dict(row) for row in rows]


def count_observations(ticker):
    with _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM price_observations WHERE ticker = ?",
            (str(ticker).upper(),)
        ).fetchone()
    return int(row["n"]) if row else 0
