"""Synthetic NGX-like data for tests only (never used for reported results).

Mimics NGX features that matter for the pipeline: many no-change days,
a +/-10% daily band, a strong late rally on some tickers (the condition that
broke the old price-level models), and a 20-minute live feed in SQLite.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

LAGOS_OFFSET = pd.Timedelta(hours=1)

TICKERS = {
    # name: (drift, rally_drift, seed)
    "SYNRALLY1": (0.0002, 0.0040, 1),
    "SYNRALLY2": (0.0001, 0.0035, 2),
    "SYNRALLY3": (0.0003, 0.0030, 3),
    "SYNUP1": (0.0006, None, 4),
    "SYNUP2": (0.0004, None, 5),
    "SYNDOWN1": (-0.0006, None, 6),
    "SYNDOWN2": (-0.0004, None, 7),
    "SYNFLAT1": (0.0, None, 8),
    "SYNFLAT2": (0.0, None, 9),
    "SYNMIX": (0.0003, -0.002, 10),
}


def make_returns(n: int, drift: float, rally_drift: float | None, seed: int, flat_prob: float = 0.35,
                 sigma: float = 0.018, phi: float = 0.15, cap: float = 0.095) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros(n)
    prev = 0.0
    rally_from = int(n * 0.6)
    for t in range(n):
        mu = rally_drift if (rally_drift is not None and t >= rally_from) else drift
        val = mu + phi * prev + rng.normal(0, sigma)
        if rng.random() < flat_prob:
            val = 0.0
        val = float(np.clip(val, -cap, cap))
        out[t] = val
        prev = val
    return out


def _write_csv(path: Path, dates: pd.DatetimeIndex, close: np.ndarray, rng) -> None:
    """Same columns as backend/data/raw/{TICKER}.csv in the project."""
    close = np.round(close, 2)
    df = pd.DataFrame({
        "timestamp": dates.strftime("%Y-%m-%d"),
        "open": close, "high": np.round(close * (1 + rng.uniform(0, 0.01, len(close))), 2),
        "low": np.round(close * (1 - rng.uniform(0, 0.01, len(close))), 2),
        "close": close, "volume": rng.integers(50_000, 5_000_000, len(close)),
    })
    df.to_csv(path, index=False)


def _write_live(con, ticker: str, days: pd.DatetimeIndex, closes: np.ndarray, rng) -> None:
    rows = []
    for day, close in zip(days, closes):
        times = [day + pd.Timedelta(hours=10) + pd.Timedelta(minutes=20 * k) for k in range(14)]
        path = np.round(close * (1 + rng.normal(0, 0.003, len(times))), 2)  # intraday wobble
        path[-1] = round(close, 2)                                           # last snapshot = close
        vol = np.cumsum(rng.integers(1_000, 50_000, len(times)))
        for ts, price, v in zip(times, path, vol):
            utc = (ts - LAGOS_OFFSET).strftime("%Y-%m-%d %H:%M:%S")  # stored as naive UTC
            rows.append((ticker, utc, float(price), None, float(v), "Kobo Terminal"))
    con.executemany("INSERT INTO price_observations (ticker, observed_at, price, change_pct, volume, source) "
                    "VALUES (?, ?, ?, ?, ?, ?)", rows)


def build_dataset(root: Path, today: pd.Timestamp | None = None, n_days: int = 2000) -> dict:
    data = root / "data"
    raw = data / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    db = data / "ngx_live.db"
    today = today or pd.Timestamp.now(tz="UTC").tz_convert("Africa/Lagos").tz_localize(None).normalize()
    live_days = pd.bdate_range(end=today - BDay(1), periods=20)
    csv_days = pd.bdate_range(end=live_days[0] - BDay(1), periods=n_days)

    con = sqlite3.connect(db)
    con.execute("CREATE TABLE IF NOT EXISTS price_observations (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "ticker TEXT, observed_at TEXT, price REAL, change_pct REAL, volume REAL, source TEXT)")
    for name, (drift, rally, seed) in TICKERS.items():
        rng = np.random.default_rng(seed + 100)
        r = make_returns(n_days + len(live_days), drift, rally, seed)
        price = 100 * np.exp(np.cumsum(r))
        _write_csv(raw / f"{name}.csv", csv_days, price[:n_days], rng)
        _write_live(con, name, live_days, price[n_days:], rng)

    # too little history
    rng = np.random.default_rng(50)
    r = make_returns(140, 0.0, None, 50)
    _write_csv(raw / "SYNSHORT.csv", pd.bdate_range(end=live_days[0] - BDay(1), periods=120),
               (50 * np.exp(np.cumsum(r)))[:120], rng)
    _write_live(con, "SYNSHORT", live_days, (50 * np.exp(np.cumsum(r)))[120:], rng)

    # CSV history stops long before the live feed starts -> gap
    rng = np.random.default_rng(60)
    r = make_returns(1500, 0.0003, None, 60)
    gap_end = live_days[0] - BDay(200)
    _write_csv(raw / "SYNGAP.csv", pd.bdate_range(end=gap_end, periods=1480),
               (80 * np.exp(np.cumsum(r)))[:1480], rng)
    _write_live(con, "SYNGAP", live_days, (80 * np.exp(np.cumsum(r)))[1480:], rng)

    # history from another exchange: same shape, 50x smaller numbers, wider daily moves
    rng = np.random.default_rng(70)
    r = make_returns(n_days + len(live_days), 0.0004, None, 70, flat_prob=0.05, sigma=0.04, cap=0.3)
    price = 5000 * np.exp(np.cumsum(r))
    _write_csv(raw / "SYNLSE.csv", csv_days, price[:n_days] / 50, rng)
    _write_live(con, "SYNLSE", live_days, price[n_days:], rng)

    con.commit()
    con.close()
    return {"data": data, "db": db, "live_days": live_days, "csv_days": csv_days, "today": today}
