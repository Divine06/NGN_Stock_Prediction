"""Builds one clean DAILY price series per ticker.

Sources, in order of priority when two give the same date:
  1. data/raw/{TICKER}.csv       NGX daily OHLCV (2014 dataset + Investo updates)
  2. data/history/{TICKER}.csv   optional extra backfill (e.g. kobo_backfill.py)
  3. Daily closes derived from the live feed (price_observations in ngx_live.db)

The live poller stores a snapshot every 20 minutes. The models work on daily
closes, so each trading day's LAST observation becomes that day's close.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

DATE_CANDIDATES = ["date", "trade_date", "tradedate", "datetime", "timestamp", "day", "pricedate"]
CLOSE_CANDIDATES = ["close", "close_price", "closeprice", "closing_price", "adj_close", "adjclose",
                    "price", "last", "last_price", "current_price"]
VOLUME_CANDIDATES = ["volume", "vol", "traded_volume", "volume_traded"]
HIGH_CANDIDATES = ["high", "high_price"]
OPEN_CANDIDATES = ["open", "open_price", "opening_price"]
LOW_CANDIDATES = ["low", "low_price"]


def _norm(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum() or ch == "_").replace("__", "_")


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lookup = {_norm(c).replace("_", ""): c for c in df.columns}
    for cand in candidates:
        key = cand.replace("_", "")
        if key in lookup:
            return lookup[key]
    return None


def _to_number(s: pd.Series) -> pd.Series:
    if s.dtype.kind in "if":
        return s.astype(float)
    cleaned = s.astype(str).str.replace(",", "", regex=False).str.replace("₦", "", regex=False).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


def _parse_dates(s: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(s, errors="coerce", format="ISO8601")
    if parsed.isna().mean() > 0.2:  # e.g. 21/10/2025 style dates
        parsed = pd.to_datetime(s, errors="coerce", dayfirst=True, format="mixed")
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_convert(config.LAGOS).dt.tz_localize(None)
    return parsed.dt.normalize()


def read_price_csv(path: Path) -> pd.DataFrame:
    """Read any daily price CSV and return columns open, high, low, close, volume indexed by date."""
    raw = pd.read_csv(path)
    date_col = _find_col(raw, DATE_CANDIDATES)
    if date_col is None and _norm(raw.columns[0]).startswith("unnamed"):
        date_col = raw.columns[0]  # date saved as an unnamed index column
    close_col = _find_col(raw, CLOSE_CANDIDATES)
    if date_col is None or close_col is None:
        raise ValueError(f"{path.name}: could not find a date and close column in {list(raw.columns)}")

    out = pd.DataFrame({"date": _parse_dates(raw[date_col]), "close": _to_number(raw[close_col])})
    for name, cands in (("open", OPEN_CANDIDATES), ("high", HIGH_CANDIDATES), ("low", LOW_CANDIDATES),
                        ("volume", VOLUME_CANDIDATES)):
        col = _find_col(raw, cands)
        out[name] = _to_number(raw[col]) if col else np.nan

    out = out.dropna(subset=["date", "close"])
    out = out[out["close"] > 0]
    out = out.drop_duplicates("date", keep="last").set_index("date").sort_index()
    return out[out.index.dayofweek < 5]


def daily_closes_from_db(ticker: str, db_path: Path | None = None, include_today: bool = False,
                         now: datetime | None = None) -> pd.DataFrame:
    """Collapse 20-minute live observations into one close per trading day."""
    db_path = Path(db_path or config.DB_PATH)
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"], index=pd.DatetimeIndex([], name="date"))
    if not db_path.exists():
        return empty
    try:
        with closing(sqlite3.connect(db_path)) as con:
            obs = pd.read_sql_query(
                f"SELECT observed_at, price, volume FROM {config.OBS_TABLE} "
                "WHERE UPPER(ticker) = ? ORDER BY observed_at",
                con, params=(ticker.upper(),))
    except Exception:
        return empty
    if obs.empty:
        return empty

    ts = pd.to_datetime(obs["observed_at"], errors="coerce", utc=True, format="ISO8601")
    obs = obs.assign(ts=ts.dt.tz_convert(config.LAGOS), price=_to_number(obs["price"]),
                     volume=_to_number(obs["volume"]))
    obs = obs.dropna(subset=["ts", "price"])
    obs = obs[obs["price"] > 0]
    if obs.empty:
        return empty
    obs["date"] = obs["ts"].dt.tz_localize(None).dt.normalize()

    daily = obs.groupby("date").agg(open=("price", "first"), high=("price", "max"), low=("price", "min"),
                                    close=("price", "last"), volume=("volume", "max"))
    daily = daily[daily.index.dayofweek < 5]

    now = now or datetime.now(config.LAGOS)
    today = pd.Timestamp(now.date())
    ready = time.fromisoformat(config.DAILY_CLOSE_READY)
    if not include_today and today in daily.index and now.time() < ready:
        daily = daily.drop(index=today)  # today's session is not finished yet
    daily.index.name = "date"
    return daily


def discover_tickers() -> list[str]:
    """The project universe (config.UNIVERSE) unless NGX_TICKERS overrides it."""
    return config.configured_tickers() or list(config.UNIVERSE)


def _case_insensitive(directory: Path, filename: str) -> Path | None:
    exact = directory / filename
    if exact.exists():
        return exact
    if directory.exists():
        for p in directory.iterdir():
            if p.name.lower() == filename.lower():
                return p
    return None


def load_history(ticker: str, include_live: bool = True, include_today: bool = False,
                 now: datetime | None = None) -> pd.DataFrame:
    """Merged daily series with a 'source' column and a 'segment' id (see add_segments)."""
    ticker = ticker.upper()
    frames = []
    for directory, filename, label in (
        (config.RAW_DIR, f"{ticker}.csv", "ngx_history"),
        (config.HISTORY_DIR, f"{ticker}.csv", "backfill"),
    ):
        path = _case_insensitive(directory, filename)
        if path is not None:
            frames.append(read_price_csv(path).assign(source=label))
    if include_live:
        live = daily_closes_from_db(ticker, include_today=include_today, now=now)
        if not live.empty:
            frames.append(live.assign(source="live_feed"))
    if not frames:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "source", "segment"])

    merged = pd.concat(frames)
    merged = merged[~merged.index.duplicated(keep="first")].sort_index()
    return add_segments(merged)


def add_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Split the series wherever it is not continuous.

    A new segment starts after a long gap (e.g. CSV ends in 2025, live data starts
    in 2026) or after a jump larger than the NGX daily band (bonus issue, split or
    bad data). Returns and indicators are never computed across a boundary.
    """
    df = df.copy()
    gap = df.index.to_series().diff().dt.days
    jump = df["close"].pct_change().abs()
    breaks = (gap > config.MAX_GAP_DAYS) | (jump > config.DISCONTINUITY_MOVE)
    df["segment"] = breaks.fillna(False).astype(int).cumsum().to_numpy()
    return df


def coverage_report(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"rows": 0}
    gaps = df.index.to_series().diff().dt.days.to_numpy()
    big = np.flatnonzero(gaps > config.MAX_GAP_DAYS)
    jumps = df["close"].pct_change().abs()
    last_seg = df[df["segment"] == df["segment"].iloc[-1]]
    return {
        "rows": int(len(df)),
        "first_date": df.index[0].date().isoformat(),
        "last_date": df.index[-1].date().isoformat(),
        "sources": {k: int(v) for k, v in df["source"].value_counts().items()},
        "gaps_over_limit": [
            {"from": df.index[i - 1].date().isoformat(), "to": df.index[i].date().isoformat(),
             "days": int(gaps[i])}
            for i in big
        ],
        "large_jumps": int((jumps > config.DISCONTINUITY_MOVE).sum()),
        "latest_segment_rows": int(len(last_seg)),
        "latest_segment_start": last_seg.index[0].date().isoformat(),
    }
