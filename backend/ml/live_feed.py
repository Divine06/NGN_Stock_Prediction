"""Free live feed: Kobo Terminal (Personal tier) and/or NGN Market (free plan). One request
returns all NGX equities; the ten project stocks are saved as observations in ngx_live.db
(same table as live_store.py).

Polling runs every 20 minutes but only between 09:55 and 15:05 WAT on weekdays
(NGX trades 10:00-14:30 and Kobo is >=30 min delayed, so the 15:05 poll catches
the close). That is ~16 requests a day, well inside the free tier's 100/day,
instead of 72+ for round-the-clock polling.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
from contextlib import closing
from datetime import datetime, time, timezone

from . import config

log = logging.getLogger("ml.live_feed")

STATE = {"stocks": [], "last_refresh": None, "error": None, "fetched_at": None, "cached": None, "sources": []}

SCHEMA = """
CREATE TABLE IF NOT EXISTS price_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    price REAL NOT NULL,
    change_pct REAL,
    volume REAL,
    source TEXT NOT NULL DEFAULT 'Kobo Terminal',
    UNIQUE(ticker, observed_at)
);
CREATE INDEX IF NOT EXISTS idx_observations_ticker_time ON price_observations(ticker, observed_at);
"""


def _connect() -> sqlite3.Connection:
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(config.DB_PATH, timeout=20)
    con.executescript(SCHEMA)
    return con


def init_db() -> None:
    with closing(_connect()) as con:
        con.commit()


def _num(v):
    try:
        x = float(v)
        return x if x == x and x not in (float("inf"), float("-inf")) else None
    except (TypeError, ValueError):
        return None


def save_observations(stocks: list[dict], observed_at: str) -> int:
    rows = [(s["ticker"], observed_at, s["price"], s.get("change_pct"), s.get("volume"),
             s.get("source", "Kobo Terminal")) for s in stocks if s.get("price")]
    with closing(_connect()) as con:
        con.executemany("INSERT OR IGNORE INTO price_observations "
                        "(ticker, observed_at, price, change_pct, volume, source) VALUES (?,?,?,?,?,?)", rows)
        con.commit()
    return len(rows)


def _market_data():
    try:
        from backend import market_data  # project layout: run from the project root
    except ImportError:  # pragma: no cover
        import market_data
    return market_data


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:  # pragma: no cover
        pass


def _http_get(url: str, params: dict, headers: dict) -> dict | list:
    import requests
    resp = requests.get(url, params=params, headers=headers, timeout=15)
    if resp.status_code == 429:
        raise RuntimeError("rate limited (429)")
    resp.raise_for_status()
    return resp.json()


class SourceUnavailable(RuntimeError):
    pass


def fetch_kobo() -> dict:
    """Kobo Terminal free Personal tier: one call returns every NGX equity (>=30 min delayed)."""
    result = _market_data().fetch_all_stocks(force_refresh=True)
    rows = {}
    for s in result.get("stocks", []):
        sym = str(s.get("symbol") or s.get("ticker") or "").upper()
        if sym:
            rows[sym] = {"price": s.get("current_price", s.get("price")), "change_pct": s.get("change_percent"),
                         "volume": s.get("volume")}
    return {"rows": rows, "fetched_at": result.get("fetched_at"), "cached": bool(result.get("cached"))}


def fetch_ngnmarket() -> dict:
    """NGN Market free plan: GET /v1/companies returns price, previous close, day high/low and
    volume for every NGX company, refreshed every 20 minutes in trading hours (3,000 calls/month)."""
    _load_env()
    key = os.getenv("NGN_MARKET_API_KEY")
    if not key:
        raise SourceUnavailable("NGN_MARKET_API_KEY not set")
    rows, page, pages = {}, 1, 1
    while page <= min(pages, 3):
        payload = _http_get(f"{config.NGN_MARKET_BASE}/companies",
                            {"limit": 200, "page": page, "sort": "symbol", "order": "asc"},
                            {"Authorization": f"Bearer {key}", "Accept": "application/json"})
        items = payload if isinstance(payload, list) else next(
            (payload[k] for k in ("data", "companies", "results", "items") if isinstance(payload.get(k), list)), [])
        for c in items:
            sym = str(c.get("symbol") or "").upper()
            if sym:
                rows[sym] = {"price": c.get("price"), "change_pct": c.get("price_change_percent"),
                             "volume": c.get("volume"), "updated": c.get("last_updated")}
        pages = int(((payload.get("pagination") or {}).get("total_pages") or 1)) if isinstance(payload, dict) else 1
        page += 1
    return {"rows": rows, "fetched_at": datetime.now(timezone.utc).isoformat(), "cached": False}


FETCHERS = {"kobo": (fetch_kobo, "Kobo Terminal"), "ngnmarket": (fetch_ngnmarket, "NGN Market")}


def normalise(raw: dict, ticker: str, fetched_at: str | None, cached: bool = False,
              source: str = "Kobo Terminal") -> dict | None:
    price = _num(raw.get("current_price", raw.get("price")))
    if not price or price <= 0:
        return None
    chg = _num(raw.get("change_percent", raw.get("change_pct")))
    return {
        "ticker": ticker,
        "name": config.company_name(ticker),
        "sector": config.SECTORS.get(ticker, raw.get("sector")),
        "price": price,
        "current_price": price,
        "change_pct": chg,
        "change_percent": chg,
        "volume": _num(raw.get("volume")),
        "source": source,
        "data_status": "Delayed Data",
        "feed_timestamp": raw.get("updated") or fetched_at,
        "cached": bool(cached),
        "fallback": False,
    }


def collect_once(force: bool = True) -> dict:
    """Fetch current prices from the free sources in LIVE_SOURCES order and store them.

    The first source is used for every ticker it returns; a later source is only called
    when an earlier one fails or misses a ticker, so the daily quotas are not wasted.
    """
    universe = config.configured_tickers() or list(config.UNIVERSE)
    found, errors, used = {}, [], []
    fetched_at, cached_any = None, False
    for name in config.LIVE_SOURCES:
        if len(found) == len(universe):
            break
        fetch, label = FETCHERS[name]
        try:
            res = fetch()
        except SourceUnavailable as exc:
            errors.append(f"{label}: {exc}")
            continue
        except Exception as exc:
            errors.append(f"{label}: {exc}")
            continue
        used.append(label)
        fetched_at = fetched_at or res["fetched_at"]
        for t in universe:
            if t not in found and t in res["rows"]:
                rec = normalise(res["rows"][t], t, res["fetched_at"], res["cached"], label)
                if rec:
                    found[t] = rec
                    cached_any = cached_any or res["cached"]
    if not found:
        raise RuntimeError("no live source returned prices: " + "; ".join(errors))
    stocks = [found[t] for t in universe if t in found]
    now = datetime.now(timezone.utc).isoformat()
    save_observations([s for s in stocks if not s["cached"]], now)
    STATE.update(stocks=stocks, last_refresh=now, error="; ".join(errors) or None, fetched_at=fetched_at,
                 cached=cached_any, sources=used)
    return {"stocks": stocks, "last_refresh": now, "fetched_at": fetched_at, "cached": cached_any,
            "sources": used, "errors": errors}


def snapshot_from_db() -> list[dict]:
    """Latest stored observation per ticker, so the dashboard has prices without spending quota."""
    if not config.DB_PATH.exists():
        return []
    with closing(_connect()) as con:
        rows = con.execute(
            "SELECT p.ticker, p.observed_at, p.price, p.change_pct, p.volume, p.source FROM price_observations p "
            "JOIN (SELECT ticker, MAX(observed_at) m FROM price_observations GROUP BY ticker) l "
            "ON l.ticker = p.ticker AND l.m = p.observed_at").fetchall()
    out = []
    universe = config.configured_tickers() or list(config.UNIVERSE)
    for t, at, price, chg, vol, src in rows:
        if t in universe:
            rec = normalise({"current_price": price, "change_percent": chg, "volume": vol}, t, at, cached=True,
                            source=src or "stored observation")
            if rec:
                rec["data_status"] = "Last stored observation"
                out.append(rec)
    return sorted(out, key=lambda r: universe.index(r["ticker"]))


def latest_observation_age_minutes(now: datetime | None = None) -> float | None:
    if not config.DB_PATH.exists():
        return None
    with closing(_connect()) as con:
        row = con.execute("SELECT MAX(observed_at) FROM price_observations").fetchone()
    if not row or not row[0]:
        return None
    try:
        ts = datetime.fromisoformat(str(row[0]).replace("Z", "+00:00"))
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    return (now - ts).total_seconds() / 60


def intraday(ticker: str, day: str | None = None) -> list[dict]:
    """20-minute observations for one Lagos calendar day (default: the latest day with data)."""
    if not config.DB_PATH.exists():
        return []
    with closing(_connect()) as con:
        rows = con.execute("SELECT observed_at, price, change_pct, volume, source FROM price_observations "
                           "WHERE UPPER(ticker)=? ORDER BY observed_at", (ticker.upper(),)).fetchall()
    out = []
    for at, price, chg, vol, src in rows:
        try:
            ts = datetime.fromisoformat(str(at).replace("Z", "+00:00"))
        except ValueError:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        local = ts.astimezone(config.LAGOS)
        out.append({"timestamp": local.isoformat(), "observed_at": str(at), "date": local.date().isoformat(),
                    "close": price, "price": price, "change_pct": chg, "volume": vol, "source": src})
    if not out:
        return []
    day = day or out[-1]["date"]
    return [r for r in out if r["date"] == day]


def market_status(now: datetime | None = None) -> dict:
    now = now or datetime.now(config.LAGOS)
    open_t, close_t = (time.fromisoformat(x) for x in config.MARKET_HOURS)
    weekday = now.weekday() < 5
    t = now.time()
    if weekday and open_t <= t < close_t:
        status, msg = "open", "NGX market is open"
    elif weekday and time(9, 30) <= t < open_t:
        status, msg = "pre_open", "NGX pre-open session"
    else:
        status, msg = "closed", "NGX market is closed"
    return {"status": status, "message": msg,
            "data_status": "Delayed Data" if status == "open" else "Market Closed",
            "timestamp": now.isoformat(), "timezone": "Africa/Lagos (WAT, UTC+1)",
            "trading_hours": f"{config.MARKET_HOURS[0]} - {config.MARKET_HOURS[1]} WAT, Monday - Friday",
            "price_delay": "Kobo Terminal prices are delayed by at least 30 minutes"}


def should_poll(now: datetime | None = None) -> bool:
    now = now or datetime.now(config.LAGOS)
    start, end = (time.fromisoformat(x) for x in config.POLL_WINDOW)
    return now.weekday() < 5 and start <= now.time() <= end


async def collector_loop() -> None:
    while True:
        if should_poll():
            try:
                await asyncio.to_thread(collect_once, True)
            except Exception as exc:  # keep polling on the next cycle
                STATE["error"] = str(exc)
                log.warning("Kobo poll failed: %s", exc)
        await asyncio.sleep(config.POLL_SECONDS)


async def startup_refresh(min_age_minutes: float = 60) -> None:
    """At start-up use stored prices; call Kobo only if the newest observation is old."""
    STATE["stocks"] = snapshot_from_db()
    age = latest_observation_age_minutes()
    if age is None or age > min_age_minutes:
        try:
            await asyncio.to_thread(collect_once, True)
        except Exception as exc:
            STATE["error"] = str(exc)
            log.warning("Initial Kobo fetch failed: %s", exc)
