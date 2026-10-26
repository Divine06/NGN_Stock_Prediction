"""Live-only NGX pipeline: fetch Kobo quotes, choose available peers,
store snapshots in SQLite, and produce cautious trend signals.

No CSV history, feature files, or old model-prediction files are used.
"""
from datetime import datetime, timezone
from statistics import mean
from math import isfinite

from backend.market_data import get_live_stock, fetch_all_stocks
from backend.live_store import save_observation, get_history, count_observations

# Ten preferred companies. Sector peers are tried only when a preferred
# ticker has no valid quote from the current Kobo response.
SECTOR_GROUPS = [
    {
        "sector": "ICT",
        "preferred": [
            ("AIRTELAFRI", "Airtel Africa"),
            ("MTNN", "MTN Nigeria"),
        ],
        "fallbacks": [
            ("CWG", "CWG Plc"),
            ("CHAMS", "Chams Holding Company"),
            ("ETRANZACT", "eTranzact International"),
        ],
    },
    {
        "sector": "Industrial Goods",
        "preferred": [
            ("BUACEMENT", "BUA Cement"),
            ("DANGCEM", "Dangote Cement"),
        ],
        "fallbacks": [
            ("BETAGLAS", "Beta Glass"),
            ("BERGER", "Berger Paints Nigeria"),
            ("CUTIX", "Cutix Plc"),
        ],
    },
    {
        "sector": "Financial Services",
        "preferred": [
            ("GTCO", "Guaranty Trust Holding Company"),
            ("ZENITHBANK", "Zenith Bank"),
        ],
        "fallbacks": [
            ("ACCESSCORP", "Access Holdings"),
            ("UBA", "United Bank for Africa"),
            ("FIDELITYBK", "Fidelity Bank"),
        ],
    },
    {
        "sector": "Consumer Goods",
        "preferred": [
            ("NESTLE", "Nestlé Nigeria"),
            ("UNILEVER", "Unilever Nigeria"),
        ],
        "fallbacks": [
            ("DANGSUGAR", "Dangote Sugar Refinery"),
            ("BUAFOODS", "BUA Foods"),
            ("NB", "Nigerian Breweries"),
        ],
    },
    {
        "sector": "Oil and Gas",
        "preferred": [
            ("SEPLAT", "Seplat Energy"),
            ("TOTAL", "TotalEnergies Marketing Nigeria"),
        ],
        "fallbacks": [
            ("CONOIL", "Conoil"),
            ("OANDO", "Oando"),
            ("ETERNA", "Eterna"),
        ],
    },
]

def _first_value(record, keys):
    for key in keys:
        value = record.get(key)
        if value is not None and value != "":
            return value
    return None


def _to_float(value):
    try:
        number = float(value)
        return number if isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _normalise_live(ticker, name, sector, raw):
    # The existing market_data.py has used both current_price and price
    # naming in different routes. Accept either.
    price = _to_float(_first_value(raw, [
        "current_price", "price", "last_price", "lastPrice",
        "close", "last", "value"
    ]))
    if price is None or price <= 0:
        return None

    change_pct = _to_float(_first_value(raw, [
        "change_percent", "change_pct", "change_percentage",
        "changePercent", "percent_change", "change"
    ]))
    volume = _to_float(_first_value(raw, [
        "volume", "traded_volume", "tradedVolume"
    ]))
    feed_timestamp = _first_value(raw, [
        "feed_timestamp", "timestamp", "updated_at", "updatedAt"
    ])

    return {
        "ticker": ticker,
        "name": name,
        "sector": sector,
        "price": price,
        "current_price": price,
        "change_pct": change_pct,
        "change_percent": change_pct,
        "volume": volume,
        "source": raw.get("source", "Kobo Terminal"),
        "data_status": raw.get("data_status", "Delayed Data"),
        "feed_timestamp": feed_timestamp,
        "cached": bool(raw.get("cached", False)),
        "fallback": bool(raw.get("fallback", False)),
    }


def _fetch_quote(ticker, name, sector):
    try:
        raw = get_live_stock(ticker)
        if not isinstance(raw, dict):
            return None
        return _normalise_live(ticker, name, sector, raw)
    except Exception:
        return None


def fetch_live_universe(force_refresh=False):
    """Return up to 10 stocks, using same-sector fallbacks only when needed."""
    # Refresh the provider's shared cache once, then get quotes from that cache.
    try:
        fetch_all_stocks(force_refresh=force_refresh)
    except Exception as exc:
        print(
            f"Full Kobo refresh failed: {exc}. "
            "Trying individual stock lookups instead."
        )

    stocks = []
    selected_tickers = set()
    for group in SECTOR_GROUPS:
        selected_in_group = []
        candidates = group["preferred"] + group["fallbacks"]
        for ticker, name in candidates:
            if ticker in selected_tickers:
                continue
            quote = _fetch_quote(ticker, name, group["sector"])
            if quote:
                selected_in_group.append(quote)
                selected_tickers.add(ticker)
                # Keep the same number of companies per sector as before.
                if len(selected_in_group) == len(group["preferred"]):
                    break
        stocks.extend(selected_in_group)

    # Save every fetched quote with a local observation timestamp.
    now = datetime.now(timezone.utc).isoformat()
    for stock in stocks:
        save_observation(
            stock["ticker"], stock["price"], stock.get("change_pct"),
            stock.get("volume"), stock.get("source", "Kobo Terminal"),
            observed_at=now
        )
        stock["observations"] = count_observations(stock["ticker"])
        stock["trend"] = build_trend(stock["ticker"])
    return stocks

from statistics import mean

def build_trend(ticker):
    # Load combined historical CSV and live observations.
    history = get_history(ticker, limit=10000)

    # Keep valid, positive prices and sort chronologically.
    valid_rows = []
    for row in history:
        try:
            price = float(row.get("close"))
            date = row.get("timestamp") or row.get("observed_at")

            if price > 0:
                valid_rows.append({
                    "date": str(date or ""),
                    "close": price
                })
        except (TypeError, ValueError):
            continue

    valid_rows.sort(key=lambda row: row["date"])

    # Use the most recent 60 records for a short-horizon estimate.
    recent_rows = valid_rows[-60:]
    prices = [row["close"] for row in recent_rows]

    if len(prices) < 5:
        return {
            "direction": "INSUFFICIENT_DATA",
            "recommendation": "INSUFFICIENT_DATA",
            "confidence": "Not available",
            "observations": len(prices),
            "predicted_price": None,
            "predicted_change_pct": None,
            "message": (
                "At least 5 valid historical or live price observations "
                "are required to estimate a trend."
            )
        }

    # Linear regression against sequential observation index.
    n = len(prices)
    x_mean = (n - 1) / 2
    y_mean = mean(prices)

    denominator = sum((i - x_mean) ** 2 for i in range(n))

    slope = (
        sum(
            (i - x_mean) * (price - y_mean)
            for i, price in enumerate(prices)
        ) / denominator
        if denominator
        else 0
    )

    current_price = prices[-1]
    predicted_price = max(0.01, current_price + slope)
    change_pct = (
        (predicted_price - current_price) / current_price
    ) * 100

    if change_pct >= 0.25:
        recommendation = "BUY"
        direction = "UP"
    elif change_pct <= -0.25:
        recommendation = "SELL"
        direction = "DOWN"
    else:
        recommendation = "HOLD"
        direction = "SIDEWAYS"

    if n < 12:
        confidence = "Low"
    elif n < 30:
        confidence = "Moderate"
    else:
        confidence = "Higher"

    return {
        "direction": direction,
        "recommendation": recommendation,
        "confidence": confidence,
        "observations": n,
        "current_price": round(current_price, 4),
        "predicted_price": round(predicted_price, 4),
        "predicted_change_pct": round(change_pct, 4),
        "message": (
            "Experimental short-horizon trend estimate based on the "
            "most recent available historical and live prices. "
            "This is a simple linear-regression baseline, not a validated "
            "investment prediction."
        )
    }

