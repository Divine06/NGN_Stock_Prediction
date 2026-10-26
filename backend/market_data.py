import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

KOBO_API_KEY = os.getenv("KOBO_API_KEY")

KOBO_BASE_URL = "https://koboterminal.com"

STOCKS_ENDPOINT = f"{KOBO_BASE_URL}/api/ngxdata/stocks"

NGX_TIMEZONE = ZoneInfo("Africa/Lagos")


# =========================================================
# PROJECT COMPANIES
# =========================================================

COMPANIES = [
    "AIRTELAFRI",
    "BUACEMENT",
    "DANGCEM",
    "GTCO",
    "MTNN",
    "NESTLE",
    "SEPLAT",
    "TOTAL",
    "UNILEVER",
    "ZENITHBANK",
]


# =========================================================
# CACHE
# =========================================================
#
# Kobo Personal tier = 100 requests/day.
#
# We therefore DO NOT call Kobo every time React refreshes.
#
# React can refresh every 30 seconds.
# Kobo will only be contacted when our cache expires.
#
# 10 minutes = about 42 external requests/day if running
# continuously.
#
# This leaves plenty of room under the 100/day limit.
# =========================================================

CACHE_SECONDS = 600

_cache = {
    "data": None,
    "fetched_at": 0,
}


# =========================================================
# API KEY CHECK
# =========================================================

def check_api_key():
    if not KOBO_API_KEY:
        raise RuntimeError(
            "KOBO_API_KEY is missing. "
            "Add KOBO_API_KEY=your_key to the .env file."
        )


# =========================================================
# FETCH ALL NGX STOCKS
# =========================================================

def fetch_all_stocks(force_refresh=False):
    """
    Fetch all NGX equities from Kobo Terminal.

    The result is cached for CACHE_SECONDS so that the
    frontend can poll our FastAPI backend frequently without
    repeatedly consuming the external API quota.
    """

    check_api_key()

    now = time.time()

    # Return cached data when it is still fresh
    if (
        not force_refresh
        and _cache["data"] is not None
        and (now - _cache["fetched_at"]) < CACHE_SECONDS
    ):
        return {
            "stocks": _cache["data"],
            "cached": True,
            "fetched_at": datetime.fromtimestamp(
                _cache["fetched_at"],
                tz=NGX_TIMEZONE
            ).isoformat(),
        }

    headers = {
        "X-API-Key": KOBO_API_KEY,
        "Content-Type": "application/json",
    }

    try:
        response = requests.get(
            STOCKS_ENDPOINT,
            headers=headers,
            timeout=15,
        )

        response.raise_for_status()

        payload = response.json()

        # Kobo currently returns the equities as a list.
        # This also handles a possible {"data": [...]} wrapper.
        if isinstance(payload, list):
            stocks = payload

        elif isinstance(payload, dict):
            stocks = (
                payload.get("data")
                or payload.get("stocks")
                or payload.get("results")
                or []
            )

        else:
            stocks = []

        if not isinstance(stocks, list):
            raise RuntimeError(
                "Unexpected response format from Kobo Terminal."
            )

        # Update cache
        _cache["data"] = stocks
        _cache["fetched_at"] = now

        return {
            "stocks": stocks,
            "cached": False,
            "fetched_at": datetime.fromtimestamp(
                now,
                tz=NGX_TIMEZONE
            ).isoformat(),
        }

    except requests.RequestException as error:
        # If the external service temporarily fails but we have
        # previous data, continue serving the cached data.
        if _cache["data"] is not None:
            return {
                "stocks": _cache["data"],
                "cached": True,
                "fallback": True,
                "fetched_at": datetime.fromtimestamp(
                    _cache["fetched_at"],
                    tz=NGX_TIMEZONE
                ).isoformat(),
                "warning": f"Kobo API temporarily unavailable: {error}",
            }

        raise RuntimeError(
            f"Unable to fetch NGX market data from Kobo Terminal: {error}"
        )


# =========================================================
# FIND ONE STOCK
# =========================================================

def get_live_stock(ticker):
    """
    Return the latest Kobo price for one of our project stocks.
    """

    ticker = ticker.upper()

    if ticker not in COMPANIES:
        raise ValueError(f"{ticker} is not in the project company list.")

    result = fetch_all_stocks()

    for stock in result["stocks"]:
        symbol = str(
            stock.get("symbol")
            or stock.get("ticker")
            or ""
        ).upper()

        if symbol == ticker:
            return {
                "ticker": ticker,
                "name": stock.get("name"),
                "price": stock.get("current_price"),
                "change_percent": stock.get("change_percent"),
                "volume": stock.get("volume"),
                "sector": stock.get("sector"),
                "source": "Kobo Terminal",
                "data_status": "Delayed Data",
                "feed_timestamp": result["fetched_at"],
                "cached": result.get("cached", False),
                "fallback": result.get("fallback", False),
            }

    raise ValueError(
        f"{ticker} was not returned by the Kobo Terminal API."
    )


# =========================================================
# MARKET STATUS
# =========================================================

def get_market_status():
    """
    NGX trading hours:
    Monday-Friday
    9:00 AM - 4:00 PM WAT
    """

    now = datetime.now(NGX_TIMEZONE)

    weekday = now.weekday()

    market_open_time = now.replace(
        hour=9,
        minute=0,
        second=0,
        microsecond=0,
    )

    market_close_time = now.replace(
        hour=16,
        minute=0,
        second=0,
        microsecond=0,
    )

    is_weekday = weekday < 5

    is_open = (
        is_weekday
        and market_open_time <= now < market_close_time
    )

    if is_open:
        status = "open"
        message = "NGX Market is Open"
        data_status = "Delayed Data"

    else:
        status = "closed"
        message = "NGX Market is Closed"
        data_status = "Market Closed"

    return {
        "status": status,
        "message": message,
        "data_status": data_status,
        "timestamp": now.isoformat(),
        "timezone": "Africa/Lagos",
        "trading_hours": "09:00 - 16:00 WAT, Monday - Friday",
    }


# =========================================================
# FORCE REFRESH
# =========================================================

def refresh_market_data():
    """
    Force a fresh request to Kobo Terminal.
    """

    return fetch_all_stocks(force_refresh=True)