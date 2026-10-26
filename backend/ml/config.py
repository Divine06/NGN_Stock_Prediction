"""Central settings for the ML prediction layer.

Every path can be overridden with an environment variable, so the package works
whether you start the API from the project root or from inside backend/.
"""
from __future__ import annotations

import json
import os
from datetime import timedelta, timezone
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent


def _path(env: str, default: Path) -> Path:
    return Path(os.getenv(env, str(default))).resolve()


# --- Where things live (matches the structure in the handover guide) ---------
DATA_DIR = _path("NGX_DATA_DIR", BACKEND_DIR / "data")
DB_PATH = _path("NGX_DB_PATH", DATA_DIR / "ngx_live.db")            # live Kobo store (live_store.py)
RAW_DIR = _path("NGX_RAW_DIR", DATA_DIR / "raw")                     # {TICKER}.csv NGX daily OHLCV 2014-
HISTORY_DIR = _path("NGX_HISTORY_DIR", DATA_DIR / "history")         # optional extra backfill {TICKER}.csv
MODELS_DIR = _path("NGX_MODELS_DIR", DATA_DIR / "ml_models")         # trained models + metrics

OBS_TABLE = os.getenv("NGX_OBS_TABLE", "price_observations")

# Nigeria is UTC+1 all year (no daylight saving). A fixed offset avoids needing
# the tzdata package on Windows.
LAGOS = timezone(timedelta(hours=1), "WAT")
DAILY_CLOSE_READY = os.getenv("NGX_DAILY_CLOSE_READY", "15:00")  # Kobo is >=30 min delayed; NGX closes 14:30
DAILY_RUN_TIME = os.getenv("NGX_DAILY_RUN_TIME", "15:10")        # when the scheduler runs predictions

# --- Market rules used as sanity checks ---------------------------------------
MAX_DAILY_MOVE = 0.10        # NGX +/-10% daily price band: a 1-day forecast beyond this is a bug
DISCONTINUITY_MOVE = 0.115   # a recorded move bigger than this = bonus/split/data error -> new segment
MAX_GAP_DAYS = 10            # calendar-day gap that breaks the series (e.g. CSV end -> live start)

# --- Modelling ----------------------------------------------------------------
WINDOW = 20                  # LSTM look-back (trading days)
HORIZON = int(os.getenv("NGX_HORIZON", "5"))   # forecast 1..HORIZON trading days ahead
BAND_QUANTILES = (0.10, 0.90)  # "likely range" = 80% of test-period errors
TEST_FRACTION = 0.20         # last 20% of usable rows = test period (chronological, never shuffled)
MIN_TRAIN_ROWS = 250         # ~1 trading year
MIN_TEST_ROWS = 60
STALE_DAYS = 7               # refuse to predict if the latest close is older than this
LSTM_EPOCHS = int(os.getenv("NGX_LSTM_EPOCHS", "60"))
RANDOM_STATE = 42

ML_MODELS = ["LSTM", "RF", "SVR", "ARIMA"]
ALLOW_GAP_BRIDGE = True      # live prediction may span a data gap (flagged, confidence forced to low)

# --- Data validation (ml/validate.py) ---------------------------------------
SCALE_MISMATCH_RATIO = 5.0   # live price vs last historical close beyond 5x = wrong exchange/currency
BAND_BREACH_WARN = 0.002     # share of consecutive-day moves beyond the NGX band
BAND_BREACH_INVALID = 0.005  # ...above this (and >= 10 cases) the history is not NGX data

# --- Kobo live feed (ml/live_feed.py) ---------------------------------------
POLL_SECONDS = int(os.getenv("NGX_POLL_SECONDS", str(20 * 60)))
POLL_WINDOW = ("09:55", "15:05")
# Free price sources, tried in this order. "kobo" needs KOBO_API_KEY, "ngnmarket" needs
# NGN_MARKET_API_KEY (both free; put them in the .env file in the project root).
LIVE_SOURCES = [s.strip() for s in os.getenv("LIVE_SOURCES", "ngnmarket,kobo").split(",") if s.strip()]
NGN_MARKET_BASE = os.getenv("NGN_MARKET_BASE", "https://api.ngnmarket.com/v1")   # NGX trades 10:00-14:30 WAT; one extra poll catches the delayed close
MARKET_HOURS = ("10:00", "14:30")

# --- Decision engine (document these in Chapter 3, section 8.5) ---------------
WEIGHTS = {"forecast": 0.5, "technical": 0.3, "sentiment": 0.2}
BUY_THRESHOLD = 0.25
SELL_THRESHOLD = -0.25
CONFIDENCE_MULTIPLIER = {"high": 1.0, "medium": 0.6, "low": 0.3}
FORECAST_SCALE_VOL = 0.25    # forecast is measured in units of 0.25 x 20-day volatility
NEUTRAL_BAND = 0.005         # price within 0.5% of its SMA (or MACD within 0.05%) counts as neutral
MIN_HEADLINES = 3            # fewer recent headlines than this -> sentiment left out
SENTIMENT_DAYS = 7
SENTIMENT_CACHE_MINUTES = 60

# --- Universe -----------------------------------------------------------------
# The ten project companies (same list as market_data.py / main.py).
UNIVERSE = {
    "AIRTELAFRI": ("Airtel Africa", "Telecommunications"),
    "BUACEMENT": ("BUA Cement", "Industrial Goods"),
    "DANGCEM": ("Dangote Cement", "Industrial Goods"),
    "GTCO": ("Guaranty Trust Holding Company", "Financial Services"),
    "MTNN": ("MTN Nigeria", "Telecommunications"),
    "NESTLE": ("Nestle Nigeria", "Consumer Goods"),
    "SEPLAT": ("Seplat Energy", "Oil and Gas"),
    "TOTAL": ("TotalEnergies Marketing Nigeria", "Oil and Gas"),
    "UNILEVER": ("Unilever Nigeria", "Consumer Goods"),
    "ZENITHBANK": ("Zenith Bank", "Financial Services"),
}
COMPANY_NAMES = {t: v[0] for t, v in UNIVERSE.items()}
SECTORS = {t: v[1] for t, v in UNIVERSE.items()}

# Google News search terms (quoted phrases keep results on the company).
NEWS_QUERIES = {
    "AIRTELAFRI": '"Airtel Africa"',
    "BUACEMENT": '"BUA Cement"',
    "DANGCEM": '"Dangote Cement"',
    "GTCO": '"GTCO" OR "Guaranty Trust Holding"',
    "MTNN": '"MTN Nigeria"',
    "NESTLE": '"Nestle Nigeria"',
    "SEPLAT": '"Seplat Energy" OR "Seplat"',
    "TOTAL": '"TotalEnergies Marketing Nigeria" OR "TotalEnergies Nigeria"',
    "UNILEVER": '"Unilever Nigeria"',
    "ZENITHBANK": '"Zenith Bank"',
}


# The list above can be replaced without editing code: backend/data/universe.json
# (written by `python -m backend.ml.replace_ticker`). Format:
# {"companies": {"OANDO": {"name": "Oando", "sector": "Oil and Gas", "news_query": "\"Oando\""}, ...}}
UNIVERSE_FILE = DATA_DIR / "universe.json"


def _load_universe_file() -> None:
    if not UNIVERSE_FILE.exists():
        return
    try:
        companies = json.loads(UNIVERSE_FILE.read_text(encoding="utf-8"))["companies"]
        universe = {t.upper(): (v["name"], v.get("sector", "")) for t, v in companies.items()}
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(f"{UNIVERSE_FILE} is not valid: {exc}") from exc
    UNIVERSE.clear()
    UNIVERSE.update(universe)
    COMPANY_NAMES.clear()
    COMPANY_NAMES.update({t: v[0] for t, v in universe.items()})
    SECTORS.clear()
    SECTORS.update({t: v[1] for t, v in universe.items()})
    for t, v in companies.items():
        if v.get("news_query"):
            NEWS_QUERIES[t.upper()] = v["news_query"]


_load_universe_file()


def configured_tickers() -> list[str] | None:
    raw = os.getenv("NGX_TICKERS", "").strip()
    return [t.strip().upper() for t in raw.split(",") if t.strip()] or None


def company_name(ticker: str) -> str:
    return COMPANY_NAMES.get(ticker.upper(), ticker.upper())
