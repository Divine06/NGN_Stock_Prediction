from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
import asyncio
import requests

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from backend.live_store import init_db, get_history, latest_observations
from backend.live_pipeline import fetch_live_universe, build_trend, SECTOR_GROUPS
from backend.news_sentiment import get_company_news

# Kobo Terminal updates its public board on a delayed cadence. Avoid polling
# more frequently than every 20 minutes; adjust only if your API plan permits.
POLL_SECONDS = 20 * 60
_latest_stocks = []
_last_refresh = None
_refresh_error = None


async def _collector():
    global _latest_stocks, _last_refresh, _refresh_error
    while True:
        try:
            stocks = await asyncio.to_thread(fetch_live_universe, True)
            _latest_stocks = stocks
            _last_refresh = datetime.now(timezone.utc).isoformat()
            _refresh_error = None
        except Exception as exc:
            _refresh_error = str(exc)
        await asyncio.sleep(POLL_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _latest_stocks, _last_refresh
    init_db()
    try:
        _latest_stocks = await asyncio.to_thread(fetch_live_universe, True)
        _last_refresh = datetime.now(timezone.utc).isoformat()
    except Exception as exc:
        global _refresh_error
        _refresh_error = str(exc)
    task = asyncio.create_task(_collector())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="NGX Live Trend Prediction and Decision Support System",
    description="Live Kobo quotes with automatically collected SQLite observations. No historical CSV dependency.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "system": "NGX Live Trend Prediction and Decision Support System",
        "status": "running",
        "backend": "FastAPI",
        "data_source": "Kobo Terminal",
        "history_source": "Live observations collected from launch onward",
        "csv_dependency": False,
    }


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "last_refresh": _last_refresh,
        "refresh_error": _refresh_error,
        "poll_interval_seconds": POLL_SECONDS,
    }


@app.get("/api/companies")
def companies():
    return {
        "count": len(_latest_stocks),
        "companies": _latest_stocks,
        "sector_groups": [
            {"sector": group["sector"], "preferred_tickers": [t for t, _ in group["preferred"]],
             "fallback_tickers": [t for t, _ in group["fallbacks"]]}
            for group in SECTOR_GROUPS
        ],
        "last_refresh": _last_refresh,
    }


@app.get("/api/market-data")
def market_data():
    if not _latest_stocks:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "No live quotes are currently available from Kobo Terminal.",
                "refresh_error": _refresh_error,
            },
        )
    return {
        "count": len(_latest_stocks),
        "stocks": _latest_stocks,
        "last_refresh": _last_refresh,
        "data_status": "Delayed Data",
    }


@app.get("/api/decisions")
def decisions():
    if not _latest_stocks:
        raise HTTPException(status_code=503, detail="No live market data is available.")
    output = []
    for stock in _latest_stocks:
        trend = build_trend(stock["ticker"])
        output.append({
            **stock,
            **trend,
            "current_price": stock["price"],
            "price_source": stock.get("source", "Kobo Terminal"),
            "market_data_timestamp": stock.get("feed_timestamp") or _last_refresh,
            "data_status": stock.get("data_status", "Delayed Data"),
        })
    return {"count": len(output), "decisions": output, "last_refresh": _last_refresh}


@app.get("/api/stocks/{ticker}")
def stock_detail(ticker: str):
    ticker = ticker.upper()
    stock = next((item for item in _latest_stocks if item["ticker"] == ticker), None)
    if not stock:
        raise HTTPException(status_code=404, detail=f"{ticker} is not currently available in the selected live universe.")
    return {**stock, **build_trend(ticker)}


@app.get("/api/stocks/{ticker}/history")
def stock_history(
    ticker: str,
    timeframe: str = Query("1m", pattern="^(1d|1w|1m|3m|6m|1y|all)$"),
):
    ticker = ticker.upper()
    if not any(item["ticker"] == ticker for item in _latest_stocks):
        raise HTTPException(status_code=404, detail=f"{ticker} is not currently available.")
    days = {"1d": 1, "1w": 7, "1m": 30, "3m": 90, "6m": 180, "1y": 365}.get(timeframe)
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat() if days else None
    history = get_history(ticker, limit=10000, since=since)
    return {"ticker": ticker, "timeframe": timeframe, "count": len(history), "data": history, "history": history}

@app.get("/api/stocks/{ticker}/news")
def stock_news(ticker: str, limit: int = 0):
    try:
        return get_company_news(ticker, limit)
    except requests.RequestException as exc:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=502,
            detail=f"News provider request failed: {exc}",
        )
    except ValueError as exc:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=502,
            detail=f"News provider returned invalid data: {exc}",
        )


@app.post("/api/refresh")
async def refresh():
    global _latest_stocks, _last_refresh, _refresh_error
    try:
        stocks = await asyncio.to_thread(fetch_live_universe, True)
        _latest_stocks = stocks
        _last_refresh = datetime.now(timezone.utc).isoformat()
        _refresh_error = None
        return {"ok": True, "count": len(stocks), "last_refresh": _last_refresh, "stocks": stocks}
    except Exception as exc:
        _refresh_error = str(exc)
        raise HTTPException(status_code=502, detail=f"Live refresh failed: {exc}")


@app.get("/api/stocks/{ticker}/sentiment")
def sentiment_disabled(ticker: str):
    # Keep the route compatible with the existing frontend, but avoid
    # pretending news sentiment is part of this live-price-only rebuild.
    return {
        "ticker": ticker.upper(),
        "sentiment_score": 0,
        "sentiment_label": "Not included in live-only baseline",
        "headline_count": 0,
        "headlines": [],
        "note": "This first simplified version focuses on live price observations."
    }
