"""NGX Stock Market Trend Prediction and Decision Support System: unified API.

Run from the project root (the folder that contains backend/):

    uvicorn backend.api.main_ml:app --reload

Data flow:
  NGX daily history (data/raw, 2014-)  ─┐
  NGN Market / Kobo prices (20 min)   ──┴─> daily series ─> LSTM / RF / SVR / ARIMA
  Google News RSS + VADER sentiment ───────────────────────> decision engine ─> dashboard

This replaces main_live.py (live trend only) and main.py (stale CSV predictions).
Route names and response fields stay compatible with the existing React dashboard.
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from backend.ml import config, dashboard, live_feed
from backend.ml.predict import run_all
from backend.ml.routes import _json_safe, router as ml_router
from backend.ml.scheduler import start_daily_predictions, stop_daily_predictions

log = logging.getLogger("ngx.api")


async def _refresh_predictions_if_stale() -> None:
    try:
        if await asyncio.to_thread(dashboard.predictions_stale):
            await asyncio.to_thread(run_all, None, True, True)
            dashboard.clear_cache()
    except Exception:
        log.exception("Start-up prediction run failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    live_feed.init_db()
    await live_feed.startup_refresh()
    collector = asyncio.create_task(live_feed.collector_loop())
    start_daily_predictions()
    warmup = asyncio.create_task(_refresh_predictions_if_stale())
    try:
        yield
    finally:
        for task in (collector, warmup):
            task.cancel()
        stop_daily_predictions()


app = FastAPI(
    title="NGX Stock Market Trend Prediction and Decision Support System",
    description="NGX historical data + Kobo Terminal live prices -> LSTM, Random Forest, SVR and ARIMA "
                "-> Google News sentiment -> decision support. Not financial advice.",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv(
        "FRONTEND_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ml_router)  # /api/predictions, /api/model-metrics, /api/data-coverage


def _ticker(ticker: str) -> str:
    t = ticker.upper()
    if t not in dashboard.universe():
        raise HTTPException(status_code=404, detail=f"Company {t} not found")
    return t


@app.get("/")
def root():
    return {"system": "NGX Stock Market Trend Prediction and Decision Support System", "status": "running",
            "backend": "FastAPI", "historical_data": "NGX daily history (backend/data/raw)",
            "live_data": "NGN Market / Kobo Terminal free tiers (delayed 20-30 min)",
            "live_sources": config.LIVE_SOURCES, "models": config.ML_MODELS,
            "sentiment": "Google News RSS + VADER", "docs": "/docs"}


@app.get("/api/health")
def health():
    return _json_safe(dashboard.health_payload())


@app.get("/api/companies")
def companies():
    return _json_safe(dashboard.companies_payload())


@app.get("/api/market-status")
def market_status():
    return live_feed.market_status()


@app.get("/api/market-data")
def market_data():
    payload = dashboard.market_data_payload()
    if not payload["stocks"]:
        raise HTTPException(status_code=503, detail={
            "message": "No live prices stored yet.", "refresh_error": payload["refresh_error"]})
    return _json_safe(payload)


@app.get("/api/stocks/{ticker}")
def stock(ticker: str):
    return _json_safe(dashboard.stock_detail(_ticker(ticker)))


@app.get("/api/stocks/{ticker}/history")
def stock_history(ticker: str, timeframe: str = Query("1y", pattern="^(1d|1w|1m|3m|6m|1y|5y|all)$")):
    return _json_safe(dashboard.history_payload(_ticker(ticker), timeframe))


@app.get("/api/stocks/{ticker}/sentiment")
async def stock_sentiment(ticker: str):
    return _json_safe(await run_in_threadpool(dashboard.sentiment_payload, _ticker(ticker)))


@app.get("/api/stocks/{ticker}/news")
async def stock_news(ticker: str, limit: int = 0):
    return _json_safe(await run_in_threadpool(dashboard.news_payload, _ticker(ticker), limit))


@app.get("/api/decisions")
def decisions():
    return _json_safe(dashboard.decisions_payload())


@app.get("/api/decisions/{ticker}")
def decision(ticker: str):
    return _json_safe(dashboard.decision_payload(_ticker(ticker)))


@app.get("/api/advisory/{ticker}")
def advisory(ticker: str):
    return decision(ticker)


@app.get("/api/data-quality")
def data_quality():
    return _json_safe(dashboard.data_quality_payload())


@app.post("/api/refresh")
async def refresh():
    """Fetch live prices now (uses one request of the free quota)."""
    try:
        result = await run_in_threadpool(live_feed.collect_once, True)
    except Exception as exc:
        live_feed.STATE["error"] = str(exc)
        raise HTTPException(status_code=502, detail=f"Live refresh failed: {exc}")
    dashboard.clear_cache()
    return _json_safe({"ok": True, "count": len(result["stocks"]), "last_refresh": result["last_refresh"],
                       "stocks": result["stocks"]})
