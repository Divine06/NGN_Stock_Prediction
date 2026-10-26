"""ML routes (included by backend/api/main_ml.py)."""
from __future__ import annotations

import math

from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool

from . import config, store
from .decision import uniformity_warning
from .history import coverage_report, discover_tickers, load_history
from .predict import predict_ticker, run_all
from .train import load_summary, metrics_table

router = APIRouter(prefix="/api", tags=["ml-predictions"])


def _json_safe(obj):
    """NaN/inf are not valid JSON (FastAPI would raise); send them as null."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


@router.get("/predictions")
def list_predictions():
    """Latest stored prediction for every ticker (run POST /api/predictions/run first)."""
    rows = store.latest_predictions()
    signals = [r["signal"] for r in rows]
    return _json_safe({"count": len(rows), "warning": uniformity_warning(signals), "predictions": rows})


@router.post("/predictions/run")
async def run_predictions(sentiment: bool = Query(True, description="fetch Google News sentiment")):
    """Recompute predictions now (normally done by the daily scheduler after market close)."""
    return _json_safe(await run_in_threadpool(run_all, None, sentiment, True))


@router.get("/predictions/accuracy")
def prediction_accuracy():
    """How past live predictions compared with the next close actually collected."""
    return _json_safe(store.live_accuracy())


@router.get("/predictions/{ticker}")
async def ticker_prediction(ticker: str, live: bool = Query(False, description="recompute instead of stored"),
                            sentiment: bool = True, history: int = Query(30, ge=1, le=365)):
    ticker = ticker.upper()
    if live:
        current = await run_in_threadpool(predict_ticker, ticker, sentiment)
    else:
        past = store.ticker_predictions(ticker, 1)
        current = past[0] if past else None
    if current is None:
        raise HTTPException(404, f"No prediction stored for {ticker}. Call POST /api/predictions/run "
                                 f"or add ?live=true")
    return _json_safe({"ticker": ticker, "current": current, "history": store.ticker_predictions(ticker, history)})


@router.get("/model-metrics")
def model_metrics():
    """Test-period metrics for every ticker and model (the table for Chapter 4)."""
    summaries = [s for s in (load_summary(t) for t in discover_tickers()) if s]
    table = metrics_table(summaries)
    return _json_safe({
        "metrics": table.to_dict(orient="records") if not table.empty else [],
        "tickers": [{k: s.get(k) for k in ("ticker", "status", "reason", "trained_at", "test_start", "test_end",
                                            "best_model", "confidence", "confidence_reason", "signal_backtest")}
                    for s in summaries],
    })


@router.get("/model-metrics/{ticker}")
def ticker_model_metrics(ticker: str):
    s = load_summary(ticker)
    if not s:
        raise HTTPException(404, f"{ticker.upper()} has not been trained. Run: python -m backend.ml.train --tickers {ticker}")
    return _json_safe(s)


@router.get("/data-coverage/{ticker}")
def data_coverage(ticker: str):
    """Shows what daily history exists, where it came from, and any gaps."""
    report = coverage_report(load_history(ticker))
    report["max_gap_days"] = config.MAX_GAP_DAYS
    return _json_safe(report)
