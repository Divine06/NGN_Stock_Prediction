"""Builds the JSON the React dashboard reads.

Field names are a superset of the old main.py and main_live.py responses, so the
existing frontend keeps working while the numbers now come from the ML pipeline.
"""
from __future__ import annotations

import math
import time
from datetime import datetime

import numpy as np
import pandas as pd

from . import config, live_feed, store
from .features import rsi
from .history import add_segments, daily_closes_from_db, discover_tickers, load_history
from .sentiment import get_news, get_sentiment, label_for
from .train import load_summary

_cache: dict = {}
TIMEFRAMES = {"1d": None, "1w": 7, "1m": 30, "3m": 90, "6m": 180, "1y": 365, "5y": 1825, "all": None}
CONF_LABEL = {"high": "High", "medium": "Moderate", "low": "Low"}


def _memo(key, ttl: float, fn):
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    value = fn()
    _cache[key] = (time.time(), value)
    return value


def clear_cache() -> None:
    _cache.clear()


def universe() -> list[str]:
    return discover_tickers()


def _f(x, nd: int = 4):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, nd) if math.isfinite(x) else None


# --- history + indicators -----------------------------------------------------------
def quality(ticker: str) -> dict:
    from .validate import validate_ticker
    return _memo(("quality", ticker), 600, lambda: validate_ticker(ticker))


def invalid_reason(ticker: str) -> str | None:
    q = quality(ticker)
    if q["status"] != "INVALID":
        return None
    bad = next(c for c in q["checks"] if c["status"] == "INVALID")
    return bad["detail"]


def merged_history(ticker: str) -> pd.DataFrame:
    """Daily bars: NGX history + live-feed closes, including today's provisional bar.
    If the historical file failed validation (e.g. London prices), only live-feed days are used."""
    def build():
        if invalid_reason(ticker):
            live = daily_closes_from_db(ticker, include_today=True)
            if live.empty:
                return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "source", "segment"])
            return add_segments(live.assign(source="live_feed"))
        return load_history(ticker, include_today=True)
    return _memo(("hist", ticker), 60, build)


def add_display_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Chart indicators over the full series (as a charting platform would draw them)."""
    out = df.copy()
    c = out["close"].astype(float)
    out["sma_20"] = c.rolling(20).mean()
    out["sma_50"] = c.rolling(50).mean()
    out["ema_20"] = c.ewm(span=20, adjust=False, min_periods=20).mean()
    out["rsi_14"] = rsi(c)
    ema12 = c.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = c.ewm(span=26, adjust=False, min_periods=26).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False, min_periods=9).mean()
    out["macd_histogram"] = out["macd"] - out["macd_signal"]
    std = c.rolling(20).std(ddof=0)
    out["bb_middle"] = out["sma_20"]
    out["bb_upper"] = out["sma_20"] + 2 * std
    out["bb_lower"] = out["sma_20"] - 2 * std
    high = out["high"].astype(float).fillna(c) if "high" in out else c
    low = out["low"].astype(float).fillna(c) if "low" in out else c
    prev = c.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    out["atr_14"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    return out


INDICATOR_KEYS = ["rsi_14", "sma_20", "sma_50", "ema_20", "macd", "macd_signal", "macd_histogram",
                  "bb_upper", "bb_middle", "bb_lower", "atr_14"]


def indicators_for(ticker: str) -> tuple[dict, str | None, pd.DataFrame]:
    hist = merged_history(ticker)
    if hist.empty:
        return {k: None for k in INDICATOR_KEYS}, None, hist
    ind = add_display_indicators(hist)
    last = ind.iloc[-1]
    return {k: _f(last[k], 6) for k in INDICATOR_KEYS}, ind.index[-1].date().isoformat(), ind


# --- live prices ------------------------------------------------------------------
def live_stocks() -> list[dict]:
    if not live_feed.STATE["stocks"]:
        live_feed.STATE["stocks"] = live_feed.snapshot_from_db()
    return live_feed.STATE["stocks"]


def live_quote(ticker: str) -> dict | None:
    return next((s for s in live_stocks() if s["ticker"] == ticker), None)


# --- predictions / decisions ------------------------------------------------------------
def _latest_prediction_rows() -> dict:
    return {r["ticker"]: r for r in store.latest_predictions()}


def _direction(pct: float | None) -> str:
    if pct is None:
        return "INSUFFICIENT_DATA"
    return "UP" if pct > 0.1 else "DOWN" if pct < -0.1 else "SIDEWAYS"


def decision_from_row(ticker: str, row: dict | None, list_style: bool) -> dict:
    live = live_quote(ticker)
    ind, ind_date, _ = indicators_for(ticker)
    base = {
        "ticker": ticker, "name": config.company_name(ticker), "sector": config.SECTORS.get(ticker),
        "rsi_14": ind["rsi_14"], "macd": ind["macd"], "macd_signal": ind["macd_signal"],
        "sma_20": ind["sma_20"], "sma_50": ind["sma_50"], "indicator_timestamp": ind_date,
        "price_source": live.get("source") if live else "NGX historical data",
        "market_data_timestamp": live.get("feed_timestamp") if live else None,
        "data_status": live.get("data_status", "Delayed Data") if live else "Historical Data",
        "change_pct": live.get("change_pct") if live else None,
        "change_percent": live.get("change_pct") if live else None,
        "volume": live.get("volume") if live else None,
    }
    if not row:
        msg = "No ML prediction yet. Train the models (python -m backend.ml.train), then POST /api/predictions/run."
        price = live["price"] if live else None
        return {**base, "timestamp": None, "current_price": price, "price": price, "predicted_price": None,
                "predicted_change_pct": None, "recommendation": "INSUFFICIENT_DATA", "signal": "INSUFFICIENT_DATA",
                "direction": "INSUFFICIENT_DATA", "confidence": "Not available", "score": None,
                "reasons": msg if list_style else [msg], "evidence": [msg], "message": msg,
                "sentiment_score": 0.0, "sentiment_label": "Neutral", "headline_count": 0,
                "models": {}, "model_agreement": None, "rf_prediction": None, "rf_probability_up": None,
                "forecast_path": [], "horizon_days": 0}

    models = row.get("model_outputs") or {}
    r = row.get("ensemble_return")
    pct = None if r is None else (math.exp(r) - 1) * 100
    signs = [np.sign(m.get("return_pct") or 0) for m in models.values()]
    ens_sign = np.sign(r or 0)
    agree = sum(1 for s in signs if s == ens_sign and s != 0)
    rf = models.get("RF", {}).get("return_pct")
    evidence = row.get("evidence") or []
    current = live["price"] if live else row.get("last_close")
    s_score = row.get("sentiment_score")
    return {
        **base,
        "timestamp": row.get("as_of_date"),
        "as_of_date": row.get("as_of_date"),
        "target_date": row.get("target_date"),
        "current_price": current,
        "price": current,
        "forecast_base_price": row.get("last_close"),
        "predicted_price": _f(row.get("ensemble_price")),
        "predicted_change_pct": _f(pct),
        "forecast_path": row.get("forecast_path") or [],
        "horizon_days": len(row.get("forecast_path") or []),
        "horizon_date": (row.get("forecast_path") or [{}])[-1].get("date"),
        "horizon_price": (row.get("forecast_path") or [{}])[-1].get("price"),
        "horizon_change_pct": (row.get("forecast_path") or [{}])[-1].get("return_pct"),
        "recommendation": row.get("signal"),
        "signal": row.get("signal"),
        "direction": _direction(pct) if row.get("signal") != "INSUFFICIENT_DATA" else "INSUFFICIENT_DATA",
        "confidence": CONF_LABEL.get(row.get("confidence"), "Not available"),
        "score": row.get("score"),
        "reasons": " | ".join(evidence) if list_style else evidence,
        "evidence": evidence,
        "message": evidence[0] if evidence else None,
        "sentiment_score": _f(s_score) if s_score is not None else 0.0,
        "sentiment_label": label_for(s_score),
        "headline_count": row.get("n_headlines") or 0,
        "models": models,
        "best_model": row.get("best_model"),
        "model_agreement": (agree >= math.ceil(0.75 * len(signs))) if signs else None,
        "models_agreeing": f"{agree}/{len(signs)}" if signs else None,
        "rf_prediction": None if rf is None else int(rf > 0),
        "rf_probability_up": None,
        "prediction_created_at": row.get("created_at"),
    }


def decisions_payload() -> dict:
    rows = _latest_prediction_rows()
    out = [decision_from_row(t, rows.get(t), list_style=True) for t in universe()]
    from .decision import uniformity_warning
    signals = [d["recommendation"] for d in out]
    return {"count": len(out), "decisions": out, "data_status": "Delayed Data",
            "signal_counts": {s: signals.count(s) for s in sorted(set(signals))},
            "warning": uniformity_warning(signals), "last_refresh": live_feed.STATE["last_refresh"],
            "method": "Ensemble of LSTM, Random Forest, SVR and ARIMA next-day return forecasts, combined with "
                      "technical indicators and Google News (VADER) sentiment"}


def decision_payload(ticker: str) -> dict:
    return decision_from_row(ticker, _latest_prediction_rows().get(ticker), list_style=False)


# --- stock detail / history --------------------------------------------------------
def stock_detail(ticker: str) -> dict:
    ind, ind_date, frame = indicators_for(ticker)
    live = live_quote(ticker)
    last = frame.iloc[-1] if not frame.empty else None
    dec = decision_payload(ticker)
    seg_rows = None
    if not frame.empty:
        seg_rows = int((frame["segment"] == frame["segment"].iloc[-1]).sum())
    bad = invalid_reason(ticker)
    note = None
    if bad:
        note = ("Historical prices failed the data-quality check, so only live-feed prices are shown and "
                f"indicators need about 50 trading days of them. {bad}")
    elif seg_rows is not None and seg_rows < 50:
        note = f"Only {seg_rows} trading days since the last data gap; indicators span the gap"
    price = live["price"] if live else (_f(last["close"]) if last is not None else None)
    return {
        "ticker": ticker, "name": config.company_name(ticker), "sector": config.SECTORS.get(ticker),
        "timestamp": live.get("feed_timestamp") if live else ind_date,
        "price": price, "current_price": price,
        "open": _bar(last, "open"), "high": _bar(last, "high"), "low": _bar(last, "low"),
        "close": _bar(last, "close"),
        "volume": live.get("volume") if live else (_f(last.get("volume"), 0) if last is not None else None),
        "change_pct": (live or {}).get("change_pct"), "change_percent": (live or {}).get("change_pct"),
        "market_data": {"source": (live or {}).get("source"), "status": (live or {}).get("data_status"),
                        "change_percent": (live or {}).get("change_pct"),
                        "feed_timestamp": (live or {}).get("feed_timestamp"),
                        "cached": (live or {}).get("cached", False), "fallback": False},
        "indicators": ind,
        "indicator_timestamp": ind_date,
        "indicator_note": note,
        "data_quality": quality(ticker)["status"],
        "price_source": live.get("source") if live else "NGX historical data",
        "historical_data_source": "NGX daily history (data/raw) + live-feed daily closes",
        "observations": int(len(frame)),
        # prediction summary (main_live.py trend fields)
        "recommendation": dec["recommendation"], "direction": dec["direction"],
        "confidence": dec["confidence"], "predicted_price": dec["predicted_price"],
        "predicted_change_pct": dec["predicted_change_pct"], "message": dec["message"],
        "prediction": dec,
    }


def _bar(row, key: str):
    """OHLC value from a daily row; live-only days have no open, so fall back to the close."""
    if row is None:
        return None
    v = row.get(key, np.nan)
    return _f(row["close"] if v is None or pd.isna(v) else v)


def history_payload(ticker: str, timeframe: str = "1y") -> dict:
    timeframe = timeframe.lower()
    if timeframe == "1d":
        rows = live_feed.intraday(ticker)
        return {"ticker": ticker, "timeframe": "1d", "count": len(rows), "data": rows, "history": rows,
                "note": "20-minute live-feed observations for the latest trading day (delayed)"}
    _, _, frame = indicators_for(ticker)
    bad_note = "Historical file failed the data-quality check; showing live-feed prices only." \
        if invalid_reason(ticker) else None
    if frame.empty:
        return {"ticker": ticker, "timeframe": timeframe, "count": 0, "data": [], "history": [], "note": bad_note}
    days = TIMEFRAMES.get(timeframe, 365)
    if days:
        frame = frame[frame.index >= frame.index[-1] - pd.Timedelta(days=days)]
    rows = []
    for d, r in frame.iterrows():
        iso = d.date().isoformat()
        row = {"timestamp": iso, "observed_at": iso, "date": iso,
               "open": _bar(r, "open"), "high": _bar(r, "high"), "low": _bar(r, "low"),
               "close": _f(r["close"]), "volume": _f(r.get("volume"), 0),
               "source": "Live feed" if r.get("source") == "live_feed" else "NGX history"}
        for k in INDICATOR_KEYS:
            row[k] = _f(r[k], 4)
        rows.append(row)
    gaps = [{"from": frame.index[i - 1].date().isoformat(), "to": frame.index[i].date().isoformat()}
            for i in range(1, len(frame)) if (frame.index[i] - frame.index[i - 1]).days > config.MAX_GAP_DAYS]
    return {"ticker": ticker, "timeframe": timeframe, "count": len(rows), "data": rows, "history": rows,
            "gaps": gaps, "first_date": rows[0]["timestamp"], "last_date": rows[-1]["timestamp"], "note": bad_note}


# --- sentiment & news -------------------------------------------------------------
def sentiment_payload(ticker: str) -> dict:
    s = get_sentiment(ticker)
    score = s.get("score")
    return {
        "ticker": ticker, "company": config.company_name(ticker),
        "sentiment_score": _f(score) if score is not None else 0.0,
        "sentiment_label": label_for(score),
        "headline_count": int(s.get("n_headlines", 0)),
        "headlines": [{"title": h["title"], "link": h.get("link"), "source": h.get("source"),
                       "published": h.get("published"),
                       "sentiment": {"compound": h.get("compound"), "label": label_for(h.get("compound"))}}
                      for h in s.get("headlines", [])],
        "window_days": s.get("window_days", config.SENTIMENT_DAYS),
        "method": "VADER compound score of Google News RSS headlines",
        "used_in_decision": int(s.get("n_headlines", 0)) >= config.MIN_HEADLINES,
        "status": "ok" if s.get("n_headlines") else ("provider_error" if s.get("error") else "no_recent_headlines"),
        "message": s.get("error"),
        "updated_at": datetime.now(config.LAGOS).isoformat(timespec="seconds"),
    }


def news_payload(ticker: str, limit: int = 0) -> dict:
    n = get_news(ticker)
    arts = n.get("articles", [])
    if limit and limit > 0:
        arts = arts[:limit]
    pos = sum(a["sentiment"] == "Positive" for a in arts)
    neg = sum(a["sentiment"] == "Negative" for a in arts)
    mean = sum(a["score"] for a in arts) / len(arts) if arts else None
    return {
        "ticker": ticker, "company_name": config.company_name(ticker),
        "status": "ok" if arts else ("provider_unavailable" if n.get("error") else "no_articles"),
        "article_count": len(arts),
        "sentiment": {"overall": label_for(mean), "score": _f(mean), "positive_count": pos,
                      "negative_count": neg, "method": "VADER"} if arts else None,
        "articles": arts, "cached": False, "message": n.get("error"),
    }


# --- universe / market / health --------------------------------------------------------
def companies_payload() -> dict:
    live = {s["ticker"]: s for s in live_stocks()}
    comps = [{"ticker": t, "name": config.company_name(t), "sector": config.SECTORS.get(t), **live.get(t, {})}
             for t in universe()]
    return {"count": len(comps), "companies": comps, "last_refresh": live_feed.STATE["last_refresh"]}


def market_data_payload() -> dict:
    stocks = live_stocks()
    return {"source": ", ".join(live_feed.STATE.get("sources") or []) or "Stored prices",
            "data_status": "Delayed Data", "count": len(stocks), "stocks": stocks,
            "last_refresh": live_feed.STATE["last_refresh"], "fetched_at": live_feed.STATE["fetched_at"],
            "cached": live_feed.STATE["cached"], "refresh_error": live_feed.STATE["error"]}


def data_quality_payload() -> dict:
    from .validate import validate_all
    reports = _memo(("quality",), 600, validate_all)
    return {"tickers": reports, "invalid": [r["ticker"] for r in reports if r["status"] == "INVALID"],
            "warnings": [r["ticker"] for r in reports if r["status"] == "WARNING"]}


def health_payload() -> dict:
    summaries = [load_summary(t) for t in universe()]
    trained = [s["ticker"] for s in summaries if s and s.get("status") == "OK"]
    rows = store.latest_predictions()
    return {"status": "healthy", "last_refresh": live_feed.STATE["last_refresh"],
            "refresh_error": live_feed.STATE["error"], "poll_interval_seconds": config.POLL_SECONDS,
            "live_sources": config.LIVE_SOURCES, "last_sources_used": live_feed.STATE.get("sources"),
            "poll_window": f"{config.POLL_WINDOW[0]}-{config.POLL_WINDOW[1]} WAT weekdays",
            "market_status": live_feed.market_status(), "models_trained_for": trained,
            "predictions_as_of": max((r["as_of_date"] for r in rows), default=None)}


def predictions_stale() -> bool:
    """True when models exist and a newer completed session exists than the stored predictions."""
    trained = [t for t in universe() if (load_summary(t) or {}).get("status") == "OK"]
    rows = _latest_prediction_rows()
    for t in trained:
        hist = load_history(t)
        if hist.empty:
            continue
        latest = hist.index[-1].date().isoformat()
        if t not in rows or rows[t]["as_of_date"] < latest:
            return True
    return False
