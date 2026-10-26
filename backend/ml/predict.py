"""Daily prediction run: latest daily close -> saved models -> decision -> SQLite.

    python -m backend.ml.predict                 # all tickers, with Google News sentiment
    python -m backend.ml.predict --no-sentiment  # offline
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from . import config, store
from .decision import decide, uniformity_warning
from .features import build_features, feature_mask, latest_indicators
from .history import discover_tickers, load_history
from .models import MODEL_CLASSES, model_dir
from .train import ensemble_mean, load_summary
from .validate import validate_ticker

_model_cache: dict[tuple[str, str], tuple[str, object]] = {}


def _load_model(ticker: str, name: str, trained_at: str):
    """Load once per training run (cache is invalidated when metrics.json changes)."""
    key = (ticker, name)
    hit = _model_cache.get(key)
    if hit and hit[0] == trained_at:
        return hit[1]
    model = MODEL_CLASSES[name].load(model_dir(ticker, name))
    _model_cache[key] = (trained_at, model)
    return model


def _next_weekday(d: pd.Timestamp) -> str:
    return _next_weekdays(d, 1)[0]


def _next_weekdays(d: pd.Timestamp, n: int) -> list[str]:
    """The next n weekdays after d (NGX public holidays are not known to the system)."""
    out, cur = [], d
    while len(out) < n:
        cur = cur + timedelta(days=1)
        if cur.weekday() < 5:
            out.append(cur.date().isoformat())
    return out


def _clamp_path(path: np.ndarray) -> tuple[np.ndarray, bool]:
    """A cumulative h-day return cannot exceed the NGX +/-10% daily band compounded h times."""
    h = np.arange(1, len(path) + 1)
    lo, hi = h * math.log(1 - config.MAX_DAILY_MOVE), h * math.log(1 + config.MAX_DAILY_MOVE)
    clipped = np.clip(path, lo, hi)
    return clipped, bool(np.any(clipped != path))


def _insufficient(ticker: str, reason: str, as_of: str | None = None, last_close=None) -> dict:
    return {"ticker": ticker, "company": config.company_name(ticker), "as_of_date": as_of,
            "last_close": last_close, "signal": "INSUFFICIENT_DATA", "score": None,
            "evidence": [reason], "models": {}}


def predict_ticker(ticker: str, use_sentiment: bool = True, now: datetime | None = None) -> dict:
    ticker = ticker.upper()
    now = now or datetime.now(config.LAGOS)
    hist = load_history(ticker, now=now)
    if hist.empty:
        return _insufficient(ticker, "No price history found for this ticker")
    feats = build_features(hist)
    as_of = feats.index[-1]
    last_close = float(feats["close"].iloc[-1])

    store.update_realised(ticker, {d.date().isoformat(): float(c) for d, c in hist["close"].items()})

    quality = validate_ticker(ticker, now=now)
    if quality["status"] == "INVALID":
        bad = next(c for c in quality["checks"] if c["status"] == "INVALID")
        return _insufficient(ticker, f"Data quality check failed ({bad['check']}): {bad['detail']}",
                             as_of.date().isoformat(), last_close)

    summary = load_summary(ticker)
    if not summary or summary.get("status") != "OK" or not summary.get("saved_models"):
        return _insufficient(ticker, "Models not trained for this ticker yet: run python -m backend.ml.train",
                             as_of.date().isoformat(), last_close)
    if summary.get("horizon", 1) != config.HORIZON:
        return _insufficient(ticker, f"Models were trained for a {summary.get('horizon', 1)}-day horizon; the system "
                                     f"now forecasts {config.HORIZON} days. Retrain: python -m backend.ml.train",
                             as_of.date().isoformat(), last_close)
    age = (pd.Timestamp(now.date()) - as_of).days
    if age > config.STALE_DAYS:
        return _insufficient(ticker, f"Latest close is from {as_of.date()} ({age} days old); "
                                     "the live feed or history needs updating", as_of.date().isoformat(), last_close)
    data_warning = None
    if not bool(feature_mask(feats).iloc[-1]):
        seg = feats[feats["segment"] == feats["segment"].iloc[-1]]
        seg_rows = len(seg)
        bridged = build_features(hist.assign(segment=0)) if config.ALLOW_GAP_BRIDGE else None
        if bridged is None or not bool(feature_mask(bridged).iloc[-1]):
            return _insufficient(ticker, f"Only {seg_rows} continuous trading days since the last gap in the "
                                         "data; indicators need about 50. Fill the gap: python -m backend.ml.import_csv",
                                 as_of.date().isoformat(), last_close)
        prev_day = hist.index[hist.index.get_loc(seg.index[0]) - 1]
        data_warning = (f"Only {seg_rows} trading days since a data gap ({prev_day.date()} to "
                        f"{seg.index[0].date()}); model inputs and indicators span the gap, so confidence is set "
                        "to low. Fill the gap: python -m backend.ml.import_csv")
        feats = bridged

    recent = feats.tail(500)  # enough history for ARIMA's filter and the LSTM window
    H = config.HORIZON
    dates = _next_weekdays(as_of, H)
    model_returns, model_paths, model_outputs, errors = {}, {}, {}, {}
    for name in summary["saved_models"]:
        try:
            m = _load_model(ticker, name, summary["trained_at"])
            path = np.asarray(m.predict(recent)[-1], dtype=float).reshape(H)
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
            continue
        if np.all(np.isfinite(path)):
            path, _ = _clamp_path(path)
            model_returns[name] = float(path[0])
            model_paths[name] = path
            model_outputs[name] = {
                "return_pct": round(float(path[0]) * 100, 4),
                "price": round(last_close * math.exp(path[0]), 4),
                "path_return_pct": [round(float(r) * 100, 4) for r in path],
                "path_price": [round(last_close * math.exp(r), 4) for r in path],
            }
    if not model_returns:
        rec = _insufficient(ticker, f"No model produced a forecast: {errors}", as_of.date().isoformat(), last_close)
        return rec

    ens_path, clamped = _clamp_path(ensemble_mean(list(model_paths.values())))
    ens = float(ens_path[0])
    bands = {b["h"]: b for b in summary.get("forecast_band") or []}
    forecast_path = []
    for h in range(1, H + 1):
        r = float(ens_path[h - 1])
        b = bands.get(h)
        forecast_path.append({
            "h": h, "date": dates[h - 1],
            "price": round(last_close * math.exp(r), 4),
            "return_pct": round((math.exp(r) - 1) * 100, 4),
            "low": round(last_close * math.exp(r + b["q_low"]), 4) if b else None,
            "high": round(last_close * math.exp(r + b["q_high"]), 4) if b else None,
            "typical_error_pct": round(b["ensemble_mape_pct"], 3) if b else None,
            "naive_error_pct": round(b["naive_mape_pct"], 3) if b else None,
        })
    signs = [np.sign(v) for v in model_returns.values()]
    majority = max(signs.count(1.0), signs.count(-1.0))
    agreement = f"{majority}/{len(signs)} models agree on direction"

    indicators = latest_indicators(feats)
    sentiment = None
    if use_sentiment:
        from .sentiment import get_sentiment
        sentiment = get_sentiment(ticker)

    confidence = "low" if data_warning else summary.get("confidence", "low")
    forecast = {"ensemble_return": ens, "confidence": confidence,
                "confidence_reason": ("forced to low because of the data gap" if data_warning
                                      else summary.get("confidence_reason")),
                "model_returns": model_returns, "agreement": agreement, "clamped": clamped}
    decision = decide(forecast, indicators, sentiment)
    last = forecast_path[-1]
    outlook = (f"{H}-day outlook: {last['return_pct']:+.2f}% to ₦{last['price']:,.2f} by {last['date']}"
               + (f" (likely range ₦{last['low']:,.2f} – ₦{last['high']:,.2f})" if last["low"] else ""))
    decision.evidence.insert(1, outlook)
    if data_warning:
        decision.evidence.insert(0, data_warning)
    for c in quality["checks"]:
        if c["status"] == "WARNING" and c["check"] in ("band",):
            decision.evidence.append(f"Data quality warning ({c['check']}): {c['detail']}")

    return {
        "ticker": ticker,
        "company": config.company_name(ticker),
        "as_of_date": as_of.date().isoformat(),
        "target_date": _next_weekday(as_of),
        "last_close": last_close,
        "data_source": str(hist["source"].iloc[-1]),
        "ensemble_return": ens,
        "ensemble_return_pct": round(ens * 100, 4),
        "ensemble_price": round(last_close * math.exp(ens), 4),
        "forecast_path": forecast_path,
        "horizon_days": H,
        "models": model_outputs,
        "best_model": summary.get("best_model"),
        "confidence": confidence,
        "data_warning": data_warning,
        "data_quality": quality["status"],
        "signal": decision.signal,
        "score": decision.score,
        "votes": decision.votes,
        "evidence": decision.evidence,
        "indicators": indicators,
        "sentiment": sentiment,
        "model_errors": errors,
    }


def run_all(tickers: list[str] | None = None, use_sentiment: bool = True, save: bool = True,
            now: datetime | None = None) -> dict:
    tickers = tickers or discover_tickers()
    records = []
    for t in tickers:
        try:
            rec = predict_ticker(t, use_sentiment=use_sentiment, now=now)
        except Exception as exc:  # one bad ticker must not stop the run
            rec = _insufficient(t.upper(), f"Prediction failed: {type(exc).__name__}: {exc}")
        records.append(rec)
        if save:
            store.save_prediction(rec)
    signals = [r["signal"] for r in records]
    return {
        "generated_at": datetime.now(config.LAGOS).isoformat(timespec="seconds"),
        "signal_counts": {s: signals.count(s) for s in sorted(set(signals))},
        "warning": uniformity_warning(signals),
        "predictions": records,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run today's NGX predictions")
    ap.add_argument("--tickers")
    ap.add_argument("--no-sentiment", action="store_true")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args(argv)
    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else None
    out = run_all(tickers, use_sentiment=not args.no_sentiment, save=not args.no_save)
    for r in out["predictions"]:
        ret = r.get("ensemble_return_pct")
        print(f"{r['ticker']:<12} {r['signal']:<18} "
              f"{'' if ret is None else f'{ret:+.2f}%':>8}  score {r.get('score')}  conf {r.get('confidence')}")
    print(json.dumps({k: v for k, v in out.items() if k != "predictions"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
