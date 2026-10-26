"""Train, evaluate and save LSTM / RF / SVR / ARIMA for every ticker.

Run from the project root (the folder containing backend/):
    python -m backend.ml.train                       # all tickers found in data/
    python -m backend.ml.train --tickers DANGCEM,MTNN
    python -m backend.ml.train --models RF,SVR,ARIMA # skip LSTM for a quick run

Evaluation protocol (report this in Chapter 3/4):
  * chronological split: last 20% of usable trading days = test period
  * every model is fitted on the training period only, then scored on the test period
  * targets: total return to 1, 2, ... HORIZON trading days ahead (default 5)
  * metrics per horizon: return MAE/RMSE, price MAE/RMSE/MAPE, directional accuracy,
    and skill versus the no-change (naive) baseline; `metrics` holds day 1
  * "likely range": 10th-90th percentile of the ensemble's test-period errors per horizon
  * the ensemble is the plain mean of the ML models (no weights fitted on the test set)
  * after scoring, each model is refitted on all data and saved for live use
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

from . import config
from .decision import decide
from .features import build_features, feature_mask, target_mask
from .history import coverage_report, discover_tickers, load_history
from .models import MODEL_CLASSES, ModelUnavailable, model_dir


def ensemble_mean(arrays: list[np.ndarray]) -> np.ndarray:
    """Plain average of the ML models' forecasts, element by element (works for 1-D and 2-D)."""
    stacked = np.stack([np.asarray(a, dtype=float) for a in arrays], axis=0)
    counts = np.isfinite(stacked).sum(axis=0)
    total = np.nansum(stacked, axis=0)
    return np.where(counts > 0, total / np.maximum(counts, 1), np.nan)


def compute_metrics(actual: np.ndarray, pred: np.ndarray, close_t: np.ndarray) -> dict:
    err = pred - actual
    p_pred = close_t * np.exp(pred)
    p_act = close_t * np.exp(actual)
    moved = np.abs(actual) > 1e-9
    out = {
        "n": int(len(actual)),
        "ret_mae_pct": float(np.mean(np.abs(err)) * 100),
        "ret_rmse_pct": float(np.sqrt(np.mean(err ** 2)) * 100),
        "price_mae": float(np.mean(np.abs(p_pred - p_act))),
        "price_rmse": float(np.sqrt(np.mean((p_pred - p_act) ** 2))),
        "price_mape_pct": float(np.mean(np.abs((p_pred - p_act) / p_act)) * 100),
        "dir_acc": None,
        "share_pred_down": float(np.mean(pred < 0)),
    }
    if moved.any() and np.any(pred != 0):
        out["dir_acc"] = float(np.mean(np.sign(pred[moved]) == np.sign(actual[moved])))
    return out


def confidence_from(ens: dict, naive: dict, dir_baseline: float) -> tuple[str, str]:
    skill = 1 - ens["ret_rmse_pct"] / naive["ret_rmse_pct"] if naive["ret_rmse_pct"] > 0 else 0.0
    beats_rmse = skill > 0
    da = ens.get("dir_acc")
    beats_dir = da is not None and da > dir_baseline
    reason = (f"test RMSE skill vs no-change baseline {skill * 100:+.1f}%, directional accuracy "
              f"{(da or 0) * 100:.1f}% vs {dir_baseline * 100:.1f}% majority-direction baseline")
    if beats_rmse and da is not None and da >= dir_baseline + 0.03:
        return "high", reason
    if beats_rmse or beats_dir:
        return "medium", reason
    return "low", reason


def _signal_backtest(feats: pd.DataFrame, rows: np.ndarray, ens: np.ndarray, confidence: str) -> dict:
    """Replay the decision engine (forecast + technicals, no sentiment) over the test period."""
    signals, hits = [], {"BUY": [], "SELL": []}
    cols = ["close", "rsi14", "sma20", "sma50", "macd_hist", "vol_20"]
    for pos in np.flatnonzero(rows):
        row = feats.iloc[pos]
        ind = {c: (None if pd.isna(row[c]) else float(row[c])) for c in cols}
        d = decide({"ensemble_return": float(ens[pos]), "confidence": confidence}, ind)
        signals.append(d.signal)
        if d.signal in hits:
            nxt = row["target_h1"]
            hits[d.signal].append(nxt > 0 if d.signal == "BUY" else nxt < 0)
    counts = pd.Series(signals).value_counts().to_dict()
    return {
        "counts": {k: int(v) for k, v in counts.items()},
        "buy_hit_rate": float(np.mean(hits["BUY"])) if hits["BUY"] else None,
        "sell_hit_rate": float(np.mean(hits["SELL"])) if hits["SELL"] else None,
    }


def train_ticker(ticker: str, models: list[str] | None = None, epochs: int | None = None,
                 refit: bool = True, log=print) -> dict:
    ticker = ticker.upper()
    models = models or list(config.ML_MODELS)
    from .validate import validate_ticker
    quality = validate_ticker(ticker)
    if quality["status"] == "INVALID":
        bad = next(c for c in quality["checks"] if c["status"] == "INVALID")
        summary = {"ticker": ticker, "trained_at": datetime.now(config.LAGOS).isoformat(timespec="seconds"),
                   "status": "INVALID_DATA", "reason": f"{bad['check']}: {bad['detail']}", "data_quality": quality}
        _write_summary(ticker, summary)
        log(f"[{ticker}] skipped: {summary['reason']}")
        return summary
    hist = load_history(ticker)
    coverage = coverage_report(hist)
    feats = build_features(hist)
    usable = feature_mask(feats) & target_mask(feats) if len(feats) else pd.Series(dtype=bool)
    n = int(usable.sum())
    summary = {"ticker": ticker, "trained_at": datetime.now(config.LAGOS).isoformat(timespec="seconds"),
               "coverage": coverage, "usable_rows": n, "data_quality": quality["status"]}
    if n < config.MIN_TRAIN_ROWS + config.MIN_TEST_ROWS:
        summary["status"] = "INSUFFICIENT_DATA"
        summary["reason"] = (f"{n} usable trading days; need at least "
                             f"{config.MIN_TRAIN_ROWS + config.MIN_TEST_ROWS}")
        _write_summary(ticker, summary)
        log(f"[{ticker}] skipped: {summary['reason']}")
        return summary

    usable_dates = feats.index[usable.to_numpy()]
    n_test = max(config.MIN_TEST_ROWS, int(round(n * config.TEST_FRACTION)))
    split_date = usable_dates[-n_test]
    train_mask = usable & (feats.index < split_date)
    test_mask = usable & (feats.index >= split_date)

    preds, errors = {}, {}
    for name in ["NAIVE"] + models:
        t0 = time.time()
        try:
            m = _make(name, epochs).fit(feats, train_mask)
            preds[name] = m.predict(feats, rows=test_mask.to_numpy())
            log(f"[{ticker}] {name:<5} fitted on {int(train_mask.sum())} days in {time.time() - t0:.1f}s")
        except ModelUnavailable as exc:
            errors[name] = str(exc)
            log(f"[{ticker}] {name:<5} skipped: {exc}")
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
            log(f"[{ticker}] {name:<5} failed: {errors[name]}")

    ml_names = [m for m in models if m in preds]
    if not ml_names:
        summary.update(status="FAILED", reason="no ML model could be trained", errors=errors)
        _write_summary(ticker, summary)
        return summary

    ens = ensemble_mean([preds[m] for m in ml_names])
    preds["ENSEMBLE"] = ens

    rows = test_mask.to_numpy() & np.all([np.isfinite(p).all(axis=1) for p in preds.values()], axis=0)
    close_t = feats["close"].to_numpy(float)[rows]
    actual = feats["target_h1"].to_numpy(float)[rows]
    metrics = {name: compute_metrics(actual, p[rows, 0], close_t) for name, p in preds.items()}

    moved = actual[np.abs(actual) > 1e-9]
    up_share = float(np.mean(moved > 0)) if len(moved) else 0.5
    dir_baseline = max(up_share, 1 - up_share)
    for name, m in metrics.items():
        m["rmse_skill_vs_naive_pct"] = (1 - m["ret_rmse_pct"] / metrics["NAIVE"]["ret_rmse_pct"]) * 100 \
            if metrics["NAIVE"]["ret_rmse_pct"] > 0 else 0.0
    confidence, reason = confidence_from(metrics["ENSEMBLE"], metrics["NAIVE"], dir_baseline)
    best = min(ml_names, key=lambda k: metrics[k]["ret_rmse_pct"])

    # --- every horizon (1..HORIZON trading days ahead) ---
    by_h = {name: [] for name in preds}
    band = []
    lo_q, hi_q = config.BAND_QUANTILES
    for h in range(1, config.HORIZON + 1):
        act_h = feats[f"target_h{h}"].to_numpy(float)[rows]
        naive_rmse = None
        for name in ["NAIVE"] + [k for k in preds if k != "NAIVE"]:
            mh = compute_metrics(act_h, preds[name][rows, h - 1], close_t)
            if name == "NAIVE":
                naive_rmse = mh["ret_rmse_pct"]
            mh["h"] = h
            mh["rmse_skill_vs_naive_pct"] = (1 - mh["ret_rmse_pct"] / naive_rmse) * 100 if naive_rmse else 0.0
            by_h[name].append(mh)
        err = act_h - ens[rows, h - 1]
        band.append({"h": h, "q_low": float(np.quantile(err, lo_q)), "q_high": float(np.quantile(err, hi_q)),
                     "ensemble_mape_pct": by_h["ENSEMBLE"][-1]["price_mape_pct"],
                     "naive_mape_pct": by_h["NAIVE"][-1]["price_mape_pct"],
                     "ensemble_dir_acc": by_h["ENSEMBLE"][-1]["dir_acc"],
                     "ensemble_skill_pct": by_h["ENSEMBLE"][-1]["rmse_skill_vs_naive_pct"]})

    summary.update(
        status="OK",
        split_date=split_date.date().isoformat(),
        train_days=int(train_mask.sum()),
        test_days=int(rows.sum()),
        test_start=feats.index[rows][0].date().isoformat(),
        test_end=feats.index[rows][-1].date().isoformat(),
        actual_share_down=float(np.mean(actual < 0)),
        actual_share_flat=float(np.mean(np.abs(actual) <= 1e-9)),
        dir_baseline=dir_baseline,
        horizon=config.HORIZON,
        metrics=metrics,
        metrics_by_horizon=by_h,
        forecast_band=band,
        best_model=best,
        ensemble_members=ml_names,
        confidence=confidence,
        confidence_reason=reason,
        signal_backtest=_signal_backtest(feats, rows, ens[:, 0], confidence),
        errors=errors,
    )

    if refit:
        all_mask = usable
        for name in ["NAIVE"] + ml_names:
            try:
                m = _make(name, epochs).fit(feats, all_mask)
                m.save(model_dir(ticker, name))
            except Exception as exc:
                summary["errors"][name] = f"refit failed: {type(exc).__name__}: {exc}"
                if name in summary["ensemble_members"]:
                    summary["ensemble_members"].remove(name)
        summary["saved_models"] = list(summary["ensemble_members"])
        if not summary["saved_models"]:
            summary["status"] = "FAILED"
    _write_summary(ticker, summary)
    return summary


def _make(name: str, epochs: int | None):
    cls = MODEL_CLASSES[name]
    if name == "LSTM" and epochs:
        return cls(epochs=epochs)
    return cls()


def _write_summary(ticker: str, summary: dict) -> None:
    out = config.MODELS_DIR / ticker
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(_clean(summary), indent=2, default=_json_default))


def _clean(obj):
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    raise TypeError(type(o))


def load_summary(ticker: str) -> dict | None:
    path = config.MODELS_DIR / ticker.upper() / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else None


def metrics_table(summaries: list[dict]) -> pd.DataFrame:
    rows = []
    for s in summaries:
        for model, m in (s.get("metrics") or {}).items():
            rows.append({"ticker": s["ticker"], "model": model, "test_days": m["n"],
                         "price_MAE": round(m["price_mae"], 4), "price_RMSE": round(m["price_rmse"], 4),
                         "MAPE_%": round(m["price_mape_pct"], 3),
                         "dir_acc_%": None if m["dir_acc"] is None else round(m["dir_acc"] * 100, 1),
                         "skill_vs_naive_%": round(m["rmse_skill_vs_naive_pct"], 2),
                         "pred_down_%": round(m["share_pred_down"] * 100, 1)})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train and evaluate NGX forecasting models")
    ap.add_argument("--tickers", help="comma-separated, default: every ticker found in data/")
    ap.add_argument("--models", default=",".join(config.ML_MODELS), help="subset of LSTM,RF,SVR,ARIMA")
    ap.add_argument("--epochs", type=int, default=None, help="LSTM max epochs")
    ap.add_argument("--no-refit", action="store_true", help="evaluate only, do not save models")
    args = ap.parse_args(argv)

    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else discover_tickers()
    if not tickers:
        print(f"No tickers found. Put CSVs in {config.RAW_DIR}.")
        return 1
    models = [m.strip().upper() for m in args.models.split(",") if m.strip()]
    summaries = [train_ticker(t, models, args.epochs, refit=not args.no_refit) for t in tickers]

    table = metrics_table(summaries)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not table.empty:
        table.to_csv(config.MODELS_DIR / "metrics_summary.csv", index=False)
        with pd.option_context("display.max_rows", 500, "display.width", 160):
            print("\nTest-period results (lower MAE/RMSE/MAPE is better; skill > 0 beats no-change):")
            print(table.to_string(index=False))
        ml = table[table["model"].isin(models)]
        if not ml.empty:
            wins = ml.loc[ml.groupby("ticker")["price_RMSE"].idxmin(), "model"].value_counts()
            print("\nBest model per ticker (lowest RMSE):", ", ".join(f"{k} {v}" for k, v in wins.items()))
    for s in summaries:
        if s.get("status") == "OK":
            bt = s["signal_backtest"]["counts"]
            print(f"{s['ticker']}: confidence {s['confidence']}, test-period signals {bt}")
        else:
            print(f"{s['ticker']}: {s.get('status')} - {s.get('reason', s.get('errors'))}")
    print(f"\nSaved to {config.MODELS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
