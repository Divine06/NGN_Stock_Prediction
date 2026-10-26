import sqlite3
import unittest
from datetime import datetime
from functools import lru_cache

import numpy as np
import pandas as pd

import _setup
from backend.ml import config, store
from backend.ml.decision import decide, uniformity_warning
from backend.ml.diagnose import diagnose_ticker
from backend.ml.predict import predict_ticker, run_all
from backend.ml.scheduler import next_run
from backend.ml.sentiment import parse_rss, score_headlines
from backend.ml.train import train_ticker

AFTER_CLOSE = datetime.now(config.LAGOS).replace(hour=20, minute=0)
FAST_MODELS = ["RF", "SVR", "ARIMA", "LSTM"]  # LSTM is skipped cleanly if TensorFlow is missing
TRAIN_SET = ["SYNRALLY1", "SYNRALLY2", "SYNUP1", "SYNDOWN1", "SYNFLAT1", "SYNMIX", "SYNSHORT", "SYNGAP",
             "SYNLSE"]


@lru_cache(maxsize=1)
def trained():
    return {t: train_ticker(t, FAST_MODELS, epochs=5, log=lambda *_: None) for t in TRAIN_SET}


# --- the old bug ------------------------------------------------------------------
def test_price_level_model_reproduces_all_sell_on_rally():
    r = diagnose_ticker("SYNRALLY1")
    assert r["test_days_above_train_max_%"] > 50
    assert r["old_model_pred_fall_%"] > 80            # old approach: "SELL" almost every day
    assert r["new_model_pred_fall_%"] < 70            # return-based model does not
    assert r["old_model_pred_fall_%"] - r["actual_fall_%"] > 40


# --- training ---------------------------------------------------------------------
def test_training_outputs_metrics_and_models():
    s = trained()["SYNUP1"]
    assert s["status"] == "OK"
    for m in ["NAIVE", "RF", "SVR", "ARIMA", "ENSEMBLE"]:
        assert m in s["metrics"], m
        assert s["metrics"][m]["n"] == s["test_days"] > 60
    assert s["confidence"] in ("high", "medium", "low")
    assert s["horizon"] == 5 and len(s["forecast_band"]) == 5
    band = s["forecast_band"]
    assert all(b["q_low"] < 0 < b["q_high"] for b in band)
    assert band[-1]["naive_mape_pct"] > band[0]["naive_mape_pct"]   # 5 days ahead is harder than 1
    assert [m["h"] for m in s["metrics_by_horizon"]["ENSEMBLE"]] == [1, 2, 3, 4, 5]
    assert set(s["saved_models"]) >= {"RF", "SVR", "ARIMA"}
    assert (config.MODELS_DIR / "SYNUP1" / "metrics.json").exists()
    if "LSTM" not in s["saved_models"]:
        assert "LSTM" in s["errors"]


def test_rally_ticker_no_longer_predicts_falls_everywhere():
    s = trained()["SYNRALLY1"]
    for m in ["RF", "SVR", "ARIMA", "ENSEMBLE"]:
        assert s["metrics"][m]["share_pred_down"] < 0.7, (m, s["metrics"][m]["share_pred_down"])
    assert s["signal_backtest"]["counts"].get("SELL", 0) < 0.5 * s["test_days"]


def test_short_history_is_refused():
    assert trained()["SYNSHORT"]["status"] == "INSUFFICIENT_DATA"


# --- prediction -------------------------------------------------------------------
def test_prediction_record_is_complete_and_sane():
    trained()
    rec = predict_ticker("SYNUP1", use_sentiment=False, now=AFTER_CLOSE)
    assert rec["signal"] in ("BUY", "HOLD", "SELL")
    assert abs(rec["ensemble_return"]) <= config.MAX_DAILY_MOVE
    assert rec["data_source"] == "live_feed"
    assert rec["as_of_date"] == _setup.DATASET["live_days"][-1].date().isoformat()
    assert {"RF", "SVR", "ARIMA"} <= set(rec["models"])
    assert any("Ensemble forecast" in e for e in rec["evidence"])
    assert np.isclose(rec["ensemble_price"], rec["last_close"] * np.exp(rec["ensemble_return"]), rtol=1e-4)
    path = rec["forecast_path"]
    assert [p["h"] for p in path] == [1, 2, 3, 4, 5]
    assert np.isclose(path[0]["price"], rec["ensemble_price"], rtol=1e-6)
    dates = [p["date"] for p in path]
    assert dates == sorted(dates) and all(pd.Timestamp(d).weekday() < 5 for d in dates)
    assert all(p["low"] <= p["price"] <= p["high"] for p in path)
    assert (path[-1]["high"] - path[-1]["low"]) > (path[0]["high"] - path[0]["low"])   # range widens
    assert all(len(m["path_price"]) == 5 for m in rec["models"].values())
    assert any("5-day outlook" in e for e in rec["evidence"])


def test_gap_is_bridged_with_warning_and_untrained_is_refused():
    trained()
    gap = predict_ticker("SYNGAP", use_sentiment=False, now=AFTER_CLOSE)
    assert gap["signal"] in ("BUY", "HOLD", "SELL")
    assert gap["data_warning"] and gap["confidence"] == "low" and "data gap" in gap["evidence"][0]
    untrained = predict_ticker("SYNFLAT2", use_sentiment=False, now=AFTER_CLOSE)
    assert untrained["signal"] == "INSUFFICIENT_DATA" and "not trained" in untrained["evidence"][0]


def test_wrong_exchange_history_is_rejected():
    from backend.ml.validate import validate_ticker
    v = validate_ticker("SYNLSE")
    assert v["status"] == "INVALID"
    assert {c["check"]: c["status"] for c in v["checks"]}["scale"] == "INVALID"
    assert validate_ticker("SYNUP1")["status"] != "INVALID"
    assert train_ticker("SYNLSE", ["RF"], log=lambda *_: None)["status"] == "INVALID_DATA"
    p = predict_ticker("SYNLSE", use_sentiment=False, now=AFTER_CLOSE)
    assert p["signal"] == "INSUFFICIENT_DATA" and "Data quality" in p["evidence"][0]


def test_stale_feed_is_refused():
    trained()
    later = AFTER_CLOSE.replace(year=AFTER_CLOSE.year + 1)
    rec = predict_ticker("SYNUP1", use_sentiment=False, now=later)
    assert rec["signal"] == "INSUFFICIENT_DATA" and "days old" in rec["evidence"][0]


def test_run_all_saves_and_scores_realised_outcomes():
    trained()
    out = run_all(TRAIN_SET, use_sentiment=False, save=True, now=AFTER_CLOSE)
    assert len(out["predictions"]) == len(TRAIN_SET)
    latest = {r["ticker"]: r for r in store.latest_predictions()}
    assert latest["SYNUP1"]["signal"] in ("BUY", "HOLD", "SELL")

    # next day's close arrives through the live feed -> outcome is filled in
    rec = latest["SYNUP1"]
    nxt = _setup.DATASET["live_days"][-1] + np.timedelta64(1, "D")
    while nxt.weekday() >= 5:
        nxt += np.timedelta64(1, "D")
    con = sqlite3.connect(config.DB_PATH)
    con.execute("INSERT INTO price_observations (ticker, observed_at, price, volume, source) VALUES (?,?,?,?,?)",
                ("SYNUP1", f"{nxt.date()} 13:20:00", rec["last_close"] * 1.02, 1000, "Kobo Terminal"))
    con.commit(); con.close()
    store.update_realised("SYNUP1", {nxt.date().isoformat(): rec["last_close"] * 1.02})
    acc = store.live_accuracy()
    row = [r for r in acc["per_ticker"] if r["ticker"] == "SYNUP1"][0]
    assert row["n"] == 1 and abs(row["mae"]) < 0.2


# --- decision engine --------------------------------------------------------------
IND_BULL = {"close": 110, "sma20": 105, "sma50": 100, "rsi14": 55, "macd_hist": 0.4, "vol_20": 0.02}
IND_BEAR = {"close": 90, "sma20": 95, "sma50": 100, "rsi14": 45, "macd_hist": -0.4, "vol_20": 0.02}


def test_decision_rules():
    assert decide(None, IND_BULL).signal == "INSUFFICIENT_DATA"
    assert decide({"ensemble_return": 0.01, "confidence": "high"}, IND_BULL).signal == "BUY"
    assert decide({"ensemble_return": -0.01, "confidence": "high"}, IND_BEAR).signal == "SELL"
    mixed = decide({"ensemble_return": 0.0005, "confidence": "low"}, {**IND_BEAR, "rsi14": 25, "macd_hist": 0.1})
    assert mixed.signal == "HOLD"
    # indicators bullish but the models expect a fall -> the ML gate turns BUY into HOLD
    gated = decide({"ensemble_return": -0.0003, "confidence": "low"}, IND_BULL, {"score": 0.6, "n_headlines": 9})
    assert gated.signal == "HOLD" and "withheld" in gated.evidence[-1]


def test_sentiment_needs_enough_headlines():
    few = decide({"ensemble_return": 0.0, "confidence": "low"}, IND_BULL, {"score": 0.6, "n_headlines": 2})
    assert "sentiment" not in few.votes
    many = decide({"ensemble_return": 0.0, "confidence": "low"}, IND_BULL, {"score": -0.6, "n_headlines": 8})
    assert many.votes["sentiment"] == -1.0


def test_uniform_signals_are_flagged():
    assert uniformity_warning(["SELL"] * 10)
    assert uniformity_warning(["SELL"] * 9 + ["HOLD"]) is None


# --- sentiment parsing ------------------------------------------------------------
RSS = """<?xml version="1.0"?><rss><channel>
<item><title>Dangote Cement posts record profit - BusinessDay</title><source>BusinessDay</source>
<pubDate>{d}</pubDate><link>http://a</link></item>
<item><title>Dangote Cement shares slump on weak outlook - Nairametrics</title><source>Nairametrics</source>
<pubDate>{d}</pubDate><link>http://b</link></item>
<item><title>Old story - X</title><source>X</source><pubDate>Mon, 01 Jan 2018 10:00:00 GMT</pubDate></item>
</channel></rss>"""


def test_rss_parsing_and_scoring():
    from email.utils import format_datetime
    from datetime import timezone
    import backend.ml.sentiment as sm
    items = parse_rss(RSS.format(d=format_datetime(datetime.now(timezone.utc))))
    assert items[0]["title"] == "Dangote Cement posts record profit"
    try:
        sm._get_analyzer()
    except RuntimeError:  # VADER not installed here: use a stand-in with the same interface
        class _Fake:
            def polarity_scores(self, text):
                return {"compound": 0.6 if "record" in text else -0.4}
        sm._analyzer = _Fake()
    res = score_headlines(items)
    assert res["n_headlines"] == 2            # the 2018 story is outside the 7-day window
    assert -1 <= res["score"] <= 1


# --- scheduler & API --------------------------------------------------------------
def test_scheduler_skips_weekends():
    fri_evening = datetime(2026, 10, 23, 16, 0, tzinfo=config.LAGOS)
    assert next_run(fri_evening).strftime("%a %H:%M") == "Mon 15:10"
    mon_morning = datetime(2026, 10, 26, 9, 0, tzinfo=config.LAGOS)
    assert next_run(mon_morning).strftime("%a %d %H:%M") == "Mon 26 15:10"


def test_ml_router():
    import asyncio
    import json
    from _standins import install_fastapi_standin
    trained()
    run_all(TRAIN_SET, use_sentiment=False, save=True, now=AFTER_CLOSE)
    install_fastapi_standin()
    import backend.ml.routes as r

    def ok(payload):
        json.dumps(payload, allow_nan=False)  # what FastAPI's JSON response requires
        return payload

    assert ok(r.model_metrics())["metrics"]
    assert ok(r.ticker_model_metrics("SYNUP1"))["status"] == "OK"
    assert ok(r.data_coverage("SYNGAP"))["gaps_over_limit"]
    assert ok(r.list_predictions())["count"] >= 6
    ok(r.prediction_accuracy())
    one = ok(asyncio.run(r.ticker_prediction("synup1", live=False, sentiment=False, history=5)))
    assert one["current"]["ticker"] == "SYNUP1"
    try:
        r.ticker_model_metrics("NOPE")
        raise AssertionError("expected 404")
    except r.HTTPException as exc:
        assert exc.status_code == 404
