import asyncio
import json
import sqlite3
import sys
import types
from datetime import datetime

import _setup
from _standins import install_fastapi_standin
from backend.ml import config, dashboard, live_feed, sentiment
from backend.ml.predict import run_all
from test_pipeline import AFTER_CLOSE, TRAIN_SET, trained


def ok(payload):
    json.dumps(payload, allow_nan=False)  # FastAPI rejects NaN/inf in JSON
    return payload


def _prepare():
    trained()
    sentiment.set_provider(lambda t: {"score": 0.2, "n_headlines": 5, "headlines": []})
    run_all(TRAIN_SET, use_sentiment=True, save=True, now=AFTER_CLOSE)
    sentiment.set_provider(None)
    dashboard.clear_cache()
    live_feed.STATE["stocks"] = []


# --- live feed --------------------------------------------------------------------
def test_market_hours_and_poll_window():
    L = config.LAGOS
    assert live_feed.market_status(datetime(2026, 10, 26, 11, 0, tzinfo=L))["status"] == "open"
    assert live_feed.market_status(datetime(2026, 10, 26, 9, 45, tzinfo=L))["status"] == "pre_open"
    assert live_feed.market_status(datetime(2026, 10, 26, 14, 40, tzinfo=L))["status"] == "closed"
    assert live_feed.market_status(datetime(2026, 10, 24, 11, 0, tzinfo=L))["status"] == "closed"  # Saturday
    assert live_feed.should_poll(datetime(2026, 10, 26, 15, 5, tzinfo=L))
    assert not live_feed.should_poll(datetime(2026, 10, 26, 15, 10, tzinfo=L))
    assert not live_feed.should_poll(datetime(2026, 10, 24, 12, 0, tzinfo=L))


def test_collect_once_keeps_universe_and_stores_observations():
    import backend
    fake = types.ModuleType("backend.market_data")
    fake.fetch_all_stocks = lambda force_refresh=False: {
        "stocks": [{"symbol": "SYNFLAT2", "current_price": "123.40", "change_percent": 1.2, "volume": 1000},
                   {"symbol": "NOTOURS", "current_price": 5.0}],
        "fetched_at": "2026-10-22T11:00:00+01:00", "cached": False}
    old_mod, old_attr = sys.modules.get("backend.market_data"), getattr(backend, "market_data", None)
    sys.modules["backend.market_data"], backend.market_data = fake, fake
    try:
        res = live_feed.collect_once(True)
        assert [s["ticker"] for s in res["stocks"]] == ["SYNFLAT2"]
        assert res["stocks"][0]["price"] == 123.4 and res["stocks"][0]["sector"] is None
        con = sqlite3.connect(config.DB_PATH)
        n = con.execute("SELECT COUNT(*) FROM price_observations WHERE ticker='SYNFLAT2' AND price=123.4").fetchone()[0]
        con.execute("DELETE FROM price_observations WHERE ticker='SYNFLAT2' AND price=123.4")
        con.commit(); con.close()
        assert n == 1
    finally:
        if old_mod is None:
            sys.modules.pop("backend.market_data", None)
        else:
            sys.modules["backend.market_data"] = old_mod
        backend.market_data = old_attr
        live_feed.STATE["stocks"] = []


# --- dashboard payloads -------------------------------------------------------------
def test_decisions_payload():
    _prepare()
    d = ok(dashboard.decisions_payload())
    assert d["count"] == len(_setup.TEST_TICKERS) and d["warning"] is None
    recs = {x["ticker"]: x for x in d["decisions"]}
    up = recs["SYNUP1"]
    assert up["recommendation"] in ("BUY", "HOLD", "SELL")
    assert isinstance(up["reasons"], str) and "Ensemble forecast" in up["reasons"]   # main.py list style
    assert up["predicted_price"] > 0 and abs(up["predicted_change_pct"]) < 10.5
    assert {"RF", "SVR", "ARIMA"} <= set(up["models"]) and up["sentiment_label"] == "Positive"
    assert up["headline_count"] == 5 and up["rsi_14"] is not None
    assert up["horizon_days"] == 5 and len(up["forecast_path"]) == 5 and up["horizon_price"] > 0
    assert recs["SYNLSE"]["recommendation"] == "INSUFFICIENT_DATA"
    assert recs["SYNFLAT2"]["recommendation"] == "INSUFFICIENT_DATA"   # never trained
    single = ok(dashboard.decision_payload("SYNUP1"))
    assert isinstance(single["reasons"], list)                         # main.py single style


def test_stock_detail_and_history():
    _prepare()
    sd = ok(dashboard.stock_detail("SYNUP1"))
    assert set(dashboard.INDICATOR_KEYS) <= set(sd["indicators"])
    assert 0 <= sd["indicators"]["rsi_14"] <= 100 and sd["price"] > 0
    assert sd["price_source"] == "Kobo Terminal" and sd["prediction"]["ticker"] == "SYNUP1"
    y1 = ok(dashboard.history_payload("SYNUP1", "1y"))
    assert 240 <= y1["count"] <= 270 and "sma_20" in y1["data"][-1] and y1["data"][-1]["source"] == "Live feed"
    assert y1["data"][0]["open"] is not None
    allh = dashboard.history_payload("SYNUP1", "all")
    assert allh["count"] >= 2000 and allh["first_date"] < y1["first_date"]
    d1 = ok(dashboard.history_payload("SYNDOWN1", "1d"))
    assert d1["count"] == 14 and d1["data"][-1]["timestamp"].endswith("+01:00")
    assert dashboard.history_payload("SYNGAP", "all")["gaps"]
    bad = ok(dashboard.stock_detail("SYNLSE"))                 # wrong-exchange history is not drawn
    assert bad["data_quality"] == "INVALID" and "data-quality" in bad["indicator_note"]
    bad_hist = dashboard.history_payload("SYNLSE", "all")
    assert bad_hist["count"] <= 20 and bad_hist["note"]
    gap_detail = dashboard.stock_detail("SYNGAP")
    assert gap_detail["indicator_note"] and "data gap" in gap_detail["indicator_note"]


def test_companies_market_health_quality():
    _prepare()
    assert ok(dashboard.companies_payload())["count"] == len(_setup.TEST_TICKERS)
    md = ok(dashboard.market_data_payload())
    assert md["count"] >= 12 and all(s["price"] > 0 for s in md["stocks"])
    h = ok(dashboard.health_payload())
    assert "SYNUP1" in h["models_trained_for"] and h["predictions_as_of"]
    q = ok(dashboard.data_quality_payload())
    assert q["invalid"] == ["SYNLSE"]


def test_sentiment_and_news_payloads():
    orig_s, orig_n = dashboard.get_sentiment, dashboard.get_news
    dashboard.get_sentiment = lambda t: {"score": -0.31, "n_headlines": 4, "window_days": 7, "headlines": [
        {"title": "Profit warning", "link": "http://x", "source": "BusinessDay", "published": None,
         "compound": -0.5}]}
    dashboard.get_news = lambda t: {"articles": [
        {"title": f"h{i}", "url": "u", "link": "u", "source": "s", "published": None,
         "sentiment": "Positive" if i % 2 else "Negative", "score": 0.4 if i % 2 else -0.4, "compound": 0}
        for i in range(5)]}
    try:
        s = ok(dashboard.sentiment_payload("SYNUP1"))
        assert s["sentiment_label"] == "Negative" and s["headline_count"] == 4 and s["used_in_decision"]
        assert s["headlines"][0]["sentiment"]["label"] == "Negative"
        n = ok(dashboard.news_payload("SYNUP1", 3))
        assert n["article_count"] == 3 and n["sentiment"]["method"] == "VADER"
    finally:
        dashboard.get_sentiment, dashboard.get_news = orig_s, orig_n


# --- unified API ------------------------------------------------------------------
def test_main_ml_routes_and_lifespan():
    import importlib
    _prepare()
    install_fastapi_standin()
    m = importlib.import_module("backend.api.main_ml")
    for fn, args in [(m.root, ()), (m.health, ()), (m.companies, ()), (m.market_status, ()),
                     (m.market_data, ()), (m.stock, ("synup1",)), (m.stock_history, ("SYNUP1", "3m")),
                     (m.decisions, ()), (m.decision, ("SYNUP1",)), (m.advisory, ("SYNUP1",)),
                     (m.data_quality, ())]:
        ok(fn(*args))
    try:
        m.stock("NOPE")
        raise AssertionError("expected 404")
    except m.HTTPException as exc:
        assert exc.status_code == 404

    # start-up and shutdown: Kobo is unreachable here (no key), which must not crash the app
    import backend
    fake = types.ModuleType("backend.market_data")

    def boom(force_refresh=False):
        raise RuntimeError("KOBO_API_KEY is missing")
    fake.fetch_all_stocks = boom
    old = getattr(backend, "market_data", None)
    sys.modules["backend.market_data"], backend.market_data = fake, fake
    sentiment.set_provider(lambda t: {"score": None, "n_headlines": 0, "headlines": []})
    orig_age = live_feed.latest_observation_age_minutes
    live_feed.latest_observation_age_minutes = lambda now=None: 600.0  # force the start-up Kobo call
    try:
        async def run():
            async with m.lifespan(m.app):
                await asyncio.sleep(0.2)
        asyncio.run(run())
        assert "KOBO_API_KEY" in (live_feed.STATE["error"] or "")
        assert live_feed.STATE["stocks"]  # fell back to stored observations
    finally:
        live_feed.latest_observation_age_minutes = orig_age
        sentiment.set_provider(None)
        backend.market_data = old
        sys.modules.pop("backend.market_data", None)


# --- gap filler ---------------------------------------------------------------------
def test_find_gaps_and_rebuild_with_fake_investo():
    import pandas as pd
    from backend.ml import fill_gaps
    today = _setup.DATASET["today"].date()
    gaps = fill_gaps.find_gaps("SYNGAP", today=today)
    assert any((pd.Timestamp(b) - pd.Timestamp(a)).days > 200 for a, b in gaps)   # the CSV -> live hole
    assert fill_gaps.find_gaps("SYNUP1", today=today)[-1][1] == today.isoformat()   # tail to today

    calls = {}
    hu = types.ModuleType("backend.data.historical_updater")
    hu.fetch_history = lambda t, a, b: [{"date": "2026-01-05", "open": 1, "high": 1, "low": 1, "close": 250.0,
                                         "volume": 10}, {"date": "2026-01-06", "open": 1, "high": 1, "low": 1,
                                                         "close": 255.0, "volume": 10}]
    hu.normalise_history = lambda rows: pd.DataFrame(
        [{"timestamp": r["date"], "open": r["open"], "high": r["high"], "low": r["low"], "close": r["close"],
          "volume": r["volume"]} for r in rows])
    hu.update_database = lambda t: calls.setdefault("db", t)
    import backend.data as bd
    old = getattr(bd, "historical_updater", None)
    sys.modules["backend.data.historical_updater"], bd.historical_updater = hu, hu
    raw = config.RAW_DIR / "SYNSHORT.csv"
    original = raw.read_text()
    try:
        fill_gaps.rebuild("SYNSHORT", "2026-01-01")
        rebuilt = pd.read_csv(raw)
        assert list(rebuilt["close"]) == [250.0, 255.0]
        assert any(p.name.startswith("SYNSHORT.csv.bak-") for p in config.RAW_DIR.iterdir())
    finally:
        raw.write_text(original)
        for p in config.RAW_DIR.glob("SYNSHORT.csv.bak-*"):
            p.unlink()
        sys.modules.pop("backend.data.historical_updater", None)
        bd.historical_updater = old


def test_ngnmarket_fallback_when_kobo_fails():
    import os
    calls = []

    def fake_get(url, params, headers):
        calls.append((url, params, headers))
        return {"data": [{"symbol": "SYNFLAT2", "price": 77.5, "price_change_percent": -0.5, "volume": 1200,
                          "last_updated": "2026-10-22T10:20:00Z"}],
                "pagination": {"page": 1, "limit": 200, "total": 1, "total_pages": 1}}

    def kobo_down():
        raise RuntimeError("Kobo daily limit reached (429)")

    orig_get, orig_kobo = live_feed._http_get, live_feed.FETCHERS["kobo"]
    live_feed._http_get = fake_get
    live_feed.FETCHERS["kobo"] = (kobo_down, "Kobo Terminal")
    os.environ["NGN_MARKET_API_KEY"] = "ngm_test"
    try:
        res = live_feed.collect_once(True)
        assert [s["ticker"] for s in res["stocks"]] == ["SYNFLAT2"]
        rec = res["stocks"][0]
        assert rec["source"] == "NGN Market" and rec["price"] == 77.5 and rec["feed_timestamp"].startswith("2026-10-22")
        assert res["sources"] == ["NGN Market"] and "429" in res["errors"][0]
        assert calls[0][2]["Authorization"] == "Bearer ngm_test" and calls[0][1]["limit"] == 200
        con = sqlite3.connect(config.DB_PATH)
        src = con.execute("SELECT source FROM price_observations WHERE ticker='SYNFLAT2' AND price=77.5").fetchone()
        con.execute("DELETE FROM price_observations WHERE ticker='SYNFLAT2' AND price=77.5")
        con.commit(); con.close()
        assert src == ("NGN Market",)
    finally:
        live_feed._http_get, live_feed.FETCHERS["kobo"] = orig_get, orig_kobo
        os.environ.pop("NGN_MARKET_API_KEY", None)
        live_feed.STATE["stocks"] = []
