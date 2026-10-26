from datetime import datetime

import numpy as np
import pandas as pd

import _setup
from backend.ml import config
from backend.ml.features import FEATURE_COLUMNS, build_features, feature_mask
from backend.ml.history import add_segments, coverage_report, daily_closes_from_db, discover_tickers, load_history, read_price_csv


def test_discovers_all_tickers():
    tickers = discover_tickers()
    assert "SYNRALLY1" in tickers and "SYNGAP" in tickers and len(tickers) == 13


def test_live_observations_become_one_close_per_day():
    live = daily_closes_from_db("SYNUP1", now=datetime.now(config.LAGOS).replace(hour=20))
    assert len(live) == 20
    # last 20-minute snapshot of each day is the close
    assert (live["close"] > 0).all() and live.index.is_monotonic_increasing


def test_unfinished_session_is_excluded():
    last_day = _setup.DATASET["live_days"][-1]
    morning = datetime(last_day.year, last_day.month, last_day.day, 11, 0, tzinfo=config.LAGOS)
    evening = morning.replace(hour=16)
    assert last_day not in daily_closes_from_db("SYNUP1", now=morning).index
    assert last_day in daily_closes_from_db("SYNUP1", now=evening).index


def test_csv_and_live_merge_without_gap():
    hist = load_history("SYNUP1", now=datetime.now(config.LAGOS).replace(hour=20))
    assert set(hist["source"]) == {"ngx_history", "live_feed"}
    assert hist["segment"].nunique() == 1  # continuous series
    assert hist.index.is_unique


def test_gap_between_csv_and_live_is_detected():
    hist = load_history("SYNGAP")
    rep = coverage_report(hist)
    assert hist["segment"].nunique() == 2
    assert rep["gaps_over_limit"] and rep["latest_segment_rows"] <= 20


def test_bonus_issue_jump_starts_new_segment():
    idx = pd.bdate_range("2024-01-01", periods=6)
    df = pd.DataFrame({"close": [10, 10.2, 10.1, 5.0, 5.1, 5.2]}, index=idx)
    assert list(add_segments(df)["segment"]) == [0, 0, 0, 1, 1, 1]


def test_day_first_dates_and_comma_prices(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text('Trade Date,Close Price,Volume\n13/01/2025,"1,250.50",100\n14/01/2025,"1,260.00",200\n')
    df = read_price_csv(p)
    assert df.index[0] == pd.Timestamp("2025-01-13") and df["close"].iloc[0] == 1250.5


def test_features_are_scale_free():
    hist = load_history("SYNRALLY1", include_live=False)
    a = build_features(hist)
    b = build_features(hist.assign(close=hist["close"] * 37.0))
    cols = [c for c in FEATURE_COLUMNS if c != "vol_z20"]
    pd.testing.assert_frame_equal(a[cols], b[cols], check_exact=False, rtol=1e-9, atol=1e-12)


def test_features_complete_after_warmup_and_target_is_next_return():
    f = build_features(load_history("SYNFLAT1", include_live=False))
    assert feature_mask(f).iloc[60:].all()
    nxt = np.log(f["close"].shift(-1) / f["close"])
    assert np.allclose(f["target"].iloc[:-1], nxt.iloc[:-1], equal_nan=True)
