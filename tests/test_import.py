import numpy as np
import pandas as pd

import _setup
from backend.ml import config
from backend.ml.import_csv import import_file, parse_dates, parse_number, read_download
from backend.ml.validate import validate_ticker


def _investing_csv(path, dates, closes, date_fmt="%b %d, %Y"):
    """Mimics an Investing.com 'Historical Data' export (newest first, K/M volumes, quoted numbers)."""
    rows = []
    for d, c in sorted(zip(dates, closes), reverse=True):
        rows.append({"Date": d.strftime(date_fmt), "Price": f"{c:,.2f}", "Open": f"{c:,.2f}",
                     "High": f"{c * 1.01:,.2f}", "Low": f"{c * 0.99:,.2f}", "Vol.": "635.54K", "Change %": "0.00%"})
    pd.DataFrame(rows).to_csv(path, index=False)


def _keep(*tickers):
    return {t: (config.RAW_DIR / f"{t}.csv").read_text() for t in tickers}


def _restore(saved):
    for t, text in saved.items():
        (config.RAW_DIR / f"{t}.csv").write_text(text)
        for p in config.RAW_DIR.glob(f"{t}.csv.bak-*"):
            p.unlink()


def test_number_and_date_parsing():
    assert parse_number("1,066.70") == 1066.7 and parse_number("635.54K") == 635540
    assert parse_number("2.48M") == 2480000 and np.isnan(parse_number("-"))
    days = pd.bdate_range("2026-08-03", periods=60)
    dmy = parse_dates(pd.Series(days.strftime("%d/%m/%Y")))
    mdy = parse_dates(pd.Series(days.strftime("%m/%d/%Y")))
    assert list(dmy) == list(days) and list(mdy) == list(days)       # day/month order detected
    assert list(parse_dates(pd.Series(days.strftime("%b %d, %Y")))) == list(days)


def test_import_fills_recent_gap(tmp_path):
    saved = _keep("SYNDOWN2")
    try:
        hist = pd.read_csv(config.RAW_DIR / "SYNDOWN2.csv")
        last_close = float(hist["close"].iloc[-1])
        days = pd.bdate_range(end=_setup.DATASET["live_days"][-1], periods=45)   # overlaps the CSV end
        closes = list(hist["close"].iloc[-25:]) + [last_close] * 20
        f = tmp_path / "SYNDOWN2 Historical Data.csv"
        _investing_csv(f, days, closes)
        assert len(read_download(f)) == 45
        r = import_file("SYNDOWN2", f)
        assert r["imported"] == 45 and not r["blocked"] and r["backup"]
        merged = pd.read_csv(config.RAW_DIR / "SYNDOWN2.csv")
        assert merged["timestamp"].is_unique and merged["timestamp"].iloc[-1] == days[-1].strftime("%Y-%m-%d")
    finally:
        _restore(saved)


def test_wrong_listing_is_blocked_and_replace_fixes_bad_history(tmp_path):
    saved = _keep("SYNUP2", "SYNLSE")
    try:
        # pence-scale file for an NGX stock -> blocked
        h = pd.read_csv(config.RAW_DIR / "SYNUP2.csv")
        days = pd.to_datetime(h["timestamp"].iloc[-60:])
        f = tmp_path / "SYNUP2.csv"
        _investing_csv(f, days, list(h["close"].iloc[-60:] / 50))
        r = import_file("SYNUP2", f)
        assert r["imported"] == 0 and r["blocked"]

        # SYNLSE history is the wrong exchange: a correct NGX file plus --replace repairs it
        assert validate_ticker("SYNLSE")["status"] == "INVALID"
        bad = pd.read_csv(config.RAW_DIR / "SYNLSE.csv")
        days = pd.to_datetime(bad["timestamp"])
        f2 = tmp_path / "SYNLSE.csv"
        _investing_csv(f2, days, list(bad["close"] * 50), date_fmt="%d/%m/%Y")
        assert import_file("SYNLSE", f2)["blocked"]                 # merge refused: prices disagree
        r2 = import_file("SYNLSE", f2, replace=True)
        assert r2["imported"] == len(bad) and not r2["blocked"]
        checks = {c["check"]: c["status"] for c in validate_ticker("SYNLSE")["checks"]}
        assert checks["scale"] == "OK"
    finally:
        _restore(saved)
