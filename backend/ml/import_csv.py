"""Import a daily price file you downloaded (e.g. Investing.com "Historical Data", Lagos
listing, currency NGN) into data/raw/{TICKER}.csv, the history the models and charts use.

    python -m backend.ml.import_csv DANGCEM "C:/Users/you/Downloads/DANGCEM Historical Data.csv"
    python -m backend.ml.import_csv --folder "C:/Users/you/Downloads/ngx"   # files named after tickers
    python -m backend.ml.import_csv SEPLAT seplat_ngx.csv --replace          # throw away the old history

Accepts the usual layouts: Date, Price/Close, Open, High, Low, Vol., Change %; dates like
"Oct 16, 2026", "16/10/2026" or "10/16/2026" (the day/month order is detected); volumes like
"635.54K" or "1.2M". A copy of a web table pasted into Excel and saved as CSV works too.

Safety checks before anything is written:
  * the file's prices must match the live Kobo price and the existing history on overlapping
    dates (within 25%), so a London-listed (pence) file cannot be merged by mistake;
  * day-to-day moves beyond the NGX +/-10% band are counted and reported.
The old file is backed up as raw/{TICKER}.csv.bak-YYYYMMDD. Imported rows replace existing rows
on the same dates; --replace discards the old history completely.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .history import (CLOSE_CANDIDATES, DATE_CANDIDATES, HIGH_CANDIDATES, LOW_CANDIDATES, OPEN_CANDIDATES,
                      VOLUME_CANDIDATES, _find_col, read_price_csv)
from .validate import latest_live_price

DATE_FORMATS = [{"format": "%b %d, %Y"}, {"format": "%B %d, %Y"}, {"format": "%d/%m/%Y"}, {"format": "%m/%d/%Y"},
                {"format": "%Y-%m-%d"}, {"format": "%d-%m-%Y"}, {"format": "%m-%d-%Y"}, {"format": "%d.%m.%Y"},
                {"format": "%d-%b-%Y"}, {"format": "%d %b %Y"}]
SUFFIX = {"K": 1e3, "M": 1e6, "B": 1e9}


def parse_number(value) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    s = str(value).strip().replace(",", "").replace("₦", "").replace("NGN", "").replace("%", "").strip()
    if s in ("", "-", "—", "nan", "None"):
        return np.nan
    mult = 1.0
    if s[-1].upper() in SUFFIX:
        mult, s = SUFFIX[s[-1].upper()], s[:-1]
    try:
        return float(s) * mult
    except ValueError:
        return np.nan


def parse_dates(series: pd.Series, today: date | None = None) -> pd.Series:
    """Try each format; keep the one that parses everything with the fewest weekend/future dates."""
    today = pd.Timestamp(today or date.today())
    s = series.astype(str).str.strip()
    best = None
    for kw in DATE_FORMATS:
        d = pd.to_datetime(s, errors="coerce", **kw)
        ok = d.notna().mean()
        if ok < 0.98:
            continue
        dd = d.dropna()
        score = (round(ok, 3), -float((dd.dt.dayofweek >= 5).mean()), -float((dd > today).mean()), dd.is_unique)
        if best is None or score > best[0]:
            best = (score, d)
    if best is None:
        raise ValueError(f"could not read the dates (first values: {list(s.head(3))})")
    return best[1].dt.normalize()


def read_download(path: Path, today: date | None = None) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    date_col = _find_col(raw, DATE_CANDIDATES)
    close_col = _find_col(raw, CLOSE_CANDIDATES)
    if date_col is None or close_col is None:
        raise ValueError(f"{path.name}: need a Date and a Price/Close column, found {list(raw.columns)}")
    out = pd.DataFrame({"timestamp": parse_dates(raw[date_col], today),
                        "close": raw[close_col].map(parse_number)})
    for name, cands in (("open", OPEN_CANDIDATES), ("high", HIGH_CANDIDATES), ("low", LOW_CANDIDATES),
                        ("volume", VOLUME_CANDIDATES)):
        col = _find_col(raw, cands)
        out[name] = raw[col].map(parse_number) if col else np.nan
    out = out.dropna(subset=["timestamp", "close"])
    out = out[(out["close"] > 0) & (out["timestamp"].dt.dayofweek < 5)]
    out = out.drop_duplicates("timestamp", keep="first").sort_values("timestamp")
    for c in ("open", "high", "low"):
        out[c] = out[c].fillna(out["close"])
    out["volume"] = out["volume"].fillna(0).round().astype("int64")
    return out[["timestamp", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def check(ticker: str, new: pd.DataFrame, existing: pd.DataFrame | None) -> list[str]:
    """Returns blocking problems (empty list = safe to import)."""
    problems = []
    live, _ = latest_live_price(ticker)
    last = float(new["close"].iloc[-1])
    if live and not (0.8 <= live / last <= 1.25) and (pd.Timestamp.today() - new["timestamp"].iloc[-1]).days < 30:
        problems.append(f"last price in the file ({last:,.2f}) is {live / last:.2f}x the live Kobo price "
                        f"({live:,.2f}); is this the Lagos (NGN) listing?")
    if existing is not None and not existing.empty:
        joined = new.set_index("timestamp")["close"].to_frame("new").join(existing["close"].rename("old"), how="inner")
        if len(joined) >= 5:
            ratio = float((joined["new"] / joined["old"]).median())
            if not 0.8 <= ratio <= 1.25:
                problems.append(f"on {len(joined)} overlapping dates the file's prices are {ratio:.2f}x your "
                                "existing history. If the existing history is the wrong one (e.g. SEPLAT), "
                                "re-run with --replace")
    return problems


def band_breaches(df: pd.DataFrame) -> int:
    gap = df["timestamp"].diff().dt.days
    move = df["close"].pct_change().abs()
    return int(((move > config.DISCONTINUITY_MOVE) & (gap <= 4)).sum())


def import_file(ticker: str, path: Path, replace: bool = False, force: bool = False,
                today: date | None = None) -> dict:
    ticker = ticker.upper()
    new = read_download(Path(path), today)
    if new.empty:
        raise ValueError(f"{path}: no usable rows")
    target = config.RAW_DIR / f"{ticker}.csv"
    existing = read_price_csv(target) if target.exists() else None
    problems = check(ticker, new, None if replace else existing)
    if problems and not force:
        return {"ticker": ticker, "imported": 0, "blocked": problems}

    if existing is not None and not replace:
        old = existing.reset_index().rename(columns={"date": "timestamp"})[
            ["timestamp", "open", "high", "low", "close", "volume"]]
        old = old[~old["timestamp"].isin(new["timestamp"])]
        merged = pd.concat([old, new]).sort_values("timestamp")
    else:
        merged = new
    merged = merged.drop_duplicates("timestamp", keep="last")
    merged["open"] = merged["open"].fillna(merged["close"])
    merged["high"] = merged["high"].fillna(merged["close"])
    merged["low"] = merged["low"].fillna(merged["close"])
    merged["volume"] = merged["volume"].fillna(0)

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    backup = None
    if target.exists():
        backup = target.with_name(f"{ticker}.csv.bak-{date.today():%Y%m%d}")
        shutil.copy2(target, backup)
    out = merged.copy()
    out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%d")
    out.to_csv(target, index=False)
    return {"ticker": ticker, "imported": len(new), "total_rows": len(merged),
            "file_range": f"{new['timestamp'].iloc[0].date()} to {new['timestamp'].iloc[-1].date()}",
            "band_breaches_in_file": band_breaches(new), "backup": backup.name if backup else None,
            "blocked": problems if force else []}


def _ticker_from_name(name: str, universe: list[str]) -> str | None:
    up = name.upper()
    hits = [t for t in universe if t in up]
    return max(hits, key=len) if hits else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Import downloaded NGX daily prices into data/raw")
    ap.add_argument("ticker", nargs="?")
    ap.add_argument("path", nargs="?")
    ap.add_argument("--folder", help="import every CSV in a folder; file names must contain the ticker")
    ap.add_argument("--replace", action="store_true", help="discard the existing history for this ticker")
    ap.add_argument("--force", action="store_true", help="import even if the safety checks fail")
    args = ap.parse_args(argv)

    jobs = []
    if args.folder:
        universe = config.configured_tickers() or list(config.UNIVERSE)
        for p in sorted(Path(args.folder).glob("*.csv")):
            t = _ticker_from_name(p.stem, universe)
            if t:
                jobs.append((t, p))
            else:
                print(f"skipped {p.name}: no ticker in the file name")
    elif args.ticker and args.path:
        jobs.append((args.ticker, Path(args.path)))
    else:
        ap.error("give TICKER PATH or --folder")

    for t, p in jobs:
        try:
            r = import_file(t, p, replace=args.replace, force=args.force)
        except Exception as exc:
            print(f"{t}: FAILED - {exc}")
            continue
        if r["imported"] == 0:
            print(f"{t}: NOT imported - " + "; ".join(r["blocked"]))
        else:
            print(f"{t}: {r['imported']} rows ({r['file_range']}), history now {r['total_rows']} rows, "
                  f"{r['band_breaches_in_file']} moves beyond the NGX band, backup: {r['backup']}")
    print("\nNext: python -m backend.ml.validate, then python -m backend.ml.train")
    return 0


if __name__ == "__main__":
    sys.exit(main())
