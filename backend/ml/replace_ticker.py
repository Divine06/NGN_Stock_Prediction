"""Remove a stock from the whole system and (optionally) put another one in its place.

Run from the project root:

    python -m backend.ml.replace_ticker SEPLAT OANDO --name "Oando"
    python -m backend.ml.replace_ticker SEPLAT --remove-only
    python -m backend.ml.replace_ticker SEPLAT OANDO --name "Oando" --dry-run   # show what would happen

What it does (everything is backed up first, nothing is permanently deleted):
  1. copies ngx_live.db and ngx.db to backend/data/backups/<date-time>/
  2. ngx_live.db: deletes the old ticker's price observations and predictions
  3. ngx.db: deletes the old company's StockPrice rows (and any linked indicator,
     prediction and sentiment rows) and its Company row; adds the new company
     (so the Investo updater can store prices for it)
  4. moves the old ticker's files (raw CSV, trained models, and the old experiment
     files in features/, model_data/, processed/, predictions/) to
     backend/data/removed/<TICKER>-<date-time>/
  5. writes backend/data/universe.json with the new company in the old one's place
     (same sector unless --sector is given); the API, collector and dashboard use it

Afterwards: import the new stock's history, check it, and train (printed at the end).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from contextlib import closing
from datetime import datetime
from pathlib import Path

from . import config

LEGACY_PATTERNS = [
    "features/{t}_features.csv", "model_data/{t}_train.csv", "model_data/{t}_test.csv", "processed/{t}.csv",
    "predictions/arima/{t}_arima_predictions.csv", "predictions/lstm/{t}_lstm.keras",
    "predictions/lstm/{t}_lstm_predictions.csv", "predictions/random_forest/{t}_feature_importance.csv",
    "predictions/random_forest/{t}_random_forest_predictions.csv", "predictions/svr/{t}_svr_predictions.csv",
]


def _tables(con) -> set[str]:
    return {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def clean_live_db(ticker: str, dry: bool) -> dict:
    out = {}
    if not config.DB_PATH.exists():
        return out
    with closing(sqlite3.connect(config.DB_PATH)) as con:
        for table in ("price_observations", "predictions"):
            if table in _tables(con):
                out[table] = con.execute(f"SELECT COUNT(*) FROM {table} WHERE UPPER(ticker)=?", (ticker,)).fetchone()[0]
                if not dry:
                    con.execute(f"DELETE FROM {table} WHERE UPPER(ticker)=?", (ticker,))
        con.commit()
    return out


def clean_legacy_db(old: str, new: str | None, name: str | None, sector: str | None, dry: bool) -> dict:
    db = config.DATA_DIR / "ngx.db"
    out = {}
    if not db.exists():
        return out
    with closing(sqlite3.connect(db)) as con:
        tables = _tables(con)
        if "Company" not in tables:
            return out
        row = con.execute("SELECT company_id FROM Company WHERE UPPER(ticker)=?", (old,)).fetchone()
        if row:
            cid = row[0]
            if "StockPrice" in tables:
                out["StockPrice"] = con.execute("SELECT COUNT(*) FROM StockPrice WHERE company_id=?", (cid,)).fetchone()[0]
                if not dry and "TechnicalIndicator" in tables:
                    con.execute("DELETE FROM TechnicalIndicator WHERE price_id IN "
                                "(SELECT price_id FROM StockPrice WHERE company_id=?)", (cid,))
                if not dry:
                    con.execute("DELETE FROM StockPrice WHERE company_id=?", (cid,))
            for t in ("Prediction", "SentimentRecord"):
                if t in tables and not dry:
                    con.execute(f"DELETE FROM {t} WHERE company_id=?", (cid,))
            if not dry:
                con.execute("DELETE FROM Company WHERE company_id=?", (cid,))
            out["Company"] = 1
        if new and not con.execute("SELECT 1 FROM Company WHERE UPPER(ticker)=?", (new,)).fetchone():
            out["added_company"] = new
            if not dry:
                con.execute("INSERT INTO Company (name, ticker, sector) VALUES (?, ?, ?)", (name, new, sector or ""))
        con.commit()
    return out


def move_files(ticker: str, dest: Path, dry: bool) -> list[str]:
    candidates = [config.RAW_DIR / f"{ticker}.csv", config.HISTORY_DIR / f"{ticker}.csv",
                  config.MODELS_DIR / ticker]
    candidates += [config.DATA_DIR / p.format(t=ticker) for p in LEGACY_PATTERNS]
    candidates += list(config.RAW_DIR.glob(f"{ticker}.csv.bak-*")) if config.RAW_DIR.exists() else []
    moved = []
    for src in candidates:
        if not src.exists():
            continue
        rel = src.relative_to(config.DATA_DIR) if config.DATA_DIR in src.parents else Path(src.name)
        moved.append(str(rel))
        if not dry:
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(target))
    return moved


def write_universe(old: str, new: str | None, name: str | None, sector: str | None,
                   news: str | None, dry: bool) -> list[str]:
    companies = {}
    for t, (n, sec) in config.UNIVERSE.items():
        if t == old:
            if new:
                companies[new] = {"name": name, "sector": sector or sec,
                                  "news_query": news or f'"{name}"'}
            continue
        companies[t] = {"name": n, "sector": sec}
        if t in config.NEWS_QUERIES:
            companies[t]["news_query"] = config.NEWS_QUERIES[t]
    if not dry:
        config.UNIVERSE_FILE.parent.mkdir(parents=True, exist_ok=True)
        config.UNIVERSE_FILE.write_text(json.dumps({"companies": companies}, indent=2, ensure_ascii=False),
                                        encoding="utf-8")
    return list(companies)


def replace(old: str, new: str | None = None, name: str | None = None, sector: str | None = None,
            news: str | None = None, dry: bool = False) -> dict:
    old = old.upper()
    new = new.upper() if new else None
    if old not in config.UNIVERSE:
        raise SystemExit(f"{old} is not in the current company list: {', '.join(config.UNIVERSE)}")
    if new and new in config.UNIVERSE:
        raise SystemExit(f"{new} is already in the company list")
    if new and not name:
        raise SystemExit("give the new company's name, e.g. --name \"Oando\"")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backups = []
    if not dry:
        bdir = config.DATA_DIR / "backups" / stamp
        for db in (config.DB_PATH, config.DATA_DIR / "ngx.db"):
            if db.exists():
                bdir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(db, bdir / db.name)
                backups.append(str(bdir / db.name))
    result = {
        "removed": old, "added": new, "backups": backups,
        "live_db_rows_deleted": clean_live_db(old, dry),
        "ngx_db": clean_legacy_db(old, new, name, sector or config.SECTORS.get(old), dry),
        "files_moved": move_files(old, config.DATA_DIR / "removed" / f"{old}-{stamp}", dry),
        "universe": write_universe(old, new, name, sector, news, dry),
        "dry_run": dry,
    }
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Remove a stock and optionally replace it with another")
    ap.add_argument("old")
    ap.add_argument("new", nargs="?")
    ap.add_argument("--name", help="new company's name, e.g. \"Oando\"")
    ap.add_argument("--sector", help="defaults to the removed company's sector")
    ap.add_argument("--news", help='Google News search, default: the quoted name, e.g. \'"Oando"\'')
    ap.add_argument("--remove-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if not args.new and not args.remove_only:
        ap.error("give the new ticker (e.g. SEPLAT OANDO --name \"Oando\") or use --remove-only")

    r = replace(args.old, None if args.remove_only else args.new, args.name, args.sector, args.news, args.dry_run)
    print(("DRY RUN, nothing changed\n" if r["dry_run"] else "") + f"Removed {r['removed']}"
          + (f", added {r['added']}" if r["added"] else ""))
    for b in r["backups"]:
        print("  backup:", b)
    print("  ngx_live.db rows:", r["live_db_rows_deleted"] or "none")
    print("  ngx.db:", r["ngx_db"] or "nothing to change")
    print(f"  files moved to data/removed/: {len(r['files_moved'])}")
    for f in r["files_moved"]:
        print("    -", f)
    print("  company list:", ", ".join(r["universe"]))
    if r["added"] and not r["dry_run"]:
        t = r["added"]
        print(f"""
Next:
  1. Download {t}'s daily history (Investing.com, Lagos listing, NGN, from 2014) and import it:
       python -m backend.ml.import_csv {t} path/to/{t}.csv --replace
     (or, if your Investo key has API access: python -m backend.ml.fill_gaps --rebuild {t})
  2. python -m backend.ml.validate
  3. python -m backend.ml.train --tickers {t}
  4. Restart the API (uvicorn backend.api.main_ml:app).""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
