"""Run the automatic pipeline without the web server.

    python -m backend.ml.collect            # fetch current prices once (ignores market hours)
    python -m backend.ml.collect --loop     # keep running: prices every 20 min in market hours,
                                            # predictions every weekday at 15:10 WAT

The API (uvicorn backend.api.main_ml:app) already does the same work while it is running, so use
this when you want collection to continue with the dashboard closed: leave it running in a terminal,
or start it with Windows Task Scheduler at log-on. It only works while the computer is awake.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime

from . import config, live_feed

log = logging.getLogger("ml.collect")


def run_loop(use_sentiment: bool = True) -> None:
    from .predict import run_all
    from .scheduler import next_run
    live_feed.init_db()
    next_poll = 0.0
    next_predict = next_run(datetime.now(config.LAGOS))
    log.info("Collector started; next prediction run %s", next_predict.isoformat())
    while True:
        now = datetime.now(config.LAGOS)
        if live_feed.should_poll(now) and time.time() >= next_poll:
            try:
                res = live_feed.collect_once(True)
                log.info("Stored %d prices from %s", len(res["stocks"]), ", ".join(res["sources"]))
            except Exception as exc:
                log.warning("Price fetch failed: %s", exc)
            next_poll = time.time() + config.POLL_SECONDS
        if now >= next_predict:
            try:
                out = run_all(None, use_sentiment, True)
                log.info("Predictions: %s", out["signal_counts"])
            except Exception:
                log.exception("Prediction run failed")
            next_predict = next_run(now)
        time.sleep(30)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Collect free NGX prices and run predictions")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--no-sentiment", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.loop:
        run_loop(not args.no_sentiment)
        return 0
    live_feed.init_db()
    res = live_feed.collect_once(True)
    for s in res["stocks"]:
        print(f"{s['ticker']:<11} N{s['price']:>12,.2f}  {s['source']}")
    for e in res["errors"]:
        print("warning:", e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
