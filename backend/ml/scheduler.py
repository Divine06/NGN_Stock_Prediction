"""Runs the prediction pipeline once per trading day after the market closes.

Started by backend/api/main_ml.py. Needs a running asyncio loop.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, time, timedelta

from . import config

log = logging.getLogger("ml.scheduler")
_task: asyncio.Task | None = None


def next_run(now: datetime, run_at: str = config.DAILY_RUN_TIME) -> datetime:
    at = time.fromisoformat(run_at)
    candidate = now.replace(hour=at.hour, minute=at.minute, second=0, microsecond=0)
    if candidate <= now:
        candidate += timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate


async def _loop(use_sentiment: bool) -> None:
    from .predict import run_all
    while True:
        now = datetime.now(config.LAGOS)
        target = next_run(now)
        await asyncio.sleep(max(1.0, (target - now).total_seconds()))
        try:
            result = await asyncio.to_thread(run_all, None, use_sentiment, True)
            from . import dashboard
            dashboard.clear_cache()
            log.info("Daily predictions done: %s", result["signal_counts"])
            if result.get("warning"):
                log.warning(result["warning"])
        except Exception:  # keep the loop alive
            log.exception("Daily prediction run failed")


def start_daily_predictions(use_sentiment: bool = True) -> asyncio.Task:
    global _task
    if _task is None or _task.done():
        _task = asyncio.get_running_loop().create_task(_loop(use_sentiment))
    return _task


def stop_daily_predictions() -> None:
    if _task is not None:
        _task.cancel()
