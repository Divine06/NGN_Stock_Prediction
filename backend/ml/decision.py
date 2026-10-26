"""Decision engine: ML forecast + technical indicators + news sentiment -> signal.

Each component is turned into a vote between -1 (bearish) and +1 (bullish):

  forecast   tanh(ensemble_return / (0.25 x 20-day volatility)), multiplied by
             the model confidence (high 1.0, medium 0.6, low 0.3) from the test period
  technical  mean of: RSI (<30 +1, >70 -1, else 0), MACD histogram sign,
             close vs SMA50, SMA20 vs SMA50
  sentiment  mean VADER compound of recent headlines / 0.5, clipped to [-1, 1];
             left out when there are fewer than MIN_HEADLINES headlines

score = weighted mean of the available votes (weights 0.5 / 0.3 / 0.2)
BUY if score >= +0.25, SELL if score <= -0.25, otherwise HOLD.
Gate: a BUY needs a forecast >= 0 and a SELL a forecast <= 0, otherwise HOLD.
INSUFFICIENT_DATA when there is no usable forecast.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from . import config

SIGNALS = ("BUY", "HOLD", "SELL", "INSUFFICIENT_DATA")


@dataclass
class Decision:
    signal: str
    score: float | None
    votes: dict = field(default_factory=dict)
    weights_used: dict = field(default_factory=dict)
    evidence: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _fmt_pct(x: float) -> str:
    return f"{x * 100:+.2f}%"


def forecast_vote(pred_return: float, vol20: float | None, confidence: str) -> tuple[float, str]:
    scale = max((vol20 or 0.0) * config.FORECAST_SCALE_VOL, 0.002)
    raw = math.tanh(pred_return / scale)
    mult = config.CONFIDENCE_MULTIPLIER.get(confidence, config.CONFIDENCE_MULTIPLIER["low"])
    return raw * mult, f"forecast vote {raw:+.2f} x {confidence} confidence ({mult})"


def technical_vote(ind: dict) -> tuple[float | None, list[str]]:
    votes, notes = [], []
    rsi = ind.get("rsi14")
    if rsi is not None:
        if rsi < 30:
            votes.append(1.0); notes.append(f"RSI {rsi:.1f}: oversold (bullish)")
        elif rsi > 70:
            votes.append(-1.0); notes.append(f"RSI {rsi:.1f}: overbought (bearish)")
        else:
            votes.append(0.0); notes.append(f"RSI {rsi:.1f}: neutral")
    close, sma20, sma50 = ind.get("close"), ind.get("sma20"), ind.get("sma50")
    hist = ind.get("macd_hist")
    # Many NGX stocks do not trade for days, so indicators are often exactly equal.
    # A tie (within NEUTRAL_BAND of the price) is neutral, not bearish.
    if hist is not None and close:
        rel = hist / close
        v = 0.0 if abs(rel) < config.NEUTRAL_BAND / 10 else (1.0 if rel > 0 else -1.0)
        votes.append(v)
        notes.append("MACD histogram flat (neutral)" if v == 0 else
                     f"MACD histogram {'above' if v > 0 else 'below'} zero ({'bullish' if v > 0 else 'bearish'})")
    for label, a, b in (("Price", close, sma50), ("20-day SMA", sma20, sma50)):
        if a is None or not b:
            continue
        rel = a / b - 1
        v = 0.0 if abs(rel) < config.NEUTRAL_BAND else (1.0 if rel > 0 else -1.0)
        votes.append(v)
        notes.append(f"{label} level with 50-day SMA (neutral)" if v == 0 else
                     f"{label} {'above' if v > 0 else 'below'} 50-day SMA ({'bullish' if v > 0 else 'bearish'})")
    if not votes:
        return None, ["Technical indicators unavailable"]
    return sum(votes) / len(votes), notes


def sentiment_vote(sentiment: dict | None) -> tuple[float | None, str]:
    if not sentiment or sentiment.get("score") is None:
        return None, "News sentiment unavailable: left out of the score"
    n = int(sentiment.get("n_headlines", 0))
    if n < config.MIN_HEADLINES:
        return None, f"Only {n} recent headline(s): sentiment left out of the score"
    s = float(sentiment["score"])
    vote = max(-1.0, min(1.0, s / 0.5))
    mood = "bullish" if s > 0.05 else "bearish" if s < -0.05 else "neutral"
    return vote, f"News sentiment {s:+.2f} (VADER) from {n} headlines ({mood})"


def decide(forecast: dict | None, indicators: dict, sentiment: dict | None = None) -> Decision:
    """forecast = {"ensemble_return", "confidence", "model_returns": {name: r}} or None."""
    if not forecast or forecast.get("ensemble_return") is None:
        reason = (forecast or {}).get("reason", "No trained model output for this ticker")
        return Decision("INSUFFICIENT_DATA", None, evidence=[reason])

    evidence, votes = [], {}
    r = float(forecast["ensemble_return"])
    conf = forecast.get("confidence", "low")
    fv, fnote = forecast_vote(r, indicators.get("vol_20"), conf)
    votes["forecast"] = round(fv, 4)

    parts = [f"{k} {_fmt_pct(v)}" for k, v in (forecast.get("model_returns") or {}).items() if v is not None]
    agree = forecast.get("agreement")
    evidence.append(f"Ensemble forecast {_fmt_pct(r)} for the next session"
                    + (f" ({', '.join(parts)})" if parts else "")
                    + (f"; {agree}" if agree else ""))
    evidence.append(f"Model confidence {conf}: {forecast.get('confidence_reason', 'see model metrics')}")
    if forecast.get("clamped"):
        evidence.append("Raw forecast exceeded the NGX +/-10% daily band and was capped; check the data")

    tv, tnotes = technical_vote(indicators)
    if tv is not None:
        votes["technical"] = round(tv, 4)
    evidence.extend(tnotes)

    sv, snote = sentiment_vote(sentiment)
    if sv is not None:
        votes["sentiment"] = round(sv, 4)
    evidence.append(snote)

    weights = {k: config.WEIGHTS[k] for k in votes}
    score = sum(votes[k] * weights[k] for k in votes) / sum(weights.values())
    if score >= config.BUY_THRESHOLD:
        signal = "BUY"
    elif score <= config.SELL_THRESHOLD:
        signal = "SELL"
    else:
        signal = "HOLD"
    evidence.append(f"Weighted score {score:+.2f} (BUY >= {config.BUY_THRESHOLD:+.2f}, "
                    f"SELL <= {config.SELL_THRESHOLD:+.2f}); {fnote}")
    # The ML forecast is the gate: indicators and news can strengthen or weaken a signal,
    # but cannot produce a BUY when the models expect a fall (or a SELL when they expect a rise).
    if (signal == "BUY" and r < 0) or (signal == "SELL" and r > 0):
        evidence.append(f"{signal} withheld: the ML forecast ({_fmt_pct(r)}) points the other way, so HOLD")
        signal = "HOLD"
    return Decision(signal, round(score, 4), votes, weights, evidence)


def uniformity_warning(signals: list[str]) -> str | None:
    """Flag the failure mode seen before: every company getting the same BUY/SELL."""
    actionable = [s for s in signals if s in ("BUY", "SELL")]
    if len(signals) >= 5 and len(actionable) == len(signals) and len(set(actionable)) == 1:
        return (f"All {len(signals)} tickers returned {actionable[0]}. Check the data feed, "
                "feature scaling and model metrics before trusting these signals.")
    return None
