"""Feature engineering.

Every feature is SCALE-FREE (a return, a ratio or an oscillator), never a raw
naira price. That is what lets the models work when prices move into a range
they never saw in training, which is what produced the all-SELL output before.

Targets: target_h{h} = ln(close[t+h] / close[t]) for h = 1..HORIZON trading days
(the total change from today's close to h sessions ahead). `target` = target_h1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

FEATURE_COLUMNS = [
    "r_0", "r_1", "r_2", "r_3", "r_4",          # last five daily log returns
    "r_5d", "r_10d", "r_20d",                    # multi-day momentum
    "vol_10", "vol_20",                          # rolling volatility of returns
    "sma20_gap", "sma50_gap", "ema20_gap", "sma20_50_gap",
    "rsi_c",                                     # RSI14 centred: (RSI-50)/50
    "macd_n", "macd_sig_n", "macd_hist_n",       # MACD / close
    "bb_pctb",                                   # Bollinger %B centred
    "atr_proxy",                                 # mean |return| over 14 days
    "flat_frac_20",                              # share of no-change days (NGX illiquidity)
    "vol_z20",                                   # volume z-score (0 when volume missing)
]

TARGET_COLUMNS = [f"target_h{h}" for h in range(1, config.HORIZON + 1)]

INDICATOR_COLUMNS = ["rsi14", "sma20", "sma50", "ema20", "macd", "macd_signal", "macd_hist",
                     "bb_upper", "bb_lower", "vol_20"]


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    out = out.where(loss != 0, np.where(gain > 0, 100.0, 50.0))  # no losses at all
    return out.where(gain.notna())


def _segment_features(seg: pd.DataFrame) -> pd.DataFrame:
    close = seg["close"].astype(float)
    logret = np.log(close).diff()
    f = pd.DataFrame(index=seg.index)
    f["close"] = close
    f["logret"] = logret
    f["target"] = logret.shift(-1)
    for h in range(1, config.HORIZON + 1):
        f[f"target_h{h}"] = np.log(close.shift(-h) / close)

    for k in range(5):
        f[f"r_{k}"] = logret.shift(k)
    for k in (5, 10, 20):
        f[f"r_{k}d"] = np.log(close / close.shift(k))
    f["vol_10"] = logret.rolling(10).std()
    f["vol_20"] = logret.rolling(20).std()

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    ema20 = close.ewm(span=20, adjust=False, min_periods=20).mean()
    f["sma20"], f["sma50"], f["ema20"] = sma20, sma50, ema20
    f["sma20_gap"] = close / sma20 - 1
    f["sma50_gap"] = close / sma50 - 1
    f["ema20_gap"] = close / ema20 - 1
    f["sma20_50_gap"] = sma20 / sma50 - 1

    f["rsi14"] = rsi(close)
    f["rsi_c"] = (f["rsi14"] - 50) / 50

    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    f["macd"], f["macd_signal"], f["macd_hist"] = macd, signal, macd - signal
    f["macd_n"], f["macd_sig_n"], f["macd_hist_n"] = macd / close, signal / close, (macd - signal) / close

    std20 = close.rolling(20).std()
    f["bb_upper"], f["bb_lower"] = sma20 + 2 * std20, sma20 - 2 * std20
    width = (f["bb_upper"] - f["bb_lower"]).replace(0, np.nan)
    f["bb_pctb"] = ((close - f["bb_lower"]) / width).fillna(0.5).where(std20.notna()) - 0.5

    f["atr_proxy"] = logret.abs().rolling(14).mean()
    f["flat_frac_20"] = (logret.abs() < 1e-9).astype(float).where(logret.notna()).rolling(20).mean()

    vol = seg["volume"].astype(float) if "volume" in seg else pd.Series(np.nan, index=seg.index)
    lv = np.log1p(vol)
    z = (lv - lv.rolling(20, min_periods=10).mean()) / lv.rolling(20, min_periods=10).std()
    f["vol_z20"] = z.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
    return f


def build_features(history: pd.DataFrame) -> pd.DataFrame:
    """Features for every row; rows without enough look-back keep NaNs."""
    if history.empty:
        return pd.DataFrame(columns=["close", "logret", "target", "segment"] + FEATURE_COLUMNS + TARGET_COLUMNS)
    parts = []
    for seg_id, seg in history.groupby("segment", sort=True):
        part = _segment_features(seg)
        part["segment"] = seg_id
        parts.append(part)
    return pd.concat(parts).sort_index()


def feature_mask(frame: pd.DataFrame) -> pd.Series:
    return frame[FEATURE_COLUMNS].notna().all(axis=1)


def target_mask(frame: pd.DataFrame) -> pd.Series:
    """Rows where every horizon's outcome is known (needed for multi-horizon training/testing)."""
    return frame[TARGET_COLUMNS].notna().all(axis=1)


def latest_indicators(frame: pd.DataFrame) -> dict:
    last = frame.iloc[-1]
    out = {"date": frame.index[-1].date().isoformat(), "close": float(last["close"])}
    for col in INDICATOR_COLUMNS:
        val = last.get(col, np.nan)
        out[col] = None if pd.isna(val) else round(float(val), 6)
    return out
