import pandas as pd
import numpy as np


def calculate_indicators(df):
    """
    Calculate technical indicators for stock-price data.

    Required indicators:
    - SMA
    - EMA
    - RSI
    - MACD
    - Bollinger Bands
    - ATR
    """

    df = df.copy()

    # ---------------------------------------------------------
    # Moving Averages
    # ---------------------------------------------------------

    df["sma_20"] = (
        df["close"]
        .rolling(window=20)
        .mean()
    )

    df["sma_50"] = (
        df["close"]
        .rolling(window=50)
        .mean()
    )

    df["ema_20"] = (
        df["close"]
        .ewm(span=20, adjust=False)
        .mean()
    )

    # ---------------------------------------------------------
    # RSI - Relative Strength Index
    # ---------------------------------------------------------

    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    rs = avg_gain / avg_loss

    df["rsi_14"] = 100 - (
        100 / (1 + rs)
    )

    # ---------------------------------------------------------
    # MACD
    # ---------------------------------------------------------

    ema_12 = (
        df["close"]
        .ewm(span=12, adjust=False)
        .mean()
    )

    ema_26 = (
        df["close"]
        .ewm(span=26, adjust=False)
        .mean()
    )

    df["macd"] = ema_12 - ema_26

    df["macd_signal"] = (
        df["macd"]
        .ewm(span=9, adjust=False)
        .mean()
    )

    df["macd_histogram"] = (
        df["macd"] -
        df["macd_signal"]
    )

    # ---------------------------------------------------------
    # Bollinger Bands
    # ---------------------------------------------------------

    middle_band = (
        df["close"]
        .rolling(window=20)
        .mean()
    )

    std_dev = (
        df["close"]
        .rolling(window=20)
        .std()
    )

    df["bb_middle"] = middle_band

    df["bb_upper"] = (
        middle_band + (2 * std_dev)
    )

    df["bb_lower"] = (
        middle_band - (2 * std_dev)
    )

    # ---------------------------------------------------------
    # ATR - Average True Range
    # ---------------------------------------------------------

    previous_close = df["close"].shift(1)

    true_range_1 = (
        df["high"] - df["low"]
    )

    true_range_2 = (
        (df["high"] - previous_close)
        .abs()
    )

    true_range_3 = (
        (df["low"] - previous_close)
        .abs()
    )

    true_range = pd.concat(
        [
            true_range_1,
            true_range_2,
            true_range_3
        ],
        axis=1
    ).max(axis=1)

    df["atr_14"] = (
        true_range
        .rolling(window=14)
        .mean()
    )

    return df