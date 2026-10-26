"""Checks your historical CSVs for the cause of the all-SELL signals.

    python -m backend.ml.diagnose

For each ticker it splits the history 80/20 in time order (like the earlier
experiments) and reports:
  * how often test-period prices were ABOVE the highest training price
    (or BELOW the lowest, which causes the mirror problem: constant BUY);
  * how often a price-level model (lagged prices, MinMax scaling, Random Forest:
    the old approach) predicts a fall, versus how often the price really fell;
  * the same for the return-based model used in this package;
  * the MAPE of simply predicting "tomorrow = today" (compare with the old 12.36%).

If test prices sit above the training maximum and the price-level model predicts
a fall on most days, that is the all-SELL bug: those models cannot predict a
price higher than anything they were trained on.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

from . import config
from .features import FEATURE_COLUMNS, build_features, feature_mask
from .history import discover_tickers, load_history

LAGS = 10


def diagnose_ticker(ticker: str) -> dict | None:
    hist = load_history(ticker, include_live=False)
    if len(hist) < 300:
        return None
    close = hist["close"].astype(float)
    n = len(close)
    split = int(n * 0.8)
    train_max = close.iloc[:split].max()
    train_min = close.iloc[:split].min()

    # --- old approach: price level in, price level out --------------------------
    X = np.column_stack([close.shift(k).to_numpy() for k in range(LAGS)])
    y = close.shift(-1).to_numpy()
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    idx = np.arange(n)
    tr, te = ok & (idx < split), ok & (idx >= split)
    xs, ys = MinMaxScaler().fit(X[tr]), MinMaxScaler().fit(y[tr].reshape(-1, 1))
    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=config.RANDOM_STATE, n_jobs=-1)
    rf.fit(xs.transform(X[tr]), ys.transform(y[tr].reshape(-1, 1)).ravel())
    pred_price = ys.inverse_transform(rf.predict(xs.transform(X[te])).reshape(-1, 1)).ravel()
    close_t, actual_next = X[te, 0], y[te]
    old_change = pred_price / close_t - 1
    actual_change = actual_next / close_t - 1

    # --- this package: scale-free features in, next-day return out ---------------
    feats = build_features(hist)
    usable = (feature_mask(feats) & feats["target"].notna()).to_numpy()
    split_date = close.index[split]
    ftr = usable & (feats.index < split_date)
    fte = usable & (feats.index >= split_date)
    rf2 = RandomForestRegressor(n_estimators=200, min_samples_leaf=20, max_features=0.5,
                                random_state=config.RANDOM_STATE, n_jobs=-1)
    rf2.fit(feats.loc[ftr, FEATURE_COLUMNS].to_numpy(float), feats.loc[ftr, "target"].to_numpy(float))
    new_pred = rf2.predict(feats.loc[fte, FEATURE_COLUMNS].to_numpy(float)) if fte.any() else np.array([])

    return {
        "ticker": ticker,
        "train_end": close.index[split - 1].date().isoformat(),
        "train_max_price": round(float(train_max), 2),
        "test_max_price": round(float(close.iloc[split:].max()), 2),
        "test_days_above_train_max_%": round(float((close.iloc[split:] > train_max).mean() * 100), 1),
        "test_days_below_train_min_%": round(float((close.iloc[split:] < train_min).mean() * 100), 1),
        "old_model_pred_fall_%": round(float(np.mean(old_change < 0) * 100), 1),
        "old_model_mean_pred_change_%": round(float(np.mean(old_change) * 100), 2),
        "new_model_pred_fall_%": round(float(np.mean(new_pred < 0) * 100), 1) if len(new_pred) else None,
        "actual_fall_%": round(float(np.mean(actual_change < 0) * 100), 1),
        "actual_flat_%": round(float(np.mean(np.abs(actual_change) < 1e-12) * 100), 1),
        "old_model_MAPE_%": round(float(np.mean(np.abs(pred_price - actual_next) / actual_next) * 100), 2),
        "naive_MAPE_%": round(float(np.mean(np.abs(close_t - actual_next) / actual_next) * 100), 2),
    }


def main(argv=None) -> int:
    tickers = discover_tickers()
    rows = [r for r in (diagnose_ticker(t) for t in tickers) if r]
    if not rows:
        print(f"No ticker has 300+ days of CSV history in {config.RAW_DIR}")
        return 1
    df = pd.DataFrame(rows)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.MODELS_DIR / "diagnosis.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(df.to_string(index=False))

    above = df["test_days_above_train_max_%"] > 20
    old_bearish = df["old_model_pred_fall_%"] > df["actual_fall_%"] + 25
    print()
    print(f"{int(above.sum())}/{len(df)} tickers spent >20% of the test period above their training maximum.")
    print(f"{int(old_bearish.sum())}/{len(df)} tickers: the price-level model predicts a fall far more often "
          "than prices actually fell.")
    if (above & old_bearish).sum() >= len(df) / 2:
        print("=> Consistent with the all-SELL bug: price-level models cannot forecast above their training "
              "range, so in a rising market they forecast falls. Predicting returns (this package) removes it.")
    below = df["test_days_below_train_min_%"] > 20
    if below.any():
        print(f"{int(below.sum())}/{len(df)} tickers fell below their training minimum: the same limit makes the "
              "price-level model predict rises there (mean predicted change "
              f"{df.loc[below, 'old_model_mean_pred_change_%'].mean():+.1f}%).")
    worse = df["old_model_MAPE_%"] > df["naive_MAPE_%"]
    print(f"{int(worse.sum())}/{len(df)} tickers: the price-level model's MAPE is worse than 'tomorrow = today' "
          f"(median naive MAPE {df['naive_MAPE_%'].median():.2f}%).")
    print(f"Saved {config.MODELS_DIR / 'diagnosis.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
