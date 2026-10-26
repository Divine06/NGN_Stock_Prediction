
from pathlib import Path
import pandas as pd

ROOT = Path("backend/data")

FEATURE_DIR = ROOT / "features"
LSTM_DIR = ROOT / "predictions" / "lstm"
RF_DIR = ROOT / "predictions" / "random_forest"

COMPANIES = [
    "AIRTELAFRI",
    "BUACEMENT",
    "DANGCEM",
    "GTCO",
    "MTNN",
    "NESTLE",
    "SEPLAT",
    "TOTAL",
    "UNILEVER",
    "ZENITHBANK",
]


def inspect_file(path, label, preferred_columns):
    print(f"\n  {label}: {path}")

    if not path.exists():
        print("    FILE MISSING")
        return

    print(
        "    Last modified:",
        pd.Timestamp(path.stat().st_mtime, unit="s")
    )

    try:
        df = pd.read_csv(path)

        if df.empty:
            print("    FILE IS EMPTY")
            return

        print("    Rows:", len(df))
        print("    Columns:", df.columns.tolist())

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(
                df["timestamp"], errors="coerce"
            )
            df = df.sort_values("timestamp")

            print(
                "    Latest timestamp:",
                df["timestamp"].iloc[-1]
            )

        columns = [
            c for c in preferred_columns
            if c in df.columns
        ]

        if columns:
            print(df[columns].tail(3).to_string(index=False))
        else:
            print(df.tail(2).to_string(index=False))

    except Exception as exc:
        print("    READ ERROR:", repr(exc))


for ticker in COMPANIES:
    print("\n" + "=" * 75)
    print("COMPANY:", ticker)

    inspect_file(
        FEATURE_DIR / f"{ticker}_features.csv",
        "FEATURES",
        [
            "timestamp", "close", "sma_20", "sma_50",
            "rsi_14", "macd", "macd_signal"
        ],
    )

    inspect_file(
        LSTM_DIR / f"{ticker}_lstm_predictions.csv",
        "LSTM PREDICTIONS",
        [
            "timestamp", "actual_close", "predicted_close",
            "close", "y_true", "y_pred"
        ],
    )

    inspect_file(
        RF_DIR / f"{ticker}_random_forest_predictions.csv",
        "RANDOM FOREST PREDICTIONS",
        [
            "timestamp", "actual_close", "close",
            "actual_target", "target", "predicted_target",
            "probability_up"
        ],
    )

print("\n" + "=" * 75)
print("DIAGNOSTIC COMPLETE")