import pandas as pd
from pathlib import Path


INPUT_DIR = Path("backend/data/features")
OUTPUT_DIR = Path("backend/data/model_data")

TRAIN_RATIO = 0.80


FEATURE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "return",
    "price_change",

    # Technical indicators
    "sma_20",
    "sma_50",
    "ema_20",
    "rsi_14",

    "macd",
    "macd_signal",
    "macd_histogram",

    "bb_middle",
    "bb_upper",
    "bb_lower",

    "atr_14"
]


TARGET_COLUMN = "target"


def prepare_company(file_path):

    ticker = file_path.stem.replace(
        "_features",
        ""
    )

    print(f"\nPreparing {ticker}...")

    df = pd.read_csv(file_path)

    # Make sure data is chronologically ordered
    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # Check required columns
    # ---------------------------------------------------------

    required_columns = (
        FEATURE_COLUMNS +
        [TARGET_COLUMN]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        print(
            f"❌ Missing columns: "
            f"{missing_columns}"
        )

        return

    # ---------------------------------------------------------
    # Remove rows with missing model features
    # ---------------------------------------------------------

    df = df.dropna(
        subset=required_columns
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # Time-based train/test split
    # ---------------------------------------------------------

    split_index = int(
        len(df) * TRAIN_RATIO
    )

    train_df = df.iloc[
        :split_index
    ].copy()

    test_df = df.iloc[
        split_index:
    ].copy()

    # ---------------------------------------------------------
    # Create output directory
    # ---------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ---------------------------------------------------------
    # Save training dataset
    # ---------------------------------------------------------

    train_file = (
        OUTPUT_DIR /
        f"{ticker}_train.csv"
    )

    train_df.to_csv(
        train_file,
        index=False
    )

    # ---------------------------------------------------------
    # Save test dataset
    # ---------------------------------------------------------

    test_file = (
        OUTPUT_DIR /
        f"{ticker}_test.csv"
    )

    test_df.to_csv(
        test_file,
        index=False
    )

    # ---------------------------------------------------------
    # Display summary
    # ---------------------------------------------------------

    print(
        f"   Total records: {len(df)}"
    )

    print(
        f"   Training records: "
        f"{len(train_df)}"
    )

    print(
        f"   Testing records: "
        f"{len(test_df)}"
    )

    print(
        f"   Training period: "
        f"{train_df['timestamp'].min().date()} "
        f"→ "
        f"{train_df['timestamp'].max().date()}"
    )

    print(
        f"   Testing period: "
        f"{test_df['timestamp'].min().date()} "
        f"→ "
        f"{test_df['timestamp'].max().date()}"
    )

    print(
        f"   ✅ Saved train/test datasets"
    )


def main():

    print("=" * 70)
    print("NGX MODEL DATA PREPARATION")
    print("=" * 70)

    files = sorted(
        INPUT_DIR.glob(
            "*_features.csv"
        )
    )

    if not files:

        print(
            "❌ No feature datasets found."
        )

        return

    processed = 0

    for file_path in files:

        prepare_company(
            file_path
        )

        processed += 1

    print("\n" + "=" * 70)
    print(
        f"✅ MODEL DATA PREPARATION COMPLETE — "
        f"{processed} companies processed"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()