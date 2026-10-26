import pandas as pd

from .technical_indicators import calculate_indicators


def main():

    file_path = "backend/data/processed/DANGCEM.csv"

    df = pd.read_csv(file_path)

    print("=" * 70)
    print("TECHNICAL INDICATOR TEST")
    print("=" * 70)

    print("\nOriginal columns:")
    print(df.columns.tolist())

    result = calculate_indicators(df)

    print("\nIndicator columns:")
    print(result.columns.tolist())

    print("\nLatest indicator values:")
    print(
        result[
            [
                "timestamp",
                "close",
                "sma_20",
                "sma_50",
                "ema_20",
                "rsi_14",
                "macd",
                "macd_signal",
                "bb_upper",
                "bb_middle",
                "bb_lower",
                "atr_14"
            ]
        ].tail(10).to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("✅ INDICATOR TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()