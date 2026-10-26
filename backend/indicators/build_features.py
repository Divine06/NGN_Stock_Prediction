"""
NGX Stock Feature Builder
-------------------------
Reads historical OHLCV data from backend/data/raw/
and generates technical-indicator datasets in
backend/data/features/.

Supported companies:
AIRTELAFRI, BUACEMENT, DANGCEM, GTCO, MTNN,
NESTLE, SEPLAT, TOTAL, UNILEVER, ZENITHBANK

Important:
- Keeps the newest available historical records.
- Does not impose a fixed end date.
- Preserves valid OHLC rows even when early indicator
  values are NaN because of rolling-window warm-up.
- Sorts dates oldest to newest and removes duplicates.
"""

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
PROJECT_DIR = BACKEND_DIR.parent

RAW_DIR = BACKEND_DIR / "data" / "raw"
FEATURE_DIR = BACKEND_DIR / "data" / "features"


# ============================================================
# COMPANIES
# ============================================================

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


# ============================================================
# COLUMN STANDARDISATION
# ============================================================

COLUMN_ALIASES = {
    "date": "timestamp",
    "datetime": "timestamp",
    "time": "timestamp",
    "timestamp": "timestamp",
    "tradedate": "timestamp",

    "open": "open",
    "openingprice": "open",

    "high": "high",
    "highestprice": "high",

    "low": "low",
    "lowestprice": "low",

    "close": "close",
    "price": "close",
    "lastprice": "close",
    "closingprice": "close",
    "adjclose": "close",
    "adjustedclose": "close",

    "volume": "volume",
    "vol": "volume",
    "vol.": "volume",
    "sharestraded": "volume",
}


def normalise_column_name(column):
    """Convert a column name into a comparable lowercase key."""
    return (
        str(column)
        .strip()
        .lower()
        .replace("_", "")
        .replace(" ", "")
        .replace("-", "")
        .replace("/", "")
        .replace(".", ".")
    )


def standardise_columns(df):
    """Rename recognised raw-data columns to standard names."""
    rename_map = {}

    for column in df.columns:
        key = normalise_column_name(column)

        if key in COLUMN_ALIASES:
            rename_map[column] = COLUMN_ALIASES[key]

    df = df.rename(columns=rename_map)

    # If multiple source columns map to the same target name,
    # keep the first occurrence.
    df = df.loc[:, ~df.columns.duplicated()]

    return df


# ============================================================
# NUMERIC CLEANING
# ============================================================

def clean_numeric(series):
    """
    Convert values such as '1,234.50' or '₦1,234.50'
    to numeric values. Invalid values become NaN.
    """
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(",", "", regex=False)
        .str.replace("₦", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("NGN", "", regex=False)
        .str.replace("%", "", regex=False)
        .replace({
            "": np.nan,
            "nan": np.nan,
            "None": np.nan,
            "null": np.nan,
            "-": np.nan,
            "--": np.nan,
        })
    )

    return pd.to_numeric(cleaned, errors="coerce")


# ============================================================
# LOAD AND CLEAN RAW DATA
# ============================================================

def load_raw_data(ticker):
    """Load, standardise, clean and chronologically sort a stock CSV."""

    raw_file = RAW_DIR / f"{ticker}.csv"

    if not raw_file.exists():
        print(f"  [SKIP] Raw file not found: {raw_file}")
        return None

    print(f"\n  Reading: {raw_file.name}")

    try:
        df = pd.read_csv(raw_file)
    except Exception as error:
        print(f"  [ERROR] Could not read CSV: {error}")
        return None

    if df.empty:
        print("  [SKIP] Raw CSV is empty.")
        return None

    df = standardise_columns(df)

    required = ["timestamp", "open", "high", "low", "close"]
    missing = [column for column in required if column not in df.columns]

    if missing:
        print(f"  [ERROR] Missing required columns: {missing}")
        print(f"  Available columns: {list(df.columns)}")
        return None

    # Parse timestamps without imposing an end-date cutoff.
    # Most project raw files use ISO-style dates.
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        dayfirst=False,
    )

    # If many dates failed to parse, retry using day-first parsing.
    if df["timestamp"].notna().sum() < max(1, int(len(df) * 0.5)):
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
            dayfirst=True,
        )

    # Convert OHLC columns to numbers.
    for column in ["open", "high", "low", "close"]:
        df[column] = clean_numeric(df[column])

    if "volume" not in df.columns:
        df["volume"] = 0
    else:
        df["volume"] = clean_numeric(df["volume"])

    # Invalid volume values should not remove otherwise valid prices.
    df["volume"] = df["volume"].fillna(0)

    # Remove rows that cannot be plotted or used for price analysis.
    df = df.dropna(
        subset=["timestamp", "open", "high", "low", "close"]
    )

    # Remove impossible or non-positive prices.
    df = df[
        (df["open"] > 0)
        & (df["high"] > 0)
        & (df["low"] > 0)
        & (df["close"] > 0)
    ]

    if df.empty:
        print("  [ERROR] No valid OHLC rows remain after cleaning.")
        return None

    # Sort before deduplicating so the final occurrence of each date
    # is retained when the source contains duplicate dates.
    df = df.sort_values("timestamp")
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    df = df.reset_index(drop=True)

    # Keep a consistent, timezone-free timestamp column.
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(
        None
    )

    # Put core columns first, followed by any extra raw columns.
    first_columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    remaining_columns = [
        column for column in df.columns
        if column not in first_columns
    ]

    df = df[first_columns + remaining_columns]

    first_date = df["timestamp"].min().strftime("%Y-%m-%d")
    last_date = df["timestamp"].max().strftime("%Y-%m-%d")

    print(f"  Valid raw records: {len(df):,}")
    print(f"  First date: {first_date}")
    print(f"  Latest date: {last_date}")

    return df


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def calculate_rsi(close, period=14):
    """Calculate RSI using exponentially smoothed gains and losses."""

    delta = close.diff()

    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    average_gain = gains.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    average_loss = losses.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    relative_strength = average_gain / average_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + relative_strength))

    # Handle edge cases without generating infinite values.
    rsi = rsi.mask(
        (average_loss == 0) & (average_gain > 0),
        100,
    )

    rsi = rsi.mask(
        (average_gain == 0) & (average_loss > 0),
        0,
    )

    rsi = rsi.mask(
        (average_gain == 0) & (average_loss == 0),
        50,
    )

    return rsi


def add_technical_indicators(df):
    """Calculate all indicators required by the existing project."""

    df = df.copy()

    close = df["close"]
    high = df["high"]
    low = df["low"]

    # --------------------------------------------------------
    # 1. SIMPLE MOVING AVERAGES
    # --------------------------------------------------------

    df["sma_20"] = close.rolling(
        window=20,
        min_periods=20,
    ).mean()

    df["sma_50"] = close.rolling(
        window=50,
        min_periods=50,
    ).mean()

    # --------------------------------------------------------
    # 2. EXPONENTIAL MOVING AVERAGE
    # --------------------------------------------------------

    df["ema_20"] = close.ewm(
        span=20,
        min_periods=20,
        adjust=False,
    ).mean()

    # --------------------------------------------------------
    # 3. RELATIVE STRENGTH INDEX
    # --------------------------------------------------------

    df["rsi_14"] = calculate_rsi(close, period=14)

    # --------------------------------------------------------
    # 4. MACD
    # --------------------------------------------------------

    ema_12 = close.ewm(
        span=12,
        min_periods=12,
        adjust=False,
    ).mean()

    ema_26 = close.ewm(
        span=26,
        min_periods=26,
        adjust=False,
    ).mean()

    df["macd"] = ema_12 - ema_26

    df["macd_signal"] = df["macd"].ewm(
        span=9,
        min_periods=9,
        adjust=False,
    ).mean()

    df["macd_histogram"] = (
        df["macd"] - df["macd_signal"]
    )

    # --------------------------------------------------------
    # 5. BOLLINGER BANDS
    # --------------------------------------------------------

    df["bb_middle"] = close.rolling(
        window=20,
        min_periods=20,
    ).mean()

    rolling_std = close.rolling(
        window=20,
        min_periods=20,
    ).std(ddof=0)

    df["bb_upper"] = df["bb_middle"] + (2 * rolling_std)
    df["bb_lower"] = df["bb_middle"] - (2 * rolling_std)

    # --------------------------------------------------------
    # 6. AVERAGE TRUE RANGE
    # --------------------------------------------------------

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["atr_14"] = true_range.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False,
    ).mean()

    # Replace infinities with NaN, but DO NOT drop rows merely
    # because an indicator has a warm-up NaN.
    df = df.replace([np.inf, -np.inf], np.nan)

    return df


# ============================================================
# BUILD ONE COMPANY'S FEATURES
# ============================================================

def build_company_features(ticker):
    """Create and save a feature CSV for one company."""

    print("\n" + "=" * 68)
    print(f"BUILDING FEATURES: {ticker}")
    print("=" * 68)

    raw_df = load_raw_data(ticker)

    if raw_df is None or raw_df.empty:
        return False

    features_df = add_technical_indicators(raw_df)

    # Confirm all required dashboard columns exist.
    required_features = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
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
        "atr_14",
    ]

    missing = [
        column for column in required_features
        if column not in features_df.columns
    ]

    if missing:
        print(f"  [ERROR] Missing generated columns: {missing}")
        return False

    # Keep all rows with valid historical prices.
    # This is essential so the chart can display the newest dates.
    features_df = features_df.dropna(
        subset=["timestamp", "open", "high", "low", "close"]
    )

    features_df = features_df.sort_values("timestamp")
    features_df = features_df.drop_duplicates(
        subset=["timestamp"],
        keep="last",
    )
    features_df = features_df.reset_index(drop=True)

    # Save timestamp as YYYY-MM-DD, compatible with the API.
    features_df["timestamp"] = pd.to_datetime(
        features_df["timestamp"]
    ).dt.strftime("%Y-%m-%d")

    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    output_file = FEATURE_DIR / f"{ticker}_features.csv"

    try:
        features_df.to_csv(output_file, index=False)
    except Exception as error:
        print(f"  [ERROR] Could not save features: {error}")
        return False

    # Verify the saved file itself.
    try:
        check_df = pd.read_csv(output_file)

        saved_dates = pd.to_datetime(
            check_df["timestamp"],
            errors="coerce",
        ).dropna()

        if saved_dates.empty:
            print("  [ERROR] Saved file contains no valid dates.")
            return False

        saved_latest = saved_dates.max().strftime("%Y-%m-%d")
        raw_latest = pd.to_datetime(
            raw_df["timestamp"]
        ).max().strftime("%Y-%m-%d")

        print(f"\n  [SUCCESS] Saved: {output_file}")
        print(f"  Raw latest date:     {raw_latest}")
        print(f"  Feature latest date: {saved_latest}")
        print(f"  Output records:      {len(check_df):,}")

        if saved_latest == raw_latest:
            print("  Date check:          PASS — latest date preserved")
        else:
            print(
                "  Date check:          WARNING — latest dates differ. "
                "Check the raw CSV and its timestamp values."
            )

        # Report warm-up NaNs without deleting those rows.
        warmup_columns = [
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
            "atr_14",
        ]

        if len(check_df) > 0:
            first_complete = check_df[warmup_columns].notna().all(axis=1)

            if first_complete.any():
                first_complete_date = check_df.loc[
                    first_complete, "timestamp"
                ].iloc[0]

                print(
                    "  All-indicator warm-up ends around: "
                    f"{first_complete_date}"
                )
            else:
                print(
                    "  [WARNING] No row has all indicators populated. "
                    "The dataset may be too short."
                )

        return True

    except Exception as error:
        print(f"  [ERROR] Verification failed: {error}")
        return False


# ============================================================
# BUILD ALL COMPANIES
# ============================================================

def main():
    print("\n" + "#" * 68)
    print("NGX STOCK MARKET — TECHNICAL FEATURE BUILDER")
    print("#" * 68)

    print(f"\nProject directory: {PROJECT_DIR}")
    print(f"Raw data directory: {RAW_DIR}")
    print(f"Feature directory: {FEATURE_DIR}")

    if not RAW_DIR.exists():
        print(f"\n[ERROR] Raw data directory does not exist: {RAW_DIR}")
        print("Check that you are using the correct project folder.")
        sys.exit(1)

    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    successful = []
    failed = []

    for ticker in COMPANIES:
        try:
            result = build_company_features(ticker)

            if result:
                successful.append(ticker)
            else:
                failed.append(ticker)

        except Exception as error:
            print(f"\n[ERROR] {ticker}: {error}")
            failed.append(ticker)

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n\n" + "#" * 68)
    print("FEATURE BUILD SUMMARY")
    print("#" * 68)

    print(f"\nSuccessful: {len(successful)}/{len(COMPANIES)}")
    print(f"Failed:     {len(failed)}/{len(COMPANIES)}")

    if successful:
        print("\nSuccessful companies:")
        for ticker in successful:
            print(f"  [OK] {ticker}")

    if failed:
        print("\nCompanies requiring attention:")
        for ticker in failed:
            print(f"  [FAILED] {ticker}")

    # Final table of saved coverage.
    print("\n\nLATEST DATE VERIFICATION")
    print("-" * 68)
    print(f"{'TICKER':<15} {'RECORDS':>10} {'FIRST DATE':>15} {'LATEST DATE':>15}")
    print("-" * 68)

    for ticker in COMPANIES:
        output_file = FEATURE_DIR / f"{ticker}_features.csv"

        if not output_file.exists():
            print(f"{ticker:<15} {'—':>10} {'—':>15} {'FILE MISSING':>15}")
            continue

        try:
            df = pd.read_csv(output_file)

            dates = pd.to_datetime(
                df["timestamp"],
                errors="coerce",
            ).dropna()

            if dates.empty:
                print(f"{ticker:<15} {len(df):>10} {'—':>15} {'NO VALID DATES':>15}")
                continue

            first_date = dates.min().strftime("%Y-%m-%d")
            last_date = dates.max().strftime("%Y-%m-%d")

            print(
                f"{ticker:<15} "
                f"{len(df):>10,} "
                f"{first_date:>15} "
                f"{last_date:>15}"
            )

        except Exception as error:
            print(f"{ticker:<15} {'ERROR':>10} {'—':>15} {str(error)[:15]:>15}")

    print("\n" + "#" * 68)

    if failed:
        print("Finished with errors. Review the messages above.")
        sys.exit(1)

    print("ALL FEATURE FILES BUILT SUCCESSFULLY.")
    print("The newest dates from each valid raw CSV should now be preserved.")
    print("#" * 68)


if __name__ == "__main__":
    main()