import sqlite3
from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR
DB_PATH = DATA_DIR / "ngx.db"
RAW_DIR = DATA_DIR / "raw"

RAW_DIR.mkdir(parents=True, exist_ok=True)


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
# REBUILD
# ============================================================

def rebuild_company(connection, ticker):

    print()
    print("=" * 70)
    print(f"REBUILDING {ticker}")
    print("=" * 70)

    query = """
        SELECT
            s.timestamp,
            s.open,
            s.high,
            s.low,
            s.close,
            s.volume
        FROM StockPrice s
        JOIN Company c
            ON c.company_id = s.company_id
        WHERE c.ticker = ?
          AND s.timeframe = '1d'
        ORDER BY s.timestamp ASC
    """

    df = pd.read_sql_query(
        query,
        connection,
        params=(ticker,)
    )

    if df.empty:
        print(f"❌ No database data found for {ticker}")
        return False

    # Convert timestamp
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    # Convert numeric columns
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Remove invalid rows
    df = df.dropna(
        subset=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    )

    # Remove duplicate dates
    df = df.drop_duplicates(
        subset=["timestamp"],
        keep="last"
    )

    # Sort oldest → newest
    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    # Standard date format
    df["timestamp"] = df[
        "timestamp"
    ].dt.strftime("%Y-%m-%d")

    # Make sure columns are exactly what the
    # feature builder expects.
    df = df[
        [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    ]

    output_file = RAW_DIR / f"{ticker}.csv"

    df.to_csv(
        output_file,
        index=False
    )

    print(f"✅ File: {output_file}")
    print(f"Rows:   {len(df)}")
    print(f"Start:  {df['timestamp'].iloc[0]}")
    print(f"Latest: {df['timestamp'].iloc[-1]}")

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("REBUILDING RAW CSV FILES FROM SQLITE DATABASE")
    print("=" * 70)

    if not DB_PATH.exists():
        print(f"❌ Database not found:")
        print(DB_PATH)
        return

    connection = sqlite3.connect(DB_PATH)

    successful = 0
    failed = 0

    try:

        for ticker in COMPANIES:

            try:

                if rebuild_company(
                    connection,
                    ticker
                ):
                    successful += 1
                else:
                    failed += 1

            except Exception as error:

                failed += 1

                print(
                    f"❌ {ticker} failed:"
                )
                print(error)

    finally:
        connection.close()

    print()
    print("=" * 70)
    print("REBUILD COMPLETE")
    print("=" * 70)
    print(f"Successful: {successful}")
    print(f"Failed:     {failed}")
    print("=" * 70)

    if failed == 0:
        print(
            "🎉 ALL 10 RAW CSV FILES HAVE BEEN REBUILT."
        )


if __name__ == "__main__":
    main()