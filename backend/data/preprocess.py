import sqlite3
import pandas as pd
from pathlib import Path


DB_PATH = "backend/data/ngx.db"
OUTPUT_DIR = Path("backend/data/processed")


def load_stock_data(ticker):
    """
    Load historical stock-price data for one company.
    """

    conn = sqlite3.connect(DB_PATH)

    query = """
        SELECT
            c.ticker,
            c.name,
            c.sector,
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
        ORDER BY s.timestamp ASC
    """

    df = pd.read_sql_query(query, conn, params=(ticker,))

    conn.close()

    return df


def clean_stock_data(df, ticker):
    """
    Clean and prepare stock-price data.
    """

    print(f"\nProcessing {ticker}...")

    if df.empty:
        print(f"❌ No data found for {ticker}")
        return None

    # ---------------------------------------------------------
    # 1. Convert timestamp to datetime
    # ---------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    # ---------------------------------------------------------
    # 2. Remove rows with invalid dates
    # ---------------------------------------------------------

    before = len(df)

    df = df.dropna(subset=["timestamp"])

    removed_dates = before - len(df)

    if removed_dates:
        print(f"   Removed {removed_dates} rows with invalid dates.")

    # ---------------------------------------------------------
    # 3. Convert numerical columns
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # 4. Remove rows with missing essential prices
    # ---------------------------------------------------------

    before = len(df)

    df = df.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close"
        ]
    )

    removed_prices = before - len(df)

    if removed_prices:
        print(
            f"   Removed {removed_prices} rows "
            f"with missing price values."
        )

    # ---------------------------------------------------------
    # 5. Sort chronologically
    # ---------------------------------------------------------

    df = df.sort_values("timestamp")

    # ---------------------------------------------------------
    # 6. Remove duplicate dates
    # ---------------------------------------------------------

    before = len(df)

    df = df.drop_duplicates(
        subset=["timestamp"],
        keep="first"
    )

    removed_duplicates = before - len(df)

    if removed_duplicates:
        print(
            f"   Removed {removed_duplicates} "
            f"duplicate records."
        )

    # ---------------------------------------------------------
    # 7. Calculate daily percentage return
    # ---------------------------------------------------------

    df["return"] = df["close"].pct_change()

    # ---------------------------------------------------------
    # 8. Calculate price movement
    # ---------------------------------------------------------

    df["price_change"] = (
        df["close"] - df["open"]
    )

    # ---------------------------------------------------------
    # 9. Create prediction target
    #
    # 1 = price goes up next trading day
    # 0 = price does not go up
    # ---------------------------------------------------------

    df["next_close"] = df["close"].shift(-1)

    df["target"] = (
        df["next_close"] > df["close"]
    ).astype(int)

    # ---------------------------------------------------------
    # 10. Remove final row
    #
    # There is no next-day price for the final record.
    # ---------------------------------------------------------

    df = df.dropna(
        subset=["next_close"]
    )

    return df


def save_processed_data(df, ticker):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = OUTPUT_DIR / f"{ticker}.csv"

    df.to_csv(
        output_file,
        index=False
    )

    print(f"   ✅ Saved: {output_file}")


def main():

    print("=" * 70)
    print("NGX STOCK MARKET DATA PREPROCESSING")
    print("=" * 70)

    conn = sqlite3.connect(DB_PATH)

    companies = pd.read_sql_query(
        """
        SELECT ticker
        FROM Company
        ORDER BY ticker
        """,
        conn
    )

    conn.close()

    total_processed = 0

    for ticker in companies["ticker"]:

        df = load_stock_data(ticker)

        cleaned_df = clean_stock_data(
            df,
            ticker
        )

        if cleaned_df is not None:

            save_processed_data(
                cleaned_df,
                ticker
            )

            print(
                f"   Records ready for modelling: "
                f"{len(cleaned_df)}"
            )

            total_processed += 1

    print("\n" + "=" * 70)
    print(
        f"✅ PREPROCESSING COMPLETE — "
        f"{total_processed} companies processed"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()