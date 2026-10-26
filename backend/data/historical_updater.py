"""
NGX Historical Data Updater
---------------------------
Updates the existing 2014-2025 NGX dataset with newer data from Investo.

Data flow:
Investo API
    ↓
backend/data/raw/*.csv
    ↓
backend/data/ngx.db

The existing historical data is preserved.
Duplicate dates are removed automatically.
"""

import os
import sqlite3
from pathlib import Path
from datetime import date, timedelta

import pandas as pd
import requests
from dotenv import load_dotenv


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent

DATA_DIR = BASE_DIR
RAW_DIR = DATA_DIR / "raw"
DB_PATH = DATA_DIR / "ngx.db"

ENV_FILE = PROJECT_ROOT / ".env"


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv(ENV_FILE)

INVESTO_API_KEY = os.getenv("INVESTO_API_KEY")

if not INVESTO_API_KEY:
    raise RuntimeError(
        "INVESTO_API_KEY was not found in .env\n"
        f"Expected .env at: {ENV_FILE}"
    )


# ============================================================
# INVESTO SETTINGS
# ============================================================

API_BASE = "https://investo.ng/api/v1"

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
# HTTP SESSION
# ============================================================

SESSION = requests.Session()

SESSION.headers.update(
    {
        "Authorization": f"Bearer {INVESTO_API_KEY}",
        "Accept": "application/json",
        "User-Agent": "NGX-Advisory-System/1.0",
    }
)


# ============================================================
# DATABASE
# ============================================================

def get_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def get_company_id(connection, ticker):
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT company_id
        FROM Company
        WHERE ticker = ?
        """,
        (ticker,),
    )

    result = cursor.fetchone()

    if result is None:
        return None

    return result[0]


def get_latest_database_date(connection, ticker):
    """
    Returns the latest date currently stored in SQLite.
    """

    company_id = get_company_id(connection, ticker)

    if company_id is None:
        return None

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT MAX(timestamp)
        FROM StockPrice
        WHERE company_id = ?
        AND timeframe = '1d'
        """,
        (company_id,),
    )

    result = cursor.fetchone()

    if not result or result[0] is None:
        return None

    return str(result[0])[:10]


# ============================================================
# RAW CSV
# ============================================================

def get_raw_file(ticker):
    return RAW_DIR / f"{ticker}.csv"


def get_latest_raw_date(ticker):
    """
    Checks the raw CSV as well as SQLite.
    """

    file_path = get_raw_file(ticker)

    if not file_path.exists():
        return None

    try:
        df = pd.read_csv(file_path)

        if df.empty or "timestamp" not in df.columns:
            return None

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce"
        )

        df = df.dropna(subset=["timestamp"])

        if df.empty:
            return None

        return df["timestamp"].max().date().isoformat()

    except Exception as error:
        print(f"Warning reading {file_path}: {error}")
        return None


# ============================================================
# INVESTO API
# ============================================================

def fetch_history(ticker, start_date, end_date):
    """
    Fetch historical daily OHLCV data from Investo.
    Uses smaller date windows and retries to avoid connection resets.
    """

    url = f"{API_BASE}/prices/{ticker}"

    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()

    all_data = []

    # Request 30-day chunks instead of one large request.
    current = start

    while current <= end:

        chunk_end = min(
            current + timedelta(days=29),
            end
        )

        print()
        print("-" * 60)
        print(f"Fetching {ticker}")
        print(f"From: {current}")
        print(f"To:   {chunk_end}")
        print("-" * 60)

        params = {
            "from": current.isoformat(),
            "to": chunk_end.isoformat(),
            "provenance": "1",
        }

        success = False

        for attempt in range(1, 4):

            try:

                response = SESSION.get(
                    url,
                    params=params,
                    timeout=(10, 60),
                )

                print(
                    f"Attempt {attempt}: "
                    f"HTTP {response.status_code}"
                )

                if response.status_code == 200:

                    payload = response.json()

                    if payload.get("ok") is True:

                        chunk_data = payload.get(
                            "data",
                            []
                        )

                        print(
                            f"Rows received: "
                            f"{len(chunk_data)}"
                        )

                        all_data.extend(
                            chunk_data
                        )

                        success = True
                        break

                    else:
                        print(
                            "Investo returned an error:"
                        )
                        print(payload)

                        # Don't retry an API-level error.
                        break

                else:

                    try:
                        print(response.json())
                    except Exception:
                        print(response.text[:500])

            except requests.exceptions.RequestException as error:

                print(
                    f"Connection error on attempt "
                    f"{attempt}: {error}"
                )

                if attempt < 3:
                    import time
                    time.sleep(2)

        if not success:
            print(
                f"⚠️ Failed chunk "
                f"{current} → {chunk_end}"
            )

        current = chunk_end + timedelta(days=1)

    print()
    print(
        f"Total rows collected for {ticker}: "
        f"{len(all_data)}"
    )

    return all_data


# ============================================================
# NORMALISE INVESTO DATA
# ============================================================

def normalise_history(data):
    """
    Convert Investo response into our existing CSV format.
    """

    rows = []

    for item in data:

        if not isinstance(item, dict):
            continue

        raw_date = item.get("date")

        if not raw_date:
            continue

        try:
            timestamp = pd.to_datetime(raw_date)
        except Exception:
            continue

        row = {
            "timestamp": timestamp.strftime("%Y-%m-%d"),
            "open": item.get("open"),
            "high": item.get("high"),
            "low": item.get("low"),
            "close": item.get("close"),
            "volume": item.get("volume"),
        }

        rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        )

    df = pd.DataFrame(rows)

    # Convert numeric fields
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Remove rows without essential OHLCV data
    df = df.dropna(
        subset=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    )

    # Remove duplicate dates
    df = df.drop_duplicates(
        subset=["timestamp"],
        keep="last"
    )

    df = df.sort_values("timestamp")

    return df[
        [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ].reset_index(drop=True)


# ============================================================
# UPDATE RAW CSV
# ============================================================

def update_raw_csv(ticker, new_data):
    """
    Merge new Investo data into the existing raw CSV.
    """

    file_path = get_raw_file(ticker)

    new_df = normalise_history(new_data)

    if new_df.empty:
        print(f"⚠️ No usable new data for {ticker}.")
        return 0

    if file_path.exists():

        try:
            existing_df = pd.read_csv(file_path)

            if not existing_df.empty:
                existing_df["timestamp"] = pd.to_datetime(
                    existing_df["timestamp"],
                    errors="coerce"
                ).dt.strftime("%Y-%m-%d")

                existing_df = existing_df.dropna(
                    subset=["timestamp"]
                )

                combined = pd.concat(
                    [
                        existing_df,
                        new_df,
                    ],
                    ignore_index=True,
                )

            else:
                combined = new_df

        except Exception as error:
            print(
                f"⚠️ Could not read existing CSV for {ticker}: {error}"
            )
            combined = new_df

    else:
        combined = new_df

    # Normalise again after merging
    combined["timestamp"] = pd.to_datetime(
        combined["timestamp"],
        errors="coerce"
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in numeric_columns:
        combined[column] = pd.to_numeric(
            combined[column],
            errors="coerce"
        )

    combined = combined.dropna(
        subset=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    )

    combined = combined.drop_duplicates(
        subset=["timestamp"],
        keep="last"
    )

    combined = combined.sort_values("timestamp")

    combined["timestamp"] = combined[
        "timestamp"
    ].dt.strftime("%Y-%m-%d")

    combined = combined[
        [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ]

    combined.to_csv(
        file_path,
        index=False
    )

    print(
        f"✅ {ticker}.csv updated: "
        f"{len(combined)} total rows"
    )

    return len(combined)


# ============================================================
# UPDATE SQLITE
# ============================================================

def update_database(ticker):
    """
    Import the complete raw CSV into SQLite without
    creating duplicate StockPrice rows.
    """

    file_path = get_raw_file(ticker)

    if not file_path.exists():
        print(f"❌ Raw file missing: {file_path}")
        return 0

    connection = get_connection()

    try:
        company_id = get_company_id(
            connection,
            ticker
        )

        if company_id is None:
            print(
                f"❌ Company {ticker} does not exist "
                "in the Company table."
            )
            return 0

        df = pd.read_csv(file_path)

        if df.empty:
            return 0

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce"
        )

        numeric_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

        for column in numeric_columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        df = df.dropna(
            subset=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        )

        df = df.drop_duplicates(
            subset=["timestamp"]
        )

        cursor = connection.cursor()

        inserted = 0

        for _, row in df.iterrows():

            timestamp = row["timestamp"].strftime(
                "%Y-%m-%d"
            )

            cursor.execute(
                """
                INSERT OR IGNORE INTO StockPrice
                (
                    company_id,
                    timestamp,
                    timeframe,
                    open,
                    high,
                    low,
                    close,
                    volume
                )
                VALUES (?, ?, '1d', ?, ?, ?, ?, ?)
                """,
                (
                    company_id,
                    timestamp,
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    int(float(row["volume"])),
                ),
            )

            if cursor.rowcount > 0:
                inserted += 1

        connection.commit()

        print(
            f"✅ {ticker}: {inserted} new database rows inserted."
        )

        return inserted

    except Exception as error:

        connection.rollback()

        print(
            f"❌ Database update failed for {ticker}: {error}"
        )

        return 0

    finally:
        connection.close()


# ============================================================
# UPDATE ONE COMPANY
# ============================================================

def update_company(ticker):
    """
    Update one company from the latest stored date
    through today.
    """

    print()
    print("#" * 70)
    print(f"UPDATING {ticker}")
    print("#" * 70)

    connection = get_connection()

    try:
        db_latest = get_latest_database_date(
            connection,
            ticker
        )
    finally:
        connection.close()

    raw_latest = get_latest_raw_date(ticker)

    print(f"Database latest: {db_latest}")
    print(f"CSV latest:      {raw_latest}")

    # Use the newest date we already have.
    existing_dates = [
        value
        for value in [
            db_latest,
            raw_latest,
        ]
        if value
    ]

    if existing_dates:
        latest_existing = max(existing_dates)

        next_date = (
            pd.to_datetime(latest_existing)
            + pd.Timedelta(days=1)
        ).date()

    else:
        # Safety fallback.
        next_date = date(2014, 1, 1)

    today = date.today()

    print(f"Next requested date: {next_date}")
    print(f"Today:                {today}")

    if next_date > today:
        print(
            f"✅ {ticker} is already up to date."
        )

        # Still ensure DB contains everything in CSV.
        update_database(ticker)
        return True

    new_data = fetch_history(
        ticker,
        next_date.isoformat(),
        today.isoformat(),
    )

    if not new_data:
        print(
            f"⚠️ No new data returned for {ticker}."
        )
        return False

    total_rows = update_raw_csv(
        ticker,
        new_data
    )

    update_database(ticker)

    # Verify final date
    connection = get_connection()

    try:
        final_date = get_latest_database_date(
            connection,
            ticker
        )
    finally:
        connection.close()

    print()
    print(
        f"✅ {ticker} complete."
    )
    print(
        f"Latest database date: {final_date}"
    )
    print(
        f"Total CSV rows:       {total_rows}"
    )

    return True


# ============================================================
# RUN ALL COMPANIES
# ============================================================

def run_update():

    print()
    print("=" * 70)
    print("NGX HISTORICAL DATA UPDATE")
    print("=" * 70)
    print("Source: Investo Data API")
    print("Target: 2026")
    print("=" * 70)

    successful = 0
    failed = 0

    for ticker in COMPANIES:

        try:

            result = update_company(ticker)

            if result:
                successful += 1
            else:
                failed += 1

        except Exception as error:

            failed += 1

            print()
            print(
                f"❌ {ticker} failed:"
            )
            print(error)

    print()
    print("=" * 70)
    print("FINAL UPDATE SUMMARY")
    print("=" * 70)
    print(f"Successful: {successful}")
    print(f"Failed:     {failed}")
    print("=" * 70)

    if successful == len(COMPANIES):
        print(
            "🎉 ALL 10 COMPANIES UPDATED SUCCESSFULLY."
        )

    elif successful > 0:
        print(
            "⚠️ Some companies updated successfully."
        )

    else:
        print(
            "❌ No companies were updated."
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_update()