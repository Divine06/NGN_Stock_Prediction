import os
import pandas as pd
import sqlite3
from datetime import datetime

# Path configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "raw")
DB_PATH = os.path.join(BASE_DIR, "ngx.db")

# Company mapping: filename → (name, sector)
COMPANIES = {
    "ZENITHBANK": ("Zenith Bank",       "Banking"),
    "GTCO":       ("GT Bank",           "Banking"),
    "SEPLAT":     ("Seplat Energy",     "Oil and Gas"),
    "TOTAL":      ("Total Energies",    "Oil and Gas"),
    "NESTLE":     ("Nestle Nigeria",    "FMCG"),
    "UNILEVER":   ("Unilever Nigeria",  "FMCG"),
    "MTNN":       ("MTN Nigeria",       "Telecommunications"),
    "AIRTELAFRI": ("Airtel Africa",     "Telecommunications"),
    "DANGCEM":    ("Dangote Cement",    "Manufacturing"),
    "BUACEMENT":  ("BUA Cement",        "Manufacturing"),
}

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Create all tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS Company (
            company_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            name         TEXT NOT NULL,
            ticker       TEXT NOT NULL UNIQUE,
            sector       TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS StockPrice (
            price_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id   INTEGER NOT NULL,
            timestamp    TEXT NOT NULL,
            timeframe    TEXT NOT NULL DEFAULT '1d',
            open         REAL,
            high         REAL,
            low          REAL,
            close        REAL,
            volume       INTEGER,
            FOREIGN KEY (company_id) REFERENCES Company(company_id),
            UNIQUE(company_id, timestamp, timeframe)
        );

        CREATE TABLE IF NOT EXISTS TechnicalIndicator (
            indicator_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            price_id         INTEGER NOT NULL UNIQUE,
            rsi              REAL,
            macd             REAL,
            macd_signal      REAL,
            moving_avg_20    REAL,
            moving_avg_50    REAL,
            bollinger_upper  REAL,
            bollinger_lower  REAL,
            atr              REAL,
            FOREIGN KEY (price_id) REFERENCES StockPrice(price_id)
        );

        CREATE TABLE IF NOT EXISTS Prediction (
            prediction_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id       INTEGER NOT NULL,
            model_type       TEXT NOT NULL,
            predicted_price  REAL,
            current_price    REAL,
            mae              REAL,
            rmse             REAL,
            advisory         TEXT,
            stop_loss        REAL,
            take_profit      REAL,
            generated_at     TEXT,
            FOREIGN KEY (company_id) REFERENCES Company(company_id)
        );

        CREATE TABLE IF NOT EXISTS SentimentRecord (
            sentiment_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id       INTEGER NOT NULL,
            headline         TEXT,
            sentiment_score  REAL,
            sentiment_label  TEXT,
            fetched_at       TEXT,
            FOREIGN KEY (company_id) REFERENCES Company(company_id)
        );
    """)
    conn.commit()
    conn.close()
    print("✅ Database tables created.")

def insert_company(ticker, name, sector):
    """Insert company if not exists. Return company_id."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR IGNORE INTO Company (name, ticker, sector) VALUES (?, ?, ?)",
        (name, ticker, sector)
    )
    conn.commit()
    cursor.execute("SELECT company_id FROM Company WHERE ticker = ?", (ticker,))
    row = cursor.fetchone()
    conn.close()
    return row["company_id"]

def clean_dataframe(df, ticker):
    """Standardise columns from Investing.com CSV format."""

    # Investing.com columns:
    # Date, Price, Open, High, Low, Vol., Change %

    col_map = {
        "Date":     "timestamp",
        "Price":    "close",
        "Open":     "open",
        "High":     "high",
        "Low":      "low",
        "Vol.":     "volume",
    }

    # Rename available columns
    available = {k: v for k, v in col_map.items() if k in df.columns}
    df = df.rename(columns=available)

    # Required columns
    required = [
        "timestamp",
        "close",
        "open",
        "high",
        "low"
    ]

    for col in required:
        if col not in df.columns:
            print(
                f"  ⚠️ Missing column '{col}' in {ticker} — skipping."
            )
            return None

    # ---------------------------------------------------------
    # DATE CLEANING
    # ---------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        dayfirst=True,
        errors="coerce"
    )

    df = df.dropna(subset=["timestamp"])

    # Store dates in YYYY-MM-DD format
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d")

    # ---------------------------------------------------------
    # PRICE CLEANING
    # ---------------------------------------------------------

    for col in ["close", "open", "high", "low"]:

        if col in df.columns:

            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
            )

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    # ---------------------------------------------------------
    # VOLUME CLEANING
    # ---------------------------------------------------------

    if "volume" in df.columns:

        df["volume"] = (
            df["volume"]
            .astype(str)
            .str.replace(",", "", regex=False)
        )

        df["volume"] = df["volume"].apply(parse_volume)

    else:
        df["volume"] = None

    # ---------------------------------------------------------
    # REMOVE INVALID PRICE ROWS
    # ---------------------------------------------------------

    df = df.dropna(
        subset=[
            "close",
            "open",
            "high",
            "low"
        ]
    )

    # ---------------------------------------------------------
    # DATE RANGE
    # ---------------------------------------------------------

    # Keep the original historical period and allow 2026 data.
    df = df[df["timestamp"] >= "2014-01-01"]
    df = df[df["timestamp"] <= "2026-12-31"]

    # ---------------------------------------------------------
    # REMOVE DUPLICATE DATES
    # ---------------------------------------------------------

    df = df.drop_duplicates(
        subset=["timestamp"],
        keep="last"
    )

    # ---------------------------------------------------------
    # SORT CHRONOLOGICALLY
    # ---------------------------------------------------------

    df = (
        df
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    return df

def parse_volume(val):
    """Convert volume strings like 1.5M, 500K to integers."""
    try:
        val = str(val).strip().upper()
        if val in ["", "-", "N/A", "NAN"]:
            return None
        if val.endswith("M"):
            return int(float(val[:-1]) * 1_000_000)
        if val.endswith("K"):
            return int(float(val[:-1]) * 1_000)
        if val.endswith("B"):
            return int(float(val[:-1]) * 1_000_000_000)
        return int(float(val))
    except:
        return None

def import_csv(ticker, name, sector):
    """Import one company's CSV into SQLite."""
    csv_path = os.path.join(RAW_DIR, f"{ticker}.csv")

    if not os.path.exists(csv_path):
        print(f"  ❌ File not found: {ticker}.csv — skipping.")
        return 0

    print(f"\n📂 Importing {name} ({ticker})...")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  ❌ Could not read CSV: {e}")
        return 0

    df = clean_dataframe(df, ticker)
    if df is None or df.empty:
        print(f"  ❌ No valid data after cleaning.")
        return 0

    company_id = insert_company(ticker, name, sector)

    conn = get_connection()
    cursor = conn.cursor()
    inserted = 0
    skipped = 0

    for _, row in df.iterrows():
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO StockPrice
                (company_id, timestamp, timeframe, open, high, low, close, volume)
                VALUES (?, ?, '1d', ?, ?, ?, ?, ?)
            """, (
                company_id,
                row["timestamp"],
                row.get("open"),
                row.get("high"),
                row.get("low"),
                row.get("close"),
                row.get("volume"),
            ))
            if cursor.rowcount > 0:
                inserted += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"  ⚠️  Row error: {e}")

    conn.commit()
    conn.close()
    print(f"  ✅ Inserted: {inserted} rows | Skipped (duplicate): {skipped}")
    return inserted

def run_import():
    """Import all companies."""
    print("=" * 60)
    print("NGX STOCK DATA IMPORTER")
    print("=" * 60)

    os.makedirs(RAW_DIR, exist_ok=True)
    init_db()

    total = 0
    for ticker, (name, sector) in COMPANIES.items():
        count = import_csv(ticker, name, sector)
        total += count

    print("\n" + "=" * 60)
    print(f"✅ IMPORT COMPLETE — Total rows inserted: {total}")
    print("=" * 60)

    # Verify
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT c.name, c.ticker, COUNT(sp.price_id) as rows
        FROM Company c
        LEFT JOIN StockPrice sp ON c.company_id = sp.company_id
        GROUP BY c.company_id
        ORDER BY c.sector, c.name
    """)
    rows = cursor.fetchall()
    conn.close()

    print("\n📊 DATABASE SUMMARY:")
    print(f"{'Company':<25} {'Ticker':<15} {'Rows'}")
    print("-" * 50)
    for row in rows:
        print(f"{row['name']:<25} {row['ticker']:<15} {row['rows']}")

if __name__ == "__main__":
    run_import()