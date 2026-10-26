import sqlite3
import pandas as pd

DB_PATH = "backend/data/ngx.db"


def validate_database():

    conn = sqlite3.connect(DB_PATH)

    print("=" * 70)
    print("NGX STOCK MARKET DATA VALIDATION")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Show companies
    # ---------------------------------------------------------

    companies = pd.read_sql_query(
        """
        SELECT company_id, ticker, name, sector
        FROM Company
        ORDER BY sector, ticker
        """,
        conn
    )

    print("\n📊 COMPANIES IN DATABASE")
    print("-" * 70)
    print(companies.to_string(index=False))

    # ---------------------------------------------------------
    # 2. Count stock-price records
    # ---------------------------------------------------------

    counts = pd.read_sql_query(
        """
        SELECT
            c.ticker,
            c.name,
            c.sector,
           COUNT(s.price_id) AS rows
        FROM Company c
        LEFT JOIN StockPrice s
            ON c.company_id = s.company_id
        GROUP BY c.company_id
        ORDER BY c.sector, c.ticker
        """,
        conn
    )

    print("\n📈 STOCK PRICE RECORDS")
    print("-" * 70)
    print(counts.to_string(index=False))

    # ---------------------------------------------------------
    # 3. Check date ranges
    # ---------------------------------------------------------

    dates = pd.read_sql_query(
        """
        SELECT
            c.ticker,
            MIN(s.timestamp) AS first_date,
            MAX(s.timestamp) AS last_date
        FROM Company c
        JOIN StockPrice s
            ON c.company_id = s.company_id
        GROUP BY c.company_id
        ORDER BY c.ticker
        """,
        conn
    )

    print("\n📅 DATE RANGES")
    print("-" * 70)
    print(dates.to_string(index=False))

    # ---------------------------------------------------------
    # 4. Check missing values
    # ---------------------------------------------------------

    missing = pd.read_sql_query(
        """
        SELECT
            c.ticker,
            SUM(
                CASE WHEN s.timestamp IS NULL THEN 1 ELSE 0 END
            ) AS missing_date,
            SUM(
                CASE WHEN s.open IS NULL THEN 1 ELSE 0 END
            ) AS missing_open,
            SUM(
                CASE WHEN s.high IS NULL THEN 1 ELSE 0 END
            ) AS missing_high,
            SUM(
                CASE WHEN s.low IS NULL THEN 1 ELSE 0 END
            ) AS missing_low,
            SUM(
                CASE WHEN s.close IS NULL THEN 1 ELSE 0 END
            ) AS missing_close,
            SUM(
                CASE WHEN s.volume IS NULL THEN 1 ELSE 0 END
            ) AS missing_volume
        FROM Company c
        LEFT JOIN StockPrice s
            ON c.company_id = s.company_id
        GROUP BY c.company_id
        ORDER BY c.ticker
        """,
        conn
    )

    print("\n🔍 MISSING VALUES")
    print("-" * 70)
    print(missing.to_string(index=False))

    # ---------------------------------------------------------
    # 5. Check duplicate dates
    # ---------------------------------------------------------

    duplicates = pd.read_sql_query(
        """
        SELECT
            c.ticker,
            s.timestamp,
            COUNT(*) AS occurrences
        FROM StockPrice s
        JOIN Company c
            ON c.company_id = s.company_id
        GROUP BY c.company_id, s.timestamp
        HAVING COUNT(*) > 1
        ORDER BY c.ticker, s.timestamp
        """,
        conn
    )

    print("\n♻️ DUPLICATE RECORDS")
    print("-" * 70)

    if duplicates.empty:
        print("✅ No duplicate stock-price records found.")
    else:
        print(duplicates.to_string(index=False))

    # ---------------------------------------------------------
    # 6. Check invalid prices
    # ---------------------------------------------------------

    invalid = pd.read_sql_query(
        """
        SELECT
            c.ticker,
            COUNT(*) AS invalid_rows
        FROM StockPrice s
        JOIN Company c
            ON c.company_id = s.company_id
        WHERE
            s.open <= 0
            OR s.high <= 0
            OR s.low <= 0
            OR s.close <= 0
            OR s.high < s.low
        GROUP BY c.company_id
        ORDER BY c.ticker
        """,
        conn
    )

    print("\n⚠️ INVALID PRICE RECORDS")
    print("-" * 70)

    if invalid.empty:
        print("✅ No invalid price records found.")
    else:
        print(invalid.to_string(index=False))

    conn.close()

    print("\n" + "=" * 70)
    print("✅ DATA VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    validate_database()