
from pathlib import Path
import pandas as pd

tickers = [
    "SEPLAT", "AIRTELAFRI", "TOTAL",
    "BUACEMENT", "DANGCEM", "GTCO",
    "MTNN", "NESTLE", "UNILEVER", "ZENITHBANK"
]

for ticker in tickers:
    print(f"\n{'=' * 65}")
    print(ticker)

    for folder in ["raw", "features"]:
        path = Path("backend/data") / folder / f"{ticker}.csv"

        if not path.exists():
            print(f"{folder}: FILE NOT FOUND")
            continue

        df = pd.read_csv(path)
        date_col = next(
            (c for c in ["timestamp", "date", "Date"]
             if c in df.columns),
            None
        )

        print(f"\n{folder}: {path}")
        print("Rows:", len(df))
        print("Columns:", df.columns.tolist())
        print("Modified:", pd.Timestamp(
            path.stat().st_mtime, unit="s"
        ))

        if date_col:
            dates = pd.to_datetime(df[date_col], errors="coerce")
            print("First date:", dates.min())
            print("Last date:", dates.max())

        if "close" in df.columns:
            close = pd.to_numeric(df["close"], errors="coerce")
            print("First close:", close.iloc[0])
            print("Last close:", close.iloc[-1])
            print("Minimum close:", close.min())
            print("Maximum close:", close.max())

        if folder == "features":
            for col in ["rsi_14", "sma_20", "sma_50"]:
                if col in df.columns:
                    values = pd.to_numeric(
                        df[col], errors="coerce"
                    )
                    print(
                        f"{col}: last={values.iloc[-1]}, "
                        f"min={values.min()}, max={values.max()}"
                    )
