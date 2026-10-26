
from pathlib import Path
import pandas as pd

for ticker in ["SEPLAT", "AIRTELAFRI", "TOTAL"]:
    path = Path("backend/data/raw") / f"{ticker}.csv"

    print(f"\n{'=' * 60}\n{ticker}: {path}")

    if not path.exists():
        print("FILE NOT FOUND")
        continue

    df = pd.read_csv(path)
    print("Columns:", df.columns.tolist())
    print("\nFirst five rows:")
    print(df.head().to_string(index=False))
    print("\nLast five rows:")
    print(df.tail().to_string(index=False))
