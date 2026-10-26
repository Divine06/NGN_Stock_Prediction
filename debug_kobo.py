
from pprint import pprint

from backend.market_data import fetch_all_stocks, get_live_stock

print("\n=== STEP A: FETCH KOBO STOCK LIST ===")

try:
    result = fetch_all_stocks(force_refresh=True)

    print("Return type:", type(result).__name__)

    if isinstance(result, list):
        print("Number of records:", len(result))
        print("First 3 records:")
        pprint(result[:3])

    elif isinstance(result, dict):
        print("Top-level keys:", list(result.keys()))
        pprint(result)

    else:
        pprint(result)

except Exception as e:
    print("FETCH ERROR:", type(e).__name__, str(e))


print("\n=== STEP B: TEST INDIVIDUAL TICKERS ===")

for ticker in ["MTNN", "GTCO", "AIRTELAFRI", "SEPLAT"]:
    print(f"\n--- {ticker} ---")

    try:
        result = get_live_stock(ticker)

        print("Return type:", type(result).__name__)

        if isinstance(result, dict):
            print("Returned keys:", list(result.keys()))
            pprint(result)
        else:
            pprint(result)

    except Exception as e:
        print("TICKER ERROR:", type(e).__name__, str(e))
