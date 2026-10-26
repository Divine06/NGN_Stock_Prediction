
from backend.market_data import fetch_all_stocks, COMPANIES

result = fetch_all_stocks(force_refresh=True)

print("Fetched at:", result["fetched_at"])
print("Used cache:", result["cached"])

for stock in result["stocks"]:
    symbol = str(
        stock.get("symbol") or stock.get("ticker") or ""
    ).upper()

    if symbol in COMPANIES:
        print({
            "ticker": symbol,
            "name": stock.get("name"),
            "current_price": stock.get("current_price"),
            "change_percent": stock.get("change_percent"),
            "volume": stock.get("volume"),
            "sector": stock.get("sector"),
        })
