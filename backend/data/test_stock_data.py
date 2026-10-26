from .stock_data import fetch_stock_data


ticker = "DANGCEM.LG"

data = fetch_stock_data(ticker)

print("\nFirst 5 records:")
print(data.head())

print("\nLast 5 records:")
print(data.tail())

print("\nNumber of records:")
print(len(data))

print("\nColumns:")
print(data.columns.tolist())