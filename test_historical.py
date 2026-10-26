
import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("KOBO_API_KEY")
if not api_key:
    raise RuntimeError("KOBO_API_KEY was not found in your .env file")

url = "https://koboterminal.com/api/ngxdata/prices/SEPLAT"

try:
    response = requests.get(
        url,
        headers={"X-API-Key": api_key},
        params={"days": 30},
        timeout=30,
    )

    print("HTTP status:", response.status_code)
    response.raise_for_status()

    data = response.json()
    print("Response type:", type(data).__name__)

    if isinstance(data, dict):
        print("Response keys:", list(data.keys()))

        # Print a small sample without exposing your API key.
        for key, value in data.items():
            if isinstance(value, list):
                print(f"{key}: {len(value)} records")
                print("First record:", value[0] if value else "No records")
                print("Last record:", value[-1] if value else "No records")
            else:
                print(f"{key}:", value)

    elif isinstance(data, list):
        print("Number of records:", len(data))
        print("First record:", data[0] if data else "No records")
        print("Last record:", data[-1] if data else "No records")

except requests.RequestException as exc:
    print("Request failed:", type(exc).__name__, str(exc))
    if getattr(exc, "response", None) is not None:
        print("Response body:", exc.response.text[:500])
