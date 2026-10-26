
import requests

url = "https://api.gdeltproject.org/api/v2/doc/doc"

params = {
    "query": '"MTN Nigeria"',
    "mode": "artlist",
    "format": "json",
    "maxrecords": 5,
    "sort": "datedesc",
}

try:
    response = requests.get(
        url,
        params=params,
        headers={"User-Agent": "Mozilla/5.0 NGXAdvisory/1.0"},
        timeout=25,
    )

    print("HTTP status:", response.status_code)
    response.raise_for_status()

    data = response.json()
    articles = data.get("articles", [])

    print("Articles returned:", len(articles))

    for article in articles:
        print("-", article.get("title"))
        print("  Publisher:", article.get("domain"))
        print("  URL:", article.get("url"))

except requests.RequestException as exc:
    print("Request failed:", type(exc).__name__, str(exc))
except ValueError as exc:
    print("Could not parse response as JSON:", str(exc))
