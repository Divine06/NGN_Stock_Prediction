
import re
import time
import requests
import xml.etree.ElementTree as ET

from threading import Lock
from urllib.parse import quote_plus

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"

COMPANIES = {
    "AIRTELAFRI": "Airtel Africa",
    "MTNN": "MTN Nigeria",
    "BUACEMENT": "BUA Cement",
    "DANGCEM": "Dangote Cement",
    "GTCO": "Guaranty Trust Holding Company",
    "ZENITHBANK": "Zenith Bank",
    "NESTLE": "Nestle Nigeria",
    "UNILEVER": "Unilever Nigeria",
    "SEPLAT": "Seplat Energy",
    "TOTAL": "TotalEnergies Nigeria",
    "ACCESSCORP": "Access Holdings",
    "UBA": "United Bank for Africa",
    "FIDELITYBK": "Fidelity Bank Nigeria",
    "BUAFOODS": "BUA Foods",
    "OANDO": "Oando",
    "CONOIL": "Conoil",
}

POSITIVE_WORDS = {
    "growth", "profit", "profits", "gain", "gains", "surge",
    "surges", "rise", "rises", "rising", "strong", "record",
    "upgrade", "upgraded", "dividend", "expansion", "success",
    "positive", "improve", "improved", "recovery", "beat",
    "beats", "increase", "increased", "boost", "boosts",
}

NEGATIVE_WORDS = {
    "loss", "losses", "decline", "declines", "fall", "falls",
    "falling", "drop", "drops", "dropping", "slump", "slumps",
    "downgrade", "downgraded", "miss", "misses", "cut", "cuts",
    "warning", "warns", "negative", "debt", "risk", "risks",
}

CACHE_TTL = 1800
FAILURE_TTL = 300

_NEWS_CACHE = {}
_NEXT_RETRY = {}
_CACHE_LOCK = Lock()


def analyze_headline(title):
    words = re.findall(r"\b[a-z]+\b", title.lower())
    positive = sum(word in POSITIVE_WORDS for word in words)
    negative = sum(word in NEGATIVE_WORDS for word in words)

    if positive > negative:
        return "Positive", positive - negative
    if negative > positive:
        return "Negative", negative - positive
    return "Neutral", 0


def _unavailable(ticker, company_name, status, message, cached=False):
    return {
        "ticker": ticker,
        "company_name": company_name,
        "status": status,
        "article_count": 0,
        "sentiment": None,
        "articles": [],
        "cached": cached,
        "message": message,
    }


def get_company_news(ticker: str, limit: int = 0):
    """
    Fetch company news from Google News RSS.

    limit=0 returns every article supplied by the RSS feed.
    A positive limit restricts the returned entries to that number.
    The provider determines how many entries the feed contains.
    """
    ticker = ticker.upper().strip()
    company_name = COMPANIES.get(ticker)

    if not company_name:
        return _unavailable(
            ticker, ticker, "unsupported_ticker",
            "This ticker is not configured for news search.",
        )

    try:
        limit = max(0, int(limit))
    except (TypeError, ValueError):
        limit = 0

    cache_key = ticker
    now = time.time()

    with _CACHE_LOCK:
        cached_entry = _NEWS_CACHE.get(cache_key)

        if cached_entry and now - cached_entry[1] < CACHE_TTL:
            result = dict(cached_entry[0])
            result["cached"] = True
            return result

        retry_until = _NEXT_RETRY.get(cache_key, 0)

        if now < retry_until:
            if cached_entry:
                result = dict(cached_entry[0])
                result["cached"] = True
                result["status"] = "stale"
                result["message"] = (
                    "Showing previously collected headlines while "
                    "Google News RSS is temporarily unavailable."
                )
                return result

            return _unavailable(
                ticker, company_name, "provider_cooldown",
                "Google News RSS is temporarily unavailable. "
                "Please try again later.",
            )

    query = quote_plus(f'"{company_name}"')
    url = (
        f"{GOOGLE_NEWS_RSS}?q={query}"
        "&hl=en-NG&gl=NG&ceid=NG:en"
    )

    try:
        response = requests.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 NGXAdvisoryStudentProject/1.0"
            },
            timeout=20,
        )
        response.raise_for_status()

        root = ET.fromstring(response.content)
        channel = root.find("channel")

        if channel is None:
            raise ValueError("RSS response has no channel element.")

        articles = []
        positive_count = 0
        negative_count = 0

        for item in channel.findall("item"):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            published = (item.findtext("pubDate") or "").strip()

            source_element = item.find("source")
            source = (
                (source_element.text or "").strip()
                if source_element is not None
                else "Google News"
            )

            if not title or not link:
                continue

            sentiment, score = analyze_headline(title)

            if sentiment == "Positive":
                positive_count += 1
            elif sentiment == "Negative":
                negative_count += 1

            articles.append({
                "title": title,
                "url": link,
                "source": source,
                "published": published,
                "sentiment": sentiment,
                "score": score,
            })

        # RSS provides the available feed entries. Apply a limit only
        # when the caller explicitly requests a positive number.
        if limit > 0:
            articles = articles[:limit]

        if not articles:
            overall = "INSUFFICIENT_DATA"
        elif positive_count > negative_count:
            overall = "Positive"
        elif negative_count > positive_count:
            overall = "Negative"
        else:
            overall = "Neutral"

        result = {
            "ticker": ticker,
            "company_name": company_name,
            "status": "ok" if articles else "no_articles",
            "article_count": len(articles),
            "sentiment": {
                "overall": overall,
                "positive_count": positive_count,
                "negative_count": negative_count,
                "method": "experimental_word_lexicon",
            } if articles else None,
            "articles": articles,
            "cached": False,
            "message": None if articles else (
                "Google News RSS returned no matching headlines."
            ),
        }

        with _CACHE_LOCK:
            _NEWS_CACHE[cache_key] = (result, time.time())
            _NEXT_RETRY.pop(cache_key, None)

        return result

    except requests.Timeout:
        status = "provider_timeout"
        message = "Google News RSS timed out. Please try again later."

    except requests.RequestException as exc:
        status = "provider_unavailable"
        message = f"Google News RSS request failed: {exc}"

    except ET.ParseError:
        status = "invalid_response"
        message = "Google News RSS returned invalid XML."

    except ValueError as exc:
        status = "invalid_response"
        message = str(exc)

    with _CACHE_LOCK:
        _NEXT_RETRY[cache_key] = time.time() + FAILURE_TTL

        cached_entry = _NEWS_CACHE.get(cache_key)
        if cached_entry:
            result = dict(cached_entry[0])
            result["cached"] = True
            result["status"] = "stale"
            result["message"] = (
                message + " Showing previously retrieved headlines."
            )
            return result

    return _unavailable(ticker, company_name, status, message)
