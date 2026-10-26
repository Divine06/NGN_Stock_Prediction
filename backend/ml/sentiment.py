"""Google News RSS headlines scored with VADER.

If your existing backend/news_sentiment.py already works, you can keep it: call
`set_provider(your_function)` where your function takes a ticker and returns
{"score": float | None, "n_headlines": int, "headlines": [...]}.
"""
from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

from . import config

RSS_URL = "https://news.google.com/rss/search?q={q}&hl=en-NG&gl=NG&ceid=NG:en"
_cache: dict[str, tuple[float, dict]] = {}
_provider = None
_analyzer = None


def set_provider(fn) -> None:
    global _provider
    _provider = fn


def _get_analyzer():
    global _analyzer
    if _analyzer is None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        except ImportError:
            try:
                from nltk.sentiment.vader import SentimentIntensityAnalyzer  # needs nltk.download("vader_lexicon")
            except ImportError as exc:
                raise RuntimeError("Install vaderSentiment: pip install vaderSentiment") from exc
        _analyzer = SentimentIntensityAnalyzer()
    return _analyzer


def parse_rss(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    items = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        source = (item.findtext("source") or "").strip()
        if source and title.endswith(f" - {source}"):
            title = title[: -len(source) - 3].strip()
        else:
            title = re.sub(r"\s+-\s+[^-]{2,60}$", "", title)
        published = None
        if item.findtext("pubDate"):
            try:
                published = parsedate_to_datetime(item.findtext("pubDate"))
                if published.tzinfo is None:
                    published = published.replace(tzinfo=timezone.utc)
            except (TypeError, ValueError):
                published = None
        items.append({"title": title, "source": source, "link": item.findtext("link"),
                      "published": published})
    return items


def score_headlines(items: list[dict], days: int = config.SENTIMENT_DAYS, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)
    seen, kept = set(), []
    for it in items:
        key = it["title"].lower()
        if not it["title"] or key in seen:
            continue
        if it["published"] is not None and it["published"] < cutoff:
            continue
        seen.add(key)
        kept.append(it)
    if not kept:
        return {"score": None, "n_headlines": 0, "headlines": []}
    analyzer = _get_analyzer()
    for it in kept:
        it["compound"] = round(analyzer.polarity_scores(it["title"])["compound"], 4)
    score = sum(it["compound"] for it in kept) / len(kept)
    kept.sort(key=lambda it: it["published"] or cutoff, reverse=True)
    return {
        "score": round(score, 4),
        "n_headlines": len(kept),
        "headlines": [{**it, "published": it["published"].isoformat() if it["published"] else None}
                      for it in kept[:10]],
    }


def _fetch_items(ticker: str, days: int) -> list[dict]:
    import requests
    base = config.NEWS_QUERIES.get(ticker) or f'"{config.company_name(ticker)}"'
    query = f"{base} when:{days}d"
    resp = requests.get(RSS_URL.format(q=quote_plus(query)), timeout=10,
                        headers={"User-Agent": "Mozilla/5.0 (NGX research project)"})
    resp.raise_for_status()
    return parse_rss(resp.text)


def _cached(key: tuple, fn):
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < config.SENTIMENT_CACHE_MINUTES * 60:
        return hit[1]
    value = fn()
    _cache[key] = (time.time(), value)
    return value


def get_sentiment(ticker: str, days: int = config.SENTIMENT_DAYS) -> dict:
    """Mean VADER compound score of the last `days` days of headlines (used by the decision engine)."""
    ticker = ticker.upper()
    if _provider is not None:
        return _provider(ticker)

    def build():
        try:
            result = score_headlines(_fetch_items(ticker, days), days=days)
        except Exception as exc:
            result = {"score": None, "n_headlines": 0, "headlines": [], "error": f"{type(exc).__name__}: {exc}"}
        result.update(window_days=days, source="Google News RSS + VADER")
        return result

    return _cached(("sentiment", ticker, days), build)


def label_for(score: float | None) -> str:
    if score is None:
        return "Neutral"
    return "Positive" if score >= 0.05 else "Negative" if score <= -0.05 else "Neutral"


def get_news(ticker: str, days: int = 30) -> dict:
    """All headlines from the last `days` days, each scored with VADER (for the news panel)."""
    ticker = ticker.upper()

    def build():
        try:
            items = _fetch_items(ticker, days)
        except Exception as exc:
            return {"articles": [], "error": f"{type(exc).__name__}: {exc}"}
        analyzer = _get_analyzer()
        seen, articles = set(), []
        for it in items:
            if not it["title"] or it["title"].lower() in seen:
                continue
            seen.add(it["title"].lower())
            c = round(analyzer.polarity_scores(it["title"])["compound"], 4)
            articles.append({"title": it["title"], "url": it.get("link"), "link": it.get("link"),
                             "source": it.get("source") or "Google News",
                             "published": it["published"].isoformat() if it.get("published") else None,
                             "sentiment": label_for(c), "score": c, "compound": c})
        articles.sort(key=lambda a: a["published"] or "", reverse=True)
        return {"articles": articles}

    return _cached(("news", ticker, days), build)
