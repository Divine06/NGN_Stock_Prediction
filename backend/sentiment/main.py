import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime


analyzer = SentimentIntensityAnalyzer()


COMPANIES = {
    "AIRTELAFRI": "Airtel Africa",
    "BUACEMENT": "BUA Cement",
    "DANGCEM": "Dangote Cement",
    "GTCO": "GTCO",
    "MTNN": "MTN Nigeria",
    "NESTLE": "Nestle Nigeria",
    "SEPLAT": "Seplat Energy",
    "TOTAL": "TotalEnergies Nigeria",
    "UNILEVER": "Unilever Nigeria",
    "ZENITHBANK": "Zenith Bank"
}


def analyse_sentiment(text):

    scores = analyzer.polarity_scores(text)

    compound = scores["compound"]

    if compound >= 0.05:
        label = "Positive"

    elif compound <= -0.05:
        label = "Negative"

    else:
        label = "Neutral"

    return {
        "compound": compound,
        "positive": scores["pos"],
        "negative": scores["neg"],
        "neutral": scores["neu"],
        "label": label
    }


def get_news_sentiment(ticker):

    if ticker not in COMPANIES:
        raise ValueError(
            f"Unknown company ticker: {ticker}"
        )

    company_name = COMPANIES[ticker]

    query = f"{company_name} Nigeria stock"

    url = (
        "https://news.google.com/rss/search?"
        f"q={query.replace(' ', '+')}"
        "&hl=en-NG"
        "&gl=NG"
        "&ceid=NG:en"
    )

    feed = feedparser.parse(url)

    articles = []

    for entry in feed.entries[:10]:

        title = entry.get(
            "title",
            ""
        )

        link = entry.get(
            "link",
            ""
        )

        published = entry.get(
            "published",
            ""
        )

        sentiment = analyse_sentiment(
            title
        )

        articles.append({
            "title": title,
            "link": link,
            "published": published,
            "sentiment": sentiment
        })

    if not articles:

        return {
            "ticker": ticker,
            "company": company_name,
            "sentiment_score": 0,
            "sentiment_label": "Neutral",
            "headline_count": 0,
            "headlines": [],
            "updated_at": datetime.utcnow().isoformat()
        }

    compound_scores = [
        article["sentiment"]["compound"]
        for article in articles
    ]

    average_score = sum(
        compound_scores
    ) / len(compound_scores)

    if average_score >= 0.05:
        overall_label = "Positive"

    elif average_score <= -0.05:
        overall_label = "Negative"

    else:
        overall_label = "Neutral"

    return {
        "ticker": ticker,
        "company": company_name,
        "sentiment_score": round(
            average_score,
            4
        ),
        "sentiment_label": overall_label,
        "headline_count": len(articles),
        "headlines": articles,
        "updated_at": datetime.utcnow().isoformat()
    }