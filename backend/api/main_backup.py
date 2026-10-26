from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.sentiment.main import get_news_sentiment
from backend.market_data import (
    get_live_stock,
    fetch_all_stocks,
    get_market_status,
)

import pandas as pd
from pathlib import Path


# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="NGX Stock Market Trend Prediction and Decision Support System",
    description=(
        "FastAPI backend for the NGX stock prediction "
        "and decision support system"
    ),
    version="1.0.0",
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# DIRECTORIES
# =========================================================

FEATURE_DIR = Path("backend/data/features")

DECISION_FILE = Path(
    "backend/data/decisions/latest_hybrid_decisions.csv"
)


# =========================================================
# COMPANIES
# =========================================================

COMPANIES = [
    {
        "ticker": "AIRTELAFRI",
        "name": "Airtel Africa",
        "sector": "Telecommunications",
    },
    {
        "ticker": "BUACEMENT",
        "name": "BUA Cement",
        "sector": "Manufacturing",
    },
    {
        "ticker": "DANGCEM",
        "name": "Dangote Cement",
        "sector": "Manufacturing",
    },
    {
        "ticker": "GTCO",
        "name": "Guaranty Trust Holding Company",
        "sector": "Banking",
    },
    {
        "ticker": "MTNN",
        "name": "MTN Nigeria",
        "sector": "Telecommunications",
    },
    {
        "ticker": "NESTLE",
        "name": "Nestlé Nigeria",
        "sector": "FMCG",
    },
    {
        "ticker": "SEPLAT",
        "name": "Seplat Energy",
        "sector": "Oil and Gas",
    },
    {
        "ticker": "TOTAL",
        "name": "TotalEnergies Marketing Nigeria",
        "sector": "Oil and Gas",
    },
    {
        "ticker": "UNILEVER",
        "name": "Unilever Nigeria",
        "sector": "FMCG",
    },
    {
        "ticker": "ZENITHBANK",
        "name": "Zenith Bank",
        "sector": "Banking",
    },
]


VALID_TICKERS = {
    company["ticker"]
    for company in COMPANIES
}


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():
    return {
        "system": (
            "NGX Stock Market Trend Prediction "
            "and Decision Support System"
        ),
        "status": "running",
        "backend": "FastAPI",
        "market_data": "Kobo Terminal",
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "message": "NGX Advisory API is running",
        "backend": "FastAPI",
    }


# =========================================================
# COMPANIES
# =========================================================

@app.get("/api/companies")
def get_companies():
    return {
        "count": len(COMPANIES),
        "companies": COMPANIES,
    }


# =========================================================
# MARKET STATUS
# =========================================================

@app.get("/api/market-status")
def market_status():
    try:
        return get_market_status()

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


# =========================================================
# LIVE MARKET DATA — ALL STOCKS
# =========================================================

@app.get("/api/market-data")
def market_data():

    try:

        result = fetch_all_stocks()

        return {
            "source": "Kobo Terminal",
            "data_status": "Delayed Data",
            "count": len(result["stocks"]),
            "cached": result.get("cached", False),
            "fallback": result.get("fallback", False),
            "fetched_at": result["fetched_at"],
            "stocks": result["stocks"],
        }

    except Exception as error:

        raise HTTPException(
            status_code=502,
            detail=str(error),
        )


# =========================================================
# LATEST STOCK DATA
#
# CURRENT PRICE = KOBO
# INDICATORS = HISTORICAL FEATURE DATA
# =========================================================

@app.get("/api/stocks/{ticker}")
def get_stock(ticker: str):

    ticker = ticker.upper()

    if ticker not in VALID_TICKERS:
        raise HTTPException(
            status_code=404,
            detail=f"Company {ticker} not found",
        )

    # -----------------------------------------------------
    # GET LIVE/DELAYED MARKET PRICE
    # -----------------------------------------------------

    try:

        live_stock = get_live_stock(ticker)

    except Exception as error:

        raise HTTPException(
            status_code=502,
            detail=(
                f"Unable to retrieve market data "
                f"for {ticker}: {error}"
            ),
        )

    # -----------------------------------------------------
    # GET HISTORICAL FEATURE DATA
    # -----------------------------------------------------

    feature_file = (
        FEATURE_DIR /
        f"{ticker}_features.csv"
    )

    if not feature_file.exists():

        raise HTTPException(
            status_code=404,
            detail=f"Feature data not found for {ticker}",
        )

    try:

        df = pd.read_csv(feature_file)

        if df.empty:

            raise HTTPException(
                status_code=404,
                detail=f"No stock data available for {ticker}",
            )

        df["timestamp"] = pd.to_datetime(
            df["timestamp"]
        )

        df = df.sort_values("timestamp")

        latest = df.iloc[-1]

        # -------------------------------------------------
        # CURRENT MARKET PRICE
        # -------------------------------------------------

        live_price = live_stock.get("price")

        if live_price is None:

            raise RuntimeError(
                f"Kobo did not return a current price "
                f"for {ticker}"
            )

        live_price = float(live_price)

        # -------------------------------------------------
        # RETURN COMBINED DATA
        # -------------------------------------------------

        return {

            "ticker": ticker,

            # Current feed timestamp
            "timestamp": live_stock.get(
                "feed_timestamp"
            ),

            # CURRENT PRICE FROM KOBO
            "price": live_price,

            # Historical OHLC values used by the existing
            # technical-analysis dataset
            "open": float(
                latest["open"]
            ),

            "high": float(
                latest["high"]
            ),

            "low": float(
                latest["low"]
            ),

            "volume": live_stock.get(
                "volume",
                float(latest["volume"])
            ),

            # -------------------------------------------------
            # MARKET DATA INFORMATION
            # -------------------------------------------------

            "market_data": {

                "source": "Kobo Terminal",

                "status": "Delayed Data",

                "change_percent": live_stock.get(
                    "change_percent"
                ),

                "feed_timestamp": live_stock.get(
                    "feed_timestamp"
                ),

                "cached": live_stock.get(
                    "cached",
                    False
                ),

                "fallback": live_stock.get(
                    "fallback",
                    False
                ),
            },

            # -------------------------------------------------
            # TECHNICAL INDICATORS
            # -------------------------------------------------

            "indicators": {

                "rsi_14": float(
                    latest["rsi_14"]
                ),

                "sma_20": float(
                    latest["sma_20"]
                ),

                "sma_50": float(
                    latest["sma_50"]
                ),

                "ema_20": float(
                    latest["ema_20"]
                ),

                "macd": float(
                    latest["macd"]
                ),

                "macd_signal": float(
                    latest["macd_signal"]
                ),

                "macd_histogram": float(
                    latest["macd_histogram"]
                ),

                "bb_upper": float(
                    latest["bb_upper"]
                ),

                "bb_middle": float(
                    latest["bb_middle"]
                ),

                "bb_lower": float(
                    latest["bb_lower"]
                ),

                "atr_14": float(
                    latest["atr_14"]
                ),
            },

            # Important for transparency
            "indicator_timestamp": (
                latest["timestamp"].isoformat()
            ),

            "price_source": "Kobo Terminal",

            "historical_data_source": (
                "Project historical feature dataset"
            ),
        }

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


# =========================================================
# HISTORICAL STOCK DATA
# =========================================================

@app.get("/api/stocks/{ticker}/history")
def get_stock_history(
    ticker: str,
    timeframe: str = "1y"
):
    ticker = ticker.upper()
    timeframe = timeframe.lower()

    if ticker not in VALID_TICKERS:
        raise HTTPException(
            status_code=404,
            detail=f"Company {ticker} not found"
        )

    valid_timeframes = [
        "1m",
        "3m",
        "6m",
        "1y",
        "5y",
        "all"
    ]

    if timeframe not in valid_timeframes:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid timeframe. "
                "Use: 1m, 3m, 6m, 1y, 5y, or all"
            )
        )

    feature_file = (
        FEATURE_DIR /
        f"{ticker}_features.csv"
    )

    if not feature_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Historical data not found for {ticker}"
        )

    try:
        df = pd.read_csv(feature_file)

        if df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No historical data available for {ticker}"
            )

        df["timestamp"] = pd.to_datetime(
            df["timestamp"]
        )

        df = df.sort_values("timestamp")

        latest_date = df["timestamp"].max()

        if timeframe == "1m":
            start_date = (
                latest_date -
                pd.Timedelta(days=30)
            )
            df = df[
                df["timestamp"] >= start_date
            ]

        elif timeframe == "3m":
            start_date = (
                latest_date -
                pd.Timedelta(days=90)
            )
            df = df[
                df["timestamp"] >= start_date
            ]

        elif timeframe == "6m":
            start_date = (
                latest_date -
                pd.Timedelta(days=180)
            )
            df = df[
                df["timestamp"] >= start_date
            ]

        elif timeframe == "1y":
            start_date = (
                latest_date -
                pd.Timedelta(days=365)
            )
            df = df[
                df["timestamp"] >= start_date
            ]

        elif timeframe == "5y":
            start_date = (
                latest_date -
                pd.Timedelta(days=365 * 5)
            )
            df = df[
                df["timestamp"] >= start_date
            ]

        # "all" needs no filtering

        df = df.dropna(
            subset=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume"
            ]
        )

        history = []

        for _, row in df.iterrows():
            history.append({
                "timestamp":
                    row["timestamp"].isoformat(),

                "open":
                    float(row["open"]),

                "high":
                    float(row["high"]),

                "low":
                    float(row["low"]),

                "close":
                    float(row["close"]),

                "volume":
                    float(row["volume"])
            })

        return {
            "ticker": ticker,
            "timeframe": timeframe,
            "count": len(history),
            "data": history
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# =========================================================
# NEWS SENTIMENT
# =========================================================

@app.get("/api/stocks/{ticker}/sentiment")
def get_sentiment(ticker: str):

    ticker = ticker.upper()

    if ticker not in VALID_TICKERS:

        raise HTTPException(
            status_code=404,
            detail=f"Company {ticker} not found",
        )

    try:

        sentiment = get_news_sentiment(
            ticker
        )

        return sentiment

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


# =========================================================
# DECISIONS
#
# IMPORTANT:
# Current price is refreshed from Kobo.
# Existing model predictions remain unchanged until
# the models are retrained.
# =========================================================

@app.get("/api/decisions")
def get_all_decisions():

    if not DECISION_FILE.exists():

        raise HTTPException(
            status_code=404,
            detail=(
                "Hybrid decision file not found. "
                "Run the decision engine first."
            ),
        )

    try:

        df = pd.read_csv(
            DECISION_FILE
        )

        if df.empty:

            raise HTTPException(
                status_code=404,
                detail="No decision data available",
            )

        decisions = []

        for _, row in df.iterrows():

            ticker = str(
                row["ticker"]
            ).upper()

            # ---------------------------------------------
            # GET CURRENT KOBO PRICE
            # ---------------------------------------------

            try:

                live_stock = get_live_stock(
                    ticker
                )

                live_price = float(
                    live_stock["price"]
                )

            except Exception:

                # Keep model result if feed temporarily fails
                live_stock = None

                live_price = float(
                    row["current_price"]
                )

            # ---------------------------------------------
            # COPY EXISTING MODEL DECISION
            # ---------------------------------------------

            decision = row.to_dict()

            old_price = float(
                row["current_price"]
            )

            predicted_price = float(
                row["predicted_price"]
            )

            # ---------------------------------------------
            # RECALCULATE EXPECTED CHANGE
            # ---------------------------------------------

            if live_price != 0:

                predicted_change = (
                    (
                        predicted_price
                        - live_price
                    )
                    / live_price
                ) * 100

            else:

                predicted_change = 0

            # ---------------------------------------------
            # UPDATE LIVE FIELDS
            # ---------------------------------------------

            decision["current_price"] = (
                live_price
            )

            decision["predicted_change_pct"] = (
                predicted_change
            )

            decision["price_source"] = (
                "Kobo Terminal"
            )

            if live_stock:

                decision["market_data_timestamp"] = (
                    live_stock.get(
                        "feed_timestamp"
                    )
                )

                decision["data_status"] = (
                    "Delayed Data"
                )

            else:

                decision["market_data_timestamp"] = (
                    None
                )

                decision["data_status"] = (
                    "Model Data"
                )

            decision["previous_model_price"] = (
                old_price
            )

            decisions.append(
                decision
            )

        return {

            "count": len(decisions),

            "data_status": "Delayed Data",

            "decisions": decisions,

        }

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


# =========================================================
# SINGLE DECISION
# =========================================================

@app.get("/api/decisions/{ticker}")
def get_company_decision(
    ticker: str
):

    ticker = ticker.upper()

    if ticker not in VALID_TICKERS:

        raise HTTPException(
            status_code=404,
            detail=f"Company {ticker} not found",
        )

    if not DECISION_FILE.exists():

        raise HTTPException(
            status_code=404,
            detail="Hybrid decision file not found",
        )

    try:

        df = pd.read_csv(
            DECISION_FILE
        )

        result = df[
            df["ticker"]
            .astype(str)
            .str.upper()
            == ticker
        ]

        if result.empty:

            raise HTTPException(
                status_code=404,
                detail=(
                    f"No decision found for {ticker}"
                ),
            )

        row = result.iloc[0]

        # ---------------------------------------------
        # CURRENT KOBO PRICE
        # ---------------------------------------------

        try:

            live_stock = get_live_stock(
                ticker
            )

            current_price = float(
                live_stock["price"]
            )

            feed_timestamp = (
                live_stock.get(
                    "feed_timestamp"
                )
            )

            data_status = "Delayed Data"

        except Exception:

            live_stock = None

            current_price = float(
                row["current_price"]
            )

            feed_timestamp = None

            data_status = "Model Data"

        predicted_price = float(
            row["predicted_price"]
        )

        # ---------------------------------------------
        # RECALCULATE EXPECTED CHANGE
        # ---------------------------------------------

        if current_price != 0:

            predicted_change = (
                (
                    predicted_price
                    - current_price
                )
                / current_price
            ) * 100

        else:

            predicted_change = 0

        # ---------------------------------------------
        # RETURN
        # ---------------------------------------------

        return {

            "ticker": ticker,

            "timestamp": str(
                row["timestamp"]
            ),

            "current_price":
                current_price,

            "predicted_price":
                predicted_price,

            "predicted_change_pct":
                predicted_change,

            "sentiment_score":
                float(
                    row.get(
                        "sentiment_score",
                        0
                    )
                ),

            "sentiment_label":
                str(
                    row.get(
                        "sentiment_label",
                        "Neutral"
                    )
                ),

            "headline_count":
                int(
                    row.get(
                        "headline_count",
                        0
                    )
                ),

            "rsi_14":
                float(
                    row["rsi_14"]
                ),

            "macd":
                float(
                    row["macd"]
                ),

            "macd_signal":
                float(
                    row["macd_signal"]
                ),

            "sma_20":
                float(
                    row["sma_20"]
                ),

            "sma_50":
                float(
                    row["sma_50"]
                ),

            "rf_prediction":
                int(
                    row["rf_prediction"]
                ),

            "rf_probability_up":
                float(
                    row["rf_probability_up"]
                ),

            "model_agreement":
                bool(
                    row["model_agreement"]
                ),

            # Existing model-engine recommendation
            "recommendation":
                str(
                    row["recommendation"]
                ),

            "confidence":
                str(
                    row["confidence"]
                ),

            "score":
                int(
                    row["score"]
                ),

            "reasons":
                str(
                    row["reasons"]
                ).split(" | "),

            "price_source":
                "Kobo Terminal",

            "data_status":
                data_status,

            "market_data_timestamp":
                feed_timestamp,

        }

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


# =========================================================
# ADVISORY
# =========================================================

@app.get("/api/advisory/{ticker}")
def get_advisory(
    ticker: str
):

    return get_company_decision(
        ticker
    )