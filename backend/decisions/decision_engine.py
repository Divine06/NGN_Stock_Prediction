import pandas as pd
from pathlib import Path

from backend.sentiment.main import get_news_sentiment
from backend.market_data import get_live_stock


# ============================================================
# DIRECTORIES
# ============================================================

FEATURE_DIR = Path(
    "backend/data/features"
)

LSTM_DIR = Path(
    "backend/data/predictions/lstm"
)

RANDOM_FOREST_DIR = Path(
    "backend/data/predictions/random_forest"
)

OUTPUT_DIR = Path(
    "backend/data/decisions"
)


# ============================================================
# COMPANIES
# ============================================================

COMPANIES = [
    "AIRTELAFRI",
    "BUACEMENT",
    "DANGCEM",
    "GTCO",
    "MTNN",
    "NESTLE",
    "SEPLAT",
    "TOTAL",
    "UNILEVER",
    "ZENITHBANK",
]


# ============================================================
# DECISION CALCULATION
# ============================================================

def calculate_decision(
    current_price,
    predicted_price,
    rsi,
    macd,
    macd_signal,
    sma_20,
    sma_50,
    rf_prediction,
    rf_probability,
    sentiment_score,
):

    reasons = []

    score = 0

    # ========================================================
    # 1. LSTM PREDICTED PRICE MOVEMENT
    # ========================================================

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

    if predicted_change > 2:

        score += 2

        reasons.append(
            "LSTM predicts a positive price movement"
        )

    elif predicted_change > 0:

        score += 1

        reasons.append(
            "LSTM predicts a small positive price movement"
        )

    elif predicted_change < -2:

        score -= 2

        reasons.append(
            "LSTM predicts a negative price movement"
        )

    else:

        reasons.append(
            "LSTM predicts limited price movement"
        )

    # ========================================================
    # 2. RANDOM FOREST DIRECTION
    # ========================================================

    if rf_prediction == 1:

        score += 2

        reasons.append(
            "Random Forest predicts upward movement "
            f"(probability: {rf_probability:.1%})"
        )

    else:

        score -= 2

        reasons.append(
            "Random Forest predicts downward movement "
            f"(probability of upward movement: "
            f"{rf_probability:.1%})"
        )

    # ========================================================
    # 3. NEWS SENTIMENT
    # ========================================================

    if sentiment_score >= 0.25:

        score += 2

        reasons.append(
            "News sentiment is strongly positive "
            f"(score: {sentiment_score:.2f})"
        )

    elif sentiment_score >= 0.05:

        score += 1

        reasons.append(
            "News sentiment is positive "
            f"(score: {sentiment_score:.2f})"
        )

    elif sentiment_score <= -0.25:

        score -= 2

        reasons.append(
            "News sentiment is strongly negative "
            f"(score: {sentiment_score:.2f})"
        )

    elif sentiment_score <= -0.05:

        score -= 1

        reasons.append(
            "News sentiment is negative "
            f"(score: {sentiment_score:.2f})"
        )

    else:

        reasons.append(
            "News sentiment is neutral "
            f"(score: {sentiment_score:.2f})"
        )

    # ========================================================
    # 4. RSI
    # ========================================================

    if rsi < 30:

        score += 1

        reasons.append(
            "RSI indicates potentially oversold conditions"
        )

    elif rsi > 70:

        score -= 1

        reasons.append(
            "RSI indicates potentially overbought conditions"
        )

    else:

        reasons.append(
            "RSI is within a neutral range"
        )

    # ========================================================
    # 5. MACD
    # ========================================================

    if macd > macd_signal:

        score += 1

        reasons.append(
            "MACD is above its signal line"
        )

    else:

        score -= 1

        reasons.append(
            "MACD is below its signal line"
        )

    # ========================================================
    # 6. MOVING AVERAGES
    # ========================================================

    if sma_20 > sma_50:

        score += 1

        reasons.append(
            "20-day moving average is above "
            "the 50-day moving average"
        )

    else:

        score -= 1

        reasons.append(
            "20-day moving average is below "
            "the 50-day moving average"
        )

    # ========================================================
    # 7. MODEL AGREEMENT
    # ========================================================

    lstm_direction = (
        1 if predicted_change > 0 else 0
    )

    if lstm_direction == rf_prediction:

        reasons.append(
            "LSTM and Random Forest agree "
            "on market direction"
        )

        model_agreement = True

    else:

        reasons.append(
            "LSTM and Random Forest disagree "
            "on market direction"
        )

        model_agreement = False

    # ========================================================
    # 8. FINAL RECOMMENDATION
    # ========================================================

    if score >= 1:

        recommendation = "BUY"

    elif score <= -1:

        recommendation = "SELL"

    else:

        recommendation = "HOLD"

    # ========================================================
    # 9. CONFIDENCE
    # ========================================================

    if model_agreement and abs(score) >= 4:

        confidence = "High"

    elif abs(score) >= 3:

        confidence = "Moderate"

    else:

        confidence = "Low"

    return (
        recommendation,
        confidence,
        score,
        predicted_change,
        model_agreement,
        reasons,
    )


# ============================================================
# PROCESS ONE COMPANY
# ============================================================

def process_company(ticker):

    # --------------------------------------------------------
    # FILES
    # --------------------------------------------------------

    feature_file = (
        FEATURE_DIR /
        f"{ticker}_features.csv"
    )

    lstm_file = (
        LSTM_DIR /
        f"{ticker}_lstm_predictions.csv"
    )

    rf_file = (
        RANDOM_FOREST_DIR /
        f"{ticker}_random_forest_predictions.csv"
    )

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    if not feature_file.exists():

        raise FileNotFoundError(
            f"Feature file not found: {feature_file}"
        )

    if not lstm_file.exists():

        raise FileNotFoundError(
            f"LSTM prediction file not found: {lstm_file}"
        )

    if not rf_file.exists():

        raise FileNotFoundError(
            f"Random Forest prediction file not found: {rf_file}"
        )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    features = pd.read_csv(
        feature_file
    )

    lstm = pd.read_csv(
        lstm_file
    )

    rf = pd.read_csv(
        rf_file
    )

    # --------------------------------------------------------
    # TIMESTAMPS
    # --------------------------------------------------------

    features["timestamp"] = pd.to_datetime(
        features["timestamp"]
    )

    lstm["timestamp"] = pd.to_datetime(
        lstm["timestamp"]
    )

    rf["timestamp"] = pd.to_datetime(
        rf["timestamp"]
    )

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    features = features.sort_values(
        "timestamp"
    )

    lstm = lstm.sort_values(
        "timestamp"
    )

    rf = rf.sort_values(
        "timestamp"
    )

    # ========================================================
    # HISTORICAL TECHNICAL DATA
    # ========================================================

    latest = features.iloc[-1]

    rsi = float(
        latest["rsi_14"]
    )

    macd = float(
        latest["macd"]
    )

    macd_signal = float(
        latest["macd_signal"]
    )

    sma_20 = float(
        latest["sma_20"]
    )

    sma_50 = float(
        latest["sma_50"]
    )

    # ========================================================
    # CURRENT MARKET PRICE
    #
    # THIS IS THE MAJOR FIX.
    #
    # We no longer use:
    # latest["close"]
    #
    # We use Kobo Terminal.
    # ========================================================

    try:

        live_stock = get_live_stock(
            ticker
        )

        current_price = float(
            live_stock["price"]
        )

        market_timestamp = (
            live_stock.get(
                "feed_timestamp"
            )
        )

        price_source = (
            "Kobo Terminal"
        )

    except Exception as error:

        print(
            f"   ⚠️ Live price unavailable "
            f"for {ticker}: {error}"
        )

        # Fallback to historical data so the
        # entire decision process does not crash.
        current_price = float(
            latest["close"]
        )

        market_timestamp = None

        price_source = (
            "Historical Feature Dataset (Fallback)"
        )

    # ========================================================
    # LSTM PREDICTION
    # ========================================================

    latest_lstm = lstm.iloc[-1]

    predicted_price = float(
        latest_lstm["predicted_close"]
    )

    # ========================================================
    # RANDOM FOREST
    # ========================================================

    latest_rf = rf.iloc[-1]

    rf_prediction = int(
        latest_rf["predicted_target"]
    )

    if "probability_up" in rf.columns:

        rf_probability = float(
            latest_rf["probability_up"]
        )

    else:

        rf_probability = (
            1.0
            if rf_prediction == 1
            else 0.0
        )

    # ========================================================
    # NEWS SENTIMENT
    # ========================================================

    try:

        sentiment_data = get_news_sentiment(
            ticker
        )

        sentiment_score = float(
            sentiment_data[
                "sentiment_score"
            ]
        )

        sentiment_label = (
            sentiment_data[
                "sentiment_label"
            ]
        )

        headline_count = int(
            sentiment_data[
                "headline_count"
            ]
        )

    except Exception as error:

        print(
            f"   ⚠️ Sentiment unavailable "
            f"for {ticker}: {error}"
        )

        sentiment_score = 0.0

        sentiment_label = "Neutral"

        headline_count = 0

    # ========================================================
    # GENERATE DECISION
    # ========================================================

    (
        recommendation,
        confidence,
        score,
        predicted_change,
        model_agreement,
        reasons,
        ) = calculate_decision(
        current_price,
        predicted_price,
        rsi,
        macd,
        macd_signal,
        sma_20,
        sma_50,
        rf_prediction,
        rf_probability,
        sentiment_score
    )

    # ========================================================
    # RESULT
    # ========================================================

    return {

        "ticker": ticker,

        "timestamp":
            str(
                latest["timestamp"]
            ),

        "market_data_timestamp":
            market_timestamp,

        "price_source":
            price_source,

        "current_price":
            current_price,

        "predicted_price":
            predicted_price,

        "predicted_change_pct":
            predicted_change,

        "sentiment_score":
            sentiment_score,

        "sentiment_label":
            sentiment_label,

        "headline_count":
            headline_count,

        "rsi_14":
            rsi,

        "macd":
            macd,

        "macd_signal":
            macd_signal,

        "sma_20":
            sma_20,

        "sma_50":
            sma_50,

        "rf_prediction":
            rf_prediction,

        "rf_probability_up":
            rf_probability,

        "model_agreement":
            model_agreement,

        "recommendation":
            recommendation,

        "confidence":
            confidence,

        "score":
            score,

        "reasons":
            " | ".join(reasons),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "NGX HYBRID DECISION SUPPORT ENGINE"
    )

    print(
        "LIVE MARKET PRICE SOURCE: KOBO TERMINAL"
    )

    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results = []

    # ========================================================
    # PROCESS ALL 10 COMPANIES
    # ========================================================

    for ticker in COMPANIES:

        print(
            f"\nProcessing {ticker}..."
        )

        try:

            result = process_company(
                ticker
            )

            results.append(
                result
            )

            print(
                f"   Current price: "
                f"{result['current_price']:.2f}"
            )

            print(
                f"   LSTM predicted price: "
                f"{result['predicted_price']:.2f}"
            )

            print(
                f"   Expected change: "
                f"{result['predicted_change_pct']:.2f}%"
            )

            print(
                f"   RF direction: "
                f"{'UP' if result['rf_prediction'] == 1 else 'DOWN'}"
            )

            print(
                f"   RF probability UP: "
                f"{result['rf_probability_up']:.2%}"
            )

            print(
                f"   Recommendation: "
                f"{result['recommendation']}"
            )

            print(
                f"   Confidence: "
                f"{result['confidence']}"
            )

            print(
                f"   Score: "
                f"{result['score']}"
            )

        except Exception as error:

            print(
                f"   ❌ Error processing "
                f"{ticker}: {error}"
            )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    if results:

        decisions = pd.DataFrame(
            results
        )

        output_file = (
            OUTPUT_DIR /
            "latest_hybrid_decisions.csv"
        )

        decisions.to_csv(
            output_file,
            index=False
        )

        print(
            "\n" + "=" * 70
        )

        print(
            "HYBRID DECISION SUMMARY"
        )

        print(
            "=" * 70
        )

        print(
            decisions[
                [
                    "ticker",
                    "current_price",
                    "predicted_price",
                    "predicted_change_pct",
                    "rf_prediction",
                    "rf_probability_up",
                    "model_agreement",
                    "recommendation",
                    "confidence",
                    "score",
                ]
            ].to_string(
                index=False
            )
        )

        print(
            "\n✅ Hybrid decisions saved:"
        )

        print(
            output_file
        )

    else:

        print(
            "\n❌ No decision results were generated."
        )

    print(
        "\n" + "=" * 70
    )

    print(
        "✅ HYBRID DECISION SUPPORT ENGINE COMPLETE"
    )

    print(
        "=" * 70
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()