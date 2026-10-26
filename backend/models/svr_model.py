import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


INPUT_DIR = Path("backend/data/model_data")
OUTPUT_DIR = Path("backend/data/predictions/svr")


FEATURE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "return",
    "price_change",
    "sma_20",
    "sma_50",
    "ema_20",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_histogram",
    "bb_middle",
    "bb_upper",
    "bb_lower",
    "atr_14"
]


def calculate_mape(actual, predicted):

    actual = np.array(actual)
    predicted = np.array(predicted)

    non_zero = actual != 0

    return (
        np.mean(
            np.abs(
                (actual[non_zero] - predicted[non_zero])
                / actual[non_zero]
            )
        )
        * 100
    )


def run_svr(ticker):

    train_file = INPUT_DIR / f"{ticker}_train.csv"
    test_file = INPUT_DIR / f"{ticker}_test.csv"

    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    train["timestamp"] = pd.to_datetime(
        train["timestamp"]
    )

    test["timestamp"] = pd.to_datetime(
        test["timestamp"]
    )

    train = train.sort_values("timestamp")
    test = test.sort_values("timestamp")

    X_train = train[FEATURE_COLUMNS]
    X_test = test[FEATURE_COLUMNS]

    y_train = train["close"].astype(float)
    y_test = test["close"].astype(float)

    print(f"\nTraining SVR for {ticker}...")
    print(f"   Training records: {len(X_train)}")
    print(f"   Testing records: {len(X_test)}")

    # ---------------------------------------------------------
    # Scale features
    # ---------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # ---------------------------------------------------------
    # Create SVR model
    # ---------------------------------------------------------

    model = SVR(
        kernel="rbf",
        C=100,
        epsilon=0.1,
        gamma="scale"
    )

    # ---------------------------------------------------------
    # Train
    # ---------------------------------------------------------

    model.fit(
        X_train_scaled,
        y_train
    )

    # ---------------------------------------------------------
    # Predict
    # ---------------------------------------------------------

    predictions = model.predict(
        X_test_scaled
    )

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    mape = calculate_mape(
        y_test,
        predictions
    )

    # ---------------------------------------------------------
    # Save predictions
    # ---------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results = pd.DataFrame({
        "timestamp": test["timestamp"],
        "actual_close": y_test,
        "predicted_close": predictions
    })

    results["error"] = (
        results["actual_close"]
        - results["predicted_close"]
    )

    output_file = (
        OUTPUT_DIR /
        f"{ticker}_svr_predictions.csv"
    )

    results.to_csv(
        output_file,
        index=False
    )

    # ---------------------------------------------------------
    # Display results
    # ---------------------------------------------------------

    print(f"   MAE:  {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAPE: {mape:.2f}%")

    print(
        f"   ✅ Predictions saved: {output_file}"
    )

    return {
        "ticker": ticker,
        "mae": mae,
        "rmse": rmse,
        "mape": mape
    }


def main():

    print("=" * 70)
    print("NGX SVR STOCK PRICE PREDICTION")
    print("=" * 70)

    train_files = sorted(
        INPUT_DIR.glob("*_train.csv")
    )

    if not train_files:

        print(
            "❌ No training datasets found."
        )

        return

    results = []

    for train_file in train_files:

        ticker = train_file.stem.replace(
            "_train",
            ""
        )

        try:

            result = run_svr(ticker)

            results.append(result)

        except Exception as error:

            print(
                f"   ❌ Error processing "
                f"{ticker}: {error}"
            )

    # ---------------------------------------------------------
    # Save summary
    # ---------------------------------------------------------

    if results:

        summary = pd.DataFrame(results)

        summary_file = (
            OUTPUT_DIR /
            "svr_summary.csv"
        )

        summary.to_csv(
            summary_file,
            index=False
        )

        print("\n" + "=" * 70)
        print("SVR MODEL SUMMARY")
        print("=" * 70)

        print(
            summary.to_string(
                index=False
            )
        )

        print(
            f"\n✅ Summary saved: {summary_file}"
        )

    print("\n" + "=" * 70)
    print("✅ SVR PHASE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()