import pandas as pd
import numpy as np
from pathlib import Path

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error


INPUT_DIR = Path("backend/data/model_data")
OUTPUT_DIR = Path("backend/data/predictions/arima")


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


def run_arima(ticker):

    train_file = INPUT_DIR / f"{ticker}_train.csv"
    test_file = INPUT_DIR / f"{ticker}_test.csv"

    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    train["timestamp"] = pd.to_datetime(train["timestamp"])
    test["timestamp"] = pd.to_datetime(test["timestamp"])

    train = train.sort_values("timestamp")
    test = test.sort_values("timestamp")

    train_close = train["close"].astype(float)
    test_close = test["close"].astype(float)

    print(f"\nTraining ARIMA for {ticker}...")
    print(f"   Training records: {len(train_close)}")
    print(f"   Testing records: {len(test_close)}")

    # ---------------------------------------------------------
    # ARIMA model
    # ---------------------------------------------------------

    model = ARIMA(
        train_close,
        order=(1, 1, 1)
    )

    fitted_model = model.fit()

    # ---------------------------------------------------------
    # Forecast the complete test period
    # ---------------------------------------------------------

    forecast = fitted_model.forecast(
        steps=len(test_close)
    )

    forecast = np.array(forecast)

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------

    mae = mean_absolute_error(
        test_close,
        forecast
    )

    rmse = np.sqrt(
        mean_squared_error(
            test_close,
            forecast
        )
    )

    mape = calculate_mape(
        test_close,
        forecast
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
        "actual_close": test_close,
        "predicted_close": forecast
    })

    results["error"] = (
        results["actual_close"]
        - results["predicted_close"]
    )

    output_file = (
        OUTPUT_DIR /
        f"{ticker}_arima_predictions.csv"
    )

    results.to_csv(
        output_file,
        index=False
    )

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
    print("NGX ARIMA STOCK PRICE PREDICTION")
    print("=" * 70)

    train_files = sorted(
        INPUT_DIR.glob("*_train.csv")
    )

    if not train_files:
        print("❌ No training datasets found.")
        return

    results = []

    for train_file in train_files:

        ticker = train_file.stem.replace(
            "_train",
            ""
        )

        try:

            result = run_arima(ticker)

            results.append(result)

        except Exception as error:

            print(
                f"   ❌ Error processing "
                f"{ticker}: {error}"
            )

    # ---------------------------------------------------------
    # Save model summary
    # ---------------------------------------------------------

    if results:

        summary = pd.DataFrame(results)

        summary_file = (
            OUTPUT_DIR /
            "arima_summary.csv"
        )

        summary.to_csv(
            summary_file,
            index=False
        )

        print("\n" + "=" * 70)
        print("ARIMA MODEL SUMMARY")
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
    print("✅ ARIMA PHASE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()