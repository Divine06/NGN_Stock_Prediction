import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


INPUT_DIR = Path("backend/data/model_data")
OUTPUT_DIR = Path("backend/data/predictions/lstm")

LOOKBACK = 30


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


def create_sequences(features, target, lookback):

    X = []
    y = []

    for i in range(lookback, len(features)):

        X.append(
            features[
                i - lookback:i
            ]
        )

        y.append(
            target[i]
        )

    return np.array(X), np.array(y)


def run_lstm(ticker):

    train_file = (
        INPUT_DIR /
        f"{ticker}_train.csv"
    )

    test_file = (
        INPUT_DIR /
        f"{ticker}_test.csv"
    )

    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    train["timestamp"] = pd.to_datetime(
        train["timestamp"]
    )

    test["timestamp"] = pd.to_datetime(
        test["timestamp"]
    )

    train = train.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    test = test.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    print(f"\nTraining LSTM for {ticker}...")
    print(f"   Training records: {len(train)}")
    print(f"   Testing records: {len(test)}")

    # ---------------------------------------------------------
    # Use close price as the main prediction series
    # ---------------------------------------------------------

    train_close = (
        train["close"]
        .astype(float)
        .values
        .reshape(-1, 1)
    )

    test_close = (
        test["close"]
        .astype(float)
        .values
        .reshape(-1, 1)
    )

    # ---------------------------------------------------------
    # Scale using TRAINING data only
    # ---------------------------------------------------------

    scaler = MinMaxScaler()

    train_scaled = scaler.fit_transform(
        train_close
    )

    test_scaled = scaler.transform(
        test_close
    )

    # ---------------------------------------------------------
    # Create training sequences
    # ---------------------------------------------------------

    X_train, y_train = create_sequences(
        train_scaled,
        train_scaled.flatten(),
        LOOKBACK
    )

    # ---------------------------------------------------------
    # Create test sequences
    #
    # We need the final LOOKBACK training observations
    # so that the first test prediction has historical context.
    # ---------------------------------------------------------

    combined = np.concatenate(
        [
            train_scaled[-LOOKBACK:],
            test_scaled
        ]
    )

    X_test, y_test = create_sequences(
        combined,
        combined.flatten(),
        LOOKBACK
    )

    print(
        f"   LSTM training sequences: "
        f"{len(X_train)}"
    )

    print(
        f"   LSTM testing sequences: "
        f"{len(X_test)}"
    )

    # ---------------------------------------------------------
    # Build LSTM
    # ---------------------------------------------------------

    model = Sequential([
        LSTM(
            64,
            return_sequences=True,
            input_shape=(
                LOOKBACK,
                1
            )
        ),

        Dropout(0.2),

        LSTM(
            32
        ),

        Dropout(0.2),

        Dense(16, activation="relu"),

        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )

    # ---------------------------------------------------------
    # Early stopping
    # ---------------------------------------------------------

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    # ---------------------------------------------------------
    # Train
    # ---------------------------------------------------------

    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=1,
        shuffle=False
    )

    # ---------------------------------------------------------
    # Predict
    # ---------------------------------------------------------

    predictions_scaled = model.predict(
        X_test,
        verbose=0
    )

    predictions = scaler.inverse_transform(
        predictions_scaled
    ).flatten()

    actual = scaler.inverse_transform(
        y_test.reshape(-1, 1)
    ).flatten()

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------

    mae = mean_absolute_error(
        actual,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predictions
        )
    )

    mape = calculate_mape(
        actual,
        predictions
    )

    # ---------------------------------------------------------
    # Save predictions
    # ---------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Test timestamps correspond directly to
    # the generated test sequences.
    prediction_dates = test[
        "timestamp"
    ].iloc[
        :len(predictions)
    ].values

    results = pd.DataFrame({
        "timestamp": prediction_dates,
        "actual_close": actual,
        "predicted_close": predictions
    })

    results["error"] = (
        results["actual_close"]
        - results["predicted_close"]
    )

    output_file = (
        OUTPUT_DIR /
        f"{ticker}_lstm_predictions.csv"
    )

    results.to_csv(
        output_file,
        index=False
    )

    # ---------------------------------------------------------
    # Save model
    # ---------------------------------------------------------

    model_file = (
        OUTPUT_DIR /
        f"{ticker}_lstm.keras"
    )

    model.save(
        model_file
    )

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    print(f"   MAE:  {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAPE: {mape:.2f}%")

    print(
        f"   ✅ Predictions saved: "
        f"{output_file}"
    )

    print(
        f"   ✅ Model saved: "
        f"{model_file}"
    )

    return {
        "ticker": ticker,
        "mae": mae,
        "rmse": rmse,
        "mape": mape
    }


def main():

    print("=" * 70)
    print("NGX LSTM STOCK PRICE PREDICTION")
    print("=" * 70)

    train_files = sorted(
        INPUT_DIR.glob(
            "*_train.csv"
        )
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

            result = run_lstm(
                ticker
            )

            results.append(
                result
            )

        except Exception as error:

            print(
                f"   ❌ Error processing "
                f"{ticker}: {error}"
            )

    # ---------------------------------------------------------
    # Save summary
    # ---------------------------------------------------------

    if results:

        summary = pd.DataFrame(
            results
        )

        summary_file = (
            OUTPUT_DIR /
            "lstm_summary.csv"
        )

        summary.to_csv(
            summary_file,
            index=False
        )

        print("\n" + "=" * 70)
        print("LSTM MODEL SUMMARY")
        print("=" * 70)

        print(
            summary.to_string(
                index=False
            )
        )

        print(
            f"\n✅ Summary saved: "
            f"{summary_file}"
        )

    print("\n" + "=" * 70)
    print("✅ LSTM PHASE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()