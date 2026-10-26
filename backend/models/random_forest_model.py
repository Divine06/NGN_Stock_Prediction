import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


INPUT_DIR = Path("backend/data/model_data")
OUTPUT_DIR = Path("backend/data/predictions/random_forest")


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

TARGET_COLUMN = "target"


def run_random_forest(ticker):

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

    # ---------------------------------------------------------
    # Select features and target
    # ---------------------------------------------------------

    X_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET_COLUMN]

    X_test = test[FEATURE_COLUMNS]
    y_test = test[TARGET_COLUMN]

    print(f"\nTraining Random Forest for {ticker}...")
    print(f"   Training records: {len(X_train)}")
    print(f"   Testing records: {len(X_test)}")

    # ---------------------------------------------------------
    # Create model
    # ---------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    # ---------------------------------------------------------
    # Train
    # ---------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )

    # ---------------------------------------------------------
    # Predict
    # ---------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    cm = confusion_matrix(
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
        "actual_target": y_test,
        "predicted_target": predictions,
        "probability_up": probabilities
    })

    output_file = (
        OUTPUT_DIR /
        f"{ticker}_random_forest_predictions.csv"
    )

    results.to_csv(
        output_file,
        index=False
    )

    # ---------------------------------------------------------
    # Feature importance
    # ---------------------------------------------------------

    importance = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": model.feature_importances_
    })

    importance = importance.sort_values(
        "importance",
        ascending=False
    )

    importance_file = (
        OUTPUT_DIR /
        f"{ticker}_feature_importance.csv"
    )

    importance.to_csv(
        importance_file,
        index=False
    )

    # ---------------------------------------------------------
    # Display results
    # ---------------------------------------------------------

    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1 Score:  {f1:.4f}")

    print("\n   Confusion Matrix:")
    print(cm)

    print("\n   Top 5 Important Features:")

    print(
        importance.head(5)
        .to_string(index=False)
    )

    print(
        f"\n   ✅ Predictions saved: "
        f"{output_file}"
    )

    return {
        "ticker": ticker,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def main():

    print("=" * 70)
    print("NGX RANDOM FOREST STOCK TREND PREDICTION")
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

            result = run_random_forest(
                ticker
            )

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

        summary = pd.DataFrame(
            results
        )

        summary_file = (
            OUTPUT_DIR /
            "random_forest_summary.csv"
        )

        summary.to_csv(
            summary_file,
            index=False
        )

        print("\n" + "=" * 70)
        print("RANDOM FOREST MODEL SUMMARY")
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
    print("✅ RANDOM FOREST PHASE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()