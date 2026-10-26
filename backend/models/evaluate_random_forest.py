import os
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ============================================================
# NGX RANDOM FOREST MODEL EVALUATION
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
    "ZENITHBANK"
]

INPUT_DIR = "backend/data/predictions/random_forest"
OUTPUT_DIR = "backend/data/predictions/random_forest/evaluation"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def evaluate_company(ticker):
    print(f"\nEvaluating {ticker}...")

    file_path = os.path.join(
        INPUT_DIR,
        f"{ticker}_random_forest_predictions.csv"
    )

    if not os.path.exists(file_path):
        print(f"   ❌ File not found: {file_path}")
        return None

    df = pd.read_csv(file_path)

    required_columns = [
        "actual_target",
        "predicted_target"
    ]

    for column in required_columns:
        if column not in df.columns:
            print(f"   ❌ Missing column: {column}")
            return None

    actual = df["actual_target"]
    predicted = df["predicted_target"]

    accuracy = accuracy_score(actual, predicted)

    precision = precision_score(
        actual,
        predicted,
        zero_division=0
    )

    recall = recall_score(
        actual,
        predicted,
        zero_division=0
    )

    f1 = f1_score(
        actual,
        predicted,
        zero_division=0
    )

    cm = confusion_matrix(actual, predicted)

    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1 Score:  {f1:.4f}")

    print("   Confusion Matrix:")
    print(cm)

    return {
        "ticker": ticker,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_negative": cm[0][0] if cm.shape == (2, 2) else 0,
        "false_positive": cm[0][1] if cm.shape == (2, 2) else 0,
        "false_negative": cm[1][0] if cm.shape == (2, 2) else 0,
        "true_positive": cm[1][1] if cm.shape == (2, 2) else 0
    }


def main():

    print("=" * 70)
    print("NGX RANDOM FOREST MODEL EVALUATION")
    print("=" * 70)

    results = []

    for ticker in COMPANIES:
        result = evaluate_company(ticker)

        if result is not None:
            results.append(result)

    if not results:
        print("\n❌ No evaluation results generated.")
        return

    results_df = pd.DataFrame(results)

    output_file = os.path.join(
        OUTPUT_DIR,
        "random_forest_evaluation.csv"
    )

    results_df.to_csv(output_file, index=False)

    print("\n" + "=" * 70)
    print("AVERAGE RANDOM FOREST PERFORMANCE")
    print("=" * 70)

    print(
        results_df[
            ["accuracy", "precision", "recall", "f1_score"]
        ].mean()
    )

    print("\n" + "=" * 70)
    print("BEST PERFORMING COMPANIES")
    print("=" * 70)

    best_accuracy = results_df.loc[
        results_df["accuracy"].idxmax()
    ]

    best_f1 = results_df.loc[
        results_df["f1_score"].idxmax()
    ]

    print(
        f"Best Accuracy: {best_accuracy['ticker']} "
        f"({best_accuracy['accuracy']:.4f})"
    )

    print(
        f"Best F1 Score: {best_f1['ticker']} "
        f"({best_f1['f1_score']:.4f})"
    )

    print("\n" + "=" * 70)
    print("✅ RANDOM FOREST EVALUATION COMPLETE")
    print("=" * 70)

    print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()