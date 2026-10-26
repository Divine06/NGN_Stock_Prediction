import pandas as pd
from pathlib import Path


BASE_DIR = Path("backend/data/predictions")

ARIMA_FILE = BASE_DIR / "arima/arima_summary.csv"
SVR_FILE = BASE_DIR / "svr/svr_summary.csv"
LSTM_FILE = BASE_DIR / "lstm/lstm_summary.csv"

OUTPUT_DIR = BASE_DIR / "comparison"


def load_model_results():

    arima = pd.read_csv(ARIMA_FILE)
    svr = pd.read_csv(SVR_FILE)
    lstm = pd.read_csv(LSTM_FILE)

    arima = arima[
        ["ticker", "mae", "rmse", "mape"]
    ].copy()

    svr = svr[
        ["ticker", "mae", "rmse", "mape"]
    ].copy()

    lstm = lstm[
        ["ticker", "mae", "rmse", "mape"]
    ].copy()

    arima["model"] = "ARIMA"
    svr["model"] = "SVR"
    lstm["model"] = "LSTM"

    return pd.concat(
        [arima, svr, lstm],
        ignore_index=True
    )


def create_comparison():

    print("=" * 70)
    print("NGX MODEL COMPARISON")
    print("=" * 70)

    # Load all model results
    comparison = load_model_results()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ---------------------------------------------------------
    # Save complete comparison
    # ---------------------------------------------------------

    comparison_file = (
        OUTPUT_DIR /
        "model_comparison.csv"
    )

    comparison.to_csv(
        comparison_file,
        index=False
    )

    # ---------------------------------------------------------
    # Display results by company
    # ---------------------------------------------------------

    print("\nMODEL PERFORMANCE BY COMPANY")
    print("-" * 70)

    print(
        comparison.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # Average performance
    # ---------------------------------------------------------

    average = (
        comparison
        .groupby("model")[
            ["mae", "rmse", "mape"]
        ]
        .mean()
        .reset_index()
    )

    print("\n" + "=" * 70)
    print("AVERAGE MODEL PERFORMANCE")
    print("=" * 70)

    print(
        average.to_string(
            index=False
        )
    )

    average_file = (
        OUTPUT_DIR /
        "average_model_performance.csv"
    )

    average.to_csv(
        average_file,
        index=False
    )

    # ---------------------------------------------------------
    # Determine best overall model
    # ---------------------------------------------------------

    best_mae = average.loc[
        average["mae"].idxmin()
    ]

    best_rmse = average.loc[
        average["rmse"].idxmin()
    ]

    best_mape = average.loc[
        average["mape"].idxmin()
    ]

    print("\n" + "=" * 70)
    print("BEST MODEL")
    print("=" * 70)

    print(
        f"Lowest MAE:  {best_mae['model']} "
        f"({best_mae['mae']:.4f})"
    )

    print(
        f"Lowest RMSE: {best_rmse['model']} "
        f"({best_rmse['rmse']:.4f})"
    )

    print(
        f"Lowest MAPE: {best_mape['model']} "
        f"({best_mape['mape']:.2f}%)"
    )

    # ---------------------------------------------------------
    # Best model for EACH company
    # ---------------------------------------------------------

    company_winners = []

    for ticker, company_data in comparison.groupby("ticker"):

        best_row = company_data.loc[
            company_data["mape"].idxmin()
        ]

        company_winners.append({
            "ticker": ticker,
            "best_model": best_row["model"],
            "best_mape": best_row["mape"]
        })

    winner_table = pd.DataFrame(
        company_winners
    )

    # Save winners
    winner_file = (
        OUTPUT_DIR /
        "best_model_by_company.csv"
    )

    winner_table.to_csv(
        winner_file,
        index=False
    )

    # ---------------------------------------------------------
    # Display winners
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("BEST MODEL BY COMPANY — BASED ON MAPE")
    print("=" * 70)

    print(
        winner_table.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # Count model wins
    # ---------------------------------------------------------

    wins = (
        winner_table["best_model"]
        .value_counts()
    )

    print("\nModel win count:")

    for model, count in wins.items():

        print(
            f"{model}: {count} companies"
        )

    # ---------------------------------------------------------
    # Final message
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("✅ MODEL COMPARISON COMPLETE")
    print("=" * 70)

    print(
        f"\nSaved: {comparison_file}"
    )

    print(
        f"Saved: {average_file}"
    )

    print(
        f"Saved: {winner_file}"
    )


if __name__ == "__main__":
    create_comparison()