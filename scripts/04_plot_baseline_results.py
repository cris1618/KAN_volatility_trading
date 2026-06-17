import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kan_volatility.config import FIGURES_DIR, TABLES_DIR

SUMMARY_FILE = "baseline_model_summary.csv"
PREDICTIONS_FILE = "baseline_test_predictions.csv"

def load_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_path = TABLES_DIR / SUMMARY_FILE
    predictions_path = TABLES_DIR / PREDICTIONS_FILE

    if not summary_path.exists():
        raise FileNotFoundError(
            f"Could not find {summary_path}. Run scripts/03_train_baselines.py first."
        )
    
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Could not find {predictions_path}. Run scripts/03_train_baselines.py first."
        )
    
    summary = pd.read_csv(summary_path)
    predictions = pd.read_csv(predictions_path)

    predictions["date"] = pd.to_datetime(predictions["date"])

    return summary, predictions

def choose_best_model(summary: pd.DataFrame) -> str:
    """
    Choose the best baseline model by test RMSE.

    We exclude the naive model so that we can compare:
        naive baseline vs best non-naive baseline.
    """
    non_naive = summary[~summary["model"].str.contains("naive", case=False)].copy()

    if non_naive.empty:
        raise ValueError("No non-naive models found in baseline summary.")
    
    best_row = non_naive.sort_values("test_rmse").iloc[0]

    return str(best_row["model"])

def pivot_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Convert long prediction data into wide format.

    Input columns:
        date, model, y_true, y_pred

    Output columns:
        date, y_true, model_1, model_2, ...
    """
    true_values = (
        predictions[["date", "y_true"]]
        .drop_duplicates(subset=["date"])
        .sort_values("date")
    )

    pred_wide = predictions.pivot(
        index="date",
        columns="model",
        values="y_pred",
    ).reset_index()

    merged = true_values.merge(pred_wide, on="date", how="inner")
    merged = merged.sort_values("date")

    return merged

def add_volatility_scale_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert log-volatility columns back to volatility scale.

    Since the models predict log future realized volatility, exponentiating gives
    annualized volatility.
    """
    data = data.copy()
    
    for column in data.columns:
        if column == "date":
            continue

        data[f"{column}_vol"] = np.exp(data[column])
    
    return data

def plot_actual_vs_predictions(
        data: pd.DataFrame,
        naive_model: str,
        best_model: str,
) -> None:
    """
    Plot actual future volatility vs naive and best baseline predictions.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))

    plt.plot(
        data["date"],
        data["y_true_vol"],
        label="Actual future 5-day realized volatility",
        linewidth=2,
    )    

    plt.plot(
        data["date"],
        data[f"{naive_model}_vol"],
        label="Naive baseline prediction",
        alpha=0.8,
    )

    plt.plot(
        data["date"],
        data[f"{best_model}_vol"],
        label=f"Best baseline prediction: {best_model}",
        alpha=0.8,
    )

    plt.title("Actual vs Predicted Future Realized Volatility")
    plt.xlabel("Date")
    plt.ylabel("Annualized volatility")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = FIGURES_DIR / "baseline_actual_vs_predicted_volatility.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    
    print(f"Saved: {output_path}")

def plot_prediction_errors(
    predictions: pd.DataFrame,
    naive_model: str,
    best_model: str,
) -> None:
    """
    Plot prediction errors over time for the naive and best baseline model.

    Errors are shown in log-volatility units.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    selected = predictions[predictions["model"].isin([naive_model, best_model])].copy()

    plt.figure(figsize=(12, 6))

    for model_name, group in selected.groupby("model"):
        group = group.sort_values("date")
        plt.plot(
            group["date"],
            group["error"],
            label=model_name,
            alpha=0.8,
        )

    plt.axhline(0.0, linestyle="--", linewidth=1)

    plt.title("Baseline Prediction Errors Over Time")
    plt.xlabel("Date")
    plt.ylabel("Prediction error: predicted log volatility - true log volatility")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = FIGURES_DIR / "baseline_prediction_errors.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved: {output_path}")

def plot_model_rmse_comparison(summary: pd.DataFrame) -> None:
    """
    Bar chart comparing validation and test RMSE across baseline models.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_data = summary.sort_values("test_rmse").copy()

    x = np.arange(len(plot_data))
    width = 0.35

    plt.figure(figsize=(12, 6))

    plt.bar(
        x - width / 2,
        plot_data["val_rmse"],
        width,
        label="Validation RMSE",
    )

    plt.bar(
        x + width / 2,
        plot_data["test_rmse"],
        width,
        label="Test RMSE",
    )

    plt.xticks(x, plot_data["model"], rotation=30, ha="right")
    plt.title("Baseline Model RMSE Comparison")
    plt.xlabel("Model")
    plt.ylabel("RMSE on log future volatility")
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()

    output_path = FIGURES_DIR / "baseline_rmse_comparison.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved: {output_path}")

def plot_scatter_actual_vs_predicted(
    predictions: pd.DataFrame,
    naive_model: str,
    best_model: str,
) -> None:
    """
    Scatter plot of actual vs predicted log volatility.

    A perfect model would lie close to the diagonal line.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    selected = predictions[predictions["model"].isin([naive_model, best_model])].copy()

    min_value = min(selected["y_true"].min(), selected["y_pred"].min())
    max_value = max(selected["y_true"].max(), selected["y_pred"].max())

    for model_name, group in selected.groupby("model"):
        plt.figure(figsize=(7, 7))

        plt.scatter(
            group["y_true"],
            group["y_pred"],
            alpha=0.5,
        )

        plt.plot(
            [min_value, max_value],
            [min_value, max_value],
            linestyle="--",
            linewidth=1,
            label="Perfect prediction",
        )

        plt.title(f"Actual vs Predicted Log Volatility: {model_name}")
        plt.xlabel("Actual log future volatility")
        plt.ylabel("Predicted log future volatility")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        safe_model_name = model_name.replace("/", "_").replace(" ", "_")
        output_path = FIGURES_DIR / f"scatter_actual_vs_predicted_{safe_model_name}.png"

        plt.savefig(output_path, dpi=200)
        plt.close()

        print(f"Saved: {output_path}")

def main() -> None:
    print("Plotting baseline results")
    print("=========================")

    summary, predictions = load_results()

    naive_model = "naive_spy_rv_5d"
    best_model = choose_best_model(summary)

    print(f"Naive model:      {naive_model}")
    print(f"Best non-naive:   {best_model}")
    print()

    wide_predictions = pivot_predictions(predictions)
    wide_predictions = add_volatility_scale_columns(wide_predictions)

    plot_actual_vs_predictions(
        data=wide_predictions,
        naive_model=naive_model,
        best_model=best_model,
    )

    plot_prediction_errors(
        predictions=predictions,
        naive_model=naive_model,
        best_model=best_model,
    )

    plot_model_rmse_comparison(summary)

    plot_scatter_actual_vs_predicted(
        predictions=predictions,
        naive_model=naive_model,
        best_model=best_model,
    )

    print()
    print("Plotting complete.")


if __name__ == "__main__":
    main()