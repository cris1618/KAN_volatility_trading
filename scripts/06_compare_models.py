import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from kan_volatility.config import FIGURES_DIR, TABLES_DIR

BASELINE_SUMMARY_FILE = "baseline_model_summary.csv"
BASELINE_PREDICTIONS_FILE = "baseline_test_predictions.csv"

MLP_SUMMARY_FILE = "mlp_model_summary.csv"
MLP_PREDICTIONS_FILE = "mlp_test_predictions.csv"

def load_required_csv(path):
    if not path.exists():
        raise FileNotFoundError(
            f" Could not find {path}."
            "Make sure you have already run the baseline and MLP scripts."
        )
    
    return pd.read_csv(path)

def load_all_results():
    baseline_summary = load_required_csv(TABLES_DIR / BASELINE_SUMMARY_FILE)
    baseline_predictions = load_required_csv(TABLES_DIR / BASELINE_PREDICTIONS_FILE)

    mlp_summary = load_required_csv(TABLES_DIR / MLP_SUMMARY_FILE)
    mlp_predictions = load_required_csv(TABLES_DIR / MLP_PREDICTIONS_FILE)

    summary = pd.concat(
        [baseline_summary, mlp_summary],
        ignore_index=True,
    )

    predictions = pd.concat(
        [baseline_predictions, mlp_predictions],
        ignore_index=True,
    )

    predictions["date"] = pd.to_datetime(predictions["date"])

    return summary, predictions

def select_models_to_plot(summary):
    """
    Select naive baseline, best tree/classical baseline, and MLP.

    We explicitely include:
        - naive_spy_rv_5d
        - random_forest if available
        - mlp_64_32 if available
    
    If random_forest is missing, we choose the best non-naive, non-MLP model.
    """
    available_models = set(summary["model"].tolist())

    selected_models = []

    if "naive_spy_rv_5d" in available_models:
        selected_models.append("naive_spy_rv_5d")
    
    if "random_forest" in available_models:
        selected_models.append("random_forest")
    else:
        non_naive_non_mlp = summary[
            ~summary["model"].str.contains("naive", case=False)
            & ~summary["model"].str.contains("mlp", case=False)
        ].copy()

        if not non_naive_non_mlp.empty:
            best_classical = non_naive_non_mlp.sort_values("test_rmse").iloc[0][
                "model"
            ]
            selected_models.append(str(best_classical))
    
    if "mlp_64_32" in available_models:
        selected_models.append("mlp_64_32")
    
    return selected_models

def pivot_predictions(predictions):
    """
    Convert long-format predictions into wide format.

    Output columns:
        date
        y_true
        model prediction columns
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

    data = true_values.merge(pred_wide, on="date", how="inner")
    data = data.sort_values("date")
 
    return data

def add_volatility_scale_columns(data):
    """
    Convert log-volatility columns to volatility scale.

    The models predict log future realized volatility, so exp(prediction)
    gives annualized realized-volatility forecasts.
    """
    data = data.copy()

    for column in data.columns:
        if column == "date":
            continue

        data[f"{column}_vol"] = np.exp(data[column])
    
    return data

def make_safe_filename(name):
    return (
        name.replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )

def plot_metric_comparison(summary):
    """
    Plot validation/test RMSE for all models.
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
    plt.title("Model RMSE Comparison")
    plt.xlabel("Model")
    plt.ylabel("RMSE on log future volatility")
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()

    output_path = FIGURES_DIR / "all_models_rmse_comparison.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved: {output_path}")


def plot_directional_accuracy_comparison(summary):
    """
    Plot validation/test directional accuracy for all models.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_data = summary.sort_values("test_directional_accuracy", ascending=False).copy()

    x = np.arange(len(plot_data))
    width = 0.35

    plt.figure(figsize=(12, 6))

    plt.bar(
        x - width / 2,
        plot_data["val_directional_accuracy"],
        width,
        label="Validation directional accuracy",
    )

    plt.bar(
        x + width / 2,
        plot_data["val_directional_accuracy"],
        width,
        label="Test directional accuracy",
    )

    plt.xticks(x, plot_data["model"], rotation=30, ha="right")
    plt.title("Model Directional Accuracy Comparison")
    plt.xlabel("Model")
    plt.ylabel("Directional accuracy")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()

    output_path = FIGURES_DIR / "all_models_directional_accuracy_comparison.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved: {output_path}")

def plot_actual_vs_selected_predictions(data, selected_models):
    """
    Plot actual future volatility against selected model predictions.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 6))

    plt.plot(
        data["date"],
        data["y_true_vol"],
        label="Actual future 5-day realized volatility",
        linewidth=2,
    )

    for model_name in selected_models:
        vol_column = f"{model_name}_vol"

        if vol_column not in data.columns:
            print(f"Skipping {model_name}; column not found.")
            continue

        plt.plot(
            data["date"],
            data[vol_column],
            label=model_name,
            alpha=0.85,
        )
    
    plt.title("Actual vs Predicted Future Realized Volatility")
    plt.xlabel("Date")
    plt.ylabel("Annualized volatility")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = FIGURES_DIR / "all_models_actual_vs_predicted_volatility.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved: {output_path}")

def plot_zoomed_actual_vs_selected_predictions(data, selected_models, last_n_days=25):
    """
    Plot a zoomed comparison over the most recent test-set period.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_data = data.tail(last_n_days).copy()

    plt.figure(figsize=(14, 6))

    plt.plot(
        plot_data["date"],
        plot_data["y_true_vol"],
        label="Actual future 5-day realized volatility",
        linewidth=2,
    )

    for model_name in selected_models:
        vol_column = f"{model_name}_vol"

        if vol_column not in plot_data.columns:
            print(f"Skipping {model_name}; column not found.")
            continue

        plt.plot(
            plot_data["date"],
            plot_data[vol_column],
            label=model_name,
            alpha=0.85,
        )
    
    plt.title(f"Actual vs Predicted Volatility, Last {last_n_days} Test Days")
    plt.xlabel("Date")
    plt.ylabel("Annualized volatility")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = FIGURES_DIR / "all_models_actual_vs_predicted_volatility_zoom.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved: {output_path}")

def plot_prediction_errors(predictions, selected_models):
    """
    Plot prediction errors for selected models.

    Error is measured in log-volatility units:
        prediction - actual
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    selected = predictions[predictions["model"].isin(selected_models)].copy()

    plt.figure(figsize=(14, 6))

    for model_name, group in selected.groupby("model"):
        group = group.sort_values("date")

        plt.plot(
            group["date"],
            group["error"],
            label=model_name,
            alpha=0.8,
        )
    
    plt.axhline(0.0, linestyle="--", linewidth=1)

    plt.title("Prediction Errors Over Time")
    plt.xlabel("Date")
    plt.ylabel("Prediction error: predicted log volatility - true log volatility")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = FIGURES_DIR / "all_models_prediction_errors.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved: {output_path}")

def plot_actual_vs_predicted_scatter(predictions, selected_models):
    """
    Create actual-vs-predicted scatter plots for selected models.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    selected = predictions[predictions["model"].isin(selected_models)].copy()

    min_value = min(selected["y_true"].min(), selected["y_pred"].min())
    max_value = max(selected["y_true"].max(), selected["y_pred"].max())

    for model_name, group in selected.groupby("model"):
        plt.figure(figsize=(7,7))

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

        safe_model_name = make_safe_filename(model_name)
        output_path = (
            FIGURES_DIR / f"all_models_scatter_actual_vs_predicted_{safe_model_name}.png"
        )

        plt.savefig(output_path, dpi=200)
        plt.close()

        print(f"Saved: {output_path}")

def save_combined_summary(summary):
    """
    Save combined baseline + MLP metrics.
    """
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    output_path = TABLES_DIR / "all_models_summary.csv"

    summary.sort_values("test_rmse").to_csv(output_path, index=False)

    print(f"Saved: {output_path}")

def main():
    print("Comparing all models so far")
    print("===========================")

    summary, predictions = load_all_results()

    selected_models = select_models_to_plot(summary)

    print("Models ranked by test RMSE:")
    print(summary.sort_values("test_rmse").to_string(index=False))
    print()

    print("Selected models for detailed plots:")
    for model_name in selected_models:
        print(f" - {model_name}")
    print()

    wide_predictions = pivot_predictions(predictions)
    wide_predictions = add_volatility_scale_columns(wide_predictions)

    save_combined_summary(summary)

    plot_metric_comparison(summary)
    plot_directional_accuracy_comparison(summary)

    plot_actual_vs_selected_predictions(
        data=wide_predictions,
        selected_models=selected_models,
    )

    plot_zoomed_actual_vs_selected_predictions(
        data=wide_predictions,
        selected_models=selected_models,
        last_n_days=252,
    )

    plot_prediction_errors(
        predictions=predictions,
        selected_models=selected_models,
    )

    plot_actual_vs_predicted_scatter(
        predictions=predictions,
        selected_models=selected_models,
    )

    print()
    print("Model comparison complete.")

if __name__ == "__main__":
    main()