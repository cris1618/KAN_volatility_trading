import itertools
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from kan_volatility.config import (
    FIGURES_DIR,
    PROCESSED_DATA_DIR,
    TABLES_DIR,
    TARGET_HORIZON_DAYS,
)
from kan_volatility.metrics import (
    directional_accuracy,
    regression_metrics,
    summarize_predictions,
)
from kan_volatility.models.kan_model import predict_kan, train_kan_model

DATASET_FILE = "spy_volatility_dataset.csv"
TARGET_COLUMN = f"log_future_rv_{TARGET_HORIZON_DAYS}d"
REFERENCE_VOL_COLUMN = "spy_rv_5d"

def load_dataset() -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / DATASET_FILE

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Run scripts/02_build_dataset.py first."
        )
    
    data = pd.read_csv(path)
    data["date"] = pd.to_datetime(data["date"])

    return data

def chronological_split(
    data: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = data.sort_values("date").reset_index(drop=True)

    n = len(data)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train = data.iloc[:train_end].copy()
    val = data.iloc[train_end:val_end].copy()
    test = data.iloc[val_end:].copy()

    return train, val, test

def get_feature_columns(data: pd.DataFrame) -> list[str]:
    excluded_columns = {
        "date",
        TARGET_COLUMN,
        f"future_rv_{TARGET_HORIZON_DAYS}d",
    }

    return [
        column 
        for column in data.columns
        if column not in excluded_columns
    ]

def format_hidden_dims(hidden_dims: list[int]) -> str:
    return "_".join(str(dim) for dim in hidden_dims)

def make_model_name(
        hidden_dims: list[int],
        grid_size: int,
        weight_decay: float,
) -> str:
    wd_string = f"{weight_decay:.0e}".replace("-", "m")
    hidden_string = format_hidden_dims(hidden_dims)

    return f"efficient_kan_{hidden_string}_grid{grid_size}_wd{wd_string}"

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    reference_vol: np.ndarray,
) -> dict[str, float]:
    metrics = regression_metrics(y_true, y_pred)

    reference_log_vol = np.log(np.maximum(reference_vol, 1e-8))

    dir_acc = directional_accuracy(
        y_true=y_true,
        y_pred=y_pred,
        reference=reference_log_vol,
    )

    return {
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "r2": metrics["r2"],
        "directional_accuracy": dir_acc,
    }

def plot_tuning_results(results: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_data = results.sort_values("val_rmse").copy()

    plt.figure(figsize=(14, 6))
    plt.bar(plot_data["model"], plot_data["val_rmse"])
    plt.xticks(rotation=45, ha="right")
    plt.title("EfficientKAN Tuning Results by Validation RMSE")
    plt.xlabel("Model configuration")
    plt.ylabel("Validation RMSE on log future volatility")
    plt.grid(True, axis="y")
    plt.tight_layout()

    output_path = FIGURES_DIR / "kan_tuning_validation_rmse.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved tuning plot to: {output_path}")

def plot_best_training_history(
    history: dict[str, list[float]],
    model_name: str,
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    epochs = np.arange(1, len(history["train_loss"]) +1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_loss"], label="Training loss")
    plt.plot(epochs, history["val_loss"], label="Validation loss")
    plt.title(f"Best EfficientKAN Training History: {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = FIGURES_DIR / "best_kan_training_history.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved best KAN training history to: {output_path}")

def main() -> None:
    print("Tuning EfficientKAN volatility models")
    print("=====================================")

    data = load_dataset()
    train, val, test = chronological_split(data)
    feature_columns = get_feature_columns(data)

    print(f"Dataset rows: {len(data)}")
    print(f"Feature columns: {len(feature_columns)}")
    print()
    print("Split date ranges:")
    print(f"  Train: {train['date'].min().date()} to {train['date'].max().date()}")
    print(f"  Val:   {val['date'].min().date()} to {val['date'].max().date()}")
    print(f"  Test:  {test['date'].min().date()} to {test['date'].max().date()}")
    print()

    X_train = train[feature_columns].to_numpy(dtype=np.float32)
    y_train = train[TARGET_COLUMN].to_numpy(dtype=np.float32)

    X_val = val[feature_columns].to_numpy(dtype=np.float32)
    y_val = val[TARGET_COLUMN].to_numpy(dtype=np.float32)

    X_test = test[feature_columns].to_numpy(dtype=np.float32)
    y_test = test[TARGET_COLUMN].to_numpy(dtype=np.float32)

    val_reference_vol = val[REFERENCE_VOL_COLUMN].to_numpy()
    test_reference_vol = test[REFERENCE_VOL_COLUMN].to_numpy()

    hidden_dims_grid = [
        [8],
        [16],
        [16, 8],
        [32, 16],
    ]

    grid_size_grid = [3, 5, 7]

    weight_decay_grid = [
        1e-4,
        1e-3,
        1e-2,
    ]

    configs = list(
        itertools.product(
            hidden_dims_grid,
            grid_size_grid,
            weight_decay_grid,
        )
    )

    print(f"Number of EfficientKAN configurations: {len(configs)}")
    print()

    tuning_rows = []

    best_val_rmse = float("inf")
    best_model = None
    best_scaler = None
    best_history = None
    best_model_name = None
    best_config = None

    for config_id, (hidden_dims, grid_size, weight_decay) in enumerate(configs, start=1):
        model_name = make_model_name(
            hidden_dims=hidden_dims,
            grid_size=grid_size,
            weight_decay=weight_decay,
        )

        print(f"[{config_id}/{len(configs)}] Training {model_name}")
        start_time = time.time()

        model, scaler, history = train_kan_model(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            hidden_dims=hidden_dims,
            grid_size=grid_size,
            spline_order=3,
            learning_rate=1e-3,
            weight_decay=weight_decay,
            batch_size=64,
            max_epochs=250,
            patience=25,
            random_state=42,
        )

        elapsed_seconds = time.time() - start_time

        val_pred = predict_kan(model, scaler, X_val)
        test_pred = predict_kan(model, scaler, X_test)

        val_metrics = evaluate_predictions(
            y_true=y_val,
            y_pred=val_pred,
            reference_vol=val_reference_vol,
        )

        test_metrics = evaluate_predictions(
            y_true=y_test,
            y_pred=test_pred,
            reference_vol=test_reference_vol,
        )

        row = {
            "model": model_name,
            "hidden_dims": str(hidden_dims),
            "grid_size": grid_size,
            "spline_order": 3,
            "weight_decay": weight_decay,
            "epochs_trained": len(history["train_loss"]),
            "elapsed_seconds": elapsed_seconds,
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_r2": val_metrics["r2"],
            "val_directional_accuracy": val_metrics["directional_accuracy"],
            "test_mae": test_metrics["mae"],
            "test_rmse": test_metrics["rmse"],
            "test_r2": test_metrics["r2"],
            "test_directional_accuracy": test_metrics["directional_accuracy"],
        }

        tuning_rows.append(row)

        print(
            f"  Val RMSE: {row['val_rmse']:.4f} | "
            f"Test RMSE: {row['test_rmse']:.4f} | "
            f"Test directional acc: {row['test_directional_accuracy']:.4f} | "
            f"Time: {elapsed_seconds:.1f}s"
        )

        if row["val_rmse"] < best_val_rmse:
            best_val_rmse = row["val_rmse"]
            best_model = model
            best_scaler = scaler
            best_history = history
            best_model_name = model_name
            best_config = row

        print()

    if best_model is None or best_scaler is None or best_history is None:
        raise RuntimeError("No EfficientKAN model was successfully trained.")

    tuning_results = pd.DataFrame(tuning_rows)
    tuning_results = tuning_results.sort_values("val_rmse")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    tuning_results_path = TABLES_DIR / "kan_tuning_results.csv"
    tuning_results.to_csv(tuning_results_path, index=False)

    print("EfficientKAN tuning complete.")
    print()
    print("Top configurations by validation RMSE:")
    print(tuning_results.head(10).to_string(index=False))
    print()
    print(f"Saved tuning results to: {tuning_results_path}")
    print()

    print("Best KAN selected by validation RMSE:")
    print(pd.DataFrame([best_config]).to_string(index=False))
    print()

    best_val_pred = predict_kan(best_model, best_scaler, X_val)
    best_test_pred = predict_kan(best_model, best_scaler, X_test)

    best_val_metrics = evaluate_predictions(
        y_true=y_val,
        y_pred=best_val_pred,
        reference_vol=val_reference_vol,
    )

    best_test_metrics = evaluate_predictions(
        y_true=y_test,
        y_pred=best_test_pred,
        reference_vol=test_reference_vol,
    )

    best_summary = {
        "model": f"best_{best_model_name}",
        "selected_by": "val_rmse",
        "val_mae": best_val_metrics["mae"],
        "val_rmse": best_val_metrics["rmse"],
        "val_r2": best_val_metrics["r2"],
        "val_directional_accuracy": best_val_metrics["directional_accuracy"],
        "test_mae": best_test_metrics["mae"],
        "test_rmse": best_test_metrics["rmse"],
        "test_r2": best_test_metrics["r2"],
        "test_directional_accuracy": best_test_metrics["directional_accuracy"],
    }

    best_summary_df = pd.DataFrame([best_summary])

    best_summary_path = TABLES_DIR / "best_kan_model_summary.csv"
    best_predictions_path = TABLES_DIR / "best_kan_test_predictions.csv"

    best_summary_df.to_csv(best_summary_path, index=False)

    best_test_predictions = summarize_predictions(
        dates=test["date"],
        y_true=y_test,
        y_pred=best_test_pred,
        model_name=f"best_{best_model_name}",
    )

    best_test_predictions.to_csv(best_predictions_path, index=False)

    plot_tuning_results(tuning_results)
    plot_best_training_history(best_history, model_name=f"best_{best_model_name}")

    print()
    print("Saved best KAN summary and predictions:")
    print(f"  Summary:     {best_summary_path}")
    print(f"  Predictions: {best_predictions_path}")


if __name__ == "__main__":
    main()