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

from kan_volatility.models.mlp import predict_mlp, train_mlp_model

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
    """
    Split data chronologically into train, validation, and test sets.
    """
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

def plot_training_history(history: dict[str, list[float]]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_loss"], label="Training loss")
    plt.plot(epochs, history["val_loss"], label="Validation loss")
    plt.title("MLP Training History")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = FIGURES_DIR / "mlp_training_history.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved training history plot to: {output_path}")

def main() -> None:
    print("Training MLP volatility model")
    print("=============================")

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

    model, scaler, history = train_mlp_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        hidden_dims=[64, 32],
        dropout=0.10,
        learning_rate=1e-3,
        weight_decay=1e-4,
        batch_size=64,
        max_epochs=300,
        patience=30,
        random_state=42,
    )

    val_pred = predict_mlp(model, scaler, X_val)
    test_pred = predict_mlp(model, scaler, X_test)

    val_metrics = regression_metrics(y_val, val_pred)
    test_metrics = regression_metrics(y_test, test_pred)

    val_reference = np.log(np.maximum(val[REFERENCE_VOL_COLUMN].to_numpy(), 1e-8))
    test_reference = np.log(np.maximum(test[REFERENCE_VOL_COLUMN].to_numpy(), 1e-8))

    val_directional_acc = directional_accuracy(
        y_true=y_val,
        y_pred=val_pred,
        reference=val_reference,
    )

    test_directional_acc = directional_accuracy(
        y_true=y_test,
        y_pred=test_pred,
        reference=test_reference,
    )

    summary = {
        "model": "mlp_64_32",
        "val_mae": val_metrics["mae"],
        "val_rmse": val_metrics["rmse"],
        "val_r2": val_metrics["r2"],
        "val_directional_accuracy": val_directional_acc,
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
        "test_directional_accuracy": test_directional_acc,
    }

    summary_df = pd.DataFrame([summary])

    test_predictions = summarize_predictions(
        dates=test["date"],
        y_true=y_test,
        y_pred=test_pred,
        model_name="mlp_64_32",
    )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = TABLES_DIR / "mlp_model_summary.csv"
    predictions_path = TABLES_DIR / "mlp_test_predictions.csv"

    summary_df.to_csv(summary_path, index=False)
    test_predictions.to_csv(predictions_path, index=False)

    plot_training_history(history)

    print()
    print("MLP training complete.")
    print()
    print("MLP summary:")
    print(summary_df.to_string(index=False))
    print()
    print(f"Saved summary to:     {summary_path}")
    print(f"Saved predictions to: {predictions_path}")


if __name__ == "__main__":
    main()