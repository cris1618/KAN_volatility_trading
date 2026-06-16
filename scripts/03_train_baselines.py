import numpy as np
import pandas as pd

from kan_volatility.config import PROCESSED_DATA_DIR, TABLES_DIR, TARGET_HORIZON_DAYS
from kan_volatility.metrics import (
    directional_accuracy,
    regression_metrics,
    summarize_predictions,
)
from kan_volatility.models.baselines import (
    NaiveVolatilityBaseline,
    make_linear_regression_model,
    make_random_forest_model,
    make_ridge_model,
)

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

    No shuffling is used because this is time-series data.
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
    """
    Select model feature columns.

    We exclude date and target columns. We also exclude the non-log future 
    volatility target to avoid leakage.
    """
    excluded_columns = {
        "date",
        TARGET_COLUMN,
        f"future_rv_{TARGET_HORIZON_DAYS}d",
    }

    feature_columns = [
        column 
        for column in data.columns
        if column not in excluded_columns
    ]

    return feature_columns

def evaluate_model(
        model_name: str,
        model,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
        feature_columns: list[str],
) -> tuple[dict[str, float | str], pd.DataFrame]:
    """
    Fit a model and evaluate it on train, validation and test sets.
    """
    X_train = train[feature_columns]
    y_train = train[TARGET_COLUMN].to_numpy()

    X_val = val[feature_columns]
    y_val = val[TARGET_COLUMN].to_numpy()

    X_test = test[feature_columns]
    y_test = test[TARGET_COLUMN].to_numpy()

    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)


    val_metrics = regression_metrics(y_val, val_pred)
    test_metrics = regression_metrics(y_test, test_pred)

    # Directional accuracy relative to current 5-day realized volatility.
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
        "model": model_name,
        "val_mae": val_metrics["mae"],
        "val_rmse": val_metrics["rmse"],
        "val_r2": val_metrics["r2"],
        "val_directional_accuracy": val_directional_acc,
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
        "test_directional_accuracy": test_directional_acc,
    }

    test_predictions = summarize_predictions(
        dates=test["date"],
        y_true=y_test,
        y_pred=test_pred,
        model_name=model_name,
    )

    return summary, test_predictions


def main() -> None:
    print("Training baseline volatility models")
    print("===================================")

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

    models = {
        "naive_spy_rv_5d": NaiveVolatilityBaseline(
            reference_column=REFERENCE_VOL_COLUMN
        ),
        "linear_regression": make_linear_regression_model(),
        "ridge_alpha_1": make_ridge_model(alpha=1.0),
        "ridge_alpha_10": make_ridge_model(alpha=10.0),
        "random_forest": make_random_forest_model(),
    }

    summaries = []
    all_test_predictions = []

    for model_name, model in models.items():
        print(f"Training {model_name}...")

        summary, test_predictions = evaluate_model(
            model_name=model_name,
            model=model,
            train=train,
            val=val,
            test=test,
            feature_columns=feature_columns,
        )

        summaries.append(summary)
        all_test_predictions.append(test_predictions)

        print(
            f"  Val RMSE:  {summary['val_rmse']:.4f} | "
            f"Test RMSE: {summary['test_rmse']:.4f} | "
            f"Test R2: {summary['test_r2']:.4f}"
        )

    summary_df = pd.DataFrame(summaries)
    summary_df = summary_df.sort_values("test_rmse")

    predictions_df = pd.concat(all_test_predictions, ignore_index=True)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = TABLES_DIR / "baseline_model_summary.csv"
    predictions_path = TABLES_DIR / "baseline_test_predictions.csv"

    summary_df.to_csv(summary_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)

    print()
    print("Baseline training complete.")
    print()
    print("Model summary:")
    print(summary_df.to_string(index=False))
    print()
    print(f"Saved summary to:     {summary_path}")
    print(f"Saved predictions to: {predictions_path}")


if __name__ == "__main__":
    main()