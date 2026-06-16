import numpy as np 
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Compute standard regression metrics.

    Parameters
    ----------
    y_true: True target values.
    y_pred: Predicted target values.

    Returns
    -------
    metrics: Dictionary containing MAE, RMSE, and R2.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }

def directional_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    reference: np.ndarray,
) -> float:
    """
    Measure whether the model correctly predicts whether volatility rises or falls
    relative to a reference value.

    Example
    -------
    If reference is today's realized volatility, this checks whether the model
    correctly predicts whether future volatility will be above or below today's 
    volatility.
    """
    true_direction = y_true > reference
    pred_direction = y_pred > reference

    return float(np.mean(true_direction == pred_direction))

def summarize_predictions(
    dates: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
) -> pd.DataFrame:
    """
    Create a DataFrame containing dates, true values, predictions, and errors.
    """
    results = pd.DataFrame({
        "date": dates.values,
        "model": model_name,
        "y_true": y_true,
        "y_pred": y_pred,
    })

    results["error"] = results["y_pred"] - results["y_true"]
    results["absolute_error"] = results["error"].abs()
    results["squared_error"] = results["error"] ** 2

    return results