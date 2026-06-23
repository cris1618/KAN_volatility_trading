import numpy as np
import pandas as pd

def compute_volatility_target_weights(
        predicted_volatility: pd.Series,
        target_volatility: float = 0.10,
        max_weight: float = 1.00,
        min_weight: float = 0.00,
) -> pd.Series:
    """
    Compute volatility-targeting portfolio weights.

    The rule is:

        weight_t = target_volatility / predicted_volatilty_t

    clipped between min_weight and max_weight.

    If predicted volatility is high, exposure decreases.
    If predicted volatility is low, exposure increases up the max_weight.

    Parameters
    ----------
    predicted_volatility: Annualized predicted volatility.
    target_volatility: Desired annualized portfolio volatility.
    max_weight: Maximum SPY allocation.
    min_weight: Minimum SPY allocation.

    Returns
    -------
    weights: Portfolio weights allocated to SPY.
    """
    predicted_volatility = predicted_volatility.copy()

    predicted_volatility = predicted_volatility.replace([np.inf, -np.inf], np.nan)
    predicted_volatility = predicted_volatility.clip(lower=1e-8)

    weights = target_volatility / predicted_volatility
    weights = weights.clip(lower=min_weight, upper=max_weight)

    return weights

def compute_threshold_weights( 
    predicted_volatility: pd.Series,
    high_volatility_threshold: float = 0.25,
    low_risk_weight: float = 0.50,
    normal_risk_weight: float = 1.00,
) -> pd.Series:
    """
    Compute simple threshold-based SPY weights.

    If predicted volatility is above threshold, hold less SPY.
    Otherwise, hold full SPY exposure.
    """
    weights = pd.Series(
        normal_risk_weight,
        index=predicted_volatility.index,
        dtype=float,
    )

    weights[predicted_volatility > high_volatility_threshold] = low_risk_weight

    return weights

def compute_buy_and_hold_weights(
    index: pd.Index,
    weight: float = 1.00,
) -> pd.Series:
    """
    Constant buy-and-hold SPY weight.
    """
    return pd.Series(weight, index=index, dtype=float)