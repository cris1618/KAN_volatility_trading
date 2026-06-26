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

def smooth_weights(
    raw_weights: pd.Series,
    smoothing_alpha: float = 1.0,
) -> pd.Series:
    """
    Smooth target weights through time.

    If smoothing_alpha = 1.0, no smoothing is applied.

    If smoothing_alpha = 0.25, the new target weight is:

        smoothed_weight_t =
            0.25 * raw_weight_t + 0.75 * smoothed_weight_{t-1}
    
    Lower alpha means smoother weights and lower turnover.
    """
    if not 0.0 < smoothing_alpha <= 1.0:
        raise ValueError("smoothing_alpha must be in the interval (0, 1].")
    
    raw_weights = raw_weights.copy().astype(float)

    smoothed = raw_weights.copy()

    if len(smoothed) == 0:
        return smoothed
    
    smoothed.iloc[0] = raw_weights.iloc[0]

    for i in range(1, len(raw_weights)):
        smoothed.iloc[i] = (
            smoothing_alpha * raw_weights.iloc[i]
            + (1.0 - smoothing_alpha) * smoothed.iloc[i-1]
        )
    
    return smoothed

def apply_rebalance_buffer(
    target_weights: pd.Series,
    rebalance_buffer: float = 0.0,
) -> pd.Series:
    """
    Apply a no-trade buffer to target weights.

    The portfolio only updates its target weight when the desired change is 
    larger than rebalance_buffer. 

    Example
    -------
    If rebalance_buffer = 0.05, then a change from 0.80 to 0.83 is ignored,
    but a change from 0.80 to 0.90 is accepted.
    """
    if rebalance_buffer < 0.0:
        raise ValueError("rebalance_buffer must be nonnegative.")
    
    target_weights = target_weights.copy().astype(float)

    if rebalance_buffer == 0.0 or len(target_weights) == 0:
        return target_weights
    
    buffered = target_weights.copy()
    buffered.iloc[0] = target_weights.iloc[0]

    current_weight = target_weights.iloc[0]

    for i in range(1, len(target_weights)):
        proposed_weight = target_weights.iloc[i]

        if abs(proposed_weight - current_weight) > rebalance_buffer:
            current_weight = proposed_weight
        
        buffered.iloc[i] = current_weight
    
    return buffered