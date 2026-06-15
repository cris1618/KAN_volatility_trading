import numpy as np
import pandas as pd
from kan_volatility.config import TRADING_DAYS_PER_YEAR

def add_future_realized_volatility_target(
    data: pd.DataFrame,
    horizon_days: int,
    return_column: str = "log_return_1d",
) -> pd.DataFrame:
    """
    Add future annualized realized volatility target.

    For each date t, this computes the annualized standard deviation of returns
    from t+1 through t+horizon_days.

    This avoids look-ahead leakege in features because today's feature row only
    uses information available through date t, while the target uses future data.
    """
    data = data.copy()

    future_squared_returns = pd.Series(0.0, index=data.index)

    for step in range(1, horizon_days + 1):
        future_squared_returns += data[return_column].shift(-step) ** 2
    
    future_variance = future_squared_returns / horizon_days

    data[f"future_rv_{horizon_days}d"] = (
        np.sqrt(future_variance) * np.sqrt(TRADING_DAYS_PER_YEAR)
    )

    data[f"log_future_rv_{horizon_days}d"] = np.log(data[f"future_rv_{horizon_days}d"])

    return data