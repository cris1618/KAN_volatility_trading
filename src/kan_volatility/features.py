import numpy as np
import pandas as pd
from kan_volatility.config import TRADING_DAYS_PER_YEAR

def add_return_features(
    data: pd.DataFrame,
    price_column: str = "adj_close",
) -> pd.DataFrame:
    """
    Add daily log returns and simple multi-day return features.
    """
    data = data.copy()
    data["log_return_1d"] = np.log(data[price_column] / data[price_column].shift(1))

    for window in [5, 10, 21, 63]:
        data[f"log_return_{window}d"] = np.log(
            data[price_column] / data[price_column].shift(window)
        )
    
    return data

def add_realized_volatility_features(
        data: pd.DataFrame,
        return_column: str = "log_return_1d",
) -> pd.DataFrame:
    """
    Add annualized realized volatility features using rolling log-return windows.
    """
    data = data.copy()

    for window in [5, 10, 21, 63]:
        data[f"rv_{window}d"] = (
            data[return_column]
            .rolling(window=window)
            .std()
            * np.sqrt(TRADING_DAYS_PER_YEAR)
        )

    return data

def add_volume_features(
    data: pd.DataFrame,
    volume_column: str = "volume",
) -> pd.DataFrame:
    """
    Add simple volume-based features.
    """
    data = data.copy()
    data["volume_change_1d"] = np.log(
        data[volume_column] / data[volume_column].shift(1)
    )

    for window in [5, 21]:
        data[f"volume_zscore_{window}d"] = (
            data[volume_column] - data[volume_column].rolling(window).mean()
        ) / data[volume_column].rolling(window).std()
    
    return data

def add_price_position_features(
    data: pd.DataFrame,
    price_column: str = "adj_close",
) -> pd.DataFrame:
    """
    Add features measuring price position relative to moving averages.
    """
    data = data.copy()

    for window in [10, 21, 63, 126, 252]:
        moving_average = data[price_column].rolling(window).mean()
        data[f"ma_distance_{window}d"] = data[price_column] / moving_average - 1.0
    
    return data

def build_single_asset_features(
    data: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    """
    Build features for one asset and prefix column names.

    Parameters:
    -----------
    data: Raw OHLCV data for one asset.
    prefix: Prefix to add to feature columns, for example "spy or "qqq".

    Returns:
    --------
    features: DataFrame with date plus prefixed features.
    """
    data = data.copy()

    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date")

    data = add_return_features(data)
    data = add_realized_volatility_features(data)
    data = add_volume_features(data)
    data = add_price_position_features(data)

    feature_columns = [
        "date",
        "log_return_1d",
        "log_return_5d",
        "log_return_10d",
        "log_return_21d",
        "log_return_63d",
        "rv_5d",
        "rv_10d",
        "rv_21d",
        "rv_63d",
        "volume_change_1d",
        "volume_zscore_5d",
        "volume_zscore_21d",
        "ma_distance_10d",
        "ma_distance_21d",
        "ma_distance_63d",
        "ma_distance_126d",
        "ma_distance_252d",
    ]

    features = data[feature_columns].copy()

    rename_map = {
        column: f"{prefix}_{column}" 
        for column in feature_columns
        if column != "date"
    }

    features = features.rename(columns=rename_map)
    
    return features

def build_vix_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build VIX-specific features.

    VIX is already a volatility index, so we do not treat it like a tradable
    asset in exactly the same way as ETFs.
    """
    data = data.copy()

    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date")

    data["vix_close"] = data["close"]
    data["vix_change_1d"] = data["vix_close"].diff()
    data["vix_pct_change_1d"] = data["vix_close"].pct_change()

    for window in [5, 10, 21, 63]:
        data[f"vix_ma_{window}d"] = data["vix_close"].rolling(window).mean()
        data[f"vix_ma_distance_{window}d"] = (
            data["vix_close"] / data[f"vix_ma_{window}d"] - 1.0 
        )

    feature_columns = [
        "date",
        "vix_close",
        "vix_change_1d",
        "vix_pct_change_1d",
        "vix_ma_distance_5d",
        "vix_ma_distance_10d",
        "vix_ma_distance_21d",
        "vix_ma_distance_63d",
    ]
    
    return data[feature_columns].copy()
    