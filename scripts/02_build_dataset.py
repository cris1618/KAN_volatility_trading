import numpy as np
import pandas as pd

from kan_volatility.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    TARGET_HORIZON_DAYS,
)
from kan_volatility.features import (
    build_single_asset_features,
    build_vix_features,
)
from kan_volatility.targets import add_future_realized_volatility_target

ASSET_FILES = {
    "spy": "SPY.csv",
    "qqq": "QQQ.csv",
    "iwm": "IWM.csv",
    "tlt": "TLT.csv",
    "gld": "GLD.csv",
}

VIX_FILE = "VIX.csv"

def load_raw_csv(filename: str) -> pd.DataFrame:
    path = RAW_DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")
    
    data = pd.read_csv(path)
    data["date"] = pd.to_datetime(data["date"])

    return data

def main() -> None:
    print("Building processed volatility dataset")
    print("=====================================")

    print("Loading SPY target asset...")
    spy_raw = load_raw_csv(ASSET_FILES["spy"])

    print("Building SPY features...")
    dataset = build_single_asset_features(spy_raw, prefix="spy")

    print("Building target...")
    spy_for_target = spy_raw.copy()
    spy_for_target["date"] = pd.to_datetime(spy_for_target["date"])
    spy_for_target = spy_for_target.sort_values("date")

    spy_for_target["log_return_1d"] = np.log(
        spy_for_target["adj_close"] / spy_for_target["adj_close"].shift(1)
    )

    spy_with_target = add_future_realized_volatility_target(
        data=spy_for_target,
        horizon_days=TARGET_HORIZON_DAYS,
        return_column="log_return_1d",
    )

    target_column = f"log_future_rv_{TARGET_HORIZON_DAYS}d"

    target_data = spy_with_target[
        ["date", f"future_rv_{TARGET_HORIZON_DAYS}d", target_column]
    ].copy()

    dataset = dataset.merge(target_data, on="date", how="inner")

    for prefix, filename in ASSET_FILES.items():
        if prefix == "spy":
            continue

        print(f"Building {prefix.upper()} features...")
        raw_data = load_raw_csv(filename)
        asset_features = build_single_asset_features(raw_data, prefix=prefix)

        dataset = dataset.merge(asset_features, on="date", how="inner")

    print("Building VIX features...")
    vix_raw = load_raw_csv(VIX_FILE)
    vix_features = build_vix_features(vix_raw)

    dataset = dataset.merge(vix_features, on="date", how="inner")

    print("Cleaning dataset...")
    dataset = dataset.sort_values("date")
    dataset = dataset.replace([np.inf, -np.inf], np.nan)
    dataset = dataset.dropna()

    output_path = PROCESSED_DATA_DIR / "spy_volatility_dataset.csv"
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    dataset.to_csv(output_path, index=False)

    print()
    print("Processed dataset complete.")
    print(f"Saved to: {output_path}")
    print(f"Rows:     {len(dataset)}")
    print(f"Columns:  {len(dataset.columns)}")
    print()
    print("Date range:")
    print(f"  Start: {dataset['date'].min().date()}")
    print(f"  End:   {dataset['date'].max().date()}")
    print()
    print("Target columns:")
    print(f"  future_rv_{TARGET_HORIZON_DAYS}d")
    print(f"  {target_column}")
    print()
    print("First few columns:")
    for column in dataset.columns[:15]:
        print(f"  - {column}")

if __name__ == "__main__":
    main()
