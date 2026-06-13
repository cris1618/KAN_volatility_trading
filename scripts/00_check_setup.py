from kan_volatility.config import (
    PROJECT_ROOT,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    LOGS_DIR,
    TICKERS,
    TARGET_ASSET,
    TARGET_HORIZON_DAYS,
)

def main() -> None:
    print("Project setup check")
    print("===================")
    print(f"Project root:        {PROJECT_ROOT}")
    print(f"Raw data directory:  {RAW_DATA_DIR}")
    print(f"Processed directory: {PROCESSED_DATA_DIR}")
    print(f"Figures directory:   {FIGURES_DIR}")
    print(f"Tables directory:    {TABLES_DIR}")
    print(f"Logs directory:      {LOGS_DIR}")
    print()
    print(f"Tickers:             {TICKERS}")
    print(f"Target asset:        {TARGET_ASSET}")
    print(f"Target horizon:      {TARGET_HORIZON_DAYS} trading days")


if __name__ == "__main__":
    main()