from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
LOGS_DIR = RESULTS_DIR / "logs"

TICKERS = ["SPY", "QQQ", "IWM", "TLT", "GLD", "^VIX"]

START_DATE = "2005-01-01"
END_DATE = None

TARGET_ASSET = "SPY"
TARGET_HORIZON_DAYS = 5
TRADING_DAYS_PER_YEAR = 252