from kan_volatility.config import RAW_DATA_DIR, START_DATE, END_DATE, TICKERS
from kan_volatility.data import download_and_save_tickers

def main() -> None: 
    print("Downoloading market data")
    print("========================")
    print(f"Tickers: {TICKERS}")
    print(f"Start date: {START_DATE}")
    print(f"End date: {END_DATE}")
    print(f"Output dir: {RAW_DATA_DIR}")
    print()

    saved_paths = download_and_save_tickers(
        tickers=TICKERS,
        start_date=START_DATE,
        end_date=END_DATE,
        output_dir=RAW_DATA_DIR,
    )

    print()
    print("Download Complete.")
    print("Saved files: ")

    for path in saved_paths:
        print(f" - {path}")

if __name__ == "__main__":
    main()
