from pathlib import Path
import pandas as pd
import yfinance as yf


EXPECTED_COLUMNS = {
    "date",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "ticker",
}


def clean_ticker_for_filename(ticker: str) -> str:
    """
    Convert ticker symbols into safe filenames.

    Example
    -------
    '^VIX' becomes 'VIX'
    'BRK-B' becomes 'BRK_B'
    """
    return ticker.replace("^", "").replace("-", "_").replace("/", "_")


def normalize_column_name(column: str) -> str:
    """
    Normalize a market-data column name.

    Examples
    --------
    'Adj Close' -> 'adj_close'
    'Date' -> 'date'
    """
    return str(column).strip().lower().replace(" ", "_")


def flatten_yfinance_columns(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Handle possible MultiIndex columns returned by yfinance.

    For a single ticker, yfinance may sometimes return columns like:

        ('Close', 'QQQ')
        ('High', 'QQQ')

    This function removes the ticker level and keeps only:

        Close
        High
    """
    if isinstance(data.columns, pd.MultiIndex):
        # Case where one level contains the ticker symbol.
        for level in range(data.columns.nlevels):
            level_values = data.columns.get_level_values(level)

            if ticker in level_values:
                data = data.xs(ticker, axis=1, level=level)
                break
        else:
            # Fallback: keep the first meaningful level.
            data.columns = [
                "_".join(str(part) for part in col if str(part) != "")
                for col in data.columns
            ]

    return data


def download_single_ticker(
    ticker: str,
    start_date: str,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Download daily historical market data for one ticker using yfinance.
    """
    data = yf.download(
        tickers=ticker,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
    )

    if data.empty:
        raise ValueError(f"No data downloaded for ticker: {ticker}")

    data = flatten_yfinance_columns(data, ticker=ticker)

    data = data.reset_index()

    data.columns = [normalize_column_name(col) for col in data.columns]

    # Standardize adjusted-close column name.
    if "adj_close" not in data.columns and "adjclose" in data.columns:
        data = data.rename(columns={"adjclose": "adj_close"})

    data["ticker"] = ticker

    missing_columns = EXPECTED_COLUMNS.difference(data.columns)

    if missing_columns:
        raise ValueError(
            f"Downloaded data for {ticker} is missing columns: {missing_columns}. "
            f"Available columns: {list(data.columns)}"
        )

    return data


def save_raw_ticker_data(
    data: pd.DataFrame,
    ticker: str,
    output_dir: Path,
) -> Path:
    """
    Save raw ticker data as a CSV file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_ticker = clean_ticker_for_filename(ticker)
    output_path = output_dir / f"{safe_ticker}.csv"

    data.to_csv(output_path, index=False)

    return output_path


def download_and_save_tickers(
    tickers: list[str],
    start_date: str,
    end_date: str | None,
    output_dir: Path,
) -> list[Path]:
    """
    Download and save raw data for multiple tickers.
    """
    saved_paths = []

    for ticker in tickers:
        print(f"Downloading {ticker}...")

        data = download_single_ticker(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
        )

        output_path = save_raw_ticker_data(
            data=data,
            ticker=ticker,
            output_dir=output_dir,
        )

        print(f"Saved {ticker} data to {output_path}")
        saved_paths.append(output_path)

    return saved_paths