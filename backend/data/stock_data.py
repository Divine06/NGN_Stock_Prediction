import yfinance as yf
import pandas as pd


def fetch_stock_data(
    ticker: str,
    start_date: str = "2014-01-01",
    end_date: str = "2026-01-01"
) -> pd.DataFrame:

    """
    Download historical OHLCV data for an NGX-listed stock.

    Parameters:
        ticker: Yahoo Finance ticker symbol
        start_date: Beginning of historical data
        end_date: End date (exclusive)

    Returns:
        Pandas DataFrame containing OHLCV data.
    """

    try:
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False
        )

        if data.empty:
            raise ValueError(
                f"No data returned for ticker: {ticker}"
            )

        # Handle yfinance MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.reset_index()

        # Keep only the fields required by the project
        required_columns = [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        data = data[required_columns]

        # Remove rows containing missing values
        data = data.dropna()

        # Make sure data is sorted chronologically
        data = data.sort_values("Date")

        return data

    except Exception as e:
        raise RuntimeError(
            f"Failed to retrieve data for {ticker}: {e}"
        )