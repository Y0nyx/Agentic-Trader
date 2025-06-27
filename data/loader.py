"""
Financial data loading and processing module using yfinance.

This module provides robust functionality for retrieving, cleaning, and preparing
historical financial data via the yfinance library. It serves as the foundation
for reliable data used by trading strategies and predictive models.
"""

import logging
from typing import List
import pandas as pd
import numpy as np
import yfinance as yf

# Configure logging
logger = logging.getLogger(__name__)


def load_financial_data(
    symbol: str,
    period: str = "1y",
    interval: str = "1d",
    clean_data: bool = True,
    detect_outliers: bool = True,
) -> pd.DataFrame:
    """
    Load historical financial data for a given symbol with robust error handling.

    This function retrieves historical financial data using yfinance and optionally
    applies data cleaning and outlier detection.

    Parameters
    ----------
    symbol : str
        Financial symbol to retrieve data for (e.g., 'BTC-USD', 'AAPL', 'EURUSD=X')
    period : str, default '1y'
        Time period for data retrieval. Valid values: '1d', '5d', '1mo', '3mo',
        '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'
    interval : str, default '1d'
        Data interval. Valid values: '1m', '2m', '5m', '15m', '30m', '60m', '90m',
        '1h', '1d', '5d', '1wk', '1mo', '3mo'
    clean_data : bool, default True
        Whether to apply data cleaning (handle missing values)
    detect_outliers : bool, default True
        Whether to detect and flag potential outliers

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized columns: Open, High, Low, Close, Volume
        and DatetimeIndex. Returns empty DataFrame if data retrieval fails.

    Examples
    --------
    >>> # Load 1 year of daily Bitcoin data
    >>> btc_data = load_financial_data('BTC-USD')
    >>> print(btc_data.head())

    >>> # Load 6 months of hourly Apple stock data
    >>> aapl_data = load_financial_data('AAPL', period='6mo', interval='1h')
    >>> print(aapl_data.shape)

    >>> # Load EUR/USD forex data without cleaning
    >>> eurusd_data = load_financial_data('EURUSD=X', clean_data=False)
    """
    try:
        # Validate inputs
        _validate_parameters(symbol, period, interval)

        # Create ticker object
        ticker = yf.Ticker(symbol)

        # Download data
        logger.info(
            f"Downloading data for {symbol} (period={period}, interval={interval})"
        )
        data = ticker.history(period=period, interval=interval)

        if data.empty:
            logger.warning(f"No data retrieved for symbol {symbol}")
            return pd.DataFrame()

        # Standardize column names and format
        data = _standardize_dataframe(data)

        # Apply data cleaning if requested
        if clean_data:
            data = clean_financial_data(data, symbol)

        # Detect outliers if requested
        if detect_outliers:
            data = _detect_and_flag_outliers(data, symbol)

        logger.info(f"Successfully loaded {len(data)} rows of data for {symbol}")
        return data

    except Exception as e:
        logger.error(f"Error loading data for {symbol}: {str(e)}")
        return pd.DataFrame()


def clean_financial_data(data: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    """
    Clean financial data by handling missing values and basic validation.

    This function applies various cleaning strategies:
    - Forward-fill for price data (assumes last known price persists)
    - Linear interpolation for small gaps
    - Volume missing values set to 0
    - Removal of rows with all NaN values

    Parameters
    ----------
    data : pd.DataFrame
        Raw financial data with standard columns
    symbol : str, optional
        Symbol name for logging purposes

    Returns
    -------
    pd.DataFrame
        Cleaned financial data
    """
    if data.empty:
        return data

    original_length = len(data)
    cleaned_data = data.copy()

    # Remove rows where all price columns are NaN
    price_columns = ["Open", "High", "Low", "Close"]
    cleaned_data = cleaned_data.dropna(subset=price_columns, how="all")

    # Handle missing values in price columns
    for col in price_columns:
        if col in cleaned_data.columns:
            missing_count = cleaned_data[col].isnull().sum()
            if missing_count > 0:
                # Forward fill first, then backward fill for any remaining
                cleaned_data[col] = cleaned_data[col].ffill().bfill()

                # If still missing (e.g., all data is NaN), use interpolation
                if cleaned_data[col].isnull().sum() > 0:
                    cleaned_data[col] = cleaned_data[col].interpolate(method="linear")

                logger.info(
                    f"Filled {missing_count} missing values in {col} for {symbol}"
                )

    # Handle Volume column - fill with 0 for missing values
    if "Volume" in cleaned_data.columns:
        volume_missing = cleaned_data["Volume"].isnull().sum()
        if volume_missing > 0:
            cleaned_data["Volume"] = cleaned_data["Volume"].fillna(0)
            logger.info(
                f"Filled {volume_missing} missing Volume values with 0 for {symbol}"
            )

    # Validate data integrity
    cleaned_data = _validate_price_relationships(cleaned_data, symbol)

    final_length = len(cleaned_data)
    if final_length != original_length:
        logger.info(
            f"Data cleaning removed {original_length - final_length} rows for {symbol}"
        )

    return cleaned_data


def _validate_parameters(symbol: str, period: str, interval: str) -> None:
    """Validate input parameters for data loading."""
    if not symbol or not isinstance(symbol, str):
        raise ValueError("Symbol must be a non-empty string")

    valid_periods = [
        "1d",
        "5d",
        "1mo",
        "3mo",
        "6mo",
        "1y",
        "2y",
        "5y",
        "10y",
        "ytd",
        "max",
    ]
    if period not in valid_periods:
        raise ValueError(f"Invalid period '{period}'. Valid options: {valid_periods}")

    valid_intervals = [
        "1m",
        "2m",
        "5m",
        "15m",
        "30m",
        "60m",
        "90m",
        "1h",
        "1d",
        "5d",
        "1wk",
        "1mo",
        "3mo",
    ]
    if interval not in valid_intervals:
        raise ValueError(
            f"Invalid interval '{interval}'. Valid options: {valid_intervals}"
        )


def _standardize_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """Standardize DataFrame format and column names."""
    if data.empty:
        return data

    # Ensure we have the expected columns
    expected_columns = ["Open", "High", "Low", "Close", "Volume"]

    # Map any variations in column names
    column_mapping = {
        "Adj Close": "AdjClose",  # Keep as separate column if present
    }

    data = data.rename(columns=column_mapping)

    # Ensure DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        if "Date" in data.columns:
            data = data.set_index("Date")
        data.index = pd.to_datetime(data.index)

    # Ensure numeric data types for price columns
    for col in expected_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    return data


def _validate_price_relationships(data: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    """Validate logical relationships between OHLC prices."""
    if data.empty:
        return data

    validated_data = data.copy()
    price_columns = ["Open", "High", "Low", "Close"]

    if all(col in validated_data.columns for col in price_columns):
        # Check for invalid OHLC relationships
        invalid_high = validated_data["High"] < validated_data[
            ["Open", "Low", "Close"]
        ].max(axis=1)
        invalid_low = validated_data["Low"] > validated_data[
            ["Open", "High", "Close"]
        ].min(axis=1)

        invalid_count = (invalid_high | invalid_low).sum()
        if invalid_count > 0:
            logger.warning(
                f"Found {invalid_count} rows with invalid OHLC relationships for {symbol}"
            )
            # Option: Remove invalid rows or correct them
            # For now, we'll log the warning and keep the data

    return validated_data


def _detect_and_flag_outliers(data: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    """Detect potential outliers in financial data using statistical methods."""
    if data.empty or "Close" not in data.columns:
        return data

    result_data = data.copy()

    # Calculate percentage change
    pct_change = result_data["Close"].pct_change(fill_method=None)

    # Remove NaN values for calculation
    pct_change_clean = pct_change.dropna()

    if len(pct_change_clean) == 0:
        return result_data

    # Use median-based outlier detection (more robust)
    median_change = pct_change_clean.median()
    mad = np.median(
        np.abs(pct_change_clean - median_change)
    )  # Median Absolute Deviation

    # If MAD is 0, use standard deviation method
    if mad == 0:
        mean_change = pct_change_clean.mean()
        std_change = pct_change_clean.std()

        if not np.isnan(std_change) and std_change > 0:
            threshold = 3
            outlier_mask = np.abs(pct_change - mean_change) > threshold * std_change
        else:
            outlier_mask = pd.Series(False, index=pct_change.index)
    else:
        # Modified Z-score using MAD (more robust)
        threshold = 3.5  # Common threshold for modified Z-score
        modified_z_scores = 0.6745 * (pct_change - median_change) / mad
        outlier_mask = np.abs(modified_z_scores) > threshold

    outlier_count = outlier_mask.sum()

    if outlier_count > 0:
        logger.warning(f"Detected {outlier_count} potential outliers for {symbol}")
        # Add outlier flag column
        result_data["Outlier_Flag"] = outlier_mask

    return result_data


def get_available_symbols() -> List[str]:
    """
    Get a list of commonly available financial symbols for testing.

    Returns
    -------
    List[str]
        List of symbol strings that are typically available
    """
    return [
        "AAPL",  # Apple Inc.
        "MSFT",  # Microsoft Corporation
        "GOOGL",  # Alphabet Inc.
        "TSLA",  # Tesla Inc.
        "BTC-USD",  # Bitcoin
        "ETH-USD",  # Ethereum
        "EURUSD=X",  # EUR/USD
        "GBPUSD=X",  # GBP/USD
        "^GSPC",  # S&P 500
        "^DJI",  # Dow Jones Industrial Average
    ]


def validate_symbol(symbol: str) -> bool:
    """
    Validate if a symbol exists and has available data.

    Parameters
    ----------
    symbol : str
        Financial symbol to validate

    Returns
    -------
    bool
        True if symbol is valid and has data, False otherwise
    """
    try:
        ticker = yf.Ticker(symbol)
        # Try to get a small amount of recent data
        test_data = ticker.history(period="5d", interval="1d")
        return not test_data.empty
    except Exception:
        return False
