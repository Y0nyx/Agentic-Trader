"""
CSV data loading module for local financial data files.

This module provides functionality for loading and processing CSV files
containing historical financial data stored locally in the data/ directory.
"""

import os
import logging
from typing import Optional
import pandas as pd
from .loader import _standardize_dataframe, clean_financial_data

# Configure logging
logger = logging.getLogger(__name__)


def load_csv_data(
    filepath: str,
    clean_data: bool = True,
    validate_format: bool = True,
) -> pd.DataFrame:
    """
    Load financial data from a local CSV file.

    This function loads CSV files containing OHLCV data and applies
    the same standardization and cleaning as the yfinance loader.

    Parameters
    ----------
    filepath : str
        Path to the CSV file to load
    clean_data : bool, default True
        Whether to apply data cleaning (handle missing values)
    validate_format : bool, default True
        Whether to validate the CSV has expected OHLCV format

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized columns: Open, High, Low, Close, Volume
        and DatetimeIndex. Returns empty DataFrame if loading fails.

    Examples
    --------
    >>> # Load Bitcoin CSV data
    >>> btc_data = load_csv_data("data/BTC-USD.csv")
    >>> print(btc_data.head())

    >>> # Load Apple CSV data without cleaning
    >>> aapl_data = load_csv_data("data/AAPL.csv", clean_data=False)
    """
    try:
        if not os.path.exists(filepath):
            logger.error(f"CSV file not found: {filepath}")
            return pd.DataFrame()

        logger.info(f"Loading CSV data from {filepath}")
        
        # Load CSV data
        data = pd.read_csv(filepath)
        
        if data.empty:
            logger.warning(f"CSV file is empty: {filepath}")
            return pd.DataFrame()

        # Validate expected columns if requested
        if validate_format:
            expected_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in expected_columns if col not in data.columns]
            
            if missing_columns:
                logger.warning(f"Missing expected columns in {filepath}: {missing_columns}")
                # Continue processing - standardization might handle some variations

        # Standardize the DataFrame format
        data = _standardize_dataframe(data)
        
        # Apply data cleaning if requested
        if clean_data:
            symbol = os.path.splitext(os.path.basename(filepath))[0]
            data = clean_financial_data(data, symbol)

        logger.info(f"Successfully loaded {len(data)} rows from {filepath}")
        return data

    except Exception as e:
        logger.error(f"Error loading CSV file {filepath}: {str(e)}")
        return pd.DataFrame()


def get_available_csv_files(data_dir: Optional[str] = None) -> list:
    """
    Get list of available CSV files in the data directory.

    Parameters
    ----------
    data_dir : str, optional
        Directory to search for CSV files. If None, uses the data/ directory
        relative to this module.

    Returns
    -------
    list
        List of CSV filenames available in the data directory
    """
    if data_dir is None:
        # Get the directory where this module is located
        data_dir = os.path.dirname(os.path.abspath(__file__))
    
    try:
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        return sorted(csv_files)
    except OSError as e:
        logger.error(f"Error accessing data directory {data_dir}: {str(e)}")
        return []


def load_symbol_from_csv(symbol: str, data_dir: Optional[str] = None, **kwargs) -> pd.DataFrame:
    """
    Load data for a specific symbol from CSV files.

    This function looks for a CSV file matching the symbol name in the data directory.

    Parameters
    ----------
    symbol : str
        Symbol to load (e.g., 'AAPL', 'BTC-USD')
    data_dir : str, optional
        Directory containing CSV files. If None, uses the data/ directory
    **kwargs
        Additional arguments passed to load_csv_data

    Returns
    -------
    pd.DataFrame
        DataFrame with financial data for the symbol
    """
    if data_dir is None:
        data_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try exact filename match first
    csv_filename = f"{symbol}.csv"
    filepath = os.path.join(data_dir, csv_filename)
    
    if os.path.exists(filepath):
        return load_csv_data(filepath, **kwargs)
    
    # If not found, look for similar filenames (case insensitive)
    available_files = get_available_csv_files(data_dir)
    symbol_lower = symbol.lower()
    
    for filename in available_files:
        if filename.lower().startswith(symbol_lower.lower()):
            filepath = os.path.join(data_dir, filename)
            logger.info(f"Loading {symbol} from {filename}")
            return load_csv_data(filepath, **kwargs)
    
    logger.warning(f"No CSV file found for symbol {symbol} in {data_dir}")
    return pd.DataFrame()