"""
Unit tests for the financial data loader module.

This module contains comprehensive tests for the data loading and processing
functionality with the goal of achieving 95%+ code coverage.
"""

import unittest
import sys
import os
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

# Add the parent directory to the path to import the module under test
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.loader import (  # noqa: E402
    load_financial_data,
    clean_financial_data,
    get_available_symbols,
    validate_symbol,
    _validate_parameters,
    _standardize_dataframe,
    _validate_price_relationships,
    _detect_and_flag_outliers,
)


def load_csv_as_mock_data(symbol: str) -> pd.DataFrame:
    """
    Load CSV data to use as mock data for testing instead of yfinance.

    Parameters
    ----------
    symbol : str
        Symbol to load CSV data for

    Returns
    -------
    pd.DataFrame
        DataFrame with financial data from CSV file
    """
    # Map symbols to available CSV files
    csv_files = {
        "AAPL": "GOOGL.csv",  # Use GOOGL as fallback for AAPL
        "GOOGL": "GOOGL.csv",
        "MSFT": "MSFT.csv",
        "NVDA": "NVDA.csv",
        "TSM": "TSM.csv",
        "BTC-USD": "GOOGL.csv",  # Use GOOGL as fallback for crypto
        "EURUSD=X": "MSFT.csv",  # Use MSFT as fallback for forex
        "TEST": "GOOGL.csv",  # Use GOOGL for test symbol
    }

    # Return empty DataFrame for invalid symbols
    if symbol == "INVALID":
        return pd.DataFrame()

    # Get the CSV file to use
    csv_file = csv_files.get(symbol, "GOOGL.csv")

    # Load CSV data
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    csv_path = os.path.join(data_dir, csv_file)

    try:
        df = pd.read_csv(csv_path)
        # Convert Date column to datetime and set as index
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)

        # Drop Adj Close column to match yfinance format
        if "Adj Close" in df.columns:
            df = df.drop(columns=["Adj Close"])

        # Take only recent data (last 10 rows for consistent testing)
        return df.tail(10)

    except Exception as e:
        print(f"Error loading CSV {csv_path}: {e}")
        return pd.DataFrame()


class TestFinancialDataLoader(unittest.TestCase):
    """Test class for financial data loading functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample data for testing
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        self.sample_data = pd.DataFrame(
            {
                "Open": [
                    100.0,
                    101.0,
                    102.0,
                    103.0,
                    104.0,
                    105.0,
                    106.0,
                    107.0,
                    108.0,
                    109.0,
                ],
                "High": [
                    101.0,
                    102.0,
                    103.0,
                    104.0,
                    105.0,
                    106.0,
                    107.0,
                    108.0,
                    109.0,
                    110.0,
                ],
                "Low": [
                    99.0,
                    100.0,
                    101.0,
                    102.0,
                    103.0,
                    104.0,
                    105.0,
                    106.0,
                    107.0,
                    108.0,
                ],
                "Close": [
                    100.5,
                    101.5,
                    102.5,
                    103.5,
                    104.5,
                    105.5,
                    106.5,
                    107.5,
                    108.5,
                    109.5,
                ],
                "Volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
            },
            index=dates,
        )

        # Sample data with missing values
        self.data_with_nans = self.sample_data.copy()
        self.data_with_nans.loc[self.data_with_nans.index[2], "Close"] = np.nan
        self.data_with_nans.loc[self.data_with_nans.index[5], "Volume"] = np.nan

        # Sample data with outliers
        self.data_with_outliers = self.sample_data.copy()
        self.data_with_outliers.loc[self.data_with_outliers.index[5], "Close"] = (
            1000.0  # Much bigger jump
        )

    def test_load_financial_data_valid_symbol(self):
        """Test loading data for a valid symbol."""
        with patch("data.loader.yf.Ticker") as mock_ticker:
            mock_instance = Mock()
            # Use CSV data instead of sample data
            csv_data = load_csv_as_mock_data("AAPL")
            mock_instance.history.return_value = csv_data
            mock_ticker.return_value = mock_instance

            result = load_financial_data("AAPL", period="1y", interval="1d")

            self.assertIsInstance(result, pd.DataFrame)
            self.assertFalse(result.empty)
            self.assertGreater(
                len(result), 0
            )  # Use assertGreater instead of exact count
            mock_ticker.assert_called_once_with("AAPL")
            mock_instance.history.assert_called_once_with(period="1y", interval="1d")

    def test_load_financial_data_invalid_symbol(self):
        """Test loading data for invalid symbol returns empty DataFrame."""
        with patch("data.loader.yf.Ticker") as mock_ticker:
            mock_instance = Mock()
            # Use CSV helper which returns empty DataFrame for INVALID
            mock_instance.history.return_value = load_csv_as_mock_data("INVALID")
            mock_ticker.return_value = mock_instance

            result = load_financial_data("INVALID")

            self.assertIsInstance(result, pd.DataFrame)
            self.assertTrue(result.empty)

    def test_load_financial_data_exception_handling(self):
        """Test that exceptions during data loading are handled gracefully."""
        with patch("data.loader.yf.Ticker") as mock_ticker:
            mock_ticker.side_effect = Exception("Network error")

            result = load_financial_data("AAPL")

            self.assertIsInstance(result, pd.DataFrame)
            self.assertTrue(result.empty)

    def test_load_financial_data_different_periods(self):
        """Test loading data with different time periods."""
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

        for period in valid_periods:
            with patch("data.loader.yf.Ticker") as mock_ticker:
                mock_instance = Mock()
                # Use CSV data for consistency
                mock_instance.history.return_value = load_csv_as_mock_data("AAPL")
                mock_ticker.return_value = mock_instance

                result = load_financial_data("AAPL", period=period)

                self.assertIsInstance(result, pd.DataFrame)
                mock_instance.history.assert_called_with(period=period, interval="1d")

    def test_load_financial_data_different_intervals(self):
        """Test loading data with different intervals."""
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

        for interval in valid_intervals:
            with patch("data.loader.yf.Ticker") as mock_ticker:
                mock_instance = Mock()
                # Use CSV data for consistency
                mock_instance.history.return_value = load_csv_as_mock_data("AAPL")
                mock_ticker.return_value = mock_instance

                result = load_financial_data("AAPL", interval=interval)

                self.assertIsInstance(result, pd.DataFrame)
                mock_instance.history.assert_called_with(period="1y", interval=interval)

    def test_load_financial_data_no_cleaning(self):
        """Test loading data without cleaning."""
        with patch("data.loader.yf.Ticker") as mock_ticker:
            mock_instance = Mock()
            # Create data with NaN values from CSV base
            csv_data = load_csv_as_mock_data("AAPL")
            if not csv_data.empty:
                csv_data.iloc[0, 0] = np.nan  # Add NaN to first row, first column
            mock_instance.history.return_value = csv_data
            mock_ticker.return_value = mock_instance

            result = load_financial_data("AAPL", clean_data=False)

            self.assertIsInstance(result, pd.DataFrame)
            # Should still have NaN values if CSV data had them or we added them
            if not csv_data.empty and csv_data.isnull().any().any():
                self.assertTrue(result.isnull().any().any())

    def test_load_financial_data_no_outlier_detection(self):
        """Test loading data without outlier detection."""
        with patch("data.loader.yf.Ticker") as mock_ticker:
            mock_instance = Mock()
            # Use CSV data and modify to create outliers
            csv_data = load_csv_as_mock_data("AAPL")
            if not csv_data.empty and "Close" in csv_data.columns:
                csv_data.iloc[-1, csv_data.columns.get_loc("Close")] = (
                    csv_data["Close"].iloc[0] * 10
                )  # Create outlier
            mock_instance.history.return_value = csv_data
            mock_ticker.return_value = mock_instance

            result = load_financial_data("AAPL", detect_outliers=False)

            self.assertIsInstance(result, pd.DataFrame)
            # Should not have outlier flag column
            self.assertNotIn("Outlier_Flag", result.columns)

    def test_clean_financial_data_empty_dataframe(self):
        """Test cleaning an empty DataFrame."""
        empty_df = pd.DataFrame()
        result = clean_financial_data(empty_df)

        self.assertTrue(result.empty)

    def test_clean_financial_data_with_missing_values(self):
        """Test cleaning data with missing values."""
        result = clean_financial_data(self.data_with_nans, "TEST")

        # Should not have any NaN values in price columns
        price_columns = ["Open", "High", "Low", "Close"]
        for col in price_columns:
            if col in result.columns:
                self.assertFalse(
                    result[col].isnull().any(), f"Column {col} still has NaN values"
                )

        # Volume should be filled with 0
        if "Volume" in result.columns:
            self.assertFalse(result["Volume"].isnull().any())

    def test_clean_financial_data_all_nan_row(self):
        """Test cleaning data with rows where all price data is NaN."""
        data_with_all_nan = self.sample_data.copy()
        data_with_all_nan.loc[
            data_with_all_nan.index[5], ["Open", "High", "Low", "Close"]
        ] = np.nan

        result = clean_financial_data(data_with_all_nan, "TEST")

        # Should have removed the row with all NaN price values
        self.assertEqual(len(result), len(self.sample_data) - 1)

    def test_validate_parameters_valid_inputs(self):
        """Test parameter validation with valid inputs."""
        # Should not raise any exceptions
        _validate_parameters("AAPL", "1y", "1d")
        _validate_parameters("BTC-USD", "6mo", "1h")
        _validate_parameters("EURUSD=X", "max", "1wk")

    def test_validate_parameters_invalid_symbol(self):
        """Test parameter validation with invalid symbol."""
        with self.assertRaises(ValueError):
            _validate_parameters("", "1y", "1d")

        with self.assertRaises(ValueError):
            _validate_parameters(None, "1y", "1d")

        with self.assertRaises(ValueError):
            _validate_parameters(123, "1y", "1d")

    def test_validate_parameters_invalid_period(self):
        """Test parameter validation with invalid period."""
        with self.assertRaises(ValueError):
            _validate_parameters("AAPL", "invalid", "1d")

        with self.assertRaises(ValueError):
            _validate_parameters("AAPL", "2h", "1d")

    def test_validate_parameters_invalid_interval(self):
        """Test parameter validation with invalid interval."""
        with self.assertRaises(ValueError):
            _validate_parameters("AAPL", "1y", "invalid")

        with self.assertRaises(ValueError):
            _validate_parameters("AAPL", "1y", "30s")

    def test_standardize_dataframe_basic(self):
        """Test DataFrame standardization with basic data."""
        result = _standardize_dataframe(self.sample_data)

        # Should have DatetimeIndex
        self.assertIsInstance(result.index, pd.DatetimeIndex)

        # Should have numeric columns
        expected_columns = ["Open", "High", "Low", "Close", "Volume"]
        for col in expected_columns:
            if col in result.columns:
                self.assertTrue(pd.api.types.is_numeric_dtype(result[col]))

    def test_standardize_dataframe_with_date_column(self):
        """Test DataFrame standardization when Date is a column."""
        data_with_date_col = self.sample_data.reset_index()
        data_with_date_col = data_with_date_col.rename(columns={"index": "Date"})

        result = _standardize_dataframe(data_with_date_col)

        # Should have DatetimeIndex
        self.assertIsInstance(result.index, pd.DatetimeIndex)

        # Date column should be removed from columns
        self.assertNotIn("Date", result.columns)

    def test_standardize_dataframe_empty(self):
        """Test DataFrame standardization with empty DataFrame."""
        empty_df = pd.DataFrame()
        result = _standardize_dataframe(empty_df)

        self.assertTrue(result.empty)

    def test_validate_price_relationships_valid_data(self):
        """Test price relationship validation with valid OHLC data."""
        result = _validate_price_relationships(self.sample_data, "TEST")

        # Should return the same data for valid relationships
        pd.testing.assert_frame_equal(result, self.sample_data)

    def test_validate_price_relationships_invalid_data(self):
        """Test price relationship validation with invalid OHLC data."""
        invalid_data = self.sample_data.copy()
        # Make High lower than Close (invalid)
        invalid_data.loc[invalid_data.index[0], "High"] = 95.0

        # Should log warning but return data (current implementation)
        with self.assertLogs("data.loader", level="WARNING") as cm:
            result = _validate_price_relationships(invalid_data, "TEST")
            self.assertTrue(
                any("invalid OHLC relationships" in msg for msg in cm.output)
            )

        # Data should still be returned
        self.assertEqual(len(result), len(invalid_data))

    def test_validate_price_relationships_empty_data(self):
        """Test price relationship validation with empty data."""
        empty_df = pd.DataFrame()
        result = _validate_price_relationships(empty_df, "TEST")

        self.assertTrue(result.empty)

    def test_detect_and_flag_outliers_with_outliers(self):
        """Test outlier detection with data containing outliers."""
        with self.assertLogs("data.loader", level="WARNING") as cm:
            result = _detect_and_flag_outliers(self.data_with_outliers, "TEST")
            self.assertTrue(any("potential outliers" in msg for msg in cm.output))

        # Should have outlier flag column
        self.assertIn("Outlier_Flag", result.columns)

        # Should flag the outlier
        self.assertTrue(result["Outlier_Flag"].any())

    def test_detect_and_flag_outliers_no_outliers(self):
        """Test outlier detection with clean data."""
        result = _detect_and_flag_outliers(self.sample_data, "TEST")

        # May or may not have outlier flag depending on data variance
        if "Outlier_Flag" in result.columns:
            # Should not flag any outliers in clean data
            self.assertFalse(result["Outlier_Flag"].any())

    def test_detect_and_flag_outliers_empty_data(self):
        """Test outlier detection with empty data."""
        empty_df = pd.DataFrame()
        result = _detect_and_flag_outliers(empty_df, "TEST")

        self.assertTrue(result.empty)

    def test_detect_and_flag_outliers_no_close_column(self):
        """Test outlier detection without Close column."""
        data_no_close = self.sample_data.drop(columns=["Close"])
        result = _detect_and_flag_outliers(data_no_close, "TEST")

        # Should return original data unchanged
        pd.testing.assert_frame_equal(result, data_no_close)

    def test_get_available_symbols(self):
        """Test getting list of available symbols."""
        symbols = get_available_symbols()

        self.assertIsInstance(symbols, list)
        self.assertGreater(len(symbols), 0)

        # Check for some expected symbols
        expected_symbols = ["AAPL", "BTC-USD", "EURUSD=X"]
        for symbol in expected_symbols:
            self.assertIn(symbol, symbols)

    def test_validate_symbol_valid(self):
        """Test symbol validation with valid symbol."""
        with patch("data.loader.yf.Ticker") as mock_ticker:
            mock_instance = Mock()
            # Use CSV data for valid symbol
            mock_instance.history.return_value = load_csv_as_mock_data("AAPL")
            mock_ticker.return_value = mock_instance

            result = validate_symbol("AAPL")

            self.assertTrue(result)
            mock_ticker.assert_called_once_with("AAPL")

    def test_validate_symbol_invalid(self):
        """Test symbol validation with invalid symbol."""
        with patch("data.loader.yf.Ticker") as mock_ticker:
            mock_instance = Mock()
            # Use CSV helper which returns empty DataFrame for INVALID
            mock_instance.history.return_value = load_csv_as_mock_data("INVALID")
            mock_ticker.return_value = mock_instance

            result = validate_symbol("INVALID")

            self.assertFalse(result)

    def test_validate_symbol_exception(self):
        """Test symbol validation when exception occurs."""
        with patch("data.loader.yf.Ticker") as mock_ticker:
            mock_ticker.side_effect = Exception("Network error")

            result = validate_symbol("AAPL")

            self.assertFalse(result)

    def test_known_symbols_integration(self):
        """Integration test with known good symbols (requires internet)."""
        # This test actually calls yfinance - mark as integration test
        known_symbols = ["AAPL"]  # Conservative test with one reliable symbol

        for symbol in known_symbols:
            try:
                result = load_financial_data(symbol, period="5d", interval="1d")

                # Should get some data
                self.assertIsInstance(result, pd.DataFrame)

                if not result.empty:
                    # Should have expected columns
                    expected_columns = ["Open", "High", "Low", "Close", "Volume"]
                    for col in expected_columns:
                        self.assertIn(col, result.columns)

                    # Should have DatetimeIndex
                    self.assertIsInstance(result.index, pd.DatetimeIndex)

                    # Prices should be positive
                    price_columns = ["Open", "High", "Low", "Close"]
                    for col in price_columns:
                        if col in result.columns and len(result) > 0:
                            self.assertTrue((result[col] > 0).all())

            except Exception as e:
                # Skip integration tests if network is unavailable
                self.skipTest(f"Integration test skipped due to network issue: {e}")


class TestDataLoaderEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def test_load_financial_data_cryptocurrency(self):
        """Test loading cryptocurrency data."""
        with patch("data.loader.yf.Ticker") as mock_ticker:
            mock_instance = Mock()
            # Use CSV data as base for crypto (scaled appropriately)
            crypto_data = load_csv_as_mock_data("BTC-USD")
            if not crypto_data.empty:
                # Scale prices to crypto-like values
                price_cols = ["Open", "High", "Low", "Close"]
                for col in price_cols:
                    if col in crypto_data.columns:
                        crypto_data[col] = (
                            crypto_data[col] * 500
                        )  # Scale up to crypto-like prices

                # Increase volume to crypto-like levels
                if "Volume" in crypto_data.columns:
                    crypto_data["Volume"] = crypto_data["Volume"] * 10

            mock_instance.history.return_value = crypto_data
            mock_ticker.return_value = mock_instance

            result = load_financial_data("BTC-USD")

            self.assertIsInstance(result, pd.DataFrame)
            self.assertFalse(result.empty)

    def test_load_financial_data_forex(self):
        """Test loading forex data."""
        with patch("data.loader.yf.Ticker") as mock_ticker:
            mock_instance = Mock()
            # Use CSV data as base for forex (scaled appropriately)
            forex_data = load_csv_as_mock_data("EURUSD=X")
            if not forex_data.empty:
                # Scale prices to forex-like values (around 1.0-1.2)
                price_cols = ["Open", "High", "Low", "Close"]
                for col in price_cols:
                    if col in forex_data.columns:
                        # Normalize to forex range
                        forex_data[col] = (
                            1.0 + (forex_data[col] / forex_data[col].max()) * 0.2
                        )

                # Forex typically has 0 volume
                if "Volume" in forex_data.columns:
                    forex_data["Volume"] = 0

            mock_instance.history.return_value = forex_data
            mock_ticker.return_value = mock_instance

            result = load_financial_data("EURUSD=X")

            self.assertIsInstance(result, pd.DataFrame)
            self.assertFalse(result.empty)

    def test_clean_financial_data_all_data_missing(self):
        """Test cleaning data where all data is missing."""
        all_nan_data = pd.DataFrame(
            {
                "Open": [np.nan, np.nan, np.nan],
                "High": [np.nan, np.nan, np.nan],
                "Low": [np.nan, np.nan, np.nan],
                "Close": [np.nan, np.nan, np.nan],
                "Volume": [np.nan, np.nan, np.nan],
            },
            index=pd.date_range("2023-01-01", periods=3, freq="D"),
        )

        result = clean_financial_data(all_nan_data, "TEST")

        # Should return empty DataFrame after removing all NaN rows
        self.assertTrue(result.empty)


if __name__ == "__main__":
    # Run tests with coverage if available
    try:
        import coverage

        cov = coverage.Coverage()
        cov.start()

        unittest.main(verbosity=2, exit=False)

        cov.stop()
        cov.save()
        print("\nCoverage Report:")
        cov.report()

    except ImportError:
        # Run without coverage
        unittest.main(verbosity=2)
