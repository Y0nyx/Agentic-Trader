"""
Tests for CSV data loading functionality.

This module tests the CSV loading functionality to ensure proper data handling,
format validation, and error handling.
"""

import unittest
import os
import pandas as pd
import numpy as np
from unittest.mock import patch, Mock
import tempfile
from data.csv_loader import (
    load_csv_data,
    get_available_csv_files,
    load_symbol_from_csv
)


class TestCSVLoader(unittest.TestCase):
    """Test class for CSV data loading functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample CSV data for testing
        self.sample_csv_content = """Date,Open,High,Low,Close,Adj Close,Volume
2023-01-01,100.0,105.0,99.0,104.0,104.0,1000000
2023-01-02,104.0,108.0,103.0,107.0,107.0,1200000
2023-01-03,107.0,109.0,105.0,106.0,106.0,1100000
2023-01-04,106.0,110.0,104.0,108.0,108.0,1300000
"""
        
        # Create sample CSV with missing data
        self.csv_with_missing = """Date,Open,High,Low,Close,Adj Close,Volume
2023-01-01,100.0,105.0,99.0,104.0,104.0,1000000
2023-01-02,,108.0,103.0,107.0,107.0,1200000
2023-01-03,107.0,109.0,,106.0,106.0,1100000
2023-01-04,106.0,110.0,104.0,,108.0,1300000
"""
        
        # Sample with invalid format
        self.invalid_csv_content = """Symbol,Price,Volume
AAPL,150.0,1000000
GOOGL,2500.0,800000
"""
    
    def test_load_csv_data_valid_file(self):
        """Test loading a valid CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            tmp_file.write(self.sample_csv_content)
            tmp_file_path = tmp_file.name
        
        try:
            result = load_csv_data(tmp_file_path)
            
            self.assertIsInstance(result, pd.DataFrame)
            self.assertFalse(result.empty)
            self.assertEqual(len(result), 4)
            
            # Check columns
            expected_columns = ['Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume']
            for col in expected_columns:
                self.assertIn(col, result.columns)
            
            # Check index is DatetimeIndex
            self.assertIsInstance(result.index, pd.DatetimeIndex)
            
        finally:
            os.unlink(tmp_file_path)
    
    def test_load_csv_data_missing_file(self):
        """Test loading a non-existent CSV file."""
        result = load_csv_data('non_existent_file.csv')
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)
    
    def test_load_csv_data_with_missing_values(self):
        """Test loading CSV with missing values."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            tmp_file.write(self.csv_with_missing)
            tmp_file_path = tmp_file.name
        
        try:
            result = load_csv_data(tmp_file_path, clean_data=True)
            
            self.assertFalse(result.empty)
            
            # Check that missing values are handled
            price_columns = ['Open', 'High', 'Low', 'Close']
            for col in price_columns:
                if col in result.columns:
                    self.assertFalse(result[col].isnull().any(), f"Column {col} still has NaN values")
            
        finally:
            os.unlink(tmp_file_path)
    
    def test_load_csv_data_no_cleaning(self):
        """Test loading CSV without cleaning."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            tmp_file.write(self.csv_with_missing)
            tmp_file_path = tmp_file.name
        
        try:
            result = load_csv_data(tmp_file_path, clean_data=False)
            
            self.assertFalse(result.empty)
            # Should still have some NaN values
            has_nans = result.isnull().any().any()
            self.assertTrue(has_nans)
            
        finally:
            os.unlink(tmp_file_path)
    
    def test_load_csv_data_empty_file(self):
        """Test loading an empty CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            tmp_file.write("")  # Empty file
            tmp_file_path = tmp_file.name
        
        try:
            result = load_csv_data(tmp_file_path)
            
            self.assertIsInstance(result, pd.DataFrame)
            self.assertTrue(result.empty)
            
        finally:
            os.unlink(tmp_file_path)
    
    def test_load_csv_data_invalid_format(self):
        """Test loading CSV with invalid format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            tmp_file.write(self.invalid_csv_content)
            tmp_file_path = tmp_file.name
        
        try:
            # Should load but warn about missing columns
            with self.assertLogs('data.csv_loader', level='WARNING') as cm:
                result = load_csv_data(tmp_file_path, validate_format=True)
                self.assertTrue(any('Missing expected columns' in msg for msg in cm.output))
            
        finally:
            os.unlink(tmp_file_path)
    
    def test_get_available_csv_files(self):
        """Test getting list of available CSV files."""
        # Test with actual data directory
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        csv_files = get_available_csv_files(data_dir)
        
        self.assertIsInstance(csv_files, list)
        self.assertGreater(len(csv_files), 0)
        
        # All files should end with .csv
        for filename in csv_files:
            self.assertTrue(filename.endswith('.csv'))
        
        # Should be sorted
        self.assertEqual(csv_files, sorted(csv_files))
    
    def test_get_available_csv_files_invalid_directory(self):
        """Test getting CSV files from invalid directory."""
        csv_files = get_available_csv_files('/non/existent/directory')
        
        self.assertIsInstance(csv_files, list)
        self.assertEqual(len(csv_files), 0)
    
    def test_load_symbol_from_csv_exact_match(self):
        """Test loading symbol with exact filename match."""
        # Test with actual data files
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        
        # Try to load GOOGL.csv
        result = load_symbol_from_csv('GOOGL', data_dir)
        
        if os.path.exists(os.path.join(data_dir, 'GOOGL.csv')):
            self.assertFalse(result.empty)
            self.assertIsInstance(result.index, pd.DatetimeIndex)
        else:
            self.assertTrue(result.empty)
    
    def test_load_symbol_from_csv_case_insensitive(self):
        """Test loading symbol with case insensitive matching."""
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        
        # Try different cases
        result1 = load_symbol_from_csv('googl', data_dir)  # lowercase
        result2 = load_symbol_from_csv('GOOGL', data_dir)  # uppercase
        
        # Both should work if file exists
        if os.path.exists(os.path.join(data_dir, 'GOOGL.csv')):
            self.assertFalse(result1.empty)
            self.assertFalse(result2.empty)
            # Should load the same data
            pd.testing.assert_frame_equal(result1, result2)
    
    def test_load_symbol_from_csv_not_found(self):
        """Test loading symbol that doesn't exist."""
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        
        result = load_symbol_from_csv('NONEXISTENT', data_dir)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)
    
    def test_load_csv_data_integration_with_actual_files(self):
        """Test loading actual CSV files from the data directory."""
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        
        # Get available files
        csv_files = get_available_csv_files(data_dir)
        
        if csv_files:
            # Test loading the first available file
            first_file = csv_files[0]
            file_path = os.path.join(data_dir, first_file)
            
            result = load_csv_data(file_path)
            
            self.assertFalse(result.empty)
            self.assertIsInstance(result.index, pd.DatetimeIndex)
            
            # Check basic OHLCV structure
            basic_columns = ['Open', 'High', 'Low', 'Close']
            for col in basic_columns:
                if col in result.columns:
                    self.assertTrue(pd.api.types.is_numeric_dtype(result[col]))


class TestCSVLoaderEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions for CSV loader."""
    
    def test_load_csv_malformed_content(self):
        """Test loading CSV with malformed content."""
        malformed_content = """Date,Open,High,Low,Close,Volume
2023-01-01,abc,105.0,99.0,104.0,1000000
2023-01-02,104.0,def,103.0,107.0,1200000
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            tmp_file.write(malformed_content)
            tmp_file_path = tmp_file.name
        
        try:
            result = load_csv_data(tmp_file_path)
            
            # Should handle conversion errors gracefully
            self.assertIsInstance(result, pd.DataFrame)
            
        finally:
            os.unlink(tmp_file_path)
    
    def test_load_csv_data_with_different_date_formats(self):
        """Test loading CSV with different date formats."""
        different_date_format = """Date,Open,High,Low,Close,Volume
01/01/2023,100.0,105.0,99.0,104.0,1000000
01/02/2023,104.0,108.0,103.0,107.0,1200000
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            tmp_file.write(different_date_format)
            tmp_file_path = tmp_file.name
        
        try:
            result = load_csv_data(tmp_file_path)
            
            self.assertFalse(result.empty)
            self.assertIsInstance(result.index, pd.DatetimeIndex)
            
        finally:
            os.unlink(tmp_file_path)


if __name__ == '__main__':
    unittest.main()