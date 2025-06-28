# Data Module

The data module provides comprehensive functionality for loading, processing, and managing financial data for trading strategies and backtesting.

## Features

- **Yahoo Finance Integration**: Load real-time and historical market data
- **CSV Data Loading**: Load historical data from local CSV files
- **Data Cleaning**: Automatic data validation, cleaning, and standardization
- **Multiple Asset Types**: Support for stocks, cryptocurrencies, forex, and indices
- **Flexible Time Periods**: Configurable time periods and intervals

## Available Functions

### Yahoo Finance Data Loading

```python
from data import load_financial_data

# Load 1 year of daily Bitcoin data
btc_data = load_financial_data('BTC-USD', period='1y', interval='1d')
print(f"Loaded {len(btc_data)} days of BTC data")

# Load 6 months of hourly Apple stock data
aapl_data = load_financial_data('AAPL', period='6mo', interval='1h')
print(f"Apple data shape: {aapl_data.shape}")

# Load EUR/USD forex data without automatic cleaning
eurusd_data = load_financial_data('EURUSD=X', period='1y', clean_data=False)
```

### CSV Data Loading

```python
from data.csv_loader import load_csv_data, load_symbol_from_csv

# Load data from specific CSV file
df = load_csv_data("data/GOOGL.csv")
print(f"Loaded {len(df)} rows of data")

# Load data by symbol name (searches for matching CSV file)
tesla_data = load_symbol_from_csv("TSLA")
```

### Data Cleaning and Processing

```python
from data import clean_financial_data

# Clean raw data manually
cleaned_data = clean_financial_data(raw_data, symbol='AAPL')

# Available symbols for testing
from data import get_available_symbols, validate_symbol

symbols = get_available_symbols()
print(f"Available test symbols: {symbols}")

# Validate a symbol before using
if validate_symbol('TSLA'):
    tesla_data = load_financial_data('TSLA')
```

## CSV File Format

CSV files should contain the following columns:
- **Date**: Date in YYYY-MM-DD format (or datetime index)
- **Open**: Opening price
- **High**: Highest price
- **Low**: Lowest price
- **Close**: Closing price
- **Volume**: Trading volume

Example CSV format:
```csv
Date,Open,High,Low,Close,Volume
2020-01-01,100.0,105.0,98.0,103.0,1000000
2020-01-02,103.0,107.0,101.0,106.0,1200000
```

## Supported Data Types

- **Stocks**: AAPL, MSFT, GOOGL, TSLA, etc.
- **Cryptocurrencies**: BTC-USD, ETH-USD, etc.
- **Forex**: EURUSD=X, GBPUSD=X, etc.
- **Indices**: ^GSPC (S&P 500), ^DJI (Dow Jones), etc.

## Parameters

### Time Periods
- **period**: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'

### Intervals
- **interval**: '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'

## Data Cleaning Features

The module automatically handles:
- **Missing values**: Forward-fill, interpolation, or removal
- **Outlier detection**: Statistical methods to identify anomalies  
- **Data standardization**: Consistent DataFrame format with OHLCV columns
- **Data validation**: Ensures price relationships are logical (High >= Low, etc.)

## Available Test Data

The data directory includes sample CSV files for testing:
- `GOOGL.csv` - Google/Alphabet stock data
- `MSFT.csv` - Microsoft stock data
- `NVDA.csv` - NVIDIA stock data
- `TSLA.csv` - Tesla stock data
- `TSM.csv` - Taiwan Semiconductor stock data

## Error Handling

The module includes robust error handling for:
- Invalid symbols
- Network connectivity issues
- Malformed CSV files
- Missing or corrupted data
- Invalid date ranges