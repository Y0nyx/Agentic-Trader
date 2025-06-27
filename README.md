# Agentic Trader

An AI-powered trading agent with advanced strategies and simulation capabilities.

## Overview

Agentic Trader is a sophisticated algorithmic trading system that leverages artificial intelligence to develop, test, and optimize trading strategies. The project provides a comprehensive framework for financial data analysis, strategy development, backtesting, and performance evaluation.

## Features

- **Data Management**: Comprehensive data collection and processing capabilities
- **Strategy Development**: Modular framework for creating and testing trading strategies
- **Simulation Engine**: Advanced backtesting and simulation tools
- **Performance Evaluation**: Detailed metrics and analysis for strategy assessment
- **Optimization Tools**: Parameter tuning and strategy optimization capabilities

## Project Structure

```
agentic-trader/
├── data/           # Data collection and processing modules
├── strategies/     # Trading strategy implementations
├── simulation/     # Backtesting and simulation engine
├── evaluation/     # Performance metrics and analysis
├── optimization/   # Parameter optimization tools
├── tests/          # Unit and integration tests
├── requirements.txt
├── setup.py
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Y0nyx/Agentic-Trader.git
cd Agentic-Trader
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Quick Start

### Loading Financial Data

The data module provides robust functionality for loading and processing financial data:

```python
from data import load_financial_data

# Load 1 year of daily Bitcoin data
btc_data = load_financial_data('BTC-USD', period='1y', interval='1d')
print(f"Loaded {len(btc_data)} days of BTC data")
print(btc_data.head())

# Load 6 months of hourly Apple stock data
aapl_data = load_financial_data('AAPL', period='6mo', interval='1h')
print(f"Apple data shape: {aapl_data.shape}")

# Load EUR/USD forex data without automatic cleaning
eurusd_data = load_financial_data('EURUSD=X', period='1y', clean_data=False)
```

### Data Cleaning and Processing

The module automatically handles:
- **Missing values**: Forward-fill, interpolation, or removal
- **Outlier detection**: Statistical methods to identify anomalies  
- **Data standardization**: Consistent DataFrame format with OHLCV columns
- **Data validation**: Ensures price relationships are logical

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

### Supported Data Types

- **Stocks**: AAPL, MSFT, GOOGL, TSLA, etc.
- **Cryptocurrencies**: BTC-USD, ETH-USD, etc.
- **Forex**: EURUSD=X, GBPUSD=X, etc.
- **Indices**: ^GSPC (S&P 500), ^DJI (Dow Jones), etc.

### Parameters

- **period**: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'
- **interval**: '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'

## Requirements

- Python 3.8+
- yfinance>=0.2.28 for financial data access
- pandas>=1.3.0 for data manipulation
- numpy>=1.16.5 for numerical operations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.