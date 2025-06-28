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

Each directory contains its own README with detailed documentation and examples.

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

Here's a simple example to get you started:

```python
from data.csv_loader import load_csv_data
from strategies.moving_average_cross import MovingAverageCrossStrategy
from simulation.backtester import Backtester

# Load historical data
df = load_csv_data("data/GOOGL.csv")

# Create and run strategy
strategy = MovingAverageCrossStrategy(short_window=20, long_window=50)
signals = strategy.generate_signals(df)

# Run backtest
backtester = Backtester(initial_capital=10000)
performance_report = backtester.run_backtest(df, signals)

# View results
print(performance_report.summary())
```

## Documentation

For detailed documentation and examples, see the README files in each module:

- **[data/](data/README.md)** - Data loading and processing
- **[strategies/](strategies/README.md)** - Trading strategy development
- **[simulation/](simulation/README.md)** - Backtesting and performance analysis
- **[evaluation/](evaluation/README.md)** - Performance metrics and evaluation
- **[optimization/](optimization/README.md)** - Parameter optimization tools
- **[tests/](tests/README.md)** - Testing framework and guidelines

## Requirements

- Python 3.8+
- yfinance>=0.2.28 for financial data access
- pandas>=1.3.0 for data manipulation
- numpy>=1.16.5 for numerical operations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.