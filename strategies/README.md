# Strategies Module

The strategies module provides a framework for implementing and testing trading strategies. It includes pre-built strategies and a foundation for developing custom trading algorithms.

## Available Strategies

### Moving Average Crossover Strategy

A classic technical analysis strategy that generates buy/sell signals based on the crossover of short-term and long-term moving averages.

#### How It Works

- **Bullish Signal (BUY)**: When the short-term moving average crosses above the long-term moving average
- **Bearish Signal (SELL)**: When the short-term moving average crosses below the long-term moving average
- **Hold Signal**: When no crossover occurs

#### Usage Example

```python
from data.csv_loader import load_csv_data
from strategies.moving_average_cross import MovingAverageCrossStrategy

# Load historical data
df = load_csv_data("data/TSLA.csv")

# Configure the strategy
strategy = MovingAverageCrossStrategy(short_window=20, long_window=50)

# Generate trading signals
signals = strategy.generate_signals(df)

# View signal summary
print("Signal Summary:")
print(signals['Signal'].value_counts())

# Get strategy performance summary
summary = strategy.get_strategy_summary(signals)
print("\nStrategy Summary:", summary)
```

#### Configuration Parameters

- **short_window** (int, default=20): Period for short-term moving average
- **long_window** (int, default=50): Period for long-term moving average

#### Signal Output

The strategy returns a DataFrame with the following columns:
- **Date**: Trading date
- **Close**: Closing price
- **Short_MA**: Short-term moving average value
- **Long_MA**: Long-term moving average value
- **Signal**: Trading signal ('BUY', 'SELL', or 'HOLD')
- **Position**: Position indicator (1 for long, 0 for flat, -1 for short)

#### Strategy Metrics

The strategy provides performance summary including:
- Total number of signals generated
- Breakdown by signal type (BUY/SELL/HOLD)
- Signal frequency statistics
- Strategy configuration parameters

## Strategy Development Framework

### Base Strategy Class

All strategies should inherit from a base strategy class and implement the following methods:

```python
class BaseStrategy:
    def generate_signals(self, data):
        """Generate trading signals from market data"""
        raise NotImplementedError
    
    def get_strategy_summary(self, signals):
        """Get summary statistics for the strategy"""
        raise NotImplementedError
```

### Custom Strategy Example

```python
from strategies.moving_average_cross import MovingAverageCrossStrategy

class CustomStrategy(MovingAverageCrossStrategy):
    def __init__(self, short_window=10, long_window=30, volume_threshold=1000000):
        super().__init__(short_window, long_window)
        self.volume_threshold = volume_threshold
    
    def generate_signals(self, data):
        # Get base MA signals
        signals = super().generate_signals(data)
        
        # Add volume filter
        signals.loc[data['Volume'] < self.volume_threshold, 'Signal'] = 'HOLD'
        
        return signals
```

## Signal Types

- **BUY**: Enter a long position
- **SELL**: Enter a short position (or exit long position)
- **HOLD**: Maintain current position (no action)

## Data Requirements

Strategies expect input data with the following columns:
- **Date**: Trading date (as index or column)
- **Open**: Opening price
- **High**: Highest price of the period
- **Low**: Lowest price of the period
- **Close**: Closing price
- **Volume**: Trading volume

## Performance Considerations

- Strategies calculate moving averages using pandas rolling windows for efficiency
- Signal generation is vectorized for fast computation on large datasets
- Memory usage is optimized for processing multi-year datasets

## Integration with Backtester

Strategies are designed to work seamlessly with the simulation module:

```python
from data.csv_loader import load_csv_data
from strategies.moving_average_cross import MovingAverageCrossStrategy
from simulation.backtester import Backtester

# Load data and generate signals
df = load_csv_data("data/GOOGL.csv")
strategy = MovingAverageCrossStrategy(short_window=20, long_window=50)
signals = strategy.generate_signals(df)

# Run backtest
backtester = Backtester(initial_capital=10000)
performance_report = backtester.run_backtest(df, signals)
```

## Future Strategy Ideas

The framework supports implementing various trading strategies:
- **Technical Indicators**: RSI, MACD, Bollinger Bands
- **Mean Reversion**: Statistical arbitrage strategies
- **Momentum**: Trend-following strategies
- **Machine Learning**: AI-powered prediction models
- **Multi-Asset**: Portfolio allocation strategies