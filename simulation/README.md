# Simulation Module

The simulation module provides a comprehensive backtesting engine for testing trading strategies against historical data. It simulates realistic trading conditions including transaction costs, slippage, and cash management.

## Features

- **Portfolio Simulation**: Track cash, positions, and portfolio value over time
- **Transaction Costs**: Include commission fees and slippage in simulations
- **Realistic Order Execution**: Proper cash management and position tracking
- **Performance Metrics**: Comprehensive performance analysis and reporting
- **Risk Management**: Calculate drawdowns, Sharpe ratio, and other risk metrics

## Backtester Class

The main backtesting engine that simulates trading based on generated signals.

### Basic Usage

```python
from data.csv_loader import load_csv_data
from strategies.moving_average_cross import MovingAverageCrossStrategy
from simulation.backtester import Backtester

# Load data and generate signals
df = load_csv_data("data/GOOGL.csv")
strategy = MovingAverageCrossStrategy(short_window=20, long_window=50)
signals = strategy.generate_signals(df)

# Configure and run backtest
backtester = Backtester(initial_capital=10000, commission=0.001, slippage=0.0005)
performance_report = backtester.run_backtest(df, signals)

# Display results
print(performance_report.summary())
```

### Configuration Parameters

- **initial_capital** (float, default=10000): Starting portfolio value
- **commission** (float, default=0.001): Commission rate per trade (0.1% default)
- **slippage** (float, default=0.0005): Slippage rate per trade (0.05% default)

### Transaction Cost Model

The backtester includes realistic transaction costs:
- **Commission**: Fixed percentage fee per trade
- **Slippage**: Market impact cost simulating bid-ask spread
- **Total Cost**: `(commission + slippage) * trade_value`

## Performance Report

The backtester returns a comprehensive performance report with detailed metrics and analysis.

### Summary Metrics

```python
summary = performance_report.summary()
print(f"Initial Capital: ${summary['initial_capital']:,.2f}")
print(f"Final Value: ${summary['final_value']:,.2f}")
print(f"Total Return: {summary['total_return_pct']:.2f}%")
print(f"Annualized Return: {summary['annualized_return_pct']:.2f}%")
print(f"Max Drawdown: {summary['max_drawdown_pct']:.2f}%")
print(f"Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
print(f"Number of Trades: {summary['num_trades']}")
print(f"Win Rate: {summary['win_rate']:.1%}")
print(f"Profit Factor: {summary['profit_factor']:.2f}")
```

### Available Metrics

#### Return Metrics
- **Total Return**: Overall portfolio return percentage
- **Annualized Return**: Compound annual growth rate
- **Excess Return**: Return above risk-free rate

#### Risk Metrics
- **Sharpe Ratio**: Risk-adjusted return metric
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Volatility**: Standard deviation of returns

#### Trading Metrics
- **Number of Trades**: Total executed transactions
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profits to gross losses
- **Average Trade Return**: Mean return per trade

### Transaction History

```python
# Get detailed transaction history
transactions = performance_report.get_transactions_df()
print("\nTransaction History:")
print(transactions.head())

# Transaction columns include:
# - Date: Trade execution date
# - Signal: BUY/SELL signal
# - Shares: Number of shares traded
# - Price: Execution price
# - Value: Total transaction value
# - Commission: Commission paid
# - Cash: Remaining cash after trade
# - Portfolio_Value: Total portfolio value
```

### Portfolio Evolution

```python
# Get daily portfolio values
portfolio_history = performance_report.get_portfolio_history()
print("\nPortfolio History:")
print(portfolio_history.tail())

# Portfolio history includes:
# - Date: Trading date
# - Cash: Available cash
# - Shares: Shares held
# - Stock_Value: Value of stock holdings
# - Portfolio_Value: Total portfolio value
# - Daily_Return: Daily return percentage
```

## Complete Example

```python
from data.csv_loader import load_csv_data
from strategies.moving_average_cross import MovingAverageCrossStrategy
from simulation.backtester import Backtester

# Load historical data
df = load_csv_data("data/GOOGL.csv")

# Configure strategy
strategy = MovingAverageCrossStrategy(short_window=20, long_window=50)
signals = strategy.generate_signals(df)

# Configure backtester with transaction costs
backtester = Backtester(
    initial_capital=10000,    # Start with $10,000
    commission=0.001,         # 0.1% commission per trade
    slippage=0.0005          # 0.05% slippage per trade
)

# Run backtest
performance_report = backtester.run_backtest(df, signals)

# Display comprehensive results
print("=== BACKTEST RESULTS ===")
summary = performance_report.summary()

print(f"Initial Capital: ${summary['initial_capital']:,.2f}")
print(f"Final Value: ${summary['final_value']:,.2f}")
print(f"Total Return: {summary['total_return_pct']:.2f}%")
print(f"Annualized Return: {summary['annualized_return_pct']:.2f}%")
print(f"Max Drawdown: {summary['max_drawdown_pct']:.2f}%")
print(f"Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
print(f"Number of Trades: {summary['num_trades']}")
print(f"Win Rate: {summary['win_rate']:.1%}")
print(f"Profit Factor: {summary['profit_factor']:.2f}")

# Show recent transactions
transactions = performance_report.get_transactions_df()
print("\nRecent Transactions:")
print(transactions.tail())
```

## Order Execution Logic

The backtester implements realistic order execution:

1. **Signal Processing**: Process BUY/SELL signals from strategy
2. **Cash Management**: Ensure sufficient funds for purchases
3. **Position Tracking**: Maintain accurate share counts
4. **Cost Calculation**: Apply commission and slippage fees
5. **Portfolio Update**: Update cash and portfolio value

## Signal Requirements

The backtester expects signals DataFrame with:
- **Date**: Trading date (as index)
- **Signal**: Trading action ('BUY', 'SELL', or 'HOLD')
- **Close**: Closing price for order execution

## Performance Calculation

### Return Calculation
- Daily returns calculated from portfolio value changes
- Total return: `(final_value - initial_capital) / initial_capital`
- Annualized return: `(final_value / initial_capital) ^ (252 / trading_days) - 1`

### Risk Metrics
- Sharpe ratio: `(mean_return - risk_free_rate) / return_std`
- Maximum drawdown: Peak-to-trough decline from running maximum
- Volatility: Standard deviation of daily returns (annualized)

### Trading Metrics
- Win rate: Percentage of trades with positive returns
- Profit factor: Sum of gains / Sum of losses
- Average trade: Mean return per executed trade

## Assumptions and Limitations

- **Market Orders**: All trades executed at closing prices
- **Full Position**: Buy/sell entire cash position or stock holdings
- **No Partial Fills**: Orders completely filled or rejected
- **Daily Frequency**: One signal processed per trading day
- **No Borrowing**: Cannot buy with insufficient cash
- **No Short Selling**: Only long positions currently supported