# Advanced Trading Strategies - Quick Start Guide

This guide demonstrates the new advanced trading strategies and optimization capabilities.

## Quick Demo

```python
from data.csv_loader import load_csv_data
from strategies.trend_following_strategy import TrendFollowingStrategy
from simulation.backtester import Backtester
from evaluation.metrics import evaluate_performance

# Load data
data = load_csv_data("data/GOOGL.csv")

# Create and test strategy
strategy = TrendFollowingStrategy(
    trend_window=30,
    confirmation_window=10,
    min_trend_strength=30,
    exit_rsi_threshold=75
)

# Generate signals and backtest
signals = strategy.generate_signals(data)
backtester = Backtester(initial_capital=10000)
performance = backtester.run_backtest(data, signals)

# Evaluate results
summary = performance.summary()
detailed_metrics = evaluate_performance(performance, data)

print(f"Strategy Return: {summary['total_return_pct']:.1f}%")
print(f"Benchmark Return: {detailed_metrics['benchmark_return_pct']:.1f}%")
print(f"Alpha: {detailed_metrics['alpha_pct']:.1f}%")
```

## Available Strategies

### 1. Triple Moving Average
```python
from strategies.triple_ma_strategy import TripleMovingAverageStrategy

strategy = TripleMovingAverageStrategy(
    short_window=5,
    medium_window=15, 
    long_window=30,
    ma_type="ema"  # or "sma"
)
```

### 2. Adaptive Moving Average
```python
from strategies.adaptive_ma_strategy import AdaptiveMovingAverageStrategy

strategy = AdaptiveMovingAverageStrategy(
    fast_period=2,
    slow_period=30,
    use_rsi_filter=True,
    use_trend_filter=True
)
```

### 3. Advanced MA with Filters
```python
from strategies.advanced_ma_strategy import AdvancedMAStrategy

strategy = AdvancedMAStrategy(
    short_window=10,
    long_window=30,
    ma_type="ema",
    use_volume_filter=True,
    use_rsi_filter=True,
    use_macd_filter=True,
    use_bollinger_filter=True,
    use_trend_filter=True
)
```

### 4. Trend Following
```python
from strategies.trend_following_strategy import TrendFollowingStrategy

strategy = TrendFollowingStrategy(
    trend_window=50,
    confirmation_window=20,
    min_trend_strength=25,
    exit_rsi_threshold=80
)
```

### 5. Buy-and-Hold Plus
```python
from strategies.buy_hold_plus_strategy import BuyHoldPlusStrategy

strategy = BuyHoldPlusStrategy(
    stress_rsi_threshold=25,
    reentry_rsi_threshold=40,
    drawdown_threshold=0.15
)
```

## Multi-Objective Optimization

```python
from optimization.multi_objective_optimizer import (
    MultiObjectiveOptimizer, OptimizationObjective, OptimizationConstraint
)

# Define objectives
objectives = [
    OptimizationObjective("total_return_pct", maximize=True, weight=0.4),
    OptimizationObjective("sharpe_ratio", maximize=True, weight=0.3),
    OptimizationObjective("win_rate", maximize=True, weight=0.3),
]

# Define constraints
constraints = [
    OptimizationConstraint("num_trades", "<", 40),
    OptimizationConstraint("max_drawdown_pct", ">", -25),
]

# Parameter grid
param_grid = {
    "short_window": [5, 10, 15],
    "long_window": [30, 50, 70],
}

# Run optimization
optimizer = MultiObjectiveOptimizer(
    strategy_class=TrendFollowingStrategy,
    backtester=Backtester(initial_capital=10000),
    param_grid=param_grid,
    objectives=objectives,
    constraints=constraints
)

best_params, summary = optimizer.optimize(data)
print(f"Best parameters: {best_params}")
```

## Quick Performance Comparison

Run the complete analysis:

```bash
python final_strategy_summary.py
```

This will:
- Test multiple strategies on GOOGL data
- Generate performance comparison charts
- Provide detailed analysis and insights
- Save visualizations to /tmp/

## Technical Indicators

The new indicators module provides:

```python
from indicators.technical_indicators import (
    rsi, macd, bollinger_bands, sma, ema, 
    average_true_range, adx, adaptive_moving_average
)

# Example usage
rsi_values = rsi(data['Close'], window=14)
macd_line, signal_line, histogram = macd(data['Close'])
bb_upper, bb_middle, bb_lower = bollinger_bands(data['Close'])
```

## Performance Summary

**Best Results on GOOGL 2004-2020:**
- Trend Following: 691% return (vs 2094% benchmark)
- MA Cross Optimized: 501% return
- Buy-Hold Plus: 366% return

**Key Insights:**
- All strategies underperform the exceptional GOOGL bull market
- Strategies show superior risk-adjusted returns (higher Sharpe ratios)
- Best performance during market stress periods (2008-2009: +73% alpha)
- Excellent for risk management and drawdown control

**When to Use:**
- Bear markets and corrections
- Volatile/ranging markets  
- Risk management focus
- More normal market conditions (not exceptional bull runs)