#!/usr/bin/env python3
"""
Visualization and Analysis Tools for Strategy Comparison.

This script creates charts and detailed analysis to understand why
strategies are underperforming and identify improvements.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any

# Add the project root to the path
project_root = "/home/runner/work/Agentic-Trader/Agentic-Trader"
sys.path.insert(0, project_root)

from data.csv_loader import load_csv_data
from simulation.backtester import Backtester
from evaluation.metrics import evaluate_performance
from strategies.moving_average_cross import MovingAverageCrossStrategy
from strategies.trend_following_strategy import TrendFollowingStrategy


def create_performance_comparison_chart(
    data: pd.DataFrame,
    strategies_data: Dict,
    save_path: str = "/tmp/strategy_comparison.png",
):
    """Create a chart comparing strategy performance vs buy-and-hold."""

    plt.figure(figsize=(15, 10))

    # Calculate buy-and-hold performance
    initial_capital = 10000
    buy_hold_value = initial_capital * (data["Close"] / data["Close"].iloc[0])

    # Plot buy-and-hold
    plt.subplot(2, 2, 1)
    plt.plot(data.index, buy_hold_value, label="Buy & Hold", linewidth=2, color="black")

    # Plot strategies
    colors = ["blue", "red", "green", "orange", "purple"]
    for i, (name, strategy_info) in enumerate(strategies_data.items()):
        portfolio_history = strategy_info["portfolio_history"]
        plt.plot(
            portfolio_history.index,
            portfolio_history["Total_Value"],
            label=name,
            linewidth=1.5,
            color=colors[i % len(colors)],
        )

    plt.title("Portfolio Value Over Time")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale("log")  # Log scale to better see differences

    # Plot drawdowns
    plt.subplot(2, 2, 2)

    # Buy-and-hold drawdown
    buy_hold_peak = buy_hold_value.expanding().max()
    buy_hold_drawdown = (buy_hold_value - buy_hold_peak) / buy_hold_peak
    plt.plot(
        data.index,
        buy_hold_drawdown * 100,
        label="Buy & Hold",
        linewidth=2,
        color="black",
    )

    # Strategy drawdowns
    for i, (name, strategy_info) in enumerate(strategies_data.items()):
        portfolio_history = strategy_info["portfolio_history"]
        portfolio_value = portfolio_history["Total_Value"]
        peak = portfolio_value.expanding().max()
        drawdown = (portfolio_value - peak) / peak
        plt.plot(
            portfolio_history.index,
            drawdown * 100,
            label=name,
            linewidth=1.5,
            color=colors[i % len(colors)],
        )

    plt.title("Drawdown Over Time")
    plt.xlabel("Date")
    plt.ylabel("Drawdown (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot rolling Sharpe ratio
    plt.subplot(2, 2, 3)

    # Buy-and-hold rolling Sharpe
    buy_hold_returns = buy_hold_value.pct_change().dropna()
    rolling_sharpe_bh = (
        buy_hold_returns.rolling(252).mean()
        / buy_hold_returns.rolling(252).std()
        * np.sqrt(252)
    )
    plt.plot(
        data.index[252:],
        rolling_sharpe_bh.iloc[252:],
        label="Buy & Hold",
        linewidth=2,
        color="black",
    )

    # Strategy rolling Sharpe
    for i, (name, strategy_info) in enumerate(strategies_data.items()):
        portfolio_history = strategy_info["portfolio_history"]
        strategy_returns = portfolio_history["Total_Value"].pct_change().dropna()
        rolling_sharpe = (
            strategy_returns.rolling(252).mean()
            / strategy_returns.rolling(252).std()
            * np.sqrt(252)
        )
        plt.plot(
            portfolio_history.index[252:],
            rolling_sharpe.iloc[252:],
            label=name,
            linewidth=1.5,
            color=colors[i % len(colors)],
        )

    plt.title("Rolling Sharpe Ratio (1 Year)")
    plt.xlabel("Date")
    plt.ylabel("Sharpe Ratio")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot trade frequency
    plt.subplot(2, 2, 4)

    strategy_names = []
    trade_counts = []
    returns = []

    for name, strategy_info in strategies_data.items():
        strategy_names.append(name)
        trade_counts.append(strategy_info["num_trades"])
        returns.append(strategy_info["total_return_pct"])

    # Add buy-and-hold (0 trades)
    strategy_names.append("Buy & Hold")
    trade_counts.append(0)
    buy_hold_return = ((data["Close"].iloc[-1] / data["Close"].iloc[0]) - 1) * 100
    returns.append(buy_hold_return)

    # Create scatter plot
    colors_extended = colors + ["black"]
    for i, (name, trades, ret) in enumerate(zip(strategy_names, trade_counts, returns)):
        plt.scatter(
            trades,
            ret,
            s=100,
            color=colors_extended[i % len(colors_extended)],
            label=name,
            alpha=0.7,
        )

    plt.title("Return vs Number of Trades")
    plt.xlabel("Number of Trades")
    plt.ylabel("Total Return (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Chart saved to: {save_path}")

    return save_path


def analyze_market_periods(data: pd.DataFrame):
    """Analyze different market periods to understand when strategies might work better."""

    print("\n" + "=" * 80)
    print("MARKET PERIOD ANALYSIS")
    print("=" * 80)

    # Calculate rolling returns for trend analysis
    data["Returns_1M"] = data["Close"].pct_change(periods=21)  # 1 month
    data["Returns_3M"] = data["Close"].pct_change(periods=63)  # 3 months
    data["Returns_6M"] = data["Close"].pct_change(periods=126)  # 6 months
    data["Returns_1Y"] = data["Close"].pct_change(periods=252)  # 1 year

    # Identify different market regimes
    # Bull market: 6M return > 10%
    # Bear market: 6M return < -10%
    # Sideways: -10% <= 6M return <= 10%

    bull_periods = data["Returns_6M"] > 0.10
    bear_periods = data["Returns_6M"] < -0.10
    sideways_periods = (data["Returns_6M"] >= -0.10) & (data["Returns_6M"] <= 0.10)

    bull_count = bull_periods.sum()
    bear_count = bear_periods.sum()
    sideways_count = sideways_periods.sum()
    total_periods = len(data.dropna())

    print(f"Market Regime Analysis:")
    print(f"  Bull Market periods: {bull_count} ({bull_count/total_periods:.1%})")
    print(f"  Bear Market periods: {bear_count} ({bear_count/total_periods:.1%})")
    print(f"  Sideways periods: {sideways_count} ({sideways_count/total_periods:.1%})")

    # Analyze volatility
    data["Volatility"] = data["Close"].pct_change().rolling(21).std() * np.sqrt(252)

    print(f"\nVolatility Analysis:")
    print(f"  Average Volatility: {data['Volatility'].mean():.1%}")
    print(f"  Max Volatility: {data['Volatility'].max():.1%}")
    print(f"  Min Volatility: {data['Volatility'].min():.1%}")

    # Identify high/low volatility periods
    high_vol_threshold = data["Volatility"].quantile(0.75)
    low_vol_threshold = data["Volatility"].quantile(0.25)

    high_vol_periods = data["Volatility"] > high_vol_threshold
    low_vol_periods = data["Volatility"] < low_vol_threshold

    print(
        f"  High Volatility periods (>75th percentile): {high_vol_periods.sum()} ({high_vol_periods.sum()/total_periods:.1%})"
    )
    print(
        f"  Low Volatility periods (<25th percentile): {low_vol_periods.sum()} ({low_vol_periods.sum()/total_periods:.1%})"
    )

    # Major drawdown periods
    rolling_max = data["Close"].expanding().max()
    drawdown = (data["Close"] - rolling_max) / rolling_max
    major_drawdown_periods = drawdown < -0.20  # 20%+ drawdown

    print(f"\nDrawdown Analysis:")
    print(
        f"  Periods with >20% drawdown: {major_drawdown_periods.sum()} ({major_drawdown_periods.sum()/total_periods:.1%})"
    )
    print(f"  Maximum drawdown: {drawdown.min():.1%}")

    # Time-based analysis
    data["Year"] = data.index.year
    yearly_returns = data.groupby("Year")["Close"].agg(["first", "last"])
    yearly_returns["Return"] = (
        yearly_returns["last"] / yearly_returns["first"] - 1
    ) * 100

    print(f"\nYearly Returns:")
    for year, return_pct in yearly_returns["Return"].items():
        print(f"  {year}: {return_pct:+.1f}%")

    positive_years = (yearly_returns["Return"] > 0).sum()
    total_years = len(yearly_returns)

    print(
        f"  Positive years: {positive_years}/{total_years} ({positive_years/total_years:.1%})"
    )

    return {
        "bull_periods": bull_count / total_periods,
        "bear_periods": bear_count / total_periods,
        "sideways_periods": sideways_count / total_periods,
        "avg_volatility": data["Volatility"].mean(),
        "max_drawdown": drawdown.min(),
        "positive_years_pct": positive_years / total_years,
    }


def test_strategy_on_subperiods(
    data: pd.DataFrame, strategy_class, params: Dict, strategy_name: str
):
    """Test a strategy on different time periods to find where it works best."""

    print(f"\n" + "=" * 60)
    print(f"SUBPERIOD ANALYSIS: {strategy_name}")
    print(f"=" * 60)

    # Define test periods
    periods = [
        ("2004-2007", "2004-08-19", "2007-12-31"),  # Pre-crisis bull market
        ("2008-2009", "2008-01-01", "2009-12-31"),  # Financial crisis
        ("2010-2015", "2010-01-01", "2015-12-31"),  # Recovery period
        ("2016-2020", "2016-01-01", "2020-04-01"),  # Late bull market
    ]

    results = []

    for period_name, start_date, end_date in periods:
        try:
            # Filter data for period
            period_data = data[(data.index >= start_date) & (data.index <= end_date)]

            if len(period_data) < 100:  # Need sufficient data
                continue

            # Test strategy
            strategy = strategy_class(**params)
            signals = strategy.generate_signals(period_data)

            backtester = Backtester(initial_capital=10000)
            performance = backtester.run_backtest(period_data, signals)

            # Calculate metrics
            summary = performance.summary()
            detailed_metrics = evaluate_performance(performance, period_data)

            result = {
                "period": period_name,
                "start_date": start_date,
                "end_date": end_date,
                "days": len(period_data),
                "strategy_return": summary["total_return_pct"],
                "benchmark_return": detailed_metrics["benchmark_return_pct"],
                "alpha": detailed_metrics["alpha_pct"],
                "beats_benchmark": detailed_metrics["outperformed_benchmark"],
                "sharpe_ratio": summary["sharpe_ratio"],
                "max_drawdown": summary["max_drawdown_pct"],
                "num_trades": summary["num_trades"],
                "win_rate": summary["win_rate"],
            }

            results.append(result)

            print(
                f"{period_name}: Strategy {result['strategy_return']:.1f}% vs Benchmark {result['benchmark_return']:.1f}% "
                f"(Alpha: {result['alpha']:+.1f}%, Beats: {'YES' if result['beats_benchmark'] else 'NO'})"
            )

        except Exception as e:
            print(f"{period_name}: ERROR - {e}")

    # Summary
    if results:
        successful_periods = [r for r in results if r["beats_benchmark"]]
        print(
            f"\nSummary: Strategy beat benchmark in {len(successful_periods)}/{len(results)} periods"
        )

        if successful_periods:
            print("Successful periods:")
            for result in successful_periods:
                print(f"  {result['period']}: +{result['alpha']:.1f}% alpha")

    return results


def main():
    """Main analysis function."""
    print("=" * 80)
    print("STRATEGY VISUALIZATION AND ANALYSIS")
    print("=" * 80)

    # Load data
    data = load_csv_data("data/GOOGL.csv")
    print(
        f"Loaded {len(data)} data points from {data.index.min()} to {data.index.max()}"
    )

    # Market analysis
    market_stats = analyze_market_periods(data)

    # Test a few key strategies for detailed analysis
    strategies_to_analyze = [
        {
            "class": MovingAverageCrossStrategy,
            "params": {"short_window": 8, "long_window": 80},
            "name": "MA_Cross_Optimized",
        },
        {
            "class": TrendFollowingStrategy,
            "params": {
                "trend_window": 30,
                "confirmation_window": 10,
                "min_trend_strength": 30,
                "exit_rsi_threshold": 75,
            },
            "name": "TrendFollowing_Best",
        },
    ]

    strategies_data = {}

    print(f"\n" + "=" * 80)
    print("DETAILED STRATEGY ANALYSIS")
    print("=" * 80)

    for strategy_config in strategies_to_analyze:
        try:
            # Test strategy
            strategy = strategy_config["class"](**strategy_config["params"])
            signals = strategy.generate_signals(data)

            backtester = Backtester(initial_capital=10000)
            performance = backtester.run_backtest(data, signals)

            summary = performance.summary()

            strategies_data[strategy_config["name"]] = {
                "portfolio_history": performance.portfolio_history,
                "total_return_pct": summary["total_return_pct"],
                "num_trades": summary["num_trades"],
                "sharpe_ratio": summary["sharpe_ratio"],
                "max_drawdown_pct": summary["max_drawdown_pct"],
            }

            # Subperiod analysis
            test_strategy_on_subperiods(
                data,
                strategy_config["class"],
                strategy_config["params"],
                strategy_config["name"],
            )

        except Exception as e:
            print(f"Error analyzing {strategy_config['name']}: {e}")

    # Create visualization
    if strategies_data:
        chart_path = create_performance_comparison_chart(data, strategies_data)

    # Final insights
    print(f"\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)

    print(f"1. Market Characteristics (GOOGL 2004-2020):")
    print(f"   - Bull market dominance: {market_stats['bull_periods']:.1%} of time")
    print(f"   - Low bear market exposure: {market_stats['bear_periods']:.1%} of time")
    print(f"   - High positive year ratio: {market_stats['positive_years_pct']:.1%}")
    print(f"   - Maximum drawdown: {market_stats['max_drawdown']:.1%}")

    print(f"\n2. Why Strategies Underperform:")
    print(f"   - GOOGL 2004-2020 was an exceptional bull market (+2094%)")
    print(f"   - Most time spent trending upward - hard to beat buy-and-hold")
    print(f"   - Trading costs reduce returns in trending markets")
    print(f"   - Timing the market is extremely difficult in strong trends")

    print(f"\n3. Recommendations:")
    print(f"   - Test strategies on different assets (more volatile stocks)")
    print(f"   - Test on different time periods (including bear markets)")
    print(f"   - Consider index ETFs where timing might be more effective")
    print(f"   - Focus on risk management rather than absolute returns")
    print(f"   - Use strategies that add value during market stress periods")


if __name__ == "__main__":
    main()
