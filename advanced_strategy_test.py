#!/usr/bin/env python3
"""
Advanced Strategy Testing and Comparison Script.

This script tests the new advanced moving average strategies against
the benchmark and compares their performance.
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.csv_loader import load_csv_data
from simulation.backtester import Backtester
from evaluation.metrics import evaluate_performance
from evaluation.report import create_hold_strategy_benchmark
from optimization.multi_objective_optimizer import (
    MultiObjectiveOptimizer,
    OptimizationObjective,
    OptimizationConstraint,
)

# Import all strategies
from strategies.moving_average_cross import MovingAverageCrossStrategy
from strategies.triple_ma_strategy import TripleMovingAverageStrategy
from strategies.adaptive_ma_strategy import AdaptiveMovingAverageStrategy
from strategies.advanced_ma_strategy import AdvancedMAStrategy


def test_strategy_performance(
    strategy_class,
    params: Dict,
    data: pd.DataFrame,
    backtester: Backtester,
    strategy_name: str,
) -> Dict[str, Any]:
    """Test a single strategy configuration and return results."""
    try:
        print(f"\nTesting {strategy_name} with params: {params}")

        # Create strategy and generate signals
        strategy = strategy_class(**params)
        signals = strategy.generate_signals(data)

        # Run backtest
        performance_report = backtester.run_backtest(data, signals)

        # Get performance metrics
        summary = performance_report.summary()
        detailed_metrics = evaluate_performance(performance_report, data)

        # Combine results
        results = {
            "strategy_name": strategy_name,
            "params": params,
            "total_return_pct": summary.get("total_return_pct", 0),
            "sharpe_ratio": summary.get("sharpe_ratio", 0),
            "max_drawdown_pct": summary.get("max_drawdown_pct", 0),
            "win_rate": summary.get("win_rate", 0),
            "num_trades": summary.get("num_trades", 0),
            "profit_factor": summary.get("profit_factor", 0),
            "calmar_ratio": summary.get("calmar_ratio", 0),
            "sortino_ratio": summary.get("sortino_ratio", 0),
            "final_value": summary.get("final_value", 10000),
            "alpha_pct": detailed_metrics.get("alpha_pct", 0),
            "benchmark_return_pct": detailed_metrics.get("benchmark_return_pct", 0),
            "outperformed_benchmark": detailed_metrics.get(
                "outperformed_benchmark", False
            ),
        }

        print(f"  Return: {results['total_return_pct']:.2f}%")
        print(f"  Sharpe: {results['sharpe_ratio']:.2f}")
        print(f"  Max DD: {results['max_drawdown_pct']:.2f}%")
        print(f"  Win Rate: {results['win_rate']:.1%}")
        print(f"  Trades: {results['num_trades']}")
        print(f"  Alpha: {results['alpha_pct']:.2f}%")

        return results

    except Exception as e:
        print(f"  ERROR: {e}")
        return {
            "strategy_name": strategy_name,
            "params": params,
            "error": str(e),
            "total_return_pct": 0,
            "outperformed_benchmark": False,
        }


def run_multi_objective_optimization(
    strategy_class, param_grid: Dict, data: pd.DataFrame, strategy_name: str
) -> Dict[str, Any]:
    """Run multi-objective optimization for a strategy."""
    print(f"\n{'='*60}")
    print(f"MULTI-OBJECTIVE OPTIMIZATION: {strategy_name}")
    print(f"{'='*60}")

    # Define objectives - trying to beat the benchmark requirements
    objectives = [
        OptimizationObjective("total_return", maximize=True, weight=0.4),
        OptimizationObjective("sharpe_ratio", maximize=True, weight=0.2),
        OptimizationObjective("win_rate", maximize=True, weight=0.2),
        OptimizationObjective(
            "num_trades", maximize=False, weight=0.1
        ),  # Minimize trades
        OptimizationObjective(
            "max_drawdown", maximize=False, weight=0.1
        ),  # Minimize drawdown
    ]

    # Define constraints from the issue requirements
    constraints = [
        OptimizationConstraint("total_return_pct", ">", 2100),  # Beat benchmark
        OptimizationConstraint("win_rate", ">", 0.5),  # >50% win rate
        OptimizationConstraint("num_trades", "<", 40),  # <40 trades
        OptimizationConstraint("sharpe_ratio", ">", 5.0),  # Maintain Sharpe >5.0
        OptimizationConstraint("max_drawdown_pct", ">", -25),  # Max drawdown <25%
    ]

    try:
        # Create optimizer
        backtester = Backtester(initial_capital=10000)
        optimizer = MultiObjectiveOptimizer(
            strategy_class=strategy_class,
            backtester=backtester,
            param_grid=param_grid,
            objectives=objectives,
            constraints=constraints,
        )

        # Run optimization
        best_params, optimization_summary = optimizer.optimize(data, data)

        print(f"Optimization completed!")
        print(f"Valid combinations: {optimization_summary['valid_combinations']}")
        print(f"Best parameters: {best_params}")
        print(f"Best score: {optimization_summary['best_score']:.4f}")

        # Test the best parameters
        if best_params:
            best_results = test_strategy_performance(
                strategy_class,
                best_params,
                data,
                backtester,
                f"{strategy_name}_Optimized",
            )

            return {
                "optimization_summary": optimization_summary,
                "best_params": best_params,
                "best_results": best_results,
                "optimizer": optimizer,
            }
        else:
            print("No valid parameters found that satisfy all constraints!")
            return None

    except Exception as e:
        print(f"Optimization failed: {e}")
        return None


def main():
    """Main function to test all advanced strategies."""
    print("=" * 80)
    print("ADVANCED STRATEGY TESTING - BEATING THE BENCHMARK")
    print("=" * 80)

    # Load data
    print("\n1. Loading data...")
    df = load_csv_data("data/GOOGL.csv")
    print(f"Loaded {len(df)} data points from {df.index.min()} to {df.index.max()}")

    # Calculate benchmark
    initial_capital = 10000
    benchmark_return = ((df["Close"].iloc[-1] / df["Close"].iloc[0]) - 1) * 100
    print(f"Buy-and-Hold benchmark return: {benchmark_return:.2f}%")

    backtester = Backtester(initial_capital=initial_capital)

    # Strategy configurations to test
    strategy_configs = [
        # Original Moving Average (baseline)
        {
            "class": MovingAverageCrossStrategy,
            "name": "Simple_MA_Cross",
            "params": {"short_window": 10, "long_window": 50},
            "param_grid": {
                "short_window": [5, 10, 15, 20],
                "long_window": [30, 50, 70, 100],
            },
        },
        # Triple Moving Average
        {
            "class": TripleMovingAverageStrategy,
            "name": "Triple_MA_SMA",
            "params": {
                "short_window": 5,
                "medium_window": 15,
                "long_window": 30,
                "ma_type": "sma",
            },
            "param_grid": {
                "short_window": [5, 8, 10],
                "medium_window": [15, 20, 25],
                "long_window": [30, 40, 50],
                "ma_type": ["sma", "ema"],
            },
        },
        # Adaptive Moving Average
        {
            "class": AdaptiveMovingAverageStrategy,
            "name": "Adaptive_MA",
            "params": {
                "fast_period": 2,
                "slow_period": 30,
                "use_rsi_filter": True,
                "use_trend_filter": True,
            },
            "param_grid": {
                "fast_period": [2, 3, 4],
                "slow_period": [20, 30, 40],
                "rsi_oversold": [25, 30, 35],
                "rsi_overbought": [65, 70, 75],
                "adx_threshold": [20, 25, 30],
            },
        },
        # Advanced MA with all filters
        {
            "class": AdvancedMAStrategy,
            "name": "Advanced_MA_Full",
            "params": {
                "short_window": 8,
                "long_window": 25,
                "ma_type": "ema",
                "use_volume_filter": True,
                "use_rsi_filter": True,
                "use_macd_filter": True,
                "use_bollinger_filter": True,
                "use_trend_filter": True,
                "adx_threshold": 25,
            },
            "param_grid": {
                "short_window": [5, 8, 10, 12],
                "long_window": [20, 25, 30, 35],
                "ma_type": ["ema"],
                "rsi_oversold": [25, 30],
                "rsi_overbought": [70, 75],
                "adx_threshold": [20, 25],
            },
        },
    ]

    # Store all results
    all_results = []
    optimization_results = []

    # Test each strategy configuration
    for config in strategy_configs:
        print(f"\n{'='*60}")
        print(f"TESTING STRATEGY: {config['name']}")
        print(f"{'='*60}")

        # Test default parameters
        default_results = test_strategy_performance(
            config["class"], config["params"], df, backtester, config["name"]
        )
        all_results.append(default_results)

        # Run multi-objective optimization
        opt_results = run_multi_objective_optimization(
            config["class"], config["param_grid"], df, config["name"]
        )

        if opt_results:
            optimization_results.append(opt_results)
            all_results.append(opt_results["best_results"])

    # Summary of results
    print(f"\n{'='*80}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"Benchmark (Buy-and-Hold): {benchmark_return:.2f}%")
    print(f"Target: Beat {benchmark_return:.2f}% (>2100%)")

    # Sort results by total return
    valid_results = [r for r in all_results if "error" not in r]
    valid_results.sort(key=lambda x: x["total_return_pct"], reverse=True)

    print(f"\nStrategy Performance Ranking:")
    print("-" * 100)
    print(
        f"{'Strategy':<25} {'Return %':<10} {'Alpha %':<10} {'Sharpe':<8} {'Win Rate':<10} {'Trades':<8} {'Beat Benchmark':<15}"
    )
    print("-" * 100)

    for result in valid_results:
        print(
            f"{result['strategy_name']:<25} "
            f"{result['total_return_pct']:<10.2f} "
            f"{result.get('alpha_pct', 0):<10.2f} "
            f"{result['sharpe_ratio']:<8.2f} "
            f"{result['win_rate']:<10.1%} "
            f"{result['num_trades']:<8.0f} "
            f"{'YES' if result.get('outperformed_benchmark', False) else 'NO':<15}"
        )

    # Check if we beat the benchmark
    successful_strategies = [
        r for r in valid_results if r.get("outperformed_benchmark", False)
    ]

    print(f"\n{'='*80}")
    print("BENCHMARK ANALYSIS")
    print(f"{'='*80}")
    print(f"Strategies that beat benchmark: {len(successful_strategies)}")

    if successful_strategies:
        best_strategy = successful_strategies[0]
        print(f"\nBest performing strategy: {best_strategy['strategy_name']}")
        print(f"  Return: {best_strategy['total_return_pct']:.2f}%")
        print(f"  Alpha: {best_strategy.get('alpha_pct', 0):.2f}%")
        print(f"  Sharpe Ratio: {best_strategy['sharpe_ratio']:.2f}")
        print(f"  Win Rate: {best_strategy['win_rate']:.1%}")
        print(f"  Max Drawdown: {best_strategy['max_drawdown_pct']:.2f}%")
        print(f"  Number of Trades: {best_strategy['num_trades']}")

        # Check if it meets all criteria
        criteria_met = (
            best_strategy["total_return_pct"] > 2100
            and best_strategy["win_rate"] > 0.5
            and best_strategy["num_trades"] < 40
            and best_strategy["sharpe_ratio"] > 5.0
            and best_strategy["max_drawdown_pct"] > -25
        )

        print(f"\nMeets all success criteria: {'YES' if criteria_met else 'NO'}")

        if criteria_met:
            print("🎉 SUCCESS! We have beaten the benchmark with all criteria met!")
        else:
            print("⚠️  Strategy beats benchmark but doesn't meet all criteria.")
    else:
        print("❌ No strategy beat the benchmark. Further optimization needed.")

    print(f"\n{'='*80}")
    print("TESTING COMPLETED")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
