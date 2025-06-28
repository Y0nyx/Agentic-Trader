#!/usr/bin/env python3
"""
Example usage of the evaluation and optimization modules.

This script demonstrates the complete workflow of:
1. Loading data
2. Optimizing strategy parameters
3. Evaluating performance with comprehensive metrics
4. Comparing against buy-and-hold benchmark
"""

import sys
import os
import numpy as np

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluation.metrics import evaluate_performance
from evaluation.report import PerformanceReporter, create_hold_strategy_benchmark
from optimization.grid_search import GridSearchOptimizer, optimize_strategy
from simulation.backtester import Backtester
from strategies.moving_average_cross import MovingAverageCrossStrategy
from data.csv_loader import load_csv_data, get_available_csv_files


def main():
    """Run the complete example."""
    print("=" * 60)
    print("AGENTIC TRADER - EVALUATION & OPTIMIZATION DEMO")
    print("=" * 60)

    # 1. Load data
    print("\n1. Loading available data...")
    available_files = get_available_csv_files()
    print(f"Available CSV files: {available_files}")

    if not available_files:
        print(
            "No CSV files found! Please ensure data files are in the data/ directory."
        )
        return

    # Use the first available file (or GOOGL if available)
    data_file = "GOOGL.csv" if "GOOGL.csv" in available_files else available_files[0]
    print(f"Using data file: {data_file}")

    # Load the data
    filepath = f"data/{data_file}"
    df = load_csv_data(filepath)

    if df.empty:
        print(f"Failed to load data from {filepath}")
        return

    print(f"Loaded {len(df)} data points from {df.index.min()} to {df.index.max()}")

    # 2. Configure optimization parameters
    print("\n2. Setting up parameter optimization...")
    param_grid = {
        "short_window": np.arange(10, 31, 5).tolist(),
        "long_window": np.arange(50, 151, 10).tolist(),
    }

    print("Parameter grid:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")

    # 3. Run optimization
    print("\n3. Running grid search optimization...")
    print("Objective: ROI (Return on Investment)")

    best_params, optimization_report = optimize_strategy(
        strategy_class=MovingAverageCrossStrategy,
        data=df,
        param_space=param_grid,
        objective="total_return",  # Optimize for ROI
        initial_capital=10000,
    )

    print(f"\nOptimization completed!")
    print(f"Best parameters found: {best_params}")
    print(f"Best ROI: {optimization_report.summary()['best_score']:.4f}")

    # Print optimization summary
    print("\n" + optimization_report.print_summary())

    # 4. Evaluate performance with best parameters
    print("\n4. Evaluating strategy performance with best parameters...")

    # Create strategy with best parameters
    strategy_best = MovingAverageCrossStrategy(**best_params)
    signals_best = strategy_best.generate_signals(df)

    # Run backtest
    backtester = Backtester(initial_capital=10000)
    performance_results = backtester.run_backtest(df, signals_best)

    # 5. Create buy-and-hold benchmark
    print("\n5. Creating buy-and-hold benchmark for comparison...")
    benchmark_data = create_hold_strategy_benchmark(df, initial_capital=10000)

    # 6. Comprehensive evaluation
    print("\n6. Generating comprehensive performance evaluation...")

    # Create performance reporter
    reporter = PerformanceReporter(performance_results)

    # Generate comprehensive report with benchmark comparison
    comprehensive_report = reporter.generate_comprehensive_report(
        benchmark_data=df, include_charts=True  # Use original price data for benchmark
    )

    # Print formatted summary
    print(reporter.print_summary_report(df))

    # 7. Display additional insights
    print("\n7. Additional Performance Insights:")
    print("-" * 40)

    # Get detailed metrics
    detailed_metrics = evaluate_performance(performance_results, df)

    print(f"Total Net Profit: ${detailed_metrics.get('total_net_profit', 0):.2f}")
    print(
        f"Average Profit per Trade: ${detailed_metrics.get('avg_profit_per_trade', 0):.2f}"
    )
    print(f"Sortino Ratio: {detailed_metrics.get('sortino_ratio', 0):.3f}")
    print(f"Calmar Ratio: {detailed_metrics.get('calmar_ratio', 0):.3f}")
    print(f"Max Consecutive Wins: {detailed_metrics.get('max_consecutive_wins', 0)}")
    print(
        f"Max Consecutive Losses: {detailed_metrics.get('max_consecutive_losses', 0)}"
    )

    if "alpha" in detailed_metrics:
        print(f"Alpha vs Buy-and-Hold: {detailed_metrics['alpha_pct']:.2f}%")
        outperformed = (
            "Yes" if detailed_metrics.get("outperformed_benchmark", False) else "No"
        )
        print(f"Outperformed Benchmark: {outperformed}")

    # 8. Parameter sensitivity analysis
    print("\n8. Parameter Sensitivity Analysis:")
    print("-" * 40)

    # Create optimizer for sensitivity analysis
    optimizer = GridSearchOptimizer(
        strategy_class=MovingAverageCrossStrategy,
        backtester=Backtester(initial_capital=10000),
        param_grid=param_grid,
        objective="total_return",
    )

    # We already have results from the optimize_strategy call, but for demo purposes:
    optimizer.results = optimization_report.results
    sensitivity = optimizer.analyze_parameter_sensitivity()

    for param, analysis in sensitivity.items():
        correlation = analysis.get("correlation", 0)
        print(f"{param}: correlation with ROI = {correlation:.3f}")

    print("\n" + "=" * 60)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    print("\nKey Results Summary:")
    print(f"- Best Strategy Parameters: {best_params}")
    print(f"- Final Portfolio Value: ${performance_results.final_value:.2f}")
    print(
        f"- Total Return: {((performance_results.final_value / performance_results.initial_capital) - 1) * 100:.2f}%"
    )
    print(f"- Total Trades: {len(performance_results.transactions)}")

    if "benchmark_return_pct" in detailed_metrics:
        print(f"- Benchmark Return: {detailed_metrics['benchmark_return_pct']:.2f}%")
        print(f"- Alpha: {detailed_metrics['alpha_pct']:.2f}%")


if __name__ == "__main__":
    main()
