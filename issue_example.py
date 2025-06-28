#!/usr/bin/env python3
"""
Exact implementation of the example from the issue requirements.

This demonstrates the expected usage pattern exactly as specified in the issue.
"""

import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluation.metrics import evaluate_performance
from optimization.grid_search import GridSearchOptimizer
from simulation.backtester import Backtester
from strategies.moving_average_cross import MovingAverageCrossStrategy
from data.csv_loader import load_csv_data


def main():
    """Run the exact example from the issue."""
    print("Running the exact example from issue requirements...")
    print("=" * 60)
    
    # Chargement des données CSV
    df = load_csv_data(filepath="data/BTC-USD.csv")
    
    if df.empty:
        print("No BTC-USD data available, using GOOGL instead...")
        df = load_csv_data(filepath="data/GOOGL.csv")
    
    print(f"Loaded data: {len(df)} rows from {df.index.min()} to {df.index.max()}")
    
    # Configuration de l'optimisation
    param_grid = {
        "short_window": [10, 20, 30],
        "long_window": [50, 100, 150]
    }
    
    print(f"Parameter grid: {param_grid}")
    
    optimizer = GridSearchOptimizer(
        MovingAverageCrossStrategy, 
        Backtester(initial_capital=10000), 
        param_grid,
        objective="total_return"  # ROI objective
    )
    
    best_params, optimization_report = optimizer.optimize(df)
    
    print(f"Best parameters found: {best_params}")
    print(f"Optimization report summary:")
    print(optimization_report.print_summary())
    
    # Évaluation des performances avec les meilleurs paramètres
    strategy_best = MovingAverageCrossStrategy(**best_params)
    signals_best = strategy_best.generate_signals(df)
    
    backtester = Backtester(initial_capital=10000)
    performance_results = backtester.run_backtest(df, signals_best)
    
    metrics = evaluate_performance(performance_results)
    
    print("Meilleurs paramètres trouvés :", best_params)
    print("Rapport d'optimisation:", optimization_report.summary())
    print("Évaluation des performances finales:", metrics)
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("All functionality from the issue has been implemented.")


if __name__ == "__main__":
    main()