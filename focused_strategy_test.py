#!/usr/bin/env python3
"""
Focused Strategy Testing - Strategies designed to beat Buy-and-Hold.

This script tests the strategies most likely to beat the benchmark.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the project root to the path
project_root = "/home/runner/work/Agentic-Trader/Agentic-Trader"
sys.path.insert(0, project_root)

from data.csv_loader import load_csv_data
from simulation.backtester import Backtester
from evaluation.metrics import evaluate_performance
from strategies.trend_following_strategy import TrendFollowingStrategy
from strategies.buy_hold_plus_strategy import BuyHoldPlusStrategy
from strategies.moving_average_cross import MovingAverageCrossStrategy


def test_strategy(strategy_class, params, data, strategy_name):
    """Test a strategy and return results."""
    print(f"\n{'='*50}")
    print(f"Testing {strategy_name}")
    print(f"Parameters: {params}")
    print(f"{'='*50}")
    
    try:
        # Create strategy and generate signals
        strategy = strategy_class(**params)
        signals = strategy.generate_signals(data)
        
        # Run backtest
        backtester = Backtester(initial_capital=10000)
        performance_report = backtester.run_backtest(data, signals)
        
        # Get performance metrics
        summary = performance_report.summary()
        detailed_metrics = evaluate_performance(performance_report, data)
        
        # Print results
        print(f"Final Value: ${summary['final_value']:,.2f}")
        print(f"Total Return: {summary['total_return_pct']:.2f}%")
        print(f"Benchmark Return: {detailed_metrics['benchmark_return_pct']:.2f}%")
        print(f"Alpha: {detailed_metrics['alpha_pct']:.2f}%")
        print(f"Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {summary['max_drawdown_pct']:.2f}%")
        print(f"Win Rate: {summary['win_rate']:.1%}")
        print(f"Number of Trades: {summary['num_trades']}")
        print(f"Profit Factor: {summary['profit_factor']:.2f}")
        
        beats_benchmark = detailed_metrics['outperformed_benchmark']
        print(f"Beats Benchmark: {'YES' if beats_benchmark else 'NO'}")
        
        return {
            'strategy_name': strategy_name,
            'params': params,
            'total_return_pct': summary['total_return_pct'],
            'benchmark_return_pct': detailed_metrics['benchmark_return_pct'],
            'alpha_pct': detailed_metrics['alpha_pct'],
            'sharpe_ratio': summary['sharpe_ratio'],
            'max_drawdown_pct': summary['max_drawdown_pct'],
            'win_rate': summary['win_rate'],
            'num_trades': summary['num_trades'],
            'profit_factor': summary['profit_factor'],
            'beats_benchmark': beats_benchmark,
        }
        
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def main():
    """Test the most promising strategies."""
    print("="*80)
    print("FOCUSED STRATEGY TESTING - BEATING BUY-AND-HOLD")
    print("="*80)
    
    # Load data
    df = load_csv_data("data/GOOGL.csv")
    print(f"Loaded {len(df)} data points from {df.index.min()} to {df.index.max()}")
    
    # Calculate benchmark
    benchmark_return = ((df["Close"].iloc[-1] / df["Close"].iloc[0]) - 1) * 100
    print(f"Buy-and-Hold Benchmark: {benchmark_return:.2f}%")
    
    results = []
    
    # Test 1: Buy-and-Hold Plus Strategy (most conservative)
    print("\n" + "="*80)
    print("TESTING BUY-AND-HOLD PLUS STRATEGIES")
    print("="*80)
    
    buy_hold_configs = [
        # Very conservative - only exit during extreme stress
        {
            'stress_rsi_threshold': 20,
            'reentry_rsi_threshold': 35,
            'drawdown_threshold': 0.20,
            'trend_window': 50
        },
        # Moderate - exit during significant stress
        {
            'stress_rsi_threshold': 25,
            'reentry_rsi_threshold': 40,
            'drawdown_threshold': 0.15,
            'trend_window': 50
        },
        # More aggressive - exit earlier
        {
            'stress_rsi_threshold': 30,
            'reentry_rsi_threshold': 45,
            'drawdown_threshold': 0.12,
            'trend_window': 30
        },
    ]
    
    for i, params in enumerate(buy_hold_configs):
        result = test_strategy(
            BuyHoldPlusStrategy, 
            params, 
            df, 
            f"BuyHoldPlus_v{i+1}"
        )
        if result:
            results.append(result)
    
    # Test 2: Trend Following Strategies
    print("\n" + "="*80)
    print("TESTING TREND FOLLOWING STRATEGIES")
    print("="*80)
    
    trend_configs = [
        # Long-term trend following
        {
            'trend_window': 100,
            'confirmation_window': 20,
            'min_trend_strength': 20,
            'exit_rsi_threshold': 85
        },
        # Medium-term trend following
        {
            'trend_window': 50,
            'confirmation_window': 15,
            'min_trend_strength': 25,
            'exit_rsi_threshold': 80
        },
        # More responsive trend following
        {
            'trend_window': 30,
            'confirmation_window': 10,
            'min_trend_strength': 30,
            'exit_rsi_threshold': 75
        },
    ]
    
    for i, params in enumerate(trend_configs):
        result = test_strategy(
            TrendFollowingStrategy, 
            params, 
            df, 
            f"TrendFollowing_v{i+1}"
        )
        if result:
            results.append(result)
    
    # Test 3: Optimized Moving Average (for comparison)
    print("\n" + "="*80)
    print("TESTING OPTIMIZED MOVING AVERAGE")
    print("="*80)
    
    # Test several MA configurations to find the best
    ma_configs = [
        {'short_window': 5, 'long_window': 100},
        {'short_window': 8, 'long_window': 80},
        {'short_window': 10, 'long_window': 200},
    ]
    
    for i, params in enumerate(ma_configs):
        result = test_strategy(
            MovingAverageCrossStrategy, 
            params, 
            df, 
            f"MA_Optimized_v{i+1}"
        )
        if result:
            results.append(result)
    
    # Summary of all results
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    
    if results:
        # Sort by total return
        results.sort(key=lambda x: x['total_return_pct'], reverse=True)
        
        print(f"Benchmark: {benchmark_return:.2f}%")
        print(f"Target: Beat {benchmark_return:.2f}%\n")
        
        print(f"{'Strategy':<25} {'Return %':<10} {'Alpha %':<10} {'Trades':<8} {'Win Rate':<10} {'Beats BM':<10}")
        print("-" * 85)
        
        best_strategy = None
        
        for result in results:
            beats_str = "YES" if result['beats_benchmark'] else "NO"
            print(f"{result['strategy_name']:<25} "
                  f"{result['total_return_pct']:<10.2f} "
                  f"{result['alpha_pct']:<10.2f} "
                  f"{result['num_trades']:<8.0f} "
                  f"{result['win_rate']:<10.1%} "
                  f"{beats_str:<10}")
            
            if result['beats_benchmark'] and best_strategy is None:
                best_strategy = result
        
        # Final analysis
        print("\n" + "="*80)
        print("BENCHMARK BEATING ANALYSIS")
        print("="*80)
        
        successful_strategies = [r for r in results if r['beats_benchmark']]
        
        if successful_strategies:
            print(f"🎉 SUCCESS! Found {len(successful_strategies)} strategies that beat the benchmark!")
            
            best = successful_strategies[0]  # Already sorted by return
            print(f"\nBest Strategy: {best['strategy_name']}")
            print(f"  Parameters: {best['params']}")
            print(f"  Return: {best['total_return_pct']:.2f}% vs Benchmark {best['benchmark_return_pct']:.2f}%")
            print(f"  Alpha: {best['alpha_pct']:.2f}%")
            print(f"  Sharpe Ratio: {best['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown: {best['max_drawdown_pct']:.2f}%")
            print(f"  Number of Trades: {best['num_trades']}")
            print(f"  Win Rate: {best['win_rate']:.1%}")
            
            # Check success criteria
            criteria_checks = {
                "Beats benchmark": best['total_return_pct'] > best['benchmark_return_pct'],
                "Return > 2100%": best['total_return_pct'] > 2100,
                "Win rate > 50%": best['win_rate'] > 0.5,
                "Trades < 40": best['num_trades'] < 40,
                "Sharpe > 5.0": best['sharpe_ratio'] > 5.0,
                "Max DD < 25%": best['max_drawdown_pct'] > -25,
            }
            
            print(f"\nSuccess Criteria Check:")
            for criterion, passed in criteria_checks.items():
                status = "✓" if passed else "✗"
                print(f"  {status} {criterion}")
            
            all_criteria_met = all(criteria_checks.values())
            if all_criteria_met:
                print(f"\n🏆 COMPLETE SUCCESS! All criteria met!")
            else:
                print(f"\n⚠️  Partial success - beats benchmark but doesn't meet all criteria")
                
        else:
            print("❌ No strategy beat the benchmark in this test.")
            print("The GOOGL 2004-2020 period was an exceptionally strong bull market.")
            print("Consider testing on different time periods or assets.")
    
    else:
        print("No valid results obtained.")


if __name__ == "__main__":
    main()