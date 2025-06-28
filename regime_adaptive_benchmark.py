#!/usr/bin/env python3
"""
Simplified Regime Adaptive Strategy Benchmark Script.

This script provides a clear and focused benchmark for the regime adaptive
strategy against standard alternatives with realistic market data.

Features:
- Uses real market data (when available) or realistic synthetic data
- Comprehensive performance comparison
- Parameter optimization for Regime Adaptive strategy
- Clear visualization of results
- Actionable insights and recommendations

Usage:
    # Basic benchmark
    python regime_adaptive_benchmark.py [--symbol GOOGL] [--years 5] [--quick]
    
    # With parameter optimization
    python regime_adaptive_benchmark.py --optimize [--full-optimization]
    
    # Example from user's output
    python regime_adaptive_benchmark.py --synthetic --years 3 --no-viz --optimize
"""

import argparse
import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import itertools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Strategy imports
from strategies.regime_adaptive_strategy import RegimeAdaptiveStrategy
from strategies.trend_following_strategy import TrendFollowingStrategy
from strategies.buy_hold_plus_strategy import BuyHoldPlusStrategy
from strategies.adaptive_ma_strategy import AdaptiveMovingAverageStrategy

# Simulation imports
from simulation.backtester import Backtester
from evaluation.metrics import evaluate_performance

# Optimization imports
from optimization.grid_search import GridSearchOptimizer

# Data import (if available)
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class RegimeAdaptiveBenchmark:
    """
    Simplified benchmark for regime adaptive trading strategy.
    
    This class provides a focused comparison of the regime adaptive strategy
    against common alternatives with clear performance metrics and insights.
    Includes optimization capabilities to improve strategy performance.
    """
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.backtester = Backtester(initial_capital=initial_capital)
        self.results = {}
        self.optimization_results = {}
        
    def optimize_regime_adaptive_strategy(
        self, 
        data: pd.DataFrame,
        quick_optimization: bool = True
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Optimize the Regime Adaptive Strategy parameters.
        
        Parameters
        ----------
        data : pd.DataFrame
            Market data for optimization
        quick_optimization : bool, default True
            If True, use a smaller parameter grid for faster optimization
            
        Returns
        -------
        Tuple[Dict[str, Any], Dict[str, Any]]
            Best parameters and optimization report
        """
        logger.info("🔧 Starting Regime Adaptive Strategy optimization...")
        
        # Define parameter grid
        if quick_optimization:
            param_grid = {
                'regime_memory': [3, 5, 7],
                'confidence_threshold': [0.5, 0.6, 0.7]
            }
        else:
            param_grid = {
                'regime_memory': [2, 3, 5, 7, 10],
                'confidence_threshold': [0.4, 0.5, 0.6, 0.7, 0.8],
            }
        
        # Custom objective function that balances returns and risk
        def regime_objective(performance_report):
            """Multi-objective scoring for regime adaptive strategies."""
            if performance_report.portfolio_history.empty:
                return -999
            
            metrics = performance_report.summary()
            
            # Extract key metrics
            sharpe = metrics.get('sharpe_ratio', 0)
            max_dd = abs(metrics.get('max_drawdown_pct', 1))
            total_return = metrics.get('total_return', 0)
            win_rate = metrics.get('win_rate', 0)
            
            # Penalize excessive drawdown and reward consistent performance
            if max_dd > 0.4:  # More than 40% drawdown is heavily penalized
                drawdown_penalty = -2.0
            elif max_dd > 0.3:  # 30-40% drawdown gets moderate penalty
                drawdown_penalty = -1.0
            elif max_dd > 0.2:  # 20-30% drawdown gets small penalty
                drawdown_penalty = -0.3
            else:
                drawdown_penalty = 0.0
            
            # Multi-objective score (higher is better)
            score = (
                sharpe * 0.4 +                           # Risk-adjusted return (40%)
                min(total_return, 1.0) * 0.3 +           # Capped total return (30%)
                win_rate * 0.2 +                         # Win rate (20%)
                (1 - min(max_dd, 1.0)) * 0.1 +          # Drawdown control (10%)
                drawdown_penalty                         # Drawdown penalty
            )
            
            return score
        
        # Initialize optimizer
        optimizer = GridSearchOptimizer(
            strategy_class=RegimeAdaptiveStrategy,
            backtester=self.backtester,
            param_grid=param_grid,
            objective=regime_objective
        )
        
        # Run optimization
        logger.info(f"Optimizing over {np.prod([len(v) for v in param_grid.values()])} parameter combinations...")
        
        try:
            best_params, optimization_report = optimizer.optimize(data)
            
            # Test the optimized strategy
            optimized_strategy = RegimeAdaptiveStrategy(**best_params)
            optimized_signals = optimized_strategy.generate_signals(data)
            optimized_performance = self.backtester.run_backtest(data, optimized_signals)
            
            optimization_summary = {
                'best_params': best_params,
                'best_score': getattr(optimization_report, 'best_score', None),
                'total_tested': len(getattr(optimization_report, 'results', [])),
                'optimized_performance': optimized_performance,
                'optimization_report': optimization_report
            }
            
            logger.info(f"✅ Optimization completed!")
            logger.info(f"Best parameters: {best_params}")
            
            return best_params, optimization_summary
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            # Return default parameters as fallback
            default_params = {'regime_memory': 5, 'confidence_threshold': 0.6}
            return default_params, {'error': str(e)}
        
    def get_market_data(
        self, 
        symbol: str = "GOOGL", 
        years: int = 5,
        use_synthetic: bool = False
    ) -> pd.DataFrame:
        """
        Get market data from Yahoo Finance or generate synthetic data.
        
        Parameters
        ----------
        symbol : str, default "GOOGL"
            Stock symbol to download
        years : int, default 5
            Number of years of historical data
        use_synthetic : bool, default False
            Force use of synthetic data instead of real data
            
        Returns
        -------
        pd.DataFrame
            Market data with OHLCV columns
        """
        if YFINANCE_AVAILABLE and not use_synthetic:
            try:
                logger.info(f"Downloading {years} years of {symbol} data...")
                end_date = datetime.now()
                start_date = end_date - timedelta(days=years*365)
                
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)
                
                if not data.empty:
                    logger.info(f"Downloaded {len(data)} days of {symbol} data")
                    return data
                else:
                    logger.warning(f"No data found for {symbol}, using synthetic data")
                    
            except Exception as e:
                logger.warning(f"Failed to download {symbol} data: {e}, using synthetic data")
        
        # Generate synthetic data
        logger.info(f"Generating {years} years of synthetic market data...")
        periods = years * 252  # Trading days
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='D')
        
        # Create realistic market data with different regimes
        prices = self._generate_realistic_synthetic_data(periods)
        
        data = pd.DataFrame({
            'Open': prices * 0.998,
            'High': prices * 1.02,
            'Low': prices * 0.98,
            'Close': prices,
            'Volume': np.random.lognormal(14, 0.5, periods)
        }, index=dates)
        
        # Ensure OHLC relationships are correct
        for i in range(len(data)):
            high = max(data.iloc[i]['Open'], data.iloc[i]['Close']) * (1 + abs(np.random.normal(0, 0.005)))
            low = min(data.iloc[i]['Open'], data.iloc[i]['Close']) * (1 - abs(np.random.normal(0, 0.005)))
            data.iloc[i, data.columns.get_loc('High')] = high
            data.iloc[i, data.columns.get_loc('Low')] = low
        
        logger.info(f"Generated {len(data)} days of synthetic market data")
        return data
    
    def _generate_realistic_synthetic_data(self, periods: int) -> np.ndarray:
        """Generate realistic synthetic price data with regime changes."""
        prices = [100.0]
        current_regime = 'moderate_bull'
        regime_length = 0
        
        # Regime characteristics
        regimes = {
            'strong_bull': {'drift': 0.0008, 'vol': 0.012, 'avg_length': 120},
            'moderate_bull': {'drift': 0.0003, 'vol': 0.015, 'avg_length': 180},
            'sideways': {'drift': 0.0001, 'vol': 0.018, 'avg_length': 150},
            'bear_market': {'drift': -0.0005, 'vol': 0.022, 'avg_length': 100},
            'crisis': {'drift': -0.002, 'vol': 0.035, 'avg_length': 40}
        }
        
        for i in range(1, periods):
            # Check for regime change
            regime_length += 1
            if regime_length > regimes[current_regime]['avg_length']:
                if np.random.random() < 0.3:  # 30% chance of regime change
                    current_regime = np.random.choice(list(regimes.keys()))
                    regime_length = 0
            
            # Generate price movement
            regime_params = regimes[current_regime]
            daily_return = np.random.normal(regime_params['drift'], regime_params['vol'])
            
            # Add some momentum and mean reversion
            if len(prices) > 10:
                momentum = np.mean([prices[j] / prices[j-1] - 1 for j in range(-5, 0)])
                daily_return += momentum * 0.1  # Momentum effect
                
                # Mean reversion
                ma_50 = np.mean(prices[-50:]) if len(prices) >= 50 else np.mean(prices)
                deviation = (prices[-1] - ma_50) / ma_50
                daily_return -= deviation * 0.05  # Mean reversion
            
            new_price = prices[-1] * (1 + daily_return)
            prices.append(max(new_price, 1.0))  # Prevent negative prices
        
        return np.array(prices)
    
    def run_strategy_comparison(
        self, 
        data: pd.DataFrame, 
        run_optimization: bool = False,
        quick_optimization: bool = True
    ) -> Dict[str, Any]:
        """
        Run comparison of regime adaptive strategy against benchmarks.
        
        Parameters
        ----------
        data : pd.DataFrame
            Market data for backtesting
        run_optimization : bool, default False
            Whether to run parameter optimization on Regime Adaptive strategy
        quick_optimization : bool, default True
            If True, use faster optimization with smaller parameter grid
            
        Returns
        -------
        Dict[str, Any]
            Strategy comparison results
        """
        logger.info("Running strategy comparison...")
        
        strategies = {
            'Regime Adaptive': RegimeAdaptiveStrategy(),
            'Trend Following': TrendFollowingStrategy(),
            'Buy Hold Plus': BuyHoldPlusStrategy(),
            'Adaptive MA': AdaptiveMovingAverageStrategy(),
            'Buy & Hold': None  # Special case
        }
        
        results = {}
        
        # Run optimization if requested
        optimized_params = None
        optimization_summary = None
        
        if run_optimization:
            logger.info("\n" + "="*50)
            logger.info("🚀 PARAMETER OPTIMIZATION")
            logger.info("="*50)
            
            optimized_params, optimization_summary = self.optimize_regime_adaptive_strategy(
                data, quick_optimization=quick_optimization
            )
            
            if 'error' not in optimization_summary:
                # Add optimized version to comparison
                strategies['Regime Adaptive (Optimized)'] = RegimeAdaptiveStrategy(**optimized_params)
                self.optimization_results = optimization_summary
        
        for name, strategy in strategies.items():
            logger.info(f"Testing {name}...")
            
            try:
                if name == 'Buy & Hold':
                    # Calculate buy and hold performance
                    initial_price = data['Close'].iloc[0]
                    final_price = data['Close'].iloc[-1]
                    total_return = (final_price - initial_price) / initial_price
                    
                    # Create portfolio history for buy and hold
                    portfolio_values = (data['Close'] / initial_price * self.initial_capital)
                    portfolio_history = pd.DataFrame({
                        'Date': data.index,
                        'Price': data['Close'],
                        'Signal': ['HOLD'] * len(data),
                        'Cash': [0.0] * len(data),
                        'Position': [self.initial_capital / initial_price] * len(data),
                        'Position_Value': portfolio_values,
                        'Total_Value': portfolio_values
                    })
                    portfolio_history.set_index('Date', inplace=True)
                    
                    from simulation.backtester import PerformanceReport
                    performance = PerformanceReport(
                        portfolio_history=portfolio_history,
                        transactions=[],
                        initial_capital=self.initial_capital,
                        final_value=portfolio_values.iloc[-1]
                    )
                else:
                    # Run strategy backtest
                    signals = strategy.generate_signals(data)
                    performance = self.backtester.run_backtest(data, signals)
                
                # Extract key metrics
                summary = performance.summary()
                
                results[name] = {
                    'performance': performance,
                    'total_return': summary.get('total_return', 0),
                    'total_return_pct': summary.get('total_return_pct', 0),
                    'sharpe_ratio': summary.get('sharpe_ratio', 0),
                    'max_drawdown': abs(summary.get('max_drawdown_pct', 0)),
                    'volatility': summary.get('volatility', 0),
                    'num_trades': summary.get('num_trades', 0),
                    'win_rate': summary.get('win_rate', 0)
                }
                
                logger.info(f"  Return: {summary.get('total_return', 0):.1%}")
                logger.info(f"  Sharpe: {summary.get('sharpe_ratio', 0):.2f}")
                logger.info(f"  Max DD: {abs(summary.get('max_drawdown_pct', 0)):.1f}%")
                
            except Exception as e:
                logger.error(f"Failed to test {name}: {e}")
                results[name] = {'error': str(e)}
        
        self.results = results
        
        # Add optimization summary to results if available
        if optimization_summary and 'error' not in optimization_summary:
            logger.info(f"\n📊 Optimization Summary:")
            logger.info(f"   Best parameters: {optimized_params}")
            logger.info(f"   Parameter combinations tested: {optimization_summary.get('total_tested', 'N/A')}")
            
        return results
    
    def generate_insights(self) -> List[str]:
        """Generate actionable insights from the benchmark results."""
        insights = []
        
        if not self.results:
            return ["No results available for analysis"]
        
        # Filter out failed strategies
        valid_results = {k: v for k, v in self.results.items() if 'error' not in v}
        
        if not valid_results:
            return ["All strategies failed to run"]
        
        # Find best performing strategies by different metrics
        best_return = max(valid_results.items(), key=lambda x: x[1]['total_return'])
        best_sharpe = max(valid_results.items(), key=lambda x: x[1]['sharpe_ratio'])
        best_drawdown = min(valid_results.items(), key=lambda x: x[1]['max_drawdown'])
        
        insights.append(f"🏆 Highest Return: {best_return[0]} ({best_return[1]['total_return']:.1%})")
        insights.append(f"📊 Best Risk-Adjusted: {best_sharpe[0]} (Sharpe: {best_sharpe[1]['sharpe_ratio']:.2f})")
        insights.append(f"🛡️ Lowest Drawdown: {best_drawdown[0]} ({best_drawdown[1]['max_drawdown']:.1f}%)")
        
        # Analyze regime adaptive strategy specifically
        ra_results = valid_results.get('Regime Adaptive')
        ra_opt_results = valid_results.get('Regime Adaptive (Optimized)')
        bh_results = valid_results.get('Buy & Hold', {})
        
        if ra_results and bh_results:
            insights.append("")
            insights.append("🎯 Regime Adaptive Analysis:")
            
            ra_return = ra_results['total_return']
            bh_return = bh_results['total_return']
            alpha = ra_return - bh_return
            
            if alpha > 0:
                insights.append(f"   ✅ Outperformed Buy & Hold by {alpha:.1%}")
            else:
                insights.append(f"   ❌ Underperformed Buy & Hold by {abs(alpha):.1%}")
            
            if ra_results['sharpe_ratio'] > bh_results['sharpe_ratio']:
                insights.append("   ✅ Better risk-adjusted returns")
            else:
                insights.append("   ⚠️ Lower risk-adjusted returns")
            
            if ra_results['max_drawdown'] < bh_results['max_drawdown']:
                insights.append("   ✅ Better drawdown control")
            else:
                insights.append("   ⚠️ Higher maximum drawdown")
        
        # Compare baseline vs optimized if both are available
        if ra_results and ra_opt_results:
            insights.append("")
            insights.append("🔧 Optimization Impact:")
            
            return_improvement = ra_opt_results['total_return'] - ra_results['total_return']
            sharpe_improvement = ra_opt_results['sharpe_ratio'] - ra_results['sharpe_ratio']
            dd_improvement = ra_results['max_drawdown'] - ra_opt_results['max_drawdown']
            
            if return_improvement > 0:
                insights.append(f"   📈 Return improved by {return_improvement:.1%}")
            else:
                insights.append(f"   📉 Return decreased by {abs(return_improvement):.1%}")
            
            if sharpe_improvement > 0:
                insights.append(f"   📊 Sharpe ratio improved by {sharpe_improvement:.2f}")
            else:
                insights.append(f"   📊 Sharpe ratio decreased by {abs(sharpe_improvement):.2f}")
            
            if dd_improvement > 0:
                insights.append(f"   🛡️ Maximum drawdown reduced by {dd_improvement:.1f}%")
            else:
                insights.append(f"   ⚠️ Maximum drawdown increased by {abs(dd_improvement):.1f}%")
            
            # Overall optimization verdict
            improvements = sum([
                1 if return_improvement > 0.01 else 0,  # 1% improvement threshold
                1 if sharpe_improvement > 0.1 else 0,   # 0.1 Sharpe improvement
                1 if dd_improvement > 1.0 else 0        # 1% drawdown reduction
            ])
            
            if improvements >= 2:
                insights.append("   ✅ Optimization was beneficial")
            elif improvements == 1:
                insights.append("   ⚠️ Optimization showed mixed results")
            else:
                insights.append("   ❌ Optimization did not improve performance")
        
        # Overall recommendations
        insights.append("")
        insights.append("💡 Recommendations:")
        
        if ra_opt_results and ra_results:
            if ra_opt_results['total_return'] > ra_results['total_return']:
                insights.append("   ✅ Use optimized parameters for better performance")
                if hasattr(self, 'optimization_results') and 'best_params' in self.optimization_results:
                    params = self.optimization_results['best_params']
                    insights.append(f"   🔧 Recommended parameters: {params}")
            else:
                insights.append("   ⚠️ Default parameters may be sufficient")
        elif ra_results:
            avg_sharpe = np.mean([r['sharpe_ratio'] for r in valid_results.values()])
            ra_sharpe = ra_results['sharpe_ratio']
            if ra_sharpe > avg_sharpe:
                insights.append("   ✅ Regime Adaptive shows above-average risk-adjusted performance")
            else:
                insights.append("   📈 Consider parameter optimization for Regime Adaptive strategy")
        
        return insights
    
    def create_visualization(self, data: pd.DataFrame) -> None:
        """Create performance visualization."""
        if not self.results:
            logger.warning("No results to visualize")
            return
        
        valid_results = {k: v for k, v in self.results.items() if 'error' not in v}
        
        if not valid_results:
            logger.warning("No valid results to visualize")
            return
        
        # Set up the plot
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Regime Adaptive Strategy Benchmark Results', fontsize=16, fontweight='bold')
        
        # 1. Portfolio value over time
        ax1 = axes[0, 0]
        for name, result in valid_results.items():
            portfolio_history = result['performance'].portfolio_history
            if not portfolio_history.empty:
                normalized_values = portfolio_history['Total_Value'] / self.initial_capital
                ax1.plot(portfolio_history.index, normalized_values, 
                        label=name, linewidth=2)
        
        ax1.set_title('Portfolio Value Over Time (Normalized)', fontweight='bold')
        ax1.set_ylabel('Portfolio Value / Initial Capital')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        
        # 2. Performance metrics comparison
        ax2 = axes[0, 1]
        strategies = list(valid_results.keys())
        returns = [valid_results[s]['total_return'] * 100 for s in strategies]
        sharpe_ratios = [valid_results[s]['sharpe_ratio'] for s in strategies]
        
        x = np.arange(len(strategies))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, returns, width, label='Total Return (%)', alpha=0.8)
        ax2_twin = ax2.twinx()
        bars2 = ax2_twin.bar(x + width/2, sharpe_ratios, width, label='Sharpe Ratio', 
                            alpha=0.8, color='orange')
        
        ax2.set_title('Return vs Risk-Adjusted Performance', fontweight='bold')
        ax2.set_xlabel('Strategy')
        ax2.set_ylabel('Total Return (%)', color='blue')
        ax2_twin.set_ylabel('Sharpe Ratio', color='orange')
        ax2.set_xticks(x)
        ax2.set_xticklabels(strategies, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        # 3. Risk metrics
        ax3 = axes[1, 0]
        max_drawdowns = [valid_results[s]['max_drawdown'] for s in strategies]
        volatilities = [valid_results[s]['volatility'] * 100 for s in strategies]
        
        scatter = ax3.scatter(volatilities, max_drawdowns, s=100, alpha=0.7)
        for i, strategy in enumerate(strategies):
            ax3.annotate(strategy, (volatilities[i], max_drawdowns[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax3.set_title('Risk Profile (Lower is Better)', fontweight='bold')
        ax3.set_xlabel('Volatility (%)')
        ax3.set_ylabel('Maximum Drawdown (%)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Strategy summary table
        ax4 = axes[1, 1]
        ax4.axis('tight')
        ax4.axis('off')
        
        # Create summary table
        table_data = []
        for strategy in strategies:
            result = valid_results[strategy]
            table_data.append([
                strategy,
                f"{result['total_return']:.1%}",
                f"{result['sharpe_ratio']:.2f}",
                f"{result['max_drawdown']:.1f}%",
                f"{result['num_trades']}"
            ])
        
        table = ax4.table(cellText=table_data,
                         colLabels=['Strategy', 'Return', 'Sharpe', 'Max DD', 'Trades'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Style the table
        for i in range(len(table_data) + 1):
            for j in range(len(table_data[0])):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#4472C4')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#F2F2F2' if i % 2 == 0 else 'white')
        
        ax4.set_title('Performance Summary', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('regime_adaptive_benchmark.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Benchmark visualization saved as 'regime_adaptive_benchmark.png'")


def main():
    """Main function to run the benchmark."""
    parser = argparse.ArgumentParser(description='Regime Adaptive Strategy Benchmark')
    parser.add_argument('--symbol', type=str, default='GOOGL', 
                       help='Stock symbol to analyze (default: GOOGL)')
    parser.add_argument('--years', type=int, default=5, 
                       help='Number of years of data (default: 5)')
    parser.add_argument('--capital', type=float, default=100000, 
                       help='Initial capital (default: 100000)')
    parser.add_argument('--synthetic', action='store_true', 
                       help='Use synthetic data instead of real market data')
    parser.add_argument('--no-viz', action='store_true', 
                       help='Skip visualization')
    parser.add_argument('--optimize', action='store_true',
                       help='Run parameter optimization on Regime Adaptive strategy')
    parser.add_argument('--full-optimization', action='store_true',
                       help='Run comprehensive optimization (slower but more thorough)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🎯 REGIME ADAPTIVE STRATEGY BENCHMARK")
    print("=" * 70)
    
    # Initialize benchmark
    benchmark = RegimeAdaptiveBenchmark(initial_capital=args.capital)
    
    try:
        # Get market data
        data = benchmark.get_market_data(
            symbol=args.symbol, 
            years=args.years, 
            use_synthetic=args.synthetic
        )
        
        # Run strategy comparison (with optimization if requested)
        results = benchmark.run_strategy_comparison(
            data, 
            run_optimization=args.optimize,
            quick_optimization=not args.full_optimization
        )
        
        # Generate insights
        insights = benchmark.generate_insights()
        
        # Display results
        print("\n" + "=" * 70)
        print("📊 BENCHMARK RESULTS")
        print("=" * 70)
        
        for insight in insights:
            print(insight)
        
        # Create visualization
        if not args.no_viz:
            print(f"\n📈 Creating performance visualization...")
            try:
                benchmark.create_visualization(data)
            except Exception as e:
                logger.warning(f"Visualization failed: {e}")
        
        print(f"\n✅ Benchmark completed successfully!")
        print(f"📊 Analyzed {len(data)} days of market data")
        print(f"💰 Initial Capital: ${args.capital:,.0f}")
        
        if args.optimize:
            print(f"🔧 Parameter optimization: {'Full' if args.full_optimization else 'Quick'}")
        
        if not args.synthetic and YFINANCE_AVAILABLE:
            print(f"📈 Symbol: {args.symbol}")
            print(f"📅 Period: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise


if __name__ == "__main__":
    main()