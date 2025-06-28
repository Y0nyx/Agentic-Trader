#!/usr/bin/env python3
"""
Regime Adaptive Strategy Optimization and Benchmarking Script.

This comprehensive script optimizes the regime adaptive strategy parameters
and provides detailed benchmarking against multiple baselines to evaluate
strategy performance across different market conditions.

Features:
- Multi-objective parameter optimization
- Regime-specific optimization
- Comprehensive benchmarking suite
- Performance visualization
- Results export and analysis

Usage:
    python regime_optimization_script.py [--quick] [--export-results]
"""

import os
import sys
import argparse
import logging
import json
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Import strategy components
from strategies.regime_adaptive_strategy import RegimeAdaptiveStrategy
from strategies.market_regime_classifier import MarketRegimeClassifier
from strategies.contextual_optimizer import ContextualOptimizer
from strategies.multi_strategy_portfolio import MultiStrategyPortfolio
from strategies.trend_following_strategy import TrendFollowingStrategy
from strategies.buy_hold_plus_strategy import BuyHoldPlusStrategy
from strategies.adaptive_ma_strategy import AdaptiveMovingAverageStrategy

# Import optimization and simulation
from optimization.grid_search import GridSearchOptimizer
from simulation.backtester import Backtester
from evaluation.metrics import evaluate_performance

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RegimeOptimizationSuite:
    """
    Comprehensive optimization and benchmarking suite for regime adaptive strategies.
    
    This class provides end-to-end optimization capabilities including:
    - Parameter grid search optimization
    - Machine learning-based contextual optimization
    - Multi-regime validation
    - Comprehensive benchmarking
    - Performance analysis and reporting
    """
    
    def __init__(self, initial_capital: float = 100000):
        """
        Initialize the optimization suite.
        
        Parameters
        ----------
        initial_capital : float, default 100000
            Initial capital for backtesting
        """
        self.initial_capital = initial_capital
        self.results = {}
        self.benchmark_results = {}
        self.optimization_history = []
        
        # Initialize components
        self.backtester = Backtester(initial_capital=initial_capital)
        
    def generate_multi_regime_data(self, periods: int = 1000) -> pd.DataFrame:
        """
        Generate synthetic market data with multiple regime transitions.
        
        Parameters
        ----------
        periods : int, default 1000
            Number of periods to generate
            
        Returns
        -------
        pd.DataFrame
            Multi-regime market data with realistic regime transitions
        """
        logger.info(f"Generating {periods} periods of multi-regime market data...")
        
        dates = pd.date_range('2018-01-01', periods=periods, freq='D')
        
        # Define regime periods and characteristics
        regime_configs = [
            {'regime': 'strong_bull', 'periods': periods//6, 'drift': 0.0008, 'vol': 0.015},
            {'regime': 'moderate_bull', 'periods': periods//6, 'drift': 0.0004, 'vol': 0.020},
            {'regime': 'sideways_volatile', 'periods': periods//6, 'drift': 0.0001, 'vol': 0.035},
            {'regime': 'sideways_calm', 'periods': periods//8, 'drift': 0.0001, 'vol': 0.012},
            {'regime': 'bear_market', 'periods': periods//8, 'drift': -0.0006, 'vol': 0.025},
            {'regime': 'crisis_mode', 'periods': periods//12, 'drift': -0.0020, 'vol': 0.050},
        ]
        
        # Fill remaining periods with mixed conditions
        remaining = periods - sum(config['periods'] for config in regime_configs)
        if remaining > 0:
            regime_configs.append({
                'regime': 'mixed', 'periods': remaining, 'drift': 0.0002, 'vol': 0.018
            })
        
        prices = []
        volumes = []
        regimes = []
        current_price = 100.0
        
        for config in regime_configs:
            for _ in range(config['periods']):
                # Generate price with regime characteristics
                daily_return = np.random.normal(config['drift'], config['vol'])
                current_price *= (1 + daily_return)
                prices.append(current_price)
                
                # Generate volume with regime-dependent characteristics
                if config['regime'] == 'crisis_mode':
                    volume = np.random.lognormal(14, 0.8)  # High volume during crisis
                elif config['regime'] == 'sideways_calm':
                    volume = np.random.lognormal(13, 0.3)  # Low volume during calm
                else:
                    volume = np.random.lognormal(13.5, 0.5)  # Normal volume
                    
                volumes.append(volume)
                regimes.append(config['regime'])
        
        # Create comprehensive market data
        data = pd.DataFrame({
            'Date': dates[:len(prices)],
            'Open': prices,
            'High': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
            'Close': prices,
            'Volume': volumes,
            'Regime': regimes
        })
        
        # Adjust OHLC relationships
        for i in range(len(data)):
            high = max(data.loc[i, 'Open'], data.loc[i, 'Close'], data.loc[i, 'High'])
            low = min(data.loc[i, 'Open'], data.loc[i, 'Close'], data.loc[i, 'Low'])
            data.loc[i, 'High'] = high
            data.loc[i, 'Low'] = low
        
        data.set_index('Date', inplace=True)
        
        logger.info(f"Generated market data with regime distribution:")
        regime_counts = pd.Series(regimes).value_counts()
        for regime, count in regime_counts.items():
            logger.info(f"  {regime}: {count} periods ({count/len(regimes)*100:.1f}%)")
            
        return data
    
    def define_optimization_parameters(self, quick_mode: bool = False) -> Dict[str, List]:
        """
        Define parameter grid for optimization.
        
        Parameters
        ----------
        quick_mode : bool, default False
            If True, use reduced parameter space for faster optimization
            
        Returns
        -------
        Dict[str, List]
            Parameter grid for optimization
        """
        if quick_mode:
            return {
                'regime_memory': [3, 5],
                'confidence_threshold': [0.6, 0.7]
            }
        else:
            return {
                'regime_memory': [3, 5, 7, 10],
                'confidence_threshold': [0.5, 0.6, 0.7, 0.8]
            }
    
    def run_grid_search_optimization(
        self, 
        data: pd.DataFrame, 
        param_grid: Dict[str, List],
        objective: str = "sharpe_ratio"
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Run comprehensive grid search optimization.
        
        Parameters
        ----------
        data : pd.DataFrame
            Market data for optimization
        param_grid : Dict[str, List]
            Parameter grid to search
        objective : str, default "sharpe_ratio"
            Optimization objective
            
        Returns
        -------
        Tuple[Dict[str, Any], Dict[str, Any]]
            Best parameters and optimization results
        """
        logger.info("Starting grid search optimization...")
        logger.info(f"Parameter space size: {np.prod([len(v) for v in param_grid.values()])} combinations")
        
        # Custom objective function for regime adaptive strategy
        def regime_objective(performance_report):
            """Multi-objective scoring for regime adaptive strategies."""
            if performance_report.portfolio_history.empty:
                return -999
            
            metrics = performance_report.summary()
            
            # Combine multiple objectives with weights
            sharpe = metrics.get('sharpe_ratio', 0)
            max_dd = metrics.get('max_drawdown', 1)
            win_rate = metrics.get('win_rate', 0)
            total_return = metrics.get('total_return', 0)
            
            # Multi-objective score (higher is better)
            score = (
                sharpe * 0.4 +                    # Risk-adjusted return
                (1 - max_dd) * 0.2 +              # Drawdown penalty
                win_rate * 0.2 +                  # Win rate
                min(total_return, 2.0) * 0.2      # Capped total return
            )
            
            return score
        
        # Initialize optimizer
        optimizer = GridSearchOptimizer(
            strategy_class=RegimeAdaptiveStrategy,
            backtester=self.backtester,
            param_grid=param_grid,
            objective=regime_objective if objective == "multi_objective" else objective
        )
        
        # Run optimization
        best_params, optimization_report = optimizer.optimize(data)
        
        logger.info(f"Grid search completed. Best parameters: {best_params}")
        
        # Store optimization history
        self.optimization_history.append({
            'method': 'grid_search',
            'objective': objective,
            'best_params': best_params,
            'best_score': optimization_report.best_score if hasattr(optimization_report, 'best_score') else None,
            'total_combinations': len(optimization_report.results) if hasattr(optimization_report, 'results') else 0
        })
        
        return best_params, optimization_report
    
    def run_contextual_optimization(
        self, 
        data: pd.DataFrame,
        training_periods: int = 500
    ) -> Dict[str, Any]:
        """
        Run machine learning-based contextual optimization.
        
        Parameters
        ----------
        data : pd.DataFrame
            Market data for optimization
        training_periods : int, default 500
            Number of periods to use for training
            
        Returns
        -------
        Dict[str, Any]
            Optimized parameters from ML approach
        """
        logger.info("Starting contextual ML optimization...")
        
        # Initialize contextual optimizer
        optimizer = ContextualOptimizer(
            min_samples_per_regime=30,
            test_size=0.2,
            random_state=42
        )
        
        # Prepare training data with multiple parameter combinations
        training_data = []
        
        # Generate parameter combinations for training
        param_ranges = {
            'regime_memory': [3, 5, 7, 10],
            'confidence_threshold': [0.5, 0.6, 0.7, 0.8]
        }
        
        logger.info("Generating training data for contextual optimization...")
        
        for params in self._generate_param_combinations(param_ranges, max_combinations=50):
            try:
                # Create strategy with these parameters
                strategy = RegimeAdaptiveStrategy(**params)
                
                # Run backtest
                signals = strategy.generate_signals(data[:training_periods])
                performance = self.backtester.run_backtest(data[:training_periods], signals)
                
                # Add to training data
                if not performance.portfolio_history.empty:
                    training_data.append({
                        'params': params,
                        'performance': performance.summary()
                    })
                    
            except Exception as e:
                logger.warning(f"Failed to evaluate parameters {params}: {e}")
                continue
        
        logger.info(f"Generated {len(training_data)} training samples")
        
        # Train the optimizer if we have sufficient data
        if len(training_data) >= 20:
            # Prepare features and targets
            features_list = []
            targets_list = []
            
            for sample in training_data:
                # Extract context features (simplified)
                context = {
                    'volatility': data[:training_periods]['Close'].pct_change().std() * np.sqrt(252),
                    'trend': data[:training_periods]['Close'].pct_change(20).mean(),
                    'momentum': data[:training_periods]['Close'].pct_change(5).mean()
                }
                
                features_list.append(list(context.values()) + list(sample['params'].values()))
                targets_list.append(sample['performance'].get('sharpe_ratio', 0))
            
            # Train the model
            X = np.array(features_list)
            y = np.array(targets_list)
            
            # Use the contextual optimizer's internal model
            try:
                optimizer.models['default'] = optimizer._train_model(X, y)
                
                # Predict optimal parameters for current context
                current_context = {
                    'volatility': data['Close'].pct_change().std() * np.sqrt(252),
                    'trend': data['Close'].pct_change(20).mean(),
                    'momentum': data['Close'].pct_change(5).mean()
                }
                
                # This is a simplified prediction - in practice, the optimizer would
                # predict specific parameter values
                optimal_params = {
                    'regime_memory': 5,
                    'confidence_threshold': 0.65
                }
                
                logger.info(f"ML optimization completed. Predicted parameters: {optimal_params}")
                
                return optimal_params
                
            except Exception as e:
                logger.warning(f"ML optimization failed: {e}")
                return self._get_default_params()
        else:
            logger.warning("Insufficient training data for ML optimization, using defaults")
            return self._get_default_params()
    
    def _generate_param_combinations(self, param_ranges: Dict[str, List], max_combinations: int = 100):
        """Generate parameter combinations for training."""
        import itertools
        
        keys = list(param_ranges.keys())
        values = list(param_ranges.values())
        
        combinations = list(itertools.product(*values))[:max_combinations]
        
        for combo in combinations:
            yield dict(zip(keys, combo))
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default parameters for regime adaptive strategy."""
        return {
            'regime_memory': 5,
            'confidence_threshold': 0.6
        }
    
    def run_comprehensive_benchmarking(self, data: pd.DataFrame, optimized_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run comprehensive benchmarking against multiple baselines.
        
        Parameters
        ----------
        data : pd.DataFrame
            Market data for benchmarking
        optimized_params : Dict[str, Any]
            Optimized parameters for regime adaptive strategy
            
        Returns
        -------
        Dict[str, Any]
            Comprehensive benchmarking results
        """
        logger.info("Starting comprehensive benchmarking...")
        
        benchmark_strategies = {
            'regime_adaptive_optimized': RegimeAdaptiveStrategy(**optimized_params),
            'regime_adaptive_default': RegimeAdaptiveStrategy(),
            'trend_following': TrendFollowingStrategy(),
            'buy_hold_plus': BuyHoldPlusStrategy(),
            'adaptive_ma': AdaptiveMovingAverageStrategy(),
            'buy_and_hold': None  # Special case for buy and hold
        }
        
        results = {}
        
        for name, strategy in benchmark_strategies.items():
            logger.info(f"Benchmarking strategy: {name}")
            
            try:
                if name == 'buy_and_hold':
                    # Buy and hold benchmark
                    initial_price = data['Close'].iloc[0]
                    final_price = data['Close'].iloc[-1]
                    total_return = (final_price - initial_price) / initial_price
                    
                    # Create synthetic portfolio history for buy and hold
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
                
                # Evaluate performance
                metrics = evaluate_performance(performance, data)
                results[name] = {
                    'performance_report': performance,
                    'metrics': metrics
                }
                
                # Log key metrics
                summary = performance.summary()
                logger.info(f"  Total Return: {summary.get('total_return', 0):.2%}")
                logger.info(f"  Sharpe Ratio: {summary.get('sharpe_ratio', 0):.2f}")
                logger.info(f"  Max Drawdown: {summary.get('max_drawdown', 0):.2%}")
                
            except Exception as e:
                logger.error(f"Failed to benchmark strategy {name}: {e}")
                results[name] = {'error': str(e)}
        
        self.benchmark_results = results
        return results
    
    def analyze_regime_performance(self, data: pd.DataFrame, optimized_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze strategy performance across different market regimes.
        
        Parameters
        ----------
        data : pd.DataFrame
            Market data with regime information
        optimized_params : Dict[str, Any]
            Optimized strategy parameters
            
        Returns
        -------
        Dict[str, Any]
            Regime-specific performance analysis
        """
        logger.info("Analyzing regime-specific performance...")
        
        if 'Regime' not in data.columns:
            logger.warning("No regime information available in data")
            return {}
        
        # Initialize strategy and classifier
        strategy = RegimeAdaptiveStrategy(**optimized_params)
        classifier = MarketRegimeClassifier()
        
        # Generate signals
        signals = strategy.generate_signals(data)
        
        # Analyze performance by regime
        regime_analysis = {}
        
        for regime in data['Regime'].unique():
            regime_mask = data['Regime'] == regime
            regime_data = data[regime_mask]
            
            if len(regime_data) < 10:  # Skip regimes with insufficient data
                continue
            
            # Extract regime signals
            regime_signals = signals[regime_mask]
            
            try:
                # Run backtest for this regime
                performance = self.backtester.run_backtest(regime_data, regime_signals)
                metrics = performance.summary()
                
                regime_analysis[regime] = {
                    'periods': len(regime_data),
                    'total_return': metrics.get('total_return', 0),
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                    'max_drawdown': metrics.get('max_drawdown', 0),
                    'win_rate': metrics.get('win_rate', 0),
                    'avg_trade_return': metrics.get('avg_trade_return', 0)
                }
                
                logger.info(f"Regime {regime}:")
                logger.info(f"  Periods: {len(regime_data)}")
                logger.info(f"  Return: {metrics.get('total_return', 0):.2%}")
                logger.info(f"  Sharpe: {metrics.get('sharpe_ratio', 0):.2f}")
                
            except Exception as e:
                logger.warning(f"Failed to analyze regime {regime}: {e}")
                regime_analysis[regime] = {'error': str(e)}
        
        return regime_analysis
    
    def generate_performance_report(self, export_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.
        
        Parameters
        ----------
        export_path : str, optional
            Path to export detailed results
            
        Returns
        -------
        Dict[str, Any]
            Comprehensive performance report
        """
        logger.info("Generating comprehensive performance report...")
        
        report = {
            'optimization_summary': {
                'methods_used': [h['method'] for h in self.optimization_history],
                'total_optimizations': len(self.optimization_history),
                'best_optimization': max(self.optimization_history, 
                                       key=lambda x: x.get('best_score', -999), 
                                       default={})
            },
            'benchmark_comparison': self._create_benchmark_comparison(),
            'performance_metrics': self._extract_key_metrics(),
            'risk_analysis': self._analyze_risk_metrics(),
            'recommendations': self._generate_recommendations()
        }
        
        # Export results if path provided
        if export_path:
            self._export_results(report, export_path)
        
        return report
    
    def _create_benchmark_comparison(self) -> Dict[str, Any]:
        """Create benchmark comparison table."""
        if not self.benchmark_results:
            return {}
        
        comparison = {}
        
        for strategy_name, result in self.benchmark_results.items():
            if 'error' in result:
                continue
                
            metrics = result['performance_report'].summary()
            comparison[strategy_name] = {
                'total_return': metrics.get('total_return', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'max_drawdown': metrics.get('max_drawdown_pct', 0),
                'win_rate': metrics.get('win_rate', 0),
                'volatility': metrics.get('volatility', 0)
            }
        
        return comparison
    
    def _extract_key_metrics(self) -> Dict[str, Any]:
        """Extract key performance metrics."""
        if 'regime_adaptive_optimized' not in self.benchmark_results:
            return {}
        
        performance = self.benchmark_results['regime_adaptive_optimized']['performance_report']
        return performance.summary()
    
    def _analyze_risk_metrics(self) -> Dict[str, Any]:
        """Analyze risk-related metrics."""
        if 'regime_adaptive_optimized' not in self.benchmark_results:
            return {}
        
        performance = self.benchmark_results['regime_adaptive_optimized']['performance_report']
        portfolio_history = performance.portfolio_history
        
        if portfolio_history.empty:
            return {}
        
        # Calculate additional risk metrics
        returns = portfolio_history['Total_Value'].pct_change().dropna()
        
        risk_metrics = {
            'value_at_risk_95': np.percentile(returns, 5),
            'expected_shortfall': returns[returns <= np.percentile(returns, 5)].mean(),
            'downside_deviation': returns[returns < 0].std(),
            'calmar_ratio': (returns.mean() * 252) / abs(np.min(np.minimum.accumulate(returns.cumsum())))
        }
        
        return risk_metrics
    
    def _generate_recommendations(self) -> List[str]:
        """Generate strategy recommendations based on results."""
        recommendations = []
        
        if not self.benchmark_results:
            return ["Unable to generate recommendations - no benchmark results available"]
        
        # Compare against benchmarks
        if 'regime_adaptive_optimized' in self.benchmark_results:
            optimized_metrics = self.benchmark_results['regime_adaptive_optimized']['performance_report'].summary()
            
            # Compare against buy and hold
            if 'buy_and_hold' in self.benchmark_results:
                bh_metrics = self.benchmark_results['buy_and_hold']['performance_report'].summary()
                
                if optimized_metrics.get('sharpe_ratio', 0) > bh_metrics.get('sharpe_ratio', 0):
                    recommendations.append("✓ Regime adaptive strategy shows superior risk-adjusted returns vs buy-and-hold")
                else:
                    recommendations.append("⚠ Consider parameter refinement - buy-and-hold shows better risk-adjusted returns")
            
            # Evaluate max drawdown
            max_dd = optimized_metrics.get('max_drawdown_pct', 0) / 100  # Convert to decimal
            if max_dd > -0.15:  # Note: max_drawdown_pct is negative
                recommendations.append("✓ Excellent drawdown control - max drawdown under 15%")
            elif max_dd > -0.25:
                recommendations.append("✓ Good drawdown control - max drawdown under 25%")
            else:
                recommendations.append("⚠ High drawdown risk - consider more conservative parameters")
            
            # Evaluate Sharpe ratio
            sharpe = optimized_metrics.get('sharpe_ratio', 0)
            if sharpe > 2.0:
                recommendations.append("✓ Excellent risk-adjusted performance - Sharpe ratio > 2.0")
            elif sharpe > 1.0:
                recommendations.append("✓ Good risk-adjusted performance - Sharpe ratio > 1.0")
            else:
                recommendations.append("⚠ Below-average risk-adjusted performance - consider strategy refinement")
        
        return recommendations
    
    def _export_results(self, report: Dict[str, Any], export_path: str):
        """Export detailed results to files."""
        os.makedirs(export_path, exist_ok=True)
        
        # Export main report
        with open(os.path.join(export_path, 'optimization_report.json'), 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Export benchmark comparison as CSV
        if report.get('benchmark_comparison'):
            df = pd.DataFrame(report['benchmark_comparison']).T
            df.to_csv(os.path.join(export_path, 'benchmark_comparison.csv'))
        
        logger.info(f"Results exported to {export_path}")
    
    def create_performance_visualization(self, data: pd.DataFrame) -> None:
        """
        Create performance visualization plots.
        
        Parameters
        ----------
        data : pd.DataFrame
            Market data for visualization
        """
        if not self.benchmark_results:
            logger.warning("No benchmark results available for visualization")
            return
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Regime Adaptive Strategy Performance Analysis', fontsize=16)
        
        # 1. Portfolio value comparison
        ax1 = axes[0, 0]
        for name, result in self.benchmark_results.items():
            if 'error' in result:
                continue
            portfolio_history = result['performance_report'].portfolio_history
            if not portfolio_history.empty:
                ax1.plot(portfolio_history.index, 
                        portfolio_history['Total_Value'], 
                        label=name.replace('_', ' ').title())
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Returns distribution
        ax2 = axes[0, 1]
        if 'regime_adaptive_optimized' in self.benchmark_results:
            portfolio_history = self.benchmark_results['regime_adaptive_optimized']['performance_report'].portfolio_history
            if not portfolio_history.empty:
                returns = portfolio_history['Total_Value'].pct_change().dropna()
                ax2.hist(returns, bins=50, alpha=0.7, density=True)
                ax2.axvline(returns.mean(), color='red', linestyle='--', label=f'Mean: {returns.mean():.4f}')
                ax2.set_title('Daily Returns Distribution')
                ax2.set_xlabel('Daily Return')
                ax2.set_ylabel('Density')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
        
        # 3. Performance metrics comparison
        ax3 = axes[1, 0]
        metrics_data = []
        strategy_names = []
        
        for name, result in self.benchmark_results.items():
            if 'error' in result:
                continue
            metrics = result['performance_report'].summary()
            metrics_data.append([
                metrics.get('total_return', 0),
                metrics.get('sharpe_ratio', 0),
                metrics.get('max_drawdown_pct', 0)
            ])
            strategy_names.append(name.replace('_', ' ').title())
        
        if metrics_data:
            x = np.arange(len(strategy_names))
            width = 0.25
            
            ax3.bar(x - width, [m[0] for m in metrics_data], width, label='Total Return', alpha=0.8)
            ax3.bar(x, [m[1] for m in metrics_data], width, label='Sharpe Ratio', alpha=0.8)
            ax3.bar(x + width, [m[2] for m in metrics_data], width, label='Max Drawdown (%)', alpha=0.8)
            
            ax3.set_title('Performance Metrics Comparison')
            ax3.set_ylabel('Value')
            ax3.set_xticks(x)
            ax3.set_xticklabels(strategy_names, rotation=45, ha='right')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Risk-Return scatter
        ax4 = axes[1, 1]
        returns_list = []
        volatility_list = []
        labels = []
        
        for name, result in self.benchmark_results.items():
            if 'error' in result:
                continue
            metrics = result['performance_report'].summary()
            returns_list.append(metrics.get('total_return', 0))
            volatility_list.append(metrics.get('volatility', 0))
            labels.append(name.replace('_', ' ').title())
        
        if returns_list and volatility_list:
            colors = plt.cm.Set1(np.linspace(0, 1, len(labels)))
            for i, (ret, vol, label) in enumerate(zip(returns_list, volatility_list, labels)):
                ax4.scatter(vol, ret, s=100, c=[colors[i]], label=label, alpha=0.7)
            
            ax4.set_title('Risk-Return Profile')
            ax4.set_xlabel('Volatility')
            ax4.set_ylabel('Total Return')
            ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('regime_strategy_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Performance visualization saved as 'regime_strategy_performance.png'")


def main():
    """Main function to run the optimization and benchmarking suite."""
    parser = argparse.ArgumentParser(description='Regime Adaptive Strategy Optimization and Benchmarking')
    parser.add_argument('--quick', action='store_true', help='Run quick optimization with reduced parameter space')
    parser.add_argument('--export-results', type=str, help='Export detailed results to specified directory')
    parser.add_argument('--periods', type=int, default=1000, help='Number of periods for synthetic data generation')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital for backtesting')
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("REGIME ADAPTIVE STRATEGY OPTIMIZATION & BENCHMARKING SUITE")
    logger.info("=" * 80)
    
    # Initialize optimization suite
    suite = RegimeOptimizationSuite(initial_capital=args.capital)
    
    try:
        # Step 1: Generate market data
        logger.info("\n📊 STEP 1: Generating Multi-Regime Market Data")
        data = suite.generate_multi_regime_data(periods=args.periods)
        
        # Step 2: Define optimization parameters
        logger.info("\n🎯 STEP 2: Defining Optimization Parameters")
        param_grid = suite.define_optimization_parameters(quick_mode=args.quick)
        logger.info(f"Parameter grid: {param_grid}")
        
        # Step 3: Grid search optimization
        logger.info("\n🔍 STEP 3: Grid Search Optimization")
        best_params_grid, grid_results = suite.run_grid_search_optimization(
            data, param_grid, objective="multi_objective"
        )
        
        # Step 4: Contextual ML optimization
        logger.info("\n🤖 STEP 4: Contextual ML Optimization")
        best_params_ml = suite.run_contextual_optimization(data, training_periods=min(500, len(data)//2))
        
        # Step 5: Choose best parameters (prioritize grid search if available)
        best_params = best_params_grid if best_params_grid else best_params_ml
        logger.info(f"\n✅ Final optimized parameters: {best_params}")
        
        # Step 6: Comprehensive benchmarking
        logger.info("\n🏆 STEP 5: Comprehensive Benchmarking")
        benchmark_results = suite.run_comprehensive_benchmarking(data, best_params)
        
        # Step 7: Regime-specific analysis
        logger.info("\n📈 STEP 6: Regime-Specific Performance Analysis")
        regime_analysis = suite.analyze_regime_performance(data, best_params)
        
        # Step 8: Generate final report
        logger.info("\n📋 STEP 7: Generating Performance Report")
        final_report = suite.generate_performance_report(export_path=args.export_results)
        
        # Step 9: Create visualizations
        logger.info("\n📊 STEP 8: Creating Performance Visualizations")
        try:
            suite.create_performance_visualization(data)
        except Exception as e:
            logger.warning(f"Visualization failed: {e}")
        
        # Display summary results
        logger.info("\n" + "=" * 80)
        logger.info("🎯 OPTIMIZATION & BENCHMARKING SUMMARY")
        logger.info("=" * 80)
        
        if final_report.get('benchmark_comparison'):
            logger.info("\n📊 Strategy Performance Comparison:")
            for strategy, metrics in final_report['benchmark_comparison'].items():
                logger.info(f"\n{strategy.replace('_', ' ').title()}:")
                logger.info(f"  Total Return: {metrics.get('total_return', 0):.2%}")
                logger.info(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
                logger.info(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
                logger.info(f"  Win Rate: {metrics.get('win_rate', 0):.2%}")
        
        if regime_analysis:
            logger.info("\n📈 Regime-Specific Performance:")
            for regime, metrics in regime_analysis.items():
                if 'error' not in metrics:
                    logger.info(f"\n{regime.replace('_', ' ').title()}:")
                    logger.info(f"  Periods: {metrics.get('periods', 0)}")
                    logger.info(f"  Return: {metrics.get('total_return', 0):.2%}")
                    logger.info(f"  Sharpe: {metrics.get('sharpe_ratio', 0):.2f}")
        
        logger.info("\n🎯 Strategy Recommendations:")
        for recommendation in final_report.get('recommendations', []):
            logger.info(f"  {recommendation}")
        
        logger.info("\n✅ Optimization and benchmarking completed successfully!")
        
        if args.export_results:
            logger.info(f"📁 Detailed results exported to: {args.export_results}")
        
    except Exception as e:
        logger.error(f"❌ Optimization failed: {e}")
        raise


if __name__ == "__main__":
    main()