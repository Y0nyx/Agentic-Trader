"""
Strategy Database module for storing and retrieving historical strategy performance data.

This module provides a simple in-memory database that can be extended to use
SQLite or other persistence mechanisms for storing strategy results.
"""

import json
import hashlib
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd


@dataclass
class StrategyResult:
    """Data class for storing strategy performance results."""
    
    strategy_type: str
    parameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    market_regime: Optional[str] = None
    timestamp: Optional[str] = None
    signature: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.signature is None:
            self.signature = self._generate_signature()
    
    def _generate_signature(self) -> str:
        """Generate a unique signature for this strategy configuration."""
        param_str = json.dumps(self.parameters, sort_keys=True)
        content = f"{self.strategy_type}_{param_str}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


@dataclass
class StrategyInsights:
    """Data class for aggregated strategy insights."""
    
    strategy_type: str
    best_params: Dict[str, Any]
    worst_params: Dict[str, Any]
    top_performers: List[Dict[str, Any]]
    worst_performers: List[Dict[str, Any]]
    success_rate: float
    regime_performance: Dict[str, Dict[str, float]]
    failure_patterns: List[str]
    suggestions: List[str]
    avg_performance: Dict[str, float]


class StrategyDatabase:
    """
    Simple in-memory database for storing strategy performance data.
    
    This class manages historical strategy results and provides methods
    for querying and analyzing performance patterns.
    """
    
    def __init__(self):
        self.results: List[StrategyResult] = []
        self._load_sample_data()
    
    def add_result(self, result: StrategyResult) -> None:
        """Add a strategy result to the database."""
        self.results.append(result)
    
    def get_strategy_results(self, strategy_type: str) -> List[StrategyResult]:
        """Get all results for a specific strategy type."""
        return [r for r in self.results if r.strategy_type == strategy_type]
    
    def find_similar_strategy(self, signature: str) -> Optional[StrategyResult]:
        """Find a strategy with similar signature."""
        for result in self.results:
            if result.signature == signature:
                return result
        return None
    
    def get_strategy_performance_summary(self, strategy_type: str) -> StrategyInsights:
        """Get aggregated performance insights for a strategy type."""
        results = self.get_strategy_results(strategy_type)
        
        if not results:
            return self._create_empty_insights(strategy_type)
        
        # Sort by Sharpe ratio (primary metric)
        sorted_results = sorted(
            results, 
            key=lambda r: r.performance_metrics.get('sharpe_ratio', 0), 
            reverse=True
        )
        
        best_result = sorted_results[0]
        worst_result = sorted_results[-1]
        
        # Calculate success rate (positive Sharpe ratio)
        successful = [r for r in results if r.performance_metrics.get('sharpe_ratio', 0) > 0]
        success_rate = (len(successful) / len(results)) * 100 if results else 0
        
        # Get top and worst performers
        top_performers = [
            {
                'params': r.parameters,
                'sharpe': r.performance_metrics.get('sharpe_ratio', 0),
                'return': r.performance_metrics.get('total_return_pct', 0)
            }
            for r in sorted_results[:3]
        ]
        
        worst_performers = [
            {
                'params': r.parameters,
                'sharpe': r.performance_metrics.get('sharpe_ratio', 0),
                'return': r.performance_metrics.get('total_return_pct', 0)
            }
            for r in sorted_results[-3:]
        ]
        
        # Analyze regime performance
        regime_performance = self._analyze_regime_performance(results)
        
        # Generate failure patterns and suggestions
        failure_patterns = self._identify_failure_patterns(results)
        suggestions = self._generate_suggestions(results)
        
        # Calculate average performance
        avg_performance = self._calculate_average_performance(results)
        
        return StrategyInsights(
            strategy_type=strategy_type,
            best_params=best_result.parameters,
            worst_params=worst_result.parameters,
            top_performers=top_performers,
            worst_performers=worst_performers,
            success_rate=success_rate,
            regime_performance=regime_performance,
            failure_patterns=failure_patterns,
            suggestions=suggestions,
            avg_performance=avg_performance
        )
    
    def get_optimal_parameters(self, strategy_type: str, market_regime: str = None) -> Dict[str, Any]:
        """Get optimal parameters for a strategy type and market regime."""
        results = self.get_strategy_results(strategy_type)
        
        if market_regime:
            results = [r for r in results if r.market_regime == market_regime]
        
        if not results:
            return self._get_default_parameters(strategy_type)
        
        # Find best performing strategy
        best_result = max(
            results, 
            key=lambda r: r.performance_metrics.get('sharpe_ratio', 0)
        )
        
        return best_result.parameters
    
    def get_successful_strategies(self, strategy_type: str) -> List[StrategyResult]:
        """Get strategies with positive performance."""
        results = self.get_strategy_results(strategy_type)
        return [r for r in results if r.performance_metrics.get('sharpe_ratio', 0) > 0]
    
    def _create_empty_insights(self, strategy_type: str) -> StrategyInsights:
        """Create empty insights when no data is available."""
        return StrategyInsights(
            strategy_type=strategy_type,
            best_params={},
            worst_params={},
            top_performers=[],
            worst_performers=[],
            success_rate=0,
            regime_performance={},
            failure_patterns=["No historical data available"],
            suggestions=["Start with default parameters", "Run initial optimization"],
            avg_performance={}
        )
    
    def _analyze_regime_performance(self, results: List[StrategyResult]) -> Dict[str, Dict[str, float]]:
        """Analyze performance by market regime."""
        regime_performance = {}
        
        for result in results:
            regime = result.market_regime or "unknown"
            if regime not in regime_performance:
                regime_performance[regime] = []
            
            regime_performance[regime].append(result.performance_metrics.get('sharpe_ratio', 0))
        
        # Calculate average performance per regime
        return {
            regime: {
                'avg_sharpe': sum(sharpes) / len(sharpes),
                'count': len(sharpes)
            }
            for regime, sharpes in regime_performance.items()
        }
    
    def _identify_failure_patterns(self, results: List[StrategyResult]) -> List[str]:
        """Identify common failure patterns."""
        patterns = []
        
        # Analyze poor performing strategies
        poor_results = [r for r in results if r.performance_metrics.get('sharpe_ratio', 0) < 0]
        
        if len(poor_results) > len(results) * 0.5:
            patterns.append("High failure rate - consider different strategy type")
        
        # Check for parameter patterns in failures
        if poor_results and len(poor_results) > 2:
            # Simple pattern detection for moving average strategies
            short_windows = [r.parameters.get('short_window', 0) for r in poor_results]
            long_windows = [r.parameters.get('long_window', 0) for r in poor_results]
            
            if sum(short_windows) > 0 and sum(long_windows) > 0:
                avg_short = sum(short_windows) / len([w for w in short_windows if w > 0])
                avg_long = sum(long_windows) / len([w for w in long_windows if w > 0])
                
                if avg_short > 15:
                    patterns.append("Short windows > 15 often underperform")
                if avg_long > 60:
                    patterns.append("Long windows > 60 may be too slow")
        
        return patterns or ["No clear failure patterns identified"]
    
    def _generate_suggestions(self, results: List[StrategyResult]) -> List[str]:
        """Generate improvement suggestions based on historical data."""
        suggestions = []
        
        successful_results = [r for r in results if r.performance_metrics.get('sharpe_ratio', 0) > 0]
        
        if successful_results:
            # Analyze successful parameter ranges
            if successful_results[0].parameters.get('short_window'):
                short_windows = [r.parameters.get('short_window', 0) for r in successful_results]
                optimal_short = sum(short_windows) / len(short_windows)
                suggestions.append(f"Optimal short window around {optimal_short:.0f}")
            
            if successful_results[0].parameters.get('long_window'):
                long_windows = [r.parameters.get('long_window', 0) for r in successful_results]
                optimal_long = sum(long_windows) / len(long_windows)
                suggestions.append(f"Optimal long window around {optimal_long:.0f}")
            
            # Check for filters or additional features
            rsi_users = [r for r in successful_results if r.parameters.get('use_rsi_filter')]
            if len(rsi_users) > len(successful_results) * 0.6:
                suggestions.append("RSI filter improves performance")
            
            volume_users = [r for r in successful_results if r.parameters.get('volume_confirmation')]
            if len(volume_users) > len(successful_results) * 0.6:
                suggestions.append("Volume confirmation reduces false signals")
        
        return suggestions or ["Try parameter optimization", "Consider adding filters"]
    
    def _calculate_average_performance(self, results: List[StrategyResult]) -> Dict[str, float]:
        """Calculate average performance metrics."""
        if not results:
            return {}
        
        metrics = {}
        metric_names = ['sharpe_ratio', 'total_return_pct', 'max_drawdown_pct', 'win_rate']
        
        for metric in metric_names:
            values = [r.performance_metrics.get(metric, 0) for r in results]
            metrics[metric] = sum(values) / len(values) if values else 0
        
        return metrics
    
    def _get_default_parameters(self, strategy_type: str) -> Dict[str, Any]:
        """Get default parameters for a strategy type."""
        defaults = {
            'moving_average': {
                'short_window': 10,
                'long_window': 30,
                'use_rsi_filter': True,
                'volume_confirmation': True
            },
            'rsi': {
                'rsi_period': 14,
                'oversold': 25,
                'overbought': 75,
                'volume_filter': True
            },
            'trend_following': {
                'lookback_period': 20,
                'ma_period': 50,
                'atr_period': 14
            }
        }
        
        return defaults.get(strategy_type, {})
    
    def _load_sample_data(self):
        """Load sample historical data for demonstration."""
        sample_results = [
            # Moving Average strategies
            StrategyResult(
                strategy_type="moving_average",
                parameters={"short_window": 10, "long_window": 30},
                performance_metrics={"sharpe_ratio": 1.35, "total_return_pct": 15.2, "max_drawdown_pct": -8.5},
                market_regime="trending"
            ),
            StrategyResult(
                strategy_type="moving_average", 
                parameters={"short_window": 8, "long_window": 25},
                performance_metrics={"sharpe_ratio": 1.18, "total_return_pct": 12.8, "max_drawdown_pct": -6.2},
                market_regime="trending"
            ),
            StrategyResult(
                strategy_type="moving_average",
                parameters={"short_window": 20, "long_window": 50},
                performance_metrics={"sharpe_ratio": 0.85, "total_return_pct": 8.5, "max_drawdown_pct": -12.1},
                market_regime="sideways"
            ),
            StrategyResult(
                strategy_type="moving_average",
                parameters={"short_window": 5, "long_window": 15},
                performance_metrics={"sharpe_ratio": -0.25, "total_return_pct": -3.2, "max_drawdown_pct": -18.5},
                market_regime="volatile"
            ),
            # RSI strategies
            StrategyResult(
                strategy_type="rsi",
                parameters={"rsi_period": 14, "oversold": 25, "overbought": 75},
                performance_metrics={"sharpe_ratio": 1.42, "total_return_pct": 18.7, "max_drawdown_pct": -5.8},
                market_regime="mean_reverting"
            ),
            StrategyResult(
                strategy_type="rsi",
                parameters={"rsi_period": 21, "oversold": 30, "overbought": 70},
                performance_metrics={"sharpe_ratio": 0.95, "total_return_pct": 9.8, "max_drawdown_pct": -9.2},
                market_regime="mean_reverting"
            ),
        ]
        
        self.results.extend(sample_results)


# Global database instance
_strategy_db = StrategyDatabase()


def get_strategy_database() -> StrategyDatabase:
    """Get the global strategy database instance."""
    return _strategy_db