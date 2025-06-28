"""
Multi-Strategy Portfolio Management System.

This module implements a sophisticated portfolio allocation system that
dynamically allocates capital across different trading strategies based
on market conditions and strategy performance.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from .regime_adaptive_strategy import RegimeAdaptiveStrategy
from .trend_following_strategy import TrendFollowingStrategy
from .adaptive_ma_strategy import AdaptiveMovingAverageStrategy
from .buy_hold_plus_strategy import BuyHoldPlusStrategy
from .specialized_strategies import IntelligentBullStrategy, CrisisProtectionStrategy, VolatilityRangeStrategy
from .market_regime_classifier import MarketRegimeClassifier

logger = logging.getLogger(__name__)


class MultiStrategyPortfolio:
    """
    Portfolio manager that dynamically allocates capital between strategies.
    
    This system manages multiple trading strategies and allocates capital
    based on:
    1. Current market regime
    2. Individual strategy performance
    3. Risk management constraints
    4. Diversification requirements
    
    Allocation Strategies:
    - Bull market: 60% trend, 20% momentum, 20% defensive
    - Bear market: 70% defensive, 30% mean reversion
    - Sideways: 50% mean reversion, 30% momentum, 20% defensive
    - Crisis: 80% cash, 20% crisis protection
    
    Parameters
    ----------
    initial_capital : float, default 100000
        Initial portfolio capital
    max_strategy_allocation : float, default 0.4
        Maximum allocation to any single strategy
    min_strategy_allocation : float, default 0.05
        Minimum allocation to maintain diversification
    rebalance_frequency : int, default 20
        Number of periods between rebalancing
    """
    
    def __init__(
        self,
        initial_capital: float = 100000,
        max_strategy_allocation: float = 0.4,
        min_strategy_allocation: float = 0.05,
        rebalance_frequency: int = 20
    ):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_strategy_allocation = max_strategy_allocation
        self.min_strategy_allocation = min_strategy_allocation
        self.rebalance_frequency = rebalance_frequency
        
        # Initialize strategies
        self.strategies = self._initialize_strategies()
        
        # Portfolio tracking
        self.allocations = self._get_default_allocations()
        self.strategy_performance = {name: [] for name in self.strategies.keys()}
        self.portfolio_history = []
        self.rebalance_count = 0
        self.last_rebalance = 0
        
        # Market regime classifier
        self.regime_classifier = MarketRegimeClassifier()
        
    def _initialize_strategies(self) -> Dict[str, Any]:
        """Initialize the strategy instances."""
        strategies = {
            'trend_following': TrendFollowingStrategy(
                trend_window=50,
                confirmation_window=20,
                min_trend_strength=25
            ),
            'mean_reversion': VolatilityRangeStrategy(
                lookback_window=50,
                volatility_window=20,
                mean_reversion_threshold=2.0
            ),
            'momentum': AdaptiveMovingAverageStrategy(
                fast_period=10,
                slow_period=30
            ),
            'defensive': IntelligentBullStrategy(
                ma_window=200,
                exit_rsi_threshold=80,
                exit_volatility_threshold=0.30
            ),
            'crisis_protection': CrisisProtectionStrategy(
                crisis_volatility_threshold=0.40,
                crisis_volume_threshold=3.0,
                recovery_periods=10
            ),
            'buy_hold_plus': BuyHoldPlusStrategy(
                stress_rsi_threshold=25,
                drawdown_threshold=0.15  # 15% drawdown threshold (positive value)
            )
        }
        
        return strategies
    
    def _get_default_allocations(self) -> Dict[str, float]:
        """Get default strategy allocations."""
        return {
            'trend_following': 0.25,
            'mean_reversion': 0.20,
            'momentum': 0.20,
            'defensive': 0.15,
            'crisis_protection': 0.10,
            'buy_hold_plus': 0.10
        }
    
    def generate_portfolio_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate portfolio signals with dynamic allocation.
        
        Parameters
        ----------
        data : pd.DataFrame
            Market data with OHLCV columns
            
        Returns
        -------
        pd.DataFrame
            Portfolio signals with allocation information
        """
        if len(data) < 50:
            logger.warning("Insufficient data for portfolio management")
            return self._create_empty_portfolio_signals(data)
        
        # Generate signals for each strategy
        strategy_signals = {}
        for name, strategy in self.strategies.items():
            try:
                signals = strategy.generate_signals(data.copy())
                strategy_signals[name] = signals
            except Exception as e:
                logger.warning(f"Error generating signals for {name}: {e}")
                strategy_signals[name] = self._create_default_signals(data)
        
        # Determine current market regime
        current_regime = self.regime_classifier.classify_regime(data)
        
        # Update allocations based on regime and performance
        if self._should_rebalance(len(data)):
            self._rebalance_portfolio(current_regime, strategy_signals, data)
        
        # Combine signals with allocations
        portfolio_signals = self._combine_strategy_signals(strategy_signals, data)
        
        # Add portfolio metadata
        portfolio_signals['Current_Regime'] = current_regime
        portfolio_signals['Rebalance_Count'] = self.rebalance_count
        
        return portfolio_signals
    
    def _should_rebalance(self, current_period: int) -> bool:
        """Determine if portfolio should be rebalanced."""
        periods_since_rebalance = current_period - self.last_rebalance
        return periods_since_rebalance >= self.rebalance_frequency
    
    def _rebalance_portfolio(
        self,
        current_regime: str,
        strategy_signals: Dict[str, pd.DataFrame],
        data: pd.DataFrame
    ):
        """Rebalance portfolio allocations based on regime and performance."""
        logger.info(f"Rebalancing portfolio for regime: {current_regime}")
        
        # Get regime-based base allocations
        regime_allocations = self._get_regime_allocations(current_regime)
        
        # Adjust based on recent strategy performance
        performance_adjustments = self._calculate_performance_adjustments(strategy_signals)
        
        # Combine regime and performance factors
        new_allocations = self._combine_allocation_factors(
            regime_allocations, performance_adjustments
        )
        
        # Apply constraints
        self.allocations = self._apply_allocation_constraints(new_allocations)
        
        # Update tracking
        self.rebalance_count += 1
        self.last_rebalance = len(data)
        
        logger.info(f"New allocations: {self.allocations}")
    
    def _get_regime_allocations(self, regime: str) -> Dict[str, float]:
        """Get base allocations for specific market regime."""
        regime_allocations = {
            'strong_bull': {
                'buy_hold_plus': 0.40,
                'trend_following': 0.25,
                'momentum': 0.20,
                'defensive': 0.10,
                'crisis_protection': 0.05,
                'mean_reversion': 0.00
            },
            'moderate_bull': {
                'trend_following': 0.35,
                'momentum': 0.25,
                'buy_hold_plus': 0.20,
                'defensive': 0.15,
                'crisis_protection': 0.05,
                'mean_reversion': 0.00
            },
            'sideways_volatile': {
                'mean_reversion': 0.40,
                'momentum': 0.25,
                'trend_following': 0.15,
                'defensive': 0.15,
                'crisis_protection': 0.05,
                'buy_hold_plus': 0.00
            },
            'sideways_calm': {
                'mean_reversion': 0.30,
                'momentum': 0.25,
                'trend_following': 0.20,
                'defensive': 0.15,
                'crisis_protection': 0.05,
                'buy_hold_plus': 0.05
            },
            'bear_market': {
                'defensive': 0.40,
                'crisis_protection': 0.25,
                'mean_reversion': 0.20,
                'trend_following': 0.10,
                'momentum': 0.05,
                'buy_hold_plus': 0.00
            },
            'crisis_mode': {
                'crisis_protection': 0.60,
                'defensive': 0.30,
                'mean_reversion': 0.10,
                'trend_following': 0.00,
                'momentum': 0.00,
                'buy_hold_plus': 0.00
            }
        }
        
        return regime_allocations.get(regime, self._get_default_allocations())
    
    def _calculate_performance_adjustments(
        self,
        strategy_signals: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """Calculate performance-based allocation adjustments."""
        adjustments = {}
        
        for name, signals in strategy_signals.items():
            if 'Position' in signals.columns and len(signals) > 20:
                # Simple performance proxy: recent return * signal consistency
                recent_positions = signals['Position'].tail(20)
                recent_returns = signals['Close'].pct_change().tail(20)
                
                # Strategy return proxy
                strategy_return = (recent_positions.shift(1) * recent_returns).sum()
                
                # Signal consistency (prefer stable strategies)
                position_changes = (recent_positions != recent_positions.shift(1)).sum()
                consistency_score = 1.0 - (position_changes / len(recent_positions))
                
                # Combined performance score
                performance_score = strategy_return * consistency_score
                adjustments[name] = performance_score
            else:
                adjustments[name] = 0.0
        
        # Normalize adjustments
        if adjustments:
            max_adj = max(abs(v) for v in adjustments.values())
            if max_adj > 0:
                adjustments = {k: v / max_adj * 0.1 for k, v in adjustments.items()}  # Max 10% adjustment
        
        return adjustments
    
    def _combine_allocation_factors(
        self,
        regime_allocations: Dict[str, float],
        performance_adjustments: Dict[str, float]
    ) -> Dict[str, float]:
        """Combine regime-based and performance-based allocations."""
        combined = {}
        
        for strategy in self.strategies.keys():
            base_allocation = regime_allocations.get(strategy, 0.0)
            performance_adj = performance_adjustments.get(strategy, 0.0)
            
            # Combine with dampening factor
            combined[strategy] = base_allocation + performance_adj
        
        # Normalize to sum to 1.0
        total_allocation = sum(combined.values())
        if total_allocation > 0:
            combined = {k: v / total_allocation for k, v in combined.items()}
        else:
            combined = self._get_default_allocations()
        
        return combined
    
    def _apply_allocation_constraints(self, allocations: Dict[str, float]) -> Dict[str, float]:
        """Apply min/max allocation constraints."""
        constrained = allocations.copy()
        
        # Apply maximum constraint
        for strategy in constrained:
            if constrained[strategy] > self.max_strategy_allocation:
                constrained[strategy] = self.max_strategy_allocation
        
        # Apply minimum constraint for active strategies
        active_strategies = [s for s in constrained if constrained[s] > 0]
        for strategy in active_strategies:
            if constrained[strategy] < self.min_strategy_allocation:
                constrained[strategy] = self.min_strategy_allocation
        
        # Renormalize
        total = sum(constrained.values())
        if total > 0:
            constrained = {k: v / total for k, v in constrained.items()}
        
        return constrained
    
    def _combine_strategy_signals(
        self,
        strategy_signals: Dict[str, pd.DataFrame],
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """Combine individual strategy signals into portfolio signals."""
        result = data.copy()
        
        # Initialize portfolio columns
        result['Portfolio_Signal'] = 'HOLD'
        result['Portfolio_Position'] = 0.0
        
        # Add allocation columns
        for strategy, allocation in self.allocations.items():
            result[f'{strategy}_allocation'] = allocation
        
        # Calculate weighted portfolio position
        for i in range(len(result)):
            total_position = 0.0
            signal_votes = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
            
            for strategy, allocation in self.allocations.items():
                if strategy in strategy_signals:
                    signals = strategy_signals[strategy]
                    if i < len(signals):
                        # Get strategy position and signal
                        position = signals['Position'].iloc[i] if 'Position' in signals.columns else 0
                        signal = signals['Signal'].iloc[i] if 'Signal' in signals.columns else 'HOLD'
                        
                        # Weight by allocation
                        weighted_position = position * allocation
                        total_position += weighted_position
                        
                        # Count signal votes
                        signal_votes[signal] += allocation
            
            # Set portfolio position
            result.loc[result.index[i], 'Portfolio_Position'] = total_position
            
            # Set portfolio signal based on weighted votes
            dominant_signal = max(signal_votes, key=signal_votes.get)
            result.loc[result.index[i], 'Portfolio_Signal'] = dominant_signal
        
        return result
    
    def _create_empty_portfolio_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create empty portfolio signals DataFrame."""
        result = data.copy()
        result['Portfolio_Signal'] = 'HOLD'
        result['Portfolio_Position'] = 0.0
        result['Current_Regime'] = 'sideways_calm'
        result['Rebalance_Count'] = 0
        
        return result
    
    def _create_default_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create default signals for failed strategy."""
        result = data.copy()
        result['Signal'] = 'HOLD'
        result['Position'] = 0
        
        return result
    
    def get_portfolio_summary(self, signals: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive portfolio summary.
        
        Parameters
        ----------
        signals : pd.DataFrame
            Portfolio signals DataFrame
            
        Returns
        -------
        Dict[str, Any]
            Portfolio performance and allocation summary
        """
        summary = {
            'portfolio_name': 'Multi-Strategy Portfolio',
            'total_signals': len(signals),
            'rebalance_count': self.rebalance_count,
            'current_allocations': self.allocations.copy(),
            'initial_capital': self.initial_capital
        }
        
        # Signal distribution
        if 'Portfolio_Signal' in signals.columns:
            signal_counts = signals['Portfolio_Signal'].value_counts()
            summary['signal_distribution'] = signal_counts.to_dict()
        
        # Position statistics
        if 'Portfolio_Position' in signals.columns:
            position_stats = signals['Portfolio_Position'].describe()
            summary['position_stats'] = position_stats.to_dict()
        
        # Regime distribution
        if 'Current_Regime' in signals.columns:
            regime_counts = signals['Current_Regime'].value_counts()
            summary['regime_distribution'] = regime_counts.to_dict()
        
        # Strategy allocation efficiency
        allocation_efficiency = self._calculate_allocation_efficiency()
        summary['allocation_efficiency'] = allocation_efficiency
        
        return summary
    
    def _calculate_allocation_efficiency(self) -> Dict[str, float]:
        """Calculate allocation efficiency metrics."""
        efficiency = {}
        
        # Diversification score (1 - Herfindahl index)
        herfindahl = sum(allocation ** 2 for allocation in self.allocations.values())
        efficiency['diversification_score'] = 1 - herfindahl
        
        # Active allocation count
        active_count = sum(1 for allocation in self.allocations.values() if allocation > 0.01)
        efficiency['active_strategies'] = active_count
        
        # Concentration risk (max allocation)
        efficiency['max_concentration'] = max(self.allocations.values())
        
        return efficiency
    
    def update_allocation(self, strategy: str, new_allocation: float):
        """
        Manually update strategy allocation.
        
        Parameters
        ----------
        strategy : str
            Strategy name to update
        new_allocation : float
            New allocation weight (0-1)
        """
        if strategy in self.allocations:
            old_allocation = self.allocations[strategy]
            self.allocations[strategy] = new_allocation
            
            # Renormalize other allocations
            other_total = sum(alloc for name, alloc in self.allocations.items() if name != strategy)
            if other_total > 0:
                scale_factor = (1.0 - new_allocation) / other_total
                for name in self.allocations:
                    if name != strategy:
                        self.allocations[name] *= scale_factor
            
            logger.info(f"Updated {strategy} allocation: {old_allocation:.3f} -> {new_allocation:.3f}")
        else:
            logger.warning(f"Unknown strategy: {strategy}")
    
    def get_allocation_history(self) -> pd.DataFrame:
        """Get historical allocation changes."""
        # This would be expanded to track allocation history over time
        return pd.DataFrame([self.allocations])
    
    def get_strategy_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """Get individual strategy performance summary."""
        performance_summary = {}
        
        for strategy_name in self.strategies.keys():
            if strategy_name in self.strategy_performance:
                perf_data = self.strategy_performance[strategy_name]
                if perf_data:
                    performance_summary[strategy_name] = {
                        'avg_performance': np.mean(perf_data),
                        'volatility': np.std(perf_data),
                        'sharpe_ratio': np.mean(perf_data) / (np.std(perf_data) + 1e-8),
                        'total_observations': len(perf_data)
                    }
        
        return performance_summary