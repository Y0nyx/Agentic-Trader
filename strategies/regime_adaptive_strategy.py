"""
Regime Adaptive Strategy Framework.

This module implements an intelligent strategy that automatically adapts
to different market regimes by switching between specialized sub-strategies
based on market conditions detected by the MarketRegimeClassifier.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional
from .market_regime_classifier import MarketRegimeClassifier
from .trend_following_strategy import TrendFollowingStrategy
from .buy_hold_plus_strategy import BuyHoldPlusStrategy
from .adaptive_ma_strategy import AdaptiveMovingAverageStrategy

logger = logging.getLogger(__name__)


class RegimeAdaptiveStrategy:
    """
    Adaptive strategy that switches between specialized strategies based on market regime.
    
    This strategy uses the MarketRegimeClassifier to detect current market conditions
    and automatically selects the most appropriate sub-strategy for optimal performance.
    
    Strategy Mapping:
    - strong_bull: BuyHoldPlusStrategy (capture upside, exit on crisis)
    - moderate_bull: TrendFollowingStrategy (follow trends with filters)
    - sideways_volatile: BollingerMeanReversionStrategy (mean reversion)
    - sideways_calm: BreakoutStrategy (await breakouts)
    - bear_market: DefensiveStrategy (capital protection)
    - crisis_mode: CashStrategy (capital preservation)
    
    Parameters
    ----------
    regime_classifier : MarketRegimeClassifier, optional
        Market regime classifier instance
    strategy_configs : Dict[str, Dict], optional
        Configuration parameters for each strategy
    regime_memory : int, default 5
        Number of periods to remember regime changes (stability)
    confidence_threshold : float, default 0.6
        Minimum confidence required for regime change
    """
    
    def __init__(
        self,
        regime_classifier: Optional[MarketRegimeClassifier] = None,
        strategy_configs: Optional[Dict[str, Dict]] = None,
        regime_memory: int = 5,
        confidence_threshold: float = 0.6
    ):
        self.regime_classifier = regime_classifier or MarketRegimeClassifier()
        self.regime_memory = regime_memory
        self.confidence_threshold = confidence_threshold
        
        # Initialize strategy configurations
        self.strategy_configs = strategy_configs or self._get_default_configs()
        
        # Initialize sub-strategies
        self.strategies = self._initialize_strategies()
        
        # Regime tracking
        self.regime_history = []
        self.current_regime = None
        self.active_strategy = None
        self.regime_changes = 0
        
    def _get_default_configs(self) -> Dict[str, Dict]:
        """Get default configuration for each strategy type."""
        return {
            'strong_bull': {
                'strategy_type': 'buy_hold_plus',
                'exit_rsi_threshold': 80,
                'exit_drawdown_threshold': -0.10,
                'reentry_rsi_threshold': 30
            },
            'moderate_bull': {
                'strategy_type': 'trend_following',
                'trend_window': 50,
                'confirmation_window': 20,
                'min_trend_strength': 25
            },
            'sideways_volatile': {
                'strategy_type': 'adaptive_ma',
                'fast_period': 10,
                'slow_period': 30,
                'rsi_threshold': 70
            },
            'sideways_calm': {
                'strategy_type': 'adaptive_ma',
                'fast_period': 15,
                'slow_period': 45,
                'rsi_threshold': 60
            },
            'bear_market': {
                'strategy_type': 'defensive',
                'max_position': 0.3,  # Limited exposure
                'stop_loss': -0.05
            },
            'crisis_mode': {
                'strategy_type': 'cash',
                'position': 0  # Cash only
            }
        }
    
    def _initialize_strategies(self) -> Dict[str, Any]:
        """Initialize sub-strategies for each regime."""
        strategies = {}
        
        # Strong bull - Buy and hold with protection
        strategies['strong_bull'] = BuyHoldPlusStrategy(
            stress_rsi_threshold=25,
            drawdown_threshold=0.10  # 10% drawdown threshold (positive value)
        )
        
        # Moderate bull - Trend following
        strategies['moderate_bull'] = TrendFollowingStrategy(
            trend_window=self.strategy_configs['moderate_bull']['trend_window'],
            confirmation_window=self.strategy_configs['moderate_bull']['confirmation_window'],
            min_trend_strength=self.strategy_configs['moderate_bull']['min_trend_strength']
        )
        
        # Sideways volatile - Adaptive MA (more responsive)
        strategies['sideways_volatile'] = AdaptiveMovingAverageStrategy(
            fast_period=self.strategy_configs['sideways_volatile']['fast_period'],
            slow_period=self.strategy_configs['sideways_volatile']['slow_period']
        )
        
        # Sideways calm - Adaptive MA (less responsive)
        strategies['sideways_calm'] = AdaptiveMovingAverageStrategy(
            fast_period=self.strategy_configs['sideways_calm']['fast_period'],
            slow_period=self.strategy_configs['sideways_calm']['slow_period']
        )
        
        # Bear market - Conservative adaptive MA
        strategies['bear_market'] = AdaptiveMovingAverageStrategy(
            fast_period=20,
            slow_period=50
        )
        
        # Crisis mode - Create simple cash strategy
        strategies['crisis_mode'] = CashStrategy()
        
        return strategies
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on adaptive regime strategy.
        
        Parameters
        ----------
        data : pd.DataFrame
            Market data with OHLCV columns
            
        Returns
        -------
        pd.DataFrame
            DataFrame with signals and regime information
        """
        if len(data) < 50:  # Minimum data requirement
            logger.warning("Insufficient data for regime adaptive strategy")
            return self._create_empty_signals(data)
        
        # Classify current market regime
        regime_summary = self.regime_classifier.get_regime_summary(data)
        new_regime = regime_summary['current_regime']
        confidence = regime_summary['regime_confidence']
        
        # Update regime tracking
        self._update_regime_tracking(new_regime, confidence)
        
        # Get active strategy based on stable regime
        active_strategy = self._get_active_strategy()
        
        # Generate signals using active strategy
        signals = active_strategy.generate_signals(data)
        
        # Add regime information to signals
        signals = self._add_regime_info(signals, regime_summary)
        
        return signals
    
    def _update_regime_tracking(self, new_regime: str, confidence: float):
        """Update regime tracking with stability filters."""
        self.regime_history.append({
            'regime': new_regime,
            'confidence': confidence,
            'timestamp': pd.Timestamp.now()
        })
        
        # Keep only recent history
        if len(self.regime_history) > self.regime_memory * 2:
            self.regime_history = self.regime_history[-self.regime_memory * 2:]
        
        # Determine stable regime
        if confidence >= self.confidence_threshold:
            # Check if regime is stable over recent periods
            recent_regimes = [h['regime'] for h in self.regime_history[-self.regime_memory:]]
            regime_counts = {}
            for regime in recent_regimes:
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
            # Use most frequent regime if it appears in majority of recent periods
            if recent_regimes:
                most_frequent = max(regime_counts, key=regime_counts.get)
                if regime_counts[most_frequent] >= self.regime_memory // 2:
                    if self.current_regime != most_frequent:
                        logger.info(f"Regime change: {self.current_regime} -> {most_frequent}")
                        self.current_regime = most_frequent
                        self.regime_changes += 1
        
        # Initialize regime if not set
        if self.current_regime is None:
            self.current_regime = new_regime
    
    def _get_active_strategy(self) -> Any:
        """Get the strategy for current regime."""
        if self.current_regime in self.strategies:
            self.active_strategy = self.strategies[self.current_regime]
        else:
            # Default to moderate bull strategy
            self.active_strategy = self.strategies['moderate_bull']
            
        return self.active_strategy
    
    def _add_regime_info(self, signals: pd.DataFrame, regime_summary: Dict) -> pd.DataFrame:
        """Add regime information to signals DataFrame."""
        signals['Current_Regime'] = regime_summary['current_regime']
        signals['Regime_Confidence'] = regime_summary['regime_confidence']
        signals['Strategy_Type'] = self.strategy_configs.get(
            regime_summary['current_regime'], {}
        ).get('strategy_type', 'unknown')
        
        # Add regime features as additional columns
        for feature_name, value in regime_summary['features'].items():
            signals[f'Regime_{feature_name}'] = value
            
        return signals
    
    def _create_empty_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create empty signals DataFrame with required columns."""
        result = data.copy()
        result['Signal'] = 'HOLD'
        result['Position'] = 0
        result['Current_Regime'] = 'sideways_calm'
        result['Regime_Confidence'] = 0.0
        result['Strategy_Type'] = 'conservative'
        
        return result
    
    def get_strategy_summary(self, signals: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive strategy summary including regime analysis.
        
        Parameters
        ----------
        signals : pd.DataFrame
            Strategy signals with regime information
            
        Returns
        -------
        Dict[str, Any]
            Strategy performance and regime summary
        """
        summary = {
            'strategy_name': 'Regime Adaptive Strategy',
            'total_signals': len(signals),
            'regime_changes': self.regime_changes,
            'current_regime': self.current_regime,
            'active_strategy_type': getattr(self.active_strategy, '__class__', {}).get('__name__', 'Unknown')
        }
        
        # Regime distribution
        if 'Current_Regime' in signals.columns:
            regime_counts = signals['Current_Regime'].value_counts()
            summary['regime_distribution'] = regime_counts.to_dict()
        
        # Signal distribution
        if 'Signal' in signals.columns:
            signal_counts = signals['Signal'].value_counts()
            summary['signal_distribution'] = signal_counts.to_dict()
        
        # Average regime confidence
        if 'Regime_Confidence' in signals.columns:
            summary['avg_regime_confidence'] = float(signals['Regime_Confidence'].mean())
        
        # Get active strategy summary if available
        if hasattr(self.active_strategy, 'get_strategy_summary'):
            try:
                active_summary = self.active_strategy.get_strategy_summary(signals)
                summary['active_strategy_summary'] = active_summary
            except Exception as e:
                logger.warning(f"Could not get active strategy summary: {e}")
        
        return summary
    
    def configure_strategy(self, regime: str, config: Dict[str, Any]):
        """
        Configure parameters for a specific regime strategy.
        
        Parameters
        ----------
        regime : str
            Market regime to configure
        config : Dict[str, Any]
            Configuration parameters
        """
        if regime in self.strategy_configs:
            self.strategy_configs[regime].update(config)
            # Reinitialize strategies with new config
            self.strategies = self._initialize_strategies()
            logger.info(f"Updated configuration for {regime} regime")
        else:
            logger.warning(f"Unknown regime: {regime}")
    
    def get_regime_history(self) -> pd.DataFrame:
        """Get regime history as DataFrame."""
        if not self.regime_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.regime_history)


class CashStrategy:
    """
    Simple cash preservation strategy for crisis periods.
    
    This strategy maintains cash position (no trades) during crisis conditions
    to preserve capital until market conditions improve.
    """
    
    def __init__(self):
        self.strategy_name = "Cash Preservation Strategy"
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate cash-only signals (no trades)."""
        result = data.copy()
        result['Signal'] = 'HOLD'
        result['Position'] = 0  # Stay in cash
        
        return result
    
    def get_strategy_summary(self, signals: pd.DataFrame) -> Dict[str, Any]:
        """Get cash strategy summary."""
        return {
            'strategy_name': self.strategy_name,
            'total_signals': len(signals),
            'buy_signals': 0,
            'sell_signals': 0,
            'hold_signals': len(signals),
            'description': 'Capital preservation - no trades during crisis'
        }