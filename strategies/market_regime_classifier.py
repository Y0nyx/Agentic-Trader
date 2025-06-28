"""
Market Regime Classification System.

This module provides intelligent classification of market regimes to enable
adaptive trading strategies that can respond to different market conditions.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple
from indicators.technical_indicators import ema, sma, rsi, average_true_range

logger = logging.getLogger(__name__)


class MarketRegimeClassifier:
    """
    Intelligent market regime classifier for adaptive trading strategies.
    
    Classifies market conditions into distinct regimes to enable
    strategy adaptation based on market characteristics.
    
    Market Regimes:
    - 'strong_bull': Strong uptrend (>15% annual, volatility < 20%)
    - 'moderate_bull': Moderate uptrend 
    - 'sideways_volatile': Range-bound with high volatility
    - 'sideways_calm': Range-bound with low volatility
    - 'bear_market': Downtrend
    - 'crisis_mode': Extreme volatility + significant decline
    
    Parameters
    ----------
    trend_window : int, default 252
        Period for trend analysis (trading days in year)
    volatility_window : int, default 60
        Period for volatility calculation
    volume_window : int, default 20
        Period for volume analysis
    """
    
    def __init__(
        self,
        trend_window: int = 252,
        volatility_window: int = 60,
        volume_window: int = 20
    ):
        self.trend_window = trend_window
        self.volatility_window = volatility_window
        self.volume_window = volume_window
        
        # Regime thresholds
        self.strong_bull_return_threshold = 0.15  # 15% annual return
        self.moderate_bull_return_threshold = 0.05  # 5% annual return
        self.bear_market_threshold = -0.05  # -5% annual return
        
        self.low_volatility_threshold = 0.15  # 15% annual volatility
        self.high_volatility_threshold = 0.30  # 30% annual volatility
        self.crisis_volatility_threshold = 0.50  # 50% annual volatility
        
        self.crisis_drawdown_threshold = -0.15  # -15% drawdown
        
    def classify_regime(self, data: pd.DataFrame) -> str:
        """
        Classify current market regime based on recent data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Market data with OHLCV columns
            
        Returns
        -------
        str
            Current market regime classification
        """
        if len(data) < max(self.trend_window, self.volatility_window):
            logger.warning("Insufficient data for regime classification")
            return "sideways_calm"  # Default conservative regime
            
        features = self.get_regime_features(data)
        
        # Extract key features
        trend_slope = features['trend_slope']
        volatility = features['volatility_annualized']
        drawdown = features['current_drawdown']
        volume_anomaly = features['volume_anomaly']
        
        # Crisis detection (highest priority)
        if (volatility > self.crisis_volatility_threshold or 
            drawdown < self.crisis_drawdown_threshold or
            volume_anomaly > 3.0):  # 3x normal volume
            return "crisis_mode"
            
        # Bear market detection
        if trend_slope < self.bear_market_threshold:
            return "bear_market"
            
        # Bull market detection
        if trend_slope > self.strong_bull_return_threshold:
            if volatility < self.low_volatility_threshold:
                return "strong_bull"
            else:
                return "moderate_bull"
        elif trend_slope > self.moderate_bull_return_threshold:
            return "moderate_bull"
            
        # Sideways market classification
        if volatility > self.high_volatility_threshold:
            return "sideways_volatile"
        else:
            return "sideways_calm"
    
    def get_regime_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate features for regime classification.
        
        Parameters
        ----------
        data : pd.DataFrame
            Market data with OHLCV columns
            
        Returns
        -------
        Dict[str, float]
            Dictionary of regime classification features
        """
        prices = data['Close'].copy()
        
        # 1. Trend slope (annualized return via linear regression)
        trend_slope = self._calculate_trend_slope(prices)
        
        # 2. Annualized volatility (rolling)
        volatility_annualized = self._calculate_volatility(prices)
        
        # 3. VIX equivalent (normalized ATR)
        vix_equivalent = self._calculate_vix_equivalent(data)
        
        # 4. Volume profile analysis
        volume_anomaly = self._calculate_volume_anomaly(data)
        
        # 5. Current drawdown from recent peak
        current_drawdown = self._calculate_current_drawdown(prices)
        
        # 6. Price momentum (shorter-term trend strength)
        price_momentum = self._calculate_price_momentum(prices)
        
        return {
            'trend_slope': trend_slope,
            'volatility_annualized': volatility_annualized,
            'vix_equivalent': vix_equivalent,
            'volume_anomaly': volume_anomaly,
            'current_drawdown': current_drawdown,
            'price_momentum': price_momentum
        }
    
    def _calculate_trend_slope(self, prices: pd.Series) -> float:
        """Calculate annualized trend slope using linear regression."""
        if len(prices) < self.trend_window:
            return 0.0
            
        recent_prices = prices.tail(self.trend_window)
        
        # Linear regression on log prices for percentage returns
        log_prices = np.log(recent_prices)
        x = np.arange(len(log_prices))
        
        # Calculate slope (daily log return)
        slope = np.polyfit(x, log_prices, 1)[0]
        
        # Annualize the slope (252 trading days)
        annualized_slope = slope * 252
        
        return annualized_slope
    
    def _calculate_volatility(self, prices: pd.Series) -> float:
        """Calculate annualized volatility."""
        if len(prices) < self.volatility_window:
            return 0.0
            
        # Calculate daily returns
        returns = prices.pct_change().dropna()
        
        # Rolling volatility
        rolling_vol = returns.rolling(window=self.volatility_window).std()
        
        # Annualize volatility (sqrt(252) scaling)
        current_vol = rolling_vol.iloc[-1] if not rolling_vol.empty else 0.0
        annualized_vol = current_vol * np.sqrt(252)
        
        return annualized_vol
    
    def _calculate_vix_equivalent(self, data: pd.DataFrame) -> float:
        """Calculate VIX-equivalent using normalized ATR."""
        if not all(col in data.columns for col in ['High', 'Low', 'Close']):
            return 0.0
            
        try:
            atr = average_true_range(
                data['High'], data['Low'], data['Close'], window=14
            )
            
            # Normalize ATR by price and annualize
            atr_normalized = (atr / data['Close']) * np.sqrt(252) * 100
            
            return atr_normalized.iloc[-1] if not atr_normalized.empty else 0.0
        except Exception as e:
            logger.warning(f"Error calculating VIX equivalent: {e}")
            return 0.0
    
    def _calculate_volume_anomaly(self, data: pd.DataFrame) -> float:
        """Calculate volume anomaly compared to recent average."""
        if 'Volume' not in data.columns or len(data) < self.volume_window:
            return 1.0
            
        volume = data['Volume']
        
        # Calculate average volume and current volume ratio
        avg_volume = volume.rolling(window=self.volume_window).mean()
        current_volume = volume.iloc[-1]
        avg_volume_recent = avg_volume.iloc[-1]
        
        if avg_volume_recent > 0:
            volume_ratio = current_volume / avg_volume_recent
        else:
            volume_ratio = 1.0
            
        return volume_ratio
    
    def _calculate_current_drawdown(self, prices: pd.Series) -> float:
        """Calculate current drawdown from recent peak."""
        if len(prices) < 2:
            return 0.0
            
        # Calculate running maximum (peak)
        running_max = prices.rolling(window=len(prices), min_periods=1).max()
        
        # Calculate drawdown
        drawdown = (prices / running_max) - 1.0
        
        return drawdown.iloc[-1]
    
    def _calculate_price_momentum(self, prices: pd.Series) -> float:
        """Calculate short-term price momentum."""
        if len(prices) < 20:
            return 0.0
            
        # 20-day momentum
        momentum = (prices.iloc[-1] / prices.iloc[-20]) - 1.0
        
        return momentum
    
    def get_regime_summary(self, data: pd.DataFrame) -> Dict[str, any]:
        """
        Get comprehensive regime analysis summary.
        
        Parameters
        ----------
        data : pd.DataFrame
            Market data with OHLCV columns
            
        Returns
        -------
        Dict[str, any]
            Comprehensive regime analysis
        """
        current_regime = self.classify_regime(data)
        features = self.get_regime_features(data)
        
        return {
            'current_regime': current_regime,
            'features': features,
            'regime_confidence': self._calculate_regime_confidence(features),
            'regime_description': self._get_regime_description(current_regime),
            'suggested_strategy_type': self._get_suggested_strategy(current_regime)
        }
    
    def _calculate_regime_confidence(self, features: Dict[str, float]) -> float:
        """Calculate confidence score for regime classification."""
        # Simple confidence based on feature strength
        # This could be enhanced with ML models
        
        trend_strength = abs(features['trend_slope'])
        vol_clarity = abs(features['volatility_annualized'] - 0.20)  # Distance from neutral vol
        
        confidence = min(1.0, (trend_strength + vol_clarity) / 2.0)
        return confidence
    
    def _get_regime_description(self, regime: str) -> str:
        """Get human-readable description of regime."""
        descriptions = {
            'strong_bull': 'Strong uptrend with low volatility - favorable for trend following',
            'moderate_bull': 'Moderate uptrend - good for trend strategies with some caution',
            'sideways_volatile': 'Range-bound with high volatility - ideal for mean reversion',
            'sideways_calm': 'Range-bound with low volatility - suitable for breakout strategies',
            'bear_market': 'Downtrend - defensive strategies recommended',
            'crisis_mode': 'High volatility/crisis conditions - capital preservation priority'
        }
        return descriptions.get(regime, 'Unknown regime')
    
    def _get_suggested_strategy(self, regime: str) -> str:
        """Get suggested strategy type for regime."""
        strategy_mapping = {
            'strong_bull': 'buy_and_hold_plus',
            'moderate_bull': 'trend_following', 
            'sideways_volatile': 'mean_reversion',
            'sideways_calm': 'breakout',
            'bear_market': 'defensive',
            'crisis_mode': 'cash_preservation'
        }
        return strategy_mapping.get(regime, 'conservative')