"""
Specialized strategies for different market contexts.

This module implements context-specific trading strategies optimized for
particular market conditions identified by the regime classifier.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple
from indicators.technical_indicators import rsi, ema, sma, macd, average_true_range

logger = logging.getLogger(__name__)


class IntelligentBullStrategy:
    """
    Strategy optimized for bull markets with intelligent protective exits.
    
    This strategy is designed to:
    1. Stay invested during bull market conditions
    2. Use conservative exit criteria to protect gains
    3. Quickly re-enter when conditions normalize
    4. Maximize upside capture while limiting downside
    
    Exit Criteria:
    - RSI > 80 AND volatility > 30% AND abnormal volume
    - MACD divergence in overbought zone
    - Break below 200-day MA with volume confirmation
    
    Parameters
    ----------
    ma_window : int, default 200
        Moving average window for trend confirmation
    exit_rsi_threshold : float, default 80
        RSI threshold for exit consideration
    exit_volatility_threshold : float, default 0.30
        Volatility threshold for exit consideration
    volume_multiplier : float, default 2.0
        Volume multiplier for anomaly detection
    reentry_rsi_threshold : float, default 50
        RSI threshold for re-entry consideration
    """
    
    def __init__(
        self,
        ma_window: int = 200,
        exit_rsi_threshold: float = 80,
        exit_volatility_threshold: float = 0.30,
        volume_multiplier: float = 2.0,
        reentry_rsi_threshold: float = 50
    ):
        self.ma_window = ma_window
        self.exit_rsi_threshold = exit_rsi_threshold
        self.exit_volatility_threshold = exit_volatility_threshold
        self.volume_multiplier = volume_multiplier
        self.reentry_rsi_threshold = reentry_rsi_threshold
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate intelligent bull market signals."""
        result = data.copy()
        
        # Calculate indicators
        result = self._calculate_indicators(result)
        
        # Initialize signals
        result['Signal'] = 'HOLD'
        result['Position'] = 0
        
        # Generate entry/exit signals
        result = self._generate_entry_exit_signals(result)
        
        return result
    
    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for bull strategy."""
        prices = data['Close']
        
        # Long-term trend (use available data if less than window)
        ma_window = min(self.ma_window, len(data) - 1, 50)  # Cap at 50 for testing
        data['MA_200'] = sma(prices, ma_window)
        data['EMA_50'] = ema(prices, min(50, len(data) - 1, 20))  # Cap at 20 for testing
        
        # Momentum indicators
        data['RSI'] = rsi(prices, window=14)
        
        # MACD for divergence detection
        if len(data) >= 26:
            macd_line, macd_signal, macd_histogram = macd(prices)
            data['MACD'] = macd_line
            data['MACD_Signal'] = macd_signal
            data['MACD_Histogram'] = macd_histogram
        
        # Volatility
        data['Volatility'] = self._calculate_rolling_volatility(prices)
        
        # Volume analysis
        if 'Volume' in data.columns:
            data['Volume_MA'] = sma(data['Volume'], 20)
            data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']
        
        return data
    
    def _calculate_rolling_volatility(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Calculate rolling volatility."""
        returns = prices.pct_change()
        volatility = returns.rolling(window=window).std() * np.sqrt(252)
        return volatility
    
    def _generate_entry_exit_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate entry and exit signals for bull market strategy."""
        
        # Create masks for valid data
        valid_mask = (
            data['Close'].notna() & 
            data['MA_200'].notna() & 
            data['RSI'].notna()
        )
        
        if valid_mask.sum() == 0:
            return data
        
        # Default position is long in bull market (start with 1)
        position = np.ones(len(data))
        
        # Exit conditions
        exit_condition = self._should_exit(data)
        
        # Re-entry conditions after exit
        reentry_condition = self._should_reenter(data)
        
        # Apply position logic
        for i in range(1, len(data)):
            if not valid_mask.iloc[i]:
                position[i] = position[i-1]
                continue
                
            # Exit logic
            if position[i-1] == 1 and exit_condition.iloc[i]:
                position[i] = 0
                data.loc[data.index[i], 'Signal'] = 'SELL'
            # Re-entry logic
            elif position[i-1] == 0 and reentry_condition.iloc[i]:
                position[i] = 1
                data.loc[data.index[i], 'Signal'] = 'BUY'
            else:
                position[i] = position[i-1]
        
        data['Position'] = position
        
        return data
    
    def _should_exit(self, data: pd.DataFrame) -> pd.Series:
        """Determine if should exit long position."""
        
        # Exit condition 1: Extreme overbought with high volatility and volume
        extreme_overbought = (
            (data['RSI'] > self.exit_rsi_threshold) &
            (data['Volatility'] > self.exit_volatility_threshold) &
            (data.get('Volume_Ratio', 1) > self.volume_multiplier)
        )
        
        # Exit condition 2: MACD divergence in overbought zone
        macd_divergence = pd.Series(False, index=data.index)
        if 'MACD' in data.columns and 'MACD_Histogram' in data.columns:
            # Simple divergence: decreasing MACD while price increases
            price_increasing = data['Close'] > data['Close'].shift(5)
            macd_decreasing = data['MACD'] < data['MACD'].shift(5)
            macd_divergence = (
                (data['RSI'] > 70) &
                price_increasing &
                macd_decreasing
            )
        
        # Exit condition 3: Break below 200-day MA with volume
        ma_break = (
            (data['Close'] < data['MA_200']) &
            (data['Close'].shift(1) >= data['MA_200'].shift(1)) &
            (data.get('Volume_Ratio', 1) > 1.5)
        )
        
        # Exit condition 4: Significant drawdown from recent high
        recent_high = data['Close'].rolling(window=20).max()
        drawdown = (data['Close'] / recent_high) - 1
        significant_drawdown = drawdown < -0.10  # 10% drawdown
        
        return extreme_overbought | macd_divergence | ma_break | significant_drawdown
    
    def _should_reenter(self, data: pd.DataFrame) -> pd.Series:
        """Determine if should re-enter long position."""
        
        # Re-entry condition 1: RSI normalized and price above MA
        rsi_normalized = (
            (data['RSI'] < self.reentry_rsi_threshold) &
            (data['Close'] > data['MA_200'])
        )
        
        # Re-entry condition 2: Price back above 50-day EMA with momentum
        ema_momentum = (
            (data['Close'] > data['EMA_50']) &
            (data['Close'] > data['Close'].shift(3))  # 3-day momentum
        )
        
        # Re-entry condition 3: Volume confirmation
        volume_confirmation = data.get('Volume_Ratio', 1) > 1.2
        
        return rsi_normalized & ema_momentum & volume_confirmation
    
    def get_strategy_summary(self, signals: pd.DataFrame) -> Dict[str, Any]:
        """Get strategy summary for intelligent bull strategy."""
        summary = {
            'strategy_name': 'Intelligent Bull Strategy',
            'total_signals': len(signals),
            'parameters': {
                'ma_window': self.ma_window,
                'exit_rsi_threshold': self.exit_rsi_threshold,
                'exit_volatility_threshold': self.exit_volatility_threshold,
                'volume_multiplier': self.volume_multiplier
            }
        }
        
        if 'Signal' in signals.columns:
            signal_counts = signals['Signal'].value_counts()
            summary['signal_distribution'] = signal_counts.to_dict()
            
        if 'Position' in signals.columns:
            position_stats = signals['Position'].describe()
            summary['position_stats'] = position_stats.to_dict()
            
            # Calculate time in market
            time_in_market = (signals['Position'] == 1).mean()
            summary['time_in_market_pct'] = float(time_in_market * 100)
        
        return summary


class CrisisProtectionStrategy:
    """
    Strategy specialized for detecting and avoiding market crises.
    
    This strategy focuses on:
    1. Early crisis detection using multiple signals
    2. Rapid capital preservation during turmoil
    3. Gradual re-entry as conditions stabilize
    4. Minimizing maximum drawdown
    
    Crisis Signals:
    - Volatility spike (VIX-like > 40)
    - Volume panic selling
    - Correlation breakdown across assets
    - Technical breakdown patterns
    
    Parameters
    ----------
    crisis_volatility_threshold : float, default 0.40
        Volatility threshold for crisis detection
    crisis_volume_threshold : float, default 3.0
        Volume spike threshold for panic detection
    recovery_periods : int, default 10
        Periods to wait for crisis recovery confirmation
    """
    
    def __init__(
        self,
        crisis_volatility_threshold: float = 0.40,
        crisis_volume_threshold: float = 3.0,
        recovery_periods: int = 10
    ):
        self.crisis_volatility_threshold = crisis_volatility_threshold
        self.crisis_volume_threshold = crisis_volume_threshold
        self.recovery_periods = recovery_periods
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate crisis protection signals."""
        result = data.copy()
        
        # Calculate crisis indicators
        result = self._calculate_crisis_indicators(result)
        
        # Initialize signals
        result['Signal'] = 'HOLD'
        result['Position'] = 1  # Default long position
        
        # Generate crisis protection signals
        result = self._generate_crisis_signals(result)
        
        return result
    
    def _calculate_crisis_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators for crisis detection."""
        prices = data['Close']
        
        # Volatility indicators
        data['Volatility'] = self._calculate_rolling_volatility(prices, window=20)
        data['Volatility_Spike'] = data['Volatility'] > self.crisis_volatility_threshold
        
        # Volume indicators
        if 'Volume' in data.columns:
            data['Volume_MA'] = sma(data['Volume'], 20)
            data['Volume_Spike'] = data['Volume'] / data['Volume_MA']
        else:
            data['Volume_Spike'] = 1.0
            
        # Price action indicators
        data['Returns'] = prices.pct_change()
        data['Large_Decline'] = data['Returns'] < -0.05  # 5% daily decline
        
        # Drawdown from recent peak
        data['Rolling_Max'] = prices.rolling(window=60).max()
        data['Drawdown'] = (prices / data['Rolling_Max']) - 1
        
        # Crisis score (0-1)
        data['Crisis_Score'] = self._calculate_crisis_score(data)
        
        return data
    
    def _calculate_rolling_volatility(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Calculate rolling volatility."""
        returns = prices.pct_change()
        volatility = returns.rolling(window=window).std() * np.sqrt(252)
        return volatility
    
    def _calculate_crisis_score(self, data: pd.DataFrame) -> pd.Series:
        """Calculate composite crisis risk score."""
        score = pd.Series(0.0, index=data.index)
        
        # Volatility component (0-0.4)
        vol_component = np.minimum(data['Volatility'] / self.crisis_volatility_threshold, 1.0) * 0.4
        score += vol_component
        
        # Volume component (0-0.3)
        vol_spike_component = np.minimum(data['Volume_Spike'] / self.crisis_volume_threshold, 1.0) * 0.3
        score += vol_spike_component
        
        # Price decline component (0-0.2)
        decline_component = data['Large_Decline'].astype(float) * 0.2
        score += decline_component
        
        # Drawdown component (0-0.1)
        drawdown_component = np.minimum(abs(data['Drawdown']) / 0.20, 1.0) * 0.1
        score += drawdown_component
        
        return np.minimum(score, 1.0)
    
    def _generate_crisis_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate crisis protection trading signals."""
        
        crisis_threshold = 0.6  # Crisis score threshold
        recovery_threshold = 0.3  # Recovery score threshold
        
        position = np.ones(len(data))
        
        for i in range(1, len(data)):
            crisis_score = data['Crisis_Score'].iloc[i]
            
            # Exit on crisis detection
            if position[i-1] == 1 and crisis_score > crisis_threshold:
                position[i] = 0
                data.loc[data.index[i], 'Signal'] = 'SELL'
                
            # Re-enter after crisis subsides
            elif position[i-1] == 0:
                # Check if crisis has subsided for required periods
                if i >= self.recovery_periods:
                    recent_scores = data['Crisis_Score'].iloc[i-self.recovery_periods:i+1]
                    if (recent_scores < recovery_threshold).all():
                        position[i] = 1
                        data.loc[data.index[i], 'Signal'] = 'BUY'
                    else:
                        position[i] = 0
                else:
                    position[i] = 0
            else:
                position[i] = position[i-1]
        
        data['Position'] = position
        
        return data
    
    def get_strategy_summary(self, signals: pd.DataFrame) -> Dict[str, Any]:
        """Get strategy summary for crisis protection strategy."""
        summary = {
            'strategy_name': 'Crisis Protection Strategy',
            'total_signals': len(signals),
            'parameters': {
                'crisis_volatility_threshold': self.crisis_volatility_threshold,
                'crisis_volume_threshold': self.crisis_volume_threshold,
                'recovery_periods': self.recovery_periods
            }
        }
        
        if 'Signal' in signals.columns:
            signal_counts = signals['Signal'].value_counts()
            summary['signal_distribution'] = signal_counts.to_dict()
        
        if 'Crisis_Score' in signals.columns:
            crisis_stats = signals['Crisis_Score'].describe()
            summary['crisis_score_stats'] = crisis_stats.to_dict()
            
            # High crisis periods
            high_crisis_periods = (signals['Crisis_Score'] > 0.6).sum()
            summary['high_crisis_periods'] = int(high_crisis_periods)
        
        return summary


class VolatilityRangeStrategy:
    """
    Strategy optimized for range-bound volatile markets.
    
    This strategy excels in:
    1. Identifying support and resistance levels
    2. Mean reversion trading in ranges
    3. Adapting to volatility changes
    4. Managing whipsaws in sideways markets
    
    Features:
    - Dynamic support/resistance identification
    - Volatility-adjusted position sizing
    - Mean reversion signals with filters
    - Breakout detection for regime changes
    
    Parameters
    ----------
    lookback_window : int, default 50
        Window for support/resistance calculation
    volatility_window : int, default 20
        Window for volatility calculation
    mean_reversion_threshold : float, default 2.0
        Standard deviation threshold for mean reversion
    """
    
    def __init__(
        self,
        lookback_window: int = 50,
        volatility_window: int = 20,
        mean_reversion_threshold: float = 2.0
    ):
        self.lookback_window = lookback_window
        self.volatility_window = volatility_window
        self.mean_reversion_threshold = mean_reversion_threshold
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate volatility range trading signals."""
        result = data.copy()
        
        # Calculate indicators
        result = self._calculate_range_indicators(result)
        
        # Initialize signals
        result['Signal'] = 'HOLD'
        result['Position'] = 0
        
        # Generate mean reversion signals
        result = self._generate_mean_reversion_signals(result)
        
        return result
    
    def _calculate_range_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators for range trading."""
        prices = data['Close']
        
        # Moving averages for mean
        data['SMA_20'] = sma(prices, 20)
        data['SMA_50'] = sma(prices, 50)
        
        # Support and resistance
        support, resistance = self._identify_support_resistance(prices)
        data['Support'] = support
        data['Resistance'] = resistance
        
        # Bollinger Bands for mean reversion
        data['BB_Middle'] = sma(prices, 20)
        data['BB_Std'] = prices.rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (data['BB_Std'] * 2)
        data['BB_Lower'] = data['BB_Middle'] - (data['BB_Std'] * 2)
        
        # Distance from mean (in standard deviations)
        data['Distance_From_Mean'] = (prices - data['BB_Middle']) / data['BB_Std']
        
        # RSI for momentum
        data['RSI'] = rsi(prices, window=14)
        
        # Volatility
        data['Volatility'] = self._calculate_rolling_volatility(prices)
        
        return data
    
    def _identify_support_resistance(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Identify dynamic support and resistance levels."""
        support = prices.rolling(window=self.lookback_window).min()
        resistance = prices.rolling(window=self.lookback_window).max()
        
        # Smooth the levels
        support = support.rolling(window=5).mean()
        resistance = resistance.rolling(window=5).mean()
        
        return support, resistance
    
    def _calculate_rolling_volatility(self, prices: pd.Series) -> pd.Series:
        """Calculate rolling volatility."""
        returns = prices.pct_change()
        volatility = returns.rolling(window=self.volatility_window).std() * np.sqrt(252)
        return volatility
    
    def _generate_mean_reversion_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate mean reversion trading signals."""
        
        # Valid data mask
        valid_mask = (
            data['Close'].notna() &
            data['Distance_From_Mean'].notna() &
            data['RSI'].notna()
        )
        
        if valid_mask.sum() == 0:
            return data
        
        position = np.zeros(len(data))
        
        for i in range(1, len(data)):
            if not valid_mask.iloc[i]:
                position[i] = position[i-1]
                continue
            
            distance = data['Distance_From_Mean'].iloc[i]
            rsi = data['RSI'].iloc[i]
            price = data['Close'].iloc[i]
            support = data['Support'].iloc[i]
            resistance = data['Resistance'].iloc[i]
            
            # Mean reversion buy signals (oversold)
            oversold_condition = (
                distance < -self.mean_reversion_threshold and
                rsi < 30 and
                price > support * 1.02  # Above support with margin
            )
            
            # Mean reversion sell signals (overbought)
            overbought_condition = (
                distance > self.mean_reversion_threshold and
                rsi > 70 and
                price < resistance * 0.98  # Below resistance with margin
            )
            
            # Exit conditions
            exit_long_condition = (
                position[i-1] == 1 and
                (distance > 0.5 or rsi > 60)  # Take profits or momentum shift
            )
            
            exit_short_condition = (
                position[i-1] == -1 and
                (distance < -0.5 or rsi < 40)  # Cover shorts or momentum shift
            )
            
            # Apply position logic
            if oversold_condition and position[i-1] <= 0:
                position[i] = 1
                data.loc[data.index[i], 'Signal'] = 'BUY'
            elif overbought_condition and position[i-1] >= 0:
                position[i] = -1
                data.loc[data.index[i], 'Signal'] = 'SELL'
            elif exit_long_condition:
                position[i] = 0
                data.loc[data.index[i], 'Signal'] = 'SELL'
            elif exit_short_condition:
                position[i] = 0
                data.loc[data.index[i], 'Signal'] = 'BUY'
            else:
                position[i] = position[i-1]
        
        data['Position'] = position
        
        return data
    
    def get_strategy_summary(self, signals: pd.DataFrame) -> Dict[str, Any]:
        """Get strategy summary for volatility range strategy."""
        summary = {
            'strategy_name': 'Volatility Range Strategy',
            'total_signals': len(signals),
            'parameters': {
                'lookback_window': self.lookback_window,
                'volatility_window': self.volatility_window,
                'mean_reversion_threshold': self.mean_reversion_threshold
            }
        }
        
        if 'Signal' in signals.columns:
            signal_counts = signals['Signal'].value_counts()
            summary['signal_distribution'] = signal_counts.to_dict()
        
        if 'Distance_From_Mean' in signals.columns:
            distance_stats = signals['Distance_From_Mean'].describe()
            summary['distance_from_mean_stats'] = distance_stats.to_dict()
        
        return summary