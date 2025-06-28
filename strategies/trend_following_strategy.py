"""
Trend Following Strategy.

This strategy uses fewer signals but aims to capture long-term trends
by staying in positions longer and using broader confirmation signals.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any
from indicators.technical_indicators import ema, rsi, adx

logger = logging.getLogger(__name__)


class TrendFollowingStrategy:
    """
    Trend Following Strategy for long-term trend capture.
    
    This strategy is designed to beat buy-and-hold by:
    1. Using long-term trend identification
    2. Minimizing trading frequency
    3. Staying in trends longer
    4. Using conservative exit signals
    
    Signal Generation Logic:
    - BUY: Long-term uptrend confirmed with multiple signals
    - SELL: Clear trend reversal or overbought conditions
    - Focus on capturing major trend moves
    
    Parameters
    ----------
    trend_window : int, default 50
        Period for trend identification
    confirmation_window : int, default 20
        Period for trend confirmation
    price_column : str, default 'Close'
        Price column to use
    min_trend_strength : float, default 25
        Minimum ADX for trend confirmation
    exit_rsi_threshold : float, default 80
        RSI threshold for profit-taking
    """
    
    def __init__(
        self,
        trend_window: int = 50,
        confirmation_window: int = 20,
        price_column: str = "Close",
        min_trend_strength: float = 25,
        exit_rsi_threshold: float = 80,
    ):
        if trend_window <= 0 or confirmation_window <= 0:
            raise ValueError("Window periods must be positive integers")
        
        if confirmation_window >= trend_window:
            raise ValueError("Confirmation window must be smaller than trend window")
            
        self.trend_window = trend_window
        self.confirmation_window = confirmation_window
        self.price_column = price_column
        self.min_trend_strength = min_trend_strength
        self.exit_rsi_threshold = exit_rsi_threshold
        
        logger.info(
            f"Initialized TrendFollowingStrategy with trend_window={trend_window}, "
            f"confirmation_window={confirmation_window}, min_trend_strength={min_trend_strength}"
        )
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trend-following indicators.
        
        Parameters
        ----------
        data : pd.DataFrame
            Financial data with OHLCV columns
            
        Returns
        -------
        pd.DataFrame
            Data with added indicator columns
        """
        result = data.copy()
        
        if self.price_column not in data.columns:
            raise KeyError(f"Price column '{self.price_column}' not found in data")
        
        prices = data[self.price_column]
        
        # Long-term trend EMA
        result["EMA_trend"] = ema(prices, self.trend_window)
        
        # Confirmation EMA (shorter)
        result["EMA_confirm"] = ema(prices, self.confirmation_window)
        
        # Price momentum
        result["Price_momentum"] = prices / prices.shift(self.confirmation_window) - 1
        
        # RSI for exit signals
        result["RSI"] = rsi(prices, window=14)
        
        # ADX for trend strength
        if all(col in data.columns for col in ["High", "Low"]):
            result["ADX"] = adx(data["High"], data["Low"], prices, window=14)
        
        # Trend slope (rate of change of trend EMA)
        result["Trend_slope"] = result["EMA_trend"].pct_change(periods=5)
        
        return result
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trend-following signals with minimal frequency.
        
        Parameters
        ----------
        data : pd.DataFrame
            Financial data with OHLCV columns and DatetimeIndex
            
        Returns
        -------
        pd.DataFrame
            Data with added signal columns
        """
        if data.empty:
            logger.warning("Empty data provided to generate_signals")
            return data
        
        # Calculate indicators
        result = self.calculate_indicators(data)
        
        # Initialize signal columns
        result["Signal"] = "HOLD"
        result["Position"] = 0
        result["Trend_Score"] = 0.0
        
        # Ensure sufficient data
        min_periods_needed = self.trend_window + 20
        if len(result) < min_periods_needed:
            logger.warning(
                f"Insufficient data for signal generation. Need at least {min_periods_needed} periods"
            )
            return result
        
        prices = result[self.price_column]
        ema_trend = result["EMA_trend"]
        ema_confirm = result["EMA_confirm"]
        rsi_values = result["RSI"]
        
        # Find valid data points
        valid_mask = (
            prices.notna() & ema_trend.notna() & ema_confirm.notna() & rsi_values.notna()
        )
        
        if valid_mask.sum() == 0:
            logger.warning("No valid indicator data for signal generation")
            return result
        
        # Calculate trend score (combination of multiple factors)
        trend_score = pd.Series(0.0, index=result.index)
        
        # Factor 1: Price above long-term trend
        price_above_trend = (prices > ema_trend) & valid_mask
        trend_score += price_above_trend.astype(float) * 2
        
        # Factor 2: Confirmation EMA above trend EMA
        confirm_above_trend = (ema_confirm > ema_trend) & valid_mask
        trend_score += confirm_above_trend.astype(float) * 2
        
        # Factor 3: Price momentum
        if "Price_momentum" in result.columns:
            momentum = result["Price_momentum"]
            positive_momentum = (momentum > 0.02) & momentum.notna()  # At least 2% momentum
            trend_score += positive_momentum.astype(float) * 1
        
        # Factor 4: Trend slope
        if "Trend_slope" in result.columns:
            slope = result["Trend_slope"]
            positive_slope = (slope > 0) & slope.notna()
            trend_score += positive_slope.astype(float) * 1
        
        # Factor 5: ADX trend strength
        if "ADX" in result.columns:
            adx_values = result["ADX"]
            strong_trend = (adx_values > self.min_trend_strength) & adx_values.notna()
            trend_score += strong_trend.astype(float) * 1
        
        result["Trend_Score"] = trend_score
        
        # Current position tracking
        # Initialize Signal and Position columns
        result["Signal"] = np.nan
        result["Position"] = 0
        
        # BUY CONDITIONS (enter long position)
        buy_condition = (trend_score >= 5) & (rsi_values < 70) & valid_mask
        
        # SELL CONDITIONS (exit long position)
        sell_condition = (
            (trend_score <= 2) |  # Trend weakening significantly
            (rsi_values > self.exit_rsi_threshold)  # Very overbought
        ) & valid_mask
        
        # Compute positions using cumulative logic
        result["Position"] = np.where(
            buy_condition, 1,  # Enter long position
            np.where(sell_condition, -1, np.nan)  # Exit long position
        )
        
        # Forward-fill positions to maintain state
        result["Position"] = result["Position"].ffill().fillna(0)
        
        # Generate signals based on position changes
        result["Signal"] = np.where(
            (result["Position"] == 1) & (result["Position"].shift(1) == 0), "BUY",
            np.where(
                (result["Position"] == -1) & (result["Position"].shift(1) == 1), "SELL",
                np.nan
            )
        )
        
        # Count signals
        buy_signals = (result["Signal"] == "BUY").sum()
        sell_signals = (result["Signal"] == "SELL").sum()
        
        logger.info(f"Generated {buy_signals} BUY and {sell_signals} SELL signals")
        
        return result
    
    def get_strategy_summary(self, signals: pd.DataFrame) -> Dict[str, Any]:
        """
        Get a summary of the strategy configuration and performance.
        
        Parameters
        ----------
        signals : pd.DataFrame
            DataFrame with generated signals
            
        Returns
        -------
        Dict[str, Any]
            Strategy summary
        """
        if signals.empty:
            return {}
        
        signal_counts = signals["Signal"].value_counts()
        
        return {
            "strategy_type": "TrendFollowing",
            "parameters": {
                "trend_window": self.trend_window,
                "confirmation_window": self.confirmation_window,
                "price_column": self.price_column,
                "min_trend_strength": self.min_trend_strength,
                "exit_rsi_threshold": self.exit_rsi_threshold,
            },
            "signals_generated": {
                "BUY": signal_counts.get("BUY", 0),
                "SELL": signal_counts.get("SELL", 0),
                "HOLD": signal_counts.get("HOLD", 0),
                "total": len(signals),
            },
            "signal_frequency": {
                "buy_frequency": signal_counts.get("BUY", 0) / len(signals),
                "sell_frequency": signal_counts.get("SELL", 0) / len(signals),
            },
        }