"""
Adaptive Moving Average Strategy.

This strategy uses Kaufman's Adaptive Moving Average (AMA) which adjusts
its smoothing constant based on market efficiency and volatility.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any
from indicators.technical_indicators import adaptive_moving_average, rsi, adx

logger = logging.getLogger(__name__)


class AdaptiveMovingAverageStrategy:
    """
    Adaptive Moving Average Trading Strategy.
    
    This strategy uses Kaufman's Adaptive Moving Average which automatically
    adjusts to market conditions - faster in trending markets, slower in 
    ranging markets.
    
    Signal Generation Logic:
    - BUY: Price crosses above AMA and trend strength confirms
    - SELL: Price crosses below AMA and trend strength confirms
    - Additional filters can be applied for enhanced signal quality
    
    Parameters
    ----------
    fast_period : int, default 2
        Fast period for AMA calculation
    slow_period : int, default 30
        Slow period for AMA calculation
    price_column : str, default 'Close'
        Price column to use for calculation
    use_rsi_filter : bool, default True
        Whether to use RSI filter for signals
    rsi_oversold : float, default 30
        RSI oversold threshold
    rsi_overbought : float, default 70
        RSI overbought threshold
    use_trend_filter : bool, default True
        Whether to use ADX trend strength filter
    adx_threshold : float, default 25
        Minimum ADX value for trend confirmation
    """
    
    def __init__(
        self,
        fast_period: int = 2,
        slow_period: int = 30,
        price_column: str = "Close",
        use_rsi_filter: bool = True,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        use_trend_filter: bool = True,
        adx_threshold: float = 25,
    ):
        if fast_period <= 0 or slow_period <= 0:
            raise ValueError("Periods must be positive integers")
        
        if fast_period >= slow_period:
            raise ValueError("Fast period must be smaller than slow period")
        
        if not (0 <= rsi_oversold <= 100 and 0 <= rsi_overbought <= 100):
            raise ValueError("RSI thresholds must be between 0 and 100")
        
        if rsi_oversold >= rsi_overbought:
            raise ValueError("RSI oversold must be less than overbought")
            
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.price_column = price_column
        self.use_rsi_filter = use_rsi_filter
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.use_trend_filter = use_trend_filter
        self.adx_threshold = adx_threshold
        
        logger.info(
            f"Initialized AdaptiveMovingAverageStrategy with fast={fast_period}, "
            f"slow={slow_period}, RSI filter={use_rsi_filter}, trend filter={use_trend_filter}"
        )
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the Adaptive Moving Average and additional indicators.
        
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
        
        # Calculate Adaptive Moving Average
        result["AMA"] = adaptive_moving_average(prices, self.fast_period, self.slow_period)
        
        # Calculate additional indicators if enabled
        if self.use_rsi_filter:
            result["RSI"] = rsi(prices, window=14)
        
        if self.use_trend_filter:
            if all(col in data.columns for col in ["High", "Low"]):
                result["ADX"] = adx(data["High"], data["Low"], prices, window=14)
            else:
                logger.warning("High/Low columns not found, disabling ADX trend filter")
                self.use_trend_filter = False
        
        return result
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on Adaptive Moving Average.
        
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
        
        # Ensure we have sufficient data
        min_periods_needed = max(30, self.slow_period)  # Need enough for AMA and RSI
        if len(result) < min_periods_needed:
            logger.warning(
                f"Insufficient data for signal generation. Need at least {min_periods_needed} periods"
            )
            return result
        
        # Get price and AMA
        prices = result[self.price_column]
        ama = result["AMA"]
        
        # Find valid data points
        valid_mask = prices.notna() & ama.notna()
        
        if valid_mask.sum() == 0:
            logger.warning("No valid indicator data for signal generation")
            return result
        
        # Previous period values for crossover detection
        prices_prev = prices.shift(1)
        ama_prev = ama.shift(1)
        
        # Basic crossover conditions
        bullish_crossover = (
            (prices > ama) &  # Current: price above AMA
            (prices_prev <= ama_prev) &  # Previous: price below or equal AMA
            valid_mask &
            valid_mask.shift(1)
        )
        
        bearish_crossover = (
            (prices < ama) &  # Current: price below AMA
            (prices_prev >= ama_prev) &  # Previous: price above or equal AMA
            valid_mask &
            valid_mask.shift(1)
        )
        
        # Apply RSI filter if enabled
        if self.use_rsi_filter and "RSI" in result.columns:
            rsi_values = result["RSI"]
            rsi_valid = rsi_values.notna()
            
            # For buy signals: RSI should not be overbought
            # For sell signals: RSI should not be oversold
            bullish_crossover = bullish_crossover & (rsi_values < self.rsi_overbought) & rsi_valid
            bearish_crossover = bearish_crossover & (rsi_values > self.rsi_oversold) & rsi_valid
        
        # Apply trend strength filter if enabled
        if self.use_trend_filter and "ADX" in result.columns:
            adx_values = result["ADX"]
            adx_valid = adx_values.notna()
            
            # Only trade when trend strength is above threshold
            trend_strong = (adx_values > self.adx_threshold) & adx_valid
            bullish_crossover = bullish_crossover & trend_strong
            bearish_crossover = bearish_crossover & trend_strong
        
        # Generate signals
        result.loc[bullish_crossover, "Signal"] = "BUY"
        result.loc[bullish_crossover, "Position"] = 1
        
        result.loc[bearish_crossover, "Signal"] = "SELL"
        result.loc[bearish_crossover, "Position"] = -1
        
        # Count signals
        buy_signals = bullish_crossover.sum()
        sell_signals = bearish_crossover.sum()
        
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
            "strategy_type": "AdaptiveMovingAverage",
            "parameters": {
                "fast_period": self.fast_period,
                "slow_period": self.slow_period,
                "price_column": self.price_column,
                "use_rsi_filter": self.use_rsi_filter,
                "rsi_oversold": self.rsi_oversold,
                "rsi_overbought": self.rsi_overbought,
                "use_trend_filter": self.use_trend_filter,
                "adx_threshold": self.adx_threshold,
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