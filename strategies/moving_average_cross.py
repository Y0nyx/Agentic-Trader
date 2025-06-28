"""
Moving Average Cross Strategy implementation.

This module implements a simple but robust trading strategy based on the crossover
of short and long period moving averages. The strategy generates clear BUY, SELL,
and HOLD signals based on the relationship between these moving averages.
"""

import logging
from typing import Optional
import pandas as pd
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


class MovingAverageCrossStrategy:
    """
    Moving Average Crossover Trading Strategy.
    
    This strategy generates trading signals based on the crossover of two
    moving averages with different periods (short and long).
    
    Signal Generation Logic:
    - BUY: Short MA crosses above Long MA (bullish crossover)
    - SELL: Short MA crosses below Long MA (bearish crossover)  
    - HOLD: No recent crossover or insufficient data
    
    Parameters
    ----------
    short_window : int, default 20
        Period for the short-term moving average
    long_window : int, default 50
        Period for the long-term moving average
    price_column : str, default 'Close'
        Price column to use for moving average calculation
    """
    
    def __init__(
        self,
        short_window: int = 20,
        long_window: int = 50,
        price_column: str = 'Close'
    ):
        if short_window <= 0 or long_window <= 0:
            raise ValueError("Window periods must be positive integers")
        
        if short_window >= long_window:
            raise ValueError("Short window must be smaller than long window")
        
        self.short_window = short_window
        self.long_window = long_window
        self.price_column = price_column
        
        logger.info(f"Initialized MovingAverageCrossStrategy with short={short_window}, long={long_window}")
    
    def calculate_moving_averages(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate short and long moving averages.
        
        Parameters
        ----------
        data : pd.DataFrame
            Financial data with OHLCV columns
            
        Returns
        -------
        pd.DataFrame
            Data with added moving average columns
        """
        if data.empty:
            return data
            
        if self.price_column not in data.columns:
            raise ValueError(f"Price column '{self.price_column}' not found in data")
        
        result = data.copy()
        
        # Calculate moving averages
        result[f'MA_{self.short_window}'] = result[self.price_column].rolling(
            window=self.short_window,
            min_periods=self.short_window
        ).mean()
        
        result[f'MA_{self.long_window}'] = result[self.price_column].rolling(
            window=self.long_window,
            min_periods=self.long_window
        ).mean()
        
        return result
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on moving average crossovers.
        
        Parameters
        ----------
        data : pd.DataFrame
            Financial data with OHLCV columns and DatetimeIndex
            
        Returns
        -------
        pd.DataFrame
            Data with added signal columns:
            - MA_{short_window}: Short-term moving average
            - MA_{long_window}: Long-term moving average
            - Signal: Trading signal ('BUY', 'SELL', 'HOLD')
            - Position: Recommended position (1 for long, -1 for short, 0 for neutral)
        """
        if data.empty:
            logger.warning("Empty data provided to generate_signals")
            return data
        
        # Calculate moving averages
        result = self.calculate_moving_averages(data)
        
        short_ma_col = f'MA_{self.short_window}'
        long_ma_col = f'MA_{self.long_window}'
        
        # Initialize signal columns
        result['Signal'] = 'HOLD'
        result['Position'] = 0
        
        # Ensure we have sufficient data for signals
        min_periods_needed = self.long_window
        if len(result) < min_periods_needed:
            logger.warning(f"Insufficient data for signal generation. Need at least {min_periods_needed} periods")
            return result
        
        # Calculate crossover conditions
        short_ma = result[short_ma_col]
        long_ma = result[long_ma_col]
        
        # Find valid data points (where both MAs are available)
        valid_mask = short_ma.notna() & long_ma.notna()
        
        if valid_mask.sum() == 0:
            logger.warning("No valid moving average data for signal generation")
            return result
        
        # Previous period comparison for crossover detection
        short_ma_prev = short_ma.shift(1)
        long_ma_prev = long_ma.shift(1)
        
        # Detect crossovers
        bullish_crossover = (
            (short_ma > long_ma) &  # Current: short above long
            (short_ma_prev <= long_ma_prev) &  # Previous: short below or equal to long
            valid_mask &  # Valid data
            valid_mask.shift(1)  # Previous period also valid
        )
        
        bearish_crossover = (
            (short_ma < long_ma) &  # Current: short below long
            (short_ma_prev >= long_ma_prev) &  # Previous: short above or equal to long
            valid_mask &  # Valid data
            valid_mask.shift(1)  # Previous period also valid
        )
        
        # Generate signals
        result.loc[bullish_crossover, 'Signal'] = 'BUY'
        result.loc[bullish_crossover, 'Position'] = 1
        
        result.loc[bearish_crossover, 'Signal'] = 'SELL'
        result.loc[bearish_crossover, 'Position'] = -1
        
        # Count signals generated
        buy_signals = (result['Signal'] == 'BUY').sum()
        sell_signals = (result['Signal'] == 'SELL').sum()
        
        logger.info(f"Generated {buy_signals} BUY and {sell_signals} SELL signals")
        
        return result
    
    def get_strategy_summary(self, signals_data: pd.DataFrame) -> dict:
        """
        Get a summary of the strategy performance on the given data.
        
        Parameters
        ----------
        signals_data : pd.DataFrame
            Data with generated signals
            
        Returns
        -------
        dict
            Summary statistics including signal counts and periods
        """
        if signals_data.empty:
            return {}
        
        summary = {
            'short_window': self.short_window,
            'long_window': self.long_window,
            'total_periods': len(signals_data),
            'buy_signals': (signals_data['Signal'] == 'BUY').sum(),
            'sell_signals': (signals_data['Signal'] == 'SELL').sum(),
            'hold_periods': (signals_data['Signal'] == 'HOLD').sum(),
        }
        
        # Calculate signal frequency
        total_signals = summary['buy_signals'] + summary['sell_signals']
        if summary['total_periods'] > 0:
            summary['signal_frequency'] = total_signals / summary['total_periods']
        else:
            summary['signal_frequency'] = 0
        
        # Add date range if available
        if isinstance(signals_data.index, pd.DatetimeIndex):
            summary['date_range'] = {
                'start': signals_data.index.min().strftime('%Y-%m-%d'),
                'end': signals_data.index.max().strftime('%Y-%m-%d')
            }
        
        return summary