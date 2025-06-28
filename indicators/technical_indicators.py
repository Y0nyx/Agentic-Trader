"""
Technical indicators for trading strategy enhancement.

This module implements common technical indicators used in trading strategies
to improve signal quality and reduce false positives.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def sma(data: pd.Series, window: int) -> pd.Series:
    """
    Simple Moving Average.
    
    Parameters
    ----------
    data : pd.Series
        Price data
    window : int
        Period for the moving average
        
    Returns
    -------
    pd.Series
        Simple moving average
    """
    return data.rolling(window=window, min_periods=window).mean()


def ema(data: pd.Series, window: int) -> pd.Series:
    """
    Exponential Moving Average.
    
    Parameters
    ----------
    data : pd.Series
        Price data
    window : int
        Period for the moving average
        
    Returns
    -------
    pd.Series
        Exponential moving average
    """
    return data.ewm(span=window, min_periods=window).mean()


def wma(data: pd.Series, window: int) -> pd.Series:
    """
    Weighted Moving Average.
    
    Parameters
    ----------
    data : pd.Series
        Price data
    window : int
        Period for the moving average
        
    Returns
    -------
    pd.Series
        Weighted moving average
    """
    weights = np.arange(1, window + 1)
    return data.rolling(window=window).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


def rsi(data: pd.Series, window: int = 14) -> pd.Series:
    """
    Relative Strength Index.
    
    Parameters
    ----------
    data : pd.Series
        Price data (typically close prices)
    window : int, default 14
        Period for RSI calculation
        
    Returns
    -------
    pd.Series
        RSI values (0-100)
    """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi_values = 100 - (100 / (1 + rs))
    
    return rsi_values


def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Moving Average Convergence Divergence.
    
    Parameters
    ----------
    data : pd.Series
        Price data (typically close prices)
    fast : int, default 12
        Fast EMA period
    slow : int, default 26
        Slow EMA period
    signal : int, default 9
        Signal line EMA period
        
    Returns
    -------
    Tuple[pd.Series, pd.Series, pd.Series]
        MACD line, Signal line, Histogram
    """
    ema_fast = ema(data, fast)
    ema_slow = ema(data, slow)
    
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def bollinger_bands(data: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands.
    
    Parameters
    ----------
    data : pd.Series
        Price data (typically close prices)
    window : int, default 20
        Period for moving average and standard deviation
    num_std : float, default 2
        Number of standard deviations for bands
        
    Returns
    -------
    Tuple[pd.Series, pd.Series, pd.Series]
        Upper band, Middle band (SMA), Lower band
    """
    middle_band = sma(data, window)
    std = data.rolling(window=window).std()
    
    upper_band = middle_band + (std * num_std)
    lower_band = middle_band - (std * num_std)
    
    return upper_band, middle_band, lower_band


def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    On Balance Volume.
    
    Parameters
    ----------
    close : pd.Series
        Close prices
    volume : pd.Series
        Volume data
        
    Returns
    -------
    pd.Series
        On Balance Volume
    """
    price_change = close.diff()
    
    # Create direction series: 1 for up, -1 for down, 0 for unchanged
    direction = pd.Series(index=close.index, dtype=float)
    direction[price_change > 0] = 1
    direction[price_change < 0] = -1
    direction[price_change == 0] = 0
    
    # Calculate OBV
    obv = (direction * volume).cumsum()
    
    return obv


def average_true_range(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """
    Average True Range.
    
    Parameters
    ----------
    high : pd.Series
        High prices
    low : pd.Series
        Low prices
    close : pd.Series
        Close prices
    window : int, default 14
        Period for ATR calculation
        
    Returns
    -------
    pd.Series
        Average True Range
    """
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=window).mean()
    
    return atr


def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """
    Average Directional Index (trend strength indicator).
    
    Parameters
    ----------
    high : pd.Series
        High prices
    low : pd.Series
        Low prices
    close : pd.Series
        Close prices
    window : int, default 14
        Period for ADX calculation
        
    Returns
    -------
    pd.Series
        ADX values
    """
    # Calculate True Range
    atr_values = average_true_range(high, low, close, window)
    
    # Calculate Directional Movement
    high_diff = high.diff()
    low_diff = low.diff()
    
    plus_dm = pd.Series(index=high.index, dtype=float)
    minus_dm = pd.Series(index=high.index, dtype=float)
    
    plus_dm[(high_diff > low_diff) & (high_diff > 0)] = high_diff
    plus_dm[~((high_diff > low_diff) & (high_diff > 0))] = 0
    
    minus_dm[(low_diff > high_diff) & (low_diff > 0)] = low_diff
    minus_dm[~((low_diff > high_diff) & (low_diff > 0))] = 0
    
    # Calculate smoothed DM and TR
    plus_dm_smooth = plus_dm.rolling(window=window).mean()
    minus_dm_smooth = minus_dm.rolling(window=window).mean()
    
    # Calculate DI+ and DI-
    plus_di = 100 * (plus_dm_smooth / atr_values)
    minus_di = 100 * (minus_dm_smooth / atr_values)
    
    # Calculate DX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    
    # Calculate ADX
    adx_values = dx.rolling(window=window).mean()
    
    return adx_values


def adaptive_moving_average(data: pd.Series, fast_period: int = 2, slow_period: int = 30) -> pd.Series:
    """
    Adaptive Moving Average (Kaufman's AMA).
    
    Parameters
    ----------
    data : pd.Series
        Price data
    fast_period : int, default 2
        Fast period for smoothing constant
    slow_period : int, default 30
        Slow period for smoothing constant
        
    Returns
    -------
    pd.Series
        Adaptive Moving Average
    """
    window = 10  # Period for efficiency ratio calculation
    
    # Calculate price change and volatility
    price_change = abs(data.diff(window))
    volatility = abs(data.diff()).rolling(window=window).sum()
    
    # Calculate efficiency ratio
    efficiency_ratio = price_change / volatility
    
    # Calculate smoothing constant
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    
    smoothing_constant = (efficiency_ratio * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate AMA
    # Initialize AMA with the first value
    ama = data.copy()
    ama.iloc[0] = data.iloc[0]
    
    # Calculate AMA iteratively using vectorized operations
    ama = ama.shift(1) + smoothing_constant * (data - ama.shift(1))
    ama.iloc[0] = data.iloc[0]  # Ensure the first value is correct
    
    return ama