"""
Buy-and-Hold Plus Strategy.

This strategy aims to beat pure buy-and-hold by avoiding major drawdowns
while staying invested most of the time.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any
from indicators.technical_indicators import ema, rsi, sma

logger = logging.getLogger(__name__)


class BuyHoldPlusStrategy:
    """
    Buy-and-Hold Plus Strategy.
    
    This strategy tries to beat buy-and-hold by:
    1. Staying invested most of the time (like buy-and-hold)
    2. Only exiting during major market stress
    3. Re-entering quickly when conditions improve
    
    Signal Generation Logic:
    - Default: STAY INVESTED (like buy-and-hold)
    - SELL: Only during extreme market stress
    - BUY: Quick re-entry after stress passes
    
    Parameters
    ----------
    stress_rsi_threshold : float, default 25
        RSI threshold below which to sell (extreme oversold)
    reentry_rsi_threshold : float, default 40
        RSI threshold above which to re-enter
    drawdown_threshold : float, default 0.15
        Maximum drawdown before selling (15%)
    trend_window : int, default 50
        Period for trend assessment
    price_column : str, default 'Close'
        Price column to use
    """
    
    def __init__(
        self,
        stress_rsi_threshold: float = 25,
        reentry_rsi_threshold: float = 40,
        drawdown_threshold: float = 0.15,
        trend_window: int = 50,
        price_column: str = "Close",
    ):
        if not (0 <= stress_rsi_threshold <= 100 and 0 <= reentry_rsi_threshold <= 100):
            raise ValueError("RSI thresholds must be between 0 and 100")
        
        if stress_rsi_threshold >= reentry_rsi_threshold:
            raise ValueError("Stress RSI must be less than reentry RSI")
        
        if not (0 < drawdown_threshold < 1):
            raise ValueError("Drawdown threshold must be between 0 and 1")
            
        self.stress_rsi_threshold = stress_rsi_threshold
        self.reentry_rsi_threshold = reentry_rsi_threshold
        self.drawdown_threshold = drawdown_threshold
        self.trend_window = trend_window
        self.price_column = price_column
        
        logger.info(
            f"Initialized BuyHoldPlusStrategy with stress_rsi={stress_rsi_threshold}, "
            f"reentry_rsi={reentry_rsi_threshold}, drawdown_threshold={drawdown_threshold}"
        )
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate indicators for stress detection.
        
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
        
        # RSI for market stress detection
        result["RSI"] = rsi(prices, window=14)
        
        # Long-term trend
        result["EMA_trend"] = ema(prices, self.trend_window)
        
        # Rolling maximum for drawdown calculation
        result["Rolling_max"] = prices.rolling(window=252, min_periods=1).max()  # 1 year rolling max
        
        # Current drawdown from peak
        result["Current_drawdown"] = (prices - result["Rolling_max"]) / result["Rolling_max"]
        
        # Price volatility (for stress assessment)
        result["Volatility"] = prices.pct_change().rolling(window=20).std()
        
        return result
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy-and-hold plus signals.
        
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
        
        # Initialize signal columns - default to BUY (stay invested)
        result["Signal"] = "BUY"  # Default is to be invested
        result["Position"] = 1    # Default position is long
        result["Market_Stress"] = False
        
        # Ensure sufficient data
        min_periods_needed = max(30, self.trend_window)
        if len(result) < min_periods_needed:
            logger.warning(
                f"Insufficient data for signal generation. Need at least {min_periods_needed} periods"
            )
            return result
        
        rsi_values = result["RSI"]
        drawdown = result["Current_drawdown"]
        
        # Find valid data points
        valid_mask = rsi_values.notna() & drawdown.notna()
        
        if valid_mask.sum() == 0:
            logger.warning("No valid indicator data for signal generation")
            return result
        
        # Track position state
        current_position = 1  # Start invested (like buy-and-hold)
        
        for i in range(len(result)):
            if not valid_mask.iloc[i]:
                continue
            
            current_rsi = rsi_values.iloc[i]
            current_drawdown = drawdown.iloc[i]
            
            # Detect market stress
            market_stress = (
                current_rsi < self.stress_rsi_threshold or  # Extreme oversold
                current_drawdown < -self.drawdown_threshold  # Large drawdown
            )
            
            result.iloc[i, result.columns.get_loc("Market_Stress")] = market_stress
            
            if current_position == 1:  # Currently invested
                if market_stress:
                    # Exit during extreme stress
                    result.iloc[i, result.columns.get_loc("Signal")] = "SELL"
                    result.iloc[i, result.columns.get_loc("Position")] = -1
                    current_position = 0
                else:
                    # Stay invested (the default)
                    result.iloc[i, result.columns.get_loc("Signal")] = "HOLD"
                    result.iloc[i, result.columns.get_loc("Position")] = 1
            
            else:  # Currently out of market
                # Re-enter when stress subsides
                if current_rsi > self.reentry_rsi_threshold:
                    result.iloc[i, result.columns.get_loc("Signal")] = "BUY"
                    result.iloc[i, result.columns.get_loc("Position")] = 1
                    current_position = 1
                else:
                    # Stay out
                    result.iloc[i, result.columns.get_loc("Signal")] = "HOLD"
                    result.iloc[i, result.columns.get_loc("Position")] = 0
        
        # Count signals
        buy_signals = (result["Signal"] == "BUY").sum()
        sell_signals = (result["Signal"] == "SELL").sum()
        stress_periods = result["Market_Stress"].sum()
        
        logger.info(
            f"Generated {buy_signals} BUY and {sell_signals} SELL signals. "
            f"Market stress detected in {stress_periods} periods ({stress_periods/len(result):.1%})"
        )
        
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
        
        # Calculate time in market
        positions_long = (signals["Position"] == 1).sum()
        time_in_market = positions_long / len(signals)
        
        # Market stress periods
        stress_periods = signals.get("Market_Stress", pd.Series(False, index=signals.index)).sum()
        stress_frequency = stress_periods / len(signals)
        
        return {
            "strategy_type": "BuyHoldPlus",
            "parameters": {
                "stress_rsi_threshold": self.stress_rsi_threshold,
                "reentry_rsi_threshold": self.reentry_rsi_threshold,
                "drawdown_threshold": self.drawdown_threshold,
                "trend_window": self.trend_window,
                "price_column": self.price_column,
            },
            "signals_generated": {
                "BUY": signal_counts.get("BUY", 0),
                "SELL": signal_counts.get("SELL", 0),
                "HOLD": signal_counts.get("HOLD", 0),
                "total": len(signals),
            },
            "market_timing": {
                "time_in_market": time_in_market,
                "stress_frequency": stress_frequency,
                "stress_periods": stress_periods,
            },
            "signal_frequency": {
                "buy_frequency": signal_counts.get("BUY", 0) / len(signals),
                "sell_frequency": signal_counts.get("SELL", 0) / len(signals),
            },
        }