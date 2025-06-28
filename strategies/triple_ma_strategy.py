"""
Triple Moving Average Strategy.

This strategy uses three moving averages (short, medium, long) to generate
more robust trading signals with better trend confirmation.
"""

import logging
import pandas as pd
from typing import Dict, Any
from indicators.technical_indicators import sma, ema

logger = logging.getLogger(__name__)


class TripleMovingAverageStrategy:
    """
    Triple Moving Average Trading Strategy.

    This strategy generates trading signals based on the alignment and crossover
    of three moving averages with different periods (short, medium, long).

    Signal Generation Logic:
    - BUY: Short MA > Medium MA > Long MA and Short MA crosses above Medium MA
    - SELL: Short MA < Medium MA < Long MA and Short MA crosses below Medium MA
    - HOLD: Mixed or sideways alignment

    Parameters
    ----------
    short_window : int, default 10
        Period for the short-term moving average
    medium_window : int, default 20
        Period for the medium-term moving average
    long_window : int, default 50
        Period for the long-term moving average
    ma_type : str, default 'sma'
        Type of moving average ('sma' or 'ema')
    price_column : str, default 'Close'
        Price column to use for moving average calculation
    """

    def __init__(
        self,
        short_window: int = 10,
        medium_window: int = 20,
        long_window: int = 50,
        ma_type: str = "sma",
        price_column: str = "Close",
    ):
        if short_window <= 0 or medium_window <= 0 or long_window <= 0:
            raise ValueError("Window periods must be positive integers")

        if not (short_window < medium_window < long_window):
            raise ValueError(
                "Windows must be in ascending order: short < medium < long"
            )

        if ma_type not in ["sma", "ema"]:
            raise ValueError("ma_type must be 'sma' or 'ema'")

        self.short_window = short_window
        self.medium_window = medium_window
        self.long_window = long_window
        self.ma_type = ma_type
        self.price_column = price_column

        logger.info(
            f"Initialized TripleMovingAverageStrategy with short={short_window}, "
            f"medium={medium_window}, long={long_window}, type={ma_type}"
        )

    def calculate_moving_averages(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the three moving averages.

        Parameters
        ----------
        data : pd.DataFrame
            Financial data with OHLCV columns

        Returns
        -------
        pd.DataFrame
            Data with added moving average columns
        """
        result = data.copy()

        if self.price_column not in data.columns:
            raise KeyError(f"Price column '{self.price_column}' not found in data")

        prices = data[self.price_column]

        # Choose MA function
        ma_func = ema if self.ma_type == "ema" else sma

        # Calculate moving averages
        short_ma_col = f"MA_{self.short_window}_{self.ma_type.upper()}"
        medium_ma_col = f"MA_{self.medium_window}_{self.ma_type.upper()}"
        long_ma_col = f"MA_{self.long_window}_{self.ma_type.upper()}"

        result[short_ma_col] = ma_func(prices, self.short_window)
        result[medium_ma_col] = ma_func(prices, self.medium_window)
        result[long_ma_col] = ma_func(prices, self.long_window)

        return result

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on triple moving average crossovers.

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

        # Calculate moving averages
        result = self.calculate_moving_averages(data)

        short_ma_col = f"MA_{self.short_window}_{self.ma_type.upper()}"
        medium_ma_col = f"MA_{self.medium_window}_{self.ma_type.upper()}"
        long_ma_col = f"MA_{self.long_window}_{self.ma_type.upper()}"

        # Initialize signal columns
        result["Signal"] = "HOLD"
        result["Position"] = 0

        # Ensure we have sufficient data
        min_periods_needed = self.long_window
        if len(result) < min_periods_needed:
            logger.warning(
                f"Insufficient data for signal generation. Need at least {min_periods_needed} periods"
            )
            return result

        # Get moving averages
        short_ma = result[short_ma_col]
        medium_ma = result[medium_ma_col]
        long_ma = result[long_ma_col]

        # Find valid data points
        valid_mask = short_ma.notna() & medium_ma.notna() & long_ma.notna()

        if valid_mask.sum() == 0:
            logger.warning("No valid moving average data for signal generation")
            return result

        # Previous period values for crossover detection
        short_ma_prev = short_ma.shift(1)
        medium_ma_prev = medium_ma.shift(1)

        # Trend alignment conditions
        bullish_alignment = (short_ma > medium_ma) & (medium_ma > long_ma)
        bearish_alignment = (short_ma < medium_ma) & (medium_ma < long_ma)

        # Crossover conditions
        bullish_crossover = (
            (short_ma > medium_ma)
            & (short_ma_prev <= medium_ma_prev)  # Current: short above medium
            & bullish_alignment  # Previous: short below or equal medium
            & valid_mask  # Bullish trend alignment
            & valid_mask.shift(1)
        )

        bearish_crossover = (
            (short_ma < medium_ma)
            & (short_ma_prev >= medium_ma_prev)  # Current: short below medium
            & bearish_alignment  # Previous: short above or equal medium
            & valid_mask  # Bearish trend alignment
            & valid_mask.shift(1)
        )

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
            "strategy_type": "TripleMovingAverage",
            "parameters": {
                "short_window": self.short_window,
                "medium_window": self.medium_window,
                "long_window": self.long_window,
                "ma_type": self.ma_type,
                "price_column": self.price_column,
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
