"""
Advanced Moving Average Strategy with Multiple Filters.

This strategy combines moving averages with multiple technical indicators
to create high-quality trading signals with reduced false positives.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any
from indicators.technical_indicators import (
    sma,
    ema,
    rsi,
    macd,
    bollinger_bands,
    on_balance_volume,
    average_true_range,
    adx,
)

logger = logging.getLogger(__name__)


class AdvancedMAStrategy:
    """
    Advanced Moving Average Strategy with Multiple Filters.

    This strategy uses traditional moving average crossovers enhanced with
    multiple technical indicators for signal confirmation and filtering.

    Signal Generation Logic:
    - Base: Moving average crossover (EMA or SMA)
    - Filters: RSI, MACD, Bollinger Bands, Volume, ADX trend strength
    - Risk Management: ATR-based position sizing and stop-loss

    Parameters
    ----------
    short_window : int, default 10
        Period for short-term moving average
    long_window : int, default 30
        Period for long-term moving average
    ma_type : str, default 'ema'
        Type of moving average ('sma' or 'ema')
    use_volume_filter : bool, default True
        Whether to use volume confirmation
    use_rsi_filter : bool, default True
        Whether to use RSI filter
    rsi_oversold : float, default 30
        RSI oversold threshold
    rsi_overbought : float, default 70
        RSI overbought threshold
    use_macd_filter : bool, default True
        Whether to use MACD confirmation
    use_bollinger_filter : bool, default True
        Whether to use Bollinger Bands filter
    use_trend_filter : bool, default True
        Whether to use ADX trend strength filter
    adx_threshold : float, default 25
        Minimum ADX for trend confirmation
    atr_multiplier : float, default 2.0
        ATR multiplier for stop-loss calculation
    price_column : str, default 'Close'
        Price column to use
    """

    def __init__(
        self,
        short_window: int = 10,
        long_window: int = 30,
        ma_type: str = "ema",
        use_volume_filter: bool = True,
        use_rsi_filter: bool = True,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        use_macd_filter: bool = True,
        use_bollinger_filter: bool = True,
        use_trend_filter: bool = True,
        adx_threshold: float = 25,
        atr_multiplier: float = 2.0,
        price_column: str = "Close",
    ):
        if short_window <= 0 or long_window <= 0:
            raise ValueError("Window periods must be positive integers")

        if short_window >= long_window:
            raise ValueError("Short window must be smaller than long window")

        if ma_type not in ["sma", "ema"]:
            raise ValueError("ma_type must be 'sma' or 'ema'")

        if not (0 <= rsi_oversold <= 100 and 0 <= rsi_overbought <= 100):
            raise ValueError("RSI thresholds must be between 0 and 100")

        if rsi_oversold >= rsi_overbought:
            raise ValueError("RSI oversold must be less than overbought")

        self.short_window = short_window
        self.long_window = long_window
        self.ma_type = ma_type
        self.use_volume_filter = use_volume_filter
        self.use_rsi_filter = use_rsi_filter
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.use_macd_filter = use_macd_filter
        self.use_bollinger_filter = use_bollinger_filter
        self.use_trend_filter = use_trend_filter
        self.adx_threshold = adx_threshold
        self.atr_multiplier = atr_multiplier
        self.price_column = price_column

        logger.info(
            f"Initialized AdvancedMAStrategy with {ma_type.upper()} {short_window}/{long_window}, "
            f"filters: volume={use_volume_filter}, RSI={use_rsi_filter}, "
            f"MACD={use_macd_filter}, BB={use_bollinger_filter}, trend={use_trend_filter}"
        )

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators.

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

        required_columns = [self.price_column]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise KeyError(f"Required columns not found: {missing_columns}")

        prices = data[self.price_column]

        # Calculate moving averages
        ma_func = ema if self.ma_type == "ema" else sma
        result[f"MA_short_{self.ma_type}"] = ma_func(prices, self.short_window)
        result[f"MA_long_{self.ma_type}"] = ma_func(prices, self.long_window)

        # RSI
        if self.use_rsi_filter:
            result["RSI"] = rsi(prices, window=14)

        # MACD
        if self.use_macd_filter:
            macd_line, signal_line, histogram = macd(prices)
            result["MACD"] = macd_line
            result["MACD_signal"] = signal_line
            result["MACD_histogram"] = histogram

        # Bollinger Bands
        if self.use_bollinger_filter:
            bb_upper, bb_middle, bb_lower = bollinger_bands(prices, window=20)
            result["BB_upper"] = bb_upper
            result["BB_middle"] = bb_middle
            result["BB_lower"] = bb_lower

        # Volume indicators
        if self.use_volume_filter and "Volume" in data.columns:
            result["OBV"] = on_balance_volume(prices, data["Volume"])
            result["Volume_MA"] = sma(data["Volume"], window=20)

        # ATR for risk management
        if all(col in data.columns for col in ["High", "Low"]):
            result["ATR"] = average_true_range(data["High"], data["Low"], prices)

        # ADX for trend strength
        if self.use_trend_filter and all(
            col in data.columns for col in ["High", "Low"]
        ):
            result["ADX"] = adx(data["High"], data["Low"], prices)

        return result

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals with multiple filters.

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
        result["Stop_Loss"] = np.nan
        result["Take_Profit"] = np.nan

        # Ensure we have sufficient data
        min_periods_needed = max(50, self.long_window + 20)  # Enough for all indicators
        if len(result) < min_periods_needed:
            logger.warning(
                f"Insufficient data for signal generation. Need at least {min_periods_needed} periods"
            )
            return result

        # Get moving averages
        short_ma = result[f"MA_short_{self.ma_type}"]
        long_ma = result[f"MA_long_{self.ma_type}"]
        prices = result[self.price_column]

        # Find valid data points
        valid_mask = short_ma.notna() & long_ma.notna() & prices.notna()

        if valid_mask.sum() == 0:
            logger.warning("No valid moving average data for signal generation")
            return result

        # Previous period values for crossover detection
        short_ma_prev = short_ma.shift(1)
        long_ma_prev = long_ma.shift(1)

        # Basic crossover conditions
        bullish_crossover = (
            (short_ma > long_ma)
            & (short_ma_prev <= long_ma_prev)
            & valid_mask
            & valid_mask.shift(1)
        )

        bearish_crossover = (
            (short_ma < long_ma)
            & (short_ma_prev >= long_ma_prev)
            & valid_mask
            & valid_mask.shift(1)
        )

        # Apply filters
        buy_filter = bullish_crossover
        sell_filter = bearish_crossover

        # RSI Filter
        if self.use_rsi_filter and "RSI" in result.columns:
            rsi_values = result["RSI"]
            rsi_valid = rsi_values.notna()

            buy_filter = buy_filter & (rsi_values < self.rsi_overbought) & rsi_valid
            sell_filter = sell_filter & (rsi_values > self.rsi_oversold) & rsi_valid

        # MACD Filter
        if self.use_macd_filter and "MACD" in result.columns:
            macd_values = result["MACD"]
            macd_signal = result["MACD_signal"]
            macd_valid = macd_values.notna() & macd_signal.notna()

            # MACD should confirm the direction
            macd_bullish = (macd_values > macd_signal) & macd_valid
            macd_bearish = (macd_values < macd_signal) & macd_valid

            buy_filter = buy_filter & macd_bullish
            sell_filter = sell_filter & macd_bearish

        # Bollinger Bands Filter
        if self.use_bollinger_filter and "BB_upper" in result.columns:
            bb_upper = result["BB_upper"]
            bb_lower = result["BB_lower"]
            bb_valid = bb_upper.notna() & bb_lower.notna()

            # Don't buy if price is near upper band, don't sell if near lower band
            not_overbought = (prices < bb_upper * 0.95) & bb_valid
            not_oversold = (prices > bb_lower * 1.05) & bb_valid

            buy_filter = buy_filter & not_overbought
            sell_filter = sell_filter & not_oversold

        # Volume Filter
        if self.use_volume_filter and "Volume_MA" in result.columns:
            volume = result["Volume"]
            volume_ma = result["Volume_MA"]
            volume_valid = volume.notna() & volume_ma.notna()

            # Volume should be above average for confirmation
            volume_confirmation = (volume > volume_ma) & volume_valid

            buy_filter = buy_filter & volume_confirmation
            sell_filter = sell_filter & volume_confirmation

        # Trend Strength Filter
        if self.use_trend_filter and "ADX" in result.columns:
            adx_values = result["ADX"]
            adx_valid = adx_values.notna()

            # Only trade in trending markets
            trend_strong = (adx_values > self.adx_threshold) & adx_valid

            buy_filter = buy_filter & trend_strong
            sell_filter = sell_filter & trend_strong

        # Generate signals
        result.loc[buy_filter, "Signal"] = "BUY"
        result.loc[buy_filter, "Position"] = 1

        result.loc[sell_filter, "Signal"] = "SELL"
        result.loc[sell_filter, "Position"] = -1

        # Calculate stop-loss and take-profit levels using ATR
        if "ATR" in result.columns:
            atr_values = result["ATR"]

            # For buy signals
            # For buy signals
            buy_condition = buy_filter & atr_values.notna()
            result.loc[buy_condition, "Stop_Loss"] = prices[buy_condition] - (
                self.atr_multiplier * atr_values[buy_condition]
            )
            result.loc[buy_condition, "Take_Profit"] = prices[buy_condition] + (
                2 * self.atr_multiplier * atr_values[buy_condition]
            )

            # For sell signals
            sell_condition = sell_filter & atr_values.notna()
            result.loc[sell_condition, "Stop_Loss"] = prices[sell_condition] + (
                self.atr_multiplier * atr_values[sell_condition]
            )
            result.loc[sell_condition, "Take_Profit"] = prices[sell_condition] - (
                2 * self.atr_multiplier * atr_values[sell_condition]
            )

        # Count signals
        buy_signals = buy_filter.sum()
        sell_signals = sell_filter.sum()

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
            "strategy_type": "AdvancedMA",
            "parameters": {
                "short_window": self.short_window,
                "long_window": self.long_window,
                "ma_type": self.ma_type,
                "filters_enabled": {
                    "volume": self.use_volume_filter,
                    "rsi": self.use_rsi_filter,
                    "macd": self.use_macd_filter,
                    "bollinger": self.use_bollinger_filter,
                    "trend": self.use_trend_filter,
                },
                "rsi_thresholds": {
                    "oversold": self.rsi_oversold,
                    "overbought": self.rsi_overbought,
                },
                "adx_threshold": self.adx_threshold,
                "atr_multiplier": self.atr_multiplier,
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
