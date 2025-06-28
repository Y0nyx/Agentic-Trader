"""
Test suite for advanced strategies and optimization framework.

This module provides comprehensive tests for the new trading strategies
and multi-objective optimization capabilities.
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from strategies.triple_ma_strategy import TripleMovingAverageStrategy
from strategies.adaptive_ma_strategy import AdaptiveMovingAverageStrategy
from strategies.advanced_ma_strategy import AdvancedMAStrategy
from strategies.trend_following_strategy import TrendFollowingStrategy
from strategies.buy_hold_plus_strategy import BuyHoldPlusStrategy
from indicators.technical_indicators import rsi, macd, sma, ema, bollinger_bands
from optimization.multi_objective_optimizer import (
    MultiObjectiveOptimizer,
    OptimizationObjective,
)
from simulation.backtester import Backtester


class TestAdvancedStrategies(unittest.TestCase):
    """Test cases for advanced trading strategies."""

    def setUp(self):
        """Set up test data."""
        # Create sample data
        dates = pd.date_range("2020-01-01", periods=252, freq="D")
        np.random.seed(42)

        # Generate realistic price data
        returns = np.random.normal(0.001, 0.02, 252)  # Daily returns
        prices = [100]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))

        self.sample_data = pd.DataFrame(
            {
                "Date": dates,
                "Open": prices[:-1],
                "High": [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices[:-1]],
                "Low": [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices[:-1]],
                "Close": prices[1:],
                "Volume": np.random.randint(100000, 1000000, 252),
            }
        )
        self.sample_data.set_index("Date", inplace=True)

    def test_triple_ma_strategy_initialization(self):
        """Test TripleMovingAverageStrategy initialization."""
        strategy = TripleMovingAverageStrategy(
            short_window=5, medium_window=15, long_window=30, ma_type="sma"
        )

        self.assertEqual(strategy.short_window, 5)
        self.assertEqual(strategy.medium_window, 15)
        self.assertEqual(strategy.long_window, 30)
        self.assertEqual(strategy.ma_type, "sma")

    def test_triple_ma_strategy_invalid_params(self):
        """Test TripleMovingAverageStrategy with invalid parameters."""
        # Test invalid window order
        with self.assertRaises(ValueError):
            TripleMovingAverageStrategy(
                short_window=30, medium_window=20, long_window=10
            )

        # Test invalid MA type
        with self.assertRaises(ValueError):
            TripleMovingAverageStrategy(ma_type="invalid")

    def test_triple_ma_strategy_signal_generation(self):
        """Test TripleMovingAverageStrategy signal generation."""
        strategy = TripleMovingAverageStrategy(
            short_window=5, medium_window=10, long_window=20, ma_type="sma"
        )
        signals = strategy.generate_signals(self.sample_data)

        # Check required columns exist
        required_columns = ["Signal", "Position", "MA_5_SMA", "MA_10_SMA", "MA_20_SMA"]
        for col in required_columns:
            self.assertIn(col, signals.columns)

        # Check signal values are valid
        valid_signals = ["BUY", "SELL", "HOLD"]
        self.assertTrue(signals["Signal"].isin(valid_signals).all())

        # Check position values are valid
        valid_positions = [-1, 0, 1]
        self.assertTrue(signals["Position"].isin(valid_positions).all())

    def test_adaptive_ma_strategy_initialization(self):
        """Test AdaptiveMovingAverageStrategy initialization."""
        strategy = AdaptiveMovingAverageStrategy(
            fast_period=2, slow_period=30, use_rsi_filter=True, use_trend_filter=True
        )

        self.assertEqual(strategy.fast_period, 2)
        self.assertEqual(strategy.slow_period, 30)
        self.assertTrue(strategy.use_rsi_filter)
        self.assertTrue(strategy.use_trend_filter)

    def test_adaptive_ma_strategy_signal_generation(self):
        """Test AdaptiveMovingAverageStrategy signal generation."""
        strategy = AdaptiveMovingAverageStrategy(
            fast_period=2, slow_period=20, use_rsi_filter=True, use_trend_filter=False
        )
        signals = strategy.generate_signals(self.sample_data)

        # Check required columns exist
        required_columns = ["Signal", "Position", "AMA", "RSI"]
        for col in required_columns:
            self.assertIn(col, signals.columns)

        # Check signal values are valid
        valid_signals = ["BUY", "SELL", "HOLD"]
        self.assertTrue(signals["Signal"].isin(valid_signals).all())

    def test_advanced_ma_strategy_initialization(self):
        """Test AdvancedMAStrategy initialization."""
        strategy = AdvancedMAStrategy(
            short_window=10,
            long_window=30,
            ma_type="ema",
            use_volume_filter=True,
            use_rsi_filter=True,
            use_macd_filter=True,
        )

        self.assertEqual(strategy.short_window, 10)
        self.assertEqual(strategy.long_window, 30)
        self.assertEqual(strategy.ma_type, "ema")
        self.assertTrue(strategy.use_volume_filter)
        self.assertTrue(strategy.use_rsi_filter)
        self.assertTrue(strategy.use_macd_filter)

    def test_advanced_ma_strategy_signal_generation(self):
        """Test AdvancedMAStrategy signal generation."""
        strategy = AdvancedMAStrategy(
            short_window=5,
            long_window=15,
            ma_type="ema",
            use_volume_filter=True,
            use_rsi_filter=True,
            use_macd_filter=False,
            use_bollinger_filter=False,
            use_trend_filter=False,
        )
        signals = strategy.generate_signals(self.sample_data)

        # Check required columns exist
        required_columns = ["Signal", "Position", "MA_short_ema", "MA_long_ema", "RSI"]
        for col in required_columns:
            self.assertIn(col, signals.columns)

        # Check signal values are valid
        valid_signals = ["BUY", "SELL", "HOLD"]
        self.assertTrue(signals["Signal"].isin(valid_signals).all())

    def test_trend_following_strategy(self):
        """Test TrendFollowingStrategy."""
        strategy = TrendFollowingStrategy(
            trend_window=30, confirmation_window=10, min_trend_strength=25
        )
        signals = strategy.generate_signals(self.sample_data)

        # Check required columns exist
        required_columns = [
            "Signal",
            "Position",
            "EMA_trend",
            "EMA_confirm",
            "Trend_Score",
        ]
        for col in required_columns:
            self.assertIn(col, signals.columns)

        # Check signal values are valid
        valid_signals = ["BUY", "SELL", "HOLD"]
        self.assertTrue(signals["Signal"].isin(valid_signals).all())

    def test_buy_hold_plus_strategy(self):
        """Test BuyHoldPlusStrategy."""
        strategy = BuyHoldPlusStrategy(
            stress_rsi_threshold=25, reentry_rsi_threshold=40, drawdown_threshold=0.15
        )
        signals = strategy.generate_signals(self.sample_data)

        # Check required columns exist
        required_columns = ["Signal", "Position", "RSI", "Market_Stress"]
        for col in required_columns:
            self.assertIn(col, signals.columns)

        # Check signal values are valid
        valid_signals = ["BUY", "SELL", "HOLD"]
        self.assertTrue(signals["Signal"].isin(valid_signals).all())


class TestTechnicalIndicators(unittest.TestCase):
    """Test cases for technical indicators."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.prices = pd.Series([100 + i + np.random.normal(0, 2) for i in range(100)])
        self.volume = pd.Series(np.random.randint(1000, 10000, 100))
        self.high = self.prices * 1.02
        self.low = self.prices * 0.98

    def test_rsi_calculation(self):
        """Test RSI calculation."""
        rsi_values = rsi(self.prices, window=14)

        # Check output is pandas Series
        self.assertIsInstance(rsi_values, pd.Series)

        # Check values are in valid range (0-100)
        valid_rsi = rsi_values.dropna()
        self.assertTrue((valid_rsi >= 0).all())
        self.assertTrue((valid_rsi <= 100).all())

        # Check length
        self.assertEqual(len(rsi_values), len(self.prices))

    def test_macd_calculation(self):
        """Test MACD calculation."""
        macd_line, signal_line, histogram = macd(self.prices)

        # Check all outputs are pandas Series
        self.assertIsInstance(macd_line, pd.Series)
        self.assertIsInstance(signal_line, pd.Series)
        self.assertIsInstance(histogram, pd.Series)

        # Check lengths
        self.assertEqual(len(macd_line), len(self.prices))
        self.assertEqual(len(signal_line), len(self.prices))
        self.assertEqual(len(histogram), len(self.prices))

    def test_bollinger_bands_calculation(self):
        """Test Bollinger Bands calculation."""
        upper, middle, lower = bollinger_bands(self.prices, window=20)

        # Check all outputs are pandas Series
        self.assertIsInstance(upper, pd.Series)
        self.assertIsInstance(middle, pd.Series)
        self.assertIsInstance(lower, pd.Series)

        # Check band relationships where data is available
        valid_data = upper.notna() & middle.notna() & lower.notna()
        self.assertTrue((upper[valid_data] >= middle[valid_data]).all())
        self.assertTrue((middle[valid_data] >= lower[valid_data]).all())

    def test_moving_averages(self):
        """Test moving average calculations."""
        sma_values = sma(self.prices, window=10)
        ema_values = ema(self.prices, window=10)

        # Check outputs are pandas Series
        self.assertIsInstance(sma_values, pd.Series)
        self.assertIsInstance(ema_values, pd.Series)

        # Check lengths
        self.assertEqual(len(sma_values), len(self.prices))
        self.assertEqual(len(ema_values), len(self.prices))

        # EMA should react faster than SMA (more responsive)
        # This is hard to test directly, but we can check they're different
        valid_data = sma_values.notna() & ema_values.notna()
        if valid_data.sum() > 1:
            # They should not be identical
            self.assertFalse(sma_values[valid_data].equals(ema_values[valid_data]))


class TestMultiObjectiveOptimizer(unittest.TestCase):
    """Test cases for multi-objective optimizer."""

    def setUp(self):
        """Set up test data and components."""
        # Create simple test data
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        np.random.seed(42)

        prices = [100]
        for _ in range(99):
            prices.append(prices[-1] * (1 + np.random.normal(0.001, 0.02)))

        self.test_data = pd.DataFrame(
            {
                "Date": dates,
                "Open": prices[:-1],
                "High": [p * 1.01 for p in prices[:-1]],
                "Low": [p * 0.99 for p in prices[:-1]],
                "Close": prices[1:],
                "Volume": np.random.randint(1000, 10000, 100),
            }
        )
        self.test_data.set_index("Date", inplace=True)

        # Setup optimizer components
        self.backtester = Backtester(initial_capital=10000)
        self.param_grid = {
            "short_window": [5, 10],
            "long_window": [20, 30],
        }

    def test_optimization_objective_creation(self):
        """Test OptimizationObjective creation."""
        obj = OptimizationObjective("total_return", maximize=True, weight=0.5)

        self.assertEqual(obj.name, "total_return")
        self.assertTrue(obj.maximize)
        self.assertEqual(obj.weight, 0.5)

    def test_multi_objective_optimizer_initialization(self):
        """Test MultiObjectiveOptimizer initialization."""
        from strategies.moving_average_cross import MovingAverageCrossStrategy

        objectives = [
            OptimizationObjective("total_return", maximize=True, weight=0.6),
            OptimizationObjective("sharpe_ratio", maximize=True, weight=0.4),
        ]

        optimizer = MultiObjectiveOptimizer(
            strategy_class=MovingAverageCrossStrategy,
            backtester=self.backtester,
            param_grid=self.param_grid,
            objectives=objectives,
        )

        self.assertEqual(len(optimizer.objectives), 2)
        self.assertEqual(optimizer.param_grid, self.param_grid)

        # Check weights are normalized
        total_weight = sum(obj.weight for obj in optimizer.objectives)
        self.assertAlmostEqual(total_weight, 1.0, places=6)

    def test_multi_objective_optimizer_small_run(self):
        """Test MultiObjectiveOptimizer with small parameter space."""
        from strategies.moving_average_cross import MovingAverageCrossStrategy

        objectives = [
            OptimizationObjective("total_return_pct", maximize=True, weight=1.0),
        ]

        optimizer = MultiObjectiveOptimizer(
            strategy_class=MovingAverageCrossStrategy,
            backtester=self.backtester,
            param_grid=self.param_grid,
            objectives=objectives,
        )

        try:
            best_params, summary = optimizer.optimize(self.test_data)

            # Should return some result
            self.assertIsNotNone(best_params)
            self.assertIsInstance(summary, dict)
            self.assertIn("best_params", summary)
            self.assertIn("total_combinations", summary)

        except ValueError as e:
            # It's OK if no valid combinations found with constraints
            self.assertIn("No valid parameter combinations", str(e))


class TestStrategyIntegration(unittest.TestCase):
    """Integration tests for complete strategy workflow."""

    def setUp(self):
        """Set up test data."""
        dates = pd.date_range("2020-01-01", periods=200, freq="D")
        np.random.seed(42)

        # Generate more realistic trending data
        trend = 0.0005  # Slight upward trend
        volatility = 0.015

        prices = [100]
        for i in range(199):
            daily_return = trend + np.random.normal(0, volatility)
            prices.append(prices[-1] * (1 + daily_return))

        self.test_data = pd.DataFrame(
            {
                "Date": dates,
                "Open": prices[:-1],
                "High": [
                    p * (1 + abs(np.random.normal(0, 0.005))) for p in prices[:-1]
                ],
                "Low": [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices[:-1]],
                "Close": prices[1:],
                "Volume": np.random.randint(100000, 1000000, 200),
            }
        )
        self.test_data.set_index("Date", inplace=True)

    def test_complete_strategy_workflow(self):
        """Test complete workflow: strategy -> backtest -> evaluation."""
        # Test with TripleMovingAverageStrategy
        strategy = TripleMovingAverageStrategy(
            short_window=5, medium_window=15, long_window=30, ma_type="sma"
        )

        # Generate signals
        signals = strategy.generate_signals(self.test_data)
        self.assertGreater(len(signals), 0)

        # Run backtest
        backtester = Backtester(initial_capital=10000)
        performance = backtester.run_backtest(self.test_data, signals)

        # Check performance report
        summary = performance.summary()
        self.assertIn("total_return", summary)
        self.assertIn("sharpe_ratio", summary)
        self.assertIn("max_drawdown_pct", summary)

        # Check portfolio history
        self.assertGreater(len(performance.portfolio_history), 0)
        self.assertIn("Total_Value", performance.portfolio_history.columns)

    def test_strategy_summary_methods(self):
        """Test strategy summary methods."""
        strategy = AdvancedMAStrategy(
            short_window=8,
            long_window=25,
            ma_type="ema",
            use_volume_filter=True,
            use_rsi_filter=True,
        )

        signals = strategy.generate_signals(self.test_data)
        summary = strategy.get_strategy_summary(signals)

        # Check summary structure
        self.assertIn("strategy_type", summary)
        self.assertIn("parameters", summary)
        self.assertIn("signals_generated", summary)
        self.assertEqual(summary["strategy_type"], "AdvancedMA")


if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2)
