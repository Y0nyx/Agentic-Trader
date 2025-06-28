"""
Tests for trading strategies.

This module tests the trading strategy implementations to ensure proper
signal generation, parameter validation, and edge case handling.
"""

import unittest
import pandas as pd
import numpy as np
from strategies.moving_average_cross import MovingAverageCrossStrategy


class TestMovingAverageCrossStrategy(unittest.TestCase):
    """Test class for MovingAverageCrossStrategy."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample price data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # Create trending price data to test crossovers
        price_base = 100
        trend = np.linspace(0, 20, 100)  # Upward trend
        noise = np.random.normal(0, 1, 100)
        prices = price_base + trend + noise
        
        self.sample_data = pd.DataFrame({
            'Open': prices * 0.99,
            'High': prices * 1.02,
            'Low': prices * 0.98,
            'Close': prices,
            'Volume': np.random.randint(1000000, 2000000, 100)
        }, index=dates)
        
        # Create data with clear crossover patterns
        self.crossover_data = self._create_crossover_test_data()
        
        # Empty DataFrame for edge case testing
        self.empty_data = pd.DataFrame()
    
    def _create_crossover_test_data(self):
        """Create data with predictable crossover patterns."""
        dates = pd.date_range('2023-01-01', periods=60, freq='D')
        
        # Create price pattern that will generate crossovers
        prices = []
        for i in range(60):
            if i < 20:
                prices.append(100 + i * 0.5)  # Slow uptrend
            elif i < 30:
                prices.append(110 - (i-20) * 2)  # Sharp decline
            elif i < 40:
                prices.append(90 + (i-30) * 0.5)  # Recovery
            else:
                prices.append(95 + (i-40) * 1.5)  # Strong uptrend
        
        return pd.DataFrame({
            'Open': [p * 0.99 for p in prices],
            'High': [p * 1.01 for p in prices],
            'Low': [p * 0.98 for p in prices],
            'Close': prices,
            'Volume': [1000000] * 60
        }, index=dates)
    
    def test_strategy_initialization_valid_params(self):
        """Test strategy initialization with valid parameters."""
        strategy = MovingAverageCrossStrategy(short_window=10, long_window=20)
        
        self.assertEqual(strategy.short_window, 10)
        self.assertEqual(strategy.long_window, 20)
        self.assertEqual(strategy.price_column, 'Close')
    
    def test_strategy_initialization_custom_price_column(self):
        """Test strategy initialization with custom price column."""
        strategy = MovingAverageCrossStrategy(
            short_window=10, 
            long_window=20, 
            price_column='High'
        )
        
        self.assertEqual(strategy.price_column, 'High')
    
    def test_strategy_initialization_invalid_params(self):
        """Test strategy initialization with invalid parameters."""
        # Short window >= long window
        with self.assertRaises(ValueError):
            MovingAverageCrossStrategy(short_window=20, long_window=20)
        
        with self.assertRaises(ValueError):
            MovingAverageCrossStrategy(short_window=30, long_window=20)
        
        # Negative or zero windows
        with self.assertRaises(ValueError):
            MovingAverageCrossStrategy(short_window=0, long_window=20)
        
        with self.assertRaises(ValueError):
            MovingAverageCrossStrategy(short_window=10, long_window=-5)
    
    def test_calculate_moving_averages_valid_data(self):
        """Test moving average calculation with valid data."""
        strategy = MovingAverageCrossStrategy(short_window=5, long_window=10)
        result = strategy.calculate_moving_averages(self.sample_data)
        
        self.assertIn('MA_5', result.columns)
        self.assertIn('MA_10', result.columns)
        
        # Check that moving averages are calculated correctly
        expected_ma5 = self.sample_data['Close'].rolling(window=5, min_periods=5).mean()
        expected_ma10 = self.sample_data['Close'].rolling(window=10, min_periods=10).mean()
        
        pd.testing.assert_series_equal(result['MA_5'], expected_ma5, check_names=False)
        pd.testing.assert_series_equal(result['MA_10'], expected_ma10, check_names=False)
    
    def test_calculate_moving_averages_missing_price_column(self):
        """Test moving average calculation with missing price column."""
        strategy = MovingAverageCrossStrategy(short_window=5, long_window=10, price_column='NonExistent')
        
        with self.assertRaises(ValueError):
            strategy.calculate_moving_averages(self.sample_data)
    
    def test_calculate_moving_averages_empty_data(self):
        """Test moving average calculation with empty data."""
        strategy = MovingAverageCrossStrategy(short_window=5, long_window=10)
        result = strategy.calculate_moving_averages(self.empty_data)
        
        self.assertTrue(result.empty)
    
    def test_generate_signals_valid_data(self):
        """Test signal generation with valid data."""
        strategy = MovingAverageCrossStrategy(short_window=5, long_window=10)
        signals = strategy.generate_signals(self.crossover_data)
        
        # Check that all required columns are present
        required_columns = ['MA_5', 'MA_10', 'Signal', 'Position']
        for col in required_columns:
            self.assertIn(col, signals.columns)
        
        # Check signal types
        unique_signals = signals['Signal'].unique()
        for signal in unique_signals:
            self.assertIn(signal, ['BUY', 'SELL', 'HOLD'])
        
        # Check position values
        unique_positions = signals['Position'].unique()
        for position in unique_positions:
            self.assertIn(position, [-1, 0, 1])
    
    def test_generate_signals_crossover_logic(self):
        """Test that crossover logic generates correct signals."""
        strategy = MovingAverageCrossStrategy(short_window=3, long_window=5)
        
        # Create specific test data where we know crossovers will occur
        dates = pd.date_range('2023-01-01', periods=20, freq='D')
        
        # Pattern: starts low, goes up (bullish crossover), then down (bearish crossover)
        prices = [100, 101, 102, 103, 105, 108, 112, 115, 110, 105, 
                  100, 95, 90, 88, 87, 86, 85, 84, 83, 82]
        
        test_data = pd.DataFrame({
            'Close': prices,
            'Open': [p * 0.99 for p in prices],
            'High': [p * 1.01 for p in prices],
            'Low': [p * 0.98 for p in prices],
            'Volume': [1000000] * 20
        }, index=dates)
        
        signals = strategy.generate_signals(test_data)
        
        # Should have some BUY and SELL signals
        buy_signals = (signals['Signal'] == 'BUY').sum()
        sell_signals = (signals['Signal'] == 'SELL').sum()
        
        self.assertGreater(buy_signals + sell_signals, 0, "Should generate some trading signals")
    
    def test_generate_signals_insufficient_data(self):
        """Test signal generation with insufficient data."""
        strategy = MovingAverageCrossStrategy(short_window=10, long_window=20)
        
        # Create data with only 5 periods (less than long_window)
        short_data = self.sample_data.head(5)
        signals = strategy.generate_signals(short_data)
        
        # Should return data with signal columns but all HOLD
        self.assertIn('Signal', signals.columns)
        self.assertTrue((signals['Signal'] == 'HOLD').all())
    
    def test_generate_signals_empty_data(self):
        """Test signal generation with empty data."""
        strategy = MovingAverageCrossStrategy(short_window=5, long_window=10)
        signals = strategy.generate_signals(self.empty_data)
        
        self.assertTrue(signals.empty)
    
    def test_get_strategy_summary_valid_data(self):
        """Test strategy summary with valid data."""
        strategy = MovingAverageCrossStrategy(short_window=5, long_window=10)
        signals = strategy.generate_signals(self.crossover_data)
        summary = strategy.get_strategy_summary(signals)
        
        # Check required summary fields
        required_fields = [
            'short_window', 'long_window', 'total_periods',
            'buy_signals', 'sell_signals', 'hold_periods',
            'signal_frequency', 'date_range'
        ]
        for field in required_fields:
            self.assertIn(field, summary)
        
        # Check data consistency
        self.assertEqual(summary['short_window'], 5)
        self.assertEqual(summary['long_window'], 10)
        self.assertEqual(summary['total_periods'], len(signals))
        
        # Signal counts should add up
        total_signals = summary['buy_signals'] + summary['sell_signals'] + summary['hold_periods']
        self.assertEqual(total_signals, summary['total_periods'])
        
        # Signal frequency should be between 0 and 1
        self.assertGreaterEqual(summary['signal_frequency'], 0)
        self.assertLessEqual(summary['signal_frequency'], 1)
    
    def test_get_strategy_summary_empty_data(self):
        """Test strategy summary with empty data."""
        strategy = MovingAverageCrossStrategy(short_window=5, long_window=10)
        summary = strategy.get_strategy_summary(self.empty_data)
        
        self.assertEqual(summary, {})
    
    def test_signal_position_consistency(self):
        """Test that signals and positions are consistent."""
        strategy = MovingAverageCrossStrategy(short_window=5, long_window=10)
        signals = strategy.generate_signals(self.crossover_data)
        
        # BUY signals should have position = 1
        buy_mask = signals['Signal'] == 'BUY'
        if buy_mask.any():
            self.assertTrue((signals.loc[buy_mask, 'Position'] == 1).all())
        
        # SELL signals should have position = -1
        sell_mask = signals['Signal'] == 'SELL'
        if sell_mask.any():
            self.assertTrue((signals.loc[sell_mask, 'Position'] == -1).all())
        
        # HOLD signals should have position = 0
        hold_mask = signals['Signal'] == 'HOLD'
        if hold_mask.any():
            self.assertTrue((signals.loc[hold_mask, 'Position'] == 0).all())
    
    def test_moving_average_relationships(self):
        """Test that moving averages follow expected relationships."""
        strategy = MovingAverageCrossStrategy(short_window=5, long_window=20)
        signals = strategy.generate_signals(self.sample_data)
        
        # Short MA should be more volatile (respond faster to price changes)
        short_ma = signals['MA_5'].dropna()
        long_ma = signals['MA_20'].dropna()
        
        if len(short_ma) > 1 and len(long_ma) > 1:
            short_volatility = short_ma.pct_change().std()
            long_volatility = long_ma.pct_change().std()
            
            # Short MA should be more volatile than long MA
            self.assertGreater(short_volatility, long_volatility)
    
    def test_different_window_sizes(self):
        """Test strategy with different window size combinations."""
        window_combinations = [
            (5, 10), (10, 20), (20, 50), (3, 7)
        ]
        
        for short, long in window_combinations:
            with self.subTest(short_window=short, long_window=long):
                strategy = MovingAverageCrossStrategy(short_window=short, long_window=long)
                signals = strategy.generate_signals(self.sample_data)
                
                # Should generate valid signals
                self.assertIn('Signal', signals.columns)
                self.assertIn('Position', signals.columns)
                
                # Should have moving average columns
                self.assertIn(f'MA_{short}', signals.columns)
                self.assertIn(f'MA_{long}', signals.columns)


if __name__ == '__main__':
    unittest.main()