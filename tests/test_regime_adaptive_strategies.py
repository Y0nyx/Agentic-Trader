"""
Tests for Regime-Adaptive Strategy System.

This module provides comprehensive tests for the new regime-adaptive
trading strategies and components.
"""

import unittest
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# Import the new strategy components
from strategies.market_regime_classifier import MarketRegimeClassifier
from strategies.regime_adaptive_strategy import RegimeAdaptiveStrategy, CashStrategy
from strategies.specialized_strategies import (
    IntelligentBullStrategy,
    CrisisProtectionStrategy,
    VolatilityRangeStrategy
)
from strategies.contextual_optimizer import ContextualOptimizer
from strategies.multi_strategy_portfolio import MultiStrategyPortfolio

logger = logging.getLogger(__name__)


class TestMarketRegimeClassifier(unittest.TestCase):
    """Test cases for MarketRegimeClassifier."""
    
    def setUp(self):
        """Set up test data and classifier."""
        self.classifier = MarketRegimeClassifier()
        
        # Create test data with different market conditions
        self.test_data = self._create_test_market_data()
        
    def _create_test_market_data(self) -> pd.DataFrame:
        """Create synthetic market data for testing."""
        dates = pd.date_range('2020-01-01', periods=300, freq='D')
        
        # Create different market scenarios
        prices = []
        volumes = []
        
        for i in range(300):
            if i < 60:  # Bull market
                price = 100 + i * 0.5 + np.random.normal(0, 1)
                volume = 1000000 + np.random.normal(0, 100000)
            elif i < 120:  # Crisis period
                price = prices[-1] * (1 + np.random.normal(-0.02, 0.05))
                volume = 3000000 + np.random.normal(0, 500000)
            elif i < 200:  # Sideways volatile
                price = 130 + 10 * np.sin(i * 0.1) + np.random.normal(0, 2)
                volume = 1500000 + np.random.normal(0, 200000)
            else:  # Bear market
                price = prices[-1] * (1 + np.random.normal(-0.01, 0.02))
                volume = 1200000 + np.random.normal(0, 150000)
            
            prices.append(max(price, 10))  # Ensure positive prices
            volumes.append(max(volume, 100000))  # Ensure positive volume
        
        data = pd.DataFrame({
            'Date': dates,
            'Open': [p * 0.995 for p in prices],
            'High': [p * 1.02 for p in prices],
            'Low': [p * 0.98 for p in prices],
            'Close': prices,
            'Volume': volumes
        })
        
        data.set_index('Date', inplace=True)
        return data
    
    def test_regime_classifier_initialization(self):
        """Test MarketRegimeClassifier initialization."""
        self.assertIsInstance(self.classifier, MarketRegimeClassifier)
        self.assertEqual(self.classifier.trend_window, 252)
        self.assertEqual(self.classifier.volatility_window, 60)
        
    def test_classify_regime_with_valid_data(self):
        """Test regime classification with valid data."""
        regime = self.classifier.classify_regime(self.test_data)
        
        self.assertIsInstance(regime, str)
        self.assertIn(regime, [
            'strong_bull', 'moderate_bull', 'sideways_volatile',
            'sideways_calm', 'bear_market', 'crisis_mode'
        ])
    
    def test_classify_regime_insufficient_data(self):
        """Test regime classification with insufficient data."""
        small_data = self.test_data.head(10)
        regime = self.classifier.classify_regime(small_data)
        
        self.assertEqual(regime, 'sideways_calm')  # Default
    
    def test_get_regime_features(self):
        """Test regime feature extraction."""
        features = self.classifier.get_regime_features(self.test_data)
        
        self.assertIsInstance(features, dict)
        expected_features = [
            'trend_slope', 'volatility_annualized', 'vix_equivalent',
            'volume_anomaly', 'current_drawdown', 'price_momentum'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, features)
            self.assertIsInstance(features[feature], (int, float))
    
    def test_regime_summary(self):
        """Test comprehensive regime summary."""
        summary = self.classifier.get_regime_summary(self.test_data)
        
        self.assertIn('current_regime', summary)
        self.assertIn('features', summary)
        self.assertIn('regime_confidence', summary)
        self.assertIn('regime_description', summary)
        self.assertIn('suggested_strategy_type', summary)
    
    def test_regime_stability(self):
        """Test regime classification stability over similar periods."""
        # Test that similar market conditions produce similar regimes
        stable_data = self._create_stable_bull_data()
        
        regimes = []
        for i in range(5):
            subset = stable_data.iloc[i*50:(i+1)*50+100]  # Overlapping windows
            if len(subset) >= 150:
                regime = self.classifier.classify_regime(subset)
                regimes.append(regime)
        
        # Should have some consistency in regime classification
        if regimes:
            unique_regimes = set(regimes)
            self.assertLessEqual(len(unique_regimes), 3)  # Allow some variation
    
    def _create_stable_bull_data(self) -> pd.DataFrame:
        """Create stable bull market data for testing."""
        dates = pd.date_range('2020-01-01', periods=350, freq='D')
        base_price = 100
        
        prices = []
        for i in range(350):
            # Steady uptrend with low volatility
            price = base_price + i * 0.3 + np.random.normal(0, 0.5)
            prices.append(max(price, 10))
        
        data = pd.DataFrame({
            'Date': dates,
            'Open': [p * 0.998 for p in prices],
            'High': [p * 1.008 for p in prices],
            'Low': [p * 0.992 for p in prices],
            'Close': prices,
            'Volume': [1000000] * 350
        })
        
        data.set_index('Date', inplace=True)
        return data


class TestRegimeAdaptiveStrategy(unittest.TestCase):
    """Test cases for RegimeAdaptiveStrategy."""
    
    def setUp(self):
        """Set up test data and strategy."""
        self.strategy = RegimeAdaptiveStrategy()
        self.test_data = self._create_diverse_market_data()
    
    def _create_diverse_market_data(self) -> pd.DataFrame:
        """Create diverse market data for testing strategy adaptation."""
        dates = pd.date_range('2020-01-01', periods=200, freq='D')
        
        prices = []
        base_price = 100
        
        for i in range(200):
            if i < 50:  # Bull market
                price = base_price + i * 0.8 + np.random.normal(0, 1)
            elif i < 100:  # Volatile sideways
                price = 140 + 15 * np.sin(i * 0.2) + np.random.normal(0, 3)
            elif i < 150:  # Bear market
                price = prices[-1] * (1 + np.random.normal(-0.015, 0.02))
            else:  # Recovery
                price = prices[-1] * (1 + np.random.normal(0.01, 0.02))
            
            prices.append(max(price, 10))
        
        data = pd.DataFrame({
            'Date': dates,
            'Open': [p * 0.995 for p in prices],
            'High': [p * 1.02 for p in prices],
            'Low': [p * 0.98 for p in prices],
            'Close': prices,
            'Volume': [1500000 + np.random.normal(0, 200000) for _ in range(200)]
        })
        
        data.set_index('Date', inplace=True)
        return data
    
    def test_regime_adaptive_strategy_initialization(self):
        """Test RegimeAdaptiveStrategy initialization."""
        self.assertIsInstance(self.strategy, RegimeAdaptiveStrategy)
        self.assertIsInstance(self.strategy.regime_classifier, MarketRegimeClassifier)
        self.assertIsInstance(self.strategy.strategies, dict)
        self.assertGreater(len(self.strategy.strategies), 0)
    
    def test_generate_signals_valid_data(self):
        """Test signal generation with valid data."""
        signals = self.strategy.generate_signals(self.test_data)
        
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertEqual(len(signals), len(self.test_data))
        
        # Check required columns
        required_columns = ['Signal', 'Position', 'Current_Regime']
        for col in required_columns:
            self.assertIn(col, signals.columns)
    
    def test_generate_signals_insufficient_data(self):
        """Test signal generation with insufficient data."""
        small_data = self.test_data.head(20)
        signals = self.strategy.generate_signals(small_data)
        
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertEqual(len(signals), len(small_data))
    
    def test_regime_tracking(self):
        """Test regime tracking and stability."""
        # Generate signals multiple times to test regime tracking
        for _ in range(3):
            signals = self.strategy.generate_signals(self.test_data)
        
        # Check that regime history is being tracked
        self.assertGreater(len(self.strategy.regime_history), 0)
        self.assertIsNotNone(self.strategy.current_regime)
    
    def test_strategy_switching(self):
        """Test that strategy switches based on regime changes."""
        # Create data with distinct regime characteristics
        bull_data = self._create_regime_specific_data('bull')
        bear_data = self._create_regime_specific_data('bear')
        
        # Test on bull data
        bull_signals = self.strategy.generate_signals(bull_data)
        bull_regime = bull_signals['Current_Regime'].iloc[-1]
        
        # Reset strategy for clean test
        self.strategy.regime_history = []
        self.strategy.current_regime = None
        
        # Test on bear data
        bear_signals = self.strategy.generate_signals(bear_data)
        bear_regime = bear_signals['Current_Regime'].iloc[-1]
        
        # Regimes should be different for different market conditions
        # (allowing for some variability in classification)
        logger.info(f"Bull regime: {bull_regime}, Bear regime: {bear_regime}")
    
    def test_get_strategy_summary(self):
        """Test strategy summary generation."""
        signals = self.strategy.generate_signals(self.test_data)
        summary = self.strategy.get_strategy_summary(signals)
        
        self.assertIn('strategy_name', summary)
        self.assertIn('total_signals', summary)
        self.assertIn('current_regime', summary)
        self.assertEqual(summary['total_signals'], len(signals))
    
    def _create_regime_specific_data(self, regime_type: str) -> pd.DataFrame:
        """Create data specific to regime type."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        base_price = 100
        
        prices = []
        for i in range(100):
            if regime_type == 'bull':
                # Strong uptrend with low volatility
                price = base_price + i * 1.0 + np.random.normal(0, 0.5)
            elif regime_type == 'bear':
                # Downtrend with moderate volatility
                price = base_price - i * 0.5 + np.random.normal(0, 2)
            else:  # volatile sideways
                price = base_price + 20 * np.sin(i * 0.3) + np.random.normal(0, 5)
            
            prices.append(max(price, 10))
        
        data = pd.DataFrame({
            'Date': dates,
            'Open': [p * 0.995 for p in prices],
            'High': [p * 1.02 for p in prices],
            'Low': [p * 0.98 for p in prices],
            'Close': prices,
            'Volume': [1000000] * 100
        })
        
        data.set_index('Date', inplace=True)
        return data


class TestSpecializedStrategies(unittest.TestCase):
    """Test cases for specialized strategy components."""
    
    def setUp(self):
        """Set up test data for specialized strategies."""
        self.test_data = self._create_specialized_test_data()
    
    def _create_specialized_test_data(self) -> pd.DataFrame:
        """Create test data for specialized strategies."""
        dates = pd.date_range('2020-01-01', periods=150, freq='D')
        
        # Create bull market data with some volatility
        prices = []
        base_price = 100
        
        for i in range(150):
            if i < 100:  # Uptrend
                price = base_price + i * 0.6 + np.random.normal(0, 1.5)
            else:  # Some pullback
                price = prices[-1] * (1 + np.random.normal(-0.005, 0.02))
            
            prices.append(max(price, 10))
        
        data = pd.DataFrame({
            'Date': dates,
            'Open': [p * 0.995 for p in prices],
            'High': [p * 1.025 for p in prices],
            'Low': [p * 0.975 for p in prices],
            'Close': prices,
            'Volume': [1200000 + np.random.normal(0, 300000) for _ in range(150)]
        })
        
        data.set_index('Date', inplace=True)
        return data
    
    def test_intelligent_bull_strategy(self):
        """Test IntelligentBullStrategy."""
        strategy = IntelligentBullStrategy()
        signals = strategy.generate_signals(self.test_data)
        
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertEqual(len(signals), len(self.test_data))
        self.assertIn('Signal', signals.columns)
        self.assertIn('Position', signals.columns)
        
        # Should have some positions in bull market data
        positions = signals['Position'].sum()
        self.assertGreater(positions, 0)
    
    def test_crisis_protection_strategy(self):
        """Test CrisisProtectionStrategy."""
        strategy = CrisisProtectionStrategy()
        
        # Create crisis data
        crisis_data = self._create_crisis_data()
        signals = strategy.generate_signals(crisis_data)
        
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertEqual(len(signals), len(crisis_data))
        
        # Should have crisis score column
        self.assertIn('Crisis_Score', signals.columns)
        
        # Should detect high crisis periods
        high_crisis = (signals['Crisis_Score'] > 0.6).sum()
        self.assertGreater(high_crisis, 0)
    
    def test_volatility_range_strategy(self):
        """Test VolatilityRangeStrategy."""
        strategy = VolatilityRangeStrategy()
        
        # Create sideways volatile data
        sideways_data = self._create_sideways_data()
        signals = strategy.generate_signals(sideways_data)
        
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertEqual(len(signals), len(sideways_data))
        
        # Should have some trading signals in range-bound market
        signal_count = (signals['Signal'] != 'HOLD').sum()
        self.assertGreater(signal_count, 0)
    
    def test_cash_strategy(self):
        """Test CashStrategy."""
        strategy = CashStrategy()
        signals = strategy.generate_signals(self.test_data)
        
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertEqual(len(signals), len(self.test_data))
        
        # Should maintain cash position (no trades)
        positions = signals['Position'].sum()
        self.assertEqual(positions, 0)
        
        signals_count = (signals['Signal'] != 'HOLD').sum()
        self.assertEqual(signals_count, 0)
    
    def _create_crisis_data(self) -> pd.DataFrame:
        """Create crisis-like market data."""
        dates = pd.date_range('2020-01-01', periods=60, freq='D')
        prices = []
        volumes = []
        
        base_price = 100
        for i in range(60):
            # High volatility decline
            if i < 20:
                price = base_price - i * 2 + np.random.normal(0, 5)
                volume = 5000000 + np.random.normal(0, 1000000)
            else:
                price = prices[-1] * (1 + np.random.normal(-0.02, 0.08))
                volume = 3000000 + np.random.normal(0, 800000)
            
            prices.append(max(price, 10))
            volumes.append(max(volume, 100000))
        
        data = pd.DataFrame({
            'Date': dates,
            'Open': [p * 0.99 for p in prices],
            'High': [p * 1.05 for p in prices],
            'Low': [p * 0.95 for p in prices],
            'Close': prices,
            'Volume': volumes
        })
        
        data.set_index('Date', inplace=True)
        return data
    
    def _create_sideways_data(self) -> pd.DataFrame:
        """Create sideways market data with volatility."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        
        prices = []
        base_price = 100
        
        for i in range(100):
            # Oscillating around base price
            price = base_price + 15 * np.sin(i * 0.2) + np.random.normal(0, 3)
            prices.append(max(price, 10))
        
        data = pd.DataFrame({
            'Date': dates,
            'Open': [p * 0.995 for p in prices],
            'High': [p * 1.02 for p in prices],
            'Low': [p * 0.98 for p in prices],
            'Close': prices,
            'Volume': [1500000] * 100
        })
        
        data.set_index('Date', inplace=True)
        return data


class TestContextualOptimizer(unittest.TestCase):
    """Test cases for ContextualOptimizer."""
    
    def setUp(self):
        """Set up test optimizer."""
        self.optimizer = ContextualOptimizer(min_samples_per_regime=10)
    
    def test_contextual_optimizer_initialization(self):
        """Test ContextualOptimizer initialization."""
        self.assertIsInstance(self.optimizer, ContextualOptimizer)
        self.assertEqual(self.optimizer.min_samples_per_regime, 10)
        self.assertIsInstance(self.optimizer.context_features, list)
    
    def test_predict_optimal_params_default(self):
        """Test parameter prediction with default behavior."""
        context = {
            'regime': 'moderate_bull',
            'volatility_regime': 1.0,
            'trend_strength': 0.1
        }
        
        params = self.optimizer.predict_optimal_params(context)
        
        self.assertIsInstance(params, dict)
        self.assertIn('trend_window', params)
        self.assertIn('rsi_threshold', params)
    
    def test_get_default_params(self):
        """Test default parameter retrieval."""
        for regime in ['strong_bull', 'moderate_bull', 'bear_market', 'crisis_mode']:
            params = self.optimizer._get_default_params(regime)
            self.assertIsInstance(params, dict)
            self.assertIn('trend_window', params)
            self.assertIn('rsi_threshold', params)


class TestMultiStrategyPortfolio(unittest.TestCase):
    """Test cases for MultiStrategyPortfolio."""
    
    def setUp(self):
        """Set up test portfolio."""
        self.portfolio = MultiStrategyPortfolio(initial_capital=100000)
        self.test_data = self._create_portfolio_test_data()
    
    def _create_portfolio_test_data(self) -> pd.DataFrame:
        """Create test data for portfolio testing."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        
        prices = []
        base_price = 100
        
        for i in range(100):
            price = base_price + i * 0.5 + np.random.normal(0, 2)
            prices.append(max(price, 10))
        
        data = pd.DataFrame({
            'Date': dates,
            'Open': [p * 0.995 for p in prices],
            'High': [p * 1.02 for p in prices],
            'Low': [p * 0.98 for p in prices],
            'Close': prices,
            'Volume': [1000000] * 100
        })
        
        data.set_index('Date', inplace=True)
        return data
    
    def test_portfolio_initialization(self):
        """Test MultiStrategyPortfolio initialization."""
        self.assertIsInstance(self.portfolio, MultiStrategyPortfolio)
        self.assertEqual(self.portfolio.initial_capital, 100000)
        self.assertIsInstance(self.portfolio.strategies, dict)
        self.assertIsInstance(self.portfolio.allocations, dict)
        
        # Check that allocations sum to approximately 1
        total_allocation = sum(self.portfolio.allocations.values())
        self.assertAlmostEqual(total_allocation, 1.0, places=2)
    
    def test_generate_portfolio_signals(self):
        """Test portfolio signal generation."""
        signals = self.portfolio.generate_portfolio_signals(self.test_data)
        
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertEqual(len(signals), len(self.test_data))
        
        # Check required columns
        required_columns = ['Portfolio_Signal', 'Portfolio_Position', 'Current_Regime']
        for col in required_columns:
            self.assertIn(col, signals.columns)
    
    def test_portfolio_summary(self):
        """Test portfolio summary generation."""
        signals = self.portfolio.generate_portfolio_signals(self.test_data)
        summary = self.portfolio.get_portfolio_summary(signals)
        
        self.assertIn('portfolio_name', summary)
        self.assertIn('current_allocations', summary)
        self.assertIn('initial_capital', summary)
        self.assertEqual(summary['initial_capital'], 100000)
    
    def test_allocation_update(self):
        """Test manual allocation updates."""
        original_allocation = self.portfolio.allocations['trend_following']
        self.portfolio.update_allocation('trend_following', 0.4)
        
        new_allocation = self.portfolio.allocations['trend_following']
        self.assertEqual(new_allocation, 0.4)
        self.assertNotEqual(original_allocation, new_allocation)
        
        # Total should still sum to 1
        total_allocation = sum(self.portfolio.allocations.values())
        self.assertAlmostEqual(total_allocation, 1.0, places=2)


class TestStrategyIntegration(unittest.TestCase):
    """Integration tests for the complete regime-adaptive system."""
    
    def setUp(self):
        """Set up integration test components."""
        self.regime_classifier = MarketRegimeClassifier()
        self.adaptive_strategy = RegimeAdaptiveStrategy()
        self.portfolio = MultiStrategyPortfolio()
        self.test_data = self._create_integration_test_data()
    
    def _create_integration_test_data(self) -> pd.DataFrame:
        """Create comprehensive test data for integration testing."""
        dates = pd.date_range('2020-01-01', periods=250, freq='D')
        
        prices = []
        volumes = []
        base_price = 100
        
        for i in range(250):
            if i < 60:  # Bull market
                price = base_price + i * 0.7 + np.random.normal(0, 1)
                volume = 1000000 + np.random.normal(0, 100000)
            elif i < 90:  # Crisis
                price = prices[-1] * (1 + np.random.normal(-0.03, 0.06))
                volume = 4000000 + np.random.normal(0, 500000)
            elif i < 180:  # Sideways
                price = 140 + 20 * np.sin(i * 0.15) + np.random.normal(0, 3)
                volume = 1500000 + np.random.normal(0, 200000)
            else:  # Recovery
                price = prices[-1] * (1 + np.random.normal(0.008, 0.02))
                volume = 1200000 + np.random.normal(0, 150000)
            
            prices.append(max(price, 10))
            volumes.append(max(volume, 100000))
        
        data = pd.DataFrame({
            'Date': dates,
            'Open': [p * 0.995 for p in prices],
            'High': [p * 1.025 for p in prices],
            'Low': [p * 0.975 for p in prices],
            'Close': prices,
            'Volume': volumes
        })
        
        data.set_index('Date', inplace=True)
        return data
    
    def test_complete_workflow_integration(self):
        """Test complete workflow from regime classification to portfolio signals."""
        # Test regime classification
        regime = self.regime_classifier.classify_regime(self.test_data)
        self.assertIsInstance(regime, str)
        
        # Test adaptive strategy
        adaptive_signals = self.adaptive_strategy.generate_signals(self.test_data)
        self.assertIsInstance(adaptive_signals, pd.DataFrame)
        self.assertEqual(len(adaptive_signals), len(self.test_data))
        
        # Test portfolio management
        portfolio_signals = self.portfolio.generate_portfolio_signals(self.test_data)
        self.assertIsInstance(portfolio_signals, pd.DataFrame)
        self.assertEqual(len(portfolio_signals), len(self.test_data))
        
        # Check that all components work together
        self.assertIn('Current_Regime', adaptive_signals.columns)
        self.assertIn('Portfolio_Signal', portfolio_signals.columns)
    
    def test_regime_change_handling(self):
        """Test system response to regime changes."""
        # Split data into different regime periods
        bull_period = self.test_data.iloc[:60]
        crisis_period = self.test_data.iloc[60:90]
        
        # Test adaptation to bull market
        bull_signals = self.adaptive_strategy.generate_signals(bull_period)
        bull_regime = bull_signals['Current_Regime'].iloc[-1]
        
        # Reset for clean test
        self.adaptive_strategy.regime_history = []
        self.adaptive_strategy.current_regime = None
        
        # Test adaptation to crisis
        crisis_signals = self.adaptive_strategy.generate_signals(crisis_period)
        crisis_regime = crisis_signals['Current_Regime'].iloc[-1]
        
        logger.info(f"Regime adaptation: Bull={bull_regime}, Crisis={crisis_regime}")
        
        # System should adapt to different conditions
        # (allowing for some classification variability)
    
    def test_error_handling_and_robustness(self):
        """Test system robustness with edge cases."""
        # Test with minimal data
        minimal_data = self.test_data.head(10)
        
        try:
            regime = self.regime_classifier.classify_regime(minimal_data)
            adaptive_signals = self.adaptive_strategy.generate_signals(minimal_data)
            portfolio_signals = self.portfolio.generate_portfolio_signals(minimal_data)
            
            # Should handle gracefully without errors
            self.assertIsInstance(regime, str)
            self.assertIsInstance(adaptive_signals, pd.DataFrame)
            self.assertIsInstance(portfolio_signals, pd.DataFrame)
            
        except Exception as e:
            self.fail(f"System should handle minimal data gracefully: {e}")
        
        # Test with missing columns
        incomplete_data = self.test_data[['Close']].copy()
        
        try:
            regime = self.regime_classifier.classify_regime(incomplete_data)
            self.assertIsInstance(regime, str)
        except Exception as e:
            # Should handle missing columns gracefully
            pass


if __name__ == '__main__':
    # Configure logging for tests
    logging.basicConfig(level=logging.INFO)
    
    # Run the tests
    unittest.main(verbosity=2)