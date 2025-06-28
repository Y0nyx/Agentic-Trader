"""
Tests for the Copilot Integration module.

This module tests the core functionality of the GitHub Copilot integration
including the strategy database, API, and decorators.
"""

import unittest
import json
from unittest.mock import patch, MagicMock
from copilot_integration.strategy_database import StrategyDatabase, StrategyResult, StrategyInsights
from copilot_integration.copilot_api import CopilotStrategyAPI
from copilot_integration.decorators import copilot_strategy_context, CopilotCommentGenerator


class TestStrategyDatabase(unittest.TestCase):
    """Test the strategy database functionality."""
    
    def setUp(self):
        """Set up test database."""
        self.db = StrategyDatabase()
        # Clear sample data for clean tests
        self.db.results = []
    
    def test_add_result(self):
        """Test adding a strategy result."""
        result = StrategyResult(
            strategy_type="test_strategy",
            parameters={"param1": 10},
            performance_metrics={"sharpe_ratio": 1.5}
        )
        
        self.db.add_result(result)
        self.assertEqual(len(self.db.results), 1)
        self.assertEqual(self.db.results[0].strategy_type, "test_strategy")
    
    def test_get_strategy_results(self):
        """Test filtering results by strategy type."""
        result1 = StrategyResult(
            strategy_type="moving_average",
            parameters={"short_window": 10},
            performance_metrics={"sharpe_ratio": 1.2}
        )
        result2 = StrategyResult(
            strategy_type="rsi",
            parameters={"period": 14},
            performance_metrics={"sharpe_ratio": 0.8}
        )
        
        self.db.add_result(result1)
        self.db.add_result(result2)
        
        ma_results = self.db.get_strategy_results("moving_average")
        self.assertEqual(len(ma_results), 1)
        self.assertEqual(ma_results[0].strategy_type, "moving_average")
    
    def test_find_similar_strategy(self):
        """Test finding strategies by signature."""
        result = StrategyResult(
            strategy_type="test_strategy",
            parameters={"param1": 10},
            performance_metrics={"sharpe_ratio": 1.5}
        )
        
        self.db.add_result(result)
        found = self.db.find_similar_strategy(result.signature)
        
        self.assertIsNotNone(found)
        self.assertEqual(found.strategy_type, "test_strategy")
    
    def test_get_strategy_performance_summary(self):
        """Test getting performance summary."""
        # Add some test results
        for i in range(3):
            result = StrategyResult(
                strategy_type="moving_average",
                parameters={"short_window": 10 + i},
                performance_metrics={"sharpe_ratio": 1.0 + i * 0.2}
            )
            self.db.add_result(result)
        
        summary = self.db.get_strategy_performance_summary("moving_average")
        
        self.assertIsInstance(summary, StrategyInsights)
        self.assertEqual(summary.strategy_type, "moving_average")
        self.assertGreater(summary.success_rate, 0)
        self.assertTrue(len(summary.top_performers) > 0)
    
    def test_get_optimal_parameters(self):
        """Test getting optimal parameters."""
        result = StrategyResult(
            strategy_type="moving_average",
            parameters={"short_window": 10, "long_window": 30},
            performance_metrics={"sharpe_ratio": 1.8}
        )
        
        self.db.add_result(result)
        optimal = self.db.get_optimal_parameters("moving_average")
        
        self.assertEqual(optimal["short_window"], 10)
        self.assertEqual(optimal["long_window"], 30)


class TestCopilotStrategyAPI(unittest.TestCase):
    """Test the Copilot Strategy API."""
    
    def test_get_strategy_insights(self):
        """Test getting strategy insights."""
        insights = CopilotStrategyAPI.get_strategy_insights("moving_average")
        
        self.assertIsInstance(insights, str)
        self.assertIn("HISTORICAL CONTEXT", insights)
        self.assertIn("OPTIMAL PARAMETERS", insights)
        self.assertIn("PERFORMANCE PATTERNS", insights)
    
    def test_suggest_parameters(self):
        """Test parameter suggestions."""
        params = CopilotStrategyAPI.suggest_parameters("moving_average")
        
        self.assertIsInstance(params, dict)
        # Should contain enhanced parameter info
        for param_name, param_info in params.items():
            if isinstance(param_info, dict):
                self.assertIn('recommended_value', param_info)
    
    def test_check_strategy_exists(self):
        """Test checking if strategy exists."""
        # Test with non-existent strategy
        result = CopilotStrategyAPI.check_strategy_exists("nonexistent_signature")
        
        self.assertIsInstance(result, dict)
        self.assertIn("exists", result)
        self.assertFalse(result["exists"])
    
    def test_get_code_patterns(self):
        """Test getting code patterns."""
        patterns = CopilotStrategyAPI.get_code_patterns("moving_average")
        
        self.assertIsInstance(patterns, str)
        self.assertIn("SIGNAL GENERATION", patterns)
        self.assertIn("RISK MANAGEMENT", patterns)


class TestCopilotDecorators(unittest.TestCase):
    """Test the Copilot decorators."""
    
    def test_copilot_strategy_context_decorator(self):
        """Test the strategy context decorator."""
        @copilot_strategy_context("moving_average")
        def test_function():
            """Test function."""
            return "test"
        
        # Check that decorator enhanced the docstring
        self.assertIsNotNone(test_function.__doc__)
        self.assertIn("HISTORICAL CONTEXT", test_function.__doc__)
        self.assertIn("COPILOT CODING GUIDELINES", test_function.__doc__)
        
        # Check metadata
        self.assertTrue(hasattr(test_function, '_copilot_strategy_type'))
        self.assertEqual(test_function._copilot_strategy_type, "moving_average")
        self.assertTrue(test_function._copilot_enhanced)
    
    def test_auto_detect_strategy_type(self):
        """Test auto-detection of strategy type."""
        @copilot_strategy_context()  # No strategy type provided
        def create_moving_average_strategy():
            """Create MA strategy."""
            return "test"
        
        # Should auto-detect as moving_average
        self.assertEqual(create_moving_average_strategy._copilot_strategy_type, "moving_average")
    
    def test_copilot_comment_generator(self):
        """Test the comment generator."""
        comments = CopilotCommentGenerator.generate_parameter_comments("moving_average")
        
        self.assertIsInstance(comments, str)
        self.assertIn("#", comments)  # Should contain comment markers
    
    def test_copilot_implementation_hints(self):
        """Test implementation hints generation."""
        hints = CopilotCommentGenerator.generate_implementation_hints("moving_average")
        
        self.assertIsInstance(hints, str)
        self.assertIn("IMPLEMENTATION HINTS", hints)


class TestStrategyResult(unittest.TestCase):
    """Test the StrategyResult data class."""
    
    def test_strategy_result_creation(self):
        """Test creating a strategy result."""
        result = StrategyResult(
            strategy_type="test",
            parameters={"param": 1},
            performance_metrics={"sharpe": 1.0}
        )
        
        self.assertEqual(result.strategy_type, "test")
        self.assertIsNotNone(result.timestamp)
        self.assertIsNotNone(result.signature)
    
    def test_signature_generation(self):
        """Test signature generation."""
        result1 = StrategyResult(
            strategy_type="test",
            parameters={"param": 1},
            performance_metrics={"sharpe": 1.0}
        )
        
        result2 = StrategyResult(
            strategy_type="test",
            parameters={"param": 1},
            performance_metrics={"sharpe": 1.5}  # Different metrics
        )
        
        # Same strategy type and parameters should have same signature
        self.assertEqual(result1.signature, result2.signature)
        
        result3 = StrategyResult(
            strategy_type="test",
            parameters={"param": 2},  # Different parameters
            performance_metrics={"sharpe": 1.0}
        )
        
        # Different parameters should have different signature
        self.assertNotEqual(result1.signature, result3.signature)


if __name__ == "__main__":
    unittest.main()