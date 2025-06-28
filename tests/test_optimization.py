"""
Tests for the optimization module.

This module contains comprehensive tests for the grid search optimization
functionality including parameter optimization and reporting.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd
from optimization.grid_search import (
    GridSearchOptimizer,
    OptimizationReport,
    optimize_strategy,
)
from simulation.backtester import Backtester, PerformanceReport
from strategies.moving_average_cross import MovingAverageCrossStrategy


class TestGridSearchOptimizer(unittest.TestCase):
    """Test class for GridSearchOptimizer functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        prices = 100 + np.cumsum(np.random.randn(100) * 0.02)  # Random walk

        self.data = pd.DataFrame(
            {
                "Open": prices * 0.99,
                "High": prices * 1.02,
                "Low": prices * 0.98,
                "Close": prices,
                "Volume": np.random.randint(1000000, 2000000, 100),
            },
            index=dates,
        )

        # Create backtester
        self.backtester = Backtester(initial_capital=10000)

        # Create parameter grid
        self.param_grid = {
            "short_window": [5, 10],
            "long_window": [20, 30],
        }

        # Create optimizer
        self.optimizer = GridSearchOptimizer(
            strategy_class=MovingAverageCrossStrategy,
            backtester=self.backtester,
            param_grid=self.param_grid,
            objective="sharpe_ratio",
        )

    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        self.assertEqual(self.optimizer.strategy_class, MovingAverageCrossStrategy)
        self.assertEqual(self.optimizer.backtester, self.backtester)
        self.assertEqual(self.optimizer.param_grid, self.param_grid)
        self.assertEqual(self.optimizer.objective, "sharpe_ratio")
        self.assertEqual(len(self.optimizer.results), 0)

    def test_optimizer_initialization_invalid_param_grid(self):
        """Test optimizer initialization with invalid parameter grid."""
        # Empty param grid
        with self.assertRaises(ValueError):
            GridSearchOptimizer(
                MovingAverageCrossStrategy, self.backtester, {}, "sharpe_ratio"
            )

        # Invalid param grid type
        with self.assertRaises(ValueError):
            GridSearchOptimizer(
                MovingAverageCrossStrategy, self.backtester, "invalid", "sharpe_ratio"
            )

        # Empty parameter values
        with self.assertRaises(ValueError):
            GridSearchOptimizer(
                MovingAverageCrossStrategy,
                self.backtester,
                {"short_window": []},
                "sharpe_ratio",
            )

    def test_count_combinations(self):
        """Test parameter combination counting."""
        count = self.optimizer._count_combinations()
        expected = len(self.param_grid["short_window"]) * len(
            self.param_grid["long_window"]
        )
        self.assertEqual(count, expected)

    def test_generate_param_combinations(self):
        """Test parameter combination generation."""
        combinations = list(self.optimizer._generate_param_combinations())

        self.assertEqual(len(combinations), 4)  # 2 * 2 = 4 combinations

        # Check all combinations are generated
        expected_combinations = [
            {"short_window": 5, "long_window": 20},
            {"short_window": 5, "long_window": 30},
            {"short_window": 10, "long_window": 20},
            {"short_window": 10, "long_window": 30},
        ]

        for expected in expected_combinations:
            self.assertIn(expected, combinations)

    def test_evaluate_objective_string_objectives(self):
        """Test objective evaluation with string objectives."""
        # Create mock performance report
        mock_report = Mock()
        mock_report.summary.return_value = {
            "sharpe_ratio": 1.5,
            "total_return": 0.15,
            "max_drawdown_pct": -5.0,
            "profit_factor": 2.0,
            "win_rate": 0.6,
        }

        # Test different string objectives
        self.optimizer.objective = "sharpe_ratio"
        self.assertEqual(self.optimizer._evaluate_objective(mock_report), 1.5)

        self.optimizer.objective = "roi"
        self.assertEqual(self.optimizer._evaluate_objective(mock_report), 0.15)

        self.optimizer.objective = "drawdown"
        self.assertEqual(
            self.optimizer._evaluate_objective(mock_report), -5.0
        )  # -abs(-5.0) = -5.0

        self.optimizer.objective = "profit_factor"
        self.assertEqual(self.optimizer._evaluate_objective(mock_report), 2.0)

        self.optimizer.objective = "win_rate"
        self.assertEqual(self.optimizer._evaluate_objective(mock_report), 0.6)

    def test_evaluate_objective_callable(self):
        """Test objective evaluation with callable objective."""

        def custom_objective(report):
            summary = report.summary()
            return summary.get("total_return", 0) * 2

        optimizer = GridSearchOptimizer(
            MovingAverageCrossStrategy,
            self.backtester,
            self.param_grid,
            custom_objective,
        )

        mock_report = Mock()
        mock_report.summary.return_value = {"total_return": 0.15}

        result = optimizer._evaluate_objective(mock_report)
        self.assertEqual(result, 0.3)  # 0.15 * 2

    def test_optimize_basic(self):
        """Test basic optimization functionality."""
        # Use smaller param grid for faster testing
        small_param_grid = {
            "short_window": [5, 10],
            "long_window": [20, 25],
        }

        optimizer = GridSearchOptimizer(
            MovingAverageCrossStrategy,
            self.backtester,
            small_param_grid,
            "total_return",
        )

        best_params, report = optimizer.optimize(self.data, verbose=False)

        # Check that optimization completed
        self.assertIsNotNone(best_params)
        self.assertIsInstance(report, OptimizationReport)
        self.assertEqual(len(optimizer.results), 4)  # 2 * 2 combinations

        # Check that best params are in the search space
        self.assertIn(best_params["short_window"], small_param_grid["short_window"])
        self.assertIn(best_params["long_window"], small_param_grid["long_window"])

    def test_optimize_empty_data(self):
        """Test optimization with empty data."""
        empty_data = pd.DataFrame()

        with self.assertRaises(ValueError):
            self.optimizer.optimize(empty_data)

    def test_optimize_with_errors(self):
        """Test optimization handling errors in strategy execution."""
        # Create optimizer with invalid parameter combinations
        # (short_window >= long_window will cause errors)
        bad_param_grid = {
            "short_window": [30, 40],  # These are >= long_window
            "long_window": [20, 25],
        }

        optimizer = GridSearchOptimizer(
            MovingAverageCrossStrategy, self.backtester, bad_param_grid, "sharpe_ratio"
        )

        best_params, report = optimizer.optimize(self.data, verbose=False)

        # Should handle errors gracefully
        self.assertEqual(len(optimizer.results), 4)
        # All results should have failed
        failed_results = [r for r in optimizer.results if not np.isfinite(r["score"])]
        self.assertEqual(len(failed_results), 4)

    def test_get_top_results(self):
        """Test getting top results."""
        # First run optimization
        self.optimizer.optimize(self.data, verbose=False)

        top_results = self.optimizer.get_top_results(n=2)

        self.assertLessEqual(len(top_results), 2)
        if len(top_results) > 1:
            # Results should be sorted by score (descending)
            self.assertGreaterEqual(top_results[0]["score"], top_results[1]["score"])

    def test_analyze_parameter_sensitivity(self):
        """Test parameter sensitivity analysis."""
        # Run optimization first
        self.optimizer.optimize(self.data, verbose=False)

        sensitivity = self.optimizer.analyze_parameter_sensitivity()

        # Should have sensitivity data for each parameter
        for param in self.param_grid.keys():
            if param in sensitivity:
                self.assertIn("correlation", sensitivity[param])
                self.assertIn("param_stats", sensitivity[param])


class TestOptimizationReport(unittest.TestCase):
    """Test class for OptimizationReport functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample optimization results
        self.results = [
            {
                "params": {"short_window": 5, "long_window": 20},
                "score": 1.2,
                "performance_summary": {"total_return": 0.12},
            },
            {
                "params": {"short_window": 10, "long_window": 20},
                "score": 1.5,
                "performance_summary": {"total_return": 0.15},
            },
            {
                "params": {"short_window": 5, "long_window": 30},
                "score": 0.8,
                "performance_summary": {"total_return": 0.08},
            },
            {
                "params": {"short_window": 10, "long_window": 30},
                "score": -float("inf"),  # Failed result
                "error": "Test error",
                "performance_summary": {},
            },
        ]

        self.param_grid = {
            "short_window": [5, 10],
            "long_window": [20, 30],
        }

        self.report = OptimizationReport(self.results, "sharpe_ratio", self.param_grid)

    def test_report_initialization(self):
        """Test report initialization."""
        self.assertEqual(len(self.report.results), 4)
        self.assertEqual(len(self.report.valid_results), 3)
        self.assertEqual(len(self.report.failed_results), 1)

    def test_summary(self):
        """Test report summary generation."""
        summary = self.report.summary()

        expected_keys = [
            "total_combinations",
            "successful_combinations",
            "failed_combinations",
            "best_score",
            "best_params",
            "score_statistics",
        ]

        for key in expected_keys:
            self.assertIn(key, summary)

        self.assertEqual(summary["total_combinations"], 4)
        self.assertEqual(summary["successful_combinations"], 3)
        self.assertEqual(summary["failed_combinations"], 1)
        self.assertEqual(summary["best_score"], 1.5)
        self.assertEqual(
            summary["best_params"], {"short_window": 10, "long_window": 20}
        )

    def test_summary_no_valid_results(self):
        """Test summary with no valid results."""
        failed_results = [
            {
                "params": {"short_window": 5, "long_window": 20},
                "score": -float("inf"),
                "error": "Test error",
            }
        ]

        report = OptimizationReport(failed_results, "sharpe_ratio", self.param_grid)
        summary = report.summary()

        self.assertEqual(summary["successful_combinations"], 0)
        self.assertIsNone(summary["best_score"])

    def test_get_parameter_impact(self):
        """Test parameter impact calculation."""
        impact = self.report.get_parameter_impact()

        # Should have impact for each parameter
        for param in self.param_grid.keys():
            self.assertIn(param, impact)
            self.assertIsInstance(impact[param], (int, float))

    def test_get_best_results(self):
        """Test getting best results."""
        best_results = self.report.get_best_results(n=2)

        self.assertEqual(len(best_results), 2)
        # Should be sorted by score
        self.assertGreaterEqual(best_results[0]["score"], best_results[1]["score"])

    def test_print_summary(self):
        """Test formatted summary printing."""
        summary_text = self.report.print_summary()

        self.assertIsInstance(summary_text, str)
        self.assertIn("GRID SEARCH OPTIMIZATION REPORT", summary_text)
        self.assertIn("Best Score:", summary_text)
        self.assertIn("Best Parameters:", summary_text)


class TestOptimizeStrategyFunction(unittest.TestCase):
    """Test class for the optimize_strategy convenience function."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        prices = 100 + np.cumsum(np.random.randn(50) * 0.01)

        self.data = pd.DataFrame(
            {
                "Open": prices * 0.99,
                "High": prices * 1.02,
                "Low": prices * 0.98,
                "Close": prices,
                "Volume": np.random.randint(1000000, 2000000, 50),
            },
            index=dates,
        )

        self.param_space = {
            "short_window": [5, 10],
            "long_window": [20, 25],
        }

    def test_optimize_strategy_function(self):
        """Test the optimize_strategy convenience function."""
        best_params, report = optimize_strategy(
            strategy_class=MovingAverageCrossStrategy,
            data=self.data,
            param_space=self.param_space,
            objective="total_return",
            initial_capital=5000,
        )

        self.assertIsNotNone(best_params)
        self.assertIsInstance(report, OptimizationReport)

        # Check that best params are valid
        self.assertIn(best_params["short_window"], self.param_space["short_window"])
        self.assertIn(best_params["long_window"], self.param_space["long_window"])

    def test_optimize_strategy_custom_objective(self):
        """Test optimize_strategy with custom objective function."""

        def custom_roi_objective(report):
            summary = report.summary()
            return summary.get("total_return", 0)

        best_params, report = optimize_strategy(
            strategy_class=MovingAverageCrossStrategy,
            data=self.data,
            param_space=self.param_space,
            objective=custom_roi_objective,
        )

        self.assertIsNotNone(best_params)
        self.assertIsInstance(report, OptimizationReport)


class TestOptimizationEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions for optimization module."""

    def setUp(self):
        """Set up test fixtures."""
        self.backtester = Backtester(initial_capital=10000)
        self.param_grid = {"short_window": [5], "long_window": [20]}

    def test_single_parameter_combination(self):
        """Test optimization with single parameter combination."""
        optimizer = GridSearchOptimizer(
            MovingAverageCrossStrategy, self.backtester, self.param_grid, "sharpe_ratio"
        )

        # Create minimal data
        data = pd.DataFrame(
            {
                "Close": [100, 101, 102, 103, 104],
                "Open": [99, 100, 101, 102, 103],
                "High": [101, 102, 103, 104, 105],
                "Low": [99, 100, 101, 102, 103],
                "Volume": [1000] * 5,
            },
            index=pd.date_range("2023-01-01", periods=5, freq="D"),
        )

        best_params, report = optimizer.optimize(data, verbose=False)

        self.assertEqual(len(optimizer.results), 1)
        self.assertEqual(best_params, {"short_window": 5, "long_window": 20})

    def test_invalid_parameter_values(self):
        """Test optimization with parameter values that cause strategy errors."""
        # Create param grid with invalid combinations
        invalid_param_grid = {
            "short_window": [50],  # Larger than long_window
            "long_window": [20],
        }

        optimizer = GridSearchOptimizer(
            MovingAverageCrossStrategy,
            self.backtester,
            invalid_param_grid,
            "sharpe_ratio",
        )

        data = pd.DataFrame(
            {
                "Close": [100, 101],
                "Open": [99, 100],
                "High": [101, 102],
                "Low": [99, 100],
                "Volume": [1000, 1000],
            },
            index=pd.date_range("2023-01-01", periods=2, freq="D"),
        )

        best_params, report = optimizer.optimize(data, verbose=False)

        # Should handle the error gracefully
        self.assertEqual(len(optimizer.results), 1)
        # The result should have failed
        self.assertFalse(np.isfinite(optimizer.results[0]["score"]))

    def test_optimization_with_insufficient_data(self):
        """Test optimization with insufficient data for strategy."""
        # Create very small dataset
        data = pd.DataFrame(
            {"Close": [100]},
            index=pd.date_range("2023-01-01", periods=1, freq="D"),
        )

        optimizer = GridSearchOptimizer(
            MovingAverageCrossStrategy, self.backtester, self.param_grid, "sharpe_ratio"
        )

        best_params, report = optimizer.optimize(data, verbose=False)

        # Should complete but may not generate meaningful results
        self.assertIsNotNone(best_params)
        self.assertIsInstance(report, OptimizationReport)


class TestOptimizationIntegration(unittest.TestCase):
    """Integration tests for optimization module with real data."""

    def setUp(self):
        """Set up test fixtures with realistic data."""
        # Create realistic price data with trend and volatility
        np.random.seed(42)  # For reproducible tests
        dates = pd.date_range("2023-01-01", periods=200, freq="D")

        # Generate realistic price series
        returns = np.random.normal(0.001, 0.02, 200)  # Daily returns
        prices = [100]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))

        self.data = pd.DataFrame(
            {
                "Open": [p * 0.995 for p in prices[:-1]],
                "High": [p * 1.01 for p in prices[:-1]],
                "Low": [p * 0.99 for p in prices[:-1]],
                "Close": prices[:-1],
                "Volume": np.random.randint(1000000, 5000000, 200),
            },
            index=dates,
        )

    def test_full_optimization_cycle(self):
        """Test complete optimization cycle with realistic parameters."""
        param_space = {
            "short_window": [10, 20, 30],
            "long_window": [50, 100, 150],
        }

        best_params, report = optimize_strategy(
            strategy_class=MovingAverageCrossStrategy,
            data=self.data,
            param_space=param_space,
            objective="sharpe_ratio",
            initial_capital=10000,
        )

        # Verify results
        self.assertIsNotNone(best_params)
        self.assertIn("short_window", best_params)
        self.assertIn("long_window", best_params)

        # Verify report
        summary = report.summary()
        self.assertEqual(summary["total_combinations"], 9)  # 3 * 3
        self.assertGreater(summary["successful_combinations"], 0)

        # Verify parameter sensitivity
        impact = report.get_parameter_impact()
        self.assertIn("short_window", impact)
        self.assertIn("long_window", impact)

    def test_optimization_different_objectives(self):
        """Test optimization with different objective functions."""
        param_space = {
            "short_window": [10, 20],
            "long_window": [50, 100],
        }

        objectives = ["sharpe_ratio", "total_return", "profit_factor", "win_rate"]

        for objective in objectives:
            with self.subTest(objective=objective):
                best_params, report = optimize_strategy(
                    strategy_class=MovingAverageCrossStrategy,
                    data=self.data,
                    param_space=param_space,
                    objective=objective,
                )

                self.assertIsNotNone(best_params)
                summary = report.summary()
                self.assertGreater(summary["successful_combinations"], 0)


if __name__ == "__main__":
    unittest.main()
