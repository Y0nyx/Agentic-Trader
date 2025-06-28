"""
Tests for the evaluation module.

This module contains comprehensive tests for the performance evaluation
functionality including metrics calculation and reporting.
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from simulation.backtester import PerformanceReport, Backtester
from evaluation.metrics import (
    evaluate_performance,
    calculate_roi_metrics,
    calculate_risk_metrics,
    calculate_trade_metrics,
    _calculate_advanced_metrics,
    _calculate_consecutive_trades,
    _calculate_benchmark_comparison,
)
from evaluation.report import PerformanceReporter, create_hold_strategy_benchmark


class TestEvaluationMetrics(unittest.TestCase):
    """Test class for evaluation metrics functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample portfolio history
        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        values = np.linspace(10000, 12000, 30)  # Upward trend
        
        self.portfolio_history = pd.DataFrame(
            {"Total_Value": values},
            index=dates
        )
        
        # Create sample transactions with P&L
        self.transactions = [
            {
                "date": dates[5],
                "type": "BUY",
                "price": 100,
                "quantity": 50,
                "pnl": 0,
            },
            {
                "date": dates[15],
                "type": "SELL",
                "price": 110,
                "quantity": 50,
                "pnl": 450,  # Profit transaction
            },
            {
                "date": dates[20],
                "type": "BUY",
                "price": 115,
                "quantity": 40,
                "pnl": 0,
            },
            {
                "date": dates[25],
                "type": "SELL",
                "price": 105,
                "quantity": 40,
                "pnl": -450,  # Loss transaction
            },
        ]
        
        self.initial_capital = 10000
        self.final_value = 12000
        
        self.performance_report = PerformanceReport(
            self.portfolio_history,
            self.transactions,
            self.initial_capital,
            self.final_value,
        )
        
        # Create benchmark data
        self.benchmark_data = pd.DataFrame(
            {"Close": np.linspace(100, 115, 30)},
            index=dates
        )

    def test_evaluate_performance_basic(self):
        """Test basic performance evaluation."""
        metrics = evaluate_performance(self.performance_report)
        
        # Check that basic metrics are included
        self.assertIn("total_return", metrics)
        self.assertIn("sharpe_ratio", metrics)
        self.assertIn("max_drawdown_pct", metrics)
        
        # Check advanced metrics are included
        self.assertIn("total_net_profit", metrics)
        self.assertIn("sortino_ratio", metrics)
        self.assertIn("var_5_percent", metrics)
        
        # Verify calculations
        expected_return = (self.final_value - self.initial_capital) / self.initial_capital
        self.assertAlmostEqual(metrics["total_return"], expected_return, places=4)

    def test_evaluate_performance_with_benchmark(self):
        """Test performance evaluation with benchmark comparison."""
        metrics = evaluate_performance(self.performance_report, self.benchmark_data)
        
        # Check benchmark metrics are included
        self.assertIn("benchmark_return", metrics)
        self.assertIn("alpha", metrics)
        self.assertIn("outperformed_benchmark", metrics)
        
        # Verify benchmark calculations
        self.assertIsInstance(metrics["benchmark_return"], float)
        self.assertIsInstance(metrics["alpha"], float)
        self.assertIsInstance(metrics["outperformed_benchmark"], bool)

    def test_evaluate_performance_empty_report(self):
        """Test evaluation with empty performance report."""
        empty_report = PerformanceReport(
            pd.DataFrame(), [], 10000, 10000
        )
        
        metrics = evaluate_performance(empty_report)
        self.assertEqual(metrics, {})

    def test_calculate_advanced_metrics(self):
        """Test advanced metrics calculation."""
        advanced = _calculate_advanced_metrics(self.performance_report)
        
        # Check all expected advanced metrics
        expected_keys = [
            "total_net_profit", "avg_profit_per_trade", "profit_std",
            "cumulative_profit_final", "sortino_ratio", "calmar_ratio",
            "downside_volatility", "var_5_percent", "cvar_5_percent",
            "max_consecutive_wins", "max_consecutive_losses", "portfolio_volatility_pct"
        ]
        
        for key in expected_keys:
            self.assertIn(key, advanced)
            self.assertIsInstance(advanced[key], (int, float))

    def test_calculate_consecutive_trades(self):
        """Test consecutive trades calculation."""
        # Test with mixed wins/losses
        transactions = [
            {"pnl": 100}, {"pnl": 200}, {"pnl": -50}, 
            {"pnl": -100}, {"pnl": -75}, {"pnl": 150}
        ]
        
        max_wins, max_losses = _calculate_consecutive_trades(transactions)
        self.assertEqual(max_wins, 2)  # First two trades
        self.assertEqual(max_losses, 3)  # Middle three trades

    def test_calculate_consecutive_trades_edge_cases(self):
        """Test consecutive trades with edge cases."""
        # Empty transactions
        max_wins, max_losses = _calculate_consecutive_trades([])
        self.assertEqual(max_wins, 0)
        self.assertEqual(max_losses, 0)
        
        # No P&L data
        max_wins, max_losses = _calculate_consecutive_trades([{"type": "BUY"}])
        self.assertEqual(max_wins, 0)
        self.assertEqual(max_losses, 0)
        
        # All wins
        all_wins = [{"pnl": 100}, {"pnl": 200}, {"pnl": 50}]
        max_wins, max_losses = _calculate_consecutive_trades(all_wins)
        self.assertEqual(max_wins, 3)
        self.assertEqual(max_losses, 0)

    def test_calculate_benchmark_comparison(self):
        """Test benchmark comparison calculation."""
        comparison = _calculate_benchmark_comparison(
            self.performance_report, self.benchmark_data
        )
        
        expected_keys = [
            "benchmark_return", "benchmark_return_pct", "benchmark_final_value",
            "benchmark_volatility", "benchmark_max_drawdown_pct",
            "alpha", "alpha_pct", "outperformed_benchmark"
        ]
        
        for key in expected_keys:
            self.assertIn(key, comparison)

    def test_calculate_roi_metrics(self):
        """Test ROI metrics calculation."""
        roi_metrics = calculate_roi_metrics(self.performance_report)
        
        expected_keys = [
            "roi_total", "roi_total_pct", "roi_annualized", 
            "roi_annualized_pct", "sharpe_ratio"
        ]
        
        for key in expected_keys:
            self.assertIn(key, roi_metrics)
            self.assertIsInstance(roi_metrics[key], (int, float))

    def test_calculate_risk_metrics(self):
        """Test risk metrics calculation."""
        risk_metrics = calculate_risk_metrics(self.performance_report)
        
        expected_keys = [
            "volatility_annualized", "volatility_pct", "max_drawdown",
            "max_drawdown_pct", "avg_drawdown", "avg_drawdown_pct",
            "downside_volatility", "downside_volatility_pct"
        ]
        
        for key in expected_keys:
            self.assertIn(key, risk_metrics)
            self.assertIsInstance(risk_metrics[key], (int, float))

    def test_calculate_trade_metrics(self):
        """Test trade metrics calculation."""
        trade_metrics = calculate_trade_metrics(self.performance_report)
        
        expected_keys = [
            "total_trades", "winning_trades", "losing_trades", "win_rate",
            "win_rate_pct", "avg_win", "avg_loss", "profit_factor",
            "largest_win", "largest_loss", "total_gross_profit",
            "total_gross_loss", "net_profit"
        ]
        
        for key in expected_keys:
            self.assertIn(key, trade_metrics)

        # Verify specific calculations
        # Note: All transactions with P&L data are counted (including BUY with pnl=0)
        self.assertEqual(trade_metrics["total_trades"], 4)  # 4 trades with P&L data
        self.assertEqual(trade_metrics["winning_trades"], 1)  # 1 trade with positive P&L
        self.assertEqual(trade_metrics["losing_trades"], 1)   # 1 trade with negative P&L
        self.assertEqual(trade_metrics["win_rate"], 0.25)     # 1/4 = 0.25
        self.assertEqual(trade_metrics["largest_win"], 450)
        self.assertEqual(trade_metrics["largest_loss"], -450)

    def test_calculate_trade_metrics_no_transactions(self):
        """Test trade metrics with no transactions."""
        empty_report = PerformanceReport(
            self.portfolio_history, [], self.initial_capital, self.final_value
        )
        
        trade_metrics = calculate_trade_metrics(empty_report)
        self.assertEqual(trade_metrics["total_trades"], 0)
        self.assertEqual(trade_metrics["win_rate"], 0)


class TestPerformanceReporter(unittest.TestCase):
    """Test class for PerformanceReporter functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample data similar to TestEvaluationMetrics
        dates = pd.date_range("2023-01-01", periods=20, freq="D")
        values = np.linspace(10000, 11000, 20)
        
        portfolio_history = pd.DataFrame(
            {"Total_Value": values},
            index=dates
        )
        
        transactions = [
            {
                "date": dates[5],
                "type": "BUY",
                "price": 100,
                "quantity": 50,
                "pnl": 0,
            },
            {
                "date": dates[15],
                "type": "SELL",
                "price": 110,
                "quantity": 50,
                "pnl": 450,
            },
        ]
        
        self.performance_report = PerformanceReport(
            portfolio_history, transactions, 10000, 11000
        )
        
        self.reporter = PerformanceReporter(self.performance_report)
        
        # Create benchmark data
        self.benchmark_data = pd.DataFrame(
            {"Close": np.linspace(100, 105, 20)},
            index=dates
        )

    def test_reporter_initialization(self):
        """Test reporter initialization."""
        self.assertEqual(self.reporter.performance_report, self.performance_report)
        self.assertIsNone(self.reporter.metrics)

    def test_generate_comprehensive_report(self):
        """Test comprehensive report generation."""
        report = self.reporter.generate_comprehensive_report()
        
        expected_sections = [
            "summary", "performance_metrics", "risk_analysis",
            "trade_analysis", "portfolio_evolution"
        ]
        
        for section in expected_sections:
            self.assertIn(section, report)

    def test_generate_comprehensive_report_with_benchmark(self):
        """Test comprehensive report with benchmark."""
        report = self.reporter.generate_comprehensive_report(self.benchmark_data)
        
        self.assertIn("benchmark_comparison", report)
        benchmark = report["benchmark_comparison"]
        self.assertIn("benchmark_return_pct", benchmark)

    def test_generate_comprehensive_report_with_charts(self):
        """Test comprehensive report with chart data."""
        report = self.reporter.generate_comprehensive_report(include_charts=True)
        
        self.assertIn("chart_data", report)
        chart_data = report["chart_data"]
        self.assertIn("portfolio_evolution", chart_data)
        self.assertIn("transactions", chart_data)

    def test_print_summary_report(self):
        """Test formatted summary report generation."""
        summary_text = self.reporter.print_summary_report()
        
        self.assertIsInstance(summary_text, str)
        self.assertIn("TRADING STRATEGY PERFORMANCE REPORT", summary_text)
        self.assertIn("EXECUTIVE SUMMARY", summary_text)
        self.assertIn("Initial Capital", summary_text)
        self.assertIn("Final Value", summary_text)

    def test_print_summary_report_with_benchmark(self):
        """Test formatted summary with benchmark."""
        summary_text = self.reporter.print_summary_report(self.benchmark_data)
        
        self.assertIn("BENCHMARK COMPARISON", summary_text)
        self.assertIn("Benchmark Return", summary_text)
        self.assertIn("Alpha", summary_text)


class TestHoldStrategyBenchmark(unittest.TestCase):
    """Test class for hold strategy benchmark functionality."""

    def setUp(self):
        """Set up test fixtures."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        prices = np.linspace(100, 110, 10)
        
        self.data = pd.DataFrame(
            {"Close": prices},
            index=dates
        )

    def test_create_hold_strategy_benchmark(self):
        """Test hold strategy benchmark creation."""
        benchmark = create_hold_strategy_benchmark(self.data, 10000)
        
        self.assertIn("Total_Value", benchmark.columns)
        self.assertIn("Daily_Return", benchmark.columns)
        
        # Check initial value calculation
        initial_price = self.data["Close"].iloc[0]
        shares = 10000 / initial_price
        expected_final_value = shares * self.data["Close"].iloc[-1]
        
        self.assertAlmostEqual(
            benchmark["Total_Value"].iloc[-1], expected_final_value, places=2
        )

    def test_create_hold_strategy_benchmark_empty_data(self):
        """Test hold strategy with empty data."""
        empty_data = pd.DataFrame()
        benchmark = create_hold_strategy_benchmark(empty_data)
        
        self.assertTrue(benchmark.empty)

    def test_create_hold_strategy_benchmark_missing_close(self):
        """Test hold strategy with missing Close column."""
        invalid_data = pd.DataFrame({"Price": [100, 110]})
        benchmark = create_hold_strategy_benchmark(invalid_data)
        
        self.assertTrue(benchmark.empty)


class TestEvaluationEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions for evaluation module."""

    def test_evaluate_performance_invalid_benchmark(self):
        """Test evaluation with invalid benchmark data."""
        # Create minimal valid performance report
        portfolio = pd.DataFrame(
            {"Total_Value": [10000, 11000]},
            index=pd.date_range("2023-01-01", periods=2, freq="D")
        )
        
        report = PerformanceReport(portfolio, [], 10000, 11000)
        
        # Test with empty benchmark
        empty_benchmark = pd.DataFrame()
        metrics = evaluate_performance(report, empty_benchmark)
        self.assertNotIn("benchmark_return", metrics)
        
        # Test with misaligned benchmark dates
        wrong_dates = pd.DataFrame(
            {"Close": [100, 110]},
            index=pd.date_range("2022-01-01", periods=2, freq="D")
        )
        metrics = evaluate_performance(report, wrong_dates)
        # Should still work but benchmark metrics might be empty

    def test_metrics_with_zero_volatility(self):
        """Test metrics calculation with zero volatility portfolio."""
        # Create flat portfolio (no volatility)
        portfolio = pd.DataFrame(
            {"Total_Value": [10000] * 10},
            index=pd.date_range("2023-01-01", periods=10, freq="D")
        )
        
        report = PerformanceReport(portfolio, [], 10000, 10000)
        metrics = evaluate_performance(report)
        
        # Should handle zero volatility gracefully
        self.assertEqual(metrics["volatility"], 0)
        self.assertEqual(metrics["sharpe_ratio"], 0)

    def test_metrics_with_single_data_point(self):
        """Test metrics with single data point."""
        portfolio = pd.DataFrame(
            {"Total_Value": [11000]},
            index=pd.date_range("2023-01-01", periods=1, freq="D")
        )
        
        report = PerformanceReport(portfolio, [], 10000, 11000)
        metrics = evaluate_performance(report)
        
        # Should handle single point gracefully
        self.assertIn("total_return", metrics)
        self.assertEqual(metrics["volatility"], 0)


if __name__ == "__main__":
    unittest.main()