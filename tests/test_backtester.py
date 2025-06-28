"""
Tests for backtesting functionality.

This module tests the backtesting engine to ensure proper portfolio simulation,
transaction execution, and performance reporting.
"""

import unittest
import pandas as pd
import numpy as np
from simulation.backtester import Backtester, PerformanceReport


class TestBacktester(unittest.TestCase):
    """Test class for Backtester functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample price data
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        prices = np.linspace(100, 150, 50)  # Steady uptrend

        self.sample_data = pd.DataFrame(
            {
                "Open": prices * 0.99,
                "High": prices * 1.02,
                "Low": prices * 0.98,
                "Close": prices,
                "Volume": np.random.randint(1000000, 2000000, 50),
            },
            index=dates,
        )

        # Create simple trading signals
        signals = ["HOLD"] * 50
        signals[10] = "BUY"  # Buy signal at day 10
        signals[30] = "SELL"  # Sell signal at day 30
        signals[40] = "BUY"  # Buy again at day 40

        self.sample_signals = pd.DataFrame({"Signal": signals}, index=dates)

        # Create data for loss scenario
        declining_prices = np.linspace(100, 70, 30)  # Declining trend
        self.declining_data = pd.DataFrame(
            {
                "Open": declining_prices * 0.99,
                "High": declining_prices * 1.02,
                "Low": declining_prices * 0.98,
                "Close": declining_prices,
                "Volume": [1000000] * 30,
            },
            index=pd.date_range("2023-01-01", periods=30, freq="D"),
        )

        declining_signals = ["HOLD"] * 30
        declining_signals[5] = "BUY"
        declining_signals[25] = "SELL"

        self.declining_signals = pd.DataFrame(
            {"Signal": declining_signals}, index=self.declining_data.index
        )

    def test_backtester_initialization_valid_params(self):
        """Test backtester initialization with valid parameters."""
        backtester = Backtester(
            initial_capital=10000, commission=0.001, slippage=0.0005
        )

        self.assertEqual(backtester.initial_capital, 10000)
        self.assertEqual(backtester.commission, 0.001)
        self.assertEqual(backtester.slippage, 0.0005)
        self.assertEqual(backtester.cash, 10000)
        self.assertEqual(backtester.position, 0)

    def test_backtester_initialization_invalid_params(self):
        """Test backtester initialization with invalid parameters."""
        # Negative initial capital
        with self.assertRaises(ValueError):
            Backtester(initial_capital=-1000)

        # Zero initial capital
        with self.assertRaises(ValueError):
            Backtester(initial_capital=0)

        # Negative commission
        with self.assertRaises(ValueError):
            Backtester(initial_capital=10000, commission=-0.01)

        # Negative slippage
        with self.assertRaises(ValueError):
            Backtester(initial_capital=10000, slippage=-0.01)

    def test_backtester_reset(self):
        """Test backtester reset functionality."""
        backtester = Backtester(initial_capital=10000)

        # Modify state
        backtester.cash = 8000
        backtester.position = 100
        backtester.transactions = [{"test": "transaction"}]
        backtester.portfolio_history = [{"test": "history"}]

        # Reset
        backtester.reset()

        self.assertEqual(backtester.cash, 10000)
        self.assertEqual(backtester.position, 0)
        self.assertEqual(len(backtester.transactions), 0)
        self.assertEqual(len(backtester.portfolio_history), 0)

    def test_calculate_transaction_cost(self):
        """Test transaction cost calculation."""
        backtester = Backtester(
            initial_capital=10000, commission=0.001, slippage=0.0005
        )

        trade_value = 1000
        cost = backtester._calculate_transaction_cost(trade_value)

        expected_cost = trade_value * (0.001 + 0.0005)
        self.assertEqual(cost, expected_cost)

    def test_execute_trade_buy_sufficient_cash(self):
        """Test executing a buy trade with sufficient cash."""
        backtester = Backtester(
            initial_capital=10000, commission=0.001, slippage=0.0005
        )

        # Buy 10 shares at $100 each
        date = pd.Timestamp("2023-01-01")
        success = backtester._execute_trade(100, 10, "BUY", date)

        self.assertTrue(success)
        self.assertEqual(backtester.position, 10)

        # Check cash after transaction costs
        trade_value = 1000
        transaction_cost = trade_value * 0.0015  # commission + slippage
        expected_cash = 10000 - trade_value - transaction_cost
        self.assertAlmostEqual(backtester.cash, expected_cash, places=2)

        # Check transaction recorded
        self.assertEqual(len(backtester.transactions), 1)
        self.assertEqual(backtester.transactions[0]["type"], "BUY")

    def test_execute_trade_buy_insufficient_cash(self):
        """Test executing a buy trade with insufficient cash."""
        backtester = Backtester(initial_capital=1000, commission=0.001, slippage=0.0005)

        # Try to buy 100 shares at $100 each (need $10,000 but only have $1,000)
        date = pd.Timestamp("2023-01-01")
        success = backtester._execute_trade(100, 100, "BUY", date)

        self.assertFalse(success)
        self.assertEqual(backtester.position, 0)
        self.assertEqual(backtester.cash, 1000)  # Cash unchanged
        self.assertEqual(len(backtester.transactions), 0)  # No transaction recorded

    def test_execute_trade_sell_sufficient_position(self):
        """Test executing a sell trade with sufficient position."""
        backtester = Backtester(initial_capital=10000)

        # First buy some shares
        date = pd.Timestamp("2023-01-01")
        backtester._execute_trade(100, 10, "BUY", date)

        # Then sell them
        sell_date = pd.Timestamp("2023-01-02")
        success = backtester._execute_trade(110, 10, "SELL", sell_date)

        self.assertTrue(success)
        self.assertEqual(backtester.position, 0)

        # Check cash increased
        self.assertGreater(backtester.cash, 10000)  # Should have made a profit

        # Check transactions recorded
        self.assertEqual(len(backtester.transactions), 2)
        self.assertEqual(backtester.transactions[1]["type"], "SELL")

    def test_execute_trade_sell_partial_position(self):
        """Test executing a sell trade with more shares than owned."""
        backtester = Backtester(initial_capital=10000)

        # Buy 5 shares
        date = pd.Timestamp("2023-01-01")
        backtester._execute_trade(100, 5, "BUY", date)

        # Try to sell 10 shares (only have 5)
        sell_date = pd.Timestamp("2023-01-02")
        success = backtester._execute_trade(110, 10, "SELL", sell_date)

        self.assertTrue(success)  # Should succeed but only sell what's available
        self.assertEqual(backtester.position, 0)  # All shares sold

        # Check that only 5 shares were sold
        sell_transaction = backtester.transactions[1]
        self.assertEqual(sell_transaction["quantity"], 5)

    def test_run_backtest_basic_scenario(self):
        """Test running a basic backtest scenario."""
        backtester = Backtester(initial_capital=10000)

        report = backtester.run_backtest(self.sample_data, self.sample_signals)

        self.assertIsInstance(report, PerformanceReport)
        self.assertFalse(report.portfolio_history.empty)
        self.assertGreater(len(report.transactions), 0)

        # Should have made some trades
        transactions_df = report.get_transactions_df()
        self.assertGreater(len(transactions_df), 0)

        # Should have both buy and sell transactions
        transaction_types = transactions_df["type"].unique()
        self.assertIn("BUY", transaction_types)
        self.assertIn("SELL", transaction_types)

    def test_run_backtest_empty_data(self):
        """Test running backtest with empty data."""
        backtester = Backtester(initial_capital=10000)

        empty_data = pd.DataFrame()
        empty_signals = pd.DataFrame()

        report = backtester.run_backtest(empty_data, empty_signals)

        self.assertIsInstance(report, PerformanceReport)
        self.assertTrue(report.portfolio_history.empty)
        self.assertEqual(len(report.transactions), 0)
        self.assertEqual(report.final_value, report.initial_capital)

    def test_run_backtest_no_signals(self):
        """Test running backtest with no trading signals."""
        backtester = Backtester(initial_capital=10000)

        # All HOLD signals
        hold_signals = pd.DataFrame(
            {"Signal": ["HOLD"] * len(self.sample_data)}, index=self.sample_data.index
        )

        report = backtester.run_backtest(self.sample_data, hold_signals)

        # Should have no transactions
        self.assertEqual(len(report.transactions), 0)
        # Final value should be initial capital (no trades made)
        self.assertEqual(report.final_value, report.initial_capital)

    def test_run_backtest_losing_scenario(self):
        """Test running backtest in a losing scenario."""
        backtester = Backtester(initial_capital=10000)

        report = backtester.run_backtest(self.declining_data, self.declining_signals)

        # Should have lost money
        self.assertLess(report.final_value, report.initial_capital)

        # Should still generate a valid report
        summary = report.summary()
        self.assertIn("total_return", summary)
        self.assertLess(summary["total_return"], 0)  # Negative return


class TestPerformanceReport(unittest.TestCase):
    """Test class for PerformanceReport functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample portfolio history
        dates = pd.date_range("2023-01-01", periods=30, freq="D")

        # Simulate portfolio growth
        initial_value = 10000
        values = [
            initial_value * (1 + i * 0.001) for i in range(30)
        ]  # Small daily growth

        self.portfolio_history = pd.DataFrame(
            {
                "Total_Value": values,
                "Cash": [initial_value * 0.5] * 30,
                "Position_Value": [initial_value * 0.5] * 30,
            },
            index=dates,
        )

        # Create sample transactions
        self.transactions = [
            {
                "date": dates[5],
                "type": "BUY",
                "price": 100,
                "quantity": 50,
                "pnl": -50,  # Loss transaction
            },
            {
                "date": dates[15],
                "type": "SELL",
                "price": 110,
                "quantity": 50,
                "pnl": 450,  # Profit transaction
            },
        ]

        self.initial_capital = 10000
        self.final_value = values[-1]

    def test_performance_report_initialization(self):
        """Test PerformanceReport initialization."""
        report = PerformanceReport(
            self.portfolio_history,
            self.transactions,
            self.initial_capital,
            self.final_value,
        )

        self.assertEqual(report.initial_capital, self.initial_capital)
        self.assertEqual(report.final_value, self.final_value)
        pd.testing.assert_frame_equal(report.portfolio_history, self.portfolio_history)
        self.assertEqual(report.transactions, self.transactions)

    def test_performance_report_summary(self):
        """Test performance report summary generation."""
        report = PerformanceReport(
            self.portfolio_history,
            self.transactions,
            self.initial_capital,
            self.final_value,
        )

        summary = report.summary()

        # Check required fields
        required_fields = [
            "initial_capital",
            "final_value",
            "total_return",
            "total_return_pct",
            "annualized_return",
            "volatility",
            "sharpe_ratio",
            "max_drawdown_pct",
            "num_trades",
            "win_rate",
            "avg_win",
            "avg_loss",
            "profit_factor",
        ]

        for field in required_fields:
            self.assertIn(field, summary)

        # Check basic calculations
        expected_return = (
            self.final_value - self.initial_capital
        ) / self.initial_capital
        self.assertAlmostEqual(summary["total_return"], expected_return, places=4)

        self.assertEqual(summary["num_trades"], len(self.transactions))

        # Win rate should be between 0 and 1
        self.assertGreaterEqual(summary["win_rate"], 0)
        self.assertLessEqual(summary["win_rate"], 1)

    def test_performance_report_summary_empty_data(self):
        """Test performance report summary with empty data."""
        empty_portfolio = pd.DataFrame()
        empty_transactions = []

        report = PerformanceReport(empty_portfolio, empty_transactions, 10000, 10000)
        summary = report.summary()

        self.assertEqual(summary, {})

    def test_get_transactions_df(self):
        """Test converting transactions to DataFrame."""
        report = PerformanceReport(
            self.portfolio_history,
            self.transactions,
            self.initial_capital,
            self.final_value,
        )

        transactions_df = report.get_transactions_df()

        self.assertIsInstance(transactions_df, pd.DataFrame)
        self.assertEqual(len(transactions_df), len(self.transactions))

        # Check columns
        expected_columns = ["date", "type", "price", "quantity", "pnl"]
        for col in expected_columns:
            self.assertIn(col, transactions_df.columns)

    def test_get_transactions_df_empty(self):
        """Test converting empty transactions to DataFrame."""
        report = PerformanceReport(
            self.portfolio_history, [], self.initial_capital, self.final_value
        )

        transactions_df = report.get_transactions_df()

        self.assertIsInstance(transactions_df, pd.DataFrame)
        self.assertTrue(transactions_df.empty)

    def test_summary_calculations_edge_cases(self):
        """Test summary calculations with edge cases."""
        # Create portfolio with no change
        flat_values = [10000] * 30
        flat_portfolio = pd.DataFrame(
            {"Total_Value": flat_values},
            index=pd.date_range("2023-01-01", periods=30, freq="D"),
        )

        report = PerformanceReport(flat_portfolio, [], 10000, 10000)
        summary = report.summary()

        # Zero return scenario
        self.assertEqual(summary["total_return"], 0)
        self.assertEqual(summary["total_return_pct"], 0)
        self.assertEqual(summary["volatility"], 0)  # No volatility in flat returns

    def test_drawdown_calculation(self):
        """Test maximum drawdown calculation."""
        # Create portfolio with significant drawdown
        values = [
            10000,
            12000,
            11000,
            8000,
            9000,
            11000,
            13000,
        ]  # Peak of 12000, trough of 8000
        drawdown_portfolio = pd.DataFrame(
            {"Total_Value": values},
            index=pd.date_range("2023-01-01", periods=7, freq="D"),
        )

        report = PerformanceReport(drawdown_portfolio, [], 10000, 13000)
        summary = report.summary()

        # Maximum drawdown should be (8000 - 12000) / 12000 = -33.33%
        expected_max_drawdown = -33.33
        self.assertAlmostEqual(
            summary["max_drawdown_pct"], expected_max_drawdown, places=1
        )


if __name__ == "__main__":
    unittest.main()
