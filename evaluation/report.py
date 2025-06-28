"""
Performance reporting and visualization module.

This module provides enhanced reporting capabilities with visualizations
and comprehensive analysis of trading strategy performance.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from simulation.backtester import PerformanceReport
from .metrics import evaluate_performance

# Configure logging
logger = logging.getLogger(__name__)


class PerformanceReporter:
    """
    Enhanced performance reporting with visualization and analysis capabilities.
    
    This class provides comprehensive reporting functionality that extends
    the basic PerformanceReport with advanced analytics and visualization.
    """
    
    def __init__(self, performance_report: PerformanceReport):
        """
        Initialize the performance reporter.
        
        Parameters
        ----------
        performance_report : PerformanceReport
            Backtest results from the simulation module
        """
        self.performance_report = performance_report
        self.metrics = None
        
    def generate_comprehensive_report(
        self, 
        benchmark_data: Optional[pd.DataFrame] = None,
        include_charts: bool = False
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report.
        
        Parameters
        ----------
        benchmark_data : pd.DataFrame, optional
            Benchmark data for comparison
        include_charts : bool, default False
            Whether to include chart data (for future visualization)
            
        Returns
        -------
        Dict[str, Any]
            Comprehensive report with all metrics and analysis
        """
        logger.info("Generating comprehensive performance report")
        
        # Calculate all metrics
        self.metrics = evaluate_performance(self.performance_report, benchmark_data)
        
        # Build comprehensive report
        report = {
            "summary": self._generate_summary(),
            "performance_metrics": self._organize_performance_metrics(),
            "risk_analysis": self._generate_risk_analysis(),
            "trade_analysis": self._generate_trade_analysis(),
            "portfolio_evolution": self._generate_portfolio_evolution(),
        }
        
        # Add benchmark comparison if available
        if benchmark_data is not None:
            report["benchmark_comparison"] = self._generate_benchmark_comparison()
        
        # Add chart data if requested
        if include_charts:
            report["chart_data"] = self._generate_chart_data()
            
        logger.info("Comprehensive report generated successfully")
        return report
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate executive summary of performance."""
        if not self.metrics:
            return {}
            
        return {
            "initial_capital": self.metrics.get("initial_capital", 0),
            "final_value": self.metrics.get("final_value", 0),
            "total_return_pct": self.metrics.get("total_return_pct", 0),
            "annualized_return_pct": self.metrics.get("annualized_return", 0) * 100,
            "sharpe_ratio": self.metrics.get("sharpe_ratio", 0),
            "max_drawdown_pct": self.metrics.get("max_drawdown_pct", 0),
            "total_trades": self.metrics.get("num_trades", 0),
            "win_rate_pct": self.metrics.get("win_rate", 0) * 100,
            "profit_factor": self.metrics.get("profit_factor", 0),
            "trading_period_days": self.metrics.get("trading_days", 0),
        }
    
    def _organize_performance_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Organize metrics into logical categories."""
        if not self.metrics:
            return {}
            
        return {
            "returns": {
                "total_return": self.metrics.get("total_return", 0),
                "total_return_pct": self.metrics.get("total_return_pct", 0),
                "annualized_return": self.metrics.get("annualized_return", 0),
                "total_net_profit": self.metrics.get("total_net_profit", 0),
                "avg_profit_per_trade": self.metrics.get("avg_profit_per_trade", 0),
            },
            "risk_adjusted": {
                "sharpe_ratio": self.metrics.get("sharpe_ratio", 0),
                "sortino_ratio": self.metrics.get("sortino_ratio", 0),
                "calmar_ratio": self.metrics.get("calmar_ratio", 0),
            },
            "volatility": {
                "volatility": self.metrics.get("volatility", 0),
                "downside_volatility": self.metrics.get("downside_volatility", 0),
                "portfolio_volatility_pct": self.metrics.get("portfolio_volatility_pct", 0),
            }
        }
    
    def _generate_risk_analysis(self) -> Dict[str, Any]:
        """Generate detailed risk analysis."""
        if not self.metrics:
            return {}
            
        return {
            "drawdown_analysis": {
                "max_drawdown_pct": self.metrics.get("max_drawdown_pct", 0),
                "avg_drawdown_pct": self.metrics.get("avg_drawdown_pct", 0),
            },
            "value_at_risk": {
                "var_5_percent": self.metrics.get("var_5_percent", 0),
                "cvar_5_percent": self.metrics.get("cvar_5_percent", 0),
            },
            "volatility_metrics": {
                "total_volatility": self.metrics.get("volatility", 0),
                "downside_volatility": self.metrics.get("downside_volatility", 0),
            }
        }
    
    def _generate_trade_analysis(self) -> Dict[str, Any]:
        """Generate detailed trade analysis."""
        if not self.metrics:
            return {}
            
        return {
            "trade_statistics": {
                "total_trades": self.metrics.get("num_trades", 0),
                "winning_trades": len([t for t in self.performance_report.transactions if t.get("pnl", 0) > 0]),
                "losing_trades": len([t for t in self.performance_report.transactions if t.get("pnl", 0) < 0]),
                "win_rate": self.metrics.get("win_rate", 0),
                "win_rate_pct": self.metrics.get("win_rate", 0) * 100,
            },
            "profit_analysis": {
                "avg_win": self.metrics.get("avg_win", 0),
                "avg_loss": self.metrics.get("avg_loss", 0),
                "profit_factor": self.metrics.get("profit_factor", 0),
                "largest_win": self.metrics.get("largest_win", 0),
                "largest_loss": self.metrics.get("largest_loss", 0),
            },
            "consecutive_trades": {
                "max_consecutive_wins": self.metrics.get("max_consecutive_wins", 0),
                "max_consecutive_losses": self.metrics.get("max_consecutive_losses", 0),
            }
        }
    
    def _generate_portfolio_evolution(self) -> Dict[str, Any]:
        """Generate portfolio evolution analysis."""
        portfolio = self.performance_report.portfolio_history
        
        if portfolio.empty:
            return {}
        
        # Calculate key evolution metrics
        portfolio_copy = portfolio.copy()
        portfolio_copy["Cumulative_Return"] = (
            (portfolio_copy["Total_Value"] / self.performance_report.initial_capital) - 1
        ) * 100
        
        return {
            "start_date": portfolio.index.min().strftime("%Y-%m-%d") if isinstance(portfolio.index, pd.DatetimeIndex) else None,
            "end_date": portfolio.index.max().strftime("%Y-%m-%d") if isinstance(portfolio.index, pd.DatetimeIndex) else None,
            "total_days": len(portfolio),
            "peak_value": portfolio["Total_Value"].max(),
            "lowest_value": portfolio["Total_Value"].min(),
            "peak_date": portfolio["Total_Value"].idxmax().strftime("%Y-%m-%d") if isinstance(portfolio.index, pd.DatetimeIndex) else None,
            "lowest_date": portfolio["Total_Value"].idxmin().strftime("%Y-%m-%d") if isinstance(portfolio.index, pd.DatetimeIndex) else None,
        }
    
    def _generate_benchmark_comparison(self) -> Dict[str, Any]:
        """Generate benchmark comparison analysis."""
        if not self.metrics:
            return {}
            
        benchmark_metrics = {
            key: value for key, value in self.metrics.items() 
            if key.startswith("benchmark_") or key in ["alpha", "alpha_pct", "outperformed_benchmark"]
        }
        
        return benchmark_metrics
    
    def _generate_chart_data(self) -> Dict[str, Any]:
        """Generate data for charts and visualizations."""
        portfolio = self.performance_report.portfolio_history
        
        if portfolio.empty:
            return {}
        
        # Prepare time series data
        portfolio_evolution = portfolio[["Total_Value"]].copy()
        portfolio_evolution["Cumulative_Return_Pct"] = (
            (portfolio_evolution["Total_Value"] / self.performance_report.initial_capital) - 1
        ) * 100
        
        # Calculate drawdown series
        cumulative_max = portfolio_evolution["Total_Value"].expanding().max()
        portfolio_evolution["Drawdown_Pct"] = (
            (portfolio_evolution["Total_Value"] - cumulative_max) / cumulative_max
        ) * 100
        
        # Prepare transaction data
        transactions_df = self.performance_report.get_transactions_df()
        
        chart_data = {
            "portfolio_evolution": portfolio_evolution.to_dict("index"),
            "transactions": transactions_df.to_dict("records") if not transactions_df.empty else [],
        }
        
        # Add profit distribution if we have P&L data
        if not transactions_df.empty and "pnl" in transactions_df.columns:
            profits = transactions_df["pnl"].dropna()
            if len(profits) > 0:
                chart_data["profit_distribution"] = {
                    "profits": profits.tolist(),
                    "bins": 20,
                    "positive_count": len(profits[profits > 0]),
                    "negative_count": len(profits[profits < 0]),
                }
        
        return chart_data
    
    def print_summary_report(self, benchmark_data: Optional[pd.DataFrame] = None) -> str:
        """
        Generate a formatted text summary report.
        
        Parameters
        ----------
        benchmark_data : pd.DataFrame, optional
            Benchmark data for comparison
            
        Returns
        -------
        str
            Formatted text report
        """
        report = self.generate_comprehensive_report(benchmark_data)
        
        lines = []
        lines.append("=" * 60)
        lines.append("TRADING STRATEGY PERFORMANCE REPORT")
        lines.append("=" * 60)
        
        # Summary section
        summary = report.get("summary", {})
        lines.append("\nEXECUTIVE SUMMARY")
        lines.append("-" * 30)
        lines.append(f"Initial Capital:      ${summary.get('initial_capital', 0):,.2f}")
        lines.append(f"Final Value:          ${summary.get('final_value', 0):,.2f}")
        lines.append(f"Total Return:         {summary.get('total_return_pct', 0):.2f}%")
        lines.append(f"Annualized Return:    {summary.get('annualized_return_pct', 0):.2f}%")
        lines.append(f"Sharpe Ratio:         {summary.get('sharpe_ratio', 0):.3f}")
        lines.append(f"Max Drawdown:         {summary.get('max_drawdown_pct', 0):.2f}%")
        lines.append(f"Total Trades:         {summary.get('total_trades', 0)}")
        lines.append(f"Win Rate:             {summary.get('win_rate_pct', 0):.1f}%")
        lines.append(f"Profit Factor:        {summary.get('profit_factor', 0):.2f}")
        
        # Risk analysis
        risk = report.get("risk_analysis", {})
        if risk:
            lines.append("\nRISK ANALYSIS")
            lines.append("-" * 30)
            drawdown = risk.get("drawdown_analysis", {})
            var = risk.get("value_at_risk", {})
            lines.append(f"Max Drawdown:         {drawdown.get('max_drawdown_pct', 0):.2f}%")
            lines.append(f"VaR (5%):             {var.get('var_5_percent', 0):.4f}")
            lines.append(f"CVaR (5%):            {var.get('cvar_5_percent', 0):.4f}")
        
        # Trade analysis
        trade_analysis = report.get("trade_analysis", {})
        if trade_analysis:
            lines.append("\nTRADE ANALYSIS")
            lines.append("-" * 30)
            stats = trade_analysis.get("trade_statistics", {})
            profit = trade_analysis.get("profit_analysis", {})
            lines.append(f"Total Trades:         {stats.get('total_trades', 0)}")
            lines.append(f"Winning Trades:       {stats.get('winning_trades', 0)}")
            lines.append(f"Losing Trades:        {stats.get('losing_trades', 0)}")
            lines.append(f"Average Win:          ${profit.get('avg_win', 0):.2f}")
            lines.append(f"Average Loss:         ${profit.get('avg_loss', 0):.2f}")
            lines.append(f"Largest Win:          ${profit.get('largest_win', 0):.2f}")
            lines.append(f"Largest Loss:         ${profit.get('largest_loss', 0):.2f}")
        
        # Benchmark comparison
        benchmark = report.get("benchmark_comparison", {})
        if benchmark:
            lines.append("\nBENCHMARK COMPARISON")
            lines.append("-" * 30)
            lines.append(f"Benchmark Return:     {benchmark.get('benchmark_return_pct', 0):.2f}%")
            lines.append(f"Alpha:                {benchmark.get('alpha_pct', 0):.2f}%")
            outperformed = benchmark.get('outperformed_benchmark', False)
            lines.append(f"Outperformed:         {'Yes' if outperformed else 'No'}")
        
        lines.append("\n" + "=" * 60)
        
        return "\n".join(lines)


def create_hold_strategy_benchmark(data: pd.DataFrame, initial_capital: float = 10000) -> pd.DataFrame:
    """
    Create a buy-and-hold benchmark strategy for comparison.
    
    Parameters
    ----------
    data : pd.DataFrame
        Price data with Close column
    initial_capital : float, default 10000
        Initial capital for the benchmark
        
    Returns
    -------
    pd.DataFrame
        Benchmark performance data
    """
    if data.empty or "Close" not in data.columns:
        logger.error("Invalid data for hold strategy benchmark")
        return pd.DataFrame()
    
    # Calculate buy-and-hold performance
    initial_price = data["Close"].iloc[0]
    shares_bought = initial_capital / initial_price
    
    benchmark = data.copy()
    benchmark["Total_Value"] = shares_bought * benchmark["Close"]
    benchmark["Daily_Return"] = benchmark["Total_Value"].pct_change(fill_method=None)
    
    return benchmark[["Close", "Total_Value", "Daily_Return"]]