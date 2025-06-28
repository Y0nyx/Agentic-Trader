"""
Performance evaluation metrics module.

This module provides comprehensive performance evaluation capabilities for trading strategies,
extending the basic metrics provided by the simulation module with advanced analytics.
"""

import logging
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from simulation.backtester import PerformanceReport

# Configure logging
logger = logging.getLogger(__name__)


def evaluate_performance(
    performance_report: PerformanceReport,
    benchmark_data: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Evaluate strategy performance with comprehensive metrics.
    
    Parameters
    ----------
    performance_report : PerformanceReport
        Backtest results from the simulation module
    benchmark_data : pd.DataFrame, optional
        Benchmark price data for comparison (e.g., buy-and-hold)
        
    Returns
    -------
    Dict[str, Any]
        Comprehensive performance metrics including:
        - Financial performance (ROI, profit metrics)
        - Risk management (drawdown, volatility)
        - Transaction performance (win rate, profit factor)
        - Benchmark comparison (if provided)
    """
    if performance_report.portfolio_history.empty:
        logger.warning("Empty performance report provided")
        return {}
    
    # Get basic metrics from existing report
    basic_metrics = performance_report.summary()
    
    # Calculate additional advanced metrics
    advanced_metrics = _calculate_advanced_metrics(performance_report)
    
    # Combine all metrics
    metrics = {**basic_metrics, **advanced_metrics}
    
    # Add benchmark comparison if provided
    if benchmark_data is not None:
        benchmark_metrics = _calculate_benchmark_comparison(
            performance_report, benchmark_data
        )
        metrics.update(benchmark_metrics)
    
    return metrics


def _calculate_advanced_metrics(performance_report: PerformanceReport) -> Dict[str, Any]:
    """Calculate advanced performance metrics."""
    portfolio = performance_report.portfolio_history.copy()
    
    if portfolio.empty:
        return {}
    
    # Calculate returns series
    portfolio["Daily_Return"] = portfolio["Total_Value"].pct_change(fill_method=None)
    returns = portfolio["Daily_Return"].dropna()
    
    # Cumulative profit over time
    cumulative_profit = portfolio["Total_Value"] - performance_report.initial_capital
    
    # Advanced risk metrics
    downside_returns = returns[returns < 0]
    downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 1 else 0
    
    # Sortino ratio (using downside volatility)
    risk_free_rate = 0.02
    excess_return = (portfolio["Total_Value"].iloc[-1] / performance_report.initial_capital - 1) * 252
    sortino_ratio = (excess_return - risk_free_rate) / downside_volatility if downside_volatility > 0 else 0
    
    # Calmar ratio (return / max drawdown)
    cumulative_max = portfolio["Total_Value"].expanding().max()
    drawdown_series = (portfolio["Total_Value"] - cumulative_max) / cumulative_max
    max_drawdown = abs(drawdown_series.min())
    calmar_ratio = (excess_return / max_drawdown) if max_drawdown > 0 else 0
    
    # Value at Risk (VaR) - 5% VaR
    var_5 = np.percentile(returns, 5) if len(returns) > 0 else 0
    
    # Conditional Value at Risk (CVaR) - Expected shortfall
    cvar_5 = returns[returns <= var_5].mean() if len(returns[returns <= var_5]) > 0 else 0
    
    # Profit metrics from transactions
    transactions = performance_report.transactions
    if transactions:
        profits = [t.get("pnl", 0) for t in transactions if "pnl" in t]
        if profits:
            total_net_profit = sum(profits)
            avg_profit_per_trade = np.mean(profits)
            profit_std = np.std(profits)
        else:
            total_net_profit = avg_profit_per_trade = profit_std = 0
    else:
        total_net_profit = avg_profit_per_trade = profit_std = 0
    
    # Maximum consecutive wins/losses
    if transactions:
        consecutive_wins, consecutive_losses = _calculate_consecutive_trades(transactions)
    else:
        consecutive_wins = consecutive_losses = 0
    
    return {
        "total_net_profit": total_net_profit,
        "avg_profit_per_trade": avg_profit_per_trade,
        "profit_std": profit_std,
        "cumulative_profit_final": cumulative_profit.iloc[-1] if len(cumulative_profit) > 0 else 0,
        "sortino_ratio": sortino_ratio,
        "calmar_ratio": calmar_ratio,
        "downside_volatility": downside_volatility,
        "var_5_percent": var_5,
        "cvar_5_percent": cvar_5,
        "max_consecutive_wins": consecutive_wins,
        "max_consecutive_losses": consecutive_losses,
        "portfolio_volatility_pct": portfolio["Daily_Return"].std() * 100 if len(returns) > 1 else 0,
    }


def _calculate_consecutive_trades(transactions: list) -> Tuple[int, int]:
    """Calculate maximum consecutive wins and losses."""
    if not transactions:
        return 0, 0
    
    profits = [t.get("pnl", 0) for t in transactions if "pnl" in t]
    if not profits:
        return 0, 0
    
    max_wins = max_losses = 0
    current_wins = current_losses = 0
    
    for profit in profits:
        if profit > 0:
            current_wins += 1
            current_losses = 0
            max_wins = max(max_wins, current_wins)
        elif profit < 0:
            current_losses += 1
            current_wins = 0
            max_losses = max(max_losses, current_losses)
        else:
            current_wins = current_losses = 0
    
    return max_wins, max_losses


def _calculate_benchmark_comparison(
    performance_report: PerformanceReport,
    benchmark_data: pd.DataFrame
) -> Dict[str, Any]:
    """Calculate performance comparison against benchmark (buy-and-hold)."""
    try:
        portfolio = performance_report.portfolio_history
        
        if portfolio.empty or benchmark_data.empty:
            return {}
        
        # Align benchmark data with portfolio timeline
        start_date = portfolio.index.min()
        end_date = portfolio.index.max()
        
        benchmark_period = benchmark_data[
            (benchmark_data.index >= start_date) & 
            (benchmark_data.index <= end_date)
        ]
        
        if benchmark_period.empty:
            logger.warning("No benchmark data available for portfolio period")
            return {}
        
        # Calculate buy-and-hold strategy performance
        initial_price = benchmark_period["Close"].iloc[0]
        final_price = benchmark_period["Close"].iloc[-1]
        
        benchmark_return = (final_price - initial_price) / initial_price
        benchmark_final_value = performance_report.initial_capital * (1 + benchmark_return)
        
        # Calculate benchmark metrics
        benchmark_returns = benchmark_period["Close"].pct_change(fill_method=None).dropna()
        benchmark_volatility = benchmark_returns.std() * np.sqrt(252) if len(benchmark_returns) > 1 else 0
        
        # Benchmark drawdown
        benchmark_cummax = benchmark_period["Close"].expanding().max()
        benchmark_drawdown = ((benchmark_period["Close"] - benchmark_cummax) / benchmark_cummax).min()
        
        # Performance comparison
        strategy_return = (performance_report.final_value - performance_report.initial_capital) / performance_report.initial_capital
        alpha = strategy_return - benchmark_return
        
        return {
            "benchmark_return": benchmark_return,
            "benchmark_return_pct": benchmark_return * 100,
            "benchmark_final_value": benchmark_final_value,
            "benchmark_volatility": benchmark_volatility,
            "benchmark_max_drawdown_pct": benchmark_drawdown * 100,
            "alpha": alpha,
            "alpha_pct": alpha * 100,
            "outperformed_benchmark": bool(strategy_return > benchmark_return),
        }
        
    except Exception as e:
        logger.error(f"Error calculating benchmark comparison: {str(e)}")
        return {}


def calculate_roi_metrics(performance_report: PerformanceReport) -> Dict[str, float]:
    """
    Calculate various ROI metrics.
    
    Parameters
    ----------
    performance_report : PerformanceReport
        Backtest results
        
    Returns
    -------
    Dict[str, float]
        ROI metrics including total, annualized, and risk-adjusted returns
    """
    if performance_report.portfolio_history.empty:
        return {}
    
    basic_summary = performance_report.summary()
    
    return {
        "roi_total": basic_summary.get("total_return", 0),
        "roi_total_pct": basic_summary.get("total_return_pct", 0),
        "roi_annualized": basic_summary.get("annualized_return", 0),
        "roi_annualized_pct": basic_summary.get("annualized_return", 0) * 100,
        "sharpe_ratio": basic_summary.get("sharpe_ratio", 0),
    }


def calculate_risk_metrics(performance_report: PerformanceReport) -> Dict[str, float]:
    """
    Calculate comprehensive risk metrics.
    
    Parameters
    ----------
    performance_report : PerformanceReport
        Backtest results
        
    Returns
    -------
    Dict[str, float]
        Risk metrics including volatility, drawdown, and downside risk measures
    """
    if performance_report.portfolio_history.empty:
        return {}
    
    portfolio = performance_report.portfolio_history.copy()
    portfolio["Daily_Return"] = portfolio["Total_Value"].pct_change(fill_method=None)
    returns = portfolio["Daily_Return"].dropna()
    
    # Basic risk metrics
    volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
    
    # Drawdown calculation
    cumulative_max = portfolio["Total_Value"].expanding().max()
    drawdown_series = (portfolio["Total_Value"] - cumulative_max) / cumulative_max
    max_drawdown = abs(drawdown_series.min())
    avg_drawdown = abs(drawdown_series[drawdown_series < 0].mean()) if len(drawdown_series[drawdown_series < 0]) > 0 else 0
    
    # Downside metrics
    downside_returns = returns[returns < 0]
    downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 1 else 0
    
    return {
        "volatility_annualized": volatility,
        "volatility_pct": volatility * 100,
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown * 100,
        "avg_drawdown": avg_drawdown,
        "avg_drawdown_pct": avg_drawdown * 100,
        "downside_volatility": downside_volatility,
        "downside_volatility_pct": downside_volatility * 100,
    }


def calculate_trade_metrics(performance_report: PerformanceReport) -> Dict[str, Any]:
    """
    Calculate detailed trading performance metrics.
    
    Parameters
    ----------
    performance_report : PerformanceReport
        Backtest results
        
    Returns
    -------
    Dict[str, Any]
        Trading metrics including win/loss ratios, profit factors, and trade distribution
    """
    transactions = performance_report.transactions
    
    if not transactions:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "profit_factor": 0,
            "largest_win": 0,
            "largest_loss": 0,
        }
    
    # Extract P&L data
    profits = [t.get("pnl", 0) for t in transactions if "pnl" in t]
    
    if not profits:
        return {"total_trades": len(transactions), "note": "No P&L data available"}
    
    winning_trades = [p for p in profits if p > 0]
    losing_trades = [p for p in profits if p < 0]
    
    # Calculate metrics
    total_trades = len(profits)
    win_count = len(winning_trades)
    loss_count = len(losing_trades)
    win_rate = win_count / total_trades if total_trades > 0 else 0
    
    avg_win = np.mean(winning_trades) if winning_trades else 0
    avg_loss = np.mean(losing_trades) if losing_trades else 0
    
    total_wins = sum(winning_trades) if winning_trades else 0
    total_losses = abs(sum(losing_trades)) if losing_trades else 0
    
    profit_factor = total_wins / total_losses if total_losses > 0 else float('inf') if total_wins > 0 else 0
    
    largest_win = max(profits) if profits else 0
    largest_loss = min(profits) if profits else 0
    
    return {
        "total_trades": total_trades,
        "winning_trades": win_count,
        "losing_trades": loss_count,
        "win_rate": win_rate,
        "win_rate_pct": win_rate * 100,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "largest_win": largest_win,
        "largest_loss": largest_loss,
        "total_gross_profit": total_wins,
        "total_gross_loss": -total_losses,
        "net_profit": total_wins - total_losses,
    }