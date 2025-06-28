"""Evaluation module for the Agentic Trader project.

This module contains performance evaluation metrics and analysis tools for trading strategies.
"""

from .metrics import (
    evaluate_performance,
    calculate_roi_metrics,
    calculate_risk_metrics,
    calculate_trade_metrics,
)
from .report import PerformanceReporter, create_hold_strategy_benchmark

__all__ = [
    "evaluate_performance",
    "calculate_roi_metrics", 
    "calculate_risk_metrics",
    "calculate_trade_metrics",
    "PerformanceReporter",
    "create_hold_strategy_benchmark",
]
