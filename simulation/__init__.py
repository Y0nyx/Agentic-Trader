"""Simulation module for the Agentic Trader project.

This module provides backtesting and simulation capabilities for trading strategies.
"""

from .backtester import Backtester, PerformanceReport

__all__ = [
    "Backtester",
    "PerformanceReport",
]
