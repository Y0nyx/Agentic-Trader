"""Optimization module for the Agentic Trader project.

This module contains tools and frameworks for automatically tuning trading strategy parameters.
"""

from .grid_search import GridSearchOptimizer, OptimizationReport, optimize_strategy

__all__ = [
    "GridSearchOptimizer",
    "OptimizationReport", 
    "optimize_strategy",
]
