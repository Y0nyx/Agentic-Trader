"""Strategies module for the Agentic Trader project.

This module contains various trading strategies and algorithmic implementations.
"""

from .moving_average_cross import MovingAverageCrossStrategy
from .triple_ma_strategy import TripleMovingAverageStrategy
from .adaptive_ma_strategy import AdaptiveMovingAverageStrategy
from .advanced_ma_strategy import AdvancedMAStrategy
from .trend_following_strategy import TrendFollowingStrategy
from .buy_hold_plus_strategy import BuyHoldPlusStrategy

__all__ = [
    "MovingAverageCrossStrategy",
    "TripleMovingAverageStrategy",
    "AdaptiveMovingAverageStrategy",
    "AdvancedMAStrategy",
    "TrendFollowingStrategy",
    "BuyHoldPlusStrategy",
]
