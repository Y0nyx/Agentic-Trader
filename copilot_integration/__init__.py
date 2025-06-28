"""
Copilot Integration module for the Agentic Trader project.

This module provides GitHub Copilot with intelligent context about historical
strategy performances, enabling better code suggestions and parameter optimization.
"""

from .strategy_database import StrategyDatabase
from .copilot_api import CopilotStrategyAPI
from .decorators import copilot_strategy_context
from .api_endpoints import create_copilot_api

__all__ = [
    "StrategyDatabase",
    "CopilotStrategyAPI", 
    "copilot_strategy_context",
    "create_copilot_api",
]