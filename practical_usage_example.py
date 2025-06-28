#!/usr/bin/env python3
"""
Practical example: Using GitHub Copilot Integration for strategy development.

This script demonstrates how a developer would use the Copilot integration
in their day-to-day strategy development workflow.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the project root to the path
project_root = "/home/runner/work/Agentic-Trader/Agentic-Trader"
sys.path.insert(0, project_root)

from copilot_integration import CopilotStrategyAPI, copilot_strategy_context
from copilot_integration.strategy_database import get_strategy_database, StrategyResult

# Example 1: Developer creating a new moving average strategy
@copilot_strategy_context("moving_average")
def create_optimized_ma_strategy():
    """
    Create a moving average crossover strategy optimized with historical data.
    
    GitHub Copilot will automatically see the enhanced context with:
    - Optimal parameters from successful strategies
    - Historical performance patterns
    - Implementation guidelines
    - Common pitfalls to avoid
    """
    # Copilot suggests these parameters based on historical data:
    # short_window=12, long_window=26 (Sharpe: 1.48)
    return {
        "short_window": 12,
        "long_window": 26,
        "use_rsi_filter": True,
        "volume_confirmation": True
    }

# Example 2: Developer creating an RSI strategy
@copilot_strategy_context("rsi")
def create_rsi_mean_reversion():
    """
    Create an RSI mean reversion strategy with Copilot intelligence.
    
    Copilot provides context about optimal RSI periods and thresholds
    based on historical performance data.
    """
    # Copilot suggests: rsi_period=14, oversold=25, overbought=75
    return {
        "rsi_period": 14,
        "oversold": 25,
        "overbought": 75,
        "volume_filter": True
    }

# Example 3: Developer exploring new strategy type
@copilot_strategy_context("bollinger_bands")
def create_bollinger_strategy():
    """
    Create a Bollinger Bands strategy with historical insights.
    
    Even for less common strategies, Copilot provides available
    historical data and suggested starting parameters.
    """
    # Copilot suggests based on available data
    return {
        "period": 20,
        "std_dev": 2.0,
        "squeeze_filter": True
    }

def demonstrate_developer_workflow():
    """Demonstrate a typical developer workflow using the system."""
    print("=" * 80)
    print("👨‍💻 DEVELOPER WORKFLOW WITH COPILOT INTEGRATION")
    print("=" * 80)
    
    print("STEP 1: Developer wants to create a new moving average strategy")
    print("-" * 60)
    
    # Check what parameters work best
    ma_params = CopilotStrategyAPI.suggest_parameters("moving_average")
    optimal = {k: v['recommended_value'] for k, v in ma_params.items() if isinstance(v, dict)}
    print(f"🎯 Copilot suggests optimal parameters: {optimal}")
    
    # Get historical context
    context = CopilotStrategyAPI.get_strategy_insights("moving_average")
    success_rate = context.split("Success rate: ")[1].split("%")[0] if "Success rate:" in context else "N/A"
    print(f"📊 Historical success rate: {success_rate}%")
    
    print("\nSTEP 2: Developer implements strategy with enhanced context")
    print("-" * 60)
    
    # Use decorator to get enhanced context
    strategy_config = create_optimized_ma_strategy()
    print(f"✅ Strategy created with config: {strategy_config}")
    print(f"🤖 Enhanced docstring available for Copilot (length: {len(create_optimized_ma_strategy.__doc__)} chars)")
    
    print("\nSTEP 3: Developer validates strategy uniqueness")
    print("-" * 60)
    
    # Check if similar strategy exists
    import hashlib
    import json
    param_str = json.dumps(strategy_config, sort_keys=True)
    signature = hashlib.md5(f"moving_average_{param_str}".encode()).hexdigest()[:12]
    validation = CopilotStrategyAPI.check_strategy_exists(signature)
    print(f"🔍 Strategy validation: exists={validation['exists']}")
    print(f"💡 Suggestion: {validation['suggestion']}")
    
    print("\nSTEP 4: Developer gets implementation guidance")
    print("-" * 60)
    
    # Get code patterns
    patterns = CopilotStrategyAPI.get_code_patterns("moving_average")
    print(f"📝 Code patterns available ({len(patterns)} chars)")
    print("Sample pattern preview:")
    print(patterns[:300] + "..." if len(patterns) > 300 else patterns)
    
    print("\nSTEP 5: Developer backtests and adds results")
    print("-" * 60)
    
    # Simulate backtest results
    backtest_results = {
        "sharpe_ratio": 1.41,
        "total_return_pct": 17.2,
        "max_drawdown_pct": -7.8,
        "win_rate": 0.61
    }
    
    # Add results to database
    new_result = StrategyResult(
        strategy_type="moving_average",
        parameters=strategy_config,
        performance_metrics=backtest_results,
        market_regime="trending"
    )
    
    db = get_strategy_database()
    initial_count = len(db.results)
    db.add_result(new_result)
    print(f"📈 Backtest completed: Sharpe {backtest_results['sharpe_ratio']}, Return {backtest_results['total_return_pct']}%")
    print(f"💾 Results added to database ({initial_count} → {len(db.results)} entries)")
    print(f"🔑 New strategy signature: {new_result.signature}")
    
    return new_result

def demonstrate_api_usage():
    """Show how external tools can use the API."""
    print("\n" + "=" * 80)
    print("🌐 EXTERNAL TOOL INTEGRATION VIA API")
    print("=" * 80)
    
    print("Example: External trading platform queries for RSI parameters")
    print("-" * 60)
    
    import requests
    
    try:
        # Query RSI parameters for trending market
        response = requests.get(
            "http://localhost:8000/api/parameters/rsi",
            params={"market_regime": "trending"},
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            params = {k: v['recommended_value'] for k, v in data['parameters'].items() if isinstance(v, dict)}
            print(f"✅ API response: {params}")
            print(f"📅 Timestamp: {data['timestamp']}")
        else:
            print(f"❌ API error: {response.status_code}")
    
    except Exception as e:
        print(f"⚠️ API connection issue: {e}")
        print("(This is expected if running without server)")

def demonstrate_continuous_learning():
    """Show how the system learns from new results."""
    print("\n" + "=" * 80)
    print("🧠 CONTINUOUS LEARNING DEMONSTRATION")
    print("=" * 80)
    
    db = get_strategy_database()
    
    # Show current best performer
    ma_results = db.get_strategy_results("moving_average")
    best_current = max(ma_results, key=lambda r: r.performance_metrics.get('sharpe_ratio', 0))
    print(f"Current best MA strategy: Sharpe {best_current.performance_metrics['sharpe_ratio']}")
    print(f"Parameters: {best_current.parameters}")
    
    # Add a new superior strategy
    superior_result = StrategyResult(
        strategy_type="moving_average",
        parameters={"short_window": 11, "long_window": 28, "use_rsi_filter": True, "volume_confirmation": True},
        performance_metrics={"sharpe_ratio": 1.62, "total_return_pct": 21.5, "max_drawdown_pct": -6.9},
        market_regime="trending"
    )
    
    db.add_result(superior_result)
    
    # Show updated recommendations
    new_insights = db.get_strategy_performance_summary("moving_average")
    print(f"\nAfter adding superior strategy:")
    print(f"New best parameters: {new_insights.best_params}")
    print(f"Updated success rate: {new_insights.success_rate:.1f}%")
    print(f"New signature: {superior_result.signature}")
    
    print("\n🔄 System has learned and updated recommendations!")

def main():
    """Run the practical demonstration."""
    print("🚀 GITHUB COPILOT INTEGRATION - PRACTICAL USAGE EXAMPLE")
    print("📋 This demonstrates real-world usage scenarios")
    
    # Demonstrate developer workflow
    new_strategy = demonstrate_developer_workflow()
    
    # Show API usage
    demonstrate_api_usage()
    
    # Show continuous learning
    demonstrate_continuous_learning()
    
    print("\n" + "=" * 80)
    print("🎯 PRACTICAL DEMONSTRATION COMPLETED")
    print("=" * 80)
    print("✅ Developers can now use Copilot for intelligent strategy development")
    print("✅ API provides real-time access to historical insights")
    print("✅ System learns and improves with each new strategy")
    print("✅ Validation prevents duplicate strategy development")
    
    print("\n🌟 KEY BENEFITS DEMONSTRATED:")
    print("  1. 🎯 Intelligent parameter suggestions based on proven results")
    print("  2. 📊 Historical performance context for better decisions")
    print("  3. 🔍 Strategy validation to avoid duplicated effort")
    print("  4. 📝 Code patterns from successful implementations")
    print("  5. 🧠 Continuous learning from new strategy results")
    print("  6. 🌐 API access for external tool integration")
    
    print(f"\n📈 Final database state: {len(get_strategy_database().results)} strategy results")

if __name__ == "__main__":
    main()