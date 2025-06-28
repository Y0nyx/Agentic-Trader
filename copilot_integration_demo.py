"""
Example script demonstrating the GitHub Copilot Integration v2.0.

This script shows how to use the new Copilot integration features
including context-aware decorators and the strategy database API.
"""

import sys
import os

# Add the project root to the path
project_root = "/home/runner/work/Agentic-Trader/Agentic-Trader"
sys.path.insert(0, project_root)

from copilot_integration import CopilotStrategyAPI, copilot_strategy_context
from copilot_integration.decorators import CopilotCommentGenerator
from copilot_integration.strategy_database import get_strategy_database, StrategyResult


def demo_strategy_context():
    """Demonstrate strategy context retrieval."""
    print("=" * 60)
    print("GITHUB COPILOT INTEGRATION v2.0 DEMO")
    print("=" * 60)
    
    # Get strategy insights
    print("\n1. Getting historical context for Moving Average strategies:")
    print("-" * 50)
    ma_context = CopilotStrategyAPI.get_strategy_insights("moving_average")
    print(ma_context)
    
    print("\n2. Getting optimal parameters:")
    print("-" * 30)
    ma_params = CopilotStrategyAPI.suggest_parameters("moving_average")
    for param, info in ma_params.items():
        if isinstance(info, dict):
            print(f"  {param}: {info.get('recommended_value')} ({info.get('description', 'N/A')})")
        else:
            print(f"  {param}: {info}")
    
    print("\n3. Getting code patterns:")
    print("-" * 25)
    patterns = CopilotStrategyAPI.get_code_patterns("moving_average")
    print(patterns[:500] + "..." if len(patterns) > 500 else patterns)


def demo_context_decorator():
    """Demonstrate the context decorator."""
    print("\n" + "=" * 60)
    print("CONTEXT DECORATOR DEMONSTRATION")
    print("=" * 60)
    
    @copilot_strategy_context("moving_average")
    def create_optimized_strategy():
        """
        Create a moving average strategy with Copilot intelligence.
        
        This function will have enhanced docstring with historical context.
        """
        return {
            "short_window": 10,
            "long_window": 30,
            "use_rsi_filter": True,
            "volume_confirmation": True
        }
    
    print("Function with enhanced docstring (first 800 chars):")
    print("-" * 50)
    print(create_optimized_strategy.__doc__[:800] + "...")
    
    print(f"\nCopilot metadata:")
    print(f"  Strategy type: {create_optimized_strategy._copilot_strategy_type}")
    print(f"  Enhanced: {create_optimized_strategy._copilot_enhanced}")


def demo_parameter_comments():
    """Demonstrate parameter comment generation."""
    print("\n" + "=" * 60)
    print("PARAMETER COMMENTS FOR COPILOT")
    print("=" * 60)
    
    print("Generated parameter comments for Moving Average strategy:")
    print("-" * 55)
    comments = CopilotCommentGenerator.generate_parameter_comments("moving_average")
    print(comments)
    
    print("\nImplementation hints:")
    print("-" * 20)
    hints = CopilotCommentGenerator.generate_implementation_hints("moving_average")
    print(hints[:600] + "..." if len(hints) > 600 else hints)


def demo_strategy_validation():
    """Demonstrate strategy validation."""
    print("\n" + "=" * 60)
    print("STRATEGY VALIDATION")
    print("=" * 60)
    
    # Test with existing strategy signature
    db = get_strategy_database()
    if db.results:
        existing_sig = db.results[0].signature
        print(f"Checking existing strategy (signature: {existing_sig}):")
        result = CopilotStrategyAPI.check_strategy_exists(existing_sig)
        
        print(f"  Exists: {result['exists']}")
        if result['exists']:
            print(f"  Performance: {result['performance']}")
            print(f"  Suggestion: {result['suggestion']}")
    
    # Test with new strategy
    print(f"\nChecking new strategy:")
    new_result = CopilotStrategyAPI.check_strategy_exists("new_strategy_12345")
    print(f"  Exists: {new_result['exists']}")
    print(f"  Suggestion: {new_result['suggestion']}")


def demo_database_operations():
    """Demonstrate database operations."""
    print("\n" + "=" * 60)
    print("STRATEGY DATABASE OPERATIONS")
    print("=" * 60)
    
    db = get_strategy_database()
    
    print(f"Current database contains {len(db.results)} strategy results")
    
    # Add a new result
    new_result = StrategyResult(
        strategy_type="moving_average",
        parameters={"short_window": 12, "long_window": 26},
        performance_metrics={
            "sharpe_ratio": 1.65,
            "total_return_pct": 22.5,
            "max_drawdown_pct": -8.2
        },
        market_regime="trending"
    )
    
    db.add_result(new_result)
    print(f"Added new result. Database now contains {len(db.results)} results")
    print(f"New strategy signature: {new_result.signature}")
    
    # Get updated insights
    print("\nUpdated insights for Moving Average strategies:")
    insights = db.get_strategy_performance_summary("moving_average")
    print(f"  Success rate: {insights.success_rate:.1f}%")
    print(f"  Best parameters: {insights.best_params}")
    print(f"  Number of top performers: {len(insights.top_performers)}")


@copilot_strategy_context("moving_average")
def example_strategy_with_context():
    """
    Example strategy function with Copilot context.
    
    This demonstrates how Copilot will see enhanced documentation
    with historical performance data and optimization suggestions.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE STRATEGY WITH COPILOT CONTEXT")
    print("=" * 60)
    
    print("This function has enhanced docstring with:")
    print("  ✓ Historical performance context")
    print("  ✓ Optimal parameter suggestions")
    print("  ✓ Implementation guidelines")
    print("  ✓ Common pitfalls to avoid")
    
    return {
        "message": "Strategy created with Copilot intelligence!",
        "recommended_params": {
            "short_window": 10,
            "long_window": 30,
            "use_rsi_filter": True,
            "volume_confirmation": True
        }
    }


def main():
    """Main demonstration function."""
    try:
        demo_strategy_context()
        demo_context_decorator()
        demo_parameter_comments()
        demo_strategy_validation()
        demo_database_operations()
        
        # Test the example strategy
        result = example_strategy_with_context()
        print(f"\nExample strategy result: {result}")
        
        print("\n" + "=" * 60)
        print("🎉 COPILOT INTEGRATION v2.0 DEMO COMPLETED!")
        print("=" * 60)
        print("✅ Strategy database working")
        print("✅ Context API functional")
        print("✅ Decorators enhance docstrings")
        print("✅ Parameter suggestions available")
        print("✅ Code patterns accessible")
        print("✅ Strategy validation working")
        
        print("\n🚀 Next steps:")
        print("  1. Start the API server: python copilot_integration/api_endpoints.py")
        print("  2. Use decorators in your strategies")
        print("  3. Let Copilot access historical context for better suggestions")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()