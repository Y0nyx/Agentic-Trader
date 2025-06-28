"""
Integration test showing how existing strategies can benefit from Copilot context.

This script demonstrates integrating the new Copilot features with existing
trading strategies in the repository.
"""

import sys
import os
import pandas as pd

# Add the project root to the path
project_root = "/home/runner/work/Agentic-Trader/Agentic-Trader"
sys.path.insert(0, project_root)

from strategies.moving_average_cross import MovingAverageCrossStrategy
from simulation.backtester import Backtester
from data.csv_loader import load_csv_data
from copilot_integration import CopilotStrategyAPI, copilot_strategy_context
from copilot_integration.strategy_database import get_strategy_database, StrategyResult


@copilot_strategy_context("moving_average")
def create_enhanced_ma_strategy():
    """
    Create an enhanced Moving Average strategy using Copilot intelligence.
    
    This function leverages historical performance data to select optimal parameters.
    """
    # Get optimal parameters from historical data
    optimal_params = CopilotStrategyAPI.suggest_parameters("moving_average")
    
    short_window = optimal_params.get('short_window', {}).get('recommended_value', 10)
    long_window = optimal_params.get('long_window', {}).get('recommended_value', 30)
    
    print(f"Creating MA strategy with Copilot-optimized parameters:")
    print(f"  Short window: {short_window}")
    print(f"  Long window: {long_window}")
    
    return MovingAverageCrossStrategy(
        short_window=short_window,
        long_window=long_window
    )


def test_strategy_with_copilot_integration():
    """Test how existing strategies work with Copilot integration."""
    print("=" * 60)
    print("TESTING EXISTING STRATEGY WITH COPILOT INTEGRATION")
    print("=" * 60)
    
    try:
        # Load sample data (small subset for demo)
        print("\n1. Loading test data...")
        data = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=100, freq='D'),
            'Open': 100 + pd.Series(range(100)) * 0.1 + pd.Series(range(100)).apply(lambda x: (x % 10) * 0.5),
            'High': 100 + pd.Series(range(100)) * 0.1 + pd.Series(range(100)).apply(lambda x: (x % 10) * 0.5) + 1,
            'Low': 100 + pd.Series(range(100)) * 0.1 + pd.Series(range(100)).apply(lambda x: (x % 10) * 0.5) - 1,
            'Close': 100 + pd.Series(range(100)) * 0.1 + pd.Series(range(100)).apply(lambda x: (x % 10) * 0.5),
            'Volume': 1000000
        })
        data.set_index('Date', inplace=True)
        print(f"   Loaded {len(data)} days of test data")
        
        # Create strategy with Copilot optimization
        print("\n2. Creating strategy with Copilot optimization...")
        strategy = create_enhanced_ma_strategy()
        
        # Generate signals
        print("\n3. Generating trading signals...")
        signals = strategy.generate_signals(data)
        print(f"   Generated {len(signals)} signal records")
        
        # Count signal types
        signal_counts = signals['Signal'].value_counts()
        print(f"   Signal distribution: {dict(signal_counts)}")
        
        # Run backtest
        print("\n4. Running backtest...")
        backtester = Backtester(initial_capital=10000)
        performance = backtester.run_backtest(data, signals)
        
        # Get performance summary
        summary = performance.summary()
        print(f"   Total return: {summary['total_return_pct']:.2f}%")
        print(f"   Sharpe ratio: {summary.get('sharpe_ratio', 'N/A')}")
        print(f"   Max drawdown: {summary.get('max_drawdown_pct', 'N/A'):.2f}%")
        
        # Add result to Copilot database for future reference
        print("\n5. Adding result to Copilot database...")
        db = get_strategy_database()
        
        result = StrategyResult(
            strategy_type="moving_average",
            parameters={
                'short_window': strategy.short_window,
                'long_window': strategy.long_window
            },
            performance_metrics={
                'sharpe_ratio': summary.get('sharpe_ratio', 0),
                'total_return_pct': summary['total_return_pct'],
                'max_drawdown_pct': summary.get('max_drawdown_pct', 0)
            },
            market_regime="test"
        )
        
        db.add_result(result)
        print(f"   Added result with signature: {result.signature}")
        
        # Show updated insights
        print("\n6. Updated Copilot insights:")
        insights = CopilotStrategyAPI.get_strategy_insights("moving_average")
        print(insights[:400] + "..." if len(insights) > 400 else insights)
        
        print("\n✅ Integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_copilot_enhancement():
    """Demonstrate how Copilot enhances existing development workflow."""
    print("\n" + "=" * 60)
    print("COPILOT ENHANCEMENT DEMONSTRATION")
    print("=" * 60)
    
    print("\n1. Traditional strategy creation (without Copilot):")
    print("   - Developer manually chooses parameters")
    print("   - No historical context available")
    print("   - Trial and error optimization")
    
    traditional_strategy = MovingAverageCrossStrategy(short_window=15, long_window=35)
    print(f"   Example: MA({traditional_strategy.short_window}, {traditional_strategy.long_window})")
    
    print("\n2. Copilot-enhanced strategy creation:")
    print("   - Historical performance data guides parameter choice")
    print("   - Automatic context injection in docstrings")
    print("   - Validation against existing strategies")
    
    # Get Copilot suggestions
    suggestions = CopilotStrategyAPI.suggest_parameters("moving_average")
    optimal_short = suggestions.get('short_window', {}).get('recommended_value', 10)
    optimal_long = suggestions.get('long_window', {}).get('recommended_value', 30)
    
    enhanced_strategy = MovingAverageCrossStrategy(short_window=optimal_short, long_window=optimal_long)
    print(f"   Copilot suggests: MA({enhanced_strategy.short_window}, {enhanced_strategy.long_window})")
    
    print("\n3. Benefits of Copilot integration:")
    print("   ✓ Parameters based on historical performance")
    print("   ✓ Automatic code suggestions with context")
    print("   ✓ Validation against duplicate strategies")
    print("   ✓ Implementation hints from successful patterns")
    print("   ✓ Continuous learning from new results")


def main():
    """Main integration test function."""
    print("GITHUB COPILOT INTEGRATION v2.0 - EXISTING STRATEGY INTEGRATION")
    
    # Test integration with existing strategies
    success = test_strategy_with_copilot_integration()
    
    # Demonstrate enhancements
    demo_copilot_enhancement()
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 INTEGRATION TEST SUCCESSFUL!")
        print("=" * 60)
        print("✅ Existing strategies work with Copilot integration")
        print("✅ Performance data automatically captured")
        print("✅ Historical context available for future development")
        print("✅ Enhanced developer experience with intelligent suggestions")
    else:
        print("\n❌ Integration test failed")
    
    return success


if __name__ == "__main__":
    main()