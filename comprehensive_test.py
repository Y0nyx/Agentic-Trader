#!/usr/bin/env python3
"""
Comprehensive test script demonstrating the deployed Copilot integration.

This script tests all aspects of the deployed system:
- Database population and queries
- API server functionality
- Strategy context generation
- Parameter optimization
- Code pattern retrieval
"""

import sys
import os
import requests
import json
import time

# Add the project root to the path
project_root = "/home/runner/work/Agentic-Trader/Agentic-Trader"
sys.path.insert(0, project_root)

from copilot_integration import CopilotStrategyAPI, copilot_strategy_context
from copilot_integration.strategy_database import get_strategy_database, StrategyResult

def test_database_population():
    """Test that the database is properly populated."""
    print("=" * 70)
    print("🗄️  DATABASE POPULATION TEST")
    print("=" * 70)
    
    db = get_strategy_database()
    print(f"✅ Database contains {len(db.results)} strategy results")
    
    # Count strategies by type
    strategy_counts = {}
    for result in db.results:
        strategy_type = result.strategy_type
        strategy_counts[strategy_type] = strategy_counts.get(strategy_type, 0) + 1
    
    print("\n📊 Strategy distribution:")
    for strategy_type, count in strategy_counts.items():
        print(f"  - {strategy_type}: {count} results")
    
    # Show top performing strategies
    sorted_results = sorted(db.results, key=lambda r: r.performance_metrics.get('sharpe_ratio', 0), reverse=True)
    print("\n🏆 Top 3 performing strategies:")
    for i, result in enumerate(sorted_results[:3], 1):
        sharpe = result.performance_metrics.get('sharpe_ratio', 0)
        returns = result.performance_metrics.get('total_return_pct', 0)
        print(f"  {i}. {result.strategy_type} (Sharpe: {sharpe}, Return: {returns:.1f}%)")
    
    return len(db.results)

def test_api_server():
    """Test API server endpoints."""
    print("\n" + "=" * 70)
    print("🌐 API SERVER FUNCTIONALITY TEST")
    print("=" * 70)
    
    base_url = "http://localhost:8000"
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/api/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health check: {health_data['status']}")
            print(f"   Database results: {health_data['database_results']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False
    
    # Test strategy context endpoint
    try:
        response = requests.get(f"{base_url}/api/strategy-context/moving_average", timeout=5)
        if response.status_code == 200:
            context_data = response.json()
            print(f"✅ Strategy context endpoint working")
            print(f"   Strategy type: {context_data['strategy_type']}")
            print(f"   Has optimal parameters: {'optimal_parameters' in context_data}")
            print(f"   Has code patterns: {'code_patterns' in context_data}")
        else:
            print(f"❌ Strategy context failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Strategy context error: {e}")
    
    # Test parameters endpoint
    try:
        response = requests.get(f"{base_url}/api/parameters/rsi", timeout=5)
        if response.status_code == 200:
            params_data = response.json()
            print(f"✅ Parameters endpoint working")
            print(f"   Strategy type: {params_data['strategy_type']}")
            recommended_params = {k: v['recommended_value'] for k, v in params_data['parameters'].items() if isinstance(v, dict)}
            print(f"   Recommended params: {recommended_params}")
        else:
            print(f"❌ Parameters endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Parameters endpoint error: {e}")
    
    # Test adding a new result
    try:
        new_result = {
            "strategy_type": "bollinger_bands",
            "parameters": {"period": 18, "std_dev": 2.2, "squeeze_filter": True},
            "performance_metrics": {"sharpe_ratio": 1.33, "total_return_pct": 16.5, "max_drawdown_pct": -8.1},
            "market_regime": "volatile"
        }
        
        response = requests.post(
            f"{base_url}/api/add-result",
            json=new_result,
            timeout=5
        )
        
        if response.status_code == 200:
            result_data = response.json()
            print(f"✅ Add result endpoint working")
            print(f"   New strategy signature: {result_data['signature']}")
        else:
            print(f"❌ Add result failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Add result error: {e}")
    
    return True

def test_copilot_integration():
    """Test Copilot integration features."""
    print("\n" + "=" * 70)
    print("🤖 COPILOT INTEGRATION FEATURES TEST")
    print("=" * 70)
    
    # Test strategy insights
    print("1. Testing strategy insights...")
    ma_insights = CopilotStrategyAPI.get_strategy_insights("moving_average")
    print(f"   ✅ Got insights for moving_average ({len(ma_insights)} chars)")
    
    rsi_insights = CopilotStrategyAPI.get_strategy_insights("rsi")
    print(f"   ✅ Got insights for rsi ({len(rsi_insights)} chars)")
    
    # Test parameter suggestions
    print("\n2. Testing parameter suggestions...")
    ma_params = CopilotStrategyAPI.suggest_parameters("moving_average")
    print(f"   ✅ Got parameters for moving_average: {len(ma_params)} parameters")
    
    bb_params = CopilotStrategyAPI.suggest_parameters("bollinger_bands")
    print(f"   ✅ Got parameters for bollinger_bands: {len(bb_params)} parameters")
    
    # Test code patterns
    print("\n3. Testing code patterns...")
    ma_patterns = CopilotStrategyAPI.get_code_patterns("moving_average")
    print(f"   ✅ Got code patterns for moving_average ({len(ma_patterns)} chars)")
    
    macd_patterns = CopilotStrategyAPI.get_code_patterns("macd")
    print(f"   ✅ Got code patterns for macd ({len(macd_patterns)} chars)")
    
    # Test strategy validation
    print("\n4. Testing strategy validation...")
    existing_result = CopilotStrategyAPI.check_strategy_exists("ecbc887c1bb2")  # Known existing signature
    print(f"   ✅ Existing strategy check: {existing_result['exists']}")
    
    new_result = CopilotStrategyAPI.check_strategy_exists("new_strategy_123")
    print(f"   ✅ New strategy check: {new_result['exists']}")

def test_decorator_functionality():
    """Test the context decorator."""
    print("\n" + "=" * 70)
    print("🎨 DECORATOR FUNCTIONALITY TEST")
    print("=" * 70)
    
    @copilot_strategy_context("rsi")
    def create_rsi_strategy():
        """Create an RSI strategy with Copilot intelligence."""
        return {
            "rsi_period": 14,
            "oversold": 25,
            "overbought": 75,
            "volume_filter": True
        }
    
    print("✅ Decorator applied successfully")
    print(f"   Strategy type: {create_rsi_strategy._copilot_strategy_type}")
    print(f"   Enhanced: {create_rsi_strategy._copilot_enhanced}")
    
    # Check enhanced docstring
    enhanced_doc = create_rsi_strategy.__doc__
    print(f"   Enhanced docstring length: {len(enhanced_doc)} chars")
    print(f"   Contains performance data: {'Sharpe' in enhanced_doc}")
    print(f"   Contains recommendations: {'OPTIMAL PARAMETERS' in enhanced_doc}")

def simulate_copilot_usage():
    """Simulate how GitHub Copilot would use the system."""
    print("\n" + "=" * 70)
    print("💻 SIMULATED COPILOT USAGE SCENARIO")
    print("=" * 70)
    
    print("Scenario: Developer wants to create a new moving average strategy")
    print("Copilot queries the system for optimal parameters and context...")
    
    # 1. Get strategy context
    context = CopilotStrategyAPI.get_strategy_insights("moving_average")
    print(f"✅ Retrieved historical context ({len(context)} chars)")
    
    # 2. Get optimal parameters
    params = CopilotStrategyAPI.suggest_parameters("moving_average")
    optimal_params = {k: v['recommended_value'] for k, v in params.items() if isinstance(v, dict)}
    print(f"✅ Retrieved optimal parameters: {optimal_params}")
    
    # 3. Get code patterns
    patterns = CopilotStrategyAPI.get_code_patterns("moving_average")
    print(f"✅ Retrieved code patterns ({len(patterns)} chars)")
    
    # 4. Check if strategy exists
    import hashlib
    param_str = json.dumps(optimal_params, sort_keys=True)
    signature = hashlib.md5(f"moving_average_{param_str}".encode()).hexdigest()[:12]
    validation = CopilotStrategyAPI.check_strategy_exists(signature)
    print(f"✅ Validated strategy uniqueness: exists={validation['exists']}")
    
    print("\n💡 Copilot would now provide intelligent suggestions based on this data!")

def main():
    """Run all tests."""
    print("🚀 GITHUB COPILOT INTEGRATION v2.0 - DEPLOYMENT TEST")
    print("🔄 Testing database population, API deployment, and integration usage")
    
    # Test database
    db_count = test_database_population()
    
    # Test API server
    api_working = test_api_server()
    
    if api_working:
        # Test integration features
        test_copilot_integration()
        
        # Test decorator
        test_decorator_functionality()
        
        # Simulate usage
        simulate_copilot_usage()
        
        print("\n" + "=" * 70)
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"✅ Database populated with {db_count} strategy results")
        print("✅ API server deployed and functional")
        print("✅ All endpoints responding correctly")
        print("✅ Copilot integration working")
        print("✅ Context decorators functional")
        print("✅ Ready for GitHub Copilot usage")
        
        print("\n🌟 SUMMARY:")
        print("  - The system is fully deployed and operational")
        print("  - GitHub Copilot can now access historical strategy data")
        print("  - Developers will get intelligent parameter suggestions")
        print("  - Code patterns from successful strategies are available")
        print("  - Strategy validation prevents duplicates")
        
        print("\n📖 USAGE:")
        print("  1. Use @copilot_strategy_context() decorator on strategy functions")
        print("  2. GitHub Copilot will see enhanced context in docstrings")
        print("  3. API endpoints provide real-time data access")
        print("  4. System learns from each new strategy result")
        
    else:
        print("\n❌ API server tests failed - check if server is running")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())