"""
Regime Adaptive Strategy System - Comprehensive Example

This example demonstrates the Strategy-Evolution v2.0 implementation with
intelligent market regime detection and adaptive strategy switching.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# Import the new regime-adaptive components
from strategies.market_regime_classifier import MarketRegimeClassifier
from strategies.regime_adaptive_strategy import RegimeAdaptiveStrategy
from strategies.specialized_strategies import (
    IntelligentBullStrategy,
    CrisisProtectionStrategy,
    VolatilityRangeStrategy
)
from strategies.contextual_optimizer import ContextualOptimizer
from strategies.multi_strategy_portfolio import MultiStrategyPortfolio

# Import existing simulation components
from simulation.backtester import Backtester
from data.csv_loader import load_csv_data

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_synthetic_market_data(periods: int = 500) -> pd.DataFrame:
    """
    Create synthetic market data representing different regime periods.
    
    This creates a realistic dataset with transitions between different
    market regimes to demonstrate the adaptive system.
    """
    dates = pd.date_range('2020-01-01', periods=periods, freq='D')
    
    prices = []
    volumes = []
    base_price = 100
    
    # Regime periods
    bull_period = periods // 5      # 20% bull market
    crisis_period = periods // 10   # 10% crisis
    sideways_period = periods // 4  # 25% sideways
    recovery_period = periods // 5  # 20% recovery
    bear_period = periods - bull_period - crisis_period - sideways_period - recovery_period
    
    current_price = base_price
    
    logger.info(f"Creating synthetic data with {periods} periods:")
    logger.info(f"Bull: {bull_period}, Crisis: {crisis_period}, Sideways: {sideways_period}, Recovery: {recovery_period}, Bear: {bear_period}")
    
    for i in range(periods):
        if i < bull_period:  # Bull market
            trend = 0.001  # 0.1% daily growth
            volatility = 0.015  # 1.5% daily volatility
            volume_base = 1000000
            
        elif i < bull_period + crisis_period:  # Crisis
            trend = -0.025  # -2.5% daily decline
            volatility = 0.06   # 6% daily volatility
            volume_base = 5000000  # High volume
            
        elif i < bull_period + crisis_period + sideways_period:  # Sideways volatile
            trend = 0.0005 * np.sin(i * 0.2)  # Oscillating
            volatility = 0.03   # 3% volatility
            volume_base = 1500000
            
        elif i < bull_period + crisis_period + sideways_period + recovery_period:  # Recovery
            trend = 0.008   # 0.8% daily growth
            volatility = 0.025  # 2.5% volatility
            volume_base = 1200000
            
        else:  # Bear market
            trend = -0.005  # -0.5% daily decline
            volatility = 0.02   # 2% volatility
            volume_base = 1000000
        
        # Generate price and volume
        daily_return = trend + np.random.normal(0, volatility)
        current_price = current_price * (1 + daily_return)
        current_price = max(current_price, 10)  # Floor price
        
        volume = max(volume_base + np.random.normal(0, volume_base * 0.3), 100000)
        
        prices.append(current_price)
        volumes.append(volume)
    
    data = pd.DataFrame({
        'Date': dates,
        'Open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
        'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'Close': prices,
        'Volume': volumes
    })
    
    data.set_index('Date', inplace=True)
    return data


def demonstrate_regime_classification():
    """Demonstrate the market regime classification system."""
    logger.info("=== Market Regime Classification Demo ===")
    
    # Create test data
    data = create_synthetic_market_data(300)
    
    # Initialize classifier
    classifier = MarketRegimeClassifier()
    
    # Analyze different periods
    periods = [
        ("Early Period (Bull)", data.iloc[:60]),
        ("Middle Period (Crisis)", data.iloc[60:90]),
        ("Late Period (Recovery)", data.iloc[180:240])
    ]
    
    for period_name, period_data in periods:
        if len(period_data) >= 60:  # Ensure sufficient data
            regime = classifier.classify_regime(period_data)
            summary = classifier.get_regime_summary(period_data)
            
            logger.info(f"\n{period_name}:")
            logger.info(f"  Regime: {regime}")
            logger.info(f"  Confidence: {summary['regime_confidence']:.3f}")
            logger.info(f"  Description: {summary['regime_description']}")
            logger.info(f"  Suggested Strategy: {summary['suggested_strategy_type']}")


def demonstrate_adaptive_strategy():
    """Demonstrate the regime adaptive strategy in action."""
    logger.info("\n=== Regime Adaptive Strategy Demo ===")
    
    # Create comprehensive test data
    data = create_synthetic_market_data(400)
    
    # Initialize adaptive strategy
    adaptive_strategy = RegimeAdaptiveStrategy(
        regime_memory=5,
        confidence_threshold=0.6
    )
    
    # Generate signals
    logger.info("Generating adaptive strategy signals...")
    signals = adaptive_strategy.generate_signals(data)
    
    # Analyze results
    summary = adaptive_strategy.get_strategy_summary(signals)
    
    logger.info(f"\nAdaptive Strategy Results:")
    logger.info(f"  Total Signals: {summary['total_signals']}")
    logger.info(f"  Regime Changes: {summary['regime_changes']}")
    logger.info(f"  Current Regime: {summary['current_regime']}")
    logger.info(f"  Active Strategy: {summary['active_strategy_type']}")
    
    if 'regime_distribution' in summary:
        logger.info(f"  Regime Distribution: {summary['regime_distribution']}")
    
    if 'signal_distribution' in summary:
        logger.info(f"  Signal Distribution: {summary['signal_distribution']}")
    
    return signals


def demonstrate_specialized_strategies():
    """Demonstrate the specialized strategies for different market contexts."""
    logger.info("\n=== Specialized Strategies Demo ===")
    
    # Create different market scenarios
    scenarios = {
        "Bull Market": create_bull_market_data(),
        "Crisis Period": create_crisis_market_data(),
        "Volatile Range": create_sideways_volatile_data()
    }
    
    strategies = {
        "Bull Market": IntelligentBullStrategy(),
        "Crisis Period": CrisisProtectionStrategy(),
        "Volatile Range": VolatilityRangeStrategy()
    }
    
    for scenario_name, scenario_data in scenarios.items():
        logger.info(f"\n{scenario_name} Scenario:")
        
        strategy = strategies[scenario_name]
        signals = strategy.generate_signals(scenario_data)
        summary = strategy.get_strategy_summary(signals)
        
        logger.info(f"  Strategy: {summary['strategy_name']}")
        logger.info(f"  Total Signals: {summary['total_signals']}")
        
        if 'signal_distribution' in summary:
            logger.info(f"  Signal Distribution: {summary['signal_distribution']}")
        
        if 'time_in_market_pct' in summary:
            logger.info(f"  Time in Market: {summary['time_in_market_pct']:.1f}%")


def demonstrate_portfolio_management():
    """Demonstrate the multi-strategy portfolio management system."""
    logger.info("\n=== Multi-Strategy Portfolio Demo ===")
    
    # Create market data
    data = create_synthetic_market_data(300)
    
    # Initialize portfolio
    portfolio = MultiStrategyPortfolio(
        initial_capital=100000,
        max_strategy_allocation=0.4,
        rebalance_frequency=20
    )
    
    # Generate portfolio signals
    logger.info("Generating portfolio allocation signals...")
    portfolio_signals = portfolio.generate_portfolio_signals(data)
    
    # Analyze portfolio results
    summary = portfolio.get_portfolio_summary(portfolio_signals)
    
    logger.info(f"\nPortfolio Management Results:")
    logger.info(f"  Initial Capital: ${summary['initial_capital']:,}")
    logger.info(f"  Rebalance Count: {summary['rebalance_count']}")
    logger.info(f"  Current Allocations:")
    
    for strategy, allocation in summary['current_allocations'].items():
        logger.info(f"    {strategy}: {allocation:.1%}")
    
    if 'regime_distribution' in summary:
        logger.info(f"  Regime Distribution: {summary['regime_distribution']}")
    
    efficiency = summary.get('allocation_efficiency', {})
    if efficiency:
        logger.info(f"  Diversification Score: {efficiency.get('diversification_score', 0):.3f}")
        logger.info(f"  Active Strategies: {efficiency.get('active_strategies', 0)}")


def run_performance_comparison():
    """Run a performance comparison between strategies."""
    logger.info("\n=== Performance Comparison ===")
    
    # Create comprehensive market data
    data = create_synthetic_market_data(400)
    
    # Initialize backtester
    backtester = Backtester(initial_capital=100000)
    
    # Test strategies
    strategies_to_test = {
        "Regime Adaptive": RegimeAdaptiveStrategy(),
        "Multi-Strategy Portfolio": MultiStrategyPortfolio(),
        "Bull Strategy": IntelligentBullStrategy(),
        "Crisis Protection": CrisisProtectionStrategy()
    }
    
    results = {}
    
    for strategy_name, strategy in strategies_to_test.items():
        try:
            logger.info(f"Testing {strategy_name}...")
            
            # Generate signals
            if hasattr(strategy, 'generate_portfolio_signals'):
                signals = strategy.generate_portfolio_signals(data)
                position_col = 'Portfolio_Position'
            else:
                signals = strategy.generate_signals(data)
                position_col = 'Position'
            
            # Run backtest (simplified)
            backtester.reset()
            
            # Calculate simple returns
            signals['Returns'] = data['Close'].pct_change()
            signals['Strategy_Returns'] = signals['Returns'] * signals[position_col].shift(1)
            
            total_return = (1 + signals['Strategy_Returns'].fillna(0)).prod() - 1
            volatility = signals['Strategy_Returns'].std() * np.sqrt(252)
            sharpe = (signals['Strategy_Returns'].mean() * 252) / (volatility + 1e-8)
            
            results[strategy_name] = {
                'total_return': total_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe,
                'max_drawdown': calculate_max_drawdown(signals['Strategy_Returns'])
            }
            
            logger.info(f"  Total Return: {total_return:.2%}")
            logger.info(f"  Sharpe Ratio: {sharpe:.3f}")
            
        except Exception as e:
            logger.error(f"Error testing {strategy_name}: {e}")
    
    return results


def calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown from returns series."""
    cumulative = (1 + returns.fillna(0)).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative / running_max) - 1
    return drawdown.min()


# Helper functions for creating specific market scenarios
def create_bull_market_data(periods: int = 150) -> pd.DataFrame:
    """Create bull market data."""
    dates = pd.date_range('2020-01-01', periods=periods, freq='D')
    base_price = 100
    
    prices = []
    for i in range(periods):
        trend = 0.005  # 0.5% daily growth
        volatility = 0.015  # 1.5% daily volatility
        daily_return = trend + np.random.normal(0, volatility)
        
        if i == 0:
            price = base_price
        else:
            price = prices[-1] * (1 + daily_return)
        
        prices.append(max(price, 10))
    
    return pd.DataFrame({
        'Open': [p * 0.995 for p in prices],
        'High': [p * 1.02 for p in prices],
        'Low': [p * 0.98 for p in prices],
        'Close': prices,
        'Volume': [1000000] * periods
    }, index=dates)


def create_crisis_market_data(periods: int = 60) -> pd.DataFrame:
    """Create crisis market data."""
    dates = pd.date_range('2020-01-01', periods=periods, freq='D')
    base_price = 100
    
    prices = []
    volumes = []
    
    for i in range(periods):
        if i < 20:
            trend = -0.03  # 3% daily decline
            volatility = 0.08  # 8% volatility
            volume = 8000000
        else:
            trend = -0.01  # 1% decline
            volatility = 0.05  # 5% volatility
            volume = 4000000
        
        daily_return = trend + np.random.normal(0, volatility)
        
        if i == 0:
            price = base_price
        else:
            price = prices[-1] * (1 + daily_return)
        
        prices.append(max(price, 10))
        volumes.append(volume + np.random.normal(0, volume * 0.3))
    
    return pd.DataFrame({
        'Open': [p * 0.98 for p in prices],
        'High': [p * 1.05 for p in prices],
        'Low': [p * 0.95 for p in prices],
        'Close': prices,
        'Volume': volumes
    }, index=dates)


def create_sideways_volatile_data(periods: int = 100) -> pd.DataFrame:
    """Create sideways volatile market data."""
    dates = pd.date_range('2020-01-01', periods=periods, freq='D')
    base_price = 100
    
    prices = []
    for i in range(periods):
        # Oscillating pattern with volatility
        trend = 0.002 * np.sin(i * 0.3)  # Oscillating trend
        volatility = 0.03  # 3% volatility
        daily_return = trend + np.random.normal(0, volatility)
        
        if i == 0:
            price = base_price
        else:
            price = prices[-1] * (1 + daily_return)
        
        prices.append(max(price, 10))
    
    return pd.DataFrame({
        'Open': [p * 0.995 for p in prices],
        'High': [p * 1.025 for p in prices],
        'Low': [p * 0.975 for p in prices],
        'Close': prices,
        'Volume': [1500000] * periods
    }, index=dates)


def main():
    """Run the comprehensive demonstration of the regime-adaptive system."""
    logger.info("🚀 Strategy-Evolution v2.0 - Regime Adaptive Trading System Demo")
    logger.info("=" * 80)
    
    try:
        # 1. Demonstrate regime classification
        demonstrate_regime_classification()
        
        # 2. Demonstrate adaptive strategy
        adaptive_signals = demonstrate_adaptive_strategy()
        
        # 3. Demonstrate specialized strategies
        demonstrate_specialized_strategies()
        
        # 4. Demonstrate portfolio management
        demonstrate_portfolio_management()
        
        # 5. Run performance comparison
        performance_results = run_performance_comparison()
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ Demo completed successfully!")
        logger.info("The regime-adaptive system demonstrates intelligent market adaptation")
        logger.info("with dynamic strategy switching and portfolio management.")
        
        return {
            'adaptive_signals': adaptive_signals,
            'performance_results': performance_results
        }
        
    except Exception as e:
        logger.error(f"❌ Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    # Run the comprehensive demonstration
    results = main()