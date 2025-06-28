"""
Copilot Strategy API for providing intelligent context to GitHub Copilot.

This module provides the main API that GitHub Copilot can use to access
historical strategy performance data and get intelligent suggestions.
"""

from typing import Dict, Any, List, Optional
from .strategy_database import get_strategy_database, StrategyResult


class CopilotStrategyAPI:
    """
    API simplifiée pour que Copilot accède aux données historiques.
    
    This class provides static methods that GitHub Copilot can easily
    access to get contextual information about trading strategies.
    """
    
    @staticmethod
    def get_strategy_insights(strategy_type: str) -> str:
        """
        Retourne un contexte textuel que Copilot peut utiliser
        dans ses suggestions de code.
        
        Args:
            strategy_type: Type of strategy ('moving_average', 'rsi', etc.)
            
        Returns:
            Formatted string with historical context for Copilot
        """
        db = get_strategy_database()
        insights = db.get_strategy_performance_summary(strategy_type)
        
        # Format parameters for display
        best_params_str = CopilotStrategyAPI._format_parameters_for_copilot(insights.best_params)
        top_performers_str = CopilotStrategyAPI._format_top_performers(insights.top_performers)
        regime_effectiveness_str = CopilotStrategyAPI._format_regime_effectiveness(insights.regime_performance)
        failure_patterns_str = CopilotStrategyAPI._format_common_failures(insights.failure_patterns)
        recommendations_str = CopilotStrategyAPI._format_recommendations(insights.suggestions)
        
        context = f"""
HISTORICAL CONTEXT FOR {strategy_type.upper()} STRATEGIES:

🎯 OPTIMAL PARAMETERS DISCOVERED:
{best_params_str}

📊 PERFORMANCE PATTERNS:
- Best performers: {top_performers_str}
- Success rate: {insights.success_rate:.1f}%
- Average Sharpe ratio: {insights.avg_performance.get('sharpe_ratio', 0):.2f}

🏮 MARKET REGIME EFFECTIVENESS:
{regime_effectiveness_str}

⚠️ COMMON PITFALLS TO AVOID:
{failure_patterns_str}

💡 RECOMMENDED IMPROVEMENTS:
{recommendations_str}
"""
        
        return context.strip()
    
    @staticmethod
    def suggest_parameters(strategy_type: str, market_regime: str = None) -> Dict[str, Any]:
        """
        Suggère les meilleurs paramètres pour un type de stratégie.
        
        Args:
            strategy_type: Type of strategy
            market_regime: Optional market regime filter
            
        Returns:
            Dictionary with optimal parameters and metadata
        """
        db = get_strategy_database()
        optimal_params = db.get_optimal_parameters(strategy_type, market_regime)
        
        # Add metadata about parameter effectiveness
        insights = db.get_strategy_performance_summary(strategy_type)
        
        enhanced_params = {}
        for param, value in optimal_params.items():
            # Calculate parameter ranges from successful strategies
            successful_results = db.get_successful_strategies(strategy_type)
            param_values = [r.parameters.get(param) for r in successful_results if param in r.parameters]
            
            if param_values:
                min_val = min(param_values)
                max_val = max(param_values)
                best_val = value
                
                # Find best performing value's Sharpe ratio
                best_sharpe = 0
                for result in successful_results:
                    if result.parameters.get(param) == value:
                        best_sharpe = max(best_sharpe, result.performance_metrics.get('sharpe_ratio', 0))
                
                enhanced_params[param] = {
                    'recommended_value': value,
                    'min_value': min_val,
                    'max_value': max_val,
                    'best_sharpe': best_sharpe,
                    'description': CopilotStrategyAPI._get_parameter_description(param)
                }
            else:
                enhanced_params[param] = {
                    'recommended_value': value,
                    'description': CopilotStrategyAPI._get_parameter_description(param)
                }
        
        return enhanced_params
    
    @staticmethod
    def check_strategy_exists(strategy_signature: str) -> Dict[str, Any]:
        """
        Vérifie si une stratégie similaire existe déjà.
        
        Args:
            strategy_signature: Unique strategy signature
            
        Returns:
            Dictionary with existence info and suggestions
        """
        db = get_strategy_database()
        existing = db.find_similar_strategy(strategy_signature)
        
        if existing:
            return {
                "exists": True,
                "performance": {
                    "sharpe_ratio": existing.performance_metrics.get('sharpe_ratio', 0),
                    "total_return_pct": existing.performance_metrics.get('total_return_pct', 0),
                    "max_drawdown_pct": existing.performance_metrics.get('max_drawdown_pct', 0)
                },
                "parameters": existing.parameters,
                "suggestion": "Consider modifying these parameters instead of creating identical strategy"
            }
        
        return {
            "exists": False,
            "suggestion": "This appears to be a new strategy configuration"
        }
    
    @staticmethod
    def get_code_patterns(strategy_type: str) -> str:
        """
        Retourne des patterns de code qui fonctionnent bien.
        
        Args:
            strategy_type: Type of strategy
            
        Returns:
            Formatted string with successful code patterns
        """
        db = get_strategy_database()
        successful_strategies = db.get_successful_strategies(strategy_type)
        
        if not successful_strategies:
            return "No successful patterns available yet"
        
        # Analyze patterns from successful strategies
        patterns = CopilotStrategyAPI._extract_code_patterns(successful_strategies, strategy_type)
        
        return f"""
SUCCESSFUL CODE PATTERNS FOR {strategy_type.upper()}:

1. SIGNAL GENERATION:
{patterns['signal_generation']}

2. RISK MANAGEMENT:
{patterns['risk_management']}

3. FILTERS AND VALIDATION:
{patterns['filters']}
"""
    
    @staticmethod
    def get_implementation_hints(strategy_type: str) -> str:
        """
        Generate implementation hints based on successful strategies.
        
        Args:
            strategy_type: Type of strategy
            
        Returns:
            Formatted implementation hints
        """
        db = get_strategy_database()
        insights = db.get_strategy_performance_summary(strategy_type)
        
        hints = []
        
        # Parameter-based hints
        if insights.best_params:
            hints.append("PARAMETER OPTIMIZATION:")
            for param, value in insights.best_params.items():
                hints.append(f"  - {param}: Use value around {value} for best performance")
        
        # Pattern-based hints
        hints.append("\nIMPLEMENTATION BEST PRACTICES:")
        if strategy_type == "moving_average":
            hints.extend([
                "  - Calculate moving averages using pandas rolling windows",
                "  - Use crossover detection with shift() to avoid look-ahead bias",
                "  - Add volume confirmation to reduce false signals",
                "  - Consider RSI filter for trend strength validation"
            ])
        elif strategy_type == "rsi":
            hints.extend([
                "  - Use traditional RSI calculation with 14-period default",
                "  - Set oversold/overbought levels based on market volatility",
                "  - Add volume filter to confirm reversal signals",
                "  - Consider multiple timeframe confirmation"
            ])
        
        # Performance hints
        hints.append(f"\nPERFORMANCE EXPECTATIONS:")
        hints.append(f"  - Success rate: {insights.success_rate:.1f}%")
        avg_sharpe = insights.avg_performance.get('sharpe_ratio', 0)
        hints.append(f"  - Expected Sharpe ratio: {avg_sharpe:.2f}")
        
        return "\n".join(hints)
    
    @staticmethod
    def _format_parameters_for_copilot(params: Dict[str, Any]) -> str:
        """Format parameters for Copilot display."""
        if not params:
            return "No optimal parameters found yet"
        
        formatted = []
        for param, value in params.items():
            formatted.append(f"  - {param}: {value}")
        
        return "\n".join(formatted)
    
    @staticmethod
    def _format_top_performers(performers: List[Dict[str, Any]]) -> str:
        """Format top performers for display."""
        if not performers:
            return "No performance data available"
        
        formatted = []
        for i, perf in enumerate(performers[:3], 1):
            params_str = ", ".join([f"{k}={v}" for k, v in perf['params'].items()])
            formatted.append(f"{i}. {params_str} (Sharpe: {perf['sharpe']:.2f})")
        
        return "; ".join(formatted)
    
    @staticmethod
    def _format_regime_effectiveness(regime_perf: Dict[str, Dict[str, float]]) -> str:
        """Format regime performance for display."""
        if not regime_perf:
            return "No regime analysis available"
        
        formatted = []
        for regime, metrics in regime_perf.items():
            avg_sharpe = metrics.get('avg_sharpe', 0)
            count = metrics.get('count', 0)
            formatted.append(f"  - {regime}: {avg_sharpe:.2f} avg Sharpe ({count} tests)")
        
        return "\n".join(formatted)
    
    @staticmethod
    def _format_common_failures(failures: List[str]) -> str:
        """Format failure patterns for display."""
        if not failures:
            return "No failure patterns identified"
        
        return "\n".join([f"  - {failure}" for failure in failures])
    
    @staticmethod
    def _format_recommendations(suggestions: List[str]) -> str:
        """Format recommendations for display."""
        if not suggestions:
            return "No specific recommendations available"
        
        return "\n".join([f"  - {suggestion}" for suggestion in suggestions])
    
    @staticmethod
    def _get_parameter_description(param: str) -> str:
        """Get description for a parameter."""
        descriptions = {
            'short_window': 'Period for short-term moving average',
            'long_window': 'Period for long-term moving average',
            'rsi_period': 'Period for RSI calculation',
            'oversold': 'RSI oversold threshold',
            'overbought': 'RSI overbought threshold',
            'use_rsi_filter': 'Whether to use RSI trend filter',
            'volume_confirmation': 'Whether to require volume confirmation',
            'lookback_period': 'Period for trend analysis',
            'ma_period': 'Moving average period',
            'atr_period': 'Average True Range period'
        }
        
        return descriptions.get(param, 'Strategy parameter')
    
    @staticmethod
    def _extract_code_patterns(successful_strategies: List[StrategyResult], strategy_type: str) -> Dict[str, str]:
        """Extract code patterns from successful strategies."""
        patterns = {
            'signal_generation': '',
            'risk_management': '',
            'filters': ''
        }
        
        if strategy_type == "moving_average":
            patterns['signal_generation'] = """
# Calculate moving averages
short_ma = data['Close'].rolling(window=short_window).mean()
long_ma = data['Close'].rolling(window=long_window).mean()

# Generate crossover signals
signals = pd.DataFrame(index=data.index)
signals['Signal'] = 'HOLD'
signals['Position'] = 0

# Bullish crossover
bullish_cross = (short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))
signals.loc[bullish_cross, 'Signal'] = 'BUY'
signals.loc[bullish_cross, 'Position'] = 1

# Bearish crossover  
bearish_cross = (short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))
signals.loc[bearish_cross, 'Signal'] = 'SELL'
signals.loc[bearish_cross, 'Position'] = -1"""
            
            patterns['risk_management'] = """
# Add stop-loss and take-profit
signals['Stop_Loss'] = data['Close'] * 0.95  # 5% stop loss
signals['Take_Profit'] = data['Close'] * 1.10  # 10% take profit

# Position sizing based on volatility
atr = data['High'].subtract(data['Low']).rolling(14).mean()
position_size = 10000 / (2 * atr)  # Risk 2 ATR per trade"""
            
            patterns['filters'] = """
# Volume confirmation filter
volume_ma = data['Volume'].rolling(20).mean()
volume_filter = data['Volume'] > volume_ma

# RSI trend filter
rsi = calculate_rsi(data['Close'], 14)
rsi_filter = (rsi > 30) & (rsi < 70)  # Avoid extreme conditions

# Apply filters to signals
signals.loc[~volume_filter, 'Signal'] = 'HOLD'
signals.loc[~rsi_filter, 'Signal'] = 'HOLD'"""
        
        elif strategy_type == "rsi":
            patterns['signal_generation'] = """
# Calculate RSI
rsi = calculate_rsi(data['Close'], rsi_period)

# Generate mean reversion signals
signals = pd.DataFrame(index=data.index)
signals['Signal'] = 'HOLD'
signals['Position'] = 0

# Oversold condition (BUY signal)
oversold_condition = rsi < oversold_threshold
signals.loc[oversold_condition, 'Signal'] = 'BUY'
signals.loc[oversold_condition, 'Position'] = 1

# Overbought condition (SELL signal)
overbought_condition = rsi > overbought_threshold
signals.loc[overbought_condition, 'Signal'] = 'SELL'
signals.loc[overbought_condition, 'Position'] = -1"""
        
        return patterns