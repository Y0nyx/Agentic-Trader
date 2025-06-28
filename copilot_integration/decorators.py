"""
Decorators for GitHub Copilot integration.

This module provides decorators that automatically inject historical strategy
context into function docstrings to guide GitHub Copilot suggestions.
"""

import functools
import inspect
from typing import Optional, Callable, Any
from .copilot_api import CopilotStrategyAPI


def copilot_strategy_context(strategy_type: Optional[str] = None):
    """
    Décorateur qui injecte automatiquement le contexte historique
    dans les docstrings pour guider Copilot.
    
    Args:
        strategy_type: Strategy type ('moving_average', 'rsi', etc.).
                      If None, attempts to auto-detect from function name.
    
    Example:
        @copilot_strategy_context("moving_average")
        def create_ma_strategy():
            '''Create optimized moving average strategy'''
            pass
    """
    def decorator(func: Callable) -> Callable:
        # Auto-detect strategy type if not provided
        detected_type = strategy_type
        if not detected_type:
            detected_type = _detect_strategy_type_from_function(func)
        
        if not detected_type:
            detected_type = "moving_average"  # Default fallback
        
        # Get historical context
        try:
            context = CopilotStrategyAPI.get_strategy_insights(detected_type)
            implementation_hints = CopilotStrategyAPI.get_implementation_hints(detected_type)
        except Exception as e:
            # Fallback if database is not available
            context = f"Context loading failed: {e}"
            implementation_hints = "Use default implementation patterns"
        
        # Enhance the docstring
        original_doc = func.__doc__ or ""
        
        enhanced_doc = f"""{original_doc}

{context}

COPILOT CODING GUIDELINES:
- Use parameters within the optimal ranges shown above
- Implement the recommended improvements
- Avoid the common pitfalls listed
- Consider market regime adaptations

{implementation_hints}
"""
        
        # Create wrapper function with enhanced docstring
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        wrapper.__doc__ = enhanced_doc.strip()
        
        # Add metadata for introspection
        wrapper._copilot_strategy_type = detected_type
        wrapper._copilot_enhanced = True
        
        return wrapper
    
    return decorator


def copilot_parameter_hints(strategy_type: str):
    """
    Decorator that adds parameter optimization hints to function docstrings.
    
    Args:
        strategy_type: Strategy type for parameter suggestions
    
    Example:
        @copilot_parameter_hints("moving_average")
        def __init__(self, short_window=10, long_window=30):
            pass
    """
    def decorator(func: Callable) -> Callable:
        try:
            # Get parameter suggestions
            param_suggestions = CopilotStrategyAPI.suggest_parameters(strategy_type)
            hints = _generate_parameter_comments(param_suggestions)
        except Exception as e:
            hints = f"Parameter hints unavailable: {e}"
        
        # Enhance docstring with parameter hints
        original_doc = func.__doc__ or ""
        enhanced_doc = f"""{original_doc}

PARAMETER OPTIMIZATION HINTS:
{hints}
"""
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        wrapper.__doc__ = enhanced_doc.strip()
        wrapper._copilot_parameter_hints = True
        
        return wrapper
    
    return decorator


def copilot_validation_check(strategy_type: str):
    """
    Decorator that adds validation against existing strategies.
    
    Args:
        strategy_type: Strategy type for validation
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract parameters from function call
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Generate strategy signature
            params = dict(bound_args.arguments)
            signature = _generate_strategy_signature(strategy_type, params)
            
            # Check if strategy exists
            try:
                validation = CopilotStrategyAPI.check_strategy_exists(signature)
                
                if validation["exists"]:
                    print(f"⚠️  Similar strategy exists!")
                    print(f"📊 Previous performance: {validation['performance']}")
                    print(f"💡 {validation['suggestion']}")
            except Exception as e:
                print(f"Validation check failed: {e}")
            
            return func(*args, **kwargs)
        
        wrapper._copilot_validation = True
        return wrapper
    
    return decorator


class CopilotCommentGenerator:
    """Génère des commentaires contextuels pour guider Copilot."""
    
    @staticmethod
    def generate_parameter_comments(strategy_type: str) -> str:
        """Génère des commentaires sur les paramètres optimaux."""
        try:
            optimal_params = CopilotStrategyAPI.suggest_parameters(strategy_type)
            
            comments = []
            for param, info in optimal_params.items():
                if isinstance(info, dict):
                    recommended = info.get('recommended_value', 'N/A')
                    min_val = info.get('min_value', 'N/A')
                    max_val = info.get('max_value', 'N/A')
                    best_sharpe = info.get('best_sharpe', 0)
                    
                    comment = f"""
# {param}: Optimal range [{min_val}, {max_val}]
# Best value found: {recommended} (Sharpe: {best_sharpe:.2f})
# {info.get('description', 'Strategy parameter')}"""
                    comments.append(comment)
                else:
                    comment = f"# {param}: {info}"
                    comments.append(comment)
            
            return "\n".join(comments)
        except Exception as e:
            return f"# Parameter comments unavailable: {e}"
    
    @staticmethod
    def generate_implementation_hints(strategy_type: str) -> str:
        """Génère des hints d'implémentation basés sur l'historique."""
        try:
            patterns = CopilotStrategyAPI.get_code_patterns(strategy_type)
            
            return f"""
# IMPLEMENTATION HINTS BASED ON SUCCESSFUL STRATEGIES:
{patterns}
"""
        except Exception as e:
            return f"# Implementation hints unavailable: {e}"


def _detect_strategy_type_from_function(func: Callable) -> Optional[str]:
    """Auto-detect strategy type from function name or module."""
    func_name = func.__name__.lower()
    module_name = func.__module__.lower() if func.__module__ else ""
    
    # Common patterns for strategy type detection
    if 'moving_average' in func_name or 'ma_' in func_name or 'moving_average' in module_name:
        return 'moving_average'
    elif 'rsi' in func_name or 'rsi' in module_name:
        return 'rsi'
    elif 'trend' in func_name or 'trend' in module_name:
        return 'trend_following'
    elif 'bollinger' in func_name or 'bollinger' in module_name:
        return 'bollinger_bands'
    elif 'macd' in func_name or 'macd' in module_name:
        return 'macd'
    
    return None


def _generate_parameter_comments(param_suggestions: dict) -> str:
    """Generate formatted parameter comments."""
    comments = []
    
    for param, info in param_suggestions.items():
        if isinstance(info, dict):
            recommended = info.get('recommended_value', 'N/A')
            description = info.get('description', 'Strategy parameter')
            comments.append(f"  - {param}: {recommended} ({description})")
        else:
            comments.append(f"  - {param}: {info}")
    
    return "\n".join(comments) if comments else "No parameter hints available"


def _generate_strategy_signature(strategy_type: str, params: dict) -> str:
    """Generate a unique signature for strategy validation."""
    import hashlib
    import json
    
    # Create signature from strategy type and parameters
    param_str = json.dumps(params, sort_keys=True)
    content = f"{strategy_type}_{param_str}"
    return hashlib.md5(content.encode()).hexdigest()[:12]