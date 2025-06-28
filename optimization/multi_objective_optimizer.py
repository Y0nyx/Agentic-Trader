"""
Multi-objective optimization for trading strategies.

This module provides optimization capabilities that can handle multiple
objectives simultaneously, such as maximizing returns while minimizing risk.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Callable, Union, Optional, Tuple
from dataclasses import dataclass
from simulation.backtester import Backtester
from evaluation.metrics import evaluate_performance

logger = logging.getLogger(__name__)


@dataclass
class OptimizationObjective:
    """
    Represents a single optimization objective.
    
    Attributes
    ----------
    name : str
        Name of the objective
    maximize : bool
        Whether to maximize (True) or minimize (False) this objective
    weight : float
        Weight for this objective in multi-objective optimization
    constraint : Optional[Dict]
        Constraint bounds for this objective (e.g., {'min': 0.5, 'max': 1.0})
    """
    name: str
    maximize: bool = True
    weight: float = 1.0
    constraint: Optional[Dict[str, float]] = None


@dataclass
class OptimizationConstraint:
    """
    Represents a constraint for optimization.
    
    Attributes
    ----------
    name : str
        Name of the constraint metric
    operator : str
        Constraint operator ('>', '<', '>=', '<=', '==')
    value : float
        Constraint value
    """
    name: str
    operator: str
    value: float


class MultiObjectiveOptimizer:
    """
    Multi-objective optimizer for trading strategies.
    
    This optimizer can handle multiple objectives simultaneously and supports
    constraints on strategy performance metrics.
    
    Parameters
    ----------
    strategy_class : class
        Strategy class to optimize
    backtester : Backtester
        Backtester instance for performance evaluation
    param_grid : Dict
        Parameter grid for optimization
    objectives : List[OptimizationObjective]
        List of optimization objectives
    constraints : List[OptimizationConstraint], optional
        List of constraints to apply
    """
    
    def __init__(
        self,
        strategy_class,
        backtester: Backtester,
        param_grid: Dict[str, List],
        objectives: List[OptimizationObjective],
        constraints: Optional[List[OptimizationConstraint]] = None,
    ):
        self.strategy_class = strategy_class
        self.backtester = backtester
        self.param_grid = param_grid
        self.objectives = objectives
        self.constraints = constraints or []
        
        # Validate objectives
        if not objectives:
            raise ValueError("At least one objective must be specified")
        
        # Normalize weights
        total_weight = sum(obj.weight for obj in objectives)
        for obj in self.objectives:
            obj.weight = obj.weight / total_weight
        
        self.results = []
        
        logger.info(
            f"Initialized MultiObjectiveOptimizer with {len(objectives)} objectives "
            f"and {len(self.constraints)} constraints"
        )
    
    def _generate_param_combinations(self) -> List[Dict]:
        """Generate all parameter combinations from the grid."""
        import itertools
        
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        
        combinations = []
        for combination in itertools.product(*values):
            param_dict = dict(zip(keys, combination))
            combinations.append(param_dict)
        
        return combinations
    
    def _evaluate_objectives(self, performance_report, benchmark_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Evaluate all objectives for a given performance report.
        
        Parameters
        ----------
        performance_report : PerformanceReport
            Backtest results
        benchmark_data : pd.DataFrame, optional
            Benchmark data for comparison
            
        Returns
        -------
        Dict[str, float]
            Objective values
        """
        summary = performance_report.summary()
        
        # Calculate additional metrics if benchmark provided
        if benchmark_data is not None:
            detailed_metrics = evaluate_performance(performance_report, benchmark_data)
            summary.update(detailed_metrics)
        
        objective_values = {}
        
        for obj in self.objectives:
            if obj.name in summary:
                value = summary[obj.name]
            elif obj.name == "total_return":
                value = summary.get("total_return", 0)
            elif obj.name == "sharpe_ratio":
                value = summary.get("sharpe_ratio", 0)
            elif obj.name == "win_rate":
                value = summary.get("win_rate", 0)
            elif obj.name == "profit_factor":
                value = summary.get("profit_factor", 0)
            elif obj.name == "max_drawdown":
                value = -abs(summary.get("max_drawdown_pct", 0))  # Make negative for minimization
            elif obj.name == "num_trades":
                value = -summary.get("num_trades", 0)  # Negative to minimize
            elif obj.name == "calmar_ratio":
                value = summary.get("calmar_ratio", 0)
            elif obj.name == "sortino_ratio":
                value = summary.get("sortino_ratio", 0)
            else:
                logger.warning(f"Unknown objective: {obj.name}")
                value = 0
            
            objective_values[obj.name] = value
        
        return objective_values
    
    def _check_constraints(self, objective_values: Dict[str, float]) -> bool:
        """
        Check if all constraints are satisfied.
        
        Parameters
        ----------
        objective_values : Dict[str, float]
            Calculated objective values
            
        Returns
        -------
        bool
            True if all constraints are satisfied
        """
        for constraint in self.constraints:
            if constraint.name not in objective_values:
                logger.warning(f"Constraint metric '{constraint.name}' not found in results")
                continue
            
            value = objective_values[constraint.name]
            
            if constraint.operator == ">":
                if not (value > constraint.value):
                    return False
            elif constraint.operator == "<":
                if not (value < constraint.value):
                    return False
            elif constraint.operator == ">=":
                if not (value >= constraint.value):
                    return False
            elif constraint.operator == "<=":
                if not (value <= constraint.value):
                    return False
            elif constraint.operator == "==":
                if not (abs(value - constraint.value) < 1e-6):
                    return False
            else:
                logger.warning(f"Unknown constraint operator: {constraint.operator}")
        
        return True
    
    def _calculate_composite_score(self, objective_values: Dict[str, float]) -> float:
        """
        Calculate weighted composite score from multiple objectives.
        
        Parameters
        ----------
        objective_values : Dict[str, float]
            Calculated objective values
            
        Returns
        -------
        float
            Composite score
        """
        score = 0.0
        
        for obj in self.objectives:
            if obj.name in objective_values:
                value = objective_values[obj.name]
                
                # Apply constraint bounds if specified
                if obj.constraint:
                    min_val = obj.constraint.get('min')
                    max_val = obj.constraint.get('max')
                    
                    if min_val is not None and value < min_val:
                        value = min_val
                    if max_val is not None and value > max_val:
                        value = max_val
                
                # Normalize and weight the objective
                if obj.maximize:
                    score += obj.weight * value
                else:
                    score -= obj.weight * value
        
        return score
    
    def optimize(self, data: pd.DataFrame, benchmark_data: Optional[pd.DataFrame] = None) -> Tuple[Dict, Dict]:
        """
        Run multi-objective optimization.
        
        Parameters
        ----------
        data : pd.DataFrame
            Historical data for backtesting
        benchmark_data : pd.DataFrame, optional
            Benchmark data for comparison
            
        Returns
        -------
        Tuple[Dict, Dict]
            Best parameters and optimization results
        """
        logger.info("Starting multi-objective optimization...")
        
        param_combinations = self._generate_param_combinations()
        logger.info(f"Testing {len(param_combinations)} parameter combinations")
        
        best_score = -float('inf')
        best_params = None
        valid_results = []
        
        for i, params in enumerate(param_combinations):
            try:
                # Create strategy with parameters
                strategy = self.strategy_class(**params)
                signals = strategy.generate_signals(data)
                
                # Run backtest
                performance_report = self.backtester.run_backtest(data, signals)
                
                # Evaluate objectives
                objective_values = self._evaluate_objectives(performance_report, benchmark_data)
                
                # Check constraints
                constraints_satisfied = self._check_constraints(objective_values)
                
                if constraints_satisfied:
                    # Calculate composite score
                    composite_score = self._calculate_composite_score(objective_values)
                    
                    # Store result
                    result = {
                        "params": params,
                        "objectives": objective_values,
                        "composite_score": composite_score,
                        "constraints_satisfied": True,
                        "performance_summary": performance_report.summary(),
                    }
                    
                    valid_results.append(result)
                    
                    # Check if this is the best result
                    if composite_score > best_score:
                        best_score = composite_score
                        best_params = params
                        
                        logger.info(
                            f"New best result found: score={composite_score:.4f}, "
                            f"params={params}"
                        )
                else:
                    logger.debug(f"Constraints not satisfied for params: {params}")
                    
            except Exception as e:
                logger.error(f"Error evaluating params {params}: {e}")
                continue
        
        # Store results
        self.results = valid_results
        
        logger.info(f"Optimization completed. {len(valid_results)} valid results found.")
        
        if best_params is None:
            raise ValueError("No valid parameter combinations found that satisfy constraints")
        
        # Prepare summary
        optimization_summary = {
            "total_combinations": len(param_combinations),
            "valid_combinations": len(valid_results),
            "best_score": best_score,
            "best_params": best_params,
            "objectives": [obj.name for obj in self.objectives],
            "constraints": [(c.name, c.operator, c.value) for c in self.constraints],
        }
        
        return best_params, optimization_summary
    
    def get_pareto_front(self) -> List[Dict]:
        """
        Get the Pareto-optimal solutions.
        
        Returns
        -------
        List[Dict]
            List of Pareto-optimal solutions
        """
        if not self.results:
            return []
        
        pareto_solutions = []
        
        for i, result_i in enumerate(self.results):
            is_dominated = False
            
            for j, result_j in enumerate(self.results):
                if i == j:
                    continue
                
                # Check if result_j dominates result_i
                dominates = True
                for obj in self.objectives:
                    obj_name = obj.name
                    
                    if obj_name not in result_i["objectives"] or obj_name not in result_j["objectives"]:
                        continue
                    
                    val_i = result_i["objectives"][obj_name]
                    val_j = result_j["objectives"][obj_name]
                    
                    if obj.maximize:
                        if val_j <= val_i:
                            dominates = False
                            break
                    else:
                        if val_j >= val_i:
                            dominates = False
                            break
                
                if dominates:
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_solutions.append(result_i)
        
        return pareto_solutions
    
    def print_results_summary(self) -> str:
        """
        Print a summary of optimization results.
        
        Returns
        -------
        str
            Formatted summary
        """
        if not self.results:
            return "No optimization results available."
        
        summary = []
        summary.append("=" * 60)
        summary.append("MULTI-OBJECTIVE OPTIMIZATION RESULTS")
        summary.append("=" * 60)
        
        summary.append(f"\nTotal combinations tested: {len(self.results)}")
        
        # Best result
        best_result = max(self.results, key=lambda x: x["composite_score"])
        summary.append(f"\nBest composite score: {best_result['composite_score']:.4f}")
        summary.append(f"Best parameters: {best_result['params']}")
        
        summary.append("\nObjective values for best result:")
        for obj in self.objectives:
            if obj.name in best_result["objectives"]:
                value = best_result["objectives"][obj.name]
                summary.append(f"  {obj.name}: {value:.4f} (weight: {obj.weight:.2f})")
        
        # Pareto front
        pareto_solutions = self.get_pareto_front()
        summary.append(f"\nPareto-optimal solutions found: {len(pareto_solutions)}")
        
        return "\n".join(summary)