"""
Grid Search optimization module for trading strategy parameter optimization.

This module provides comprehensive grid search capabilities for automatically
optimizing trading strategy parameters to maximize specified performance metrics.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional, Callable, Union
import itertools
import pandas as pd
import numpy as np
from simulation.backtester import Backtester, PerformanceReport
from evaluation.metrics import evaluate_performance

# Configure logging
logger = logging.getLogger(__name__)


class GridSearchOptimizer:
    """
    Grid Search optimizer for trading strategy parameters.

    This class performs exhaustive search over parameter combinations to find
    the optimal parameters that maximize a specified objective function.

    Parameters
    ----------
    strategy_class : class
        Trading strategy class to optimize
    backtester : Backtester
        Backtester instance for running simulations
    param_grid : Dict[str, List]
        Parameter grid defining the search space
    objective : str or callable, default "sharpe_ratio"
        Optimization objective. Can be a metric name or custom function
    """

    def __init__(
        self,
        strategy_class,
        backtester: Backtester,
        param_grid: Dict[str, List],
        objective: Union[str, Callable] = "sharpe_ratio",
    ):
        self.strategy_class = strategy_class
        self.backtester = backtester
        self.param_grid = param_grid
        self.objective = objective

        # Validation
        self._validate_param_grid()

        # Results storage
        self.results = []
        self.best_params = None
        self.best_score = -float("inf")

        logger.info(
            f"Initialized GridSearchOptimizer with {self._count_combinations()} parameter combinations"
        )

    def _validate_param_grid(self):
        """Validate the parameter grid."""
        if not isinstance(self.param_grid, dict):
            raise ValueError("param_grid must be a dictionary")

        if not self.param_grid:
            raise ValueError("param_grid cannot be empty")

        for param_name, values in self.param_grid.items():
            if not isinstance(values, (list, tuple, np.ndarray)):
                raise ValueError(
                    f"Parameter '{param_name}' values must be a list, tuple, or array"
                )
            if len(values) == 0:
                raise ValueError(
                    f"Parameter '{param_name}' must have at least one value"
                )

    def _count_combinations(self) -> int:
        """Count total number of parameter combinations."""
        return np.prod([len(values) for values in self.param_grid.values()])

    def _generate_param_combinations(self):
        """Generate all parameter combinations from the grid."""
        param_names = list(self.param_grid.keys())
        param_values = [self.param_grid[name] for name in param_names]

        for combination in itertools.product(*param_values):
            yield dict(zip(param_names, combination))

    def _evaluate_objective(self, performance_report: PerformanceReport) -> float:
        """Evaluate the objective function for a performance report."""
        if callable(self.objective):
            return self.objective(performance_report)

        # Handle string objectives
        summary = performance_report.summary()

        if isinstance(self.objective, str):
            if self.objective in summary:
                return summary[self.objective]
            elif self.objective.lower() == "roi":
                return summary.get("total_return", -float("inf"))
            elif self.objective.lower() == "drawdown":
                # For drawdown, we want to minimize (so return negative)
                return -abs(summary.get("max_drawdown_pct", float("inf")))
            elif self.objective.lower() == "profit_factor":
                return summary.get("profit_factor", 0)
            elif self.objective.lower() == "win_rate":
                return summary.get("win_rate", 0)
            else:
                logger.warning(
                    f"Unknown objective '{self.objective}', using sharpe_ratio"
                )
                return summary.get("sharpe_ratio", -float("inf"))

        return -float("inf")

    def optimize(
        self,
        data: pd.DataFrame,
        validation_split: Optional[float] = None,
        verbose: bool = True,
    ) -> Tuple[Dict[str, Any], "OptimizationReport"]:
        """
        Run the grid search optimization.

        Parameters
        ----------
        data : pd.DataFrame
            Historical price data for backtesting
        validation_split : float, optional
            Fraction of data to reserve for validation (not yet implemented)
        verbose : bool, default True
            Whether to print progress information

        Returns
        -------
        Tuple[Dict[str, Any], OptimizationReport]
            Best parameters found and detailed optimization report
        """
        if data.empty:
            raise ValueError("Data cannot be empty")

        logger.info("Starting grid search optimization")
        self.results = []
        self.best_params = None
        self.best_score = -float("inf")

        total_combinations = self._count_combinations()

        for i, params in enumerate(self._generate_param_combinations()):
            if verbose and i % max(1, total_combinations // 10) == 0:
                logger.info(f"Progress: {i+1}/{total_combinations} combinations tested")

            try:
                # Create strategy with current parameters
                strategy = self.strategy_class(**params)

                # Generate signals
                signals = strategy.generate_signals(data)

                # Run backtest
                self.backtester.reset()
                performance_report = self.backtester.run_backtest(data, signals)

                # Evaluate objective
                score = self._evaluate_objective(performance_report)

                # Store results
                result = {
                    "params": params.copy(),
                    "score": score,
                    "performance_summary": performance_report.summary(),
                }
                self.results.append(result)

                # Update best if this is better
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params.copy()

            except Exception as e:
                logger.warning(f"Error testing parameters {params}: {str(e)}")
                # Store failed result
                result = {
                    "params": params.copy(),
                    "score": -float("inf"),
                    "error": str(e),
                    "performance_summary": {},
                }
                self.results.append(result)

        logger.info(f"Grid search completed. Best score: {self.best_score:.4f}")

        # Create optimization report
        report = OptimizationReport(self.results, self.objective, self.param_grid)

        return self.best_params, report

    def get_top_results(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        Get the top N parameter combinations.

        Parameters
        ----------
        n : int, default 5
            Number of top results to return

        Returns
        -------
        List[Dict[str, Any]]
            Top N results sorted by score
        """
        if not self.results:
            return []

        sorted_results = sorted(self.results, key=lambda x: x["score"], reverse=True)
        return sorted_results[:n]

    def analyze_parameter_sensitivity(self) -> Dict[str, Dict[str, float]]:
        """
        Analyze parameter sensitivity by calculating correlation between parameters and scores.

        Returns
        -------
        Dict[str, Dict[str, float]]
            Parameter sensitivity analysis
        """
        if not self.results:
            return {}

        # Create DataFrame from results
        data_for_analysis = []
        for result in self.results:
            if not np.isfinite(result["score"]):
                continue
            row = result["params"].copy()
            row["score"] = result["score"]
            data_for_analysis.append(row)

        if not data_for_analysis:
            return {}

        df = pd.DataFrame(data_for_analysis)

        sensitivity = {}
        for param in self.param_grid.keys():
            if param in df.columns:
                correlation = df[param].corr(df["score"])

                # Calculate performance statistics for this parameter
                param_stats = (
                    df.groupby(param)["score"]
                    .agg(["mean", "std", "min", "max", "count"])
                    .to_dict("index")
                )

                sensitivity[param] = {
                    "correlation": correlation if not np.isnan(correlation) else 0,
                    "param_stats": param_stats,
                }

        return sensitivity


class OptimizationReport:
    """
    Comprehensive report of optimization results.

    This class provides detailed analysis and reporting capabilities
    for grid search optimization results.
    """

    def __init__(
        self,
        results: List[Dict[str, Any]],
        objective: Union[str, Callable],
        param_grid: Dict[str, List],
    ):
        self.results = results
        self.objective = objective
        self.param_grid = param_grid

        # Process results
        self.valid_results = [r for r in results if np.isfinite(r["score"])]
        self.failed_results = [r for r in results if not np.isfinite(r["score"])]

    def summary(self) -> Dict[str, Any]:
        """Generate a summary of the optimization results."""
        if not self.valid_results:
            return {
                "total_combinations": len(self.results),
                "successful_combinations": 0,
                "failed_combinations": len(self.failed_results),
                "best_score": None,
                "best_params": None,
            }

        best_result = max(self.valid_results, key=lambda x: x["score"])
        scores = [r["score"] for r in self.valid_results]

        return {
            "total_combinations": len(self.results),
            "successful_combinations": len(self.valid_results),
            "failed_combinations": len(self.failed_results),
            "best_score": best_result["score"],
            "best_params": best_result["params"],
            "score_statistics": {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "min": np.min(scores),
                "max": np.max(scores),
                "median": np.median(scores),
            },
        }

    def get_parameter_impact(self) -> Dict[str, float]:
        """
        Calculate the impact of each parameter on the objective score.

        Returns
        -------
        Dict[str, float]
            Parameter impact scores (correlation with objective)
        """
        if not self.valid_results:
            return {}

        # Create DataFrame for analysis
        data = []
        for result in self.valid_results:
            row = result["params"].copy()
            row["score"] = result["score"]
            data.append(row)

        df = pd.DataFrame(data)

        impact = {}
        for param in self.param_grid.keys():
            if param in df.columns:
                correlation = df[param].corr(df["score"])
                impact[param] = correlation if not np.isnan(correlation) else 0

        return impact

    def get_best_results(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get the top N best results."""
        if not self.valid_results:
            return []

        return sorted(self.valid_results, key=lambda x: x["score"], reverse=True)[:n]

    def print_summary(self) -> str:
        """Generate a formatted summary report."""
        summary = self.summary()

        lines = []
        lines.append("=" * 50)
        lines.append("GRID SEARCH OPTIMIZATION REPORT")
        lines.append("=" * 50)

        lines.append(f"\nOptimization Objective: {self.objective}")
        lines.append(f"Total Combinations: {summary['total_combinations']}")
        lines.append(f"Successful: {summary['successful_combinations']}")
        lines.append(f"Failed: {summary['failed_combinations']}")

        if summary["best_score"] is not None:
            lines.append(f"\nBest Score: {summary['best_score']:.4f}")
            lines.append("Best Parameters:")
            for param, value in summary["best_params"].items():
                lines.append(f"  {param}: {value}")

            stats = summary["score_statistics"]
            lines.append(f"\nScore Statistics:")
            lines.append(f"  Mean: {stats['mean']:.4f}")
            lines.append(f"  Std:  {stats['std']:.4f}")
            lines.append(f"  Min:  {stats['min']:.4f}")
            lines.append(f"  Max:  {stats['max']:.4f}")

        # Parameter impact
        impact = self.get_parameter_impact()
        if impact:
            lines.append(f"\nParameter Impact (Correlation with {self.objective}):")
            for param, corr in sorted(
                impact.items(), key=lambda x: abs(x[1]), reverse=True
            ):
                lines.append(f"  {param}: {corr:.3f}")

        lines.append("\n" + "=" * 50)

        return "\n".join(lines)


def optimize_strategy(
    strategy_class,
    data: pd.DataFrame,
    param_space: Dict[str, List],
    objective: Union[str, Callable] = "sharpe_ratio",
    initial_capital: float = 10000,
    commission: float = 0.001,
    slippage: float = 0.0005,
) -> Tuple[Dict[str, Any], OptimizationReport]:
    """
    Convenient function to optimize a strategy with grid search.

    Parameters
    ----------
    strategy_class : class
        Trading strategy class to optimize
    data : pd.DataFrame
        Historical price data
    param_space : Dict[str, List]
        Parameter search space
    objective : str or callable, default "sharpe_ratio"
        Optimization objective
    initial_capital : float, default 10000
        Initial capital for backtesting
    commission : float, default 0.001
        Commission rate
    slippage : float, default 0.0005
        Slippage rate

    Returns
    -------
    Tuple[Dict[str, Any], OptimizationReport]
        Best parameters and optimization report
    """
    # Create backtester
    backtester = Backtester(
        initial_capital=initial_capital, commission=commission, slippage=slippage
    )

    # Create optimizer
    optimizer = GridSearchOptimizer(
        strategy_class=strategy_class,
        backtester=backtester,
        param_grid=param_space,
        objective=objective,
    )

    # Run optimization
    return optimizer.optimize(data)
