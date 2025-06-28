# Evaluation Module

The evaluation module provides comprehensive performance metrics and analysis tools for trading strategies and portfolio management. This module is designed to complement the simulation framework with advanced statistical analysis and risk assessment capabilities.

## Features (Planned)

- **Advanced Performance Metrics**: Beyond basic returns and Sharpe ratio
- **Risk Assessment**: Comprehensive risk analysis including VaR and CVaR
- **Benchmark Comparison**: Compare strategies against market benchmarks
- **Statistical Analysis**: Statistical significance testing and confidence intervals
- **Portfolio Attribution**: Performance attribution analysis for multi-strategy portfolios

## Planned Metrics

### Risk-Adjusted Returns
- **Sharpe Ratio**: Risk-adjusted return using standard deviation
- **Sortino Ratio**: Downside risk-adjusted return
- **Calmar Ratio**: Return vs. maximum drawdown
- **Information Ratio**: Active return vs. tracking error

### Risk Metrics
- **Value at Risk (VaR)**: Potential loss at given confidence level
- **Conditional VaR (CVaR)**: Expected loss beyond VaR threshold
- **Maximum Drawdown**: Peak-to-trough decline analysis
- **Ulcer Index**: Downside risk measure
- **Beta**: Systematic risk relative to market

### Performance Attribution
- **Factor Analysis**: Decompose returns by risk factors
- **Sector Attribution**: Performance by sector exposure
- **Style Analysis**: Growth vs. value attribution
- **Market Timing**: Timing vs. selection contribution

### Statistical Tests
- **T-Tests**: Statistical significance of returns
- **Jarque-Bera**: Test for normality of returns
- **Ljung-Box**: Test for autocorrelation
- **ARCH Test**: Test for heteroskedasticity

## Usage Example (Future)

```python
from evaluation import PerformanceAnalyzer, RiskAnalyzer, BenchmarkComparison

# Analyze strategy performance
analyzer = PerformanceAnalyzer(performance_report)
metrics = analyzer.calculate_advanced_metrics()

# Risk analysis
risk_analyzer = RiskAnalyzer(returns_series)
var_95 = risk_analyzer.calculate_var(confidence_level=0.95)
cvar_95 = risk_analyzer.calculate_cvar(confidence_level=0.95)

# Benchmark comparison
benchmark_data = load_market_benchmark('SPY')
comparison = BenchmarkComparison(strategy_returns, benchmark_data)
alpha, beta = comparison.calculate_alpha_beta()
```

## Integration with Simulation

The evaluation module is designed to work seamlessly with backtesting results:

```python
# Run backtest (from simulation module)
performance_report = backtester.run_backtest(df, signals)

# Advanced evaluation (future functionality)
evaluator = PerformanceEvaluator(performance_report)
risk_metrics = evaluator.analyze_risk()
benchmark_analysis = evaluator.compare_to_benchmark('SPY')
```

## Current Status

This module is currently in development. The simulation module already provides basic performance metrics including:

- Total and annualized returns
- Sharpe ratio
- Maximum drawdown
- Win rate and profit factor
- Transaction analysis

For current evaluation capabilities, refer to the [simulation module](../simulation/README.md) which includes the `PerformanceReport` class with basic metrics.

## Future Development

Planned enhancements include:

1. **Advanced Risk Metrics**: VaR, CVaR, and tail risk measures
2. **Factor Models**: Fama-French factor analysis
3. **Regime Analysis**: Performance across different market conditions
4. **Monte Carlo Simulation**: Stress testing and scenario analysis
5. **Optimization Integration**: Performance-based parameter tuning

## Contributing

This module is open for contributions. Key areas for development:

- Risk metric implementations
- Benchmark data integration
- Statistical testing frameworks
- Performance visualization tools
- Factor model implementations

Please see the main project README for contribution guidelines.