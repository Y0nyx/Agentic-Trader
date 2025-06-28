# Tests Module

The tests module contains comprehensive unit and integration tests for all components of the Agentic Trader system. The test suite ensures code quality, reliability, and proper functionality across all modules.

## Test Structure

```
tests/
├── __init__.py
├── test_backtester.py      # Backtesting engine tests
├── test_csv_loader.py      # CSV data loading tests
├── test_data_loader.py     # Yahoo Finance data loading tests
├── test_project_structure.py # Project structure validation
└── test_strategies.py      # Trading strategy tests
```

## Running Tests

### Run All Tests
```bash
# Run all tests with verbose output
python -m pytest tests/ -v

# Run tests with coverage report
python -m pytest tests/ --cov=. --cov-report=term-missing --cov-report=html
```

### Run Specific Test Files
```bash
# Test specific module
python -m pytest tests/test_backtester.py -v
python -m pytest tests/test_strategies.py -v
python -m pytest tests/test_csv_loader.py -v
```

### Run Specific Test Functions
```bash
# Test specific functionality
python -m pytest tests/test_backtester.py::test_backtest_basic_functionality -v
python -m pytest tests/test_strategies.py::test_moving_average_signals -v
```

## Test Coverage

The test suite provides comprehensive coverage across all modules:

- **Data Module**: 95% coverage
  - CSV loading functionality
  - Data validation and cleaning
  - Error handling for invalid inputs
  
- **Strategies Module**: 95% coverage
  - Signal generation algorithms
  - Strategy configuration
  - Edge case handling
  
- **Simulation Module**: 95% coverage
  - Backtesting engine
  - Performance calculations
  - Transaction processing

## Test Categories

### Unit Tests
Test individual functions and methods in isolation:

```python
# Example from test_strategies.py
def test_moving_average_calculation():
    """Test moving average calculation accuracy"""
    strategy = MovingAverageCrossStrategy(short_window=2, long_window=3)
    data = create_test_data()
    signals = strategy.generate_signals(data)
    
    # Verify moving averages are calculated correctly
    assert abs(signals['Short_MA'].iloc[1] - expected_short_ma) < 0.01
    assert abs(signals['Long_MA'].iloc[2] - expected_long_ma) < 0.01
```

### Integration Tests
Test interaction between multiple components:

```python
# Example from test_backtester.py
def test_full_backtest_workflow():
    """Test complete workflow from data loading to performance reporting"""
    # Load data
    df = load_test_data()
    
    # Generate signals
    strategy = MovingAverageCrossStrategy()
    signals = strategy.generate_signals(df)
    
    # Run backtest
    backtester = Backtester(initial_capital=10000)
    report = backtester.run_backtest(df, signals)
    
    # Verify results
    assert report.summary()['initial_capital'] == 10000
    assert report.summary()['final_value'] > 0
```

### Edge Case Tests
Test handling of unusual or boundary conditions:

```python
def test_insufficient_data():
    """Test behavior with insufficient data for moving averages"""
    strategy = MovingAverageCrossStrategy(short_window=20, long_window=50)
    short_data = create_data_with_10_rows()
    
    signals = strategy.generate_signals(short_data)
    # Should handle gracefully without errors
    assert len(signals) == 10
```

## Test Data

### Sample Data Creation
Tests use standardized sample data for consistent results:

```python
def create_test_data(num_days=100, start_price=100):
    """Create sample OHLCV data for testing"""
    dates = pd.date_range('2020-01-01', periods=num_days, freq='D')
    np.random.seed(42)  # Reproducible results
    
    data = {
        'Date': dates,
        'Open': start_price + np.random.randn(num_days).cumsum(),
        'High': start_price + np.random.randn(num_days).cumsum() + 1,
        'Low': start_price + np.random.randn(num_days).cumsum() - 1,
        'Close': start_price + np.random.randn(num_days).cumsum(),
        'Volume': np.random.randint(1000000, 5000000, num_days)
    }
    
    return pd.DataFrame(data).set_index('Date')
```

### CSV Test Files
Tests use actual market data files for realistic testing:
- `data/GOOGL.csv` - Google stock data for backtesting tests
- `data/TSLA.csv` - Tesla data for strategy tests
- Other CSV files for various test scenarios

## Test Fixtures

Common test fixtures for repeated use:

```python
@pytest.fixture
def sample_strategy():
    """Provide a configured strategy for testing"""
    return MovingAverageCrossStrategy(short_window=20, long_window=50)

@pytest.fixture
def sample_backtester():
    """Provide a configured backtester for testing"""
    return Backtester(initial_capital=10000, commission=0.001)

@pytest.fixture
def sample_data():
    """Provide sample market data for testing"""
    return create_test_data(num_days=252)  # 1 year of data
```

## Performance Tests

Tests include performance benchmarks to ensure efficiency:

```python
def test_backtest_performance():
    """Test backtesting performance with large datasets"""
    large_data = create_test_data(num_days=5000)  # ~20 years
    strategy = MovingAverageCrossStrategy()
    
    start_time = time.time()
    signals = strategy.generate_signals(large_data)
    signal_time = time.time() - start_time
    
    backtester = Backtester(initial_capital=10000)
    start_time = time.time()
    report = backtester.run_backtest(large_data, signals)
    backtest_time = time.time() - start_time
    
    # Performance benchmarks
    assert signal_time < 1.0  # Signal generation under 1 second
    assert backtest_time < 2.0  # Backtesting under 2 seconds
```

## Error Handling Tests

Comprehensive testing of error conditions:

```python
def test_invalid_csv_file():
    """Test handling of invalid CSV files"""
    with pytest.raises(FileNotFoundError):
        load_csv_data("nonexistent_file.csv")

def test_insufficient_funds():
    """Test backtester behavior with insufficient funds"""
    backtester = Backtester(initial_capital=1)  # Very low capital
    # Should handle gracefully without crashing
    report = backtester.run_backtest(data, signals)
    assert report.summary()['num_trades'] == 0
```

## Mock and Patch Tests

Tests that require external data use mocking:

```python
@patch('yfinance.download')
def test_yahoo_finance_connection(mock_download):
    """Test Yahoo Finance data loading with mocked response"""
    mock_download.return_value = create_mock_yahoo_data()
    
    data = load_financial_data('AAPL', period='1y')
    assert len(data) > 0
    mock_download.assert_called_once()
```

## Continuous Integration

Tests are designed to run in CI/CD environments:

- **No External Dependencies**: Tests don't require internet access
- **Deterministic Results**: Fixed random seeds for reproducible results
- **Fast Execution**: Optimized for quick CI feedback
- **Comprehensive Coverage**: All critical paths tested

## Test Guidelines

### Writing New Tests

1. **Test Naming**: Use descriptive test function names
   ```python
   def test_moving_average_crossover_generates_buy_signal():
   ```

2. **Test Structure**: Follow Arrange-Act-Assert pattern
   ```python
   def test_example():
       # Arrange
       data = create_test_data()
       strategy = MovingAverageCrossStrategy()
       
       # Act
       signals = strategy.generate_signals(data)
       
       # Assert
       assert 'Signal' in signals.columns
   ```

3. **Edge Cases**: Always test boundary conditions
4. **Error Conditions**: Test invalid inputs and error handling
5. **Documentation**: Include docstrings for complex tests

### Test Data Management

- Use fixtures for commonly needed test data
- Keep test data small but representative
- Use deterministic data generation (fixed random seeds)
- Clean up any temporary files after tests

## Running Tests in Development

For development workflow:

```bash
# Quick test run during development
python -m pytest tests/ -x --ff  # Stop on first failure, run failures first

# Watch mode (requires pytest-watch)
ptw tests/  # Automatically re-run tests on file changes

# Test specific functionality you're working on
python -m pytest tests/test_backtester.py::TestBacktester::test_transaction_processing -v
```

## Coverage Reports

Generate detailed coverage reports:

```bash
# Terminal coverage report
python -m pytest tests/ --cov=. --cov-report=term-missing

# HTML coverage report (creates htmlcov/ directory)
python -m pytest tests/ --cov=. --cov-report=html

# XML coverage report (for CI tools)
python -m pytest tests/ --cov=. --cov-report=xml
```

## Contributing Tests

When contributing new features:

1. **Write tests first** (test-driven development)
2. **Maintain high coverage** (aim for >90%)
3. **Test both success and failure paths**
4. **Include integration tests** for new modules
5. **Update this README** if adding new test categories