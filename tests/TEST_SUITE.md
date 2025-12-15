# Smart Recycling Detection System - Test Suite

## Overview

This comprehensive test suite provides thorough testing coverage for the Smart Recycling Detection System, including unit tests, integration tests, performance benchmarks, and various testing utilities.

## Test Structure

```
tests/
├── conftest.py                    # Shared fixtures and configuration
├── test_counter.py               # Enhanced counter tests
├── test_detector.py              # Enhanced detector tests  
├── test_integration.py           # Integration tests
├── test_runner.py                # Test execution script
├── pytest.ini                   # Pytest configuration
├── .coveragerc                   # Coverage configuration
├── requirements-test.txt         # Test dependencies
└── TEST_SUITE.md                # This documentation
```

## Test Categories

### 🧪 Unit Tests
- **Counter Tests**: Line crossing detection, object tracking, anti-double counting
- **Detector Tests**: Model loading, inference, performance monitoring
- **Edge Cases**: Invalid inputs, error conditions, boundary values
- **Performance**: Latency, throughput, memory usage

### 🔗 Integration Tests
- **End-to-End Workflows**: Complete detection and counting pipelines
- **Multi-Component**: Detector + Counter integration
- **Real-World Scenarios**: Conveyor belt simulation, recycling center workflow
- **Concurrent Processing**: Thread safety, parallel execution

### ⚡ Performance Tests
- **Benchmarking**: FPS measurement, latency analysis
- **Scalability**: High-volume processing, memory efficiency
- **Resource Usage**: CPU, memory, GPU utilization
- **Stress Testing**: Large datasets, extended operation

### 🐌 Slow Tests
- **Long-Running**: Extended operation scenarios
- **Memory-Intensive**: Large dataset processing
- **Comprehensive**: Full system validation

## Quick Start

### Install Test Dependencies
```bash
pip install -r requirements-test.txt
```

### Run Basic Tests
```bash
# All unit tests
python test_runner.py --unit

# All integration tests  
python test_runner.py --integration

# Quick test suite
python test_runner.py --unit --coverage
```

### Run Comprehensive Tests
```bash
# Full test suite
python test_runner.py --all

# Performance benchmarks
python test_runner.py --performance

# Memory-intensive tests
python test_runner.py --memory
```

## Test Runner Usage

The `test_runner.py` script provides a comprehensive interface for running tests:

### Basic Commands
```bash
# Run all tests
python test_runner.py --all

# Unit tests with coverage
python test_runner.py --unit --coverage

# Integration tests only
python test_runner.py --integration

# Performance tests
python test_runner.py --performance

# Specific test file
python test_runner.py --specific tests/test_counter.py

# Specific test method
python test_runner.py --specific tests/test_counter.py::TestCountingLine::test_vertical_line_creation
```

### Options
```bash
# Quiet output
python test_runner.py --unit --quiet

# Save results to file
python test_runner.py --all --save-results my_results.json

# Don't save results
python test_runner.py --unit --no-save
```

## Coverage Reports

Coverage reports are generated in multiple formats:

### HTML Report
```bash
python test_runner.py --coverage
# View: htmlcov/index.html
```

### Terminal Report
```bash
python test_runner.py --unit --coverage
# Shows missing lines in terminal
```

### XML Report
```bash
# Generated automatically: coverage.xml
# For CI/CD integration
```

## Test Markers

Tests are organized using pytest markers:

- `@pytest.mark.slow` - Long-running tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.performance` - Performance benchmarks
- `@pytest.mark.memory_intensive` - Memory-heavy tests
- `@pytest.mark.gpu` - GPU-required tests

### Running Specific Markers
```bash
# Run only fast tests
pytest -m "not slow"

# Run only integration tests
pytest -m "integration"

# Run performance tests
pytest -m "performance"

# Exclude memory-intensive tests
pytest -m "not memory_intensive"
```

## Fixtures and Utilities

### Common Fixtures
- `sample_image_*` - Test images of various sizes
- `sample_detections` - Mock detection objects
- `mock_yolo_model*` - YOLO model mocks
- `performance_timer` - Performance measurement
- `memory_monitor` - Memory usage tracking

### Test Utilities
- `TestUtils` - Helper functions for test data creation
- `DetectionValidator` - Validation utilities
- `ErrorSimulator` - Error condition simulation
- `BenchmarkRunner` - Performance benchmarking

## Test Data Management

### Fixtures
Shared test data is managed through pytest fixtures in `conftest.py`:

```python
# Use fixtures in tests
def test_detection(sample_detections, mock_yolo_model):
    # Test implementation
    pass
```

### Mock Objects
Comprehensive mocking for external dependencies:
- YOLO model mocking
- File system operations
- Time/date mocking
- Network requests

## Performance Testing

### Benchmarking
```python
def test_performance(benchmark_runner):
    result = benchmark_runner.run_benchmark(
        'detection_speed',
        detector.detect,
        iterations=100,
        image
    )
    assert result['avg_fps'] > 30
```

### Memory Monitoring
```python
def test_memory_usage(memory_monitor):
    # Run test
    memory_monitor.update()
    peak_usage = memory_monitor.get_peak_usage_mb()
    assert peak_usage < 500  # MB
```

## Error Testing

### Error Simulation
```python
def test_error_handling(error_simulator):
    # Simulate various error conditions
    with pytest.raises(ValueError):
        error_simulator.simulate_intermittent_failure()
```

### Edge Cases
- Invalid inputs (None, empty arrays, wrong types)
- Boundary values (min/max coordinates, extreme confidence)
- Resource constraints (out of memory, disk space)

## Continuous Integration

### GitHub Actions Example
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        pip install -r requirements-test.txt
    - name: Run tests
      run: |
        python test_runner.py --all --save-results ci_results.json
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

## Test Configuration

### pytest.ini
- Test discovery settings
- Custom markers
- Output formatting
- Coverage integration

### .coveragerc
- Source code inclusion/exclusion
- Branch coverage
- Report formatting
- Minimum coverage thresholds

## Best Practices

### Writing Tests
1. **Use descriptive names**: `test_crossing_detection_with_multiple_objects`
2. **Test one thing**: Each test should verify one specific behavior
3. **Use fixtures**: Reuse common test data and setup
4. **Mock external dependencies**: Keep tests isolated and fast
5. **Add performance tests**: Verify system performance requirements

### Test Organization
1. **Group related tests**: Use test classes for logical grouping
2. **Use appropriate markers**: Tag tests by category and requirements
3. **Document test purpose**: Clear docstrings explaining what is tested
4. **Parameterize tests**: Use `@pytest.mark.parametrize` for multiple inputs

### Performance Testing
1. **Set realistic thresholds**: Based on actual system requirements
2. **Account for environment variance**: CI environments may be slower
3. **Test both speed and accuracy**: Ensure optimizations don't hurt quality
4. **Monitor resource usage**: CPU, memory, GPU utilization

## Troubleshooting

### Common Issues

#### Tests Running Slowly
```bash
# Run only fast tests
pytest -m "not slow and not memory_intensive"

# Parallel execution (if pytest-xdist installed)
pytest -n auto
```

#### Memory Issues
```bash
# Run tests with limited memory usage
pytest -m "not memory_intensive"

# Monitor memory usage
pytest --memory-profiler
```

#### Coverage Issues
```bash
# Generate detailed coverage report
pytest --cov=src --cov-report=html --cov-report=term-missing

# Check specific files
pytest --cov=src.core.detector --cov-report=term-missing
```

### Debug Mode
```bash
# Run with debug output
pytest -s -vv --tb=long

# Drop into debugger on failure
pytest --pdb

# Debug specific test
pytest --pdb tests/test_counter.py::TestCountingLine::test_vertical_line_creation
```

## Contributing

### Adding New Tests
1. Follow existing naming conventions
2. Add appropriate markers
3. Use fixtures for common setup
4. Include both positive and negative test cases
5. Add performance tests for new features

### Test Documentation
1. Clear docstrings for test purpose
2. Comments for complex test logic
3. Examples of expected behavior
4. Update this documentation for major changes

## Reporting Issues

When reporting test failures:
1. Include full test output
2. Specify environment details (OS, Python version)
3. List installed package versions
4. Provide reproduction steps
5. Include any custom configuration

---

For more information, see the main project documentation or contact the development team.