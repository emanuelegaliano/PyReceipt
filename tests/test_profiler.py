"""Unit tests for the performance profiler decorator."""

import logging
import time
import pytest

from pyreceipt.utils.profiler import monitor_performance


def test_monitor_performance_preserves_metadata():
    """Verify that monitor_performance preserves function __name__ and __doc__."""

    @monitor_performance
    def sample_function():
        """Sample docstring."""
        return "result"

    assert sample_function.__name__ == "sample_function"
    assert sample_function.__doc__ == "Sample docstring."


def test_monitor_performance_returns_value():
    """Verify that wrapped function returns expected value."""

    @monitor_performance
    def add(a: int, b: int) -> int:
        return a + b

    assert add(3, 5) == 8


def test_monitor_performance_logs_execution(caplog):
    """Verify that performance profiler logs execution time and memory."""

    @monitor_performance
    def compute():
        # Allocate some temporary memory and simulate work
        data = [x for x in range(100000)]
        time.sleep(0.01)
        return len(data)

    with caplog.at_level(logging.INFO):
        res = compute()

    assert res == 100000
    assert "compute" in caplog.text
    assert "Peak RAM" in caplog.text
