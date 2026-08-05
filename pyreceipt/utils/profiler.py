"""Performance profiler decorator for resource-constrained environments (e.g., Raspberry Pi).

Measures function execution duration and peak RAM usage using time.perf_counter and tracemalloc.
"""

from functools import wraps
import logging
import time
import tracemalloc
from typing import Any, Callable

logger = logging.getLogger("pyreceipt.profiler")


def monitor_performance(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to monitor and log function execution time and peak memory consumption.

    Args:
        func: The target function to profile.

    Returns:
        The wrapped function returning original output while logging execution time (s)
        and peak RAM usage (MB).
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        was_tracing = tracemalloc.is_tracing()
        if not was_tracing:
            tracemalloc.start()

        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed_time = time.perf_counter() - start_time
            _, peak_bytes = tracemalloc.get_traced_memory()
            if not was_tracing:
                tracemalloc.stop()

            peak_mb = peak_bytes / (1024 * 1024)
            log_msg = (
                f"[PERFORMANCE] '{func.__name__}' executed in {elapsed_time:.4f}s "
                f"| Peak RAM: {peak_mb:.2f} MB"
            )
            logger.info(log_msg)

    return wrapper
