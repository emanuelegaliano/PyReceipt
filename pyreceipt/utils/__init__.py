"""Utilities and Telemetry Tools for PyReceipt.

Provides hardware monitoring, memory tracing, and execution timing decorators
designed for resource-constrained edge computing environments.
"""

from pyreceipt.utils.profiler import monitor_performance

__all__ = ["monitor_performance"]

