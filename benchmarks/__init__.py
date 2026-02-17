"""
XMH Benchmarks Module

Standard benchmark functions for testing and validation.
"""

from .functions import (
    sphere,
    rosenbrock,
    rastrigin,
    schwefel,
    ackley,
    griewank,
    BenchmarkFunction,
    get_benchmark,
    list_benchmarks
)

__all__ = [
    "sphere",
    "rosenbrock",
    "rastrigin",
    "schwefel",
    "ackley",
    "griewank",
    "BenchmarkFunction",
    "get_benchmark",
    "list_benchmarks",
]
