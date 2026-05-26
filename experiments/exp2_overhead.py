#!/usr/bin/env python3
"""
Experiment 2: Computational Overhead Analysis

Measures the computational overhead of XMH instrumentation at different levels.

Research Question: What is the computational cost of instrumenting metaheuristics?
Hypothesis: Overhead < 20% for STANDARD instrumentation level.
"""

import sys
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import numpy as np

# Add parent of xmh to path for package imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from xmh.algorithms.instrumented_de import InstrumentedDE
from xmh.algorithms.instrumented_ga import InstrumentedGA
from xmh.algorithms.instrumented_pso import InstrumentedPSO
from xmh.benchmarks.functions import get_benchmark
from xmh.core.trace_logger import InstrumentationLevel


@dataclass
class OverheadResult:
    """Results from overhead measurement."""
    algorithm: str
    function: str
    dimension: int
    instrumentation_level: str

    # Timing
    base_time: float          # Time with NONE level
    instrumented_time: float  # Time with specified level
    overhead_percent: float

    # Memory (events logged)
    events_logged: int
    snapshots_stored: int

    # Quality check
    fitness_difference: float  # Should be ~0 (same seed)


def measure_single_overhead(
    algorithm_class,
    algorithm_name: str,
    func_name: str,
    dim: int,
    level: InstrumentationLevel,
    n_runs: int = 5,
    seed: int = 42
) -> OverheadResult:
    """Measure overhead for single configuration."""

    benchmark = get_benchmark(func_name)
    lower, upper = benchmark.get_bounds(dim)

    base_kwargs = {
        'dim': dim,
        'bounds': (lower, upper),
        'pop_size': 50,
        'max_generations': 100,
        'seed': seed
    }

    # Measure baseline (NONE level)
    base_times = []
    base_fitness = []

    for i in range(n_runs):
        kwargs = {**base_kwargs, 'instrumentation_level': InstrumentationLevel.NONE, 'seed': seed + i}
        alg = algorithm_class(**kwargs)

        start = time.perf_counter()
        result = alg.run(benchmark, verbose=False)
        elapsed = time.perf_counter() - start

        base_times.append(elapsed)
        base_fitness.append(result['best_fitness'])

    avg_base_time = np.mean(base_times)
    avg_base_fitness = np.mean(base_fitness)

    # Measure with instrumentation
    inst_times = []
    inst_fitness = []
    events_logged = 0
    snapshots_stored = 0

    for i in range(n_runs):
        kwargs = {**base_kwargs, 'instrumentation_level': level, 'seed': seed + i}
        alg = algorithm_class(**kwargs)

        start = time.perf_counter()
        result = alg.run(benchmark, verbose=False)
        elapsed = time.perf_counter() - start

        inst_times.append(elapsed)
        inst_fitness.append(result['best_fitness'])

        trace = result['trace']
        events_logged = len(trace.events)
        snapshots_stored = len(trace.snapshots)

    avg_inst_time = np.mean(inst_times)
    avg_inst_fitness = np.mean(inst_fitness)

    # Calculate overhead
    overhead = (avg_inst_time - avg_base_time) / avg_base_time * 100 if avg_base_time > 0 else 0

    return OverheadResult(
        algorithm=algorithm_name,
        function=func_name,
        dimension=dim,
        instrumentation_level=level.name,
        base_time=avg_base_time,
        instrumented_time=avg_inst_time,
        overhead_percent=overhead,
        events_logged=events_logged,
        snapshots_stored=snapshots_stored,
        fitness_difference=abs(avg_inst_fitness - avg_base_fitness)
    )


def run_overhead_analysis(
    algorithms: List[Tuple[Any, str]] = None,
    functions: List[str] = None,
    dimensions: List[int] = None,
    levels: List[InstrumentationLevel] = None,
    n_runs: int = 3,
    verbose: bool = True
) -> List[OverheadResult]:
    """
    Run complete overhead analysis experiment.

    Args:
        algorithms: List of (algorithm_class, name) tuples
        functions: List of benchmark function names
        dimensions: List of dimensions to test
        levels: List of instrumentation levels
        n_runs: Number of runs per configuration
        verbose: Print progress

    Returns:
        List of OverheadResult objects
    """
    # Defaults
    if algorithms is None:
        algorithms = [
            (InstrumentedDE, "DE"),
            (InstrumentedGA, "GA"),
            (InstrumentedPSO, "PSO"),
        ]

    if functions is None:
        functions = ["sphere", "rastrigin"]

    if dimensions is None:
        dimensions = [10, 30, 50]

    if levels is None:
        levels = [
            InstrumentationLevel.LIGHT,
            InstrumentationLevel.STANDARD,
            InstrumentationLevel.FULL,
        ]

    results = []
    total_configs = len(algorithms) * len(functions) * len(dimensions) * len(levels)
    current = 0

    if verbose:
        print("=" * 70)
        print("EXPERIMENT 2: COMPUTATIONAL OVERHEAD ANALYSIS")
        print("=" * 70)
        print(f"Algorithms: {[a[1] for a in algorithms]}")
        print(f"Functions: {functions}")
        print(f"Dimensions: {dimensions}")
        print(f"Levels: {[l.name for l in levels]}")
        print(f"Total configurations: {total_configs}")
        print()

    start_time = time.time()

    for alg_class, alg_name in algorithms:
        for func_name in functions:
            for dim in dimensions:
                for level in levels:
                    current += 1

                    if verbose:
                        print(f"[{current}/{total_configs}] {alg_name}/{func_name}/D{dim}/{level.name}...", end=" ")

                    try:
                        result = measure_single_overhead(
                            alg_class, alg_name, func_name, dim, level,
                            n_runs=n_runs
                        )
                        results.append(result)

                        if verbose:
                            print(f"overhead={result.overhead_percent:.1f}%")

                    except Exception as e:
                        if verbose:
                            print(f"ERROR: {e}")

    elapsed = time.time() - start_time

    if verbose:
        print()
        print("=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        print_overhead_summary(results)
        print(f"\nTotal experiment time: {elapsed:.1f}s")

    return results


def print_overhead_summary(results: List[OverheadResult]):
    """Print summary of overhead results."""

    # Group by level
    by_level = {}
    for r in results:
        if r.instrumentation_level not in by_level:
            by_level[r.instrumentation_level] = []
        by_level[r.instrumentation_level].append(r)

    print("\nOVERHEAD BY INSTRUMENTATION LEVEL:")
    print(f"\n{'Level':<12} {'Avg Overhead':>15} {'Min':>10} {'Max':>10} {'Events':>10}")
    print("-" * 60)

    for level, level_results in sorted(by_level.items()):
        overheads = [r.overhead_percent for r in level_results]
        events = [r.events_logged for r in level_results]

        print(f"{level:<12} {np.mean(overheads):>14.1f}% {np.min(overheads):>9.1f}% {np.max(overheads):>9.1f}% {int(np.mean(events)):>10}")

    # Group by algorithm
    by_algorithm = {}
    for r in results:
        if r.algorithm not in by_algorithm:
            by_algorithm[r.algorithm] = []
        by_algorithm[r.algorithm].append(r)

    print("\nOVERHEAD BY ALGORITHM (STANDARD level only):")
    print(f"\n{'Algorithm':<10} {'Avg Overhead':>15} {'Base Time':>12} {'Inst Time':>12}")
    print("-" * 55)

    for alg, alg_results in by_algorithm.items():
        std_results = [r for r in alg_results if r.instrumentation_level == "STANDARD"]
        if std_results:
            avg_overhead = np.mean([r.overhead_percent for r in std_results])
            avg_base = np.mean([r.base_time for r in std_results])
            avg_inst = np.mean([r.instrumented_time for r in std_results])
            print(f"{alg:<10} {avg_overhead:>14.1f}% {avg_base:>11.3f}s {avg_inst:>11.3f}s")

    # Group by dimension
    by_dim = {}
    for r in results:
        if r.dimension not in by_dim:
            by_dim[r.dimension] = []
        by_dim[r.dimension].append(r)

    print("\nOVERHEAD SCALING WITH DIMENSION (STANDARD level):")
    print(f"\n{'Dimension':<10} {'Avg Overhead':>15} {'Avg Base Time':>15}")
    print("-" * 45)

    for dim in sorted(by_dim.keys()):
        dim_results = [r for r in by_dim[dim] if r.instrumentation_level == "STANDARD"]
        if dim_results:
            avg_overhead = np.mean([r.overhead_percent for r in dim_results])
            avg_base = np.mean([r.base_time for r in dim_results])
            print(f"D={dim:<7} {avg_overhead:>14.1f}% {avg_base:>14.3f}s")

    # Detailed table
    print("\n\nDETAILED RESULTS:")
    print(f"{'Alg':<5} {'Func':<12} {'D':>4} {'Level':<10} {'Overhead':>10} {'Base(s)':>10} {'Inst(s)':>10} {'Events':>8}")
    print("-" * 80)

    for r in sorted(results, key=lambda x: (x.algorithm, x.function, x.dimension, x.instrumentation_level)):
        print(f"{r.algorithm:<5} {r.function:<12} {r.dimension:>4} {r.instrumentation_level:<10} {r.overhead_percent:>9.1f}% {r.base_time:>10.3f} {r.instrumented_time:>10.3f} {r.events_logged:>8}")

    # Verify fitness consistency
    print("\n\nFITNESS CONSISTENCY CHECK:")
    max_diff = max(r.fitness_difference for r in results)
    print(f"Max fitness difference (should be ~0): {max_diff:.6e}")
    if max_diff < 1e-6:
        print("✓ Instrumentation does not affect algorithm behavior")
    else:
        print("⚠ Some fitness differences detected (may be due to logging side effects)")


if __name__ == "__main__":
    results = run_overhead_analysis(
        functions=["sphere", "rastrigin"],
        dimensions=[10, 30],
        n_runs=3,
        verbose=True
    )
