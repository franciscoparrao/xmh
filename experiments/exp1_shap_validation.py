#!/usr/bin/env python3
"""
Experiment 1: Operator-SHAP Validation

Validates that Operator-SHAP attributions correlate with true operator
contributions measured via ablation study.

Research Question: How accurate are Operator-SHAP attributions?
Hypothesis: R² > 0.80 correlation between SHAP and ablation values.
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
from xmh.explanation.operator_shap import QuickSHAP
from xmh.core.trace_logger import InstrumentationLevel


@dataclass
class ValidationResult:
    """Results from one validation run."""
    algorithm: str
    function: str
    dimension: int

    # Full algorithm result
    full_fitness: float

    # Ablation results (fitness without each operator)
    ablation_fitness: Dict[str, float]
    ablation_contributions: Dict[str, float]

    # SHAP results
    shap_values: Dict[str, float]

    # Correlation metrics
    pearson_r: float
    spearman_rho: float
    ranking_agreement: float


def run_ablation_study(
    algorithm_class,
    algorithm_kwargs: Dict,
    objective_func,
    n_runs: int = 5
) -> Tuple[float, Dict[str, float]]:
    """
    Run ablation study to measure true operator contributions.

    Returns:
        Tuple of (full_fitness, {operator: ablated_fitness})
    """
    # Run full algorithm
    full_results = []
    for _ in range(n_runs):
        alg = algorithm_class(**algorithm_kwargs)
        result = alg.run(objective_func, verbose=False)
        full_results.append(result['best_fitness'])

    full_fitness = np.mean(full_results)

    # Get operator names (run briefly to get initialized operators)
    alg = algorithm_class(**algorithm_kwargs)
    alg._initialize_operators()
    operators = alg.get_operators()

    # Run with each operator disabled
    ablation_fitness = {}

    for op_name in operators:
        ablated_results = []
        for _ in range(n_runs):
            alg = algorithm_class(**algorithm_kwargs)
            # Set the operator to disable - will be applied after _initialize_operators in run()
            alg._operator_to_disable = op_name
            # Monkey-patch to disable after initialization
            original_init = alg._initialize_operators
            def patched_init(self=alg, orig=original_init, op=op_name):
                orig()
                self.disable_operator(op)
            alg._initialize_operators = patched_init
            result = alg.run(objective_func, verbose=False)
            ablated_results.append(result['best_fitness'])

        ablation_fitness[op_name] = np.mean(ablated_results)

    return full_fitness, ablation_fitness


def calculate_ablation_contributions(
    full_fitness: float,
    ablation_fitness: Dict[str, float]
) -> Dict[str, float]:
    """
    Calculate operator contributions from ablation study.

    Contribution = fitness_without_operator - full_fitness
    (Higher contribution = more important for minimization)
    """
    contributions = {}
    for op, ablated in ablation_fitness.items():
        # Positive means operator helped (ablated is worse)
        contributions[op] = ablated - full_fitness
    return contributions


def calculate_correlations(
    ablation: Dict[str, float],
    shap: Dict[str, float]
) -> Tuple[float, float, float]:
    """
    Calculate correlation metrics between ablation and SHAP values.

    Returns:
        (pearson_r, spearman_rho, ranking_agreement)
    """
    # Get common operators
    common_ops = set(ablation.keys()) & set(shap.keys())
    if len(common_ops) < 2:
        return 0.0, 0.0, 0.0

    ops = sorted(common_ops)
    ablation_vals = np.array([ablation[op] for op in ops])
    shap_vals = np.array([shap[op] for op in ops])

    # Pearson correlation
    if np.std(ablation_vals) > 0 and np.std(shap_vals) > 0:
        pearson_r = np.corrcoef(ablation_vals, shap_vals)[0, 1]
    else:
        pearson_r = 0.0

    # Spearman correlation (rank-based)
    ablation_ranks = np.argsort(np.argsort(ablation_vals))
    shap_ranks = np.argsort(np.argsort(shap_vals))

    n = len(ops)
    d_squared = np.sum((ablation_ranks - shap_ranks) ** 2)
    spearman_rho = 1 - (6 * d_squared) / (n * (n**2 - 1)) if n > 1 else 0.0

    # Ranking agreement (top operator match)
    ablation_top = ops[np.argmax(ablation_vals)]
    shap_top = ops[np.argmax(shap_vals)]
    ranking_agreement = 1.0 if ablation_top == shap_top else 0.0

    return float(pearson_r), float(spearman_rho), ranking_agreement


def validate_single_configuration(
    algorithm_class,
    algorithm_name: str,
    func_name: str,
    dim: int,
    n_ablation_runs: int = 5,
    seed: int = 42
) -> ValidationResult:
    """Run validation for a single algorithm-function-dimension configuration."""

    benchmark = get_benchmark(func_name)
    lower, upper = benchmark.get_bounds(dim)

    # Common kwargs
    kwargs = {
        'dim': dim,
        'bounds': (lower, upper),
        'pop_size': 30,
        'max_generations': 50,
        'instrumentation_level': InstrumentationLevel.STANDARD,
        'seed': seed
    }

    # Run ablation study
    full_fitness, ablation_fitness = run_ablation_study(
        algorithm_class, kwargs, benchmark, n_ablation_runs
    )

    # Calculate ablation contributions
    ablation_contributions = calculate_ablation_contributions(
        full_fitness, ablation_fitness
    )

    # Run full algorithm and get QuickSHAP
    alg = algorithm_class(**kwargs)
    result = alg.run(benchmark, verbose=False)
    trace = result['trace']

    quick_shap = QuickSHAP()
    explanation = quick_shap.explain_from_trace(trace)

    # Calculate correlations
    pearson_r, spearman_rho, ranking_agreement = calculate_correlations(
        ablation_contributions, explanation.shap_values
    )

    return ValidationResult(
        algorithm=algorithm_name,
        function=func_name,
        dimension=dim,
        full_fitness=full_fitness,
        ablation_fitness=ablation_fitness,
        ablation_contributions=ablation_contributions,
        shap_values=explanation.shap_values,
        pearson_r=pearson_r,
        spearman_rho=spearman_rho,
        ranking_agreement=ranking_agreement
    )


def run_shap_validation(
    algorithms: List[Tuple[Any, str]] = None,
    functions: List[str] = None,
    dimensions: List[int] = None,
    n_runs: int = 3,
    verbose: bool = True
) -> List[ValidationResult]:
    """
    Run complete SHAP validation experiment.

    Args:
        algorithms: List of (algorithm_class, name) tuples
        functions: List of benchmark function names
        dimensions: List of dimensions to test
        n_runs: Number of ablation runs per configuration
        verbose: Print progress

    Returns:
        List of ValidationResult objects
    """
    # Defaults
    if algorithms is None:
        algorithms = [
            (InstrumentedDE, "DE"),
            (InstrumentedGA, "GA"),
            (InstrumentedPSO, "PSO"),
        ]

    if functions is None:
        functions = ["sphere", "rastrigin", "rosenbrock"]

    if dimensions is None:
        dimensions = [10, 20]

    results = []
    total_configs = len(algorithms) * len(functions) * len(dimensions)
    current = 0

    if verbose:
        print("=" * 70)
        print("EXPERIMENT 1: OPERATOR-SHAP VALIDATION")
        print("=" * 70)
        print(f"Algorithms: {[a[1] for a in algorithms]}")
        print(f"Functions: {functions}")
        print(f"Dimensions: {dimensions}")
        print(f"Total configurations: {total_configs}")
        print()

    start_time = time.time()

    for alg_class, alg_name in algorithms:
        for func_name in functions:
            for dim in dimensions:
                current += 1

                if verbose:
                    print(f"[{current}/{total_configs}] {alg_name} on {func_name} (D={dim})...", end=" ")

                try:
                    result = validate_single_configuration(
                        alg_class, alg_name, func_name, dim,
                        n_ablation_runs=n_runs
                    )
                    results.append(result)

                    if verbose:
                        print(f"r={result.pearson_r:.3f}, ρ={result.spearman_rho:.3f}")

                except Exception as e:
                    if verbose:
                        print(f"ERROR: {e}")

    elapsed = time.time() - start_time

    if verbose:
        print()
        print("=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        print_validation_summary(results)
        print(f"\nTotal time: {elapsed:.1f}s")

    return results


def print_validation_summary(results: List[ValidationResult]):
    """Print summary of validation results."""

    # Group by algorithm
    by_algorithm = {}
    for r in results:
        if r.algorithm not in by_algorithm:
            by_algorithm[r.algorithm] = []
        by_algorithm[r.algorithm].append(r)

    print(f"\n{'Algorithm':<10} {'Avg Pearson r':>15} {'Avg Spearman ρ':>15} {'Rank Agreement':>15}")
    print("-" * 60)

    for alg, alg_results in by_algorithm.items():
        avg_pearson = np.mean([r.pearson_r for r in alg_results if not np.isnan(r.pearson_r)])
        avg_spearman = np.mean([r.spearman_rho for r in alg_results if not np.isnan(r.spearman_rho)])
        avg_ranking = np.mean([r.ranking_agreement for r in alg_results])

        print(f"{alg:<10} {avg_pearson:>15.3f} {avg_spearman:>15.3f} {avg_ranking:>15.1%}")

    # Overall
    all_pearson = [r.pearson_r for r in results if not np.isnan(r.pearson_r)]
    all_spearman = [r.spearman_rho for r in results if not np.isnan(r.spearman_rho)]
    all_ranking = [r.ranking_agreement for r in results]

    print("-" * 60)
    print(f"{'OVERALL':<10} {np.mean(all_pearson):>15.3f} {np.mean(all_spearman):>15.3f} {np.mean(all_ranking):>15.1%}")

    # Detailed breakdown
    print("\n\nDETAILED RESULTS:")
    print(f"{'Algorithm':<8} {'Function':<12} {'Dim':>4} {'Pearson':>10} {'Spearman':>10} {'RankAgr':>8}")
    print("-" * 60)

    for r in results:
        print(f"{r.algorithm:<8} {r.function:<12} {r.dimension:>4} {r.pearson_r:>10.3f} {r.spearman_rho:>10.3f} {r.ranking_agreement:>8.0%}")

    # Ablation vs SHAP comparison for one example
    print("\n\nEXAMPLE: Detailed comparison (first result)")
    r = results[0]
    print(f"\nAlgorithm: {r.algorithm}, Function: {r.function}, Dim: {r.dimension}")
    print(f"Full algorithm fitness: {r.full_fitness:.6e}")
    print(f"\n{'Operator':<30} {'Ablation Contrib':>18} {'SHAP Value':>18}")
    print("-" * 70)

    for op in r.ablation_contributions.keys():
        abl = r.ablation_contributions.get(op, 0)
        shp = r.shap_values.get(op, 0)
        print(f"{op:<30} {abl:>18.6e} {shp:>18.6e}")


if __name__ == "__main__":
    results = run_shap_validation(
        functions=["sphere", "rastrigin"],
        dimensions=[10],
        n_runs=3,
        verbose=True
    )
