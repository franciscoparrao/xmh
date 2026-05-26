#!/usr/bin/env python3
"""
Experiment 3: Cross-Algorithm Generalization Study

Tests whether operator attributions are consistent across different algorithms
and problem instances.

Research Question: Do operators exhibit consistent behavior patterns?
Hypothesis: Similar operators show similar attribution patterns across algorithms.
"""

import sys
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

# Add parent of xmh to path for package imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from xmh.algorithms.instrumented_de import InstrumentedDE
from xmh.algorithms.instrumented_ga import InstrumentedGA
from xmh.algorithms.instrumented_pso import InstrumentedPSO
from xmh.benchmarks.functions import get_benchmark
from xmh.explanation.operator_shap import QuickSHAP
from xmh.explanation.attribution import AttributionAnalyzer
from xmh.core.trace_logger import InstrumentationLevel


@dataclass
class OperatorProfile:
    """Profile of an operator's behavior across runs."""
    operator_name: str
    algorithm: str

    # Attribution statistics
    mean_attribution: float
    std_attribution: float

    # Phase-specific attributions
    early_phase_attr: float    # First 1/3 of generations
    middle_phase_attr: float   # Middle 1/3
    late_phase_attr: float     # Last 1/3

    # Consistency metrics
    consistency_score: float   # How stable across runs

    # Operator category (for cross-algorithm comparison)
    category: str  # 'exploration', 'exploitation', 'selection', 'adaptation'


@dataclass
class GeneralizationResult:
    """Results from generalization analysis."""
    function: str
    dimension: int

    # Per-algorithm profiles
    de_profiles: List[OperatorProfile] = field(default_factory=list)
    ga_profiles: List[OperatorProfile] = field(default_factory=list)
    pso_profiles: List[OperatorProfile] = field(default_factory=list)

    # Cross-algorithm metrics
    exploration_consistency: float = 0.0  # Do exploration ops behave similarly?
    exploitation_consistency: float = 0.0
    phase_pattern_similarity: float = 0.0  # Similar early/mid/late patterns?

    # Per-algorithm performance
    de_fitness: float = 0.0
    ga_fitness: float = 0.0
    pso_fitness: float = 0.0


# Operator category mappings - using actual operator names from instrumented algorithms
OPERATOR_CATEGORIES = {
    # DE operators (actual names: mutation_rand/1, crossover_binomial, selection_greedy)
    'mutation_rand/1': 'exploration',
    'mutation_best/1': 'exploitation',
    'mutation_current_to_best/1': 'exploitation',
    'mutation_rand/2': 'exploration',
    'mutation_best/2': 'exploitation',
    'crossover_binomial': 'recombination',
    'crossover_exponential': 'recombination',
    'selection_greedy': 'selection',

    # GA operators (actual names: selection_tournament, crossover_sbx, mutation_polynomial)
    'selection_tournament': 'selection',
    'selection_roulette': 'selection',
    'selection_rank': 'selection',
    'selection_sus': 'selection',
    'crossover_sbx': 'recombination',
    'crossover_one_point': 'recombination',
    'crossover_two_point': 'recombination',
    'crossover_uniform': 'recombination',
    'mutation_gaussian': 'exploration',
    'mutation_uniform': 'exploration',
    'mutation_polynomial': 'exploration',

    # PSO operators (actual names: velocity_constriction, position_update, topology_gbest)
    'velocity_standard': 'exploration',
    'velocity_constriction': 'exploration',
    'velocity_inertia_linear': 'adaptation',
    'velocity_inertia_nonlinear': 'adaptation',
    'position_update': 'exploitation',
    'topology_gbest': 'exploitation',
    'topology_lbest': 'exploitation',
    'topology_ring': 'exploitation',
}


def get_operator_category(op_name: str) -> str:
    """Get category for an operator."""
    return OPERATOR_CATEGORIES.get(op_name, 'unknown')


def analyze_phase_attributions(
    trace,
    quick_shap: QuickSHAP
) -> Dict[str, Tuple[float, float, float]]:
    """
    Analyze attributions by search phase.

    Returns:
        Dict mapping operator names to (early, middle, late) attributions
    """
    n_generations = len(trace.snapshots)
    if n_generations < 3:
        return {}

    third = n_generations // 3

    # Split trace by phase
    phases_def = {
        'early': (0, third),
        'middle': (third, 2*third),
        'late': (2*third, n_generations)
    }

    phase_attrs = {}

    for phase_name, (start, end) in phases_def.items():
        # Get events in this phase
        phase_events = [
            e for e in trace.events
            if start <= e.generation < end
        ]

        if phase_events:
            # Calculate attributions for this phase
            phase_contributions = {}
            for event in phase_events:
                # Operator name is in event.data
                op = event.data.get('operator_name', event.data.get('operator', 'unknown'))
                if op == 'unknown':
                    continue
                if op not in phase_contributions:
                    phase_contributions[op] = []

                # Use fitness improvement as contribution
                if event.data and 'fitness_delta' in event.data:
                    phase_contributions[op].append(event.data['fitness_delta'])

            for op, contribs in phase_contributions.items():
                if op not in phase_attrs:
                    phase_attrs[op] = {'early': 0, 'middle': 0, 'late': 0}
                phase_attrs[op][phase_name] = np.mean(contribs) if contribs else 0

    # Convert to tuple format
    result = {}
    for op, phases_data in phase_attrs.items():
        result[op] = (phases_data['early'], phases_data['middle'], phases_data['late'])

    return result


def build_operator_profiles(
    algorithm_class,
    algorithm_name: str,
    func_name: str,
    dim: int,
    n_runs: int = 5,
    seed: int = 42
) -> Tuple[List[OperatorProfile], float]:
    """
    Build operator profiles from multiple runs.

    Returns:
        Tuple of (profiles, average_fitness)
    """
    benchmark = get_benchmark(func_name)
    lower, upper = benchmark.get_bounds(dim)

    kwargs = {
        'dim': dim,
        'bounds': (lower, upper),
        'pop_size': 50,
        'max_generations': 100,
        'instrumentation_level': InstrumentationLevel.STANDARD,
    }

    # Collect attributions across runs
    all_attributions = {}  # op -> list of attribution values
    all_phase_attrs = {}   # op -> list of (early, mid, late) tuples
    fitness_results = []

    quick_shap = QuickSHAP()

    for i in range(n_runs):
        run_kwargs = {**kwargs, 'seed': seed + i}
        alg = algorithm_class(**run_kwargs)
        result = alg.run(benchmark, verbose=False)

        trace = result['trace']
        fitness_results.append(result['best_fitness'])

        # Get SHAP values
        explanation = quick_shap.explain_from_trace(trace)

        for op, value in explanation.shap_values.items():
            if op not in all_attributions:
                all_attributions[op] = []
            all_attributions[op].append(value)

        # Get phase attributions
        phase_attrs = analyze_phase_attributions(trace, quick_shap)
        for op, phases in phase_attrs.items():
            if op not in all_phase_attrs:
                all_phase_attrs[op] = []
            all_phase_attrs[op].append(phases)

    # Build profiles
    profiles = []

    for op in all_attributions.keys():
        attrs = all_attributions[op]

        # Phase attributions
        if op in all_phase_attrs and all_phase_attrs[op]:
            phase_data = all_phase_attrs[op]
            early = np.mean([p[0] for p in phase_data])
            middle = np.mean([p[1] for p in phase_data])
            late = np.mean([p[2] for p in phase_data])
        else:
            early = middle = late = 0.0

        # Consistency score (inverse of coefficient of variation)
        mean_attr = np.mean(attrs)
        std_attr = np.std(attrs)
        consistency = 1.0 / (1.0 + std_attr / (abs(mean_attr) + 1e-10))

        profile = OperatorProfile(
            operator_name=op,
            algorithm=algorithm_name,
            mean_attribution=mean_attr,
            std_attribution=std_attr,
            early_phase_attr=early,
            middle_phase_attr=middle,
            late_phase_attr=late,
            consistency_score=consistency,
            category=get_operator_category(op)
        )
        profiles.append(profile)

    avg_fitness = np.mean(fitness_results)
    return profiles, avg_fitness


def calculate_category_consistency(
    profiles_by_alg: Dict[str, List[OperatorProfile]],
    category: str
) -> float:
    """
    Calculate consistency of a category across algorithms.

    Measures whether operators in the same category have similar
    relative importance across different algorithms.
    """
    category_attrs = {}

    for alg, profiles in profiles_by_alg.items():
        cat_profiles = [p for p in profiles if p.category == category]
        if cat_profiles:
            # Normalize attributions within algorithm
            total = sum(abs(p.mean_attribution) for p in cat_profiles)
            if total > 0:
                category_attrs[alg] = sum(p.mean_attribution for p in cat_profiles) / total
            else:
                category_attrs[alg] = 0

    if len(category_attrs) < 2:
        return 1.0  # Can't measure consistency with < 2 algorithms

    values = list(category_attrs.values())
    # Consistency = 1 - coefficient of variation
    mean_val = np.mean(values)
    std_val = np.std(values)

    if abs(mean_val) < 1e-10:
        return 1.0

    cv = std_val / abs(mean_val)
    return max(0, 1 - cv)


def calculate_phase_pattern_similarity(
    profiles_by_alg: Dict[str, List[OperatorProfile]]
) -> float:
    """
    Calculate similarity of phase patterns across algorithms.

    Checks if operators show similar early/middle/late progression.
    """
    phase_patterns = []

    for alg, profiles in profiles_by_alg.items():
        if not profiles:
            continue

        # Aggregate phase pattern for this algorithm
        early_total = sum(p.early_phase_attr for p in profiles)
        middle_total = sum(p.middle_phase_attr for p in profiles)
        late_total = sum(p.late_phase_attr for p in profiles)

        total = abs(early_total) + abs(middle_total) + abs(late_total)
        if total > 0:
            pattern = (early_total/total, middle_total/total, late_total/total)
            phase_patterns.append(pattern)

    if len(phase_patterns) < 2:
        return 1.0

    # Calculate cosine similarity between patterns
    similarities = []
    for i in range(len(phase_patterns)):
        for j in range(i+1, len(phase_patterns)):
            p1 = np.array(phase_patterns[i])
            p2 = np.array(phase_patterns[j])

            norm1 = np.linalg.norm(p1)
            norm2 = np.linalg.norm(p2)

            if norm1 > 0 and norm2 > 0:
                sim = np.dot(p1, p2) / (norm1 * norm2)
                similarities.append(sim)

    return np.mean(similarities) if similarities else 1.0


def run_generalization_study(
    functions: List[str] = None,
    dimensions: List[int] = None,
    n_runs: int = 5,
    verbose: bool = True
) -> List[GeneralizationResult]:
    """
    Run complete generalization study.

    Args:
        functions: List of benchmark function names
        dimensions: List of dimensions to test
        n_runs: Number of runs per configuration
        verbose: Print progress

    Returns:
        List of GeneralizationResult objects
    """
    if functions is None:
        functions = ["sphere", "rastrigin", "rosenbrock"]

    if dimensions is None:
        dimensions = [10, 30]

    algorithms = [
        (InstrumentedDE, "DE"),
        (InstrumentedGA, "GA"),
        (InstrumentedPSO, "PSO"),
    ]

    results = []
    total_configs = len(functions) * len(dimensions)
    current = 0

    if verbose:
        print("=" * 70)
        print("EXPERIMENT 3: CROSS-ALGORITHM GENERALIZATION STUDY")
        print("=" * 70)
        print(f"Functions: {functions}")
        print(f"Dimensions: {dimensions}")
        print(f"Algorithms: {[a[1] for a in algorithms]}")
        print(f"Runs per config: {n_runs}")
        print(f"Total configurations: {total_configs}")
        print()

    start_time = time.time()

    for func_name in functions:
        for dim in dimensions:
            current += 1

            if verbose:
                print(f"\n[{current}/{total_configs}] {func_name} D={dim}")
                print("-" * 40)

            result = GeneralizationResult(
                function=func_name,
                dimension=dim
            )

            profiles_by_alg = {}

            # Build profiles for each algorithm
            for alg_class, alg_name in algorithms:
                if verbose:
                    print(f"  Building {alg_name} profiles...", end=" ")

                try:
                    profiles, avg_fitness = build_operator_profiles(
                        alg_class, alg_name, func_name, dim, n_runs
                    )

                    profiles_by_alg[alg_name] = profiles

                    if alg_name == "DE":
                        result.de_profiles = profiles
                        result.de_fitness = avg_fitness
                    elif alg_name == "GA":
                        result.ga_profiles = profiles
                        result.ga_fitness = avg_fitness
                    elif alg_name == "PSO":
                        result.pso_profiles = profiles
                        result.pso_fitness = avg_fitness

                    if verbose:
                        print(f"{len(profiles)} operators, fitness={avg_fitness:.6e}")

                except Exception as e:
                    if verbose:
                        print(f"ERROR: {e}")

            # Calculate cross-algorithm metrics
            if verbose:
                print("  Computing cross-algorithm metrics...")

            result.exploration_consistency = calculate_category_consistency(
                profiles_by_alg, 'exploration'
            )
            result.exploitation_consistency = calculate_category_consistency(
                profiles_by_alg, 'exploitation'
            )
            result.phase_pattern_similarity = calculate_phase_pattern_similarity(
                profiles_by_alg
            )

            if verbose:
                print(f"  Exploration consistency: {result.exploration_consistency:.3f}")
                print(f"  Exploitation consistency: {result.exploitation_consistency:.3f}")
                print(f"  Phase pattern similarity: {result.phase_pattern_similarity:.3f}")

            results.append(result)

    elapsed = time.time() - start_time

    if verbose:
        print()
        print("=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        print_generalization_summary(results)
        print(f"\nTotal experiment time: {elapsed:.1f}s")

    return results


def print_generalization_summary(results: List[GeneralizationResult]):
    """Print summary of generalization results."""

    print("\nCROSS-ALGORITHM CONSISTENCY METRICS:")
    print(f"\n{'Function':<12} {'Dim':>4} {'Explor. Cons.':>14} {'Exploit. Cons.':>15} {'Phase Sim.':>12}")
    print("-" * 65)

    for r in results:
        print(f"{r.function:<12} {r.dimension:>4} {r.exploration_consistency:>14.3f} "
              f"{r.exploitation_consistency:>15.3f} {r.phase_pattern_similarity:>12.3f}")

    # Averages
    avg_explor = np.mean([r.exploration_consistency for r in results])
    avg_exploit = np.mean([r.exploitation_consistency for r in results])
    avg_phase = np.mean([r.phase_pattern_similarity for r in results])

    print("-" * 65)
    print(f"{'AVERAGE':<12} {'':<4} {avg_explor:>14.3f} {avg_exploit:>15.3f} {avg_phase:>12.3f}")

    # Performance comparison
    print("\n\nALGORITHM PERFORMANCE COMPARISON:")
    print(f"\n{'Function':<12} {'Dim':>4} {'DE':>15} {'GA':>15} {'PSO':>15} {'Best':>8}")
    print("-" * 75)

    for r in results:
        best = "DE" if r.de_fitness <= min(r.ga_fitness, r.pso_fitness) else \
               "GA" if r.ga_fitness <= r.pso_fitness else "PSO"
        print(f"{r.function:<12} {r.dimension:>4} {r.de_fitness:>15.6e} "
              f"{r.ga_fitness:>15.6e} {r.pso_fitness:>15.6e} {best:>8}")

    # Operator profiles detail
    print("\n\nOPERATOR CATEGORY ANALYSIS (aggregated):")

    category_stats = {}

    for r in results:
        all_profiles = r.de_profiles + r.ga_profiles + r.pso_profiles

        for p in all_profiles:
            if p.category not in category_stats:
                category_stats[p.category] = {
                    'attributions': [],
                    'consistencies': [],
                    'count': 0
                }
            category_stats[p.category]['attributions'].append(p.mean_attribution)
            category_stats[p.category]['consistencies'].append(p.consistency_score)
            category_stats[p.category]['count'] += 1

    print(f"\n{'Category':<15} {'Count':>6} {'Avg Attribution':>18} {'Avg Consistency':>18}")
    print("-" * 60)

    for cat, stats in sorted(category_stats.items()):
        avg_attr = np.mean(stats['attributions'])
        avg_cons = np.mean(stats['consistencies'])
        print(f"{cat:<15} {stats['count']:>6} {avg_attr:>18.6e} {avg_cons:>18.3f}")

    # Phase analysis
    print("\n\nPHASE CONTRIBUTION ANALYSIS (aggregated):")

    phase_totals = {'early': [], 'middle': [], 'late': []}

    for r in results:
        all_profiles = r.de_profiles + r.ga_profiles + r.pso_profiles

        for p in all_profiles:
            phase_totals['early'].append(p.early_phase_attr)
            phase_totals['middle'].append(p.middle_phase_attr)
            phase_totals['late'].append(p.late_phase_attr)

    print(f"\n{'Phase':<10} {'Avg Contribution':>20} {'Std Dev':>15}")
    print("-" * 50)

    for phase, values in phase_totals.items():
        print(f"{phase.capitalize():<10} {np.mean(values):>20.6e} {np.std(values):>15.6e}")


if __name__ == "__main__":
    results = run_generalization_study(
        functions=["sphere", "rastrigin"],
        dimensions=[10],
        n_runs=3,
        verbose=True
    )
