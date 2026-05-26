#!/usr/bin/env python3
"""
Full Experimental Study for IEEE TEVC Paper

Comprehensive validation of XMH framework with:
- 9 benchmark functions (unimodal + multimodal)
- Multiple dimensions: D = {10, 30, 50}
- 30 independent runs per configuration
- Statistical analysis: Wilcoxon signed-rank, Friedman test
- SHAP validation against ablation
- Checkpoint/Resume support for long experiments

Usage:
    python exp_full_study.py                    # Full study (takes hours)
    python exp_full_study.py --quick            # Quick test (5 runs, D=10 only)
    python exp_full_study.py --resume           # Resume from checkpoint
    python exp_full_study.py --output myresults # Custom output directory
"""

import sys
import os
import argparse
import time
import json
import pickle
import shutil
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any, Set
from datetime import datetime
import numpy as np
from scipy import stats
import warnings

# Force unbuffered output for real-time logging
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from xmh.algorithms.instrumented_de import InstrumentedDE
from xmh.algorithms.instrumented_ga import InstrumentedGA
from xmh.algorithms.instrumented_pso import InstrumentedPSO
from xmh.benchmarks.functions import get_benchmark, list_benchmarks
from xmh.explanation.operator_shap import QuickSHAP
from xmh.core.trace_logger import InstrumentationLevel


# ==================== CONFIGURATION ====================

@dataclass
class ExperimentConfig:
    """Configuration for the full study."""
    # Benchmark functions to test
    functions: List[str] = field(default_factory=lambda: [
        "sphere", "rosenbrock", "rastrigin", "schwefel",
        "ackley", "griewank", "levy", "zakharov", "dixon_price"
    ])

    # Dimensions to test
    dimensions: List[int] = field(default_factory=lambda: [10, 30, 50])

    # Number of independent runs
    n_runs: int = 30

    # Algorithm parameters
    pop_size: int = 50
    max_generations: int = 500
    max_fes: int = 50000  # Max function evaluations (alternative stopping)

    # Ablation parameters
    ablation_runs: int = 5  # Runs per operator ablation

    # Output settings
    output_dir: str = "results"
    save_traces: bool = False  # Save full traces (large files)

    # Seed for reproducibility
    base_seed: int = 42


@dataclass
class RunResult:
    """Result from a single algorithm run."""
    algorithm: str
    function: str
    dimension: int
    run_id: int
    seed: int
    best_fitness: float
    convergence_gen: int  # Generation where 99% of improvement achieved
    final_population_diversity: float
    execution_time: float
    shap_values: Dict[str, float]
    operator_stats: Dict[str, Dict[str, float]]


@dataclass
class AblationResult:
    """Result from ablation study."""
    algorithm: str
    function: str
    dimension: int
    operator: str
    mean_fitness: float
    std_fitness: float
    contribution: float  # difference from full algorithm


@dataclass
class StatisticalResult:
    """Statistical comparison results."""
    function: str
    dimension: int

    # Pairwise Wilcoxon results (algorithm pairs)
    wilcoxon_pvalues: Dict[str, float]
    wilcoxon_significant: Dict[str, bool]

    # Friedman test (all algorithms)
    friedman_statistic: float
    friedman_pvalue: float

    # Rankings
    mean_ranks: Dict[str, float]

    # Best algorithm
    best_algorithm: str


@dataclass
class ExperimentCheckpoint:
    """Checkpoint for resuming experiments."""
    config: ExperimentConfig
    completed_configs: Set[Tuple[str, int, str]]  # (function, dim, algorithm)
    run_results: List[RunResult]
    ablation_results: List[AblationResult]
    start_time: float
    last_update: float

    def get_progress_string(self) -> str:
        """Get human-readable progress."""
        total = len(self.config.functions) * len(self.config.dimensions) * 3  # 3 algorithms
        completed = len(self.completed_configs)
        pct = 100 * completed / total if total > 0 else 0
        return f"{completed}/{total} ({pct:.1f}%)"


# ==================== CHECKPOINT FUNCTIONS ====================

CHECKPOINT_FILE = "checkpoint.pkl"


def save_checkpoint(checkpoint: ExperimentCheckpoint, output_dir: str):
    """Save checkpoint atomically."""
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, CHECKPOINT_FILE)
    temp_path = checkpoint_path + ".tmp"

    checkpoint.last_update = time.time()

    # Write to temp file first
    with open(temp_path, 'wb') as f:
        pickle.dump(checkpoint, f)

    # Atomic rename
    shutil.move(temp_path, checkpoint_path)


def load_checkpoint(output_dir: str) -> Optional[ExperimentCheckpoint]:
    """Load checkpoint if exists."""
    checkpoint_path = os.path.join(output_dir, CHECKPOINT_FILE)

    if not os.path.exists(checkpoint_path):
        return None

    try:
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        return checkpoint
    except Exception as e:
        print(f"Warning: Could not load checkpoint: {e}")
        return None


def is_config_completed(checkpoint: ExperimentCheckpoint, func: str, dim: int, alg: str) -> bool:
    """Check if a configuration has been completed."""
    return (func, dim, alg) in checkpoint.completed_configs


def mark_config_completed(checkpoint: ExperimentCheckpoint, func: str, dim: int, alg: str):
    """Mark a configuration as completed."""
    checkpoint.completed_configs.add((func, dim, alg))


# ==================== ALGORITHMS ====================

ALGORITHMS = {
    "DE": InstrumentedDE,
    "GA": InstrumentedGA,
    "PSO": InstrumentedPSO,
}


def get_algorithm_kwargs(alg_name: str, dim: int, bounds: Tuple,
                         config: ExperimentConfig, seed: int) -> Dict:
    """Get algorithm-specific kwargs."""
    base_kwargs = {
        'dim': dim,
        'bounds': bounds,
        'pop_size': config.pop_size,
        'max_generations': config.max_generations,
        'instrumentation_level': InstrumentationLevel.STANDARD,
        'seed': seed,
    }

    # Algorithm-specific adjustments
    if alg_name == "DE":
        base_kwargs['F'] = 0.5
        base_kwargs['CR'] = 0.9
    elif alg_name == "GA":
        base_kwargs['mutation_prob'] = 1.0 / dim  # Standard: 1/dim
        base_kwargs['crossover_prob'] = 0.9
    elif alg_name == "PSO":
        pass  # Use defaults (constriction coefficient handles w, c1, c2)

    return base_kwargs


# ==================== METRICS ====================

def calculate_convergence_generation(fitness_history: List[float],
                                     threshold: float = 0.99) -> int:
    """
    Find generation where threshold% of total improvement was achieved.
    """
    if not fitness_history or len(fitness_history) < 2:
        return 0

    initial = fitness_history[0]
    final = fitness_history[-1]

    if initial == final:
        return 0

    target_improvement = initial - (initial - final) * threshold

    for gen, fitness in enumerate(fitness_history):
        if fitness <= target_improvement:
            return gen

    return len(fitness_history) - 1


def calculate_population_diversity(population: np.ndarray) -> float:
    """Calculate population diversity as average pairwise distance."""
    if population is None or len(population) < 2:
        return 0.0

    n = len(population)
    total_dist = 0.0
    count = 0

    for i in range(n):
        for j in range(i + 1, n):
            total_dist += np.linalg.norm(population[i] - population[j])
            count += 1

    return total_dist / count if count > 0 else 0.0


# ==================== EXPERIMENT RUNNERS ====================

def run_single_experiment(
    alg_class,
    alg_name: str,
    benchmark,
    func_name: str,
    dim: int,
    config: ExperimentConfig,
    run_id: int
) -> RunResult:
    """Run a single algorithm on a single problem."""
    seed = config.base_seed + run_id
    lower, upper = benchmark.get_bounds(dim)

    kwargs = get_algorithm_kwargs(alg_name, dim, (lower, upper), config, seed)

    start_time = time.time()
    alg = alg_class(**kwargs)
    result = alg.run(benchmark, verbose=False)
    execution_time = time.time() - start_time

    # Extract trace data
    trace = result['trace']

    # Get SHAP values
    quick_shap = QuickSHAP()
    explanation = quick_shap.explain_from_trace(trace)

    # Get operator statistics
    op_stats = {}
    contributions = trace.get_operator_contributions()
    for op, stats in contributions.items():
        op_stats[op] = {
            'total_used': stats['total_used'],
            'total_success': stats['total_success'],
            'success_rate': stats['success_rate'],
        }

    # Calculate convergence
    fitness_history = [s.best_fitness for s in trace.snapshots]
    conv_gen = calculate_convergence_generation(fitness_history)

    # Population diversity (final)
    final_pop = trace.snapshots[-1].population if trace.snapshots else None
    diversity = calculate_population_diversity(final_pop)

    return RunResult(
        algorithm=alg_name,
        function=func_name,
        dimension=dim,
        run_id=run_id,
        seed=seed,
        best_fitness=result['best_fitness'],
        convergence_gen=conv_gen,
        final_population_diversity=diversity,
        execution_time=execution_time,
        shap_values=explanation.shap_values,
        operator_stats=op_stats,
    )


def run_ablation_study(
    alg_class,
    alg_name: str,
    benchmark,
    func_name: str,
    dim: int,
    config: ExperimentConfig,
    full_fitness: float
) -> List[AblationResult]:
    """Run ablation study for an algorithm."""
    lower, upper = benchmark.get_bounds(dim)
    base_kwargs = get_algorithm_kwargs(alg_name, dim, (lower, upper), config, config.base_seed)

    # Get operators
    alg = alg_class(**base_kwargs)
    alg._initialize_operators()
    operators = alg.get_operators()

    results = []

    for op_name in operators:
        ablated_fitness = []

        for i in range(config.ablation_runs):
            seed = config.base_seed + 1000 + i
            kwargs = {**base_kwargs, 'seed': seed}
            alg = alg_class(**kwargs)

            # Monkey-patch to disable after init
            original_init = alg._initialize_operators
            def patched_init(self=alg, orig=original_init, op=op_name):
                orig()
                self.disable_operator(op)
            alg._initialize_operators = patched_init

            result = alg.run(benchmark, verbose=False)
            ablated_fitness.append(result['best_fitness'])

        mean_fit = np.mean(ablated_fitness)
        std_fit = np.std(ablated_fitness)
        contribution = mean_fit - full_fitness  # Higher = operator more important

        results.append(AblationResult(
            algorithm=alg_name,
            function=func_name,
            dimension=dim,
            operator=op_name,
            mean_fitness=mean_fit,
            std_fitness=std_fit,
            contribution=contribution,
        ))

    return results


# ==================== STATISTICAL ANALYSIS ====================

def wilcoxon_test(data1: List[float], data2: List[float]) -> Tuple[float, float]:
    """Perform Wilcoxon signed-rank test."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            statistic, pvalue = stats.wilcoxon(data1, data2)
        except ValueError:
            # Identical distributions
            return 0.0, 1.0
    return statistic, pvalue


def friedman_test(data: Dict[str, List[float]]) -> Tuple[float, float, Dict[str, float]]:
    """
    Perform Friedman test for multiple algorithms.

    Returns:
        Tuple of (statistic, pvalue, mean_ranks)
    """
    algorithms = list(data.keys())
    n_algs = len(algorithms)
    n_runs = len(data[algorithms[0]])

    # Create matrix
    matrix = np.array([data[alg] for alg in algorithms]).T  # runs x algorithms

    # Compute ranks per run
    ranks = np.zeros_like(matrix)
    for i in range(n_runs):
        ranks[i] = stats.rankdata(matrix[i])

    mean_ranks = {alg: ranks[:, j].mean() for j, alg in enumerate(algorithms)}

    # Friedman statistic
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            statistic, pvalue = stats.friedmanchisquare(*[data[alg] for alg in algorithms])
        except ValueError:
            return 0.0, 1.0, mean_ranks

    return statistic, pvalue, mean_ranks


def analyze_statistical_results(
    run_results: List[RunResult],
    func_name: str,
    dim: int,
    alpha: float = 0.05
) -> StatisticalResult:
    """Perform statistical analysis for a function/dimension."""
    # Group by algorithm
    data = {}
    for r in run_results:
        if r.function == func_name and r.dimension == dim:
            if r.algorithm not in data:
                data[r.algorithm] = []
            data[r.algorithm].append(r.best_fitness)

    algorithms = list(data.keys())

    # Pairwise Wilcoxon tests
    wilcoxon_pvalues = {}
    wilcoxon_significant = {}

    for i, alg1 in enumerate(algorithms):
        for alg2 in algorithms[i+1:]:
            key = f"{alg1}_vs_{alg2}"
            _, pvalue = wilcoxon_test(data[alg1], data[alg2])
            wilcoxon_pvalues[key] = pvalue
            wilcoxon_significant[key] = pvalue < alpha

    # Friedman test
    friedman_stat, friedman_p, mean_ranks = friedman_test(data)

    # Best algorithm (lowest mean rank)
    best_alg = min(mean_ranks, key=mean_ranks.get)

    return StatisticalResult(
        function=func_name,
        dimension=dim,
        wilcoxon_pvalues=wilcoxon_pvalues,
        wilcoxon_significant=wilcoxon_significant,
        friedman_statistic=friedman_stat,
        friedman_pvalue=friedman_p,
        mean_ranks=mean_ranks,
        best_algorithm=best_alg,
    )


# ==================== SHAP VALIDATION ====================

def validate_shap_vs_ablation(
    run_results: List[RunResult],
    ablation_results: List[AblationResult],
    func_name: str,
    dim: int
) -> Dict[str, Dict[str, float]]:
    """Compare SHAP values against ablation contributions."""
    validation = {}

    for alg_name in ALGORITHMS.keys():
        # Get average SHAP values across runs
        shap_values = {}
        for r in run_results:
            if r.algorithm == alg_name and r.function == func_name and r.dimension == dim:
                for op, val in r.shap_values.items():
                    if op not in shap_values:
                        shap_values[op] = []
                    shap_values[op].append(val)

        mean_shap = {op: np.mean(vals) for op, vals in shap_values.items()}

        # Get ablation contributions
        ablation = {}
        for r in ablation_results:
            if r.algorithm == alg_name and r.function == func_name and r.dimension == dim:
                ablation[r.operator] = r.contribution

        # Compute correlation
        operators = list(set(mean_shap.keys()) & set(ablation.keys()))
        if len(operators) >= 2:
            shap_vals = [mean_shap[op] for op in operators]
            abl_vals = [ablation[op] for op in operators]

            if np.std(shap_vals) > 0 and np.std(abl_vals) > 0:
                pearson = np.corrcoef(shap_vals, abl_vals)[0, 1]
                spearman, _ = stats.spearmanr(shap_vals, abl_vals)
            else:
                pearson = 0.0
                spearman = 0.0

            # Rank agreement
            shap_ranking = sorted(operators, key=lambda x: -mean_shap[x])
            abl_ranking = sorted(operators, key=lambda x: -ablation[x])
            rank_match = 1.0 if shap_ranking[0] == abl_ranking[0] else 0.0
        else:
            pearson = 0.0
            spearman = 0.0
            rank_match = 0.0

        validation[alg_name] = {
            'pearson': pearson,
            'spearman': spearman,
            'rank_agreement': rank_match,
        }

    return validation


# ==================== OUTPUT ====================

def save_results(
    run_results: List[RunResult],
    ablation_results: List[AblationResult],
    statistical_results: List[StatisticalResult],
    shap_validation: Dict,
    config: ExperimentConfig,
    output_dir: str
):
    """Save all results to files."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save run results as CSV
    csv_path = os.path.join(output_dir, f"run_results_{timestamp}.csv")
    with open(csv_path, 'w') as f:
        headers = ["algorithm", "function", "dimension", "run_id", "seed",
                   "best_fitness", "convergence_gen", "diversity", "time"]
        f.write(",".join(headers) + "\n")

        for r in run_results:
            row = [r.algorithm, r.function, str(r.dimension), str(r.run_id),
                   str(r.seed), f"{r.best_fitness:.10e}", str(r.convergence_gen),
                   f"{r.final_population_diversity:.6f}", f"{r.execution_time:.2f}"]
            f.write(",".join(row) + "\n")

    print(f"  Run results saved to: {csv_path}")

    # Save ablation results
    abl_path = os.path.join(output_dir, f"ablation_results_{timestamp}.csv")
    with open(abl_path, 'w') as f:
        headers = ["algorithm", "function", "dimension", "operator",
                   "mean_fitness", "std_fitness", "contribution"]
        f.write(",".join(headers) + "\n")

        for r in ablation_results:
            row = [r.algorithm, r.function, str(r.dimension), r.operator,
                   f"{r.mean_fitness:.10e}", f"{r.std_fitness:.10e}",
                   f"{r.contribution:.10e}"]
            f.write(",".join(row) + "\n")

    print(f"  Ablation results saved to: {abl_path}")

    # Save statistical results as JSON
    stats_path = os.path.join(output_dir, f"statistical_results_{timestamp}.json")
    stats_data = [asdict(s) for s in statistical_results]
    # Convert numpy types to native Python types
    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        elif isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    stats_data = convert_numpy(stats_data)
    with open(stats_path, 'w') as f:
        json.dump(stats_data, f, indent=2)

    print(f"  Statistical results saved to: {stats_path}")

    # Save SHAP validation
    shap_path = os.path.join(output_dir, f"shap_validation_{timestamp}.json")
    shap_validation = convert_numpy(shap_validation)
    with open(shap_path, 'w') as f:
        json.dump(shap_validation, f, indent=2)

    print(f"  SHAP validation saved to: {shap_path}")

    # Save config
    config_path = os.path.join(output_dir, f"config_{timestamp}.json")
    with open(config_path, 'w') as f:
        json.dump(asdict(config), f, indent=2)


def print_summary(
    run_results: List[RunResult],
    statistical_results: List[StatisticalResult],
    shap_validation: Dict,
    config: ExperimentConfig
):
    """Print summary of results."""
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)

    # Performance summary
    print("\n1. ALGORITHM PERFORMANCE (Mean Best Fitness)")
    print("-" * 80)

    # Group by function and dimension
    for func in config.functions:
        for dim in config.dimensions:
            print(f"\n{func} (D={dim}):")

            for alg in ALGORITHMS.keys():
                fitness_vals = [r.best_fitness for r in run_results
                               if r.algorithm == alg and r.function == func and r.dimension == dim]
                if fitness_vals:
                    mean_fit = np.mean(fitness_vals)
                    std_fit = np.std(fitness_vals)
                    print(f"  {alg:>5}: {mean_fit:.6e} ± {std_fit:.6e}")

    # Statistical summary
    print("\n\n2. STATISTICAL ANALYSIS")
    print("-" * 80)

    wins = {alg: 0 for alg in ALGORITHMS.keys()}

    for sr in statistical_results:
        wins[sr.best_algorithm] += 1
        if sr.friedman_pvalue < 0.05:
            sig = "**"
        else:
            sig = ""
        print(f"{sr.function} D={sr.dimension}: Best={sr.best_algorithm} "
              f"(Friedman p={sr.friedman_pvalue:.4f}{sig})")

    print(f"\nTotal wins: {wins}")

    # SHAP validation summary
    print("\n\n3. SHAP VALIDATION (Avg Spearman correlation with Ablation)")
    print("-" * 80)

    for alg in ALGORITHMS.keys():
        correlations = []
        for key, val in shap_validation.items():
            if alg in val:
                correlations.append(val[alg]['spearman'])
        if correlations:
            avg_corr = np.mean([c for c in correlations if not np.isnan(c)])
            print(f"  {alg}: ρ = {avg_corr:.3f}")


# ==================== MAIN ====================

def run_full_study(config: ExperimentConfig, verbose: bool = True, resume: bool = False):
    """Run the complete experimental study with checkpoint support."""

    total_configs = len(config.functions) * len(config.dimensions) * len(ALGORITHMS)

    # Try to load existing checkpoint
    checkpoint = None
    if resume:
        checkpoint = load_checkpoint(config.output_dir)
        if checkpoint:
            print(f"\n*** RESUMING FROM CHECKPOINT ***")
            print(f"Progress: {checkpoint.get_progress_string()}")
            print(f"Last update: {datetime.fromtimestamp(checkpoint.last_update).strftime('%Y-%m-%d %H:%M:%S')}")
            print()
        else:
            print("No checkpoint found, starting fresh...")

    # Create new checkpoint if not resuming
    if checkpoint is None:
        checkpoint = ExperimentCheckpoint(
            config=config,
            completed_configs=set(),
            run_results=[],
            ablation_results=[],
            start_time=time.time(),
            last_update=time.time()
        )

    all_statistical_results = []
    all_shap_validation = {}

    current = 0
    skipped = 0

    if verbose:
        print("=" * 80)
        print("XMH FRAMEWORK - FULL EXPERIMENTAL STUDY")
        print("=" * 80)
        print(f"Functions: {len(config.functions)}")
        print(f"Dimensions: {config.dimensions}")
        print(f"Algorithms: {list(ALGORITHMS.keys())}")
        print(f"Runs per config: {config.n_runs}")
        print(f"Total configurations: {total_configs}")
        print(f"Estimated runs: {total_configs * config.n_runs}")
        if resume and checkpoint.completed_configs:
            print(f"Already completed: {len(checkpoint.completed_configs)}")
        print(f"\nCheckpoint will be saved to: {config.output_dir}/{CHECKPOINT_FILE}")
        print()

    for func_name in config.functions:
        benchmark = get_benchmark(func_name)

        for dim in config.dimensions:
            if verbose:
                print(f"\n{'='*60}")
                print(f"FUNCTION: {func_name} | DIMENSION: {dim}")
                print(f"{'='*60}")

            func_run_results = []
            func_ablation_results = []

            for alg_name, alg_class in ALGORITHMS.items():
                current += 1

                # Check if already completed
                if is_config_completed(checkpoint, func_name, dim, alg_name):
                    skipped += 1
                    if verbose:
                        print(f"\n[{current}/{total_configs}] {alg_name}... SKIPPED (already completed)")

                    # Get existing results for statistical analysis
                    func_run_results.extend([
                        r for r in checkpoint.run_results
                        if r.algorithm == alg_name and r.function == func_name and r.dimension == dim
                    ])
                    func_ablation_results.extend([
                        r for r in checkpoint.ablation_results
                        if r.algorithm == alg_name and r.function == func_name and r.dimension == dim
                    ])
                    continue

                if verbose:
                    print(f"\n[{current}/{total_configs}] {alg_name}...")

                # Run all repetitions
                alg_results = []
                for run_id in range(config.n_runs):
                    if verbose and (run_id % 10 == 0 or run_id == config.n_runs - 1):
                        print(f"  Run {run_id + 1}/{config.n_runs}...", end="\r")

                    result = run_single_experiment(
                        alg_class, alg_name, benchmark, func_name,
                        dim, config, run_id
                    )
                    alg_results.append(result)

                func_run_results.extend(alg_results)
                checkpoint.run_results.extend(alg_results)

                # Mean fitness for ablation baseline
                mean_fitness = np.mean([r.best_fitness for r in alg_results])

                if verbose:
                    std_fitness = np.std([r.best_fitness for r in alg_results])
                    print(f"  {alg_name}: {mean_fitness:.6e} ± {std_fitness:.6e}")

                # Ablation study
                if verbose:
                    print(f"  Running ablation...", end=" ")

                ablation = run_ablation_study(
                    alg_class, alg_name, benchmark, func_name,
                    dim, config, mean_fitness
                )
                func_ablation_results.extend(ablation)
                checkpoint.ablation_results.extend(ablation)

                if verbose:
                    print("done")

                # Mark as completed and save checkpoint
                mark_config_completed(checkpoint, func_name, dim, alg_name)
                save_checkpoint(checkpoint, config.output_dir)

                if verbose:
                    progress = checkpoint.get_progress_string()
                    print(f"  [Checkpoint saved: {progress}]")

            # Statistical analysis for this function/dimension (only if we have all algorithms)
            all_algs_completed = all(
                is_config_completed(checkpoint, func_name, dim, alg)
                for alg in ALGORITHMS.keys()
            )

            if all_algs_completed and func_run_results:
                stat_result = analyze_statistical_results(
                    func_run_results, func_name, dim
                )
                all_statistical_results.append(stat_result)

                # SHAP validation
                key = f"{func_name}_D{dim}"
                all_shap_validation[key] = validate_shap_vs_ablation(
                    func_run_results, func_ablation_results, func_name, dim
                )

                if verbose:
                    print(f"\n  Best algorithm: {stat_result.best_algorithm}")
                    print(f"  Friedman p-value: {stat_result.friedman_pvalue:.4f}")

    elapsed = time.time() - checkpoint.start_time

    if verbose:
        print(f"\n\nTotal experiment time: {elapsed/60:.1f} minutes")
        if skipped > 0:
            print(f"Configurations skipped (from checkpoint): {skipped}")

    # Save final results
    if verbose:
        print(f"\nSaving final results to {config.output_dir}/...")

    save_results(
        checkpoint.run_results, checkpoint.ablation_results, all_statistical_results,
        all_shap_validation, config, config.output_dir
    )

    # Remove checkpoint file after successful completion
    checkpoint_path = os.path.join(config.output_dir, CHECKPOINT_FILE)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        if verbose:
            print("Checkpoint file removed (experiment completed successfully)")

    # Print summary
    print_summary(
        checkpoint.run_results, all_statistical_results,
        all_shap_validation, config
    )

    return checkpoint.run_results, checkpoint.ablation_results, all_statistical_results


def main():
    parser = argparse.ArgumentParser(description="XMH Full Experimental Study")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test mode (5 runs, D=10 only)")
    parser.add_argument("--medium", action="store_true",
                        help="Medium mode (10 runs, D=10,30)")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory for results")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing checkpoint")
    parser.add_argument("--functions", type=str, nargs="+",
                        help="Specific functions to test")
    parser.add_argument("--dimensions", type=int, nargs="+",
                        help="Specific dimensions to test")
    parser.add_argument("--runs", type=int, default=30,
                        help="Number of runs per configuration")

    args = parser.parse_args()

    # Build config
    config = ExperimentConfig()
    config.output_dir = args.output

    if args.quick:
        config.n_runs = 5
        config.dimensions = [10]
        config.functions = ["sphere", "rastrigin", "ackley"]
        config.ablation_runs = 3
        config.max_generations = 100
        print("Running in QUICK mode (test configuration)")
    elif args.medium:
        config.n_runs = 10
        config.dimensions = [10, 30]
        config.ablation_runs = 3
        config.max_generations = 200
        print("Running in MEDIUM mode")
    else:
        config.n_runs = args.runs
        print("Running FULL experimental study")

    if args.functions:
        config.functions = args.functions
    if args.dimensions:
        config.dimensions = args.dimensions

    # Always try to resume - if no checkpoint exists, starts fresh
    run_full_study(config, resume=args.resume or True)  # Default: always try to resume


if __name__ == "__main__":
    main()
