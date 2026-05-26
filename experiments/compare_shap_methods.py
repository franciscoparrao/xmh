#!/usr/bin/env python3
"""
Compare QuickSHAP vs KernelSHAP vs Ablation.

This script evaluates the accuracy of different SHAP approximation methods
by comparing them against ablation study results.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from xmh.algorithms.instrumented_de import InstrumentedDE
from xmh.algorithms.instrumented_ga import InstrumentedGA
from xmh.benchmarks.functions import get_benchmark
from xmh.explanation.operator_shap import QuickSHAP, KernelSHAP
from xmh.core.trace_logger import InstrumentationLevel
import numpy as np
from scipy import stats
import time


def run_ablation(algorithm_class, kwargs, objective_func, n_runs=3):
    """Run ablation study to get ground truth contributions."""
    # Full run
    full_results = []
    for i in range(n_runs):
        alg = algorithm_class(**{**kwargs, 'seed': kwargs.get('seed', 42) + i})
        result = alg.run(objective_func, verbose=False)
        full_results.append(result['best_fitness'])
    full_fitness = np.mean(full_results)

    # Get operators
    alg = algorithm_class(**kwargs)
    alg._initialize_operators()
    operators = alg.get_operators()

    # Ablation for each operator
    ablation_contrib = {}
    for op_name in operators:
        ablated_results = []
        for i in range(n_runs):
            alg = algorithm_class(**{**kwargs, 'seed': kwargs.get('seed', 42) + i})
            original_init = alg._initialize_operators
            def patched_init(self=alg, orig=original_init, op=op_name):
                orig()
                self.disable_operator(op)
            alg._initialize_operators = patched_init
            result = alg.run(objective_func, verbose=False)
            ablated_results.append(result['best_fitness'])
        ablation_contrib[op_name] = np.mean(ablated_results) - full_fitness

    return full_fitness, ablation_contrib, operators


def run_quick_shap(algorithm_class, kwargs, objective_func):
    """Run QuickSHAP."""
    alg = algorithm_class(**kwargs)
    result = alg.run(objective_func, verbose=False)
    trace = result['trace']

    quick_shap = QuickSHAP()
    explanation = quick_shap.explain_from_trace(trace)

    return explanation.shap_values


def run_kernel_shap(algorithm_class, kwargs, objective_func, n_coalitions=None):
    """Run Kernel SHAP."""
    kernel_shap = KernelSHAP(
        n_samples=50,
        n_runs_per_coalition=2,
        background_samples=3
    )
    explanation = kernel_shap.explain(
        algorithm_class, kwargs, objective_func, n_coalitions
    )
    return explanation.shap_values


def compute_correlations(ablation, shap_values, operators):
    """Compute Pearson and Spearman correlations."""
    abl_vals = np.array([ablation[op] for op in operators])
    shap_vals = np.array([shap_values.get(op, 0) for op in operators])

    # Handle constant arrays
    if np.std(abl_vals) == 0 or np.std(shap_vals) == 0:
        pearson = 0.0
    else:
        pearson = np.corrcoef(abl_vals, shap_vals)[0, 1]

    # Spearman
    if len(operators) >= 2:
        spearman, _ = stats.spearmanr(abl_vals, shap_vals)
    else:
        spearman = 0.0

    # Rank agreement (top operator matches)
    abl_top = operators[np.argmax(abl_vals)]
    shap_top = max(shap_values.items(), key=lambda x: x[1])[0] if shap_values else None
    rank_match = 1.0 if abl_top == shap_top else 0.0

    return pearson, spearman, rank_match


def compare_methods(algorithm_class, alg_name, func_name, dim=10):
    """Compare all SHAP methods for a given algorithm and function."""
    print(f"\n{'='*70}")
    print(f"COMPARING SHAP METHODS: {alg_name} on {func_name} (D={dim})")
    print(f"{'='*70}")

    benchmark = get_benchmark(func_name)
    lower, upper = benchmark.get_bounds(dim)

    kwargs = {
        'dim': dim,
        'bounds': (lower, upper),
        'pop_size': 30,
        'max_generations': 50,
        'instrumentation_level': InstrumentationLevel.STANDARD,
        'seed': 42
    }

    # 1. Ablation (ground truth)
    print("\n[1/3] Running Ablation Study (ground truth)...", end=" ", flush=True)
    start = time.time()
    full_fitness, ablation, operators = run_ablation(algorithm_class, kwargs, benchmark, n_runs=3)
    t_ablation = time.time() - start
    print(f"done ({t_ablation:.1f}s)")

    # 2. QuickSHAP
    print("[2/3] Running QuickSHAP...", end=" ", flush=True)
    start = time.time()
    quick_shap = run_quick_shap(algorithm_class, kwargs, benchmark)
    t_quick = time.time() - start
    print(f"done ({t_quick:.1f}s)")

    # 3. Kernel SHAP
    print("[3/3] Running Kernel SHAP...", end=" ", flush=True)
    start = time.time()
    kernel_shap = run_kernel_shap(algorithm_class, kwargs, benchmark)
    t_kernel = time.time() - start
    print(f"done ({t_kernel:.1f}s)")

    # Compute correlations
    quick_pearson, quick_spearman, quick_rank = compute_correlations(ablation, quick_shap, operators)
    kernel_pearson, kernel_spearman, kernel_rank = compute_correlations(ablation, kernel_shap, operators)

    # Print results
    print(f"\nFull algorithm fitness: {full_fitness:.6e}")
    print(f"\n{'Operator':<30} {'Ablation':>12} {'QuickSHAP':>12} {'KernelSHAP':>12}")
    print("-" * 70)

    for op in operators:
        abl = ablation[op]
        qs = quick_shap.get(op, 0)
        ks = kernel_shap.get(op, 0)
        print(f"{op:<30} {abl:>12.4f} {qs:>12.4f} {ks:>12.4f}")

    print("-" * 70)
    print(f"{'TOTAL':<30} {sum(ablation.values()):>12.4f} {sum(quick_shap.values()):>12.4f} {sum(kernel_shap.values()):>12.4f}")

    print(f"\n{'Metric':<20} {'QuickSHAP':>15} {'KernelSHAP':>15}")
    print("-" * 52)
    print(f"{'Pearson r':<20} {quick_pearson:>15.3f} {kernel_pearson:>15.3f}")
    print(f"{'Spearman ρ':<20} {quick_spearman:>15.3f} {kernel_spearman:>15.3f}")
    print(f"{'Top-1 Match':<20} {quick_rank:>15.0%} {kernel_rank:>15.0%}")
    print(f"{'Time (s)':<20} {t_quick:>15.1f} {t_kernel:>15.1f}")

    return {
        'algorithm': alg_name,
        'function': func_name,
        'quick_pearson': quick_pearson,
        'quick_spearman': quick_spearman,
        'quick_rank': quick_rank,
        'kernel_pearson': kernel_pearson,
        'kernel_spearman': kernel_spearman,
        'kernel_rank': kernel_rank,
        't_quick': t_quick,
        't_kernel': t_kernel
    }


if __name__ == "__main__":
    results = []

    # Test configurations
    configs = [
        (InstrumentedDE, "DE", "sphere"),
        (InstrumentedDE, "DE", "rastrigin"),
        (InstrumentedGA, "GA", "sphere"),
        (InstrumentedGA, "GA", "rastrigin"),
    ]

    for alg_class, alg_name, func_name in configs:
        result = compare_methods(alg_class, alg_name, func_name)
        results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: QuickSHAP vs KernelSHAP")
    print("=" * 70)

    print(f"\n{'Config':<20} {'Quick ρ':>10} {'Kernel ρ':>10} {'Quick Top1':>12} {'Kernel Top1':>12}")
    print("-" * 70)

    for r in results:
        config = f"{r['algorithm']}/{r['function']}"
        print(f"{config:<20} {r['quick_spearman']:>10.3f} {r['kernel_spearman']:>10.3f} "
              f"{r['quick_rank']:>12.0%} {r['kernel_rank']:>12.0%}")

    # Averages
    avg_quick_spearman = np.mean([r['quick_spearman'] for r in results if not np.isnan(r['quick_spearman'])])
    avg_kernel_spearman = np.mean([r['kernel_spearman'] for r in results if not np.isnan(r['kernel_spearman'])])
    avg_quick_rank = np.mean([r['quick_rank'] for r in results])
    avg_kernel_rank = np.mean([r['kernel_rank'] for r in results])

    print("-" * 70)
    print(f"{'AVERAGE':<20} {avg_quick_spearman:>10.3f} {avg_kernel_spearman:>10.3f} "
          f"{avg_quick_rank:>12.0%} {avg_kernel_rank:>12.0%}")

    print(f"\n{'Method':<15} {'Avg Time (s)':>15}")
    print("-" * 32)
    print(f"{'QuickSHAP':<15} {np.mean([r['t_quick'] for r in results]):>15.1f}")
    print(f"{'KernelSHAP':<15} {np.mean([r['t_kernel'] for r in results]):>15.1f}")
