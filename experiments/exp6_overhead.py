"""
Experiment 6: Detailed Overhead Analysis

Separates overhead into components:
  1. Instrumentation overhead (logging events during the run)
  2. QuickSHAP overhead (post-hoc computation from trace)
  3. Exact Shapley cost (2^n coalition evaluations)
  4. KernelSHAP cost (sampled coalition evaluations)

Reports wall-clock time, FES (function evaluations), and relative overhead.
"""

import argparse
import json
import os
import time
from typing import Dict

import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from xmh.algorithms.instrumented_de import InstrumentedDE
from xmh.algorithms.instrumented_ga import InstrumentedGA
from xmh.algorithms.instrumented_pso import InstrumentedPSO
from xmh.benchmarks.functions import get_benchmark
from xmh.core.trace_logger import InstrumentationLevel
from xmh.explanation.operator_shap import ExactOperatorSHAP, QuickSHAP, KernelSHAP
from xmh.explanation.tracking_attribution import TrackingAttribution

HAS_PANDAS = False


ALGORITHMS = {
    "DE": {
        "class": InstrumentedDE,
        "kwargs": {
            "F": 0.5, "CR": 0.9,
            "mutation_strategy": "rand/1",
            "crossover_type": "binomial",
        },
    },
    "GA": {
        "class": InstrumentedGA,
        "kwargs": {
            "selection_method": "tournament",
            "crossover_type": "sbx",
            "mutation_type": "polynomial",
            "crossover_prob": 0.9,
            "tournament_size": 3,
        },
    },
    "PSO": {
        "class": InstrumentedPSO,
        "kwargs": {
            "topology": "gbest",
            "velocity_strategy": "constriction",
        },
    },
}

POP_SIZE = 50
MAX_GENERATIONS = 100
N_TIMING_RUNS = 10
SEED = 42


def _time_run(alg_class, alg_kwargs, func, level, seed):
    """Time a single algorithm run with a given instrumentation level."""
    kwargs = {**alg_kwargs, "seed": seed, "instrumentation_level": level}
    alg = alg_class(**kwargs)
    t0 = time.perf_counter()
    result = alg.run(func, verbose=False)
    elapsed = time.perf_counter() - t0
    return elapsed, result


def run_experiment(
    output_dir: str = "results_overhead",
    quick: bool = False,
):
    """Run the detailed overhead analysis."""
    os.makedirs(output_dir, exist_ok=True)

    functions = ["sphere", "rastrigin", "ackley"]
    dimensions = [10, 30] if not quick else [10]
    n_runs = 3 if quick else N_TIMING_RUNS

    all_results = []

    for alg_name, alg_info in ALGORITHMS.items():
        alg_class = alg_info["class"]
        alg_extra = alg_info["kwargs"]

        for func_name in functions:
            benchmark = get_benchmark(func_name)

            for dim in dimensions:
                print(f"\n{'='*60}")
                print(f"  {alg_name} | {func_name} | D={dim}")
                print(f"{'='*60}")

                lower, upper = benchmark.get_bounds(dim)
                bounds = (lower, upper)
                base_kwargs = {
                    "dim": dim,
                    "bounds": bounds,
                    "pop_size": POP_SIZE,
                    "max_generations": MAX_GENERATIONS,
                    **alg_extra,
                }

                # ── 1. Baseline: NONE instrumentation ──
                none_times = []
                for i in range(n_runs):
                    t, _ = _time_run(
                        alg_class, base_kwargs, benchmark.function,
                        InstrumentationLevel.NONE, SEED + i,
                    )
                    none_times.append(t)
                baseline_mean = np.mean(none_times)
                baseline_std = np.std(none_times)

                # ── 2. STANDARD instrumentation ──
                std_times = []
                trace_result = None
                for i in range(n_runs):
                    t, result = _time_run(
                        alg_class, base_kwargs, benchmark.function,
                        InstrumentationLevel.STANDARD, SEED + i,
                    )
                    std_times.append(t)
                    if i == 0:
                        trace_result = result
                std_mean = np.mean(std_times)
                std_std = np.std(std_times)

                instr_overhead = (std_mean - baseline_mean) / baseline_mean * 100

                # ── 3. QuickSHAP cost ──
                quick_times = []
                for _ in range(n_runs):
                    qs = QuickSHAP()
                    t0 = time.perf_counter()
                    qs.explain_from_trace(trace_result["trace"])
                    quick_times.append(time.perf_counter() - t0)
                quick_mean = np.mean(quick_times)

                # ── 4. Tracking Attribution cost ──
                track_times = []
                for _ in range(n_runs):
                    ta = TrackingAttribution()
                    t0 = time.perf_counter()
                    ta.explain_from_trace(trace_result["trace"])
                    track_times.append(time.perf_counter() - t0)
                track_mean = np.mean(track_times)

                # ── 5. KernelSHAP cost ──
                kernel_shap = KernelSHAP(
                    n_samples=100, n_runs_per_coalition=3,
                )
                t0 = time.perf_counter()
                kernel_shap.explain(alg_class, base_kwargs, benchmark.function)
                kernel_time = time.perf_counter() - t0

                # ── 6. Exact Shapley cost ──
                exact_shap = ExactOperatorSHAP(
                    n_runs_per_coalition=5,
                    base_seed=SEED,
                    compute_interactions=False,
                )
                t0 = time.perf_counter()
                exact_shap.explain(alg_class, base_kwargs, benchmark.function)
                exact_time = time.perf_counter() - t0

                # Count FES
                n_operators = len(trace_result["trace"].get_operator_contributions())
                n_coalitions = 2 ** n_operators
                exact_fes = n_coalitions * 5 * POP_SIZE * MAX_GENERATIONS
                single_fes = POP_SIZE * MAX_GENERATIONS

                row = {
                    "algorithm": alg_name,
                    "function": func_name,
                    "dimension": dim,
                    "n_operators": n_operators,
                    "n_coalitions": n_coalitions,
                    # Times
                    "baseline_time_s": baseline_mean,
                    "baseline_std_s": baseline_std,
                    "instrumented_time_s": std_mean,
                    "instrumented_std_s": std_std,
                    "quickshap_time_s": quick_mean,
                    "tracking_time_s": track_mean,
                    "kernelshap_time_s": kernel_time,
                    "exact_time_s": exact_time,
                    # Overheads
                    "instrumentation_overhead_pct": instr_overhead,
                    "quickshap_overhead_pct": quick_mean / baseline_mean * 100,
                    "tracking_overhead_pct": track_mean / baseline_mean * 100,
                    "kernelshap_multiplier": kernel_time / baseline_mean,
                    "exact_multiplier": exact_time / baseline_mean,
                    # FES
                    "single_run_fes": single_fes,
                    "exact_total_fes": exact_fes,
                }

                all_results.append(row)

                print(f"  Baseline (NONE):  {baseline_mean:.4f}s ± {baseline_std:.4f}s")
                print(f"  Instrumented:     {std_mean:.4f}s ± {std_std:.4f}s  ({instr_overhead:+.2f}%)")
                print(f"  QuickSHAP:        {quick_mean:.6f}s  ({quick_mean/baseline_mean*100:.4f}% of run)")
                print(f"  Tracking:         {track_mean:.6f}s  ({track_mean/baseline_mean*100:.4f}% of run)")
                print(f"  KernelSHAP:       {kernel_time:.2f}s  ({kernel_time/baseline_mean:.1f}x single run)")
                print(f"  Exact Shapley:    {exact_time:.2f}s  ({exact_time/baseline_mean:.1f}x single run)")

    # Save
    results_file = os.path.join(output_dir, "overhead_results.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    if HAS_PANDAS:
        df = pd.DataFrame(all_results)
        csv_path = os.path.join(output_dir, "overhead_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nCSV saved to {csv_path}")

        print("\n" + "=" * 70)
        print("OVERHEAD SUMMARY (mean across configs)")
        print("=" * 70)
        cols = [
            "instrumentation_overhead_pct",
            "quickshap_overhead_pct",
            "tracking_overhead_pct",
            "kernelshap_multiplier",
            "exact_multiplier",
        ]
        summary = df.groupby("algorithm")[cols].mean()
        print(summary.to_string(float_format="%.4f"))

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detailed overhead analysis"
    )
    parser.add_argument("--output", default="results_overhead")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    run_experiment(output_dir=args.output, quick=args.quick)
