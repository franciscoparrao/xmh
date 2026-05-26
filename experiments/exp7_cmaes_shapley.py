#!/usr/bin/env python3 -u
"""
Experiment 7: Exact Shapley Values for CMA-ES Operator Attribution

Same protocol as exp4 but for CMA-ES:
  - 9 functions x 3 dimensions (10, 30, 50)
  - 2^3 = 8 coalitions (recombination, covariance, step-size)
  - 30 runs per coalition with controlled seeds (42 + i)
  - pop_size=50, max_gen=100 (FES=5050, matching DE/GA/PSO protocol)
  - Total: 9 x 3 x 8 x 30 = 6,480 runs

Memory-optimized: saves incrementally, GC between combos.
"""

import argparse
import gc
import json
import math
import os
import sys
import time
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from xmh.algorithms.instrumented_cmaes import InstrumentedCMAES
from xmh.benchmarks.functions import get_benchmark, list_benchmarks


# ── Configuration ───────────────────────────────────────────────────

FUNCTIONS = list_benchmarks()
DIMENSIONS = [10, 30, 50]
N_RUNS = 30
BASE_SEED = 42
POP_SIZE = 50
MAX_GENERATIONS = 100


def run_single_config(func, dim, bounds, n_runs, base_seed):
    """Run all 8 coalitions for one function-dimension pair.

    Memory-efficient: processes one run at a time, no caching of algorithm
    instances.
    """
    alg_kwargs = {
        "dim": dim,
        "bounds": bounds,
        "pop_size": POP_SIZE,
        "max_generations": MAX_GENERATIONS,
    }

    # Get operator names from reference instance
    ref = InstrumentedCMAES(**alg_kwargs, seed=0)
    ref._initialize_operators()
    operators = ref.get_operators()
    n = len(operators)
    del ref
    gc.collect()

    # Enumerate all 2^n coalitions
    all_coalitions = []
    for size in range(n + 1):
        for subset in combinations(range(n), size):
            all_coalitions.append(frozenset(subset))

    # Evaluate each coalition
    coalition_values = {}
    for ci, coalition in enumerate(all_coalitions):
        active = [operators[i] for i in coalition]
        ops_to_disable = [op for op in operators if op not in active]

        values = []
        for r in range(n_runs):
            kwargs = {**alg_kwargs, "seed": base_seed + r}
            alg = InstrumentedCMAES(**kwargs)
            if ops_to_disable:
                orig = alg._initialize_operators
                def patched(_self=alg, _orig=orig, _ops=ops_to_disable):
                    _orig()
                    for op in _ops:
                        _self.disable_operator(op)
                alg._initialize_operators = patched
            result = alg.run(func, verbose=False)
            values.append(result["best_fitness"])
            del alg, result

        coalition_values[coalition] = values
        gc.collect()

    # Compute per-run Shapley values
    per_run_shapley = {op: [] for op in operators}
    for r in range(n_runs):
        for idx, op in enumerate(operators):
            phi = 0.0
            for coalition in coalition_values:
                if idx not in coalition:
                    s = len(coalition)
                    weight = (
                        math.factorial(s) * math.factorial(n - s - 1)
                        / math.factorial(n)
                    )
                    with_i = coalition | {idx}
                    marginal = coalition_values[coalition][r] - coalition_values[with_i][r]
                    phi += weight * marginal
            per_run_shapley[op].append(phi)

    # Average Shapley values and CIs
    shapley_values = {}
    confidence_intervals = {}
    z = 1.96
    for op in operators:
        vals = np.array(per_run_shapley[op])
        mean_phi = float(np.mean(vals))
        std_phi = float(np.std(vals, ddof=1)) if n_runs > 1 else 0.0
        se = std_phi / np.sqrt(n_runs)
        shapley_values[op] = mean_phi
        confidence_intervals[op] = [mean_phi - z * se, mean_phi + z * se]

    # Interaction indices
    interactions = {}
    for i in range(n):
        for j in range(i + 1, n):
            interaction = 0.0
            for coalition in coalition_values:
                if i not in coalition and j not in coalition:
                    s = len(coalition)
                    weight = (
                        math.factorial(s) * math.factorial(n - s - 2)
                        / math.factorial(n - 1)
                    )
                    v_neither = np.mean(coalition_values[coalition])
                    v_i = np.mean(coalition_values[coalition | {i}])
                    v_j = np.mean(coalition_values[coalition | {j}])
                    v_both = np.mean(coalition_values[coalition | {i, j}])
                    interaction += weight * (v_neither - v_i - v_j + v_both)
            interactions[f"{operators[i]}_x_{operators[j]}"] = float(interaction)

    # Base and total
    empty = frozenset()
    full = frozenset(range(n))
    base_value = float(np.mean(coalition_values[empty]))
    full_value = float(np.mean(coalition_values[full]))
    total_contribution = base_value - full_value

    serializable_coalitions = {
        str(tuple(sorted(k))): v for k, v in coalition_values.items()
    }

    return {
        "algorithm": "CMA-ES",
        "function": None,
        "dimension": dim,
        "operators": operators,
        "shapley_values": shapley_values,
        "interactions": interactions,
        "coalitions": serializable_coalitions,
        "total_contribution": total_contribution,
        "base_value": base_value,
        "confidence_intervals": confidence_intervals,
        "n_runs": n_runs,
    }


def run_experiment(
    output_dir: str = "results_cmaes_shapley",
    quick: bool = False,
    n_runs: int = N_RUNS,
):
    """Run exact Shapley experiment for CMA-ES."""
    os.makedirs(output_dir, exist_ok=True)

    if quick:
        functions = ["sphere", "rastrigin", "rosenbrock"]
        dimensions = [10]
        n_runs = 5
    else:
        functions = FUNCTIONS
        dimensions = DIMENSIONS

    # Load existing results if resuming
    results_file = os.path.join(output_dir, "cmaes_shapley_results.json")
    if os.path.exists(results_file):
        with open(results_file) as f:
            all_results = json.load(f)
        print(f"Resuming from {results_file}")
    else:
        all_results = {"CMA-ES": {}}

    total_start = time.time()
    combo_count = 0
    total_combos = len(functions) * len(dimensions)

    for func_name in functions:
        benchmark = get_benchmark(func_name)
        if func_name not in all_results.get("CMA-ES", {}):
            all_results.setdefault("CMA-ES", {})[func_name] = {}

        for dim in dimensions:
            combo_count += 1

            # Skip if already computed
            if str(dim) in all_results.get("CMA-ES", {}).get(func_name, {}):
                existing = all_results["CMA-ES"][func_name][str(dim)]
                if existing.get("n_runs", 0) >= n_runs:
                    print(f"  [{combo_count}/{total_combos}] {func_name} D={dim} — already done, skipping")
                    continue

            print(f"\n{'='*60}", flush=True)
            print(f"  [{combo_count}/{total_combos}] CMA-ES | {func_name} | D={dim}", flush=True)
            print(f"{'='*60}", flush=True)

            lower, upper = benchmark.get_bounds(dim)
            bounds = (lower, upper)

            t0 = time.time()
            result = run_single_config(
                benchmark.function, dim, bounds, n_runs, BASE_SEED
            )
            elapsed = time.time() - t0

            result["function"] = func_name
            result["elapsed_seconds"] = elapsed
            all_results["CMA-ES"][func_name][str(dim)] = result

            # Print summary
            print(f"  Time: {elapsed:.1f}s", flush=True)
            tc = result["total_contribution"]
            print(f"  Total contribution: {tc:.6e}", flush=True)
            for op, val in sorted(
                result["shapley_values"].items(),
                key=lambda x: abs(x[1]),
                reverse=True,
            ):
                pct = abs(val) / abs(tc) * 100 if tc != 0 else 0
                ci = result["confidence_intervals"][op]
                print(f"    {op:35s} {val:+.4e} ({pct:5.1f}%) [{ci[0]:+.2e}, {ci[1]:+.2e}]", flush=True)

            shapley_sum = sum(result["shapley_values"].values())
            print(f"  Σφ = {shapley_sum:.6e}  (should ≈ {tc:.6e})", flush=True)

            # Save incrementally after each combo
            with open(results_file, "w") as f:
                json.dump(all_results, f, indent=2, default=str)

            # ETA
            avg_time = (time.time() - total_start) / combo_count
            remaining = (total_combos - combo_count) * avg_time
            print(f"  Saved. ETA: {remaining/60:.0f} min remaining", flush=True)

            # Free memory
            del result
            gc.collect()

    total_elapsed = time.time() - total_start
    print(f"\nTotal experiment time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)", flush=True)
    print(f"Results saved to {results_file}", flush=True)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exact Shapley values for CMA-ES operator attribution"
    )
    parser.add_argument(
        "--output", default="results_cmaes_shapley",
        help="Output directory"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode (3 functions, D=10, 5 runs)"
    )
    parser.add_argument(
        "--runs", type=int, default=N_RUNS,
        help=f"Runs per coalition (default: {N_RUNS})"
    )
    args = parser.parse_args()

    run_experiment(
        output_dir=args.output,
        quick=args.quick,
        n_runs=args.runs,
    )
