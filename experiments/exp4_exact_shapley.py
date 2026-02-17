"""
Experiment 4: Exact Shapley Values for Operator Attribution

Computes exact Shapley values by enumerating all 2^n coalitions for each
algorithm-function-dimension combination.

Protocol:
  - 3 algorithms (DE, GA, PSO) x 9 functions x 3 dimensions (10, 30, 50)
  - 2^n coalitions per algorithm (n = number of operators)
  - 30 runs per coalition with controlled seeds (42 + i, i = 0..29)
  - Total ~ 17,010 runs (estimated 2-4 hours)

Outputs:
  - Coalition values v(S) for every coalition and run (JSON)
  - Exact Shapley values per operator (CSV)
  - Shapley interaction indices (CSV)
  - Heatmaps: operator x function (PDF)
"""

import argparse
import json
import os
import time
from itertools import combinations

import numpy as np

HAS_PANDAS = False

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from xmh.algorithms.instrumented_de import InstrumentedDE
from xmh.algorithms.instrumented_ga import InstrumentedGA
from xmh.algorithms.instrumented_pso import InstrumentedPSO
from xmh.benchmarks.functions import get_benchmark, list_benchmarks
from xmh.explanation.operator_shap import ExactOperatorSHAP


# ── Configuration ───────────────────────────────────────────────────

ALGORITHMS = {
    "DE": {
        "class": InstrumentedDE,
        "kwargs": {
            "F": 0.5,
            "CR": 0.9,
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

FUNCTIONS = list_benchmarks()
DIMENSIONS = [10, 30, 50]
N_RUNS = 30
BASE_SEED = 42
POP_SIZE = 50
MAX_GENERATIONS = 100


def run_experiment(
    output_dir: str = "results_exact_shapley",
    quick: bool = False,
    n_runs: int = N_RUNS,
):
    """Run the exact Shapley experiment."""
    os.makedirs(output_dir, exist_ok=True)

    if quick:
        functions = ["sphere", "rastrigin"]
        dimensions = [10]
        n_runs = 5
    else:
        functions = FUNCTIONS
        dimensions = DIMENSIONS

    all_results = {}
    total_start = time.time()

    for alg_name, alg_info in ALGORITHMS.items():
        alg_class = alg_info["class"]
        alg_extra = alg_info["kwargs"]
        all_results[alg_name] = {}

        for func_name in functions:
            benchmark = get_benchmark(func_name)
            all_results[alg_name][func_name] = {}

            for dim in dimensions:
                print(f"\n{'='*60}")
                print(f"  {alg_name} | {func_name} | D={dim}")
                print(f"{'='*60}")

                lower, upper = benchmark.get_bounds(dim)
                bounds = (lower, upper)

                alg_kwargs = {
                    "dim": dim,
                    "bounds": bounds,
                    "pop_size": POP_SIZE,
                    "max_generations": MAX_GENERATIONS,
                    **alg_extra,
                }

                exact_shap = ExactOperatorSHAP(
                    n_runs_per_coalition=n_runs,
                    base_seed=BASE_SEED,
                    compute_interactions=True,
                )

                t0 = time.time()
                raw = exact_shap.get_coalition_values(
                    alg_class, alg_kwargs, benchmark.function,
                )
                elapsed = time.time() - t0

                explanation = exact_shap.explain(
                    alg_class, alg_kwargs, benchmark.function,
                )

                result = {
                    "algorithm": alg_name,
                    "function": func_name,
                    "dimension": dim,
                    "operators": raw["operators"],
                    "shapley_values": raw["shapley_values"],
                    "interactions": {
                        f"{k[0]}_x_{k[1]}": v
                        for k, v in (raw["interactions"] or {}).items()
                    },
                    "coalitions": {
                        str(k): v for k, v in raw["coalitions"].items()
                    },
                    "total_contribution": explanation.total_contribution,
                    "base_value": explanation.base_value,
                    "confidence_intervals": {
                        k: list(v)
                        for k, v in explanation.confidence_intervals.items()
                    },
                    "elapsed_seconds": elapsed,
                    "n_runs": n_runs,
                }

                all_results[alg_name][func_name][str(dim)] = result

                # Print summary
                print(f"  Time: {elapsed:.1f}s")
                print(f"  Total contribution: {explanation.total_contribution:.6e}")
                for op, val in sorted(
                    explanation.shap_values.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True,
                ):
                    pct = (
                        abs(val) / abs(explanation.total_contribution) * 100
                        if explanation.total_contribution != 0
                        else 0
                    )
                    print(f"    {op:35s} {val:+.4e} ({pct:.1f}%)")

                # Sanity checks
                shapley_sum = sum(explanation.shap_values.values())
                print(f"  Σφ = {shapley_sum:.6e}  (should ≈ {explanation.total_contribution:.6e})")

                efficiency_error = abs(
                    shapley_sum - explanation.total_contribution
                )
                if explanation.total_contribution != 0:
                    rel_error = efficiency_error / abs(explanation.total_contribution)
                    if rel_error > 0.05:
                        print(f"  ⚠ Efficiency violation: {rel_error:.2%}")

    # Save all results
    results_file = os.path.join(output_dir, "exact_shapley_results.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_file}")

    # Generate summary CSV if pandas available
    if HAS_PANDAS:
        _generate_summary_csv(all_results, output_dir)

    total_elapsed = time.time() - total_start
    print(f"\nTotal experiment time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")

    return all_results


def _generate_summary_csv(results: dict, output_dir: str):
    """Generate summary CSV from results."""
    rows = []
    for alg_name, funcs in results.items():
        for func_name, dims in funcs.items():
            for dim_str, data in dims.items():
                for op, val in data["shapley_values"].items():
                    ci = data["confidence_intervals"].get(op, [None, None])
                    rows.append({
                        "algorithm": alg_name,
                        "function": func_name,
                        "dimension": int(dim_str),
                        "operator": op,
                        "shapley_value": val,
                        "ci_lower": ci[0],
                        "ci_upper": ci[1],
                        "total_contribution": data["total_contribution"],
                        "pct_contribution": (
                            abs(val) / abs(data["total_contribution"]) * 100
                            if data["total_contribution"] != 0
                            else 0
                        ),
                    })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "shapley_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Summary CSV saved to {csv_path}")

    # Interaction summary
    int_rows = []
    for alg_name, funcs in results.items():
        for func_name, dims in funcs.items():
            for dim_str, data in dims.items():
                for pair_str, val in data.get("interactions", {}).items():
                    int_rows.append({
                        "algorithm": alg_name,
                        "function": func_name,
                        "dimension": int(dim_str),
                        "interaction": pair_str,
                        "value": val,
                    })

    if int_rows:
        df_int = pd.DataFrame(int_rows)
        int_path = os.path.join(output_dir, "interactions_summary.csv")
        df_int.to_csv(int_path, index=False)
        print(f"Interactions CSV saved to {int_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exact Shapley values for operator attribution"
    )
    parser.add_argument(
        "--output", default="results_exact_shapley",
        help="Output directory"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode (2 functions, D=10, 5 runs)"
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
