"""
Experiment 5: Comparison of Attribution Methods

Compares four attribution methods against Exact Shapley (ground truth):
  1. Exact Shapley (ground truth)
  2. QuickSHAP (fast normalised proxy from a single run)
  3. KernelSHAP (weighted linear regression approximation)
  4. Tracking Attribution (EvoMapX-style direct tracking)

Metrics:
  - Spearman ρ (rank correlation)
  - Pearson r (linear correlation)
  - Rank agreement (% of matching top-operator)
  - RMSE (normalised)

Reports by algorithm and by function.
"""

import argparse
import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats as sp_stats

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from xmh.algorithms.instrumented_de import InstrumentedDE
from xmh.algorithms.instrumented_ga import InstrumentedGA
from xmh.algorithms.instrumented_pso import InstrumentedPSO
from xmh.benchmarks.functions import get_benchmark, list_benchmarks
from xmh.explanation.operator_shap import ExactOperatorSHAP, QuickSHAP, KernelSHAP
from xmh.explanation.tracking_attribution import TrackingAttribution

HAS_PANDAS = False


# ── Configuration ───────────────────────────────────────────────────

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
N_EXACT_RUNS = 30
BASE_SEED = 42


def _normalise(values: Dict[str, float]) -> Dict[str, float]:
    """Normalise values so their absolute sum equals 1."""
    total = sum(abs(v) for v in values.values())
    if total == 0:
        n = len(values)
        return {k: 1.0 / n for k in values}
    return {k: v / total for k, v in values.items()}


def _compare_dicts(
    exact: Dict[str, float],
    approx: Dict[str, float],
) -> Dict[str, float]:
    """Compare two attribution dictionaries on shared keys."""
    shared = sorted(set(exact) & set(approx))
    if len(shared) < 2:
        return {
            "spearman": float("nan"),
            "pearson": float("nan"),
            "rank_agreement": float("nan"),
            "rmse": float("nan"),
        }

    exact_vals = np.array([exact[k] for k in shared])
    approx_vals = np.array([approx[k] for k in shared])

    # Spearman (handle constant vectors gracefully)
    if np.std(exact_vals) < 1e-15 or np.std(approx_vals) < 1e-15:
        rho = float("nan")
    else:
        rho, _ = sp_stats.spearmanr(exact_vals, approx_vals)
        if np.isnan(rho):
            rho = float("nan")

    # Pearson (handle constant vectors gracefully)
    if np.std(exact_vals) < 1e-15 or np.std(approx_vals) < 1e-15:
        r = float("nan")
    else:
        r, _ = sp_stats.pearsonr(exact_vals, approx_vals)
        if np.isnan(r):
            r = float("nan")

    # Rank agreement (top operator matches)
    exact_rank = sorted(shared, key=lambda k: abs(exact[k]), reverse=True)
    approx_rank = sorted(shared, key=lambda k: abs(approx[k]), reverse=True)
    rank_agreement = 1.0 if exact_rank[0] == approx_rank[0] else 0.0

    # Normalised RMSE
    norm_exact = _normalise(exact)
    norm_approx = _normalise(approx)
    diffs = [norm_exact.get(k, 0) - norm_approx.get(k, 0) for k in shared]
    rmse = np.sqrt(np.mean(np.array(diffs) ** 2))

    # Cosine similarity (robust for small n)
    dot = np.dot(exact_vals, approx_vals)
    norm_e = np.linalg.norm(exact_vals)
    norm_a = np.linalg.norm(approx_vals)
    cosine = float(dot / (norm_e * norm_a)) if norm_e > 0 and norm_a > 0 else float("nan")

    return {
        "spearman": float(rho),
        "pearson": float(r),
        "cosine": cosine,
        "rank_agreement": rank_agreement,
        "rmse": float(rmse),
    }


def run_experiment(
    output_dir: str = "results_method_comparison",
    quick: bool = False,
):
    """Run the method comparison experiment."""
    os.makedirs(output_dir, exist_ok=True)

    if quick:
        functions = ["sphere", "rastrigin"]
        dimensions = [10]
        n_exact_runs = 5
    else:
        functions = list_benchmarks()
        dimensions = [10, 30]
        n_exact_runs = N_EXACT_RUNS

    all_results = []
    total_start = time.time()

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
                alg_kwargs = {
                    "dim": dim,
                    "bounds": bounds,
                    "pop_size": POP_SIZE,
                    "max_generations": MAX_GENERATIONS,
                    **alg_extra,
                }

                # 1) Exact Shapley (ground truth)
                t0 = time.time()
                exact_shap = ExactOperatorSHAP(
                    n_runs_per_coalition=n_exact_runs,
                    base_seed=BASE_SEED,
                    compute_interactions=False,
                )
                exact_result = exact_shap.explain(
                    alg_class, alg_kwargs, benchmark.function,
                )
                exact_time = time.time() - t0
                exact_values = exact_result.shap_values

                # 2) QuickSHAP (from a single run)
                t0 = time.time()
                alg_instance = alg_class(**{**alg_kwargs, "seed": BASE_SEED})
                run_result = alg_instance.run(
                    benchmark.function, verbose=False,
                )
                quick_shap = QuickSHAP()
                quick_result = quick_shap.explain_from_trace(run_result["trace"])
                quick_time = time.time() - t0
                quick_values = quick_result.shap_values

                # 3) KernelSHAP
                t0 = time.time()
                kernel_shap = KernelSHAP(
                    n_samples=100,
                    n_runs_per_coalition=3,
                )
                kernel_result = kernel_shap.explain(
                    alg_class, alg_kwargs, benchmark.function,
                )
                kernel_time = time.time() - t0
                kernel_values = kernel_result.shap_values

                # 4) Tracking Attribution
                t0 = time.time()
                tracker = TrackingAttribution()
                tracking_result = tracker.explain_from_trace(run_result["trace"])
                tracking_time = time.time() - t0
                tracking_values = tracking_result.attributions

                # Compare each method against exact
                methods = {
                    "QuickSHAP": (quick_values, quick_time),
                    "KernelSHAP": (kernel_values, kernel_time),
                    "Tracking": (tracking_values, tracking_time),
                }

                for method_name, (method_values, method_time) in methods.items():
                    metrics = _compare_dicts(exact_values, method_values)
                    row = {
                        "algorithm": alg_name,
                        "function": func_name,
                        "dimension": dim,
                        "method": method_name,
                        "spearman": metrics["spearman"],
                        "pearson": metrics["pearson"],
                        "cosine": metrics["cosine"],
                        "rank_agreement": metrics["rank_agreement"],
                        "rmse": metrics["rmse"],
                        "exact_time_s": exact_time,
                        "method_time_s": method_time,
                        "speedup": exact_time / method_time if method_time > 0 else float("inf"),
                    }
                    all_results.append(row)

                    rho_str = f"{metrics['spearman']:+.3f}" if not np.isnan(metrics['spearman']) else "  N/A"
                    r_str = f"{metrics['pearson']:+.3f}" if not np.isnan(metrics['pearson']) else "  N/A"
                    print(
                        f"  {method_name:12s}  ρ={rho_str}  "
                        f"r={r_str}  "
                        f"cos={metrics['cosine']:+.3f}  "
                        f"rank={metrics['rank_agreement']:.0f}  "
                        f"RMSE={metrics['rmse']:.4f}  "
                        f"time={method_time:.2f}s"
                    )

    # Save results
    results_file = os.path.join(output_dir, "method_comparison.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    if HAS_PANDAS:
        df = pd.DataFrame(all_results)
        csv_path = os.path.join(output_dir, "method_comparison.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nCSV saved to {csv_path}")

        # Summary tables
        print("\n" + "=" * 70)
        print("SUMMARY BY METHOD (mean across all configs)")
        print("=" * 70)
        summary = df.groupby("method")[
            ["spearman", "pearson", "cosine", "rank_agreement", "rmse", "speedup"]
        ].mean()
        print(summary.to_string())

        print("\n" + "=" * 70)
        print("SUMMARY BY ALGORITHM × METHOD")
        print("=" * 70)
        alg_summary = df.groupby(["algorithm", "method"])[
            ["spearman", "pearson", "cosine", "rank_agreement", "rmse"]
        ].mean()
        print(alg_summary.to_string())

    total_elapsed = time.time() - total_start
    print(f"\nTotal experiment time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare attribution methods against Exact Shapley"
    )
    parser.add_argument("--output", default="results_method_comparison")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    run_experiment(output_dir=args.output, quick=args.quick)
