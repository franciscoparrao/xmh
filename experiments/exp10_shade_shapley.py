"""Experiment 10 — Exact Shapley for SHADE (n = 4 operators, 16 coalitions).

Demonstrates that the cooperative-game framework remains tractable at n = 4,
the upper end of the n <= 4 regime claimed in the paper. SHADE (Tanabe &
Fukunaga 2013) is the natural test case because its history-based parameter
adaptation is a distinct, semantically meaningful "operator" beyond the
mutation/crossover/selection triplet of vanilla DE.

Setup mirrors exp4 (the GA/DE/PSO exact-Shapley experiment): 9 benchmark
functions × 3 dimensions × 30 runs per coalition × 2^4 = 16 coalitions.

Outputs results_shade_shapley/shade_shapley_results.json with the same shape
as exact_shapley_results.json, so downstream figure scripts and statistical
analyses extend without modification.
"""
from __future__ import annotations

import json
import os
import time

from xmh.algorithms.instrumented_shade import InstrumentedSHADE
from xmh.benchmarks.functions import get_benchmark
from xmh.explanation.operator_shap import ExactOperatorSHAP


FUNCTIONS = [
    "sphere", "rosenbrock", "rastrigin", "schwefel",
    "ackley", "griewank", "levy", "zakharov", "dixon_price",
]
DIMENSIONS = [10, 30, 50]
POP_SIZE = 50
MAX_GENERATIONS = 200
N_RUNS = 30
BASE_SEED = 1


def main(outdir: str | None = None):
    if outdir is None:
        outdir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "results_shade_shapley",
        )
    os.makedirs(outdir, exist_ok=True)

    all_results = {"SHADE": {}}
    t_start = time.time()

    for func_name in FUNCTIONS:
        benchmark = get_benchmark(func_name)
        all_results["SHADE"][func_name] = {}
        for dim in DIMENSIONS:
            print(f"\n[SHADE {func_name} D={dim}]")
            lower, upper = benchmark.get_bounds(dim)
            bounds = (lower, upper)
            alg_kwargs = dict(
                dim=dim, bounds=bounds,
                pop_size=POP_SIZE, max_generations=MAX_GENERATIONS,
            )
            exact = ExactOperatorSHAP(
                n_runs_per_coalition=N_RUNS,
                base_seed=BASE_SEED,
                compute_interactions=True,
            )
            t0 = time.time()
            raw = exact.get_coalition_values(
                InstrumentedSHADE, alg_kwargs, benchmark.function,
            )
            explanation = exact.explain(
                InstrumentedSHADE, alg_kwargs, benchmark.function,
            )
            elapsed = time.time() - t0
            print(f"  elapsed: {elapsed:.1f}s")

            interactions = {}
            if raw.get("interactions"):
                interactions = {
                    f"{k[0]}_x_{k[1]}": v for k, v in raw["interactions"].items()
                }

            entry = {
                "algorithm": "SHADE",
                "function": func_name,
                "dimension": dim,
                "operators": raw["operators"],
                "shapley_values": dict(raw["shapley_values"]),
                "interactions": interactions,
                "coalitions": {str(k): v for k, v in raw["coalitions"].items()},
                "total_contribution": explanation.total_contribution,
                "base_value": explanation.base_value,
                "confidence_intervals": {
                    k: list(v) for k, v in explanation.confidence_intervals.items()
                },
                "elapsed_seconds": elapsed,
                "n_runs": N_RUNS,
            }
            all_results["SHADE"][func_name][str(dim)] = entry

            # Report
            total = abs(explanation.total_contribution)
            for op, val in sorted(explanation.shap_values.items(),
                                  key=lambda x: abs(x[1]), reverse=True):
                pct = (abs(val) / total * 100) if total > 0 else 0
                print(f"    {op:35s} {val:+.4e}  ({pct:.1f}%)")

    out_path = os.path.join(outdir, "shade_shapley_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nTotal elapsed: {(time.time() - t_start) / 60:.1f} min")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
