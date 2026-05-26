"""Experiment 12 — Exact Shapley for JADE (n = 4 operators, 16 coalitions).

Second n = 4 case used to corroborate the SHADE result (Experiment 6). JADE
[Zhang & Sanner, 2009] shares the first three operators with SHADE
(current-to-pbest mutation with archive, binomial crossover, greedy
selection) and differs only in the parameter-adaptation mechanism:
SHADE keeps a history memory; JADE keeps two scalar moving averages.

Comparing the per-operator Shapley breakdowns of JADE and SHADE on the same
benchmarks therefore isolates the contribution of the parameter-adaptation
mechanism choice (history-memory vs. Lehmer-mean) to the overall operator
attribution.
"""
from __future__ import annotations

import json
import os
import time

from xmh.algorithms.instrumented_jade import InstrumentedJADE
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
            "results_jade_shapley",
        )
    os.makedirs(outdir, exist_ok=True)

    all_results = {"JADE": {}}
    t_start = time.time()

    for func_name in FUNCTIONS:
        benchmark = get_benchmark(func_name)
        all_results["JADE"][func_name] = {}
        for dim in DIMENSIONS:
            print(f"\n[JADE {func_name} D={dim}]")
            lower, upper = benchmark.get_bounds(dim)
            bounds = (lower, upper)
            alg_kwargs = dict(dim=dim, bounds=bounds,
                              pop_size=POP_SIZE,
                              max_generations=MAX_GENERATIONS)
            exact = ExactOperatorSHAP(n_runs_per_coalition=N_RUNS,
                                      base_seed=BASE_SEED,
                                      compute_interactions=False)
            t0 = time.time()
            raw = exact.get_coalition_values(InstrumentedJADE, alg_kwargs,
                                             benchmark.function)
            explanation = exact.explain(InstrumentedJADE, alg_kwargs,
                                        benchmark.function)
            elapsed = time.time() - t0
            print(f"  elapsed: {elapsed:.1f}s")

            entry = {
                "algorithm": "JADE", "function": func_name, "dimension": dim,
                "operators": raw["operators"],
                "shapley_values": dict(raw["shapley_values"]),
                "coalitions": {str(k): v for k, v in raw["coalitions"].items()},
                "total_contribution": explanation.total_contribution,
                "base_value": explanation.base_value,
                "confidence_intervals": {
                    k: list(v) for k, v in explanation.confidence_intervals.items()
                },
                "elapsed_seconds": elapsed, "n_runs": N_RUNS,
            }
            all_results["JADE"][func_name][str(dim)] = entry

            total = abs(explanation.total_contribution)
            for op, val in sorted(explanation.shap_values.items(),
                                  key=lambda x: abs(x[1]), reverse=True):
                pct = (abs(val) / total * 100) if total > 0 else 0
                print(f"    {op:35s} {val:+.4e}  ({pct:.1f}%)")

    out_path = os.path.join(outdir, "jade_shapley_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nTotal: {(time.time()-t_start)/60:.1f} min")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
