"""Experiment 9 — Budget sensitivity: how operator attribution changes with
population size and max generations. Reviewer-driven (R#1.2 of SWEVO).

Two sweeps:
  Sweep A: vary max_generations ∈ {25, 50, 100, 200, 500} with pop=50.
  Sweep B: vary pop_size ∈ {10, 25, 50, 100, 200} with max_generations=200.

For each (function, level) the script computes exact Shapley (8 coalitions × 30
runs) and a 30-run convergence pass with per-generation best-fitness traces.

Functions: Sphere (unimodal) and Rastrigin (multimodal) at D=30.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

import numpy as np
from scipy import stats

from xmh.algorithms.instrumented_ga import InstrumentedGA
from xmh.benchmarks.functions import get_benchmark
from xmh.explanation.operator_shap import ExactOperatorSHAP


# --- Fixed setup ----------------------------------------------------------
FUNCTIONS = ["sphere", "rastrigin"]
DIMENSION = 30
N_RUNS_PER_COALITION = 30
BASE_SEED = 1
SELECTION = "tournament"
CX_TYPE = "sbx"
MUT_TYPE = "polynomial"
CX_PROB = 0.9
MUT_PROB = 0.1
TOURN_K = 3

# Sweep A: vary max_gen
SWEEP_A_GENS = [25, 50, 100, 200, 500]
POP_FIXED = 50

# Sweep B: vary pop
SWEEP_B_POPS = [10, 25, 50, 100, 200]
GEN_FIXED = 200


def _ga_kwargs(pop, gen, bounds):
    return {
        "dim": DIMENSION, "bounds": bounds, "pop_size": pop,
        "max_generations": gen,
        "selection_method": SELECTION, "crossover_type": CX_TYPE,
        "mutation_type": MUT_TYPE, "crossover_prob": CX_PROB,
        "mutation_prob": MUT_PROB, "tournament_size": TOURN_K,
    }


def _shapley_for(func, pop, gen):
    benchmark = get_benchmark(func)
    lower, upper = benchmark.get_bounds(DIMENSION)
    bounds = (lower, upper)
    kwargs = _ga_kwargs(pop, gen, bounds)

    exact = ExactOperatorSHAP(
        n_runs_per_coalition=N_RUNS_PER_COALITION,
        base_seed=BASE_SEED, compute_interactions=False,
    )
    t0 = time.time()
    raw = exact.get_coalition_values(InstrumentedGA, kwargs, benchmark.function)
    explanation = exact.explain(InstrumentedGA, kwargs, benchmark.function)
    elapsed = time.time() - t0

    full_coal_key = tuple(range(len(raw["operators"])))
    final_fitness = list(raw["coalitions"][full_coal_key])

    return {
        "pop_size": pop, "max_generations": gen,
        "operators": raw["operators"],
        "shapley_values": dict(raw["shapley_values"]),
        "total_contribution": explanation.total_contribution,
        "base_value": explanation.base_value,
        "confidence_intervals": {
            k: list(v) for k, v in explanation.confidence_intervals.items()
        },
        "final_fitness_per_run": final_fitness,
        "elapsed_seconds": elapsed,
        "n_runs_per_coalition": N_RUNS_PER_COALITION,
    }


def _convergence_traces(func, pop, gen, n_runs=30):
    """Run vanilla GA n_runs times, return a (n_runs × gen) array of best-fitness."""
    benchmark = get_benchmark(func)
    lower, upper = benchmark.get_bounds(DIMENSION)
    bounds = (lower, upper)
    traces = []
    for seed in range(BASE_SEED, BASE_SEED + n_runs):
        ga = InstrumentedGA(seed=seed, **_ga_kwargs(pop, gen, bounds))
        ga.run(benchmark.function, verbose=False)
        snaps = ga.logger.snapshots
        # best_fitness per generation; snapshot 0 is the initial population
        bf = [s.best_fitness for s in snaps]
        # Pad / truncate to max_generations + 1 entries
        if len(bf) < gen + 1:
            bf = bf + [bf[-1]] * (gen + 1 - len(bf))
        traces.append(bf[: gen + 1])
    return traces


def _pct(case):
    total = sum(abs(v) for v in case["shapley_values"].values())
    if total < 1e-15:
        return {op: 0.0 for op in case["operators"]}
    return {op: 100 * abs(case["shapley_values"][op]) / total for op in case["operators"]}


def run_sweep_a(func: str) -> dict:
    """Vary max_generations with pop fixed at 50."""
    per_level = {}
    # Convergence: one long max_gen=500 trace, truncate for shorter levels.
    print(f"\n[Sweep A | {func}] long-trace pass (gen={max(SWEEP_A_GENS)}, pop={POP_FIXED})")
    t0 = time.time()
    long_traces = _convergence_traces(func, POP_FIXED, max(SWEEP_A_GENS), n_runs=30)
    print(f"    convergence traces: {time.time()-t0:.1f}s")

    for gen in SWEEP_A_GENS:
        print(f"\n[Sweep A | {func}] gen={gen} pop={POP_FIXED}")
        case = _shapley_for(func, POP_FIXED, gen)
        case["convergence_traces"] = [t[: gen + 1] for t in long_traces]
        per_level[str(gen)] = case
        pct = _pct(case)
        for op, p in pct.items():
            print(f"    {op:30s} φ = {p:5.2f}%")
        print(f"    fitness mean = {np.mean(case['final_fitness_per_run']):.3e}  "
              f"elapsed = {case['elapsed_seconds']:.1f}s")
    return per_level


def run_sweep_b(func: str) -> dict:
    """Vary pop_size with max_generations fixed at 200."""
    per_level = {}
    for pop in SWEEP_B_POPS:
        print(f"\n[Sweep B | {func}] pop={pop} gen={GEN_FIXED}")
        # Separate convergence pass per pop (dynamics differ)
        t0 = time.time()
        traces = _convergence_traces(func, pop, GEN_FIXED, n_runs=30)
        case = _shapley_for(func, pop, GEN_FIXED)
        case["convergence_traces"] = traces
        per_level[str(pop)] = case
        pct = _pct(case)
        for op, p in pct.items():
            print(f"    {op:30s} φ = {p:5.2f}%")
        print(f"    fitness mean = {np.mean(case['final_fitness_per_run']):.3e}  "
              f"shap_elapsed = {case['elapsed_seconds']:.1f}s  "
              f"total = {time.time()-t0:.1f}s")
    return per_level


def main(outdir: str | None = None):
    if outdir is None:
        outdir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "results_budget",
        )
    os.makedirs(outdir, exist_ok=True)

    print("=" * 70)
    print("Experiment 9 — Budget sensitivity (pop / max_gen sweep)")
    print("=" * 70)

    t_total = time.time()
    sweep_a, sweep_b = {}, {}
    for func in FUNCTIONS:
        sweep_a[func] = run_sweep_a(func)
        sweep_b[func] = run_sweep_b(func)

    # Save
    out_a = os.path.join(outdir, "exp9_sweep_a.json")
    with open(out_a, "w") as f:
        json.dump({
            "sweep": "A (max_gen, pop=50)",
            "fixed": {"pop_size": POP_FIXED, **{k: v for k, v in {
                "dim": DIMENSION, "selection": SELECTION, "cx_type": CX_TYPE,
                "mut_type": MUT_TYPE, "cx_prob": CX_PROB, "mut_prob": MUT_PROB,
                "tourn_k": TOURN_K,
            }.items()}},
            "levels": SWEEP_A_GENS,
            "functions": FUNCTIONS,
            "per_func": sweep_a,
        }, f, indent=2, default=float)
    print(f"\n→ saved {out_a}")

    out_b = os.path.join(outdir, "exp9_sweep_b.json")
    with open(out_b, "w") as f:
        json.dump({
            "sweep": "B (pop_size, max_gen=200)",
            "fixed": {"max_generations": GEN_FIXED, **{k: v for k, v in {
                "dim": DIMENSION, "selection": SELECTION, "cx_type": CX_TYPE,
                "mut_type": MUT_TYPE, "cx_prob": CX_PROB, "mut_prob": MUT_PROB,
                "tourn_k": TOURN_K,
            }.items()}},
            "levels": SWEEP_B_POPS,
            "functions": FUNCTIONS,
            "per_func": sweep_b,
        }, f, indent=2, default=float)
    print(f"→ saved {out_b}")
    print(f"\nTotal elapsed: {(time.time() - t_total) / 60:.1f} min")


if __name__ == "__main__":
    main()
