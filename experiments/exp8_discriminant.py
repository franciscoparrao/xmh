"""Experiment 8 — Discriminating XMH from fANOVA-style hyperparameter analysis.

Two-part experiment showing that XMH and fANOVA answer different questions:

Part A: Vary only the CATEGORICAL crossover operator; hyperparameters fixed.
    fANOVA on hyperparameters returns nothing (no variation).
    fANOVA on the categorical factor reports between-type variance (η²).
    XMH reports per-type within-run Shapley values for {sel, cx, mut}.

Part B: Fix the operator (SBX), vary the CONTINUOUS crossover_prob.
    fANOVA on cx_rate reports continuous sensitivity (R²).
    XMH per cx_rate shows the smooth modulation of the cx operator's role.

Combined: complementary tools, neither subsumes the other.
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


# --- Shared experimental setup -------------------------------------------
FUNCTION = "rastrigin"
DIMENSION = 30
POP_SIZE = 50
MAX_GENERATIONS = 200
N_RUNS_PER_COALITION = 30
BASE_SEED = 1
SELECTION_TYPE = "tournament"
MUTATION_TYPE = "polynomial"
MUTATION_PROB = 0.1
TOURNAMENT_SIZE = 3

# Part A: categorical
CX_TYPES = ["sbx", "blx_alpha", "arithmetic", "uniform", "two_point", "single_point"]
CX_RATE_FIXED = 0.9

# Part B: continuous
CX_RATES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
CX_TYPE_FIXED = "sbx"

OUTDIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "results_discriminant",
)
os.makedirs(OUTDIR, exist_ok=True)


def _ga_kwargs(cx_type: str, cx_rate: float, dim: int, bounds) -> dict:
    return {
        "dim": dim,
        "bounds": bounds,
        "pop_size": POP_SIZE,
        "max_generations": MAX_GENERATIONS,
        "selection_method": SELECTION_TYPE,
        "crossover_type": cx_type,
        "mutation_type": MUTATION_TYPE,
        "crossover_prob": cx_rate,
        "mutation_prob": MUTATION_PROB,
        "tournament_size": TOURNAMENT_SIZE,
    }


def _shapley_for_config(cx_type: str, cx_rate: float, bounds) -> dict[str, Any]:
    alg_kwargs = _ga_kwargs(cx_type, cx_rate, DIMENSION, bounds)
    benchmark = get_benchmark(FUNCTION)
    exact = ExactOperatorSHAP(
        n_runs_per_coalition=N_RUNS_PER_COALITION,
        base_seed=BASE_SEED,
        compute_interactions=False,
    )
    t0 = time.time()
    raw = exact.get_coalition_values(InstrumentedGA, alg_kwargs, benchmark.function)
    explanation = exact.explain(InstrumentedGA, alg_kwargs, benchmark.function)
    elapsed = time.time() - t0

    full_coalition_key = tuple(range(len(raw["operators"])))
    final_fitness_per_run = list(raw["coalitions"][full_coalition_key])

    return {
        "operators": raw["operators"],
        "shapley_values": dict(raw["shapley_values"]),
        "total_contribution": explanation.total_contribution,
        "base_value": explanation.base_value,
        "confidence_intervals": {
            k: list(v) for k, v in explanation.confidence_intervals.items()
        },
        "final_fitness_per_run": final_fitness_per_run,
        "elapsed_seconds": elapsed,
        "n_runs_per_coalition": N_RUNS_PER_COALITION,
    }


def _shapley_pct(result: dict) -> dict[str, float]:
    total = abs(result["total_contribution"])
    if total < 1e-15:
        return {op: 0.0 for op in result["operators"]}
    return {
        op: 100.0 * abs(val) / total
        for op, val in result["shapley_values"].items()
    }


# --- Part A ---------------------------------------------------------------
def run_part_a() -> dict:
    benchmark = get_benchmark(FUNCTION)
    lower, upper = benchmark.get_bounds(DIMENSION)
    bounds = (lower, upper)

    per_cx_type: dict[str, Any] = {}
    final_fitness_groups: list[list[float]] = []
    final_fitness_flat: list[float] = []
    group_labels: list[str] = []

    for cx_type in CX_TYPES:
        print(f"\n[Part A] cx_type = {cx_type}")
        result = _shapley_for_config(cx_type, CX_RATE_FIXED, bounds)
        per_cx_type[cx_type] = result
        ff = result["final_fitness_per_run"]
        final_fitness_groups.append(ff)
        final_fitness_flat.extend(ff)
        group_labels.extend([cx_type] * len(ff))
        pct = _shapley_pct(result)
        for op, p in pct.items():
            print(f"    {op:30s} φ = {p:5.2f}%")
        print(f"    final fitness mean = {np.mean(ff):.4e}  std = {np.std(ff):.4e}")
        print(f"    elapsed: {result['elapsed_seconds']:.1f}s")

    f_stat, p_value = stats.f_oneway(*final_fitness_groups)

    grand_mean = np.mean(final_fitness_flat)
    ss_between = sum(
        len(g) * (np.mean(g) - grand_mean) ** 2 for g in final_fitness_groups
    )
    ss_total = sum((x - grand_mean) ** 2 for x in final_fitness_flat)
    eta_squared = ss_between / ss_total if ss_total > 0 else 0.0

    return {
        "part": "A",
        "varied": "crossover_type (categorical)",
        "fixed": {
            "crossover_prob": CX_RATE_FIXED,
            "pop_size": POP_SIZE, "max_generations": MAX_GENERATIONS,
            "mutation_prob": MUTATION_PROB, "tournament_size": TOURNAMENT_SIZE,
            "mutation_type": MUTATION_TYPE, "selection_method": SELECTION_TYPE,
        },
        "function": FUNCTION,
        "dimension": DIMENSION,
        "per_cx_type": per_cx_type,
        "anova": {
            "f_statistic": float(f_stat),
            "p_value": float(p_value),
            "eta_squared": float(eta_squared),
            "interpretation": (
                "η² = fraction of final-fitness variance explained by crossover_type. "
                "Compare against XMH's per-cx_type Shapley (within-run attribution): "
                "different question, different answer."
            ),
        },
        "fanova_on_hyperparams": {
            "variance_explained": 0.0,
            "reason": (
                "No hyperparameter variation across runs (all hyperparams are fixed); "
                "fANOVA-on-hyperparameters has no variance to decompose. This is the "
                "central discriminating evidence: a tool that requires varying knobs "
                "is silent on the question 'which operator type works here'."
            ),
        },
    }


# --- Part B ---------------------------------------------------------------
def run_part_b() -> dict:
    benchmark = get_benchmark(FUNCTION)
    lower, upper = benchmark.get_bounds(DIMENSION)
    bounds = (lower, upper)

    per_cx_rate: dict[str, Any] = {}
    rates_flat: list[float] = []
    fitness_flat: list[float] = []

    for cx_rate in CX_RATES:
        print(f"\n[Part B] cx_rate = {cx_rate:.1f}")
        result = _shapley_for_config(CX_TYPE_FIXED, cx_rate, bounds)
        per_cx_rate[f"{cx_rate:.2f}"] = result
        ff = result["final_fitness_per_run"]
        rates_flat.extend([cx_rate] * len(ff))
        fitness_flat.extend(ff)
        pct = _shapley_pct(result)
        for op, p in pct.items():
            print(f"    {op:30s} φ = {p:5.2f}%")
        print(f"    final fitness mean = {np.mean(ff):.4e}  std = {np.std(ff):.4e}")
        print(f"    elapsed: {result['elapsed_seconds']:.1f}s")

    rates_arr = np.array(rates_flat)
    fit_arr = np.array(fitness_flat)
    spearman_r, spearman_p = stats.spearmanr(rates_arr, fit_arr)

    coeffs = np.polyfit(rates_arr, fit_arr, deg=2)
    fitted = np.polyval(coeffs, rates_arr)
    ss_res = float(np.sum((fit_arr - fitted) ** 2))
    ss_tot = float(np.sum((fit_arr - fit_arr.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    cx_op = next(op for op in next(iter(per_cx_rate.values()))["operators"] if op.startswith("crossover"))
    cx_shapley_pct_by_rate = {
        rate: _shapley_pct(res).get(cx_op, 0.0)
        for rate, res in per_cx_rate.items()
    }
    sorted_rates = sorted(cx_shapley_pct_by_rate.keys(), key=lambda s: float(s))
    pct_series = [cx_shapley_pct_by_rate[r] for r in sorted_rates]
    cx_pct_spearman, _ = stats.spearmanr(
        [float(r) for r in sorted_rates], pct_series
    )

    return {
        "part": "B",
        "varied": "crossover_prob (continuous, 0.1..0.9)",
        "fixed": {
            "crossover_type": CX_TYPE_FIXED, "pop_size": POP_SIZE,
            "max_generations": MAX_GENERATIONS, "mutation_prob": MUTATION_PROB,
            "tournament_size": TOURNAMENT_SIZE, "mutation_type": MUTATION_TYPE,
            "selection_method": SELECTION_TYPE,
        },
        "function": FUNCTION,
        "dimension": DIMENSION,
        "per_cx_rate": per_cx_rate,
        "fanova_on_cx_rate": {
            "spearman_rho_rate_vs_fitness": float(spearman_r),
            "spearman_p_value": float(spearman_p),
            "quadratic_r_squared": float(r_squared),
            "interpretation": (
                "R² = fraction of variance explained by a 2nd-degree polynomial in cx_rate. "
                "This is the 'sensitivity' fANOVA-style methods report for a continuous knob."
            ),
        },
        "xmh_cx_shapley_vs_rate": {
            "rates": [float(r) for r in sorted_rates],
            "cx_shapley_pct": pct_series,
            "spearman_rho_rate_vs_cx_pct": float(cx_pct_spearman),
            "interpretation": (
                "XMH's crossover Shapley % at each cx_rate. Smooth modulation expected "
                "(higher rate → operator applied more often), but the operator's identity "
                "is stable; no switching behavior."
            ),
        },
    }


def main():
    print("=" * 70)
    print("Experiment 8 — Discriminating XMH from fANOVA hyperparameter analysis")
    print("=" * 70)
    print(f"Function: {FUNCTION}  D={DIMENSION}  pop={POP_SIZE}  gen={MAX_GENERATIONS}")
    print(f"n_runs_per_coalition = {N_RUNS_PER_COALITION}")
    print()

    t0 = time.time()
    print("\n##### PART A — Vary categorical operator #####")
    part_a = run_part_a()
    out_a = os.path.join(OUTDIR, "exp8_part_a.json")
    with open(out_a, "w") as f:
        json.dump(part_a, f, indent=2)
    print(f"\n→ saved {out_a}")

    print("\n##### PART B — Vary continuous cx_rate #####")
    part_b = run_part_b()
    out_b = os.path.join(OUTDIR, "exp8_part_b.json")
    with open(out_b, "w") as f:
        json.dump(part_b, f, indent=2)
    print(f"\n→ saved {out_b}")

    elapsed = time.time() - t0
    print(f"\n{'='*70}\nTotal elapsed: {elapsed/60:.1f} min")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
