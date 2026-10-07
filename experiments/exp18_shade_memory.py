"""Experiment 18 — SHADE with the published memory size (H = |P|).

Experiment 5 runs SHADE with h_memory_size = 5. SHADE as published (Tanabe &
Fukunaga 2013) uses a success-history memory of size |P|, the population size.
A reviewer can reasonably object that a result about "SHADE's success-history
mechanism" measured with a 5-entry memory is a result about a different
configuration, since the memory is precisely the mechanism under test.

This experiment repeats the Experiment 5 grid with h_memory_size = pop_size and
everything else held fixed, so the two runs differ in exactly one parameter.
JADE needs no counterpart: its p_best_rate = 0.05 and c = 0.1 already match the
published JADE, and it carries no history memory.

Grid: 9 functions x 3 dimensions x 2^4 = 16 coalitions x 30 runs per coalition,
pop = 50, gen = 200 (10,000 FES), base seed 1 — identical to exp10.
"""
from __future__ import annotations

import argparse, json, os, time

from xmh.algorithms.instrumented_shade import InstrumentedSHADE
from xmh.benchmarks.functions import get_benchmark
from xmh.core.trace_logger import InstrumentationLevel
from xmh.explanation.operator_shap import ExactCoalitionSHAP

FUNCTIONS = ["sphere", "rosenbrock", "rastrigin", "schwefel",
             "ackley", "griewank", "levy", "zakharov", "dixon_price"]
POP_SIZE, MAX_GENERATIONS, N_RUNS, BASE_SEED = 50, 200, 30, 1


class CapturingSHAP(ExactCoalitionSHAP):
    """Retiene los valores de coalicion por corrida que explain() ya evalua."""
    def _compute_shapley(self, operators, coalition_values):
        self.captured_coalitions = coalition_values
        return super()._compute_shapley(operators, coalition_values)


class LightSHADE(InstrumentedSHADE):
    """SHADE con el logger en LIGHT; equivalencia con STANDARD verificada (diff 0.0)."""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.logger.level = InstrumentationLevel.LIGHT


def run_cell(func_name, dim, outdir):
    b = get_benchmark(func_name)
    lo, hi = b.get_bounds(dim)
    kwargs = dict(dim=dim, bounds=(lo, hi), pop_size=POP_SIZE,
                  max_generations=MAX_GENERATIONS,
                  h_memory_size=POP_SIZE)      # <- H = |P|, el valor publicado
    exact = CapturingSHAP(n_runs_per_coalition=N_RUNS, base_seed=BASE_SEED,
                          compute_interactions=False)
    t0 = time.time()
    ex = exact.explain(LightSHADE, kwargs, b.function)
    elapsed = time.time() - t0
    shap = dict(ex.shap_values)
    den = sum(shap.values())
    entry = {
        "algorithm": "SHADE", "function": func_name, "dimension": dim,
        "pop_size": POP_SIZE, "max_generations": MAX_GENERATIONS,
        "fes_budget": POP_SIZE * MAX_GENERATIONS, "n_runs": N_RUNS,
        "base_seed": BASE_SEED, "h_memory_size": POP_SIZE,
        "p_best_rate": 0.1, "instrumentation_level": "LIGHT",
        "coalitions": {str(tuple(sorted(k))): v for k, v in exact.captured_coalitions.items()},
        "shapley_values": shap,
        "share_pct_signed_denominator": {k: (100.0*v/den if den else 0.0) for k, v in shap.items()},
        "confidence_intervals": {k: list(v) for k, v in ex.confidence_intervals.items()},
        "elapsed_seconds": elapsed,
    }
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"shade_H{POP_SIZE}_{func_name}_D{dim}.json"), "w") as f:
        json.dump(entry, f, indent=2, default=float)
    return entry


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--func", required=True); p.add_argument("--dim", type=int, required=True)
    p.add_argument("--outdir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "results_shade_memory"))
    a = p.parse_args()
    e = run_cell(a.func, a.dim, a.outdir)
    ad = e["share_pct_signed_denominator"].get("parameter_adaptation")
    print(f"{a.func} D={a.dim} H={POP_SIZE} adaptation={ad:+.2f}%  ({e['elapsed_seconds']/60:.1f} min)")
