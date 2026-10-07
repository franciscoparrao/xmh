"""Experiment 19 — Dimension effect at constant FES per dimension.

Experiment 1 holds the evaluation budget fixed at 5,000 FES across D in
{10, 30, 50}, so FES per dimension falls 500 -> 167 -> 100. Experiment 9 then
shows that operator shares move monotonically with budget. The two manipulations
are therefore confounded in the main grid: the reported growth of GA selection
and PSO topology with dimension could be produced by the 5x reduction in
effective budget alone.

This experiment separates them. It re-runs GA and PSO with the budget scaled to
the dimension, FES = 1000 * D, so FES per dimension is constant at 1,000. If the
trends survive, they are dimension effects; if they vanish, they were budget
effects and the corresponding claims must be withdrawn.

Four functions spanning smooth and multimodal landscapes, including the two the
budget sweep of Experiment 9 used. Grid: 2 algorithms x 4 functions x 3
dimensions x 2^3 = 8 coalitions x 30 runs, base seed 42 as in exp4.
"""
from __future__ import annotations

import argparse, json, os, time

from xmh.algorithms.instrumented_ga import InstrumentedGA
from xmh.algorithms.instrumented_pso import InstrumentedPSO
from xmh.benchmarks.functions import get_benchmark
from xmh.core.trace_logger import InstrumentationLevel
from xmh.explanation.operator_shap import ExactCoalitionSHAP

FUNCTIONS = ["sphere", "rastrigin", "ackley", "griewank"]
DIMENSIONS = [10, 30, 50]
POP_SIZE, N_RUNS, BASE_SEED = 50, 30, 42
FES_PER_DIM = 1000                      # presupuesto que escala con la dimension

ALG = {
    "GA": (InstrumentedGA, dict(selection_method="tournament", crossover_type="sbx",
                                mutation_type="polynomial", crossover_prob=0.9,
                                tournament_size=3)),
    "PSO": (InstrumentedPSO, dict(topology="gbest", velocity_strategy="constriction")),
}


class CapturingSHAP(ExactCoalitionSHAP):
    def _compute_shapley(self, operators, coalition_values):
        self.captured_coalitions = coalition_values
        return super()._compute_shapley(operators, coalition_values)


def light(cls):
    class _L(cls):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.logger.level = InstrumentationLevel.LIGHT
    _L.__name__ = "Light" + cls.__name__
    return _L


def run_cell(alg_name, func_name, dim, outdir):
    cls, kw = ALG[alg_name]
    b = get_benchmark(func_name)
    lo, hi = b.get_bounds(dim)
    gens = (FES_PER_DIM * dim) // POP_SIZE          # FES = 1000*D a poblacion fija
    kwargs = dict(dim=dim, bounds=(lo, hi), pop_size=POP_SIZE,
                  max_generations=gens, **kw)
    exact = CapturingSHAP(n_runs_per_coalition=N_RUNS, base_seed=BASE_SEED,
                          compute_interactions=False)
    t0 = time.time()
    ex = exact.explain(light(cls), kwargs, b.function)
    elapsed = time.time() - t0
    shap = dict(ex.shap_values); den = sum(shap.values())
    entry = {
        "algorithm": alg_name, "function": func_name, "dimension": dim,
        "pop_size": POP_SIZE, "max_generations": gens,
        "fes_budget": POP_SIZE * gens, "fes_per_dimension": FES_PER_DIM,
        "n_runs": N_RUNS, "base_seed": BASE_SEED, "instrumentation_level": "LIGHT",
        "coalitions": {str(tuple(sorted(k))): v for k, v in exact.captured_coalitions.items()},
        "shapley_values": shap,
        "share_pct_signed_denominator": {k: (100.0*v/den if den else 0.0) for k, v in shap.items()},
        "confidence_intervals": {k: list(v) for k, v in ex.confidence_intervals.items()},
        "elapsed_seconds": elapsed,
    }
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"{alg_name}_{func_name}_D{dim}.json"), "w") as f:
        json.dump(entry, f, indent=2, default=float)
    return entry


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--alg", required=True); p.add_argument("--func", required=True)
    p.add_argument("--dim", type=int, required=True)
    p.add_argument("--outdir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "results_fes_per_dim"))
    a = p.parse_args()
    e = run_cell(a.alg, a.func, a.dim, a.outdir)
    key = "selection" if a.alg == "GA" else "topology"
    v = next((x for k, x in e["share_pct_signed_denominator"].items() if key in k), None)
    print(f"{a.alg} {a.func} D={a.dim} FES={e['fes_budget']} {key}={v:+.2f}%  ({e['elapsed_seconds']/60:.1f} min)")
