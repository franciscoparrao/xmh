"""Experiment 11 — Exact Shapley validation on the CEC2017 benchmark suite.

Replicates the GA and DE operator-attribution analysis (Section 5.1) on a
subset of CEC2017 functions to test whether the qualitative patterns
reported on the 9 classical functions (Sphere, Rastrigin, etc., used in
Experiments 1-4) hold on the more challenging CEC benchmark suite, which
features shift, rotation, and composition functions.

A subset of 10 representative CEC2017 functions covering the three
categories (unimodal, simple multimodal, hybrid) is used. The full
suite has 29 functions; the subset balances coverage with compute cost.

Outputs results_cec2017_shapley/cec2017_shapley_results.json with the
same structure as exact_shapley_results.json.
"""
from __future__ import annotations

import json
import os
import time
from typing import Callable

import numpy as np

from xmh.algorithms.instrumented_de import InstrumentedDE
from xmh.algorithms.instrumented_ga import InstrumentedGA
from xmh.explanation.operator_shap import ExactOperatorSHAP


# Representative CEC2017 subset:
# F1 (Bent Cigar) — unimodal, ill-conditioned
# F3 (Zakharov) — unimodal, non-separable
# F4 (Rosenbrock) — multimodal valley
# F5 (Rastrigin) — multimodal, separable
# F6 (Schaffer F7) — multimodal, non-separable
# F7 (Schwefel-like) — multimodal
# F9 (Levy) — multimodal
# F10 (Schwefel) — multimodal, deceptive
# F14 (Hybrid Function 1)
# F17 (Hybrid Function 4)
CEC_FUNCTIONS = [
    ('F1',  'F12017'),
    ('F3',  'F32017'),
    ('F4',  'F42017'),
    ('F5',  'F52017'),
    ('F6',  'F62017'),
    ('F7',  'F72017'),
    ('F9',  'F92017'),
    ('F10', 'F102017'),
    ('F14', 'F142017'),
    ('F17', 'F172017'),
]

ALGORITHMS = {
    'DE': {
        'class': InstrumentedDE,
        'kwargs': {'F': 0.5, 'CR': 0.9,
                   'mutation_strategy': 'rand/1',
                   'crossover_type': 'binomial'},
    },
    'GA': {
        'class': InstrumentedGA,
        'kwargs': {'selection_method': 'tournament',
                   'crossover_type': 'sbx',
                   'mutation_type': 'polynomial',
                   'crossover_prob': 0.9,
                   'tournament_size': 3},
    },
}

DIMENSIONS = [10, 30]
POP_SIZE = 50
MAX_GENERATIONS = 200
N_RUNS = 30
BASE_SEED = 1


def make_cec_benchmark(cec_class_name: str, dim: int) -> tuple[Callable, tuple]:
    """Return (objective_func, (lb_array, ub_array)) for a CEC2017 function."""
    from opfunu.cec_based import cec2017
    cls = getattr(cec2017, cec_class_name)
    f = cls(ndim=dim)
    lb = np.array(f.lb, dtype=float)
    ub = np.array(f.ub, dtype=float)
    # opfunu functions expect numpy arrays; pass through .evaluate.
    return f.evaluate, (lb, ub)


def main(outdir: str | None = None):
    if outdir is None:
        outdir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'results_cec2017_shapley',
        )
    os.makedirs(outdir, exist_ok=True)

    all_results: dict = {}
    t_start = time.time()

    for alg_name, alg_info in ALGORITHMS.items():
        alg_class = alg_info['class']
        alg_extra = alg_info['kwargs']
        all_results[alg_name] = {}
        for func_label, cec_cls in CEC_FUNCTIONS:
            all_results[alg_name][func_label] = {}
            for dim in DIMENSIONS:
                print(f"\n[{alg_name} {func_label} D={dim}]")
                fn, bounds = make_cec_benchmark(cec_cls, dim)
                alg_kwargs = dict(dim=dim, bounds=bounds,
                                  pop_size=POP_SIZE,
                                  max_generations=MAX_GENERATIONS,
                                  **alg_extra)
                exact = ExactOperatorSHAP(
                    n_runs_per_coalition=N_RUNS,
                    base_seed=BASE_SEED,
                    compute_interactions=False,
                )
                t0 = time.time()
                raw = exact.get_coalition_values(alg_class, alg_kwargs, fn)
                expl = exact.explain(alg_class, alg_kwargs, fn)
                elapsed = time.time() - t0

                entry = {
                    'algorithm': alg_name,
                    'function': func_label,
                    'cec_class': cec_cls,
                    'dimension': dim,
                    'operators': raw['operators'],
                    'shapley_values': dict(raw['shapley_values']),
                    'coalitions': {str(k): v for k, v in raw['coalitions'].items()},
                    'total_contribution': expl.total_contribution,
                    'base_value': expl.base_value,
                    'confidence_intervals': {
                        k: list(v) for k, v in expl.confidence_intervals.items()
                    },
                    'elapsed_seconds': elapsed,
                    'n_runs': N_RUNS,
                }
                all_results[alg_name][func_label][str(dim)] = entry

                print(f"  elapsed: {elapsed:.1f}s")
                total = abs(expl.total_contribution)
                for op, val in sorted(expl.shap_values.items(),
                                       key=lambda x: abs(x[1]), reverse=True):
                    pct = (abs(val) / total * 100) if total > 0 else 0
                    print(f"    {op:30s} {val:+.3e}  ({pct:.1f}%)")

    out_path = os.path.join(outdir, 'cec2017_shapley_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nTotal: {(time.time()-t_start)/60:.1f} min")
    print(f"Saved {out_path}")


if __name__ == '__main__':
    main()
