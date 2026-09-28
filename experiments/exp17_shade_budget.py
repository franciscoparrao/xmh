"""Experiment 17 — Budget sensitivity of SHADE's parameter-adaptation attribution.

Experiment 9 sweeps the evaluation budget for GA only. The manuscript's most
exposed empirical claim, that SHADE's history-based parameter adaptation carries
a negative Shapley share, is measured at a single budget (pop=50, gen=200, i.e.
10,000 FES). Adaptive parameter control is a long-horizon mechanism, so a
reviewer can reasonably object that the measurement was taken in the wrong
regime.

This experiment answers that objection directly: four functions at D=30, each at
gen in {200, 500, 1000, 2000} (10k to 100k FES; the 10,000*D convention would be
300k at D=30), with everything else held at the exp10 protocol. Functions are
chosen so that the baseline adaptation share has both signs: ackley and griewank
are negative at gen=200, rastrigin and schwefel positive.

One (function, budget) cell per process; the driver fans them out. Output goes to
results_shade_budget/shade_budget_<func>_g<gens>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import time

from xmh.algorithms.instrumented_shade import InstrumentedSHADE
from xmh.benchmarks.functions import get_benchmark
from xmh.core.trace_logger import InstrumentationLevel
from xmh.explanation.operator_shap import ExactCoalitionSHAP


class CapturingSHAP(ExactCoalitionSHAP):
    """ExactCoalitionSHAP que retiene los valores de coalicion por corrida.

    explain() ya los evalua internamente y los pasa a _compute_shapley; aqui solo
    se interceptan para archivarlos. Costo cero: no se reevalua ninguna coalicion,
    y los valores de Shapley y los intervalos salen del mismo camino de codigo que
    en exp10, no de una reimplementacion.
    """

    def _compute_shapley(self, operators, coalition_values):
        self.captured_coalitions = coalition_values
        return super()._compute_shapley(operators, coalition_values)


class LightSHADE(InstrumentedSHADE):
    """SHADE identico, con el TraceLogger en LIGHT.

    El juego de coaliciones solo consume el fitness final de cada corrida; los
    eventos por aplicacion de operador que STANDARD acumula (~4 por individuo por
    generacion) no los lee nadie en este camino y a gen=2000 agotan la RAM.
    Verificado contra STANDARD en ackley D=30 gen=200: los valores de Shapley
    coinciden exactamente (diferencia 0.0) y el codigo es determinista entre
    corridas con el mismo seed.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger.level = InstrumentationLevel.LIGHT


DIM = 30
POP_SIZE = 50
N_RUNS = 30
BASE_SEED = 1  # same as exp10, so gen=200 reproduces the published cell


def run_cell(func_name: str, gens: int, outdir: str) -> dict:
    benchmark = get_benchmark(func_name)
    lower, upper = benchmark.get_bounds(DIM)
    alg_kwargs = dict(
        dim=DIM, bounds=(lower, upper),
        pop_size=POP_SIZE, max_generations=gens,
    )
    exact = CapturingSHAP(
        n_runs_per_coalition=N_RUNS,
        base_seed=BASE_SEED,
        compute_interactions=False,
    )
    t0 = time.time()
    explanation = exact.explain(LightSHADE, alg_kwargs, benchmark.function)
    elapsed = time.time() - t0

    shap = dict(explanation.shap_values)
    den_abs = sum(abs(v) for v in shap.values())
    den_sgn = sum(shap.values())
    # El manuscrito normaliza por el denominador CON SIGNO (Seccion 5.1); el de valores
    # absolutos se guarda porque es el que la Seccion 5.5 usa para CMA-ES. Nombrar ambos
    # evita que se confundan: difieren hasta 0.7 pp cuando un operador es negativo.
    share_sgn = {k: (100.0 * v / den_sgn if den_sgn != 0 else 0.0) for k, v in shap.items()}
    share_abs = {k: (100.0 * v / den_abs if den_abs > 0 else 0.0) for k, v in shap.items()}

    entry = {
        "algorithm": "SHADE",
        "function": func_name,
        "dimension": DIM,
        "pop_size": POP_SIZE,
        "max_generations": gens,
        "fes_budget": POP_SIZE * gens,
        "n_runs": N_RUNS,
        "base_seed": BASE_SEED,
        "instrumentation_level": "LIGHT",
        # valores crudos de coalicion, una lista de R corridas por coalicion, con la
        # misma serializacion de claves que exp10 para que el parsing sea comun
        "coalitions": {str(tuple(sorted(k))): v
                       for k, v in exact.captured_coalitions.items()},
        "shapley_values": shap,
        "share_pct_signed_denominator": share_sgn,
        "share_pct_abs_denominator": share_abs,
        "confidence_intervals": {k: list(v) for k, v in explanation.confidence_intervals.items()},
        "total_contribution": explanation.total_contribution,
        "base_value": explanation.base_value,
        "elapsed_seconds": elapsed,
    }
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"shade_budget_{func_name}_g{gens}.json")
    with open(path, "w") as f:
        json.dump(entry, f, indent=2, default=float)
    return entry


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--func", required=True)
    p.add_argument("--gens", type=int, required=True)
    p.add_argument("--outdir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "results_shade_budget"))
    a = p.parse_args()
    e = run_cell(a.func, a.gens, a.outdir)
    ad = e["share_pct_signed_denominator"].get("parameter_adaptation")
    print(f"{a.func} g={a.gens} FES={e['fes_budget']} "
          f"adaptation={ad:+.2f}%  ({e['elapsed_seconds']/60:.1f} min)")
