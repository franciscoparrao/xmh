#!/usr/bin/env python3
"""
Experimento 13: sensibilidad de la atribucion al diseño del operador neutral.

Responde a los issues 3 de Reviewer 1 y 3 de Reviewer 3 (Information Sciences,
2026-08-21): las conclusiones dependen de como se construye el reemplazo neutral,
y esa eleccion nunca se sometio a analisis de sensibilidad.

Diseño: GA sobre Sphere y Rastrigin. Por cada componente se prueban DOS neutrales
alternativos ademas del usado en el paper, variando uno a la vez (one-at-a-time):

  seleccion   S0 uniforme aleatorio con reemplazo   <- el del paper
              S1 torneo con k=1 (misma ruta de codigo, sin presion de fitness)
              S2 permutacion aleatoria (sin reemplazo: elimina la deriva genetica)

  crossover   C0 pass-through, los padres no se cruzan   <- el del paper
              C1 SBX con eta_c = 1e6 (ruta real, hijos numericamente ~ padres)
              C2 intercambio de padres (recombinacion trivial, conserva el material)

  mutacion    M0 pass-through   <- el del paper
              M1 polinomial con eta_m = 1e6 (ruta real, perturbacion despreciable)
              M2 pass-through que consume el mismo RNG que el operador real
                 (controla el confound del stream de numeros aleatorios)

Salida: results_neutral_sensitivity/neutral_sensitivity.json
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from xmh.algorithms.instrumented_ga import InstrumentedGA
from xmh.benchmarks.functions import get_benchmark
from xmh.core.instrumentation import Operator
from xmh.explanation.operator_shap import ExactCoalitionSHAP

FUNCTIONS  = ["sphere", "rastrigin"]
DIMENSIONS = [10, 30]
N_RUNS     = 30
BASE_SEED  = 42
POP, GEN   = 50, 100
OPS = ["selection_tournament", "crossover_sbx", "mutation_polynomial"]
OUT = Path(__file__).resolve().parents[2] / "results_neutral_sensitivity"


# ---------- neutrales alternativos ----------
def _sel_tournament_k1(fitness=None, rng=None, **kw):
    """Torneo con k=1: la ruta real del operador, pero sin presion de fitness."""
    r = rng if rng is not None else np.random
    return int(r.integers(0, len(fitness)))

def _make_sel_permutation():
    """Muestreo sin reemplazo: mismo conjunto de padres, orden barajado."""
    state = {"perm": None, "i": 0}
    def f(fitness=None, rng=None, **kw):
        r = rng if rng is not None else np.random
        n = len(fitness)
        if state["perm"] is None or state["i"] >= n:
            state["perm"] = r.permutation(n); state["i"] = 0
        idx = int(state["perm"][state["i"]]); state["i"] += 1
        return idx
    return f

def _cx_sbx_flat(parent1=None, parent2=None, lower=None, upper=None, rng=None, **kw):
    """SBX con eta enorme: el kernel real, con hijos numericamente iguales a los padres."""
    r = rng if rng is not None else np.random
    eta = 1e6
    u = r.random(parent1.shape)
    beta = np.where(u <= 0.5, (2*u)**(1/(eta+1)), (1/(2*(1-u)))**(1/(eta+1)))
    c1 = 0.5*((1+beta)*parent1 + (1-beta)*parent2)
    c2 = 0.5*((1-beta)*parent1 + (1+beta)*parent2)
    if lower is not None:
        c1 = np.clip(c1, lower, upper); c2 = np.clip(c2, lower, upper)
    return c1, c2

def _cx_swap(parent1=None, parent2=None, **kw):
    """Intercambio de padres: recombinacion trivial que conserva el material genetico."""
    return parent2.copy(), parent1.copy()

def _mut_poly_flat(individual=None, prob=None, lower=None, upper=None, rng=None, **kw):
    """Mutacion polinomial con eta enorme: ruta real, perturbacion despreciable."""
    r = rng if rng is not None else np.random
    eta = 1e6
    x = individual.copy()
    mask = r.random(x.shape) < (prob if prob is not None else 0.0)
    if mask.any():
        u = r.random(x.shape)
        d = np.where(u < 0.5, (2*u)**(1/(eta+1)) - 1, 1 - (2*(1-u))**(1/(eta+1)))
        rng_span = (upper - lower) if lower is not None else 1.0
        x = np.where(mask, x + d*rng_span, x)
        if lower is not None:
            x = np.clip(x, lower, upper)
    return x

def _mut_passthrough_rng(individual=None, prob=None, rng=None, **kw):
    """Identidad, pero consumiendo el mismo numero de aleatorios que la mutacion real."""
    r = rng if rng is not None else np.random
    _ = r.random(individual.shape)
    _ = r.random(individual.shape)
    return individual.copy()


ALTS = {
    "selection_tournament": {
        "S1_tournament_k1":   ("selection", _sel_tournament_k1),
        "S2_permutation":     ("selection", None),   # fabrica con estado, ver abajo
    },
    "crossover_sbx": {
        "C1_sbx_eta_1e6":     ("crossover", _cx_sbx_flat),
        "C2_parent_swap":     ("crossover", _cx_swap),
    },
    "mutation_polynomial": {
        "M1_poly_eta_1e6":    ("mutation", _mut_poly_flat),
        "M2_passthrough_rng": ("mutation", _mut_passthrough_rng),
    },
}


def make_patched_class(target_op, variant_name):
    """Devuelve una subclase de InstrumentedGA cuyo disable_operator usa el neutral alternativo."""
    if variant_name is None:
        return InstrumentedGA
    op_type, func = ALTS[target_op][variant_name]

    class _GA(InstrumentedGA):
        def disable_operator(self, name):
            if name == target_op:
                f = _make_sel_permutation() if variant_name == "S2_permutation" else func
                self.operators[name] = Operator(
                    name=f"neutral_{variant_name}", operator_type=op_type,
                    function=f, parameters={}, description=f"neutral alternativo {variant_name}")
            else:
                super().disable_operator(name)
    return _GA


def shares_from(rec_values, ops):
    """Media de los shares por corrida, la convencion declarada en el paper."""
    from itertools import combinations
    from math import factorial
    n = len(ops); R = len(next(iter(rec_values.values())))
    per = {o: [] for o in ops}
    for r in range(R):
        phis = []
        for i in range(n):
            phi = 0.0; others = [j for j in range(n) if j != i]
            for size in range(n):
                for sub in combinations(others, size):
                    S = frozenset(sub); Si = S | {i}
                    w = factorial(len(S))*factorial(n-len(S)-1)/factorial(n)
                    phi += w*(rec_values[S][r] - rec_values[Si][r])
            phis.append(phi)
        tot = sum(phis)
        for i, o in enumerate(ops):
            per[o].append(100*phis[i]/tot if tot else 0.0)
    return {o: float(np.mean(v)) for o, v in per.items()}


def run_config(cls, func_name, dim):
    b = get_benchmark(func_name); lo, hi = b.get_bounds(dim)
    kw = dict(dim=dim, bounds=(lo, hi), pop_size=POP, max_generations=GEN,
              selection_method="tournament", crossover_type="sbx",
              mutation_type="polynomial", crossover_prob=0.9, tournament_size=3)
    es = ExactCoalitionSHAP(n_runs_per_coalition=N_RUNS, base_seed=BASE_SEED,
                           compute_interactions=False)
    from itertools import combinations
    vals = {}
    for size in range(len(OPS)+1):
        for sub in combinations(range(len(OPS)), size):
            active = [OPS[i] for i in sub]
            vals[frozenset(sub)] = es._evaluate_coalition(cls, kw, b.function, active, OPS)
    return shares_from(vals, OPS)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    configs = [("baseline", None, None)]
    for op, alts in ALTS.items():
        for name in alts:
            configs.append((name, op, name))

    results = {}
    t0 = time.time()
    total = len(configs)*len(FUNCTIONS)*len(DIMENSIONS)
    done = 0
    for cfg_name, target_op, variant in configs:
        cls = make_patched_class(target_op, variant) if target_op else InstrumentedGA
        results[cfg_name] = {}
        for f in FUNCTIONS:
            results[cfg_name][f] = {}
            for d in DIMENSIONS:
                sh = run_config(cls, f, d)
                results[cfg_name][f][str(d)] = sh
                done += 1
                print(f"[{done:2}/{total}] {cfg_name:20} {f:10} D={d:2}  "
                      + "  ".join(f"{k.split('_')[0]}={v:6.2f}%" for k, v in sh.items())
                      + f"   ({time.time()-t0:.0f}s)", flush=True)
    meta = dict(n_runs=N_RUNS, base_seed=BASE_SEED, pop=POP, gen=GEN,
                functions=FUNCTIONS, dimensions=DIMENSIONS, operators=OPS,
                total_runs=total*8*N_RUNS, elapsed_s=round(time.time()-t0, 1))
    json.dump({"meta": meta, "results": results},
              open(OUT / "neutral_sensitivity.json", "w"), indent=1)
    print(f"\nlisto en {meta['elapsed_s']}s -> {OUT/'neutral_sensitivity.json'}")
    print(f"corridas totales: {meta['total_runs']}")
