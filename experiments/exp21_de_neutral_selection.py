#!/usr/bin/env python3
"""
Experimento 21: los neutrales de DE bajo la Definicion 2 y la Tabla S1.

Asimetria detectada en la revision interna (2026-10). El codigo neutralizaba
dos operadores de DE de forma distinta a lo que el manuscrito declara:

  * seleccion greedy: el codigo retenia el target (ningun trial aceptado);
    la Definicion 2 exige, para operadores que *eligen*, muestrear uniformemente
    entre las salidas admisibles, que es lo que hace el neutral de SHADE/JADE
    para la misma seleccion (reemplazo aleatorio, p = 0.5);
  * cruce binomial: el codigo devolvia el target (descartaba el mutante);
    la Tabla S1 declara "mutant passes through unchanged", que es tambien el
    neutral de SHADE/JADE (trial <- mutante).

Con ambos neutrales inertes toda coalicion propia de DE valia v(S) = v(vacio) y
el axioma de simetria forzaba el reparto en tercios de la Seccion 5.1. Ese
reparto era una propiedad de la construccion, no de DE.

Este script recomputa las 27 instancias de DE (9 funciones x 3 dimensiones,
protocolo de results_exact_shapley: R = 30, seed base 42, pop 50, 100
generaciones, F = 0.5, CR = 0.9) bajo tres construcciones explicitas, cada una
con sus dos neutrales fijados en la subclase (independiente de lo que haga el
core en cada version):

  published    seleccion: target retenido      cruce: target      (v1.0-v1.5, JSON publicado)
  sel_uniform  seleccion: uniforme p = 0.5     cruce: target      (solo la seleccion corregida)
  table_s1     seleccion: uniforme p = 0.5     cruce: mutante     (Def. 2 + Tabla S1 = SHADE/JADE)

`table_s1` es la construccion adoptada desde v1.6 (core/instrumentation.py) y
la que merge_de_table_s1.py vuelca en results_exact_shapley. Las coaliciones
con seleccion y cruce activos no tocan ningun neutral y deben reproducir bit a
bit el JSON publicado: es el control.

Uso:  python3 exp21_de_neutral_selection.py [--variants sel_uniform,table_s1] [--smoke]
Salida: results_neutral_sensitivity/de_neutral_variants.json
"""
import argparse, json, sys, time
from itertools import combinations
from math import factorial
from multiprocessing import Pool
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from xmh.algorithms.instrumented_de import InstrumentedDE
from xmh.benchmarks.functions import get_benchmark
from xmh.core.instrumentation import Operator
from xmh.explanation.operator_shap import ExactCoalitionSHAP

PUBLISHED = ROOT / "results_exact_shapley" / "exact_shapley_results.json"
OUT = ROOT / "results_neutral_sensitivity" / "de_neutral_variants.json"
N_RUNS, BASE_SEED, POP, GEN = 30, 42, 50, 100
DE_KW = dict(pop_size=POP, max_generations=GEN, F=0.5, CR=0.9,
             mutation_strategy="rand/1", crossover_type="binomial")
OPS = ["mutation_rand/1", "crossover_binomial", "selection_greedy"]


# ---------- neutrales explicitos ----------
def sel_retain_target(target=None, target_fitness=None, trial=None, trial_fitness=None,
                      rng=None, **kw):
    """Ningun trial se acepta (construccion publicada hasta v1.5)."""
    return target.copy(), target_fitness


def sel_uniform(target=None, target_fitness=None, trial=None, trial_fitness=None,
                rng=None, **kw):
    """Uniforme sobre {target, trial}: Definicion 2, identico al neutral de SHADE."""
    if rng.random() < 0.5:
        return trial, trial_fitness
    return target, target_fitness


def cx_target(target=None, mutant=None, **kw):
    """Devuelve el target: descarta el mutante (codigo hasta v1.5)."""
    return target.copy()


def cx_mutant(target=None, mutant=None, **kw):
    """El mutante pasa sin cambios (Tabla S1, SHADE/JADE)."""
    return mutant.copy()


VARIANTS = {
    "published":   dict(selection=sel_retain_target, crossover=cx_target),
    "sel_uniform": dict(selection=sel_uniform,       crossover=cx_target),
    "table_s1":    dict(selection=sel_uniform,       crossover=cx_mutant),
}


def make_class(variant):
    neutrals = VARIANTS[variant]

    class _DE(InstrumentedDE):
        def disable_operator(self, name):
            if name == "selection_greedy":
                self.operators[name] = Operator(
                    name=f"neutral_selection_{variant}", operator_type="selection",
                    function=neutrals["selection"], parameters={})
            elif name == "crossover_binomial":
                self.operators[name] = Operator(
                    name=f"neutral_crossover_{variant}", operator_type="crossover",
                    function=neutrals["crossover"], parameters={})
            else:
                super().disable_operator(name)
    _DE.__name__ = f"DE_{variant}"
    return _DE


# ---------- Shapley per-run (misma convencion que recompute_de_selection.py) ----------
def shapley_from(coal, ops):
    n = len(ops); R = len(next(iter(coal.values())))
    cv = {frozenset(k): np.asarray(v, float) for k, v in coal.items()}
    per = {o: [] for o in ops}; per_share = {o: [] for o in ops}
    for r in range(R):
        phis = []
        for i in range(n):
            phi = 0.0; others = [j for j in range(n) if j != i]
            for size in range(n):
                for sub in combinations(others, size):
                    S = frozenset(sub); Si = S | {i}
                    w = factorial(len(S)) * factorial(n - len(S) - 1) / factorial(n)
                    phi += w * (cv[S][r] - cv[Si][r])
            phis.append(phi)
        tot = sum(phis)
        for i, o in enumerate(ops):
            per[o].append(phis[i])
            per_share[o].append(100 * phis[i] / tot if tot else 0.0)
    shap = {o: float(np.mean(v)) for o, v in per.items()}
    ci = {o: float(1.96 * np.std(per_share[o], ddof=1) / np.sqrt(R)) for o in ops}
    share = {o: float(np.mean(per_share[o])) for o in ops}
    return shap, share, ci


def run_instance(args):
    variant, func_name, dim = args
    cls = make_class(variant)
    b = get_benchmark(func_name); lo, hi = b.get_bounds(dim)
    kw = dict(dim=dim, bounds=(lo, hi), **DE_KW)
    es = ExactCoalitionSHAP(n_runs_per_coalition=N_RUNS, base_seed=BASE_SEED,
                            compute_interactions=False)
    t0 = time.time()
    coal = {}
    for size in range(len(OPS) + 1):
        for sub in combinations(range(len(OPS)), size):
            coal[tuple(sorted(sub))] = es._evaluate_coalition(
                cls, kw, b.function, [OPS[i] for i in sub], OPS)
    shap, share, ci = shapley_from(coal, OPS)
    return variant, func_name, str(dim), dict(
        coalitions={str(k): list(map(float, v)) for k, v in coal.items()},
        shapley_values=shap, shapley_shares_per_run=share, shares_ci95=ci,
        base_value=float(np.mean(coal[()])),
        total_contribution=float(np.mean(coal[()]) - np.mean(coal[(0, 1, 2)])),
        elapsed_seconds=round(time.time() - t0, 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="sel_uniform,table_s1")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    variants = a.variants.split(",")
    pub = json.load(open(PUBLISHED))["DE"]
    jobs = [(v, f, int(d)) for v in variants for f in pub for d in pub[f]]
    if a.smoke:
        jobs = [(v, "sphere", 10) for v in variants]
    print(f"{len(jobs)} instancias ({', '.join(variants)}), {8 * N_RUNS} runs cada una", flush=True)
    t0 = time.time()
    results = json.load(open(OUT))["results"] if OUT.exists() else {}
    with Pool(min(a.workers, len(jobs))) as pool:
        for v, f, d, rec in pool.imap_unordered(run_instance, jobs):
            results.setdefault(v, {}).setdefault(f, {})[d] = rec
            old = pub[f][d]
            # control: con seleccion y cruce activos no interviene ningun neutral
            ctrl_ok = all(np.array_equal(np.asarray(rec["coalitions"][k]),
                                         np.asarray(old["coalitions"][k]))
                          for k in ("(1, 2)", "(0, 1, 2)"))
            rec["control_identical_to_published"] = bool(ctrl_ok)
            rec["published_shares"] = old["shapley_shares_per_run"]
            sh = rec["shapley_shares_per_run"]
            print(f"  {v:12} {f:12} D={d:>2}  " +
                  "  ".join(f"{o.split('_')[0][:4]}={sh[o]:7.2f}%" for o in OPS) +
                  f"   control={'OK' if ctrl_ok else 'DIFIERE'}   ({time.time() - t0:.0f}s)",
                  flush=True)
            meta = dict(n_runs=N_RUNS, base_seed=BASE_SEED, pop=POP, gen=GEN, de_kwargs=DE_KW,
                        operators=OPS,
                        variants={k: {o: fn.__name__ for o, fn in v_.items()}
                                  for k, v_ in VARIANTS.items()},
                        adopted="table_s1", elapsed_s=round(time.time() - t0, 1))
            OUT.parent.mkdir(exist_ok=True)
            json.dump({"meta": meta, "results": results}, open(OUT, "w"), indent=1)

    print("\n=== shares medios por dimension ===")
    for v in results:
        for d in ("10", "30", "50"):
            rows = [results[v][f][d] for f in results[v] if d in results[v][f]]
            if not rows: continue
            m = {o: np.mean([r["shapley_shares_per_run"][o] for r in rows]) for o in OPS}
            print(f"  {v:12} D={d}: " + "  ".join(f"{o.split('_')[0][:4]}={m[o]:6.1f}%" for o in OPS)
                  + f"  (n={len(rows)})")
    print(f"\nlisto -> {OUT}")
