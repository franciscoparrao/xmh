#!/usr/bin/env python3
"""
Re-computo de GA tras el arreglo del RNG sin sembrar de NeutralSelectionOperator.

El bug hacia que toda coalicion de GA que neutraliza la seleccion corriera con
aleatoriedad no controlada (3 de las 8 coaliciones). DE y PSO no estan afectados
--- en DE la seleccion neutral toma una rama determinista y ningun otro operador
neutral tiene RNG propio --- de modo que este script:

  1. recomputa las 27 instancias de GA,
  2. verifica que una muestra de DE y PSO sigue reproduciendo bit a bit el JSON
     publicado, lo que confirma que el alcance del bug era solo GA,
  3. reescribe results_exact_shapley/exact_shapley_results.json conservando las
     secciones de DE y PSO intactas.
"""
import json, sys, time
from itertools import combinations
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from xmh.algorithms.instrumented_ga import InstrumentedGA
from xmh.algorithms.instrumented_de import InstrumentedDE
from xmh.algorithms.instrumented_pso import InstrumentedPSO
from xmh.benchmarks.functions import get_benchmark
from xmh.explanation.operator_shap import ExactCoalitionSHAP

RES = ROOT / "results_exact_shapley" / "exact_shapley_results.json"
N_RUNS, BASE_SEED, POP, GEN = 30, 42, 50, 100
GA_KW = dict(pop_size=POP, max_generations=GEN, selection_method="tournament",
             crossover_type="sbx", mutation_type="polynomial",
             crossover_prob=0.9, tournament_size=3)
CHECK = {  # muestra de control: deben reproducir exactamente
    "DE":  (InstrumentedDE,  dict(F=0.5, CR=0.9, mutation_strategy="rand/1", crossover_type="binomial")),
    "PSO": (InstrumentedPSO, dict(topology="gbest", velocity_strategy="constriction")),
}


def coalitions_for(cls, kw, func, ops):
    es = ExactCoalitionSHAP(n_runs_per_coalition=N_RUNS, base_seed=BASE_SEED,
                           compute_interactions=False)
    out = {}
    for size in range(len(ops) + 1):
        for sub in combinations(range(len(ops)), size):
            out[tuple(sorted(sub))] = es._evaluate_coalition(
                cls, kw, func, [ops[i] for i in sub], ops)
    return out


def shapley_from(coal, ops):
    """Shapley per-run y shares per-run, la convencion declarada en el paper."""
    from math import factorial
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
                    w = factorial(len(S))*factorial(n-len(S)-1)/factorial(n)
                    phi += w*(cv[S][r] - cv[Si][r])
            phis.append(phi)
        tot = sum(phis)
        for i, o in enumerate(ops):
            per[o].append(phis[i])
            per_share[o].append(100*phis[i]/tot if tot else 0.0)
    shap = {o: float(np.mean(v)) for o, v in per.items()}
    ci = {o: float(1.96*np.std(per_share[o], ddof=1)/np.sqrt(R)) for o in ops}
    share = {o: float(np.mean(per_share[o])) for o in ops}
    return shap, share, ci


if __name__ == "__main__":
    data = json.load(open(RES))
    t0 = time.time()

    print("=== control: DE y PSO deben reproducir bit a bit ===", flush=True)
    okc = badc = 0
    for alg, (cls, extra) in CHECK.items():
        for f in ("sphere", "rastrigin"):
            for dim in ("10",):
                rec = data[alg][f][dim]; ops = rec["operators"]
                b = get_benchmark(f); lo, hi = b.get_bounds(int(dim))
                kw = dict(dim=int(dim), bounds=(lo, hi), pop_size=POP,
                          max_generations=GEN, **extra)
                got = coalitions_for(cls, kw, b.function, ops)
                for k, v in got.items():
                    old = rec["coalitions"][str(k)]
                    if np.allclose(np.mean(v), np.mean(old), rtol=0, atol=1e-12): okc += 1
                    else:
                        badc += 1
                        print(f"  DIFIERE {alg} {f} D{dim} {k}: {np.mean(old):.8f} -> {np.mean(v):.8f}")
            print(f"  {alg} {f}: control hecho ({time.time()-t0:.0f}s)", flush=True)
    print(f"  coaliciones de control identicas: {okc}, distintas: {badc}")
    if badc:
        print("  ATENCION: el bug no estaba limitado a GA. Abortando.")
        sys.exit(1)

    print("\n=== recomputo de GA (27 instancias) ===", flush=True)
    funcs = list(data["GA"].keys())
    for f in funcs:
        for dim in data["GA"][f]:
            rec = data["GA"][f][dim]; ops = rec["operators"]
            b = get_benchmark(f); lo, hi = b.get_bounds(int(dim))
            kw = dict(dim=int(dim), bounds=(lo, hi), **GA_KW)
            t1 = time.time()
            coal = coalitions_for(InstrumentedGA, kw, b.function, ops)
            shap, share, ci = shapley_from(coal, ops)
            old_share = None
            rec["coalitions"] = {str(k): list(map(float, v)) for k, v in coal.items()}
            rec["shapley_values"] = shap
            rec["shapley_shares_per_run"] = share
            rec["shares_ci95"] = ci
            rec["base_value"] = float(np.mean(coal[()]))
            rec["total_contribution"] = float(np.mean(coal[()]) - np.mean(coal[tuple(range(len(ops)))]))
            rec["elapsed_seconds"] = round(time.time()-t1, 1)
            rec["recomputed_after_rng_fix"] = True
            print(f"  GA {f:12} D={dim:2}  " +
                  "  ".join(f"{o.split('_')[0][:4]}={share[o]:6.2f}%" for o in ops)
                  + f"   ({time.time()-t0:.0f}s)", flush=True)

    json.dump(data, open(RES, "w"), indent=2, default=str)
    print(f"\nlisto en {(time.time()-t0)/60:.1f} min -> {RES}")
