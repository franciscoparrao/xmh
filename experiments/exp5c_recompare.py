#!/usr/bin/env python3
"""
Recalculo de la Tabla 2 con la metrica corregida.

`_compare_dicts` comparaba metodo exacto y aproximado solo sobre las claves
comunes. Como la traza de ejecucion no registra la seleccion greedy de DE ni el
position update de PSO, esos operadores desaparecian de la comparacion y con
ellos el lugar donde los metodos de una sola corrida fallan. En DE quedaba
comparando [33.3, 33.3] contra [50, 50] --- vectores paralelos, coseno 1.00 ---
pese a que QuickSHAP no habia visto un tercio de la atribucion.

Aqui los operadores ausentes se tratan como atribucion cero, que es lo que esos
metodos efectivamente predicen. No se vuelve a correr el metodo exacto: se leen
sus valores de results_exact_shapley y solo se genera una traza por instancia
para QuickSHAP y Tracking.

Salida: results_method_comparison/method_comparison_fullset.json
"""
import json, re, sys, time
from itertools import combinations
from math import factorial
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from xmh.algorithms.instrumented_de import InstrumentedDE
from xmh.algorithms.instrumented_ga import InstrumentedGA
from xmh.algorithms.instrumented_pso import InstrumentedPSO
from xmh.benchmarks.functions import get_benchmark
from xmh.explanation.operator_shap import QuickSHAP
from xmh.explanation.tracking_attribution import TrackingAttribution

CFG = {
    "DE":  (InstrumentedDE,  dict(F=0.5, CR=0.9, mutation_strategy="rand/1",
                                  crossover_type="binomial")),
    "GA":  (InstrumentedGA,  dict(selection_method="tournament", crossover_type="sbx",
                                  mutation_type="polynomial", crossover_prob=0.9,
                                  tournament_size=3)),
    "PSO": (InstrumentedPSO, dict(topology="gbest", velocity_strategy="constriction")),
}
POP, GEN, SEED = 50, 100, 42


def exact_from(rec):
    """Shapley exacto medio por operador, desde los valores de coalicion guardados."""
    ops = rec["operators"]; n = len(ops)
    cv = {frozenset(int(x) for x in re.findall(r"\d+", k)): float(np.mean(v))
          for k, v in rec["coalitions"].items()}
    out = {}
    for i, o in enumerate(ops):
        acc = 0.0; others = [j for j in range(n) if j != i]
        for size in range(n):
            for sub in combinations(others, size):
                S = frozenset(sub); Si = S | {i}
                w = factorial(len(S))*factorial(n-len(S)-1)/factorial(n)
                acc += w*(cv[S] - cv[Si])
        out[o] = acc
    return out


def compare(exact, approx, full_set=True):
    keys = sorted(exact) if full_set else sorted(set(exact) & set(approx))
    if len(keys) < 2:
        return dict(cosine=float("nan"), rank_agreement=float("nan"))
    e = np.array([exact[k] for k in keys], dtype=float)
    a = np.array([float(approx.get(k, 0.0)) for k in keys], dtype=float)
    ne, na = np.linalg.norm(e), np.linalg.norm(a)
    cos = float(e @ a / (ne*na)) if ne > 0 and na > 0 else float("nan")
    # Misma regla que exp5: coincide o no el operador de mayor |atribucion|.
    # El desempate solo se relaja cuando el metodo exacto empata de verdad
    # (DE reparte un tercio a cada operador), en cuyo caso cualquier eleccion
    # del aproximado es igualmente correcta. La tolerancia es relativa a cada
    # vector, no absoluta: escalarla por max|e| la volvia enorme en las
    # instancias de fitness grande (Zakharov llega a 1e11) y hacia coincidir
    # cualquier cosa.
    ae, aa = np.abs(e), np.abs(a)
    tie_e = (ae.max() - ae.min()) <= 1e-12 * max(1.0, ae.max())
    if tie_e:
        return dict(cosine=cos, rank_agreement=1.0)
    top_e = int(np.argmax(ae)); top_a = int(np.argmax(aa))
    return dict(cosine=cos, rank_agreement=1.0 if top_e == top_a else 0.0)


if __name__ == "__main__":
    data = json.load(open(ROOT / "results_exact_shapley" / "exact_shapley_results.json"))
    rows = []; t0 = time.time()
    for alg, (cls, extra) in CFG.items():
        for f in data[alg]:
            for dim in data[alg][f]:
                rec = data[alg][f][dim]
                ex = exact_from(rec)
                b = get_benchmark(f); lo, hi = b.get_bounds(int(dim))
                kw = dict(dim=int(dim), bounds=(lo, hi), pop_size=POP,
                          max_generations=GEN, seed=SEED, **extra)
                run = cls(**kw).run(b.function, verbose=False)
                q = QuickSHAP().explain_from_trace(run["trace"]).shap_values
                t = TrackingAttribution().explain_from_trace(run["trace"]).attributions
                for name, vals in (("QuickSHAP", q), ("Tracking", t)):
                    full = compare(ex, vals, True); inter = compare(ex, vals, False)
                    rows.append(dict(algorithm=alg, function=f, dimension=int(dim),
                                     method=name,
                                     n_ops_exact=len(ex), n_ops_reported=len(vals),
                                     missing=sorted(set(ex) - set(vals)),
                                     cosine=full["cosine"], rank_agreement=full["rank_agreement"],
                                     cosine_intersection_only=inter["cosine"]))
        print(f"  {alg} listo ({time.time()-t0:.0f}s)", flush=True)
    out = ROOT / "results_method_comparison" / "method_comparison_fullset.json"
    json.dump(rows, open(out, "w"), indent=1)

    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        agg[r["algorithm"]][r["method"]].append((r["cosine"], r["rank_agreement"],
                                                 r["cosine_intersection_only"]))
    print(f"\n{'':6}{'metodo':12} {'cosine (nuevo)':>15} {'rank':>7} {'cosine (viejo)':>16}")
    for alg in ("DE", "GA", "PSO"):
        for m in ("QuickSHAP", "Tracking"):
            v = np.array(agg[alg][m])
            print(f"  {alg:4} {m:12} {v[:,0].mean():15.3f} {v[:,1].mean():7.2f} {v[:,2].mean():16.3f}")
    allv = defaultdict(list)
    for r in rows: allv[r["method"]].append((r["cosine"], r["rank_agreement"]))
    print("\npromedio global:")
    for m, v in allv.items():
        v = np.array(v)
        print(f"   {m:12} cosine={v[:,0].mean():.3f}  rank={v[:,1].mean():.3f}")
