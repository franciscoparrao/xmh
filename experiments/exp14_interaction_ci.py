#!/usr/bin/env python3
"""
Experimento 14: intervalos de confianza y correccion por comparaciones multiples
para los indices de interaccion de Shapley.

Responde al issue 4 de Reviewer 1 (Information Sciences, 2026-08-21): las 81
configuraciones reportan I_ij sin CI ni correccion, de modo que afirmaciones como
"la redundancia crossover-mutacion de GA pasa de -78% a -48%" podrian ser ruido.

Metodo: bootstrap no parametrico sobre las 30 corridas por coalicion (B=10000),
recomputando I_ij en cada replica; CI percentil al 95%; p-value bootstrap de la
hipotesis I_ij = 0; correccion Holm sobre la familia completa de pruebas.
Ademas, consistencia de signo por par de operadores a traves de configuraciones.

Salida: results_interaction_ci/interaction_ci.json
"""
import json, sys
from itertools import combinations
from math import factorial
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results_interaction_ci"
B = 10000
RNG = np.random.default_rng(20260822)


def load(path):
    d = json.load(open(ROOT / path)); rows = {}
    def walk(n, p=()):
        if isinstance(n, dict) and "shapley_values" in n: rows[p] = n
        elif isinstance(n, dict):
            for k, v in n.items(): walk(v, p + (k,))
    walk(d); return rows


def coalition_matrix(rec):
    """{frozenset(indices): array(R,)} a partir del JSON."""
    import re
    return {frozenset(int(x) for x in re.findall(r"\d+", k)): np.asarray(v, float)
            for k, v in rec["coalitions"].items()}


def interactions_from(means, n):
    """I_ij normalizado como % de la mejora total.

    `means` llega como fitness final por coalicion (menor es mejor), pero la
    funcion caracteristica del paper (Definicion 1) es la MEJORA
    v(S) = f_0 - f_G^(S). Las constantes f_0 se cancelan en la diferencia de
    segundo orden, de modo que basta invertir el signo del delta calculado sobre
    el fitness. Sin esa inversion el acoplamiento mutacion-crossover de DE sale
    como -100% (antagonismo) en vez de +100% (sinergia), que es la lectura
    correcta y la que aparece en el manuscrito.
    """
    full = frozenset(range(n)); empty = frozenset()
    total = means[empty] - means[full]
    out = {}
    for i, j in combinations(range(n), 2):
        others = [k for k in range(n) if k not in (i, j)]
        acc = 0.0
        for size in range(n - 1):
            for sub in combinations(others, size):
                S = frozenset(sub)
                delta = means[S] - means[S | {i}] - means[S | {j}] + means[S | {i, j}]
                w = factorial(len(S)) * factorial(n - len(S) - 2) / factorial(n - 1)
                acc += w * delta
        out[(i, j)] = -100 * acc / total if total else 0.0
    return out


def holm(pvals):
    idx = np.argsort(pvals); m = len(pvals); adj = np.empty(m); run = 0.0
    for rank, k in enumerate(idx):
        val = (m - rank) * pvals[k]
        run = max(run, val); adj[k] = min(1.0, run)
    return adj


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    SOURCES = {
        "DE/GA/PSO": "results_exact_shapley/exact_shapley_results.json",
    }
    rows = load(SOURCES["DE/GA/PSO"])
    records = []
    for path, rec in rows.items():
        alg, func, dim = path[0], path[-2], path[-1]
        ops = rec["operators"]; n = len(ops)
        cv = coalition_matrix(rec)
        R = len(next(iter(cv.values())))
        point = interactions_from({S: v.mean() for S, v in cv.items()}, n)
        # bootstrap: resamplear corridas en bloque (misma remuestra para todas las coaliciones)
        boot = {k: np.empty(B) for k in point}
        for b in range(B):
            pick = RNG.integers(0, R, R)
            means = {S: v[pick].mean() for S, v in cv.items()}
            for k, val in interactions_from(means, n).items():
                boot[k][b] = val
        for (i, j), val in point.items():
            d = boot[(i, j)]
            lo, hi = np.percentile(d, [2.5, 97.5])
            # p-value bootstrap bilateral contra I=0
            p = 2 * min((d <= 0).mean(), (d >= 0).mean())
            records.append(dict(algorithm=alg, function=func, dimension=dim,
                                pair=f"{ops[i]}|{ops[j]}", i=ops[i], j=ops[j],
                                value=float(val), ci_low=float(lo), ci_high=float(hi),
                                p_boot=float(min(1.0, max(p, 1.0 / B)))))
        print(f"  {alg:5} {func:12} D={dim:2}  " +
              "  ".join(f"{ops[i][:4]}-{ops[j][:4]}={v:+7.1f}%" for (i, j), v in point.items()),
              flush=True)

    p = np.array([r["p_boot"] for r in records])
    padj = holm(p)
    for r, pa in zip(records, padj):
        r["p_holm"] = float(pa); r["significant_holm"] = bool(pa < 0.05)
        r["ci_excludes_zero"] = bool(r["ci_low"] > 0 or r["ci_high"] < 0)

    # consistencia de signo por (algoritmo, par)
    consistency = {}
    for r in records:
        consistency.setdefault((r["algorithm"], r["pair"]), []).append(r["value"])
    cons_out = {}
    for (alg, pair), vals in consistency.items():
        v = np.array(vals); npos = int((v > 0).sum()); nneg = int((v < 0).sum())
        frac = max(npos, nneg) / len(v)
        cons_out[f"{alg}::{pair}"] = dict(n=len(v), positive=npos, negative=nneg,
                                          dominant_sign_fraction=round(float(frac), 3),
                                          robust=bool(frac >= 0.8),
                                          mean=float(v.mean()))
    json.dump(dict(meta=dict(B=B, n_tests=len(records), seed=20260822),
                   records=records, sign_consistency=cons_out),
              open(OUT / "interaction_ci.json", "w"), indent=1)

    nsig = sum(r["significant_holm"] for r in records)
    nci = sum(r["ci_excludes_zero"] for r in records)
    print(f"\npruebas: {len(records)}")
    print(f"  CI 95% que excluye el cero:      {nci}")
    print(f"  significativas tras Holm:        {nsig}")
    print(f"\nconsistencia de signo por (algoritmo, par):")
    for k, v in sorted(cons_out.items()):
        print(f"  {'ROBUSTO' if v['robust'] else '  ----- '} {k:58} "
              f"{v['dominant_sign_fraction']:.2f}  media={v['mean']:+7.1f}%")
