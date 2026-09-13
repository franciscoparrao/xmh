#!/usr/bin/env python3
"""
Fila de KernelSHAP de la Tabla 2: equivalencia con el Shapley de coalicion exacta.

El JSON method_comparison_v2.json que alimenta esa fila del manuscrito no tenia
script generador en el repositorio, de modo que no era reproducible y quien
corriera exp5 obtenia cifras distintas (~0.99 en vez de 1.00). La diferencia no
es un desacuerdo entre metodos: exp5 evalua KernelSHAP con solo 3 corridas por
coalicion frente a las 30 del exacto, asi que mide el error de submuestreo de
v(S), no la calidad de la regresion.

Este script produce la comparacion que el manuscrito reporta: KernelSHAP canonico
(reformulacion drop-one-column) resuelto sobre EXACTAMENTE los mismos valores de
coalicion que usa la formula combinatoria. Con las 2^n coaliciones presentes el
sistema queda determinado y ambas rutas coinciden hasta precision numerica; es
una identidad algebraica y asi debe leerse, no como evidencia empirica de que
KernelSHAP aproxime bien con pocas muestras.

Salida: results_method_comparison/method_comparison_v2.json
"""
import json, re, sys
from itertools import combinations
from math import factorial
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SRC = ROOT / "results_exact_shapley" / "exact_shapley_results.json"
OUT = ROOT / "results_method_comparison" / "method_comparison_v2.json"


def exact_shapley(cv, n):
    """Formula combinatoria sobre las medias de coalicion."""
    phi = []
    for i in range(n):
        acc = 0.0; others = [j for j in range(n) if j != i]
        for size in range(n):
            for sub in combinations(others, size):
                S = frozenset(sub); Si = S | {i}
                w = factorial(len(S))*factorial(n-len(S)-1)/factorial(n)
                acc += w*(cv[S] - cv[Si])
        phi.append(acc)
    return np.array(phi)


def kernel_shap(cv, n, ops):
    """Kernel SHAP canonico con las 2^n coaliciones, restriccion de eficiencia
    impuesta por reformulacion drop-one-column.

    Se resuelve aqui en vez de llamar a ``KernelSHAP._solve_kernel_shap`` porque
    ese solver pierde precision en las instancias de escala grande (en Zakharov,
    con valores del orden de 1e11, la diferencia frente a la formula combinatoria
    llega a 1.6e9, o sea ~1.6% relativo). La formulacion de abajo centra el
    sistema antes de resolverlo y reproduce la identidad a precision de maquina.
    """
    subsets = [frozenset(s) for size in range(n+1) for s in combinations(range(n), size)]
    full, empty = frozenset(range(n)), frozenset()
    v0, vN = cv[empty], cv[full]
    rows, ys, ws = [], [], []
    for S in subsets:
        z = len(S)
        if z == 0 or z == n:
            continue
        w = (n - 1) / ((factorial(n) / (factorial(z)*factorial(n-z))) * z * (n - z))
        x = np.zeros(n)
        for i in S: x[i] = 1.0
        xr = x[:-1] - x[-1]
        yr = (v0 - cv[S]) - x[-1] * (v0 - vN)
        rows.append(xr); ys.append(yr); ws.append(w)
    X = np.array(rows); y = np.array(ys); W = np.diag(ws)
    XtWX = X.T @ W @ X; XtWy = X.T @ W @ y
    try:
        phi_red = np.linalg.solve(XtWX, XtWy)
    except np.linalg.LinAlgError:
        phi_red = np.linalg.lstsq(XtWX, XtWy, rcond=None)[0]
    return np.append(phi_red, (v0 - vN) - phi_red.sum())


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na*nb)) if na and nb else 1.0


if __name__ == "__main__":
    data = json.load(open(SRC))
    rows = []
    for alg, fs in data.items():
        for f, ds in fs.items():
            for dim, rec in ds.items():
                ops = rec["operators"]; n = len(ops)
                cv = {frozenset(int(x) for x in re.findall(r"\d+", k)): float(np.mean(v))
                      for k, v in rec["coalitions"].items()}
                ex = exact_shapley(cv, n); ks = kernel_shap(cv, n, ops)
                # misma definicion que exp5: coincide o no el operador de mayor atribucion
                # con empates exactos (DE reparte 50/50) cualquier desempate es
                # valido, asi que el top se compara con tolerancia relativa
                tol = 1e-9 * max(1.0, float(np.max(np.abs(ex))))
                top_ex = int(np.argmax(ex))
                top_ks = top_ex if abs(ks[top_ex] - np.max(ks)) <= tol else int(np.argmax(ks))
                rows.append(dict(
                    algorithm=alg, function=f, dimension=int(dim), method="KernelSHAP",
                    cosine=round(cosine(ex, ks), 12),
                    rank_agreement=1.0 if top_ex == top_ks else 0.0,
                    max_abs_diff=float(np.max(np.abs(ex - ks))),
                    note=("canonical Kernel SHAP (drop-one-column constraint); "
                          "reuses exact-Shapley coalition values, no algorithm re-run"),
                ))
    OUT.parent.mkdir(exist_ok=True)
    json.dump(rows, open(OUT, "w"), indent=1)
    c = np.array([r["cosine"] for r in rows]); m = np.array([r["max_abs_diff"] for r in rows])
    ra = np.array([r["rank_agreement"] for r in rows])
    print(f"instancias: {len(rows)}")
    print(f"  cosine:          min={c.min():.15f}  media={c.mean():.15f}")
    print(f"  rank agreement:  min={ra.min():.3f}   media={ra.mean():.3f}")
    print(f"  |diferencia| max entre ambas rutas: {m.max():.3e}")
