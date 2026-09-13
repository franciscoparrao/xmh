#!/usr/bin/env python3
"""
Experimento 16: validacion en las funciones de composicion de CEC2017.

Responde al issue 5 de Reviewer 1 (Information Sciences, 2026-08-21). La
validacion de la Seccion 5.8 usa diez funciones de las categorias unimodal,
multimodal simple e hibrida, y excluye deliberadamente las de composicion
(F21--F30) por tener senal mas ruidosa. El reviewer pide correr al menos dos de
ellas con R aumentado, o bien acotar la afirmacion del resumen y las
conclusiones.

Aqui se corren cuatro (F21, F22, F23 y F24) con R=50 corridas por coalicion, en
vez de las dos y las R=30 pedidas, porque la categoria completa abarca diez
funciones y dos serian una muestra tan estrecha como la que motivo la objecion.
Las corridas son independientes, de modo que se reparten entre procesos.

Salida: results_cec_composition/cec_composition.json
"""
import json, os, sys, time
from itertools import combinations
from math import factorial
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from concurrent.futures import ProcessPoolExecutor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

FUNCTIONS = [("F21", "F212017"), ("F22", "F222017"),
             ("F23", "F232017"), ("F24", "F242017")]
DIMENSIONS = [10, 30]
N_RUNS = 50
BASE_SEED = 1
POP, GEN = 50, 200
OPS = ["selection_tournament", "crossover_sbx", "mutation_polynomial"]
OUT = ROOT / "results_cec_composition"


def one_run(args):
    import warnings; warnings.filterwarnings("ignore")
    cec_cls, dim, active, seed = args
    from opfunu.cec_based import cec2017
    from xmh.algorithms.instrumented_ga import InstrumentedGA
    f = getattr(cec2017, cec_cls)(ndim=dim)
    lb = np.array(f.lb, dtype=float); ub = np.array(f.ub, dtype=float)
    alg = InstrumentedGA(dim=dim, bounds=(lb, ub), pop_size=POP, max_generations=GEN,
                         selection_method="tournament", crossover_type="sbx",
                         mutation_type="polynomial", crossover_prob=0.9,
                         tournament_size=3, seed=seed)
    disable = [o for o in OPS if o not in active]
    if disable:
        orig = alg._initialize_operators
        def patched(_s=alg, _o=orig, _d=disable):
            _o()
            for op in _d: _s.disable_operator(op)
        alg._initialize_operators = patched
    return float(alg.run(f.evaluate, verbose=False)["best_fitness"])


def shares(cv, ops):
    n = len(ops); R = len(next(iter(cv.values())))
    per = {o: [] for o in ops}
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
            per[o].append(100*phis[i]/tot if tot else 0.0)
    out = {}
    for o in ops:
        v = np.array(per[o])
        out[o] = dict(share=float(v.mean()),
                      ci95=float(1.96*v.std(ddof=1)/np.sqrt(len(v))))
    return out


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    workers = max(1, min(14, (os.cpu_count() or 4) - 2))
    jobs = []
    for label, cls in FUNCTIONS:
        for dim in DIMENSIONS:
            for size in range(len(OPS)+1):
                for sub in combinations(range(len(OPS)), size):
                    active = tuple(OPS[i] for i in sub)
                    for r in range(N_RUNS):
                        jobs.append((cls, dim, active, BASE_SEED + r))
    print(f"{len(jobs)} corridas en {workers} procesos "
          f"({len(FUNCTIONS)} funciones x {len(DIMENSIONS)} dims x 8 coaliciones x {N_RUNS} corridas)",
          flush=True)

    t0 = time.time(); acc = {}
    keys = [(c, d, a) for c, d, a, _ in jobs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, val in zip(keys, ex.map(one_run, jobs, chunksize=4)):
            acc.setdefault(k[:2], {}).setdefault(k[2], []).append(val)
            n = sum(len(v) for d in acc.values() for v in d.values())
            if n % 200 == 0:
                el = time.time()-t0
                print(f"  {n}/{len(jobs)} ({el/60:.0f} min, ETA {el/n*(len(jobs)-n)/60:.0f} min)", flush=True)

    res = {}
    label_of = {c: l for l, c in FUNCTIONS}
    for (cls, dim), coals in acc.items():
        cv = {frozenset(OPS.index(o) for o in k): np.array(v) for k, v in coals.items()}
        res.setdefault(label_of[cls], {})[str(dim)] = dict(
            shares=shares(cv, OPS), n_runs=N_RUNS,
            coalitions={str(sorted(k)): list(map(float, v)) for k, v in cv.items()})
    json.dump(res, open(OUT / "cec_composition.json", "w"), indent=1)

    print(f"\n{'funcion':8} {'D':>3}   " + "  ".join(f"{o.split('_')[0][:4]:>14}" for o in OPS))
    for L in sorted(res):
        for dim in sorted(res[L], key=int):
            sh = res[L][dim]["shares"]
            print(f"  {L:6} {dim:>3}   " + "  ".join(f"{sh[o]['share']:7.2f}±{sh[o]['ci95']:5.2f}" for o in OPS))
    print(f"\nlisto en {(time.time()-t0)/60:.1f} min -> {OUT/'cec_composition.json'}")
