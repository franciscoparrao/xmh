#!/usr/bin/env python3
"""
Experimento 15: atribucion exact-coalition para PSO en el caso de uso de HPO.

Responde al issue 3 de Reviewer 3 (Information Sciences, 2026-08-21): el caso de
uso concluye que la topologia gbest no aporta nada y recomienda cambiarla, pero
lo hace con QuickSHAP, que es justamente el metodo que el propio paper reporta
como fallido en PSO. Con la metrica corregida ese fallo es aun mayor de lo que
se creia --- coseno 0.02 sobre las 27 instancias de benchmark --- de modo que la
recomendacion practica descansa sobre el metodo menos fiable disponible.

Aqui se recomputa la atribucion de PSO en los cuatro conjuntos de datos por
evaluacion de coaliciones, que es el metodo que el paper valida. Cada coalicion
se evalua con R corridas independientes; las corridas son independientes entre
si, de modo que se reparten entre procesos.

Presupuesto: R=10 corridas por coalicion en vez de las 30 usadas en los
benchmarks sinteticos. La reduccion es deliberada y se declara: cada evaluacion
entrena un XGBoost con validacion cruzada de 5 pliegues, y R=30 sobre los cuatro
conjuntos costaria del orden de 120 horas de CPU. Con R=10 el intervalo de
confianza es mas ancho, lo que se reporta.

Salida: results_hpo_exact/hpo_pso_exact.json
"""
import json, os, sys, time
from itertools import combinations
from math import factorial
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from concurrent.futures import ProcessPoolExecutor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

R_RUNS = 10
BASE_SEED = 42
DATASETS = ["breast_cancer", "wine", "diabetes", "vehicle"]
OPS = ["velocity_constriction", "position_update", "topology_gbest"]
OUT = ROOT / "results_hpo_exact"


def one_run(args):
    """Una corrida de PSO con la coalicion indicada. Se ejecuta en un proceso aparte."""
    import warnings; warnings.filterwarnings("ignore")
    ds, active, seed = args
    from xmh.algorithms.instrumented_pso import InstrumentedPSO
    from xmh.experiments.exp_hpo_application import XGBoostHPO, load_dataset, HPOConfig
    cfg = HPOConfig()
    X, y, _ = load_dataset(ds)
    obj = XGBoostHPO(X, y, cfg.hyperparameters, cv_folds=cfg.cv_folds)
    dim = len(cfg.hyperparameters)
    alg = InstrumentedPSO(dim=dim, bounds=(np.zeros(dim), np.ones(dim)),
                          pop_size=cfg.pop_size, max_generations=cfg.max_generations,
                          topology="gbest", velocity_strategy="constriction", seed=seed)
    disable = [o for o in OPS if o not in active]
    if disable:
        orig = alg._initialize_operators
        def patched(_s=alg, _o=orig, _d=disable):
            _o()
            for op in _d: _s.disable_operator(op)
        alg._initialize_operators = patched
    return float(alg.run(obj, verbose=False)["best_fitness"])


def shapley(cv, ops):
    """Shapley per-run y share medio, la convencion del paper."""
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
    for ds in DATASETS:
        for size in range(len(OPS)+1):
            for sub in combinations(range(len(OPS)), size):
                active = [OPS[i] for i in sub]
                for r in range(R_RUNS):
                    jobs.append((ds, tuple(active), BASE_SEED + r))
    print(f"{len(jobs)} corridas en {workers} procesos "
          f"({len(DATASETS)} datasets x 8 coaliciones x {R_RUNS} corridas)", flush=True)

    t0 = time.time(); res = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for (ds, active, seed), val in zip(jobs, ex.map(one_run, jobs, chunksize=1)):
            res.setdefault(ds, {}).setdefault(active, []).append(val)
            done = sum(len(v) for d in res.values() for v in d.values())
            if done % 40 == 0:
                el = time.time()-t0
                print(f"  {done}/{len(jobs)}  ({el/60:.1f} min, "
                      f"ETA {el/done*(len(jobs)-done)/60:.0f} min)", flush=True)

    out = {}
    for ds, coals in res.items():
        cv = {frozenset(OPS.index(o) for o in k): np.array(v) for k, v in coals.items()}
        out[ds] = dict(shapley=shapley(cv, OPS), n_runs=R_RUNS,
                       coalitions={str(sorted(k)): list(map(float, v)) for k, v in cv.items()})
        print(f"\n{ds}:")
        for o, d in out[ds]["shapley"].items():
            print(f"   {o:24} {d['share']:7.2f}% +- {d['ci95']:.2f}")
    json.dump(out, open(OUT / "hpo_pso_exact.json", "w"), indent=1)
    print(f"\nlisto en {(time.time()-t0)/60:.1f} min -> {OUT/'hpo_pso_exact.json'}")
