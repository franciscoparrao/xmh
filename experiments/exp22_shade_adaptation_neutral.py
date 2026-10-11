#!/usr/bin/env python3
"""
Experimento 22: sensibilidad del resultado de SHADE al neutral de la adaptacion.

La share negativa de la adaptacion de parametros (Seccion 5.6) se mide contra
un neutral de valor fijo: (F, CR) = (0.5, 0.5) sin actualizacion de memoria
(Tabla S1). Es un tercer tipo de neutral, ni pass-through ni "uninformed", y
hasta ahora no se habia variado. Tras la experiencia de DE (Exp. 21: dos
neutrales inertes manufacturaron un reparto en tercios), este experimento
repite el protocolo del Sweep C (cuatro funciones a D = 30, G = 200, R = 30,
16 coaliciones, semilla base 1) bajo tres construcciones del neutral:

  published      (F, CR) = (0.5, 0.5) fijos, sin muestreo ni actualizacion  (Tabla S1)
  frozen_memory  muestreo de SHADE (Cauchy para F, normal para CR) alrededor de
                 una memoria congelada en 0.5, sin actualizacion: conserva el
                 mecanismo y quita la senal (la construccion de la Def. 2)
  uniform        F ~ U(0.05, 1], CR ~ U[0, 1] por individuo, sin memoria:
                 "uninformed" sobre el rango admisible

La coalicion completa no toca el neutral y debe coincidir entre variantes
(control). Se reporta la share con signo de la adaptacion por funcion y
variante, con IC95 por corrida.

Uso: python3 exp22_shade_adaptation_neutral.py [--workers 12] [--smoke]
Salida: results_shade_adaptation_neutral/shade_adaptation_neutral.json
"""
import argparse, json, sys, time
from itertools import combinations
from math import factorial
from multiprocessing import Pool
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from xmh.experiments.exp17_shade_budget import LightSHADE, CapturingSHAP, DIM, POP_SIZE, N_RUNS, BASE_SEED
from xmh.benchmarks.functions import get_benchmark

OUT = ROOT / "results_shade_adaptation_neutral" / "shade_adaptation_neutral.json"
FUNCS = ["ackley", "griewank", "rastrigin", "schwefel"]
GENS = 200


class SHADEFrozenMemory(LightSHADE):
    """Muestreo de SHADE alrededor de M_F = M_CR = 0.5 congelados (sin actualizacion)."""
    def _neutral_parameters(self):
        cr = float(np.clip(self.rng.normal(0.5, 0.1), 0.0, 1.0))
        f = -1.0
        while f <= 0:
            f = float(self.rng.standard_cauchy() * 0.1 + 0.5)
        return float(min(f, 1.0)), cr


class SHADEUniform(LightSHADE):
    """(F, CR) uniformes sobre el rango admisible, por individuo, sin memoria."""
    def _neutral_parameters(self):
        return float(self.rng.uniform(0.05, 1.0)), float(self.rng.uniform(0.0, 1.0))


VARIANTS = {"published": LightSHADE, "frozen_memory": SHADEFrozenMemory, "uniform": SHADEUniform}


def per_run_shares(coal, ops):
    n = len(ops); R = len(next(iter(coal.values())))
    cv = {frozenset(k): np.asarray(v, float) for k, v in coal.items()}
    out = np.zeros((R, n))
    for r in range(R):
        phis = []
        for i in range(n):
            phi = 0.0
            for size in range(n):
                for sub in combinations([j for j in range(n) if j != i], size):
                    S = frozenset(sub); w = factorial(len(S)) * factorial(n - len(S) - 1) / factorial(n)
                    phi += w * (cv[S][r] - cv[S | {i}][r])
            phis.append(phi)
        tot = sum(phis); out[r] = [100 * p / tot if tot else 0.0 for p in phis]
    return out


def run_cell(args):
    variant, func_name = args
    cls = VARIANTS[variant]
    b = get_benchmark(func_name); lo, hi = b.get_bounds(DIM)
    kw = dict(dim=DIM, bounds=(lo, hi), pop_size=POP_SIZE, max_generations=GENS)
    exact = CapturingSHAP(n_runs_per_coalition=N_RUNS, base_seed=BASE_SEED, compute_interactions=False)
    t0 = time.time()
    expl = exact.explain(cls, kw, b.function)
    ops = list(expl.shap_values.keys())
    coal = {tuple(sorted(k)): list(map(float, v)) for k, v in exact.captured_coalitions.items()}
    shares = per_run_shares(coal, ops)
    mean = shares.mean(0); ci = 1.96 * shares.std(0, ddof=1) / np.sqrt(shares.shape[0])
    return variant, func_name, dict(
        operators=ops, coalitions={str(k): v for k, v in coal.items()},
        shapley_values={o: float(v) for o, v in expl.shap_values.items()},
        share_pct_per_run={o: float(m) for o, m in zip(ops, mean)},
        share_ci95={o: float(c) for o, c in zip(ops, ci)},
        base_value=float(expl.base_value), total_contribution=float(expl.total_contribution),
        elapsed_seconds=round(time.time() - t0, 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=12); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    jobs = [(v, f) for v in VARIANTS for f in FUNCS]
    if a.smoke: jobs = [(v, "ackley") for v in VARIANTS]
    print(f"{len(jobs)} celdas, {16 * N_RUNS} runs cada una", flush=True)
    t0 = time.time(); results = {}
    OUT.parent.mkdir(exist_ok=True)
    with Pool(min(a.workers, len(jobs))) as pool:
        for v, f, rec in pool.imap_unordered(run_cell, jobs):
            results.setdefault(v, {})[f] = rec
            sh = rec["share_pct_per_run"]; ci = rec["share_ci95"]
            ad = next(o for o in sh if "adapt" in o)
            print(f"  {v:14} {f:10} adapt={sh[ad]:+7.2f} ± {ci[ad]:.2f}   " +
                  "  ".join(f"{o.split('_')[0][:4]}={sh[o]:6.1f}" for o in sh) + f"   ({time.time()-t0:.0f}s)", flush=True)
            json.dump(dict(meta=dict(dim=DIM, pop=POP_SIZE, gens=GENS, n_runs=N_RUNS, base_seed=BASE_SEED,
                                     functions=FUNCS, variants=list(VARIANTS), elapsed_s=round(time.time()-t0, 1)),
                           results=results), open(OUT, "w"), indent=1)
    # control: coalicion completa identica entre variantes
    for f in FUNCS:
        fulls = [np.asarray(results[v][f]["coalitions"]["(0, 1, 2, 3)"]) for v in VARIANTS if f in results.get(v, {})]
        if len(fulls) == len(VARIANTS):
            print(f"  control {f}: coalicion completa identica = {all(np.array_equal(fulls[0], x) for x in fulls[1:])}")
    print(f"listo en {(time.time()-t0)/60:.1f} min -> {OUT}")
