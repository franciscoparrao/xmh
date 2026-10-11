#!/usr/bin/env python3
"""
Experimento 23: recomputo de la familia SHADE/JADE bajo el neutral de adaptacion
de la Definicion 2 (memoria congelada en 0.5 con el muestreo propio de SHADE;
v1.6.0, motivado por exp22).

Reproduce, con una sola evaluacion por coalicion y el logger en LIGHT:
  * exp10  SHADE, 9 funciones x 3 dimensiones, G = 200, R = 30, 16 coaliciones,
           con indices de interaccion            -> results_shade_shapley/shade_shapley_results.json
  * exp12  JADE, idem sin interacciones          -> results_jade_shapley/jade_shapley_results.json
  * exp18  SHADE con H = |P| (Exp. 5b), 27 celdas -> results_shade_memory/shade_H50_<f>_D<d>.json
  * exp17  Sweep C, 4 funciones x G in {200,500,1000,2000} -> results_shade_budget_def2/

Un solo Pool sobre las 97 celdas, ordenadas de mayor a menor costo. Cada celda
escribe su archivo al terminar; los JSON agregados de SHADE/JADE se reescriben
tras cada celda, asi que una interrupcion no pierde trabajo.
"""
import json, os, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "1")
from xmh.benchmarks.functions import get_benchmark
from xmh.core.trace_logger import InstrumentationLevel
from xmh.algorithms.instrumented_jade import InstrumentedJADE
from xmh.experiments import exp17_shade_budget as e17, exp18_shade_memory as e18
from xmh.experiments.exp10_shade_shapley import FUNCTIONS, DIMENSIONS, POP_SIZE, MAX_GENERATIONS, N_RUNS, BASE_SEED

OUT_SHADE = ROOT / "results_shade_shapley" / "shade_shapley_results.json"
OUT_JADE = ROOT / "results_jade_shapley" / "jade_shapley_results.json"
OUT_MEM = ROOT / "results_shade_memory"
OUT_SWEEP = ROOT / "results_shade_budget_def2"
SWEEP_FUNCS = ["ackley", "griewank", "rastrigin", "schwefel"]
SWEEP_GENS = [200, 500, 1000, 2000]


class LightJADE(InstrumentedJADE):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw); self.logger.level = InstrumentationLevel.LIGHT


def grid_cell(alg, func_name, dim, with_interactions):
    cls = e17.LightSHADE if alg == "SHADE" else LightJADE
    b = get_benchmark(func_name); lo, hi = b.get_bounds(dim)
    kw = dict(dim=dim, bounds=(lo, hi), pop_size=POP_SIZE, max_generations=MAX_GENERATIONS)
    exact = e17.CapturingSHAP(n_runs_per_coalition=N_RUNS, base_seed=BASE_SEED, compute_interactions=False)
    t0 = time.time()
    expl = exact.explain(cls, kw, b.function)
    ops = list(expl.shap_values.keys()); cv = exact.captured_coalitions
    entry = {
        "algorithm": alg, "function": func_name, "dimension": dim, "operators": ops,
        "shapley_values": {o: float(v) for o, v in expl.shap_values.items()},
        "coalitions": {str(tuple(sorted(k))): list(map(float, v)) for k, v in cv.items()},
        "total_contribution": float(expl.total_contribution), "base_value": float(expl.base_value),
        "confidence_intervals": {k: list(map(float, v)) for k, v in expl.confidence_intervals.items()},
        "elapsed_seconds": round(time.time() - t0, 1), "n_runs": N_RUNS,
        "neutral_construction": "parameter adaptation: memory frozen at 0.5, SHADE sampling kept (Definition 2, v1.6.0)",
    }
    if with_interactions:
        inter = exact._compute_interaction_indices(ops, cv)
        entry["interactions"] = {f"{a}_x_{b_}": float(v) for (a, b_), v in inter.items()}
    return entry


def run_job(job):
    kind, f, x = job
    t0 = time.time()
    if kind == "shade": rec = grid_cell("SHADE", f, x, True)
    elif kind == "jade": rec = grid_cell("JADE", f, x, False)
    elif kind == "mem": rec = e18.run_cell(f, x, str(OUT_MEM))
    elif kind == "sweep": rec = e17.run_cell(f, x, str(OUT_SWEEP))
    sh = rec.get("shapley_values", {}); tot = sum(sh.values()) or 1.0
    ad = next((o for o in sh if "adapt" in o), None)
    return kind, f, x, rec, (100 * sh[ad] / tot if ad else float("nan")), round(time.time() - t0)


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    jobs = [("sweep", f, g) for g in (2000, 1000, 500) for f in SWEEP_FUNCS]
    jobs += [("shade", f, d) for f in FUNCTIONS for d in DIMENSIONS]
    jobs += [("jade", f, d) for f in FUNCTIONS for d in DIMENSIONS]
    jobs += [("mem", f, d) for f in FUNCTIONS for d in DIMENSIONS]
    jobs += [("sweep", f, 200) for f in SWEEP_FUNCS]
    shade = {"SHADE": {}}; jade = {"JADE": {}}
    OUT_SHADE.parent.mkdir(exist_ok=True); OUT_JADE.parent.mkdir(exist_ok=True)
    print(f"{len(jobs)} celdas, {workers} workers", flush=True)
    t0 = time.time(); done = 0
    with Pool(workers) as pool:
        for kind, f, x, rec, ad_share, el in pool.imap_unordered(run_job, jobs):
            done += 1
            if kind == "shade":
                shade["SHADE"].setdefault(f, {})[str(x)] = rec
                json.dump(shade, open(OUT_SHADE, "w"), indent=2, default=float)
            elif kind == "jade":
                jade["JADE"].setdefault(f, {})[str(x)] = rec
                json.dump(jade, open(OUT_JADE, "w"), indent=2, default=float)
            print(f"[{done:2}/{len(jobs)}] {kind:5} {f:11} {x:>4}  adapt(share of mean)={ad_share:+7.2f}%  {el}s  ({(time.time()-t0)/60:.0f} min)", flush=True)
    print(f"listo en {(time.time()-t0)/3600:.1f} h", flush=True)
