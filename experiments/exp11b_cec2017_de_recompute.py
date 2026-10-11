#!/usr/bin/env python3
"""
Recomputo paralelo del bloque DE de results_cec2017_shapley tras adoptar los
neutrales de la Definicion 2 / Tabla S1 (v1.6, ver exp21). Mismo protocolo que
exp11 (pop 50, 200 generaciones, R = 30, seed base 1, 10 funciones x D in
{10, 30}); el bloque GA se conserva intacto. El manuscrito no reporta cifras de
DE en CEC2017 (Seccion 5.7 se centra en GA), pero el archivo publicado debe ser
coherente con la construccion adoptada.
"""
import json, os, shutil, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from xmh.experiments.exp11_cec2017 import (ALGORITHMS, CEC_FUNCTIONS, DIMENSIONS, POP_SIZE,
                                           MAX_GENERATIONS, N_RUNS, BASE_SEED, make_cec_benchmark)
from xmh.explanation.operator_shap import ExactCoalitionSHAP

OUT = ROOT / "results_cec2017_shapley" / "cec2017_shapley_results.json"


def run_instance(args):
    func_label, cec_cls, dim = args
    alg = ALGORITHMS["DE"]
    fn, bounds = make_cec_benchmark(cec_cls, dim)
    kw = dict(dim=dim, bounds=bounds, pop_size=POP_SIZE, max_generations=MAX_GENERATIONS,
              **alg["kwargs"])
    exact = ExactCoalitionSHAP(n_runs_per_coalition=N_RUNS, base_seed=BASE_SEED,
                               compute_interactions=False)
    t0 = time.time()
    raw = exact.get_coalition_values(alg["class"], kw, fn)
    expl = exact.explain(alg["class"], kw, fn)
    return func_label, str(dim), {
        "algorithm": "DE", "function": func_label, "cec_class": cec_cls, "dimension": dim,
        "operators": raw["operators"], "shapley_values": dict(raw["shapley_values"]),
        "coalitions": {str(k): v for k, v in raw["coalitions"].items()},
        "total_contribution": expl.total_contribution, "base_value": expl.base_value,
        "confidence_intervals": {k: list(v) for k, v in expl.confidence_intervals.items()},
        "elapsed_seconds": time.time() - t0, "n_runs": N_RUNS,
        "neutral_construction": "selection uniform p = 0.5; crossover mutant pass-through (v1.6)",
    }


if __name__ == "__main__":
    data = json.load(open(OUT))
    bak = OUT.with_name(OUT.name + f".bak-preneutralfix-{time.strftime('%Y%m%d')}")
    if not bak.exists():
        shutil.copy(OUT, bak); print(f"respaldo -> {bak}")
    jobs = [(fl, cc, d) for fl, cc in CEC_FUNCTIONS for d in DIMENSIONS]
    print(f"{len(jobs)} instancias DE CEC2017, {8 * N_RUNS} runs cada una", flush=True)
    t0 = time.time()
    with Pool(min(12, len(jobs))) as pool:
        for fl, d, rec in pool.imap_unordered(run_instance, jobs):
            old = data["DE"][fl][d]
            ctrl = np.array_equal(np.asarray(rec["coalitions"]["(0, 1, 2)"]),
                                  np.asarray(old["coalitions"]["(0, 1, 2)"]))
            data["DE"][fl][d] = rec
            tot = abs(rec["total_contribution"]) or 1.0
            print(f"  DE {fl:4} D={d:>2}  " + "  ".join(
                f"{op.split('_')[0][:4]}={100*v/tot:7.1f}%" for op, v in rec["shapley_values"].items())
                + f"   control_full={'OK' if ctrl else 'DIFIERE'}   ({time.time()-t0:.0f}s)", flush=True)
            json.dump(data, open(OUT, "w"), indent=2, default=float)
    print(f"listo en {(time.time()-t0)/60:.1f} min -> {OUT}")
