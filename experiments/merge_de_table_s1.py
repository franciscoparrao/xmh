#!/usr/bin/env python3
"""
Vuelca la construccion `table_s1` de exp21 (neutrales de DE segun Definicion 2 y
Tabla S1, adoptados en v1.6) en results_exact_shapley/exact_shapley_results.json,
conservando GA y PSO intactos, y regenera los CSV de resumen con el mismo
generador de exp4. Guarda respaldo del JSON anterior.
"""
import json, shutil, sys, time
from itertools import combinations
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from xmh.explanation.operator_shap import ExactCoalitionSHAP
import pandas as pd
import xmh.experiments.exp4_exact_shapley as _e4
_e4.pd = pd  # exp4 importa pandas dentro de main()
from xmh.experiments.exp4_exact_shapley import _generate_summary_csv

RES = ROOT / "results_exact_shapley" / "exact_shapley_results.json"
SRC = ROOT / "results_neutral_sensitivity" / "de_neutral_variants.json"
VARIANT = "table_s1"

if __name__ == "__main__":
    data = json.load(open(RES))
    new = json.load(open(SRC))["results"][VARIANT]
    stamp = time.strftime("%Y%m%d")
    bak = RES.with_name(RES.name + f".bak-preneutralfix-{stamp}")
    if not bak.exists():
        shutil.copy(RES, bak); print(f"respaldo -> {bak}")
    es = ExactCoalitionSHAP(n_runs_per_coalition=30, base_seed=42, compute_interactions=True)
    n_done = 0
    for f in data["DE"]:
        for dim in data["DE"][f]:
            rec = data["DE"][f][dim]; src = new[f][dim]; ops = rec["operators"]
            assert src["control_identical_to_published"], f"control fallo en {f} D{dim}"
            cv = {frozenset(eval(k)): list(v) for k, v in src["coalitions"].items()}
            shap, var = es._compute_shapley(ops, cv)
            inter = es._compute_interaction_indices(ops, cv)
            rec["coalitions"] = src["coalitions"]
            rec["shapley_values"] = {o: float(shap[o]) for o in ops}
            rec["interactions"] = {f"{a}_x_{b}": float(v) for (a, b), v in inter.items()}
            rec["confidence_intervals"] = {o: [float(shap[o] - 1.96 * np.sqrt(var[o])),
                                               float(shap[o] + 1.96 * np.sqrt(var[o]))] for o in ops}
            rec["shapley_shares_per_run"] = src["shapley_shares_per_run"]
            rec["shares_ci95"] = src["shares_ci95"]
            rec["base_value"] = src["base_value"]
            rec["total_contribution"] = src["total_contribution"]
            rec["elapsed_seconds"] = src["elapsed_seconds"]
            rec["neutral_construction"] = ("selection: uniform over {target, trial} (p = 0.5); "
                                           "crossover: mutant passes through (Definition 2 / Table S1, v1.6)")
            rec.pop("recomputed_after_selection_instrumentation", None)
            n_done += 1
    json.dump(data, open(RES, "w"), indent=2, default=str)
    _generate_summary_csv(data, str(RES.parent))
    print(f"{n_done} instancias de DE actualizadas -> {RES}")
