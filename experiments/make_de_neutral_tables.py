#!/usr/bin/env python3
"""
Tablas LaTeX para el SM a partir de exp21: DE bajo las tres construcciones de
neutral (published / sel_uniform / table_s1). Lee de_neutral_variants.json
(table_s1 y, si se fundio, sel_uniform) y de_neutral_selection.json (corrida
original de sel_uniform, core v1.5) y el JSON publicado (published, de respaldo).
Imprime: (a) promedios por dimension y construccion; (b) grilla por funcion de
table_s1. Uso: python3 make_de_neutral_tables.py [--fold]  (--fold funde
sel_uniform en de_neutral_variants.json; hacerlo solo con exp21 terminado).
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RS = ROOT / "results_neutral_sensitivity"
OPS = ["mutation_rand/1", "crossover_binomial", "selection_greedy"]
FUNCS = ["sphere", "rosenbrock", "rastrigin", "ackley", "griewank", "schwefel",
         "levy", "zakharov", "dixon_price"]
FLAB = dict(sphere="Sphere", rosenbrock="Rosenbrock", rastrigin="Rastrigin", ackley="Ackley",
            griewank="Griewank", schwefel="Schwefel", levy="Levy", zakharov="Zakharov",
            dixon_price="Dixon--Price")


def load():
    var = json.load(open(RS / "de_neutral_variants.json"))["results"]
    if "sel_uniform" not in var and (RS / "de_neutral_selection.json").exists():
        var["sel_uniform"] = json.load(open(RS / "de_neutral_selection.json"))["results"]
    pubfile = sorted(ROOT.glob("results_exact_shapley/exact_shapley_results.json.bak-preneutralfix-*"))
    pub = json.load(open(pubfile[-1] if pubfile else ROOT / "results_exact_shapley/exact_shapley_results.json"))["DE"]
    var["published"] = {f: {d: {"shapley_shares_per_run": pub[f][d]["shapley_shares_per_run"]}
                            for d in pub[f]} for f in pub}
    return var


def fmt(x):
    return f"$-{abs(x):.1f}$" if x < 0 else f"{x:.1f}"


if __name__ == "__main__":
    if "--fold" in sys.argv:
        p = RS / "de_neutral_variants.json"; d = json.load(open(p))
        if "sel_uniform" not in d["results"]:
            d["results"]["sel_uniform"] = json.load(open(RS / "de_neutral_selection.json"))["results"]
            json.dump(d, open(p, "w"), indent=1); print("sel_uniform fundido en", p)
    var = load()
    NAMES = {"published": "Inert (v1.0--v1.5): target retained, mutant discarded",
             "sel_uniform": "Selection uniform only",
             "table_s1": "Definition~2 / Table~S1 (adopted)"}
    print("% ---- (a) promedios por dimension ----")
    for v in ("published", "sel_uniform", "table_s1"):
        if v not in var: continue
        for d in ("10", "30", "50"):
            rows = [var[v][f][d]["shapley_shares_per_run"] for f in FUNCS if f in var[v] and d in var[v][f]]
            if not rows: continue
            m = {o: np.mean([r[o] for r in rows]) for o in OPS}
            lo = {o: min(r[o] for r in rows) for o in OPS}; hi = {o: max(r[o] for r in rows) for o in OPS}
            print(f"{NAMES[v] if d=='10' else ''} & $D = {d}$ & " +
                  " & ".join(f"{fmt(m[o])} [{fmt(lo[o])}, {fmt(hi[o])}]" for o in OPS) + f" \\\\  % n={len(rows)}")
    print("\n% ---- (b) grilla por funcion, table_s1 ----")
    if "table_s1" in var:
        for f in FUNCS:
            if f not in var["table_s1"]: continue
            cells = []
            for d in ("10", "30", "50"):
                r = var["table_s1"][f].get(d)
                cells += [fmt(r["shapley_shares_per_run"][o]) for o in OPS] if r else ["--"] * 3
            print(f"{FLAB[f]} & " + " & ".join(cells) + " \\\\")
    print("\n% ---- (c) resumen table_s1 ----")
    if "table_s1" in var:
        allr = [(f, d, var["table_s1"][f][d]) for f in FUNCS if f in var["table_s1"] for d in var["table_s1"][f]]
        cx = [r["shapley_shares_per_run"][OPS[1]] for _, _, r in allr]
        se = [r["shapley_shares_per_run"][OPS[2]] for _, _, r in allr]
        mu = [r["shapley_shares_per_run"][OPS[0]] for _, _, r in allr]
        neg = sum(1 for x in cx if x < 0)
        ci_cx_excl0 = sum(1 for _, _, r in allr if r["shapley_shares_per_run"][OPS[1]] + r["shares_ci95"][OPS[1]] < 0)
        print(f"n={len(allr)}  mutation {min(mu):.1f}--{max(mu):.1f} (median {np.median(mu):.1f})  "
              f"crossover {min(cx):.1f}--{max(cx):.1f} (median {np.median(cx):.1f}), negative in {neg}/{len(allr)}, CI excludes 0 in {ci_cx_excl0}  "
              f"selection {min(se):.1f}--{max(se):.1f} (median {np.median(se):.1f})")
        for f, d, r in allr:
            v = {k: np.mean(vv) for k, vv in r["coalitions"].items()}; v0 = r["base_value"]; tot = r["total_contribution"]
            print(f"   {f:12} D{d:>2}  v(mut)={(v0-v['(0,)'])/tot:5.2f} v(mut,cx)={(v0-v['(0, 1)'])/tot:5.2f} "
                  f"v(mut,sel)={(v0-v['(0, 2)'])/tot:5.2f} v(cx,sel)={(v0-v['(1, 2)'])/tot:5.2f}")
