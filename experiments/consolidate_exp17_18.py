#!/usr/bin/env python3
"""Consolida Sweep C (exp17) y Experimento 5b (exp18) en los JSON de reproduce/data.

Misma convencion que analyze_exp17_perrun.py / analyze_exp18_19.py: media de los
shares por corrida, intervalo del 95 % sobre esa misma distribucion. Desde v1.6.0
las celdas crudas vienen de results_shade_budget_def2/ y results_shade_memory/
(neutral de adaptacion de la Definicion 2, memoria congelada con muestreo), y se
produjeron en x86-64.
"""
import itertools, json, math, os, platform, statistics as st

B = "/home/franciscoparrao/proyectos/metaheuristicas"
FU9 = ["sphere","rosenbrock","rastrigin","schwefel","ackley","griewank","levy","zakharov","dixon_price"]
NEUTRAL = "parameter adaptation: memory frozen at 0.5, SHADE sampling kept (Definition 2, v1.6.0)"
CONV = "mean of per-run shares, phi_i^(r) / sum_j phi_j^(r) averaged over the R runs; this is the convention of manuscript Section 5.1"


def per_run(cell):
    ops = list(cell["shapley_values"]); n = len(ops); co = {}
    for k, v in cell["coalitions"].items():
        inner = k.strip("()").replace(",", " ").split()
        co[frozenset(int(x) for x in inner)] = v
    R = len(next(iter(co.values()))); acc = {o: [] for o in ops}
    for r in range(R):
        phi = []
        for i in range(n):
            tot = 0.0; others = [j for j in range(n) if j != i]
            for size in range(n):
                for S in itertools.combinations(others, size):
                    Sf = frozenset(S)
                    w = math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
                    tot += w * (co[Sf | {i}][r] - co[Sf][r])
            phi.append(-tot)
        s = sum(phi)
        if abs(s) > 1e-12:
            for o, v in zip(ops, phi): acc[o].append(100 * v / s)
    return ({o: st.mean(v) for o, v in acc.items()},
            {o: (1.96 * st.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0) for o, v in acc.items()})


def exp17(src=f"{B}/results_shade_budget_def2", dst=f"{B}/xmh/reproduce/data/exp17_shade_budget.json"):
    FU, GE = ["ackley","griewank","rastrigin","schwefel"], [200, 500, 1000, 2000]
    first = json.load(open(f"{src}/shade_budget_ackley_g200.json"))
    out = {"experiment": "exp17_shade_budget", "algorithm": "SHADE", "dimension": first["dimension"],
           "pop_size": first["pop_size"], "n_runs_per_coalition": first["n_runs"], "n_coalitions": 16,
           "base_seed": first["base_seed"], "instrumentation_level": first["instrumentation_level"],
           "architecture": platform.machine(), "neutral_construction": NEUTRAL,
           "share_convention": CONV, "functions": FU, "generations": GE, "per_cell": {}}
    for f in FU:
        out["per_cell"][f] = {}
        for g in GE:
            c = json.load(open(f"{src}/shade_budget_{f}_g{g}.json"))
            m, ci = per_run(c)
            out["per_cell"][f][str(g)] = {"fes_budget": c["fes_budget"], "shapley_values": c["shapley_values"],
                "share_pct_per_run": m, "share_ci95_halfwidth": ci,
                "phi_confidence_intervals": c["confidence_intervals"], "coalitions": c["coalitions"]}
    json.dump(out, open(dst, "w"), indent=1)
    print("exp17 ->", dst)


def exp18(src=f"{B}/results_shade_memory", grid=f"{B}/results_shade_shapley/shade_shapley_results.json",
          dst=f"{B}/xmh/reproduce/data/exp18_shade_memory.json"):
    pub = json.load(open(grid))["SHADE"]
    first = json.load(open(f"{src}/shade_H50_sphere_D10.json"))
    out = {"experiment": "exp18_shade_memory", "algorithm": "SHADE", "pop_size": first["pop_size"],
           "max_generations": first["max_generations"], "fes_budget": first["fes_budget"],
           "n_runs_per_coalition": first["n_runs"], "n_coalitions": 16, "base_seed": first["base_seed"],
           "instrumentation_level": first["instrumentation_level"], "architecture": platform.machine(),
           "h_memory_size": first["h_memory_size"], "p_best_rate": first["p_best_rate"],
           "neutral_construction": NEUTRAL,
           "purpose": "Repeats the Experiment 5 grid with the memory size SHADE's specification uses (H = |P|) instead of H = 5, changing exactly one parameter, so the parameter-adaptation share can be read as a property of the operator rather than of a configuration choice.",
           "share_convention": "mean of per-run shares, phi_i^(r) / sum_j phi_j^(r)", "per_cell": {}}
    for f in FU9:
        out["per_cell"][f] = {}
        for d in (10, 30, 50):
            c = json.load(open(f"{src}/shade_H50_{f}_D{d}.json"))
            m, ci = per_run(c); m5, _ = per_run(pub[f][str(d)])
            out["per_cell"][f][str(d)] = {"fes_budget": c["fes_budget"], "shapley_values": c["shapley_values"],
                "share_pct_per_run": m, "share_ci95_halfwidth": ci, "share_pct_per_run_H5_reference": m5,
                "coalitions": c["coalitions"], "elapsed_seconds": c["elapsed_seconds"]}
    json.dump(out, open(dst, "w"), indent=1)
    print("exp18 ->", dst)


if __name__ == "__main__":
    exp17(); exp18()
