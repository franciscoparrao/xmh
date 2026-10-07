"""Experiment 20 — The complementarity claim against IOHxplainer itself.

Section 5.8 argues that XMH and fANOVA-style hyperparameter analysis rank
operator choices differently. The comparator used there is a one-way ANOVA
eta^2 over the six categorical crossover variants, built for the purpose. A
reviewer can fairly object that this is a stand-in: the claim is about
hyperparameter-sensitivity tooling, and the reference implementation of that
tooling is IOHxplainer (van Stein et al., ACM TELO 2025), which was never run.

This experiment runs it. The paper's GA is wrapped as an IOH-compatible
optimizer, the six crossover variants of Experiment 7 become a categorical
hyperparameter, and IOHxplainer is given a configuration space with three more
knobs so its CatBoost-plus-SHAP surrogate has something to decompose rather
than a single degenerate factor. The output is its SHAP attribution over
hyperparameters, which we then rank against XMH's crossover-Shapley ranking.

Runs inside the container built from Dockerfile.iohx: IOHxplainer pins
numpy 1.24.4 and five other 2023-era versions that have no Python 3.12 wheels.
"""
from __future__ import annotations

import json, os, sys

import numpy as np
from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter
from iohxplainer import explainer

from xmh.algorithms.instrumented_ga import InstrumentedGA

# Mismos seis operadores y misma configuracion base que el Experimento 7
CX_TYPES = ["sbx", "blx_alpha", "arithmetic", "uniform", "two_point", "single_point"]
DIM, BUDGET, POP = 30, 10000, 50
FIDS = [3]          # BBOB f3 = Rastrigin, la funcion del Experimento 7
IIDS = [1, 2]
REPS = 5


def ga_optimizer(problem, config, budget=BUDGET, dim=DIM, seed=0):
    """Adaptador: corre el GA del paper sobre un problema IOH.

    IOHxplainer invoca optimizer(problem, config, budget, dim, seed) y lee el
    resultado del estado del problema, asi que aqui solo hay que traducir la
    configuracion y dejar que el GA evalue `problem`.
    """
    lo = np.full(dim, problem.bounds.lb[0], dtype=float)
    hi = np.full(dim, problem.bounds.ub[0], dtype=float)
    ga = InstrumentedGA(
        dim=dim, bounds=(lo, hi), pop_size=POP,
        max_generations=max(1, budget // POP),
        selection_method="tournament",
        crossover_type=str(config["crossover_type"]),
        mutation_type="polynomial",
        crossover_prob=float(config["crossover_prob"]),
        mutation_prob=float(config["mutation_prob"]),
        tournament_size=int(config["tournament_size"]),
        seed=int(seed),
    )
    ga.run(lambda x: float(problem(list(x))), verbose=False)


def build_space() -> ConfigurationSpace:
    # ConfigSpace 0.6.1 (la version que IOHxplainer fija) usa add_hyperparameters
    cs = ConfigurationSpace()
    cs.add_hyperparameters([
        CategoricalHyperparameter("crossover_type", CX_TYPES),
        CategoricalHyperparameter("crossover_prob", [0.6, 0.9]),
        CategoricalHyperparameter("mutation_prob", [0.05, 0.1]),
        CategoricalHyperparameter("tournament_size", [2, 3]),
    ])
    return cs


def main(outdir):
    os.makedirs(outdir, exist_ok=True)
    cs = build_space()
    ex = explainer(
        ga_optimizer, cs, algname="xmh-GA",
        dims=[DIM], fids=FIDS, iids=IIDS, reps=REPS,
        sampling_method="grid", grid_steps_dict={},
        budget=BUDGET, seed=1, verbose=True,
    )
    ex.run(paralell=True, checkpoint_file=os.path.join(outdir, "iohx_intermediate.csv"))
    ex.save_results(os.path.join(outdir, "iohx_results.pkl"))
    df = ex.explain(partial_dependence=False, best_config=False,
                    file_prefix=os.path.join(outdir, "iohx"))
    try:
        df.to_csv(os.path.join(outdir, "iohx_shap.csv"), index=False)
        print("SHAP guardado:", df.shape)
    except Exception as e:
        print("explain() no devolvio un DataFrame:", type(e).__name__, e)
    return ex


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/work/results_iohxplainer"
    main(out)
