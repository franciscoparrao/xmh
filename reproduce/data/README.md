# Raw result data

Every quantitative claim in the manuscript derives from one of these files. Each entry names the
section, table or figure it supports, so a reader can go from a number in the paper to the array it
came from without running anything.

The Shapley files store, per (algorithm, function, dimension) instance, the value of every coalition
`v(S)` across the 30 independent runs, plus the per-run Shapley values and their confidence
intervals. They are the raw coalition values the Data availability statement refers to.

| File | Supports |
|---|---|
| `exact_shapley_results.json` | Section 5.2, Experiment 1 — DE/GA/PSO, 9 functions x 3 dimensions x 8 coalitions x 30 runs (19,440 runs). Table 3, Figures 1 and 7 |
| `cmaes_shapley_results.json` | Section 5.5 — CMA-ES attribution over the 27 configurations. Table S12 |
| `shade_shapley_results.json` | Section 5.6 — SHADE at n = 4, including the negative parameter-adaptation share |
| `jade_shapley_results.json` | Section 5.6 — JADE, the independent corroboration of the SHADE result |
| `cec2017_shapley_results.json` | Section 5.7 — CEC2017 validation |
| `cec_composition.json` | Section 5.7 — CEC2017 composition functions (F21-F24) |
| `method_comparison_fullset.json` | Section 5.3 and Table 5 — QuickSHAP and Tracking vs exact, full operator set, 3 dimensions |
| `method_comparison_v2.json` | Section 5.3 — canonical KernelSHAP at K = 2^n |
| `method_comparison.json` | Superseded two-dimension comparison, kept for provenance. Not used by the manuscript |
| `interaction_ci.json` | Confidence intervals for the pairwise interaction indices. Tables S2 and S7 |
| `neutral_sensitivity.json` | Section 5.8 — sensitivity to the neutral-operator construction. Table S13 |
| `exp8_part_a.json`, `exp8_part_b.json` | Section 5.7 — complementarity with fANOVA-style analyses. Table S8, Figure 8 |
| `exp9_sweep_a.json`, `exp9_sweep_b.json` | Section 5.9 — budget sensitivity. Figure 9 |
| `algorithm_friedman.json`, `operator_friedman.json` | Section 5.11 — Friedman tests with Holm correction over the six algorithms |
| `overhead_results.json`, `overhead_results.csv` | Section 5.4 — wall-clock overhead by component. Table 4, Figure S5 |
| `hpo_results.json` | Section 6.2 — accuracy of DE/GA/PSO on the four UCI datasets, 10 runs each. Table S14 |
| `hpo_convergence.json` | Section 6.2 — convergence traces of the same runs |
| `hpo_pso_exact.json` | Section 6.3 — exact-coalition attribution for PSO on the case study. Table 7 |

## Note on the overhead multipliers

`overhead_results.json` records `shap_runs_per_coalition` (the R used for the measurement) next to
`protocol_runs_per_coalition` (the R = 30 of the manuscript protocol), and `n_operators` is taken
from the algorithm rather than from the execution trace. Multipliers should be read together with
the R that produced them.
