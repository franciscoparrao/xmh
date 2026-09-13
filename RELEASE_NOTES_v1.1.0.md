# XMH v1.1.0 — Corrected instrumentation and extended experiments

Release accompanying the manuscript:

> "XMH: Explainable Metaheuristics via Exact-Coalition Operator-Level Shapley Value Attribution"
> F. Parra Ortiz, Universidad de Santiago de Chile. Under submission.

**This is the release that reproduces the results reported in the manuscript. Use it instead of
v1.0.0**, whose instrumentation contained the defects listed below.

## Corrected instrumentation (changes reported results)

- **DE greedy selection is now routed through the registered operator.** In v1.0.0 the acceptance
  comparison was inlined in the evolution loop, so `selection_op` was fetched but never applied.
  Neutralizing selection therefore had no effect and its Shapley value was zero by construction,
  yielding a 50/50/0 split. With the operator routed correctly, DE's three operators are totally
  interdependent and the split is 33.3/33.3/33.3 — no proper subcoalition improves fitness at all.
- **The neutral-operator factory raises instead of silently returning a generic neutral.** A
  missing operator type used to fail quietly; it now fails loudly. This is the fix that closes the
  whole class of defects, since all four bugs in this release shared that failure mode.
- **PSO neutral operators cover every registered type**, with no silent fallback.
- **Per-run Shapley convention applied consistently**: percentage shares are the mean of the
  per-run shares, not the share of the mean, which is what the reported confidence intervals
  require.

## Overhead measurement

`exp6_overhead.py` derived the number of operators from the execution trace, so operators that
register no improvements of their own — DE's greedy selection, PSO's position update — were not
counted, and those algorithms were measured with 4 coalitions instead of 8. The operator count now
comes from the algorithm itself, and the number of runs per coalition used for the measurement is
recorded alongside the protocol value.

## New experiments

- `exp13_neutral_sensitivity.py` — sensitivity of the attributions to the neutral-operator
  construction (7 configurations x 2 functions x 2 dimensions).
- `exp14_interaction_ci.py` — confidence intervals for the pairwise interaction indices.
- `exp15_hpo_exact_pso.py` — exact-coalition attribution for the hyperparameter-optimization case
  study, replacing the trace-based diagnosis.
- `exp16_cec_composition.py` — CEC2017 composition functions.
- `exp5b`/`exp5c` — KernelSHAP identity check and method re-comparison over the full operator set.

## Naming

`ExactOperatorSHAP` is now `ExactCoalitionSHAP`, matching the manuscript's terminology. The old
name remains available as a backwards-compatible alias.

## Figures

Figure generation is available both in Python (`reproduce/generate_paper_figures.py`) and in
R/ggplot2 (`reproduce/generate_paper_figures.R`); the R port is the one deployed for the
manuscript.

## How to reproduce the manuscript figures

```bash
git clone https://github.com/franciscoparrao/xmh
cd xmh
pip install -r requirements.txt
cd reproduce
Rscript generate_paper_figures.R     # or: python generate_paper_figures.py
```

Figures land under `reproduce/figures_R/` (R) or `reproduce/figures/` (Python). The raw result
JSONs the manuscript tables and figures derive from are under `reproduce/data/`.

## Citation

If you use this code, please cite the accompanying manuscript and this software release.
