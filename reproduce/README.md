# Reproducing the manuscript figures and tables

The figures in the manuscript were produced by the **R pipeline**, not by the Python one.
Both are kept here; `generate_paper_figures.py` is the earlier version and is retained only
for provenance.

```bash
Rscript generate_paper_figures.R       # writes figures_R/
```

Requires R >= 4.2 with `ggplot2`, `dplyr`, `tidyr`, `jsonlite`, `patchwork`, `scales`,
`viridis`, `stringr`, `forcats` and `rlang`. It reads the JSONs in `data/` and writes PDFs
to `figures_R/`; nothing is recomputed, so it runs in seconds.

## Figure-number mapping

The filenames predate the current figure order, so they do not match the manuscript's
numbering. This table is the mapping.

| Manuscript | File | What it shows |
|---|---|---|
| Figure 1 | *(drawn in LaTeX, no file)* | Framework architecture |
| Figure 2 | `fig7_summary_all_algorithms.pdf` | Operator shares across the six algorithms |
| Figure 3 | `fig1_ga_shapley_heatmap.pdf` | GA Shapley heatmap by function and dimension |
| Figure 4 | `fig8_discriminant.pdf` | Complementarity with fANOVA-style analysis |
| Figure 5 | `fig9_budget.pdf` | Budget sensitivity, Sweeps A and B (GA) |
| Figure 6 | `fig10_shade_budget.pdf` | Sweep C, SHADE parameter adaptation vs budget |
| Figure S3 | `fig4_method_comparison.pdf` | Method comparison, secondary view |
| Figure S4 | `fig2_pso_shapley_heatmap.pdf` | PSO Shapley heatmap |
| Figure S5 | `fig5_overhead.pdf` | Overhead on a log scale |

`fig3_dimensionality_effect.pdf` and `fig6_de_invariance.pdf` are outputs of the pipeline
that the current manuscript does not use.

## Data

`data/` holds the raw result sets: for the Shapley experiments, the value of every coalition
`v(S)` across all 30 runs, not only the derived Shapley values. `data/README.md` maps each
file to the section, table or figure it supports, and records the architecture the values
were computed on.

## Reproducing the numbers rather than the plots

Every quantitative claim derives from `data/`. To recompute a published value from the raw
coalitions instead of trusting the summary, apply Eq. 3 of the manuscript per run and then
average the per-run shares, which is the convention declared in Section 5.1. Re-running the
experiments themselves (`../experiments/exp*.py`) costs the budgets stated in the paper and
reproduces these JSONs bit-for-bit on aarch64 with the archived seeds.
