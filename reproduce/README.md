# Reproducing the Paper Figures

This directory regenerates the seven figures shipped with the paper from the raw
experiment results.

## Contents

- `generate_paper_figures.py` — the figure-generation script.
- `data/` — the three raw-result JSONs read by the script:
  - `exact_shapley_results.json` (Experiment 1: exact Shapley values per algorithm/function/dimension).
  - `method_comparison.json` (Experiment 2: QuickSHAP / KernelSHAP / Tracking vs Exact).
  - `overhead_results.json` (Experiment 3: wall-clock overhead per component).
- `figures/` — created on first run; contains the seven PDFs.

## Usage

From this directory:

```bash
python generate_paper_figures.py
```

Outputs:

- `figures/fig1_ga_shapley_heatmap.pdf`
- `figures/fig2_pso_shapley_heatmap.pdf`
- `figures/fig3_dimensionality_effect.pdf`
- `figures/fig4_method_comparison.pdf`
- `figures/fig5_overhead.pdf`
- `figures/fig6_de_invariance.pdf`
- `figures/fig7_summary_all_algorithms.pdf`

## Regenerating the raw results

The JSONs in `data/` are produced by the experiment scripts in
`xmh/experiments/`:

| JSON | Source script |
|------|---------------|
| `exact_shapley_results.json` | `xmh/experiments/exp4_exact_shapley.py` |
| `method_comparison.json` | `xmh/experiments/exp5_method_comparison.py` |
| `overhead_results.json` | `xmh/experiments/exp6_overhead.py` |

Each experiment fixes random seeds for reproducibility; the published JSONs
shipped with this directory are the exact ones used to render the submitted
figures.

## Requirements

```
numpy
matplotlib
```
