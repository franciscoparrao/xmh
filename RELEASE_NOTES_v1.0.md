# XMH v1.0.0 — Initial Release

First public release of the XMH framework, accompanying the manuscript:

> "XMH: Explainable Metaheuristics via Exact Operator-Level Shapley Value Attribution"
> F. Parra Orellana. Submitted to *Information Sciences* (Elsevier).

## What is in this release

- **Core framework** (`xmh/core/`): instrumentation base classes, neutral operators, trace logger.
- **Six instrumented algorithms** (`xmh/algorithms/`): DE, GA, PSO, CMA-ES, SHADE, JADE.
- **Four attribution methods** (`xmh/explanation/`): Exact Operator-Shapley, QuickSHAP heuristic, canonical KernelSHAP (drop-one-column reformulation), tracking-based attribution.
- **Experiment scripts** (`xmh/experiments/`): exp1 (exact Shapley), exp5 (method comparison), exp6 (overhead), exp8 (XMH vs fANOVA discriminant), exp9 (budget sensitivity), exp10 (SHADE Shapley), exp11 (CEC2017 validation), exp12 (JADE Shapley), statistical_tests (Friedman + Holm).
- **Reproducibility** (`xmh/reproduce/`): figure-generation scripts plus the raw result JSONs the manuscript figures and tables are derived from.
- **Tests** (`xmh/tests/`): 44 unit tests, including a cross-check of XMH's canonical KernelSHAP against the official `shap.KernelExplainer` library.

## What is new vs the development history

This is the first tagged release. All experiments reported in the manuscript were produced with the code at this commit.

## How to reproduce the manuscript figures

```bash
git clone https://github.com/franciscoparrao/xmh
cd xmh
pip install -r requirements.txt
cd reproduce
python generate_paper_figures.py
```

Figures land under `reproduce/figures/`. Raw input data is under `reproduce/data/`.

## Citation

If you use this code, please cite the accompanying paper (under review at *Information Sciences*) and this software release (DOI to be issued by Zenodo upon publication of this release).
