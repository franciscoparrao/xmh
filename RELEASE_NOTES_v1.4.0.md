# XMH v1.4.0 — The complementarity claim against the published tool

Section 5.8 of the manuscript argues that XMH and fANOVA-style hyperparameter analysis rank
operator choices differently. Until now the comparator was a one-way ANOVA eta^2 built by us
for that experiment. This release runs the comparison against **IOHxplainer** (van Stein et
al., ACM TELO 2025), the reference implementation of that kind of tooling, and ships the
container it needs.

Supersedes v1.3.0 (10.5281/zenodo.23218303) as the archive of record. No previously reported
number changes.

## Experiment 7b — IOHxplainer on the Experiment 7 setup

`experiments/exp20_iohxplainer.py`, data in `reproduce/data/exp20_iohxplainer.json`.

The paper's GA is wrapped as an IOH-compatible optimizer and handed to IOHxplainer with a
configuration space of four hyperparameters: the crossover type over the same six variants as
Experiment 7, plus crossover rate, mutation rate and tournament size. The four knobs are
deliberate — given a single categorical factor the CatBoost surrogate has nothing to
decompose, and the comparison would be unfair to the tool. Grid of 48 configurations on BBOB
f3 (Rastrigin) at D = 30, two instances, five repetitions, budget 10,000 FES: 480 runs.

**IOHxplainer sees the factor.** It ranks crossover type as the most influential of the four
hyperparameters (mean |SHAP| 0.0035, against 0.0031 for mutation rate and 0.0011 for crossover
rate).

**The two tools rank the variants differently.** IOHxplainer: uniform > SBX > two-point >
single-point > BLX-alpha > arithmetic. XMH by crossover Shapley share: BLX-alpha (41.0%) > SBX
(37.7%) > arithmetic (36.2%) > uniform (28.2%) > two-point (25.9%) > single-point (23.9%). The
extremes invert. Spearman rho = -0.20.

**What it establishes.** With n = 6 the correlation carries p = 0.70, so the defensible
statement is that no agreement is detectable, not that the rankings are negatively related.
The divergence is expected by construction — IOHxplainer attributes predicted performance to a
configuration choice, XMH attributes realized within-run improvement to an operator — and what
the experiment adds is that it survives when the comparator is the published tool.

## The container is part of the artifact

`experiments/Dockerfile.iohxplainer`. IOHxplainer 0.9.1 pins numpy 1.24.4, pandas 2.0.3,
scikit-learn 1.3.0, xgboost 1.7.6 and modcma 1.0.4; none has a Python 3.12 wheel, so the
experiment cannot be run in a current environment and installing it fails while building numpy
from source. Python 3.10 resolves the pins natively and XMH itself runs unchanged against
numpy 1.24. Without this file the experiment is not reproducible, so it ships with it.

A second practical note recorded in `reproduce/README.md`: IOHxplainer's `explain()` writes a
plot and returns nothing, so the values reported in the manuscript come from re-running its
own pipeline (CatBoost surrogate, `shap.TreeExplainer`) over the results it saves.

## Not included

EvoMapX, the other external baseline a reviewer would want, is still not run. Its code is not
available to us, and the manuscript says so rather than substituting another stand-in.
