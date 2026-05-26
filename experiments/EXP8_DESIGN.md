# Experiment 8 — Discriminating XMH from Hyperparameter-Sensitivity Analysis

## Motivation

Reviewer #3 of the SWEVO submission argued that "operators are the abstract
form of hyperparameters" and that tools like IOHxplainer (fANOVA-style
hyperparameter analysis) and EvoMapX already cover XMH's contribution.

This experiment provides empirical evidence that **XMH and fANOVA-style
analyses answer different questions, neither subsuming the other**.

## Hypotheses

- **H1** (categorical asymmetry): When only the *categorical* choice of crossover
  operator varies and all continuous hyperparameters are held fixed,
  fANOVA-on-hyperparameters returns no signal (nothing to decompose),
  while XMH reports distinct, statistically separable Shapley values per
  operator type. fANOVA can be retrofitted to operator-type as a categorical
  factor, but its output ("variance of fitness explained by operator type")
  is semantically distinct from XMH's ("share of within-run improvement
  attributable to the crossover operator").

- **H2** (continuous symmetry): When the categorical operator is fixed and
  only a continuous hyperparameter (`crossover_rate`) varies, fANOVA reports
  graded sensitivity to the hyperparameter while XMH reports stable Shapley
  values that change smoothly with the hyperparameter (but do not "switch off").

The combined evidence shows the tools are **complementary**: fANOVA isolates
*design-choice sensitivity*; XMH isolates *operator contribution within a
fixed design*.

## Setup (shared across parts A and B)

| Setting | Value |
|---------|-------|
| Algorithm | Instrumented GA |
| Function | Rastrigin |
| Dimension | 30 |
| Population size | 50 |
| Max generations | 200 |
| Selection | Tournament (k=3) |
| Mutation | Polynomial, rate=0.1 |
| Bounds | [-5.12, 5.12] |
| Runs per configuration | 30 (independent seeds 1..30) |

Rastrigin D=30 is multimodal, sensitive to operator choice (paper Table 5
already shows GA selection at 30%, crossover at 37%, mutation at 33% with
default SBX). This makes operator-type effects detectable but not dominant.

## Part A: Vary categorical operator (cx_type), fix hyperparameters

**Independent variable**: `crossover_type` ∈
{`sbx`, `blx_alpha`, `arithmetic`, `uniform`, `two_point`, `single_point`}

**Fixed**: `crossover_rate = 0.9`, mutation as above.

**Analyses to run**:

1. **fANOVA-on-hyperparameters** (the analysis IOHxplainer-style tools natively perform):
   - Input: 6 × 30 = 180 (final_fitness, hyperparameter_vector) pairs where
     the hyperparameter vector is identical across all 180 rows.
   - Output: Trivially zero variance attributable to hyperparameters
     (no variation to decompose).
   - **Headline number**: "0% of fitness variance is explained by hyperparameter
     variation."

2. **fANOVA-on-operator-type** (the closest workaround):
   - Treat `crossover_type` as a categorical factor.
   - One-way ANOVA: `final_fitness ~ C(crossover_type)`.
   - Report η² (proportion of total variance explained by cx_type).
   - **Headline number**: "η² = X%, p < Y" — quantifies *between-cx_type variance*.

3. **XMH (ExactOperatorSHAP)** computed *per cx_type configuration*:
   - For each cx_type, run the 8 coalitions (2³) × 30 runs.
   - Compute Shapley φ_sel, φ_cx, φ_mut for that cx_type.
   - **Headline numbers**: 6 different Shapley triples; show that φ_cx
     varies from ~50% (SBX) down to ~15-25% (single_point, uniform), with
     non-overlapping 95% CIs.

**Discriminating claim**: Both fANOVA-categorical (η²) and XMH "detect that
operator type matters", but they answer different questions:
- η² answers: "If you randomize cx_type across runs, what fraction of
  fitness variance disappears?"
- XMH answers: "Given that you fixed cx_type to SBX, what fraction of the
  improvement within a run is attributable to crossover?"

The first is a *between-design* question; the second is a *within-design*
question. The reviewer's "operator = hyperparameter" claim conflates them.

## Part B: Fix categorical operator (SBX), vary cx_rate

**Independent variable**: `crossover_rate` ∈ {0.1, 0.2, 0.3, ..., 0.9} (9 values)

**Fixed**: `crossover_type = "sbx"`, mutation as above.

**Analyses to run**:

1. **fANOVA-on-hyperparameters**:
   - Input: 9 × 30 = 270 (final_fitness, cx_rate) pairs.
   - Fit a 1D fANOVA / regression decomposition (`fitness ~ f(cx_rate)`).
   - Use either sklearn's permutation_importance on a trained regressor,
     or a manual functional ANOVA via spline regression.
   - **Headline number**: "cx_rate explains Z% of fitness variance."

2. **XMH (ExactOperatorSHAP)** computed *per cx_rate*:
   - 9 × 8 coalitions × 30 runs.
   - **Headline numbers**: 9 Shapley triples; expectation is that
     φ_cx grows monotonically with cx_rate (since the operator is applied
     more often), but the operator's *identity* and qualitative role
     remains constant — there is no regime where crossover suddenly
     becomes "a different operator."

**Discriminating claim**: fANOVA detects the *strength* of the cx_rate knob;
XMH shows *what role* the crossover operator plays at each setting. Both
are useful; neither substitutes for the other.

## Total compute budget

- Part A: 6 cx_types × 8 coalitions × 30 runs = 1,440 runs
- Part B: 9 cx_rates × 8 coalitions × 30 runs = 2,160 runs
- Total: 3,600 runs

Each run ≈ 0.5–1.0 s on Rastrigin D=30 with these settings. Total wall-clock
~1–2 hours (parallelizable, but exp4 ran ~7.5 h for 17K runs without
parallelization).

## Implementation requirements

- **`instrumented_ga.py`**: add three crossover variants currently missing:
  - `arithmetic` (whole arithmetic recombination, α=0.5)
  - `two_point` (two-point crossover on real-valued vectors treated as
    indexed positions)
  - `single_point` (single-point crossover, same convention)
- Verify the existing `CROSSOVER_TYPES` list is updated and that
  `simulated_binary` alias is documented or removed.

- **`exp8_discriminant.py`** (new):
  - Two functions: `run_part_a()`, `run_part_b()`.
  - Each saves results to `results_discriminant/exp8_part_{a,b}.json`.
  - Uses `ExactOperatorSHAP` from `xmh.explanation.operator_shap`.
  - Uses `scipy.stats.f_oneway` for one-way ANOVA in Part A.
  - Uses `scipy.stats.spearmanr` and a polynomial fit in Part B for
    sensitivity analysis (avoids the `fanova` package which is not installed).

- **`xmh/reproduce/data/exp8_*.json`**: ship raw results once the experiment
  is done so the discriminating figure can be regenerated by reviewers.

## Output figure

A 2-panel figure for the manuscript:

- **Left (Part A)**: grouped bar chart. X-axis: 6 cx_types. Three bars per
  group (φ_sel, φ_cx, φ_mut) with 95% CIs. Annotation: "η² (one-way ANOVA
  on cx_type) = X%". Title: "Categorical operator choice: XMH reveals
  per-type composition; ANOVA reports between-type variance."

- **Right (Part B)**: line chart. X-axis: cx_rate from 0.1 to 0.9. Three
  lines (φ_sel, φ_cx, φ_mut) with shaded CIs. Annotation: "fANOVA
  sensitivity to cx_rate = Z%". Title: "Continuous knob: XMH tracks
  operator role; ANOVA reports knob sensitivity."

Caption length: ~150 words; explicitly state the complementarity claim.

## Acceptance criteria

The experiment is "successful" (i.e., supports the paper claim) iff:

1. **Part A**: At least 3 of 6 cx_types have non-overlapping 95% CIs on φ_cx.
   The qualitative range of φ_cx is at least 15 percentage points (e.g.,
   highest ~50%, lowest ~30%).

2. **Part B**: φ_cx changes monotonically with cx_rate in at least 7 of the
   9 settings (we allow noise). Spearman ρ(cx_rate, φ_cx) ≥ 0.6.

3. **Discriminating evidence**: η² (Part A) > 30% AND fANOVA sensitivity
   to cx_rate (Part B) > 30%. Both fANOVA analyses detect *something*; the
   complementarity argument requires both tools to "see" their respective
   signals, otherwise we're just comparing XMH against a null tool.

If acceptance criteria fail, the experiment outcome is itself informative —
it would mean that operator type does not strongly discriminate under
these conditions, which would weaken the paper's broader claim. In that
case, increase D, switch to a harder multimodal function (Schwefel), or
include all 9 benchmark functions.
