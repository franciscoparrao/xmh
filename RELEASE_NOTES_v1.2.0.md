# XMH v1.2.0 — Budget sweep of SHADE's parameter adaptation

Release accompanying the manuscript:

> "XMH: Explainable Metaheuristics via Exact-Coalition Operator-Level Shapley Value Attribution"
> F. Parra Ortiz and M. Marín, Universidad de Santiago de Chile. Under submission.

**This release supersedes v1.1.0 as the archive of record for the manuscript.** v1.1.0 does not
contain the data behind Section 5.10 (Sweep C), Table S15 or Figure 6, which the manuscript added
after that archive was published. No previously reported number changes.

## New experiment

- `experiments/exp17_shade_budget.py` — budget sensitivity of SHADE's parameter-adaptation
  attribution. Four functions at D = 30, each re-run at G ∈ {200, 500, 1000, 2000} (10,000 to
  100,000 FES), R = 30 runs per coalition, all 16 coalitions. The functions are chosen so that the
  adaptation share has both signs at the base budget: Ackley and Griewank negative, Rastrigin and
  Schwefel positive.

  The result: extending the budget tenfold does not turn the negative share positive. The only
  sign change runs the other way (Rastrigin, +1.46% → −1.16%), and the count of negative cells
  rises from two of four to three of four. All 16 confidence intervals exclude zero.

  The G = 200 column repeats the configuration of Section 5.6, so it reproduces that section's
  published values and can be checked against main Table 6.

## Data

- `reproduce/data/exp17_shade_budget.json` — per cell: the raw value of every coalition across the
  30 runs, the Shapley values, the 95% confidence intervals, and the shares under both
  normalizations (`share_pct_signed_denominator`, which is the manuscript's convention from
  Section 5.1, and `share_pct_abs_denominator`, the one Section 5.5 uses for CMA-ES). The two
  differ by up to 0.7 pp wherever an operator is negative, so they are named rather than implied.

## Determinism and architecture

Results are bit-reproducible on **aarch64** with the same seeds: re-running the SHADE protocol on a
Hetzner CAX31 (8 ARM cores, Ubuntu 24.04, NumPy 1.26.4) reproduces the published coalition values
and Shapley values of `shade_shapley_results.json` exactly (difference 0.0, raw coalition values
included).

On **x86-64** the same code with the same seeds diverges at the level of floating-point
association: up to 6.7e-4 absolute on Ackley and 3.8e-2 on Griewank at D = 30, which is ~5e-5 to
7e-5 relative to the largest Shapley value in the instance. No qualitative result depends on this,
but a reader reproducing the numbers to the last decimal should use ARM. Every experiment in this
archive was produced on ARM.

## Instrumentation note

`exp17` runs the coalition game with `InstrumentationLevel.LIGHT`. The game consumes only the final
best fitness of each run, while the event-level trace that `STANDARD` accumulates is what the
attribution methods of Section 5.3 need. Verified on Ackley D = 30 at G = 200: `LIGHT` and
`STANDARD` return identical Shapley values (difference 0.0), and the code is deterministic across
runs with the same seed. Without it the sweep does not fit in memory at G = 2000.

## Figures

- `reproduce/generate_paper_figures.R` — adds `fig10_shade_budget.pdf` (main Figure 6).
