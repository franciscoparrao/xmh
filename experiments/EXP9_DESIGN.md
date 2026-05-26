# Experiment 9 — Budget Sensitivity (Pop / Max-Gen Effect on Operator Attribution)

## Motivation

Reviewer #1 (SWEVO):
> I suggest further discussing the effect of the population size and the
> maximum number of generations (or FES budget) on the reported results.
> For instance, it would be useful to analyze how operator attribution
> changes when convergence is achieved early versus late in the optimization
> process. Including convergence plots would strengthen the analysis.

## Headline questions

1. **Does Shapley attribution stabilize with budget?** Do attributions
   computed at short budgets (e.g., gen=50) differ meaningfully from those
   at large budgets (gen=500)? If they do, the original paper's
   gen=200-derived attributions could be artifacts of the chosen budget.
2. **Does population size shift relative operator importance?** Larger
   populations dilute selection pressure but provide more crossover
   diversity. Operator attribution should shift accordingly, and XMH
   should make this visible.
3. **What do per-generation convergence trajectories look like across
   coalitions?** Coalition-conditional convergence plots make the
   neutral-operator mechanism concrete.

## Design

**Fixed**: GA (SBX, polynomial mutation, tournament selection), $D{=}30$,
crossover\_rate $=0.9$, mutation\_rate $=0.1$, tournament size 3, 30 runs
per coalition. Two benchmark functions: Sphere (unimodal, fast
convergence) and Rastrigin (multimodal, slow convergence).

**Sweep A — Max generations**: with `pop_size = 50`,
`max_generations ∈ {25, 50, 100, 200, 500}`.

**Sweep B — Population size**: with `max_generations = 200`,
`pop_size ∈ {10, 25, 50, 100, 200}`.

The configuration `(pop=50, gen=200)` appears in both sweeps and is
computed once. The Rastrigin case at this configuration also matches
exp4's existing GA/Rastrigin/D=30 result; reusing it is acceptable.

**Convergence traces**: For each `(function, level)` configuration, the
8 coalition × 30 run set already includes 30 runs of the *full coalition*
(all operators active). Per-generation best-fitness is extracted from
those runs' trace loggers to produce convergence curves with 30-run
median ± IQR shading.

## Compute budget

| Sweep | Configs | Coalitions/cfg | Runs/coal | Total runs |
|-------|---------|----------------|-----------|------------|
| A     | 5 × 2 = 10 | 8           | 30        | 2{,}400    |
| B     | 5 × 2 = 10 | 8           | 30        | 2{,}400    |
| Shared baseline    | -1 × 2 = -2 |          |           | -480      |
| **Total** |     |                |           | **4{,}320** |

The compute scales with `pop × gen`. The most expensive single config is
Sweep A's gen=500 with pop=50 (12.5× the cost of pop=50/gen=200), and
Sweep B's pop=200 with gen=200 (4× the baseline). All others are equal
to or cheaper than the exp8 baseline (~16 min per config on CAX31 ARM).

Estimated wall-clock with 8-way parallelism: 60--90 minutes total.

## Outputs

`results_budget/exp9_sweep_a.json` and `exp9_sweep_b.json` with the same
structure as `exp8_part_*.json`:
- `per_config`: dict keyed by level → `{shapley_values, confidence_intervals,
  final_fitness_per_run, convergence_traces (30 × max_gen array)}`.
- Per-sweep summary statistics (per-operator Shapley by level).

## Acceptance criteria

1. Sweep A: monotonic trend in at least one operator's Shapley with
   gen. The expectation is **selection grows with gen** (more time for
   pressure to act) and **mutation shrinks** as the search converges.
   Final-budget (gen=500) Shapley should be the most stable.
2. Sweep B: detectable change in operator share with pop. Expected
   pattern: smaller populations lean on mutation; larger populations on
   crossover.
3. Convergence plots: show qualitatively different early/late phases,
   ideally with coalition-conditioned curves distinguishing operators.

## Implementation plan

`xmh/experiments/exp9_budget.py`:
- Reuse the same shapley_for_config infrastructure as exp8.
- Add `_extract_convergence_traces` that pulls per-generation best fitness
  from the trace logger of full-coalition runs.
- Save both Shapley and convergence data per config.

Driver `exp9_parallel.py` (uploaded to CAX31): runs all 18 unique configs
in a multiprocessing pool, mirrors exp8's pattern.
