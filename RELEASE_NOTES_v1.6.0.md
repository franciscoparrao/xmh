# XMH v1.6.0 — DE's neutral operators follow Definition 2

Changes two neutral operators that DE used, and with them every DE attribution reported
since v1.1.0. No other algorithm is affected: GA, PSO, CMA-ES, SHADE and JADE reproduce
bit for bit.

Supersedes v1.5.0 as the archive of record.

## What was wrong

The manuscript's Definition 2 neutralizes an operator that *chooses* among candidates by
choosing uniformly at random, and an operator that *transforms* a candidate by letting the
candidate through unchanged. Table S1 declared, for DE, that the mutant passes through the
crossover neutral and that SHADE's identical greedy acceptance is neutralized by random
replacement with p = 0.5. The code did neither for DE:

- `NeutralSelectionOperator`, on the target/trial signature, retained the target
  unconditionally: no trial was ever accepted.
- `NeutralCrossoverOperator`, on the DE signature, returned the target and discarded the
  mutant.

With both neutrals inert, every proper coalition of DE had v(S) = v(∅), the three operators
were exchangeable in the game, and the symmetry axiom forced the one-third split that
v1.1.0–v1.5.0 reported as a sanity check. It was a property of the construction, not of DE,
and it contradicted the paper's own definition and the neutral used for the same operator in
SHADE and JADE.

## What changed

`core/instrumentation.py`:

- The DE-style acceptance neutral chooses uniformly between target and trial (p = 0.5),
  using the algorithm's seeded generator. Identical to SHADE's and JADE's.
- The DE-style crossover neutral lets the mutant through (trial ← mutant).

`experiments/exp21_de_neutral_selection.py` recomputes DE's 27 instances under three
explicit constructions (the inert one, acceptance corrected alone, both corrected), each
with its neutrals fixed in a subclass so the comparison is reproducible under any core
version. The coalitions in which acceptance and crossover are both active reproduce the
previous JSON bit for bit in all 27 instances. `merge_de_table_s1.py` writes the adopted
construction into `results_exact_shapley/`; `exp11b_cec2017_de_recompute.py` does the same
for the DE block of the CEC2017 results, which the manuscript does not cite.

## What the numbers now say

Under the adopted construction (R = 30, 8 coalitions, 27 instances):

| | Mutation | Crossover | Selection |
|---|---|---|---|
| D = 10 | 79.7 | −4.2 | 24.5 |
| D = 30 | 80.1 | −7.9 | 27.8 |
| D = 50 | 81.0 | −9.4 | 28.4 |

Mutation dominates in all 27 configurations (57–97 %). Binomial crossover at CR = 0.9 is
negative in 25 of 27, because the {mutation, selection} coalition, in which the mutant
passes through, improves as much as the full algorithm (ratio 0.81–1.27). Selection is
moderate because mutation alone, with trials accepted at random, already reaches 28–100 %
of the full improvement: the characteristic function credits the best point observed, and
random acceptance of rand/1 mutants is a form of sampling. The reading matches SHADE and
JADE, whose mutation likewise dominates and whose neutrals are the same.

Downstream: DE's interaction indices (mutation–selection +38 to +49 %, mutation–crossover
−11 to −25 %), the method-comparison figures for DE (cosine 0.61 against the trace-based
methods, which tie mutation and crossover and never single out the dominant operator), and
the per-operator Friedman test for DE (unchanged at χ² = 54.0: the ranking is the same in
all 27 instances).

## Tests

`test_neutral_crossover_de_style` now asserts that the mutant passes through;
`test_neutral_selection_de_style_uniform` asserts the uniform choice and its
reproducibility under a caller-supplied generator.

## Figures

`reproduce/generate_paper_figures.R` now computes operator shares the way the manuscript's
tables do: per-run Shapley values from each run's coalition values, shares `100·φ_i/Σφ_j`
per run, then the mean over runs. The earlier helper normalised the run-averaged φ by Σ|φ|,
which put DE's mutation at 83 % in Figure 7 against 79.7 % in the text and GA's crossover
and mutation at 41/50 against 39.2/51.6. The R values now reproduce the Python ones to two
decimals in every cell checked.

## SHADE's and JADE's adaptation neutral (same principle, second instance)

The same audit applied to SHADE's fourth operator. Its neutral fixed `(F, CR) = (0.5, 0.5)`
for every individual and performed no memory update. That removes two things at once: the
success-history signal (what the operator is) and the per-individual sampling of `(F, CR)`
around the memory (how DE-family algorithms draw their parameters whether or not they adapt).
`experiments/exp22_shade_adaptation_neutral.py` repeats the Sweep C protocol (four functions,
D = 30, G = 200, 16 coalitions, R = 30) under three constructions. With the memory frozen at
0.5 and SHADE's own sampling kept, the adaptation share moves from −5.5 to −0.5 % on Ackley
and from −2.6 to −0.8 % on Griewank, and rises on Rastrigin (+1.5 → +4.3 %) and Schwefel
(+12.7 → +13.0 %); a uniform draw gives the same picture. The negative shares reported since
v1.2 under the fixed construction were therefore mostly the cost of removing the sampling.

`InstrumentedSHADE._neutral_parameters()` now implements the frozen-memory construction and
`InstrumentedJADE` inherits it. `experiments/exp23_recompute_shade_family.py` recomputes the
SHADE and JADE grids (exp10, exp12), Experiment 5b (exp18) and Sweep C (exp17) under it, with
one evaluation per coalition (97 cells, 10.9 h wall-clock on 14 x86-64 cores). The previous
result sets are kept as `*.bak-preneutralfix-20261010`.

What changes in the numbers (per-run shares, 27 configurations):

- SHADE adaptation: median −2.6 % → −0.7 %; negative in 22 → 21 of 27; within 1 pp of zero in
  12 of 27; mean |share| 3.6 % → 2.6 %. In the 19 cells negative under both constructions the mean
  share moves from −3.1 % to −1.4 %. The five cells with a significantly positive share are the
  same as before (Rastrigin D ≥ 30, Schwefel at every dimension) and larger (Schwefel D = 50:
  +14.8 → +13.4 %, Rastrigin D = 30: +1.5 → +4.3 %).
- JADE adaptation: median −2.6 % → −1.1 %; negative in 23 → 21 of 27; same sign as SHADE in
  25 of 27 cells (r = 0.97). Per-operator mean differences between SHADE and JADE stay under 1 pp.
- Experiment 5b (H = |P|): median −1.97 % → −0.47 %; negative in 22 → 17 of 27; the four sign
  changes all go from a negative share to +0.01 % or less with an interval covering zero.
- Sweep C: the only sign change is now Ackley, from −0.54 % at G = 200 to +0.25 % at G = 2000
  through intervals that cover zero; Rastrigin (+4.3 → +1.6 %) and Schwefel (+13.0 → +5.0 %)
  keep their sign and lose about three fifths of their share; Griewank stays near −0.8 %.
  13 of 16 intervals exclude zero (all 16 did under the fixed construction).
- Friedman over operators: SHADE's adaptation and crossover remain tied at the bottom
  (mean ranks 3.48 and 3.52; JADE 3.41 and 3.59); the algorithm-level Friedman is unchanged.

The manuscript's headline on SHADE changes accordingly: the adaptation share is small and its
sign depends on the landscape, and the negative shares reported since v1.2 were in large part
an artefact of the fixed construction. Full-coalition values, which involve no neutral, reproduce
the aarch64 archive run for run on JADE and 5b and in 798 of SHADE's 810 runs; the other 12
diverge after a floating-point ordering difference flips one selection (x86-64 vs aarch64).
