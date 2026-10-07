# XMH v1.3.0 — Two robustness experiments

Adds the two experiments that test the design choices most likely to have produced the
manuscript's headline findings. Both were run because a reviewer could reasonably argue the
findings were artifacts of those choices. **No previously reported number changes**; what
changes is how confidently two of them can be stated.

Supersedes v1.2.1 (10.5281/zenodo.23175745) as the archive of record.

## Experiment 5b — the adaptation result at the specified memory size

`experiments/exp18_shade_memory.py`, data in `reproduce/data/exp18_shade_memory.json`.

The negative Shapley share of SHADE's parameter adaptation is a claim about the
success-history mechanism, and that mechanism is parameterized by the memory size H.
Experiment 5 runs H = 5; SHADE as specified uses H = |P|. This experiment repeats the whole
27-configuration grid at H = |P| = 50 with every other setting held fixed, including the base
seed, so the two runs differ in exactly one parameter.

**The finding does not depend on the memory size.** Adaptation is negative in 22 of 27
configurations at H = |P|, the same count as at H = 5, with a median share of -1.97% against
-2.62%. Two cells change sign and both run positive to negative (Rastrigin at D = 30 and
D = 50). Two further apparent changes are Zakharov at D = 30 and D = 50, where the share moves
between -0.00% and +0.01%; that cell is degenerate and we do not count it as a reversal.

The archived JSON carries the H = 5 value alongside each H = |P| value, so the comparison is
checkable without opening a second file.

## Experiment 9b — dimension separated from budget

`experiments/exp19_fes_per_dim.py`, data in `reproduce/data/exp19_fes_per_dim.json`.

Experiment 1 holds the budget at 5,000 FES while D grows, so FES per dimension falls fivefold
and the dimension manipulation is confounded with a budget reduction. This experiment reruns
GA and PSO with the budget scaled to the dimension (FES = 1000*D) on four functions, so FES
per dimension is constant and only D varies.

**The trends are dimension effects.** GA's selection share and PSO's topology share both grow
monotonically with D in all four functions: eight of eight.

**The magnitudes are budget-dependent, and the distortion is algorithm-specific.** For PSO the
confound is almost immaterial (topology grows +23.0 pp at constant FES/D against +21.5 pp in
the main grid on Sphere). For GA it is large and runs in both directions: on the smooth
functions the main grid overstates the effect roughly eightfold (Sphere +2.8 pp here against
+21.8 pp there), while on Ackley it understates it (+15.3 against +8.1 pp).

The archived JSON carries the main-grid value alongside each constant-budget value.

## Unchanged

Every other data set, every experiment script, and every number in the manuscript that
predates these two experiments. Values remain bit-reproducible on aarch64 with the archived
seeds; both new experiments were computed there.
