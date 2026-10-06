# XMH v1.2.1 — Artifact hygiene

**No reported result changes.** This release fixes defects in the *released artifact*, not in
the research: documentation that advertised superseded numbers, a results directory that
shipped retracted values beside the corrected ones, an unusable install path, and an API that
promised parallelism it never delivered. All of it was found by auditing the deposit against
the manuscript.

Supersedes v1.2.0 (10.5281/zenodo.23023518) as the archive of record.

## Corrected documentation

- **`README.md` advertised retracted results.** The "Key Results" table reported DE at
  50%/50%/0% — the figure the manuscript identifies as an instrumentation artifact and
  corrects to 33.3% each — and the method-comparison table carried superseded cosine and
  rank-agreement values (0.88/0.76). Both tables are now generated from the current data:
  DE at 33.3/33.3/33.3, and per-algorithm cosine / rank agreement for KernelSHAP, QuickSHAP
  and Tracking. The DE entry now also states that the one-third split is a sanity check
  forced by the symmetry axiom, not a discovery about DE.
- The instrumented-algorithm list said "DE, GA, and PSO"; it is six.
- PSO topology growth said 7% to 23%; under the per-run convention the manuscript declares,
  it is 8% to 22%.
- The citation block listed the author as `Anonymous`.
- The documented `ExactCoalitionSHAP.compute(...)` call does not exist; the method is
  `explain(algorithm_class, algorithm_kwargs, objective_func)`.

## Removed data that contradicted the paper

`results/` is deleted. It was tracked in git and still contained the pre-v1.1.0 coalition
values, including DE's retracted 50/50/0, sitting beside the corrected copies in
`reproduce/data/` with nothing to distinguish them — while the README pointed readers at
`results/` as the authoritative coalition values. `reproduce/data/` is now the only result
set in the archive.

## Installable, with the dependencies it actually uses

- Added `pyproject.toml`, so `pip install -e .` works. The documented install path previously
  raised `ModuleNotFoundError: No module named 'xmh'`, because there was no package to
  install and `PYTHONPATH` was documented nowhere.
- `requirements.txt` declared only NumPy (with `scipy` commented out) while the repository
  imports `scipy`, `pandas`, `scikit-learn`, `xgboost`, `opfunu`, `shap` and `seaborn`. All
  are now declared and grouped into optional extras (`experiments`, `casestudy`, `figures`,
  `crosscheck`, `dev`).
- The R toolchain behind the manuscript figures — ten packages — was declared nowhere. It is
  now stated in both READMEs.

## Removed a false promise of parallelism

`ExactCoalitionSHAP`'s ancestor accepted `parallel` and `n_jobs` ("-1 for all cores"),
documented them, stored them, and never read them: a caller asking for all cores silently got
one. `ProcessPoolExecutor` was imported and never called. Both are removed rather than left
as a false promise, and the docstring now states plainly that the coalition loop is executed
serially and how to distribute it ((coalition, run) pairs are independent, seeded
`base_seed + r`).

## Reproduction documentation

`reproduce/README.md` described seven Python figures and three JSONs; the manuscript's figures
come from the R pipeline and `data/` holds 22 files. It now documents the R pipeline, lists
its package requirements, and carries the **figure-number mapping** the filenames do not
reveal (manuscript Figure 2 is `fig7_summary_all_algorithms.pdf`, Figure 3 is
`fig1_ga_shapley_heatmap.pdf`, and so on).

## Unchanged

The data in `reproduce/data/`, the experiment scripts and every number in the manuscript. The
test suite passes as before (43 passed, 1 skipped).
