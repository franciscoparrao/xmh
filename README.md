# XMH: eXplainable MetaHeuristics via Exact Operator-Level Shapley Value Attribution

A Python framework for explaining metaheuristic algorithm behavior through game-theoretic operator attribution.

## Key Idea

Metaheuristics use few operators (n = 2-4), making **exact Shapley value** computation feasible (2^n <= 16 coalitions). XMH replaces inactive operators with neutral (identity) counterparts and evaluates all coalitions to obtain rigorous, axiomatic operator attribution.

## Features

- **Exact Operator-Shapley**: Computes exact Shapley values, confidence intervals, and interaction indices for operator attribution
- **QuickSHAP**: Lightweight normalized heuristic computable from a single run (validated against exact ground truth)
- **KernelSHAP**: Sampling-based Shapley approximation adapted for operator games
- **Tracking Attribution**: EvoMapX-style direct tracking of operator improvements
- **Instrumented Algorithms**: DE, GA, PSO, CMA-ES, SHADE and JADE with configurable operator-level logging
- **Neutral Operators**: Identity replacements for coalition evaluation

## Installation

```bash
git clone https://github.com/franciscoparrao/xmh.git
cd xmh
pip install -e .                      # core framework
pip install -e '.[experiments,casestudy,figures,dev]'   # everything the paper uses
```

## Quick Start

### Run an instrumented algorithm

```python
from xmh.algorithms.instrumented_de import InstrumentedDE
from xmh.benchmarks.functions import get_benchmark

benchmark = get_benchmark("rastrigin")
dim = 10
lower, upper = benchmark.get_bounds(dim)

de = InstrumentedDE(
    dim=dim, bounds=(lower, upper),
    pop_size=50, max_generations=100,
    F=0.5, CR=0.9, seed=42
)
result = de.run(benchmark, problem_name="Rastrigin")
trace = result['trace']
print(trace.summary())
```

### Compute Exact Shapley Values

```python
from xmh.explanation.operator_shap import ExactCoalitionSHAP

exact = ExactCoalitionSHAP(n_runs_per_coalition=30, base_seed=42)
result = exact.explain(AlgorithmClass, algorithm_kwargs, objective_func)
print(result['shapley_values'])       # {operator: value}
print(result['confidence_intervals']) # {operator: (low, high)}
print(result['interaction_indices'])  # {(op_i, op_j): value}
```

### QuickSHAP from a single trace

```python
from xmh.explanation.operator_shap import QuickSHAP

quick = QuickSHAP()
explanation = quick.explain_from_trace(trace)
print(explanation.summary())
```

## Project Structure

```
xmh/
├── core/                        # Core infrastructure
│   ├── instrumentation.py       # Base InstrumentedAlgorithm, Operator, NeutralOperator
│   ├── trace_logger.py          # TraceLogger, events, snapshots, operator stats
│   └── checkpoint.py            # State checkpointing
├── algorithms/                  # Instrumented algorithm implementations
│   ├── instrumented_de.py       # DE/rand/1/bin (mutation, crossover, selection)
│   ├── instrumented_ga.py       # GA with SBX + polynomial mutation
│   └── instrumented_pso.py      # PSO gbest with constriction
├── explanation/                 # Attribution methods
│   ├── operator_shap.py         # ExactCoalitionSHAP, QuickSHAP, KernelSHAP
│   ├── tracking_attribution.py  # EvoMapX-style tracking
│   ├── counterfactual.py        # Counterfactual analysis
│   └── attribution.py           # Simple attribution methods
├── benchmarks/                  # Test functions (9 standard benchmarks)
│   └── functions.py
├── experiments/                 # Experiment scripts
│   ├── exp4_exact_shapley.py    # Experiment 1: DE/GA/PSO, 19,440 runs
│   ├── exp5_method_comparison.py # Experiment 2: attribution methods vs exact
│   ├── exp6_overhead.py         # Experiment 3: wall-clock overhead
│   ├── exp7_cmaes_shapley.py    # Experiment 4: CMA-ES
│   ├── exp10_shade_shapley.py   # Experiment 5: SHADE at n = 4
│   ├── exp12_jade_shapley.py    # Experiment 5: JADE corroboration
│   ├── exp11_cec2017.py         # Experiment 6: CEC2017
│   ├── exp8_discriminant.py     # Experiment 7: complementarity with fANOVA
│   ├── exp13_neutral_sensitivity.py # Experiment 8: neutral-operator sensitivity
│   ├── exp9_budget.py           # Experiment 9: budget sweeps A and B
│   └── exp17_shade_budget.py    # Experiment 9: Sweep C, SHADE budget
├── reproduce/
│   ├── data/                    # Raw per-run coalition values v(S) behind every
│   │                            # number in the manuscript (see its README for the
│   │                            # file-to-table mapping)
│   ├── generate_paper_figures.R # The pipeline that produced the manuscript figures
│   └── figures_R/               # Its output
├── tests/
│   └── test_core.py             # 43 unit tests
└── visualization/
    └── plots.py
```

## Reproducing Paper Results

### Run all tests
```bash
python -m pytest tests/ -v
```

### Exact Shapley experiment (~7.5 hours)
```bash
python -m xmh.experiments.exp4_exact_shapley
```

### Method comparison (~4 hours)
```bash
python -m xmh.experiments.exp5_method_comparison
```

### Overhead analysis (~30 min)
```bash
python -m xmh.experiments.exp6_overhead
```

### Quick mode for testing (minutes)
```bash
python -m xmh.experiments.exp4_exact_shapley --quick
```

## Validating the instrumentation first

The framework fails quietly. If an operator is registered but the evolution loop never applies it
through its registered handle, or no neutral exists for its type, neutralizing it changes nothing
and its Shapley value comes out as exactly zero — indistinguishable from a legitimate null player.
This happened twice during development (DE's greedy acceptance inlined rather than routed; PSO
registering `velocity_update` against a factory that knew only `velocity`).

The empty-coalition check the method performs does **not** catch it: with every operator neutralized
the trial equals its target, nothing is accepted, and v(0) = 0 holds either way.

```python
from xmh.core.validation import validate_instrumentation
report = validate_instrumentation(InstrumentedDE, kwargs, objective_func)
```

It neutralizes each operator in turn against the full coalition over shared seeds and flags any
whose neutralization leaves every run identical. A flag is not proof of a bug — a real null player
behaves the same way — but it means that operator's attribution is not interpretable until its
registration has been checked by hand.

Note that `disable_operator()` on its own does **not** survive `run()`, which re-initializes the
operator registry. Coalition evaluation patches the initializer to reapply it; calling the public
method directly and then running gives you the full algorithm back, silently.

## Key Results

| Algorithm | Operator 1 | Operator 2 | Operator 3 | Pattern |
|-----------|-----------|-----------|-----------|---------|
| **DE** | Mutation: 80% | Crossover: -7% | Selection: 27% | Mutation-dominated, crossover redundant at CR=0.9 |
| **GA** | Crossover: 41% | Mutation: 43% | Selection: 16% | Landscape-dependent |
| **PSO** | Velocity: 42% | Position: 42% | Topology: 16% | Dimension-dependent |

- **DE**: Mutation dominates (57-97%, median 80%); binomial crossover at CR=0.9 receives a negative share in 25 of 27 configurations because the {mutation, selection} coalition already matches the full algorithm. Earlier versions (<= v1.5) reported an exact one-third split: that was an artefact of inert neutral operators (acceptance retained the target, crossover discarded the mutant), fixed in v1.6 to follow Definition 2 / Table S1, the same neutrals SHADE and JADE use. See `experiments/exp21_de_neutral_selection.py`.
- **GA**: Selection importance ranges from 0% (Zakharov) to 34% (Ackley D=50)
- **PSO**: Topology contribution grows from 8% (D=10) to 22% (D=50)
- **SHADE / JADE** (n = 4): mutation current-to-pbest dominates (55-100%); the parameter-adaptation share is small, median -0.7% (SHADE) and -1.1% (JADE), within 1 pp of zero in 12 of 27 configurations and substantially positive only on Rastrigin and Schwefel. Under the fixed (F, CR) = (0.5, 0.5) neutral used up to v1.5 the share was 3-5 pp more negative; v1.6.0 freezes the memory and keeps SHADE's sampling (Definition 2)

## Method Comparison

Cosine similarity / rank agreement against the exact-coalition values, averaged over the
27 (function, dimension) instances per algorithm.

| Method | DE | GA | PSO | Cost |
|--------|------|------|------|------|
| KernelSHAP | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | ~30x single run |
| QuickSHAP | 0.82 / 1.00 | 0.92 / 0.48 | 0.02 / 0.00 | O(1) |
| Tracking | 0.82 / 1.00 | 0.92 / 0.52 | 0.02 / 0.00 | O(1) |

At n <= 4, KernelSHAP with all 2^n coalitions is algebraically equivalent to exact-coalition
Shapley, so its row is an upper bound rather than a separate method. QuickSHAP and Tracking
recover DE's operator ordering but not its magnitudes, and both collapse on PSO, where the
execution trace never credits the position update.

**Cost of the exact method**: R * 2^n complete runs, i.e. ~215x a single run at the protocol's
R = 30. Attaching the instrumentation costs 7.1% mean wall-clock overhead (median 7.9%, std
20.9% on sub-second runs); computing a trace-based attribution from an existing trace adds
< 0.5%. The coalition loop is embarrassingly parallel but **is executed serially** in this
release.

## Benchmark Functions

Sphere, Rosenbrock, Zakharov, Dixon-Price (unimodal); Rastrigin, Schwefel, Ackley, Griewank, Levy (multimodal). Dimensions: D = {10, 30, 50}.

## Requirements

Runtime (core framework):
- Python >= 3.10
- NumPy >= 1.20

Experiments and the case study additionally need `scipy`, `pandas`, `scikit-learn`,
`xgboost` and `opfunu`; figure generation needs either `matplotlib` (legacy Python
pipeline) or R >= 4.2 with `ggplot2`, `patchwork`, `dplyr`, `tidyr`, `jsonlite`,
`scales`, `viridis`, `stringr`, `forcats` and `rlang` (the pipeline that produced the
manuscript figures). Tests need `pytest`. See `pyproject.toml` for the pinned sets.

## Citation

If you use XMH in your research, please cite:

```bibtex
@article{xmh2026,
  title = {XMH: Explainable Metaheuristics via Exact Operator-Level
           Shapley Value Attribution},
  author = {Parra Ortiz, Francisco and Mar\'in, Mauricio},
  year = {2026},
  note = {Under review},
  doi = {10.5281/zenodo.22731834}
}
```

## License

MIT License. See [LICENSE](LICENSE).
