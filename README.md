# XMH: eXplainable MetaHeuristics via Exact Operator-Level Shapley Value Attribution

A Python framework for explaining metaheuristic algorithm behavior through game-theoretic operator attribution.

## Key Idea

Metaheuristics use few operators (n = 2-4), making **exact Shapley value** computation feasible (2^n <= 16 coalitions). XMH replaces inactive operators with neutral (identity) counterparts and evaluates all coalitions to obtain rigorous, axiomatic operator attribution.

## Features

- **Exact Operator-Shapley**: Computes exact Shapley values, confidence intervals, and interaction indices for operator attribution
- **QuickSHAP**: Lightweight normalized heuristic computable from a single run (validated against exact ground truth)
- **KernelSHAP**: Sampling-based Shapley approximation adapted for operator games
- **Tracking Attribution**: EvoMapX-style direct tracking of operator improvements
- **Instrumented Algorithms**: DE, GA, and PSO with configurable operator-level logging
- **Neutral Operators**: Identity replacements for coalition evaluation

## Installation

```bash
git clone https://github.com/franciscoparrao/xmh.git
cd xmh
pip install -r requirements.txt
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
from xmh.explanation.operator_shap import ExactOperatorSHAP

exact = ExactOperatorSHAP(n_runs_per_coalition=30, base_seed=42)
result = exact.compute(algorithm_factory, objective_func, operator_names)
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
│   ├── operator_shap.py         # ExactOperatorSHAP, QuickSHAP, KernelSHAP
│   ├── tracking_attribution.py  # EvoMapX-style tracking
│   ├── counterfactual.py        # Counterfactual analysis
│   └── attribution.py           # Simple attribution methods
├── benchmarks/                  # Test functions (9 standard benchmarks)
│   └── functions.py
├── experiments/                 # Experiment scripts
│   ├── exp4_exact_shapley.py    # Exact Shapley (3 alg x 9 func x 3 dim x 30 runs)
│   ├── exp5_method_comparison.py # QuickSHAP vs KernelSHAP vs Tracking vs Exact
│   └── exp6_overhead.py         # Computational overhead analysis
├── results/                     # Experimental results (JSON/CSV)
│   ├── exact_shapley/           # All coalition values v(S) and Shapley values
│   ├── method_comparison/       # Cosine similarity, rank agreement
│   └── overhead/                # Wall-clock overhead measurements
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

## Key Results

| Algorithm | Operator 1 | Operator 2 | Operator 3 | Pattern |
|-----------|-----------|-----------|-----------|---------|
| **DE** | Mutation: 50% | Crossover: 50% | Selection: 0% | Invariant (coupling) |
| **GA** | Crossover: 41% | Mutation: 43% | Selection: 16% | Landscape-dependent |
| **PSO** | Velocity: 42% | Position: 42% | Topology: 16% | Dimension-dependent |

- **DE**: Perfect 50/50 coupling between mutation and crossover (selection is null player)
- **GA**: Selection importance ranges from 0% (Zakharov) to 34% (Ackley D=50)
- **PSO**: Topology contribution grows from 7% (D=10) to 23% (D=50)

## Method Comparison

| Method | Cosine Similarity | Rank Agreement | Cost |
|--------|------------------|----------------|------|
| KernelSHAP | 0.88 | 87% | ~30x single run |
| QuickSHAP | 0.76 | 41% | O(1) |
| Tracking | 0.76 | 43% | O(1) |

## Benchmark Functions

Sphere, Rosenbrock, Zakharov, Dixon-Price (unimodal); Rastrigin, Schwefel, Ackley, Griewank, Levy (multimodal). Dimensions: D = {10, 30, 50}.

## Requirements

- Python >= 3.10
- NumPy >= 1.20
- matplotlib >= 3.5 (for visualization)
- pytest >= 7.0 (for testing)

## Citation

If you use XMH in your research, please cite:

```bibtex
@article{xmh2026,
  title = {XMH: Explainable Metaheuristics via Exact Operator-Level
           Shapley Value Attribution},
  author = {Anonymous},
  year = {2026},
  note = {Under review}
}
```

## License

MIT License. See [LICENSE](LICENSE).
