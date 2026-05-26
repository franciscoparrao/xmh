"""Friedman + post-hoc Holm tests over the exact-Shapley result set.

Implements the standard non-parametric comparison protocol for metaheuristics
recommended by Derrac et al. (2011): rank algorithms per problem instance,
run Friedman's test for overall significance, then apply Holm's step-down
correction for pairwise comparisons.

Operates on two complementary axes:

  A) Algorithm performance across problems. For each (function, dimension),
     extract the mean final fitness of each algorithm under the full
     coalition (all operators active). Friedman + Holm identifies which
     algorithms differ in convergence quality.

  B) Operator-attribution differences across problems. For each algorithm
     separately, rank its operators per problem by their |Shapley value|
     percentage, then Friedman + Holm identifies which operator ranks
     differ from the others (i.e., is there one consistently-dominant
     operator within the algorithm?).

Reads:
  results_exact_shapley/exact_shapley_results.json   (DE, GA, PSO)
  results_cmaes_shapley/cmaes_shapley_results.json   (CMA-ES)
  results_shade_shapley/shade_shapley_results.json   (SHADE, if available)

Writes:
  results_statistical/algorithm_friedman.json
  results_statistical/operator_friedman.json
"""
from __future__ import annotations

import itertools
import json
import os
import statistics
from typing import Any

import numpy as np
from scipy import stats


ROOT = '/home/franciscoparrao/proyectos/metaheuristicas'
OUTDIR = os.path.join(ROOT, 'results_statistical')
os.makedirs(OUTDIR, exist_ok=True)


def load_results() -> dict:
    """Combine exact-Shapley results from all algorithms into one dict."""
    combined = {}
    paths = [
        os.path.join(ROOT, 'results_exact_shapley/exact_shapley_results.json'),
        os.path.join(ROOT, 'results_cmaes_shapley/cmaes_shapley_results.json'),
        os.path.join(ROOT, 'results_shade_shapley/shade_shapley_results.json'),
        os.path.join(ROOT, 'results_jade_shapley/jade_shapley_results.json'),
    ]
    for path in paths:
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            for alg, by_func in data.items():
                combined[alg] = by_func
    return combined


def parse_key(k: str) -> tuple:
    inner = k.strip('()').strip()
    if not inner:
        return ()
    return tuple(int(x.strip()) for x in inner.split(',') if x.strip())


def full_coalition_fitness(case: dict) -> list[float]:
    """Return the n_runs final-fitness values from the full coalition."""
    n = len(case['operators'])
    full = tuple(range(n))
    coal = {parse_key(k): v for k, v in case['coalitions'].items()}
    return list(coal[full])


def holm_post_hoc(stats_table: list[tuple[str, str, float]]) -> list[dict]:
    """Apply Holm's step-down correction.

    Input: list of (group_a, group_b, raw_p_value) tuples.
    Output: list of dicts with adjusted p-values and reject decisions at α = 0.05.
    """
    m = len(stats_table)
    # Sort by raw p-value ascending
    indexed = sorted(enumerate(stats_table), key=lambda x: x[1][2])
    results = [None] * m
    prev_p_adj = 0.0
    for rank, (orig_idx, (a, b, p)) in enumerate(indexed):
        # Holm: adjusted p = (m - rank) * p, but monotonic
        p_adj = (m - rank) * p
        p_adj = max(p_adj, prev_p_adj)
        p_adj = min(p_adj, 1.0)
        prev_p_adj = p_adj
        results[orig_idx] = {
            'a': a, 'b': b, 'p_raw': float(p),
            'p_holm': float(p_adj),
            'reject_at_0.05': bool(p_adj < 0.05),
        }
    return results


def algorithm_comparison(combined: dict) -> dict:
    """Axis A: rank algorithms per problem by mean final fitness, Friedman + Holm."""
    algorithms = sorted(combined.keys())
    # Build a 2D matrix: rows = problems (function × dim), cols = algorithms
    problems: list[tuple[str, str]] = []
    matrix: list[list[float]] = []
    # Use the intersection of (function, dim) configurations across all algorithms
    common = None
    for alg in algorithms:
        configs = set()
        for func, by_dim in combined[alg].items():
            for dim in by_dim.keys():
                configs.add((func, dim))
        common = configs if common is None else common & configs

    for func, dim in sorted(common):
        row = []
        for alg in algorithms:
            ff = full_coalition_fitness(combined[alg][func][dim])
            row.append(statistics.mean(ff))
        problems.append((func, dim))
        matrix.append(row)

    M = np.array(matrix)
    # Friedman test (scipy expects each algorithm's samples as a separate array)
    f_stat, p_val = stats.friedmanchisquare(*[M[:, i] for i in range(len(algorithms))])

    # Per-problem ranks (1 = lowest fitness)
    ranks = np.array([stats.rankdata(row) for row in M])
    mean_ranks = ranks.mean(axis=0)

    # Pairwise Wilcoxon signed-rank tests, then Holm correction
    pair_table = []
    for i, j in itertools.combinations(range(len(algorithms)), 2):
        try:
            _, p = stats.wilcoxon(M[:, i], M[:, j])
        except ValueError:
            p = 1.0
        pair_table.append((algorithms[i], algorithms[j], float(p)))
    holm = holm_post_hoc(pair_table)

    return {
        'algorithms': algorithms,
        'problems': [{'function': f, 'dimension': d} for f, d in problems],
        'mean_ranks': {alg: float(r) for alg, r in zip(algorithms, mean_ranks)},
        'friedman': {
            'statistic': float(f_stat),
            'p_value': float(p_val),
            'reject_null_at_0.05': bool(p_val < 0.05),
            'n_problems': len(problems),
            'k_algorithms': len(algorithms),
        },
        'holm_pairwise': holm,
        'fitness_matrix': M.tolist(),
    }


def operator_dominance(combined: dict) -> dict:
    """Axis B: per-algorithm Friedman over operators across problems.

    For each algorithm, build a matrix where rows are problems and columns
    are operators (in canonical order). Cell value = |φ_i| percentage of
    total |Shapley|. Friedman test asks whether the operators differ in
    contribution consistently across problems.
    """
    out: dict[str, Any] = {}
    for alg in sorted(combined.keys()):
        # Find a representative operator list
        sample_func = next(iter(combined[alg]))
        sample_dim = next(iter(combined[alg][sample_func]))
        operators = combined[alg][sample_func][sample_dim]['operators']

        problems: list[tuple[str, str]] = []
        matrix: list[list[float]] = []
        for func, by_dim in sorted(combined[alg].items()):
            for dim, case in sorted(by_dim.items()):
                shap = case['shapley_values']
                total = sum(abs(v) for v in shap.values())
                if total < 1e-15:
                    continue
                row = [100 * abs(shap[op]) / total for op in operators]
                problems.append((func, dim))
                matrix.append(row)

        if len(problems) < 2 or len(operators) < 2:
            out[alg] = {'note': 'insufficient data'}
            continue

        M = np.array(matrix)
        f_stat, p_val = stats.friedmanchisquare(*[M[:, i] for i in range(len(operators))])
        ranks = np.array([stats.rankdata(-row) for row in M])  # higher pct = rank 1
        mean_ranks = ranks.mean(axis=0)

        pair_table = []
        for i, j in itertools.combinations(range(len(operators)), 2):
            try:
                _, p = stats.wilcoxon(M[:, i], M[:, j])
            except ValueError:
                p = 1.0
            pair_table.append((operators[i], operators[j], float(p)))
        holm = holm_post_hoc(pair_table)

        out[alg] = {
            'operators': operators,
            'problems': [{'function': f, 'dimension': d} for f, d in problems],
            'mean_ranks': {op: float(r) for op, r in zip(operators, mean_ranks)},
            'friedman': {
                'statistic': float(f_stat),
                'p_value': float(p_val),
                'reject_null_at_0.05': bool(p_val < 0.05),
                'n_problems': len(problems),
                'k_operators': len(operators),
            },
            'holm_pairwise': holm,
            'attribution_matrix_pct': M.tolist(),
        }
    return out


def main():
    combined = load_results()
    print(f"Loaded algorithms: {sorted(combined.keys())}")

    # Axis A: algorithm performance comparison
    alg_results = algorithm_comparison(combined)
    print("\n=== Axis A: Algorithm performance comparison ===")
    fr = alg_results['friedman']
    print(f"Friedman χ² = {fr['statistic']:.2f}, p = {fr['p_value']:.3e}, "
          f"reject null = {fr['reject_null_at_0.05']}")
    print(f"Mean ranks (lower = better fitness):")
    for alg, r in sorted(alg_results['mean_ranks'].items(), key=lambda x: x[1]):
        print(f"  {alg:8s} {r:5.2f}")
    print(f"\nHolm-adjusted pairwise (Wilcoxon signed-rank, n={fr['n_problems']} problems):")
    for h in alg_results['holm_pairwise']:
        marker = "**" if h['reject_at_0.05'] else "  "
        print(f"  {marker} {h['a']:>8s} vs {h['b']:<8s}  "
              f"p_raw = {h['p_raw']:.3e}  p_Holm = {h['p_holm']:.3e}")

    out_a = os.path.join(OUTDIR, 'algorithm_friedman.json')
    with open(out_a, 'w') as f:
        json.dump(alg_results, f, indent=2)
    print(f"\nSaved {out_a}")

    # Axis B: per-algorithm operator dominance
    op_results = operator_dominance(combined)
    print("\n=== Axis B: Per-algorithm operator dominance ===")
    for alg, res in op_results.items():
        if 'friedman' not in res:
            continue
        fr = res['friedman']
        print(f"\n{alg}: Friedman χ² = {fr['statistic']:.2f}, p = {fr['p_value']:.3e}")
        print(f"  Mean ranks (1 = highest %|φ|):")
        for op, r in sorted(res['mean_ranks'].items(), key=lambda x: x[1]):
            print(f"    {op:35s} {r:5.2f}")
        any_sig = any(h['reject_at_0.05'] for h in res['holm_pairwise'])
        if any_sig:
            print("  Significant Holm-adjusted pairwise differences:")
            for h in res['holm_pairwise']:
                if h['reject_at_0.05']:
                    print(f"    {h['a']:25s} vs {h['b']:25s}  p_Holm = {h['p_holm']:.3e}")

    out_b = os.path.join(OUTDIR, 'operator_friedman.json')
    with open(out_b, 'w') as f:
        json.dump(op_results, f, indent=2)
    print(f"\nSaved {out_b}")


if __name__ == '__main__':
    main()
