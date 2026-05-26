#!/usr/bin/env python3
"""Analyze GA operator contributions for debugging."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from xmh.algorithms.instrumented_ga import InstrumentedGA
from xmh.benchmarks.functions import get_benchmark
from xmh.explanation.operator_shap import QuickSHAP
from xmh.core.trace_logger import InstrumentationLevel
from xmh.core.instrumentation import get_neutral_operator
import numpy as np

def analyze_ga(func_name='sphere'):
    # Run GA on specified function
    benchmark = get_benchmark(func_name)
    dim = 10
    print(f'\n{"="*60}')
    print(f'ANALYZING GA ON {func_name.upper()}')
    print(f'{"="*60}')
    lower, upper = benchmark.get_bounds(dim)

    kwargs = {
        'dim': dim,
        'bounds': (lower, upper),
        'pop_size': 30,
        'max_generations': 50,
        'instrumentation_level': InstrumentationLevel.STANDARD,
        'seed': 42
    }

    # Full run
    print('=== FULL GA RUN ===')
    alg = InstrumentedGA(**kwargs)
    result = alg.run(benchmark, verbose=False)
    trace = result['trace']
    full_fitness = result['best_fitness']

    print(f'Final fitness: {full_fitness:.6e}')
    print()

    # Get operator contributions from trace
    contributions = trace.get_operator_contributions()
    print('Operator Statistics from Trace:')
    for op, stats in contributions.items():
        print(f'  {op}: used={stats["total_used"]}, success={stats["total_success"]}, rate={stats["success_rate"]*100:.1f}%')

    print()

    # QuickSHAP values
    quick_shap = QuickSHAP()
    explanation = quick_shap.explain_from_trace(trace)
    print('QuickSHAP Values:')
    for op, val in sorted(explanation.shap_values.items(), key=lambda x: -abs(x[1])):
        print(f'  {op}: {val:.6e}')

    print()
    print(f'Baseline (initial best): {explanation.base_value:.6e}')
    print(f'Final fitness: {full_fitness:.6e}')
    print(f'Total improvement: {explanation.total_contribution:.6e}')

    # Now run ablation study
    print()
    print('=== ABLATION STUDY ===')

    # Get operators
    alg = InstrumentedGA(**kwargs)
    alg._initialize_operators()
    operators = alg.get_operators()

    ablation_results = {}

    for op_name in operators:
        # Create new instance and disable operator
        alg = InstrumentedGA(**kwargs)

        # Patch to disable after init
        original_init = alg._initialize_operators
        def patched_init(self=alg, orig=original_init, op=op_name):
            orig()
            self.disable_operator(op)
        alg._initialize_operators = patched_init

        result = alg.run(benchmark, verbose=False)
        ablation_results[op_name] = result['best_fitness']

        contribution = ablation_results[op_name] - full_fitness
        print(f'  {op_name}: fitness={ablation_results[op_name]:.6e}, contribution={contribution:.6e}')

    print()
    print('=== COMPARISON ===')
    print(f'{"Operator":<30} {"Ablation":>15} {"SHAP":>15} {"Same Sign?":>12}')
    print('-' * 75)

    for op in operators:
        abl = ablation_results[op] - full_fitness
        shp = explanation.shap_values.get(op, 0)
        same_sign = "YES" if (abl > 0 and shp > 0) or (abl < 0 and shp < 0) or (abl == 0 and shp == 0) else "NO"
        print(f'{op:<30} {abl:>15.6e} {shp:>15.6e} {same_sign:>12}')


if __name__ == "__main__":
    analyze_ga('sphere')
    analyze_ga('rastrigin')
