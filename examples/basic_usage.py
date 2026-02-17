#!/usr/bin/env python3
"""
XMH Basic Usage Example

Demonstrates how to use the XMH framework to:
1. Run an instrumented metaheuristic
2. Generate operator attribution explanations
3. Create visualizations
4. Perform counterfactual analysis
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from algorithms.instrumented_de import InstrumentedDE, AdaptiveDE
from benchmarks.functions import get_benchmark, list_benchmarks
from explanation.operator_shap import OperatorSHAP, QuickSHAP
from explanation.attribution import AttributionAnalyzer
from core.trace_logger import InstrumentationLevel


def main():
    """Main example function."""
    print("=" * 70)
    print("XMH Framework - Basic Usage Example")
    print("=" * 70)
    print()

    # =========================================
    # 1. Setup: Define problem and algorithm
    # =========================================
    print("1. SETUP")
    print("-" * 40)

    # Get benchmark function
    benchmark = get_benchmark("rastrigin")
    dim = 10

    print(f"Benchmark: {benchmark.name}")
    print(f"Dimension: {dim}")
    print(f"Bounds: {benchmark.bounds}")
    print(f"Characteristics: {benchmark.characteristics}")
    print()

    # Get bounds
    lower, upper = benchmark.get_bounds(dim)

    # Create instrumented DE
    de = InstrumentedDE(
        dim=dim,
        bounds=(lower, upper),
        pop_size=50,
        max_generations=100,
        F=0.5,
        CR=0.9,
        mutation_strategy="rand/1",
        crossover_type="binomial",
        instrumentation_level=InstrumentationLevel.STANDARD,
        seed=42
    )

    print(f"Algorithm: {de.__class__.__name__}")
    print(f"Configuration: {de.config}")
    print()

    # =========================================
    # 2. Run optimization
    # =========================================
    print("2. RUNNING OPTIMIZATION")
    print("-" * 40)

    result = de.run(
        objective_func=benchmark,
        problem_name=benchmark.name,
        verbose=True
    )

    print()
    print(f"Best fitness found: {result['best_fitness']:.6e}")
    print(f"Total evaluations: {result['evaluations']}")
    print()

    # =========================================
    # 3. Quick attribution analysis
    # =========================================
    print("3. QUICK ATTRIBUTION ANALYSIS")
    print("-" * 40)

    trace = result['trace']
    analyzer = AttributionAnalyzer()
    attribution = analyzer.analyze(trace)

    print(attribution.summary())
    print()

    # =========================================
    # 4. Quick SHAP analysis (from trace)
    # =========================================
    print("4. QUICK SHAP ANALYSIS (trace-based)")
    print("-" * 40)

    quick_shap = QuickSHAP()
    explanation = quick_shap.explain_from_trace(trace)

    print(explanation.summary())
    print()

    # =========================================
    # 5. Trace summary
    # =========================================
    print("5. TRACE SUMMARY")
    print("-" * 40)

    print(trace.summary())
    print()

    # =========================================
    # 6. Critical generations
    # =========================================
    print("6. CRITICAL GENERATIONS")
    print("-" * 40)

    critical = trace.get_critical_generations(5)
    print(f"Generations with major improvements: {critical}")
    print()

    # =========================================
    # 7. Operator contributions
    # =========================================
    print("7. OPERATOR CONTRIBUTIONS")
    print("-" * 40)

    contributions = trace.get_operator_contributions()
    for op, stats in contributions.items():
        print(f"{op}:")
        print(f"  Total uses: {stats['total_used']}")
        print(f"  Total successes: {stats['total_success']}")
        print(f"  Success rate: {stats['success_rate']*100:.1f}%")
    print()

    # =========================================
    # 8. Compare with different strategy
    # =========================================
    print("8. COMPARISON: Different mutation strategy")
    print("-" * 40)

    de_best = InstrumentedDE(
        dim=dim,
        bounds=(lower, upper),
        pop_size=50,
        max_generations=100,
        F=0.5,
        CR=0.9,
        mutation_strategy="best/1",  # Different strategy
        crossover_type="binomial",
        instrumentation_level=InstrumentationLevel.STANDARD,
        seed=42
    )

    result_best = de_best.run(
        objective_func=benchmark,
        problem_name=benchmark.name,
        verbose=False
    )

    print(f"DE/rand/1 best fitness: {result['best_fitness']:.6e}")
    print(f"DE/best/1 best fitness: {result_best['best_fitness']:.6e}")
    print()

    # Compare attributions
    print("Attribution comparison:")
    print(analyzer.compare_runs(
        [trace, result_best['trace']],
        labels=["rand/1", "best/1"]
    ))
    print()

    # =========================================
    # 9. Try visualization (if matplotlib available)
    # =========================================
    print("9. VISUALIZATION")
    print("-" * 40)

    try:
        from visualization.plots import create_summary_figure
        import matplotlib.pyplot as plt

        fig = create_summary_figure(
            trace,
            explanation,
            save_path="xmh_example_output.png"
        )
        plt.close(fig)
        print("Summary figure saved to 'xmh_example_output.png'")
    except ImportError:
        print("matplotlib not available - skipping visualization")
    print()

    # =========================================
    # 10. Save trace
    # =========================================
    print("10. SAVING TRACE")
    print("-" * 40)

    trace.save("xmh_example_trace.json")
    print("Trace saved to 'xmh_example_trace.json'")
    print()

    print("=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


def run_full_shap_example():
    """
    Example of full Operator-SHAP computation.

    Note: This is computationally expensive as it requires
    multiple algorithm runs. Use QuickSHAP for fast approximations.
    """
    print("=" * 70)
    print("Full Operator-SHAP Example (Computationally Intensive)")
    print("=" * 70)
    print()

    benchmark = get_benchmark("sphere")  # Simple function for demo
    dim = 5  # Low dimension for speed
    lower, upper = benchmark.get_bounds(dim)

    # Factory function to create fresh algorithm instances
    def algorithm_factory():
        return InstrumentedDE(
            dim=dim,
            bounds=(lower, upper),
            pop_size=20,
            max_generations=50,
            F=0.5,
            CR=0.9,
            mutation_strategy="rand/1",
            crossover_type="binomial",
            instrumentation_level=InstrumentationLevel.LIGHT,
            seed=None  # Different seed each time
        )

    # Compute full SHAP values
    print("Computing Operator-SHAP values (this may take a while)...")
    shap = OperatorSHAP(
        n_samples=50,  # Reduced for demo
        n_runs_per_coalition=3,
        compute_interactions=True
    )

    explanation = shap.explain(
        algorithm_factory=algorithm_factory,
        objective_func=benchmark
    )

    print(explanation.summary())


if __name__ == "__main__":
    main()

    # Uncomment to run full SHAP example (slow):
    # run_full_shap_example()
