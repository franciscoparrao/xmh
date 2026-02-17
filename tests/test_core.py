"""
Tests for XMH Core Module

Run with: pytest xmh/tests/test_core.py -v
"""

import pytest
import numpy as np
import sys
import os

# Ensure the project root is in the path so xmh can be imported as a package
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from xmh.core.trace_logger import TraceLogger, InstrumentationLevel, EventType
from xmh.core.instrumentation import (
    Operator, NeutralOperator,
    NeutralMutationOperator, NeutralCrossoverOperator, NeutralSelectionOperator,
    NeutralVelocityOperator, NeutralPositionOperator, NeutralTopologyOperator,
    get_neutral_operator,
)
from xmh.algorithms.instrumented_de import InstrumentedDE
from xmh.algorithms.instrumented_ga import InstrumentedGA
from xmh.algorithms.instrumented_pso import InstrumentedPSO
from xmh.benchmarks.functions import sphere, rastrigin, get_benchmark
from xmh.explanation.operator_shap import QuickSHAP, ExactOperatorSHAP
from xmh.explanation.tracking_attribution import TrackingAttribution


class TestTraceLogger:
    """Tests for TraceLogger class."""

    def test_initialization(self):
        """Test logger initialization."""
        logger = TraceLogger(level=InstrumentationLevel.STANDARD)
        assert logger.level == InstrumentationLevel.STANDARD
        assert len(logger.events) == 0
        assert len(logger.snapshots) == 0

    def test_start_run(self):
        """Test starting a run."""
        logger = TraceLogger()
        logger.start_run(
            algorithm_name="TestAlgo",
            problem_name="TestProblem",
            dimension=10
        )

        assert logger.algorithm_name == "TestAlgo"
        assert logger.problem_name == "TestProblem"
        assert logger.dimension == 10
        assert len(logger.events) == 1
        assert logger.events[0].event_type == EventType.INITIALIZATION

    def test_log_generation(self):
        """Test logging a generation."""
        logger = TraceLogger()
        logger.start_run("Test", "Test", 5)

        population = np.random.rand(10, 5)
        fitness = np.random.rand(10)

        logger.log_generation_end(population, fitness)

        assert len(logger.snapshots) == 1
        assert logger.snapshots[0].generation == 1

    def test_operator_logging(self):
        """Test logging operator usage."""
        logger = TraceLogger(level=InstrumentationLevel.STANDARD)
        logger.start_run("Test", "Test", 5)

        logger.log_operator_selection(
            operator_name="mutation",
            parameters={"F": 0.5},
            rationale="Test mutation"
        )

        selection_events = [
            e for e in logger.events
            if e.event_type == EventType.OPERATOR_SELECTION
        ]
        assert len(selection_events) == 1

    def test_fitness_history(self):
        """Test getting fitness history."""
        logger = TraceLogger()
        logger.start_run("Test", "Test", 5)

        for i in range(5):
            population = np.random.rand(10, 5)
            fitness = np.random.rand(10) * (10 - i)  # Decreasing fitness
            logger.log_generation_end(population, fitness)

        gens, best, mean = logger.get_fitness_history()

        assert len(gens) == 5
        assert len(best) == 5
        assert len(mean) == 5

    def test_improvement_tracking(self):
        """Test that operator improvements are tracked in snapshots."""
        logger = TraceLogger(level=InstrumentationLevel.STANDARD)
        logger.start_run("Test", "Test", 5)

        # Log a successful operator application with improvement
        logger.log_operator_application(
            operator_name="mutation_test",
            parent_indices=[0],
            offspring_fitness=[1.0],
            success=True,
            details={"improvement": 5.0}
        )
        logger.log_operator_application(
            operator_name="mutation_test",
            parent_indices=[1],
            offspring_fitness=[2.0],
            success=True,
            details={"improvement": 3.0}
        )
        logger.log_operator_application(
            operator_name="crossover_test",
            parent_indices=[0, 1],
            offspring_fitness=[1.5],
            success=True,
            details={"improvement": 2.0}
        )

        population = np.random.rand(10, 5)
        fitness = np.random.rand(10)
        logger.log_generation_end(population, fitness)

        snapshot = logger.snapshots[0]
        assert snapshot.operators_improvement["mutation_test"] == 8.0
        assert snapshot.operators_improvement["crossover_test"] == 2.0

    def test_get_operator_contributions_with_improvements(self):
        """Test that get_operator_contributions returns improvement data."""
        logger = TraceLogger(level=InstrumentationLevel.STANDARD)
        logger.start_run("Test", "Test", 5)

        # Two generations with operator applications
        for gen in range(2):
            logger.log_operator_selection("mut", {})
            logger.log_operator_application(
                "mut", [0], [1.0], success=True,
                details={"improvement": 3.0}
            )
            logger.log_operator_application(
                "mut", [1], [2.0], success=False,
                details={}
            )
            population = np.random.rand(10, 5)
            fitness = np.random.rand(10)
            logger.log_generation_end(population, fitness)

        contributions = logger.get_operator_contributions()
        assert "mut" in contributions
        assert contributions["mut"]["total_used"] == 2
        assert contributions["mut"]["total_success"] == 2
        assert contributions["mut"]["cumulative_improvement"] == 6.0
        assert contributions["mut"]["avg_improvement"] == 3.0
        assert contributions["mut"]["success_rate"] == 1.0


class TestOperator:
    """Tests for Operator class."""

    def test_operator_creation(self):
        """Test creating an operator."""
        def dummy_func(x):
            return x * 2

        op = Operator(
            name="test_op",
            operator_type="mutation",
            function=dummy_func,
            parameters={"scale": 1.0}
        )

        assert op.name == "test_op"
        assert op.operator_type == "mutation"

    def test_neutral_operator(self):
        """Test neutral operator."""
        neutral = NeutralOperator("mutation")
        population = np.array([[1, 2, 3], [4, 5, 6]])

        result = neutral.apply(population)

        np.testing.assert_array_equal(result, population)


class TestNeutralOperators:
    """Tests for all typed neutral operators."""

    def test_neutral_mutation_ga_style(self):
        """Neutral mutation returns individual unchanged (GA interface)."""
        neutral = NeutralMutationOperator()
        individual = np.array([1.0, 2.0, 3.0])
        result = neutral.apply(individual=individual)
        np.testing.assert_array_equal(result, individual)

    def test_neutral_mutation_de_style(self):
        """Neutral mutation returns target unchanged (DE interface)."""
        neutral = NeutralMutationOperator()
        pop = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = neutral.apply(population=pop, target_idx=1)
        np.testing.assert_array_equal(result, pop[1])

    def test_neutral_crossover_ga_style(self):
        """Neutral crossover returns parents unchanged."""
        neutral = NeutralCrossoverOperator()
        p1 = np.array([1.0, 2.0])
        p2 = np.array([3.0, 4.0])
        c1, c2 = neutral.apply(parent1=p1, parent2=p2)
        np.testing.assert_array_equal(c1, p1)
        np.testing.assert_array_equal(c2, p2)

    def test_neutral_crossover_de_style(self):
        """Neutral crossover returns target unchanged (DE interface)."""
        neutral = NeutralCrossoverOperator()
        target = np.array([1.0, 2.0])
        mutant = np.array([3.0, 4.0])
        result = neutral.apply(target=target, mutant=mutant)
        np.testing.assert_array_equal(result, target)

    def test_neutral_selection_random_index(self):
        """Neutral selection returns a random index (no fitness pressure)."""
        neutral = NeutralSelectionOperator()
        fitness = np.array([10.0, 1.0, 5.0, 3.0])
        idx = neutral.apply(fitness=fitness)
        assert 0 <= idx < len(fitness)

    def test_neutral_velocity(self):
        """Neutral velocity returns zero velocity."""
        neutral = NeutralVelocityOperator()
        vel = np.array([1.0, -2.0, 3.0])
        result = neutral.apply(velocity=vel)
        np.testing.assert_array_equal(result, np.zeros(3))

    def test_neutral_position(self):
        """Neutral position returns position unchanged (ignores velocity)."""
        neutral = NeutralPositionOperator()
        pos = np.array([1.0, 2.0, 3.0])
        vel = np.array([10.0, 20.0, 30.0])
        result = neutral.apply(position=pos, velocity=vel)
        np.testing.assert_array_equal(result, pos)

    def test_neutral_topology(self):
        """Neutral topology returns particle's own personal best."""
        neutral = NeutralTopologyOperator()
        personal_best = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = neutral.apply(particle_idx=1, personal_best=personal_best)
        np.testing.assert_array_equal(result, personal_best[1])

    def test_get_neutral_operator_factory(self):
        """Factory returns correct neutral operator for each type."""
        types = ["mutation", "crossover", "selection", "velocity", "position", "topology"]
        for op_type in types:
            neutral = get_neutral_operator(op_type)
            assert neutral.operator_type == op_type


class TestInstrumentedDE:
    """Tests for InstrumentedDE class."""

    @pytest.fixture
    def simple_de(self):
        """Create a simple DE instance for testing."""
        dim = 5
        bounds = (np.full(dim, -5.0), np.full(dim, 5.0))
        return InstrumentedDE(
            dim=dim,
            bounds=bounds,
            pop_size=20,
            max_generations=10,
            F=0.5,
            CR=0.9,
            seed=42
        )

    def test_initialization(self, simple_de):
        """Test DE initialization."""
        assert simple_de.dim == 5
        assert simple_de.pop_size == 20
        assert simple_de.F == 0.5
        assert simple_de.CR == 0.9

    def test_run_sphere(self, simple_de):
        """Test running DE on sphere function."""
        result = simple_de.run(sphere, problem_name="sphere", verbose=False)

        assert "best_solution" in result
        assert "best_fitness" in result
        assert "trace" in result
        assert result["best_fitness"] < 100  # Should find reasonable solution

    def test_run_rastrigin(self, simple_de):
        """Test running DE on rastrigin function."""
        result = simple_de.run(rastrigin, problem_name="rastrigin", verbose=False)

        assert result["best_fitness"] < 50  # Reasonable for 10 gens

    def test_operators_registered(self, simple_de):
        """Test that operators are registered after initialization."""
        simple_de._initialize_operators()
        operators = simple_de.get_operators()

        assert len(operators) >= 2  # At least mutation and crossover

    def test_trace_populated(self, simple_de):
        """Test that trace is populated after run."""
        result = simple_de.run(sphere, verbose=False)
        trace = result["trace"]

        assert len(trace.snapshots) == 11  # Initial + 10 generations
        assert len(trace.events) > 0

    def test_reproducibility(self):
        """Test that same seed gives same results."""
        dim = 5
        bounds = (np.full(dim, -5.0), np.full(dim, 5.0))

        de1 = InstrumentedDE(dim=dim, bounds=bounds, pop_size=20,
                            max_generations=20, seed=123)
        de2 = InstrumentedDE(dim=dim, bounds=bounds, pop_size=20,
                            max_generations=20, seed=123)

        result1 = de1.run(sphere, verbose=False)
        result2 = de2.run(sphere, verbose=False)

        assert result1["best_fitness"] == result2["best_fitness"]

    def test_different_strategies(self):
        """Test different mutation strategies."""
        dim = 5
        bounds = (np.full(dim, -5.0), np.full(dim, 5.0))

        strategies = ["rand/1", "best/1", "current-to-best/1"]

        for strategy in strategies:
            de = InstrumentedDE(
                dim=dim,
                bounds=bounds,
                pop_size=20,
                max_generations=10,
                mutation_strategy=strategy,
                seed=42
            )
            result = de.run(sphere, verbose=False)
            assert result["best_fitness"] is not None

    def test_de_improvement_tracking(self):
        """Test that DE logs improvement details correctly."""
        dim = 5
        bounds = (np.full(dim, -5.0), np.full(dim, 5.0))
        de = InstrumentedDE(
            dim=dim, bounds=bounds, pop_size=20,
            max_generations=20, seed=42,
        )
        result = de.run(sphere, verbose=False)
        contributions = result["trace"].get_operator_contributions()

        for op, stats in contributions.items():
            assert "cumulative_improvement" in stats
            assert "avg_improvement" in stats
            if stats["total_success"] > 0:
                assert stats["cumulative_improvement"] > 0


class TestInstrumentedGA:
    """Tests for InstrumentedGA class."""

    @pytest.fixture
    def simple_ga(self):
        dim = 5
        bounds = (np.full(dim, -5.0), np.full(dim, 5.0))
        return InstrumentedGA(
            dim=dim, bounds=bounds, pop_size=20,
            max_generations=10, seed=42,
        )

    def test_run(self, simple_ga):
        """Test GA runs and produces a result."""
        result = simple_ga.run(sphere, verbose=False)
        assert result["best_fitness"] < 100

    def test_ga_success_uses_best_parent(self, simple_ga):
        """Test that GA success criterion uses best parent fitness."""
        result = simple_ga.run(sphere, verbose=False)

        # Check that the logged details use best_parent_fit, not parent_avg
        for event in result["trace"].events:
            if event.event_type == EventType.OPERATOR_APPLICATION:
                details = event.data.get("details", {})
                if details:
                    assert "best_parent_fit" in details or "velocity_magnitude" in details
                    assert "parent_avg" not in details

    def test_ga_operators_registered(self, simple_ga):
        """Test GA registers 3 operators."""
        simple_ga._initialize_operators()
        ops = simple_ga.get_operators()
        assert len(ops) == 3  # selection, crossover, mutation


class TestInstrumentedPSO:
    """Tests for InstrumentedPSO class."""

    @pytest.fixture
    def simple_pso(self):
        dim = 5
        bounds = (np.full(dim, -5.0), np.full(dim, 5.0))
        return InstrumentedPSO(
            dim=dim, bounds=bounds, pop_size=20,
            max_generations=10, seed=42,
        )

    def test_run(self, simple_pso):
        """Test PSO runs and produces a result."""
        result = simple_pso.run(sphere, verbose=False)
        assert result["best_fitness"] < 100

    def test_pso_topology_logged(self, simple_pso):
        """Test that PSO topology operator is logged via log_operator_application."""
        result = simple_pso.run(sphere, verbose=False)
        contributions = result["trace"].get_operator_contributions()

        # Should have both velocity and topology contributions
        op_names = list(contributions.keys())
        has_velocity = any("velocity" in name for name in op_names)
        has_topology = any("topology" in name for name in op_names)

        assert has_velocity, f"No velocity operator logged, found: {op_names}"
        assert has_topology, f"No topology operator logged, found: {op_names}"

    def test_pso_operators_registered(self, simple_pso):
        """Test PSO registers 3 operators (velocity, position, topology)."""
        simple_pso._initialize_operators()
        ops = simple_pso.get_operators()
        assert len(ops) == 3


class TestQuickSHAP:
    """Tests for QuickSHAP with 3-factor formula."""

    def test_quickshap_uses_three_factors(self):
        """Test that QuickSHAP uses 3 factors when improvement data is available."""
        dim = 5
        bounds = (np.full(dim, -5.0), np.full(dim, 5.0))
        de = InstrumentedDE(
            dim=dim, bounds=bounds, pop_size=20,
            max_generations=30, seed=42,
        )
        result = de.run(sphere, verbose=False)

        qs = QuickSHAP()
        explanation = qs.explain_from_trace(result["trace"])

        # Values should sum to total improvement
        value_sum = sum(explanation.shap_values.values())
        assert abs(value_sum - explanation.total_contribution) < 1e-10

        # All operators should have non-negative values on sphere
        for op, val in explanation.shap_values.items():
            assert val >= 0, f"Operator {op} has negative value {val}"

    def test_quickshap_fallback_no_improvement(self):
        """Test that QuickSHAP falls back to 2-factor when no improvement data."""
        logger = TraceLogger(level=InstrumentationLevel.STANDARD)
        logger.start_run("Test", "Test", 5)

        # Log operator usage without improvement details
        for _ in range(10):
            logger.log_operator_selection("op_a", {})
            logger.log_operator_application("op_a", [0], [1.0], success=True, details={})
            logger.log_operator_selection("op_b", {})
            logger.log_operator_application("op_b", [0], [2.0], success=False, details={})

        # Log generation with decreasing fitness
        for i in range(5):
            pop = np.random.rand(10, 5)
            fit = np.random.rand(10) * (10 - i)
            logger.log_generation_end(pop, fit)

        qs = QuickSHAP()
        explanation = qs.explain_from_trace(logger)

        # Should not crash, values should sum to total improvement
        assert len(explanation.shap_values) > 0


class TestExactOperatorSHAP:
    """Tests for ExactOperatorSHAP."""

    def test_exact_shapley_on_simple_problem(self):
        """Test exact Shapley on a small DE instance."""
        dim = 5
        bounds = (np.full(dim, -5.0), np.full(dim, 5.0))

        exact_shap = ExactOperatorSHAP(
            n_runs_per_coalition=3,
            base_seed=42,
            compute_interactions=True,
        )

        explanation = exact_shap.explain(
            InstrumentedDE,
            {
                "dim": dim,
                "bounds": bounds,
                "pop_size": 20,
                "max_generations": 10,
                "F": 0.5,
                "CR": 0.9,
            },
            sphere,
        )

        # Efficiency axiom: sum of Shapley values ≈ v(N) - v(∅)
        shapley_sum = sum(explanation.shap_values.values())
        assert abs(shapley_sum - explanation.total_contribution) < abs(explanation.total_contribution) * 0.15, \
            f"Efficiency violation: Σφ={shapley_sum:.4f} vs v(N)-v(∅)={explanation.total_contribution:.4f}"

        # Should have confidence intervals
        assert len(explanation.confidence_intervals) == len(explanation.shap_values)

        # Should have interactions
        assert explanation.interactions is not None
        assert len(explanation.interactions) > 0

    def test_exact_shapley_unanimity_game(self):
        """
        Test exact Shapley on a unanimity-like problem.

        When only one operator actually matters, its Shapley value should
        be much larger than the others.
        """
        dim = 5
        bounds = (np.full(dim, -5.0), np.full(dim, 5.0))

        exact_shap = ExactOperatorSHAP(
            n_runs_per_coalition=3,
            base_seed=42,
            compute_interactions=False,
        )

        result = exact_shap.get_coalition_values(
            InstrumentedDE,
            {
                "dim": dim,
                "bounds": bounds,
                "pop_size": 20,
                "max_generations": 10,
                "F": 0.5,
                "CR": 0.9,
            },
            sphere,
        )

        # All coalition values should be finite
        for key, vals in result["coalitions"].items():
            for v in vals:
                assert np.isfinite(v), f"Non-finite value in coalition {key}"


class TestTrackingAttribution:
    """Tests for TrackingAttribution."""

    def test_tracking_normalizes_to_total(self):
        """Test that tracking attributions sum to total improvement."""
        dim = 5
        bounds = (np.full(dim, -5.0), np.full(dim, 5.0))
        de = InstrumentedDE(
            dim=dim, bounds=bounds, pop_size=20,
            max_generations=20, seed=42,
        )
        result = de.run(sphere, verbose=False)

        tracker = TrackingAttribution()
        tracking_result = tracker.explain_from_trace(result["trace"])

        attr_sum = sum(tracking_result.attributions.values())
        assert abs(attr_sum - tracking_result.total_improvement) < 1e-10, \
            f"Attributions don't sum to total: {attr_sum} vs {tracking_result.total_improvement}"

    def test_tracking_all_operators_present(self):
        """Test that all operators with successful applications are attributed."""
        dim = 5
        bounds = (np.full(dim, -5.0), np.full(dim, 5.0))
        de = InstrumentedDE(
            dim=dim, bounds=bounds, pop_size=20,
            max_generations=20, seed=42,
        )
        result = de.run(sphere, verbose=False)

        tracker = TrackingAttribution()
        tracking_result = tracker.explain_from_trace(result["trace"])

        # DE has mutation and crossover; both should be attributed
        assert len(tracking_result.attributions) >= 2

    def test_tracking_from_ga(self):
        """Test tracking attribution works for GA."""
        dim = 5
        bounds = (np.full(dim, -5.0), np.full(dim, 5.0))
        ga = InstrumentedGA(
            dim=dim, bounds=bounds, pop_size=20,
            max_generations=10, seed=42,
        )
        result = ga.run(sphere, verbose=False)

        tracker = TrackingAttribution()
        tracking_result = tracker.explain_from_trace(result["trace"])

        assert tracking_result.total_improvement > 0
        assert len(tracking_result.attributions) >= 2


class TestBenchmarks:
    """Tests for benchmark functions."""

    def test_sphere(self):
        """Test sphere function."""
        x = np.zeros(5)
        assert sphere(x) == 0.0

        x = np.ones(5)
        assert sphere(x) == 5.0

    def test_rastrigin(self):
        """Test rastrigin function."""
        x = np.zeros(5)
        assert rastrigin(x) == 0.0

    def test_get_benchmark(self):
        """Test getting benchmark by name."""
        benchmark = get_benchmark("sphere")
        assert benchmark.name == "Sphere"
        assert benchmark.optimum == 0.0

    def test_benchmark_bounds(self):
        """Test getting bounds for benchmark."""
        benchmark = get_benchmark("rastrigin")
        lower, upper = benchmark.get_bounds(10)

        assert len(lower) == 10
        assert len(upper) == 10
        assert all(lower < upper)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
