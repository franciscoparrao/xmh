"""
Operator-SHAP: Shapley value-based attribution for metaheuristic operators.

Adapts SHAP (SHapley Additive exPlanations) to quantify the contribution
of each operator to the final optimization result.
"""

from dataclasses import dataclass, field
from math import factorial
from typing import Any, Callable, Dict, List, Optional, Tuple
from itertools import combinations
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings


@dataclass
class SHAPExplanation:
    """
    Contains SHAP values and related explanation data for operators.

    Attributes:
        shap_values: Dict mapping operator names to their Shapley values
        base_value: Expected value without any operators (random search baseline)
        operator_names: List of operator names in order
        total_contribution: Sum of all SHAP values
        interactions: Optional pairwise interaction effects
    """
    shap_values: Dict[str, float]
    base_value: float
    operator_names: List[str]
    total_contribution: float
    interactions: Optional[Dict[Tuple[str, str], float]] = None

    # Additional metadata
    n_samples: int = 0
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    def get_ranking(self) -> List[Tuple[str, float]]:
        """Get operators ranked by absolute SHAP value."""
        return sorted(
            self.shap_values.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

    def get_positive_contributors(self) -> Dict[str, float]:
        """Get operators with positive contribution (improved fitness)."""
        return {k: v for k, v in self.shap_values.items() if v > 0}

    def get_negative_contributors(self) -> Dict[str, float]:
        """Get operators with negative contribution (worsened fitness)."""
        return {k: v for k, v in self.shap_values.items() if v < 0}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "shap_values": self.shap_values,
            "base_value": self.base_value,
            "total_contribution": self.total_contribution,
            "ranking": self.get_ranking(),
            "n_samples": self.n_samples,
            "confidence_intervals": self.confidence_intervals
        }

    def summary(self) -> str:
        """Generate text summary of the explanation."""
        lines = [
            "=" * 60,
            "OPERATOR-SHAP EXPLANATION",
            "=" * 60,
            f"Base value (random search): {self.base_value:.6e}",
            f"Total improvement: {self.total_contribution:.6e}",
            "",
            "OPERATOR CONTRIBUTIONS (ranked by importance):",
        ]

        for op, value in self.get_ranking():
            pct = (abs(value) / abs(self.total_contribution) * 100
                   if self.total_contribution != 0 else 0)
            direction = "+" if value > 0 else ""
            bar_len = int(pct / 5)
            bar = "#" * bar_len

            ci = self.confidence_intervals.get(op, (None, None))
            ci_str = f" [{ci[0]:.2e}, {ci[1]:.2e}]" if ci[0] is not None else ""

            lines.append(f"  {op:30s} {direction}{value:12.6e} ({pct:5.1f}%) {bar}{ci_str}")

        if self.interactions:
            lines.extend([
                "",
                "TOP INTERACTIONS:",
            ])
            sorted_interactions = sorted(
                self.interactions.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5]
            for (op1, op2), value in sorted_interactions:
                lines.append(f"  {op1} x {op2}: {value:+.6e}")

        lines.append("=" * 60)
        return "\n".join(lines)


class OperatorSHAP:
    """
    Computes Shapley values for metaheuristic operators.

    Uses sampling-based approximation for efficiency when the number
    of operators is large.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        n_runs_per_coalition: int = 5,
        compute_interactions: bool = False,
        confidence_level: float = 0.95,
        parallel: bool = False,
        n_jobs: int = -1
    ):
        """
        Initialize Operator-SHAP calculator.

        Args:
            n_samples: Number of permutation samples for approximation
            n_runs_per_coalition: Runs per coalition to reduce variance
            compute_interactions: Whether to compute pairwise interactions
            confidence_level: Confidence level for intervals
            parallel: Whether to use parallel computation
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.n_samples = n_samples
        self.n_runs_per_coalition = n_runs_per_coalition
        self.compute_interactions = compute_interactions
        self.confidence_level = confidence_level
        self.parallel = parallel
        self.n_jobs = n_jobs

    def explain(
        self,
        algorithm_factory: Callable,
        objective_func: Callable,
        baseline_value: Optional[float] = None
    ) -> SHAPExplanation:
        """
        Compute SHAP values for all operators in an algorithm.

        Args:
            algorithm_factory: Function that returns a fresh algorithm instance
            objective_func: The objective function being optimized
            baseline_value: Optional baseline (computed if not provided)

        Returns:
            SHAPExplanation with computed values
        """
        # Get reference algorithm to identify operators
        ref_alg = algorithm_factory()
        ref_alg._initialize_operators()
        operators = ref_alg.get_operators()
        n_operators = len(operators)

        if n_operators == 0:
            raise ValueError("Algorithm has no registered operators")

        # Compute baseline if not provided
        if baseline_value is None:
            baseline_value = self._compute_baseline(algorithm_factory, objective_func)

        # Choose method based on number of operators
        if n_operators <= 6:
            # Exact computation feasible
            shap_values, variances = self._compute_exact(
                algorithm_factory, objective_func, operators
            )
        else:
            # Use sampling approximation
            shap_values, variances = self._compute_sampled(
                algorithm_factory, objective_func, operators
            )

        # Compute confidence intervals
        confidence_intervals = {}
        z = 1.96 if self.confidence_level == 0.95 else 2.576
        for op in operators:
            std = np.sqrt(variances.get(op, 0))
            confidence_intervals[op] = (
                shap_values[op] - z * std,
                shap_values[op] + z * std
            )

        # Compute total contribution
        total_contribution = sum(shap_values.values())

        # Compute interactions if requested
        interactions = None
        if self.compute_interactions:
            interactions = self._compute_interactions(
                algorithm_factory, objective_func, operators
            )

        return SHAPExplanation(
            shap_values=shap_values,
            base_value=baseline_value,
            operator_names=operators,
            total_contribution=total_contribution,
            interactions=interactions,
            n_samples=self.n_samples,
            confidence_intervals=confidence_intervals
        )

    def _compute_baseline(
        self,
        algorithm_factory: Callable,
        objective_func: Callable
    ) -> float:
        """Compute baseline with all operators disabled (random search)."""
        values = []

        for _ in range(self.n_runs_per_coalition):
            alg = algorithm_factory()
            alg._initialize_operators()

            # Disable all operators
            for op_name in alg.get_operators():
                alg.disable_operator(op_name)

            result = alg.run(objective_func, verbose=False)
            values.append(result["best_fitness"])

        return np.mean(values)

    def _evaluate_coalition(
        self,
        algorithm_factory: Callable,
        objective_func: Callable,
        active_operators: List[str],
        all_operators: List[str]
    ) -> float:
        """
        Evaluate algorithm with only specified operators active.

        Args:
            algorithm_factory: Factory for creating algorithm instances
            objective_func: Objective function
            active_operators: List of operator names to keep active
            all_operators: List of all operator names

        Returns:
            Average best fitness over multiple runs
        """
        values = []

        for _ in range(self.n_runs_per_coalition):
            alg = algorithm_factory()
            alg._initialize_operators()

            # Disable operators not in coalition
            for op_name in all_operators:
                if op_name not in active_operators:
                    alg.disable_operator(op_name)

            result = alg.run(objective_func, verbose=False)
            values.append(result["best_fitness"])

        return np.mean(values)

    def _compute_exact(
        self,
        algorithm_factory: Callable,
        objective_func: Callable,
        operators: List[str]
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Compute exact Shapley values by enumerating all coalitions.

        Only feasible for small number of operators (≤6).
        """
        n = len(operators)
        shap_values = {op: 0.0 for op in operators}
        marginal_contributions = {op: [] for op in operators}

        # Cache coalition values
        coalition_cache = {}

        def get_coalition_value(coalition: frozenset) -> float:
            if coalition not in coalition_cache:
                coalition_cache[coalition] = self._evaluate_coalition(
                    algorithm_factory, objective_func,
                    list(coalition), operators
                )
            return coalition_cache[coalition]

        # Empty coalition
        empty_value = get_coalition_value(frozenset())

        # Enumerate all subsets for each operator
        for i, op in enumerate(operators):
            other_operators = [o for o in operators if o != op]

            # For each subset S of operators without op
            for r in range(n):
                for subset in combinations(other_operators, r):
                    S = frozenset(subset)
                    S_with_op = S | {op}

                    # Marginal contribution
                    v_with = get_coalition_value(S_with_op)
                    v_without = get_coalition_value(S) if S else empty_value

                    marginal = v_without - v_with  # Improvement (lower is better)

                    # Weight: |S|!(n-|S|-1)!/n!
                    weight = (factorial(len(S)) *
                             factorial(n - len(S) - 1) /
                             factorial(n))

                    shap_values[op] += weight * marginal
                    marginal_contributions[op].append(marginal)

        # Compute variances
        variances = {
            op: np.var(marginal_contributions[op])
            for op in operators
        }

        return shap_values, variances

    def _compute_sampled(
        self,
        algorithm_factory: Callable,
        objective_func: Callable,
        operators: List[str]
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Compute approximate Shapley values using permutation sampling.

        Efficient for large number of operators.
        """
        n = len(operators)
        marginal_contributions = {op: [] for op in operators}

        for _ in range(self.n_samples):
            # Random permutation of operators
            perm = np.random.permutation(operators).tolist()

            # Build coalitions incrementally
            prev_value = self._evaluate_coalition(
                algorithm_factory, objective_func, [], operators
            )

            coalition = []
            for op in perm:
                coalition.append(op)
                curr_value = self._evaluate_coalition(
                    algorithm_factory, objective_func, coalition, operators
                )

                # Marginal contribution (improvement = prev - curr for minimization)
                marginal = prev_value - curr_value
                marginal_contributions[op].append(marginal)

                prev_value = curr_value

        # Average marginal contributions
        shap_values = {
            op: np.mean(marginal_contributions[op])
            for op in operators
        }

        # Variances
        variances = {
            op: np.var(marginal_contributions[op]) / self.n_samples
            for op in operators
        }

        return shap_values, variances

    def _compute_interactions(
        self,
        algorithm_factory: Callable,
        objective_func: Callable,
        operators: List[str]
    ) -> Dict[Tuple[str, str], float]:
        """
        Compute pairwise SHAP interaction values.

        Interaction effect = joint contribution - sum of individual contributions.
        """
        interactions = {}

        for i, op1 in enumerate(operators):
            for op2 in operators[i+1:]:
                # Value with both
                v_both = self._evaluate_coalition(
                    algorithm_factory, objective_func,
                    [op1, op2], operators
                )

                # Value with only op1
                v_op1 = self._evaluate_coalition(
                    algorithm_factory, objective_func,
                    [op1], operators
                )

                # Value with only op2
                v_op2 = self._evaluate_coalition(
                    algorithm_factory, objective_func,
                    [op2], operators
                )

                # Value with neither
                v_none = self._evaluate_coalition(
                    algorithm_factory, objective_func,
                    [], operators
                )

                # Interaction effect
                interaction = ((v_none - v_both) -
                              (v_none - v_op1) -
                              (v_none - v_op2))

                interactions[(op1, op2)] = interaction

        return interactions


class KernelSHAP(OperatorSHAP):
    """
    Kernel SHAP implementation for operator attribution.

    Uses weighted linear regression to approximate Shapley values efficiently.
    More accurate than QuickSHAP but requires coalition evaluations.
    """

    def __init__(
        self,
        n_samples: int = 100,
        n_runs_per_coalition: int = 3,
        background_samples: int = 5,
        regularization: float = 0.01
    ):
        """
        Initialize Kernel SHAP.

        Args:
            n_samples: Number of coalition samples for approximation
            n_runs_per_coalition: Runs per coalition for variance reduction
            background_samples: Samples for baseline computation
            regularization: L2 regularization for linear regression
        """
        super().__init__(
            n_samples=n_samples,
            n_runs_per_coalition=n_runs_per_coalition
        )
        self.background_samples = background_samples
        self.regularization = regularization

    def explain(
        self,
        algorithm_class,
        algorithm_kwargs: dict,
        objective_func,
        n_coalitions: int = None
    ) -> SHAPExplanation:
        """
        Compute Kernel SHAP values for operators.

        Args:
            algorithm_class: The algorithm class to instantiate
            algorithm_kwargs: Arguments for the algorithm
            objective_func: Objective function to optimize
            n_coalitions: Number of coalitions to sample (default: 2^n for small n)

        Returns:
            SHAPExplanation with computed SHAP values
        """
        # Get operator names
        ref_alg = algorithm_class(**algorithm_kwargs)
        ref_alg._initialize_operators()
        operators = ref_alg.get_operators()
        n_operators = len(operators)

        if n_operators == 0:
            raise ValueError("Algorithm has no registered operators")

        # Determine number of coalitions
        if n_coalitions is None:
            if n_operators <= 5:
                n_coalitions = 2 ** n_operators  # All coalitions
            else:
                n_coalitions = min(self.n_samples, 2 ** n_operators)

        # Compute baseline (all operators disabled)
        baseline_fitness = self._evaluate_coalition_patched(
            algorithm_class, algorithm_kwargs, objective_func,
            [], operators, self.background_samples
        )

        # Compute full model (all operators enabled)
        full_fitness = self._evaluate_coalition_patched(
            algorithm_class, algorithm_kwargs, objective_func,
            operators, operators, self.background_samples
        )

        # Sample coalitions and compute values
        coalitions, values, weights = self._sample_coalitions(
            algorithm_class, algorithm_kwargs, objective_func,
            operators, n_coalitions
        )

        # Solve weighted linear regression for SHAP values
        shap_values = self._solve_kernel_shap(
            coalitions, values, weights, baseline_fitness, full_fitness, operators
        )

        # Total contribution
        total_contribution = baseline_fitness - full_fitness

        return SHAPExplanation(
            shap_values=shap_values,
            base_value=baseline_fitness,
            operator_names=operators,
            total_contribution=total_contribution,
            n_samples=n_coalitions
        )

    def _evaluate_coalition_patched(
        self,
        algorithm_class,
        algorithm_kwargs: dict,
        objective_func,
        active_operators: List[str],
        all_operators: List[str],
        n_runs: int = None
    ) -> float:
        """
        Evaluate algorithm with specific operators active.
        Uses the patched approach to properly disable operators.
        """
        if n_runs is None:
            n_runs = self.n_runs_per_coalition

        values = []
        operators_to_disable = [op for op in all_operators if op not in active_operators]

        for i in range(n_runs):
            # Create algorithm with modified seed
            kwargs = {**algorithm_kwargs, 'seed': algorithm_kwargs.get('seed', 42) + i * 1000}
            alg = algorithm_class(**kwargs)

            if operators_to_disable:
                # Patch to disable operators after initialization
                original_init = alg._initialize_operators
                def patched_init(self=alg, orig=original_init, ops=operators_to_disable):
                    orig()
                    for op in ops:
                        self.disable_operator(op)
                alg._initialize_operators = patched_init

            result = alg.run(objective_func, verbose=False)
            values.append(result["best_fitness"])

        return np.mean(values)

    def _sample_coalitions(
        self,
        algorithm_class,
        algorithm_kwargs: dict,
        objective_func,
        operators: List[str],
        n_coalitions: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample coalitions and compute their values and Kernel SHAP weights.

        Returns:
            coalitions: Binary matrix (n_coalitions x n_operators)
            values: Fitness values for each coalition
            weights: Kernel SHAP weights for each coalition
        """
        n_ops = len(operators)
        coalitions = []
        values = []
        weights = []

        # Always include empty and full coalitions
        coalition_set = set()

        # Add empty coalition
        empty = tuple([0] * n_ops)
        coalition_set.add(empty)

        # Add full coalition
        full = tuple([1] * n_ops)
        coalition_set.add(full)

        # Add single-operator coalitions
        for i in range(n_ops):
            single = [0] * n_ops
            single[i] = 1
            coalition_set.add(tuple(single))

        # Add complement coalitions (all but one)
        for i in range(n_ops):
            complement = [1] * n_ops
            complement[i] = 0
            coalition_set.add(tuple(complement))

        # Sample additional coalitions if needed
        rng = np.random.default_rng(42)
        attempts = 0
        max_attempts = n_coalitions * 10

        while len(coalition_set) < n_coalitions and attempts < max_attempts:
            # Random coalition
            z = tuple(rng.integers(0, 2, n_ops).tolist())
            coalition_set.add(z)
            attempts += 1

        # Evaluate each coalition
        for z in coalition_set:
            active = [operators[i] for i in range(n_ops) if z[i] == 1]
            value = self._evaluate_coalition_patched(
                algorithm_class, algorithm_kwargs, objective_func,
                active, operators
            )

            # Compute Kernel SHAP weight
            weight = self._kernel_shap_weight(z, n_ops)

            coalitions.append(list(z))
            values.append(value)
            weights.append(weight)

        return np.array(coalitions), np.array(values), np.array(weights)

    def _kernel_shap_weight(self, z: tuple, M: int) -> float:
        """
        Compute Kernel SHAP weight for a coalition.

        Weight = (M-1) / (C(M,|z|) * |z| * (M-|z|))

        where M is number of operators and |z| is coalition size.
        """
        z_size = sum(z)

        # Edge cases: empty or full coalition get high weight
        if z_size == 0 or z_size == M:
            return 1e6  # Very high weight for boundary cases

        # Binomial coefficient C(M, z_size)
        from math import comb
        binom = comb(M, z_size)

        # Kernel SHAP weight
        weight = (M - 1) / (binom * z_size * (M - z_size))

        return weight

    def _solve_kernel_shap(
        self,
        coalitions: np.ndarray,
        values: np.ndarray,
        weights: np.ndarray,
        baseline: float,
        full_value: float,
        operators: List[str]
    ) -> Dict[str, float]:
        """
        Solve weighted linear regression to get SHAP values.

        We solve: min_φ Σ w_i (f(z_i) - (φ_0 + Σ φ_j z_ij))^2 + λ||φ||^2

        Subject to efficiency: Σ φ_j = f(x) - f(∅)
        """
        n_ops = len(operators)

        # Normalize weights
        weights = weights / np.sum(weights)

        # Convert fitness to "value" (improvement over baseline)
        # For minimization: value = baseline - fitness
        y = baseline - values

        # Weighted least squares: (X^T W X + λI)^{-1} X^T W y
        X = coalitions
        W = np.diag(weights)

        # Add regularization
        reg = self.regularization * np.eye(n_ops)

        try:
            # Solve normal equations
            XtWX = X.T @ W @ X + reg
            XtWy = X.T @ W @ y
            phi = np.linalg.solve(XtWX, XtWy)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            phi = np.linalg.lstsq(X, y, rcond=None)[0]

        # Enforce efficiency constraint: sum(phi) = baseline - full_value
        total_improvement = baseline - full_value
        current_sum = np.sum(phi)

        if current_sum != 0:
            phi = phi * (total_improvement / current_sum)

        # Create dictionary
        shap_values = {operators[i]: float(phi[i]) for i in range(n_ops)}

        return shap_values


class QuickSHAP(OperatorSHAP):
    """
    Fast approximation of Operator-SHAP using trace analysis.

    Instead of re-running the algorithm, uses the logged trace
    to estimate operator contributions based on success rates
    and improvement magnitudes.
    """

    def __init__(self):
        super().__init__(n_samples=0, n_runs_per_coalition=0)

    def explain_from_trace(
        self,
        trace,
        baseline_fitness: Optional[float] = None
    ) -> SHAPExplanation:
        """
        Compute approximate SHAP values from execution trace.

        This is a fast approximation that doesn't require re-running
        the algorithm, but is less accurate than full Operator-SHAP.

        Args:
            trace: TraceLogger instance with recorded execution
            baseline_fitness: Starting fitness (uses first generation if not provided)

        Returns:
            SHAPExplanation with approximate values
        """
        if not trace.snapshots:
            raise ValueError("Trace has no recorded data")

        # Get operator contribution stats
        contributions = trace.get_operator_contributions()

        if not contributions:
            raise ValueError("No operator statistics in trace")

        # Baseline: first generation best fitness
        if baseline_fitness is None:
            baseline_fitness = trace.snapshots[0].best_fitness

        # Final fitness
        final_fitness = trace.snapshots[-1].best_fitness

        # Total improvement
        total_improvement = baseline_fitness - final_fitness

        # Approximate attribution using 3 factors:
        # usage × success_rate × avg_improvement
        shap_values = {}
        total_weighted = 0.0

        for op, stats in contributions.items():
            weighted = (
                stats["total_used"]
                * stats["success_rate"]
                * stats.get("avg_improvement", 0.0)
            )
            total_weighted += weighted

        # If no improvement data available, fall back to 2-factor formula
        if total_weighted == 0.0:
            for op, stats in contributions.items():
                weighted = stats["success_rate"] * stats["total_used"]
                total_weighted += weighted

            for op, stats in contributions.items():
                weighted = stats["success_rate"] * stats["total_used"]
                if total_weighted > 0:
                    proportion = weighted / total_weighted
                else:
                    proportion = 1.0 / len(contributions)
                shap_values[op] = total_improvement * proportion
        else:
            # Distribute total improvement proportionally using 3 factors
            for op, stats in contributions.items():
                weighted = (
                    stats["total_used"]
                    * stats["success_rate"]
                    * stats.get("avg_improvement", 0.0)
                )
                proportion = weighted / total_weighted
                shap_values[op] = total_improvement * proportion

        return SHAPExplanation(
            shap_values=shap_values,
            base_value=baseline_fitness,
            operator_names=list(contributions.keys()),
            total_contribution=total_improvement,
            n_samples=0,
            confidence_intervals={}
        )


class ExactOperatorSHAP:
    """
    Exact Shapley value computation for metaheuristic operators.

    Enumerates all 2^n coalitions (feasible because n is typically 2-4
    for metaheuristic operators). Uses controlled seeds across multiple
    runs per coalition to compute confidence intervals and exact Shapley
    interaction indices.
    """

    def __init__(
        self,
        n_runs_per_coalition: int = 30,
        base_seed: int = 42,
        compute_interactions: bool = True,
    ):
        """
        Args:
            n_runs_per_coalition: Runs per coalition for variance reduction.
            base_seed: Base random seed; run i uses base_seed + i.
            compute_interactions: Whether to compute pairwise interaction indices.
        """
        self.n_runs_per_coalition = n_runs_per_coalition
        self.base_seed = base_seed
        self.compute_interactions = compute_interactions

    def explain(
        self,
        algorithm_class,
        algorithm_kwargs: dict,
        objective_func: Callable,
    ) -> SHAPExplanation:
        """
        Compute exact Shapley values for all operators.

        Args:
            algorithm_class: The algorithm class to instantiate.
            algorithm_kwargs: Arguments for the algorithm constructor.
            objective_func: Objective function to optimize.

        Returns:
            SHAPExplanation with exact Shapley values, confidence
            intervals, and optionally interaction indices.
        """
        # Get operator names from a reference instance
        ref_alg = algorithm_class(**algorithm_kwargs)
        ref_alg._initialize_operators()
        operators = ref_alg.get_operators()
        n = len(operators)

        if n == 0:
            raise ValueError("Algorithm has no registered operators")
        if n > 6:
            warnings.warn(
                f"Exact Shapley with {n} operators requires {2**n} "
                f"coalitions. Consider using sampling approximation."
            )

        # Enumerate all 2^n coalitions and evaluate each
        coalition_values = {}  # frozenset -> list of fitness values
        all_coalitions = []

        for size in range(n + 1):
            for subset in combinations(range(n), size):
                coalition = frozenset(subset)
                all_coalitions.append(coalition)

        # Evaluate each coalition with the same set of seeds
        for coalition in all_coalitions:
            active_ops = [operators[i] for i in coalition]
            values = self._evaluate_coalition(
                algorithm_class, algorithm_kwargs, objective_func,
                active_ops, operators,
            )
            coalition_values[coalition] = values

        # Compute Shapley values from coalition values
        shap_values, shap_variances = self._compute_shapley(
            operators, coalition_values
        )

        # Confidence intervals (95%)
        z = 1.96
        confidence_intervals = {}
        for op in operators:
            std = np.sqrt(shap_variances[op])
            confidence_intervals[op] = (
                shap_values[op] - z * std,
                shap_values[op] + z * std,
            )

        # Interaction indices
        interactions = None
        if self.compute_interactions:
            interactions = self._compute_interaction_indices(
                operators, coalition_values
            )

        # Base and full values
        empty_mean = np.mean(coalition_values[frozenset()])
        full_mean = np.mean(coalition_values[frozenset(range(n))])
        total_contribution = empty_mean - full_mean  # improvement for minimization

        return SHAPExplanation(
            shap_values=shap_values,
            base_value=empty_mean,
            operator_names=operators,
            total_contribution=total_contribution,
            interactions=interactions,
            n_samples=self.n_runs_per_coalition,
            confidence_intervals=confidence_intervals,
        )

    def get_coalition_values(
        self,
        algorithm_class,
        algorithm_kwargs: dict,
        objective_func: Callable,
    ) -> Dict[str, Any]:
        """
        Return the raw coalition values for external analysis.

        Returns a dict with:
            - 'operators': list of operator names
            - 'coalitions': dict mapping coalition tuple -> list of fitness values
            - 'shapley_values': dict of operator -> Shapley value
            - 'interactions': dict of (op_i, op_j) -> interaction value
        """
        ref_alg = algorithm_class(**algorithm_kwargs)
        ref_alg._initialize_operators()
        operators = ref_alg.get_operators()
        n = len(operators)

        coalition_values = {}
        for size in range(n + 1):
            for subset in combinations(range(n), size):
                coalition = frozenset(subset)
                active_ops = [operators[i] for i in coalition]
                values = self._evaluate_coalition(
                    algorithm_class, algorithm_kwargs, objective_func,
                    active_ops, operators,
                )
                coalition_values[coalition] = values

        shap_values, _ = self._compute_shapley(operators, coalition_values)

        interactions = None
        if self.compute_interactions:
            interactions = self._compute_interaction_indices(
                operators, coalition_values
            )

        # Convert frozenset keys to serializable tuples
        serializable_coalitions = {
            tuple(sorted(k)): v for k, v in coalition_values.items()
        }

        return {
            "operators": operators,
            "coalitions": serializable_coalitions,
            "shapley_values": shap_values,
            "interactions": interactions,
        }

    def _evaluate_coalition(
        self,
        algorithm_class,
        algorithm_kwargs: dict,
        objective_func: Callable,
        active_operators: List[str],
        all_operators: List[str],
    ) -> List[float]:
        """
        Evaluate a coalition using controlled seeds.

        Returns a list of best_fitness values (one per run).
        """
        operators_to_disable = [
            op for op in all_operators if op not in active_operators
        ]
        values = []

        for i in range(self.n_runs_per_coalition):
            seed = self.base_seed + i
            kwargs = {**algorithm_kwargs, "seed": seed}
            alg = algorithm_class(**kwargs)

            if operators_to_disable:
                original_init = alg._initialize_operators

                def patched_init(
                    _self=alg, _orig=original_init, _ops=operators_to_disable
                ):
                    _orig()
                    for op in _ops:
                        _self.disable_operator(op)

                alg._initialize_operators = patched_init

            result = alg.run(objective_func, verbose=False)
            values.append(result["best_fitness"])

        return values

    def _compute_shapley(
        self,
        operators: List[str],
        coalition_values: Dict[frozenset, List[float]],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Compute exact Shapley values from coalition values.

        For each run r, compute Shapley values using v_r(S), then
        average across runs. Also compute variance.
        """
        n = len(operators)
        n_runs = self.n_runs_per_coalition

        # Per-run Shapley values
        per_run_shapley = {op: [] for op in operators}

        for r in range(n_runs):
            # Build v(S) for this run
            def v(S):
                return coalition_values[S][r]

            for i, op in enumerate(operators):
                phi_i = 0.0
                other_indices = [j for j in range(n) if j != i]

                for size in range(n):
                    for subset in combinations(other_indices, size):
                        S = frozenset(subset)
                        S_with_i = S | {i}

                        marginal = v(S) - v(S_with_i)  # improvement (minimization)

                        weight = (
                            factorial(len(S))
                            * factorial(n - len(S) - 1)
                            / factorial(n)
                        )
                        phi_i += weight * marginal

                per_run_shapley[op].append(phi_i)

        # Average and variance
        shap_values = {
            op: np.mean(per_run_shapley[op]) for op in operators
        }
        shap_variances = {
            op: np.var(per_run_shapley[op], ddof=1) / n_runs
            for op in operators
        }

        return shap_values, shap_variances

    def _compute_interaction_indices(
        self,
        operators: List[str],
        coalition_values: Dict[frozenset, List[float]],
    ) -> Dict[Tuple[str, str], float]:
        """
        Compute exact Shapley interaction indices.

        The interaction index I_{ij} is defined as:
        I_{ij} = sum over S not containing i,j of
            w(S) * [v(S∪{i,j}) - v(S∪{i}) - v(S∪{j}) + v(S)]

        where w(S) = |S|!(n-|S|-2)! / (n-1)!
        """
        n = len(operators)
        interactions = {}

        # Average v(S) across runs
        def v_mean(S):
            return np.mean(coalition_values[S])

        for i in range(n):
            for j in range(i + 1, n):
                other_indices = [k for k in range(n) if k != i and k != j]
                interaction = 0.0

                for size in range(n - 1):
                    for subset in combinations(other_indices, size):
                        S = frozenset(subset)
                        S_i = S | {i}
                        S_j = S | {j}
                        S_ij = S | {i, j}

                        # For minimization: improvement = v_without - v_with
                        delta = (
                            v_mean(S) - v_mean(S_i) - v_mean(S_j) + v_mean(S_ij)
                        )

                        s = len(S)
                        weight = (
                            factorial(s)
                            * factorial(n - s - 2)
                            / factorial(n - 1)
                        )
                        interaction += weight * delta

                interactions[(operators[i], operators[j])] = interaction

        return interactions
