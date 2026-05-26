"""
Instrumented CMA-ES for the XMH framework.

Implements CMA-ES (Covariance Matrix Adaptation Evolution Strategy) following
Hansen (2016) "The CMA Evolution Strategy: A Tutorial", with operator-level
instrumentation for Shapley value attribution.

Operators for the cooperative game (n=3, 2^3=8 coalitions):
  1. Weighted recombination  - fitness-proportionate mean update
  2. Covariance adaptation   - rank-1 + rank-mu C matrix update
  3. Step-size adaptation    - cumulative step-size adaptation (CSA)

Neutral operators for ablation:
  - Neutral recombination:  mean unchanged (no selection pressure)
  - Neutral covariance:     C = I throughout (isotropic search)
  - Neutral step-size:      sigma fixed at sigma0 (no scale adaptation)
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, Callable

from ..core.instrumentation import (
    InstrumentedAlgorithm, Operator, InstrumentationLevel,
    get_neutral_operator
)


# ---------------------------------------------------------------------------
# Neutral operators for CMA-ES ablation studies
# ---------------------------------------------------------------------------

class NeutralRecombinationOperator(Operator):
    """Neutral recombination: keeps mean unchanged (no selection pressure)."""

    def __init__(self):
        super().__init__(
            name="neutral_recombination",
            operator_type="recombination",
            function=self._neutral,
            parameters={},
            description="Neutral: mean unchanged (no weighted recombination)"
        )

    def _neutral(self, old_mean=None, **kwargs):
        return old_mean.copy()


class NeutralCovarianceOperator(Operator):
    """Neutral covariance: C stays as identity (isotropic Gaussian)."""

    def __init__(self, dim: int):
        self._dim = dim
        super().__init__(
            name="neutral_covariance",
            operator_type="covariance",
            function=self._neutral,
            parameters={},
            description="Neutral: C = I, p_c = 0 (isotropic search)"
        )

    def _neutral(self, **kwargs):
        return np.eye(self._dim), np.zeros(self._dim)


class NeutralStepSizeOperator(Operator):
    """Neutral step-size: sigma fixed at initial value."""

    def __init__(self, sigma0: float):
        self._sigma0 = sigma0
        super().__init__(
            name="neutral_step_size",
            operator_type="step_size",
            function=self._neutral,
            parameters={},
            description=f"Neutral: sigma fixed at {sigma0:.4f}"
        )

    def _neutral(self, p_sigma=None, **kwargs):
        return self._sigma0, np.zeros_like(p_sigma)


# ---------------------------------------------------------------------------
# InstrumentedCMAES
# ---------------------------------------------------------------------------

class InstrumentedCMAES(InstrumentedAlgorithm):
    """
    CMA-ES with operator-level instrumentation for Shapley attribution.

    Implements Hansen's (μ/μ_w, λ)-CMA-ES with three ablatable operators
    for the cooperative game:

      - **recombination_weighted**: weighted mean from μ best offspring
        (selection pressure). Neutral: mean stays at initial position.
      - **covariance_adaptation**: rank-1 + rank-μ update of C
        (distribution shape learning). Neutral: C = I (isotropic).
      - **step_size_adaptation**: cumulative step-size adaptation of σ
        (global scale control). Neutral: σ fixed at σ₀.

    Parameters
    ----------
    dim : int
        Problem dimensionality.
    bounds : tuple
        (lower_bounds, upper_bounds) as numpy arrays or lists.
    pop_size : int, optional
        Population size λ. Default: 4 + floor(3·ln(dim)).
    max_generations : int
        Maximum number of generations.
    sigma0 : float, optional
        Initial step size. Default: mean(upper−lower) / 4.
    mean0 : array-like, optional
        Initial mean vector. Default: random point in bounds (via RNG).
        Important: center of bounds coincides with the optimum for many
        benchmarks (sphere, rastrigin, ackley), biasing ablation results.
    seed : int, optional
        Random seed for reproducibility.
    instrumentation_level : InstrumentationLevel
        Logging detail level.
    """

    def __init__(
        self,
        dim: int,
        bounds: Tuple[np.ndarray, np.ndarray],
        pop_size: Optional[int] = None,
        max_generations: int = 100,
        sigma0: Optional[float] = None,
        mean0: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        instrumentation_level: InstrumentationLevel = InstrumentationLevel.STANDARD,
    ):
        if pop_size is None:
            pop_size = 4 + int(3 * np.log(dim))

        super().__init__(
            dim=dim,
            bounds=bounds,
            pop_size=pop_size,
            max_generations=max_generations,
            instrumentation_level=instrumentation_level,
            seed=seed,
        )

        # --- CMA-ES strategy parameters (from Hansen 2016, Table 1) ---
        self.mu = pop_size // 2

        # Recombination weights (log-linear, positive only)
        weights_prime = np.array([
            np.log(self.mu + 0.5) - np.log(i + 1) for i in range(self.mu)
        ])
        self.weights = weights_prime / np.sum(weights_prime)
        self.mu_eff = 1.0 / np.sum(self.weights ** 2)

        # Step-size control
        self.c_sigma = (self.mu_eff + 2) / (dim + self.mu_eff + 5)
        self.d_sigma = (
            1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (dim + 1)) - 1)
            + self.c_sigma
        )

        # Covariance adaptation
        self.c_c = (4 + self.mu_eff / dim) / (dim + 4 + 2 * self.mu_eff / dim)
        self.c_1 = 2 / ((dim + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(
            1 - self.c_1,
            2 * (self.mu_eff - 2 + 1 / self.mu_eff)
            / ((dim + 2) ** 2 + self.mu_eff),
        )

        # Initial step-size
        if sigma0 is None:
            self.sigma0 = float(
                np.mean(np.asarray(bounds[1]) - np.asarray(bounds[0]))
            ) / 4.0
        else:
            self.sigma0 = float(sigma0)

        # Initial mean (stored for _initialize_operators)
        self.mean0 = np.asarray(mean0, dtype=float) if mean0 is not None else None

        # Expected ||N(0,I)||
        self.chi_n = np.sqrt(dim) * (
            1 - 1 / (4 * dim) + 1 / (21 * dim ** 2)
        )

        # Distribution state (initialized in _initialize_operators)
        self.mean: Optional[np.ndarray] = None
        self.sigma: Optional[float] = None
        self.C: Optional[np.ndarray] = None
        self.p_sigma: Optional[np.ndarray] = None
        self.p_c: Optional[np.ndarray] = None
        self.eigenvalues: Optional[np.ndarray] = None
        self.eigenvectors: Optional[np.ndarray] = None
        self.invsqrt_C: Optional[np.ndarray] = None

        self.config = {
            "algorithm": "CMA-ES",
            "dim": dim,
            "pop_size": pop_size,
            "mu": self.mu,
            "sigma0": self.sigma0,
            "mean0": self.mean0.tolist() if self.mean0 is not None else None,
            "mu_eff": float(self.mu_eff),
            "c_sigma": float(self.c_sigma),
            "d_sigma": float(self.d_sigma),
            "c_c": float(self.c_c),
            "c_1": float(self.c_1),
            "c_mu": float(self.c_mu),
            "max_generations": max_generations,
            "seed": seed,
        }

    # ------------------------------------------------------------------
    # Framework hooks
    # ------------------------------------------------------------------

    def _initialize_operators(self):
        """Register CMA-ES operators and initialize distribution state."""
        self.register_operator(Operator(
            name="recombination_weighted",
            operator_type="recombination",
            function=self._weighted_recombination,
            parameters={},
            description="CMA-ES weighted mean recombination from mu best",
        ))
        self.register_operator(Operator(
            name="covariance_adaptation",
            operator_type="covariance",
            function=self._adapt_covariance,
            parameters={},
            description="CMA-ES rank-1 + rank-mu covariance adaptation",
        ))
        self.register_operator(Operator(
            name="step_size_adaptation",
            operator_type="step_size",
            function=self._adapt_step_size,
            parameters={},
            description="CMA-ES cumulative step-size adaptation (CSA)",
        ))

        # Initialize distribution mean
        if self.mean0 is not None:
            self.mean = self.mean0.copy()
        else:
            # Random point in bounds (avoids center-of-bounds bias)
            self.mean = self.rng.uniform(self.lower_bounds, self.upper_bounds)
        self.sigma = self.sigma0
        self.C = np.eye(self.dim)
        self.p_sigma = np.zeros(self.dim)
        self.p_c = np.zeros(self.dim)
        self._update_eigen()

    def disable_operator(self, name: str):
        """Replace CMA-ES operator with the appropriate neutral version."""
        operator = self.get_operator(name)
        op_type = operator.operator_type

        if op_type == "recombination":
            neutral = NeutralRecombinationOperator()
        elif op_type == "covariance":
            neutral = NeutralCovarianceOperator(self.dim)
        elif op_type == "step_size":
            neutral = NeutralStepSizeOperator(self.sigma0)
        else:
            neutral = get_neutral_operator(op_type)

        self.replace_operator(name, neutral)

    def initialize_population(self) -> np.ndarray:
        """Sample initial population from N(mean, sigma0^2 · I)."""
        population = np.empty((self.pop_size, self.dim))
        for i in range(self.pop_size):
            z = self.rng.standard_normal(self.dim)
            population[i] = self.mean + self.sigma0 * z
        return self.clip_to_bounds(population)

    # ------------------------------------------------------------------
    # Operator functions
    # ------------------------------------------------------------------

    def _weighted_recombination(
        self, offspring, sorted_indices, weights, mu, old_mean, **kwargs
    ):
        """Weighted mean of μ best offspring (selection + recombination)."""
        selected = offspring[sorted_indices[:mu]]
        new_mean = np.zeros(self.dim)
        for i in range(mu):
            new_mean += weights[i] * selected[i]
        return new_mean

    def _adapt_step_size(
        self, sigma, mean, old_mean, p_sigma, invsqrt_C,
        c_sigma, d_sigma, chi_n, mu_eff, **kwargs
    ):
        """CSA: update evolution path p_sigma and step-size sigma."""
        mean_diff = (mean - old_mean) / sigma
        new_p_sigma = (
            (1 - c_sigma) * p_sigma
            + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff)
            * (invsqrt_C @ mean_diff)
        )
        new_sigma = sigma * np.exp(
            (c_sigma / d_sigma) * (np.linalg.norm(new_p_sigma) / chi_n - 1)
        )
        new_sigma = np.clip(new_sigma, 1e-20, 1e10)
        return new_sigma, new_p_sigma

    def _adapt_covariance(
        self, C, mean, old_mean, sigma, offspring, sorted_indices,
        weights, mu, p_c, p_sigma, c_c, c_1, c_mu, mu_eff, dim,
        generation, chi_n, c_sigma, **kwargs
    ):
        """Rank-1 + rank-μ covariance matrix adaptation."""
        mean_diff = (mean - old_mean) / sigma

        # h_sigma stall indicator (Hansen 2016, eq. 8)
        p_sigma_norm = np.linalg.norm(p_sigma)
        threshold = (1.4 + 2.0 / (dim + 1)) * chi_n
        denom = np.sqrt(1 - (1 - c_sigma) ** (2 * (generation + 1)))
        h_sigma = 1.0 if (denom > 0 and p_sigma_norm / denom < threshold) else 0.0

        # Update evolution path p_c
        new_p_c = (
            (1 - c_c) * p_c
            + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * mean_diff
        )

        # Rank-μ matrix
        selected = offspring[sorted_indices[:mu]]
        artmp = (selected - old_mean) / sigma  # (mu, dim)
        rank_mu = np.zeros((dim, dim))
        for i in range(mu):
            rank_mu += weights[i] * np.outer(artmp[i], artmp[i])

        # Full C update (Hansen 2016, eq. 11)
        new_C = (
            (1 - c_1 - c_mu) * C
            + c_1 * (
                np.outer(new_p_c, new_p_c)
                + (1 - h_sigma) * c_c * (2 - c_c) * C
            )
            + c_mu * rank_mu
        )

        return new_C, new_p_c

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_eigen(self):
        """Eigendecompose C for sampling and invsqrt computation."""
        self.C = (self.C + self.C.T) / 2  # enforce symmetry
        eigenvalues, eigenvectors = np.linalg.eigh(self.C)
        self.eigenvalues = np.maximum(eigenvalues, 1e-20)
        self.eigenvectors = eigenvectors
        self.invsqrt_C = (
            eigenvectors
            @ np.diag(1.0 / np.sqrt(self.eigenvalues))
            @ eigenvectors.T
        )

    # ------------------------------------------------------------------
    # Main generation loop
    # ------------------------------------------------------------------

    def _generation_step(self) -> Tuple[np.ndarray, np.ndarray]:
        """One CMA-ES generation: sample, evaluate, adapt distribution."""
        recomb_op = self.get_operator("recombination_weighted")
        step_op = self.get_operator("step_size_adaptation")
        cov_op = self.get_operator("covariance_adaptation")

        old_mean = self.mean.copy()
        old_best = self.best_fitness

        # 1. Sample λ offspring from N(mean, σ² C)
        offspring = np.empty((self.pop_size, self.dim))
        for i in range(self.pop_size):
            z = self.rng.standard_normal(self.dim)
            y = self.eigenvectors @ (np.sqrt(self.eigenvalues) * z)
            offspring[i] = self.mean + self.sigma * y
        offspring = self.clip_to_bounds(offspring)

        # 2. Evaluate
        offspring_fitness = np.empty(self.pop_size)
        for i in range(self.pop_size):
            offspring_fitness[i] = self._current_objective(offspring[i])
            self.evaluations += 1

        # 3. Sort by fitness (minimization)
        sorted_indices = np.argsort(offspring_fitness)

        # 4. Weighted recombination (mean update)
        self.logger.log_operator_selection(
            operator_name=recomb_op.name,
            parameters={"mu": self.mu},
            rationale=f"Weighted recombination of {self.mu} best offspring",
        )
        self.mean = recomb_op.apply(
            offspring=offspring,
            sorted_indices=sorted_indices,
            weights=self.weights,
            mu=self.mu,
            old_mean=old_mean,
        )

        # 5. Step-size adaptation (CSA)
        self.logger.log_operator_selection(
            operator_name=step_op.name,
            parameters={"sigma": float(self.sigma)},
            rationale="Cumulative step-size adaptation",
        )
        self.sigma, self.p_sigma = step_op.apply(
            sigma=self.sigma,
            mean=self.mean,
            old_mean=old_mean,
            p_sigma=self.p_sigma,
            invsqrt_C=self.invsqrt_C,
            c_sigma=self.c_sigma,
            d_sigma=self.d_sigma,
            chi_n=self.chi_n,
            mu_eff=self.mu_eff,
        )

        # 6. Covariance adaptation
        self.logger.log_operator_selection(
            operator_name=cov_op.name,
            parameters={"c_1": self.c_1, "c_mu": self.c_mu},
            rationale="Rank-1 + rank-mu covariance adaptation",
        )
        self.C, self.p_c = cov_op.apply(
            C=self.C,
            mean=self.mean,
            old_mean=old_mean,
            sigma=self.sigma,
            offspring=offspring,
            sorted_indices=sorted_indices,
            weights=self.weights,
            mu=self.mu,
            p_c=self.p_c,
            p_sigma=self.p_sigma,
            c_c=self.c_c,
            c_1=self.c_1,
            c_mu=self.c_mu,
            mu_eff=self.mu_eff,
            dim=self.dim,
            generation=self.generation,
            chi_n=self.chi_n,
            c_sigma=self.c_sigma,
        )

        # 7. Eigendecomposition for next generation's sampling
        self._update_eigen()

        # 8. Log operator applications (all share the same success flag)
        gen_best = offspring_fitness[sorted_indices[0]]
        success = gen_best < old_best
        improvement = float(old_best - gen_best) if success else 0.0

        details = {
            "improvement": improvement,
            "sigma": float(self.sigma),
            "mean_shift": float(np.linalg.norm(self.mean - old_mean)),
            "condition_number": float(
                self.eigenvalues[-1] / self.eigenvalues[0]
            ) if self.eigenvalues[0] > 0 else float("inf"),
        }

        parent_indices = sorted_indices[: self.mu].tolist()
        for op in [recomb_op, step_op, cov_op]:
            self.logger.log_operator_application(
                operator_name=op.name,
                parent_indices=parent_indices,
                offspring_fitness=[float(gen_best)],
                success=success,
                details=details,
            )

        return offspring, offspring_fitness
