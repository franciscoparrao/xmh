"""
Instrumented Genetic Algorithm (GA)

A fully instrumented implementation of GA with detailed logging
of all operator decisions and outcomes.
"""

from typing import Callable, Dict, List, Optional, Tuple, Any
import numpy as np

from ..core.instrumentation import InstrumentedAlgorithm, Operator
from ..core.trace_logger import InstrumentationLevel


class InstrumentedGA(InstrumentedAlgorithm):
    """
    Instrumented Genetic Algorithm for continuous optimization.

    Supports multiple selection, crossover, and mutation operators
    with full logging of operator selections and outcomes.
    """

    # Available selection methods
    SELECTION_METHODS = ["tournament", "roulette", "rank", "random"]

    # Available crossover types
    CROSSOVER_TYPES = ["sbx", "blx_alpha", "uniform", "simulated_binary"]

    # Available mutation types
    MUTATION_TYPES = ["polynomial", "gaussian", "uniform"]

    def __init__(
        self,
        dim: int,
        bounds: Tuple[np.ndarray, np.ndarray],
        pop_size: int = 50,
        max_generations: int = 100,
        selection_method: str = "tournament",
        crossover_type: str = "sbx",
        mutation_type: str = "polynomial",
        crossover_prob: float = 0.9,
        mutation_prob: float = None,  # Default: 1/dim
        tournament_size: int = 3,
        sbx_eta: float = 20.0,
        mutation_eta: float = 20.0,
        blx_alpha: float = 0.5,
        gaussian_sigma: float = 0.1,
        elitism: int = 1,
        instrumentation_level: InstrumentationLevel = InstrumentationLevel.STANDARD,
        seed: Optional[int] = None
    ):
        """
        Initialize Instrumented GA.

        Args:
            dim: Problem dimensionality
            bounds: Tuple of (lower_bounds, upper_bounds)
            pop_size: Population size (should be even for pairing)
            max_generations: Maximum generations
            selection_method: One of SELECTION_METHODS
            crossover_type: One of CROSSOVER_TYPES
            mutation_type: One of MUTATION_TYPES
            crossover_prob: Probability of crossover (pc)
            mutation_prob: Probability of mutation per gene (pm)
            tournament_size: Size for tournament selection
            sbx_eta: Distribution index for SBX crossover
            mutation_eta: Distribution index for polynomial mutation
            blx_alpha: Alpha parameter for BLX-alpha crossover
            gaussian_sigma: Sigma for Gaussian mutation (relative to range)
            elitism: Number of best individuals to preserve
            instrumentation_level: Logging detail level
            seed: Random seed
        """
        # Ensure even population size
        if pop_size % 2 != 0:
            pop_size += 1

        super().__init__(
            dim=dim,
            bounds=bounds,
            pop_size=pop_size,
            max_generations=max_generations,
            instrumentation_level=instrumentation_level,
            seed=seed
        )

        # GA-specific parameters
        self.selection_method = selection_method
        self.crossover_type = crossover_type
        self.mutation_type = mutation_type
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob if mutation_prob else 1.0 / dim
        self.tournament_size = tournament_size
        self.sbx_eta = sbx_eta
        self.mutation_eta = mutation_eta
        self.blx_alpha = blx_alpha
        self.gaussian_sigma = gaussian_sigma
        self.elitism = elitism

        # Store configuration
        self.config = {
            "selection_method": selection_method,
            "crossover_type": crossover_type,
            "mutation_type": mutation_type,
            "crossover_prob": crossover_prob,
            "mutation_prob": self.mutation_prob,
            "tournament_size": tournament_size,
            "sbx_eta": sbx_eta,
            "mutation_eta": mutation_eta,
            "blx_alpha": blx_alpha,
            "gaussian_sigma": gaussian_sigma,
            "elitism": elitism,
            "pop_size": pop_size
        }

        # Validate parameters
        if selection_method not in self.SELECTION_METHODS:
            raise ValueError(f"Unknown selection method: {selection_method}")
        if crossover_type not in self.CROSSOVER_TYPES:
            raise ValueError(f"Unknown crossover type: {crossover_type}")
        if mutation_type not in self.MUTATION_TYPES:
            raise ValueError(f"Unknown mutation type: {mutation_type}")

    def _initialize_operators(self):
        """Initialize GA operators."""

        # Selection operator
        selection_func = self._get_selection_function(self.selection_method)
        self.register_operator(Operator(
            name=f"selection_{self.selection_method}",
            operator_type="selection",
            function=selection_func,
            parameters={},
            description=f"GA selection: {self.selection_method}"
        ))

        # Crossover operator
        crossover_func = self._get_crossover_function(self.crossover_type)
        self.register_operator(Operator(
            name=f"crossover_{self.crossover_type}",
            operator_type="crossover",
            function=crossover_func,
            parameters={},
            description=f"GA crossover: {self.crossover_type}"
        ))

        # Mutation operator
        mutation_func = self._get_mutation_function(self.mutation_type)
        self.register_operator(Operator(
            name=f"mutation_{self.mutation_type}",
            operator_type="mutation",
            function=mutation_func,
            parameters={},
            description=f"GA mutation: {self.mutation_type}"
        ))

    def _get_selection_function(self, method: str) -> Callable:
        """Get selection function for method."""
        methods = {
            "tournament": self._selection_tournament,
            "roulette": self._selection_roulette,
            "rank": self._selection_rank,
            "random": self._selection_random,
        }
        return methods[method]

    def _get_crossover_function(self, crossover_type: str) -> Callable:
        """Get crossover function."""
        types = {
            "sbx": self._crossover_sbx,
            "blx_alpha": self._crossover_blx_alpha,
            "uniform": self._crossover_uniform,
            "simulated_binary": self._crossover_sbx,  # Alias
        }
        return types[crossover_type]

    def _get_mutation_function(self, mutation_type: str) -> Callable:
        """Get mutation function."""
        types = {
            "polynomial": self._mutation_polynomial,
            "gaussian": self._mutation_gaussian,
            "uniform": self._mutation_uniform,
        }
        return types[mutation_type]

    def _generation_step(self) -> Tuple[np.ndarray, np.ndarray]:
        """Perform one generation of GA."""

        # Get operators
        selection_op = self.get_operator(f"selection_{self.selection_method}")
        crossover_op = self.get_operator(f"crossover_{self.crossover_type}")
        mutation_op = self.get_operator(f"mutation_{self.mutation_type}")

        # Create offspring array
        offspring = np.empty_like(self.population)
        offspring_fitness = np.empty(self.pop_size)

        # Elitism: copy best individuals
        if self.elitism > 0:
            elite_indices = np.argsort(self.fitness)[:self.elitism]
            offspring[:self.elitism] = self.population[elite_indices]
            offspring_fitness[:self.elitism] = self.fitness[elite_indices]

        # Generate rest of offspring
        idx = self.elitism
        while idx < self.pop_size:
            # Selection
            self.logger.log_operator_selection(
                operator_name=selection_op.name,
                parameters={"tournament_size": self.tournament_size},
                rationale="Selecting parents for reproduction"
            )

            parent1_idx = selection_op.apply(
                fitness=self.fitness,
                tournament_size=self.tournament_size,
                rng=self.rng
            )
            parent2_idx = selection_op.apply(
                fitness=self.fitness,
                tournament_size=self.tournament_size,
                rng=self.rng
            )

            parent1 = self.population[parent1_idx]
            parent2 = self.population[parent2_idx]

            # Crossover
            if self.rng.random() < self.crossover_prob:
                self.logger.log_operator_selection(
                    operator_name=crossover_op.name,
                    parameters={"prob": self.crossover_prob},
                    rationale="Applying crossover"
                )

                child1, child2 = crossover_op.apply(
                    parent1=parent1,
                    parent2=parent2,
                    eta=self.sbx_eta,
                    alpha=self.blx_alpha,
                    lower=self.lower_bounds,
                    upper=self.upper_bounds,
                    rng=self.rng
                )
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # Mutation
            self.logger.log_operator_selection(
                operator_name=mutation_op.name,
                parameters={"prob": self.mutation_prob},
                rationale="Applying mutation"
            )

            child1 = mutation_op.apply(
                individual=child1,
                prob=self.mutation_prob,
                eta=self.mutation_eta,
                sigma=self.gaussian_sigma,
                lower=self.lower_bounds,
                upper=self.upper_bounds,
                rng=self.rng
            )
            child2 = mutation_op.apply(
                individual=child2,
                prob=self.mutation_prob,
                eta=self.mutation_eta,
                sigma=self.gaussian_sigma,
                lower=self.lower_bounds,
                upper=self.upper_bounds,
                rng=self.rng
            )

            # Clip to bounds
            child1 = self.clip_to_bounds(child1)
            child2 = self.clip_to_bounds(child2)

            # Evaluate
            fit1 = self._current_objective(child1)
            fit2 = self._current_objective(child2)
            self.evaluations += 2

            # Store offspring
            offspring[idx] = child1
            offspring_fitness[idx] = fit1

            # Success criterion: offspring better than best parent
            best_parent_fit = min(self.fitness[parent1_idx],
                                  self.fitness[parent2_idx])
            success1 = fit1 < best_parent_fit
            improvement1 = best_parent_fit - fit1 if success1 else 0.0

            # Log all operators that contributed to this offspring
            self.logger.log_operator_application(
                operator_name=selection_op.name,
                parent_indices=[parent1_idx, parent2_idx],
                offspring_fitness=[fit1],
                success=success1,
                details={
                    "best_parent_fit": best_parent_fit,
                    "improvement": improvement1,
                }
            )

            self.logger.log_operator_application(
                operator_name=crossover_op.name,
                parent_indices=[parent1_idx, parent2_idx],
                offspring_fitness=[fit1],
                success=success1,
                details={
                    "best_parent_fit": best_parent_fit,
                    "improvement": improvement1,
                }
            )

            self.logger.log_operator_application(
                operator_name=mutation_op.name,
                parent_indices=[parent1_idx],
                offspring_fitness=[fit1],
                success=success1,
                details={
                    "best_parent_fit": best_parent_fit,
                    "improvement": improvement1,
                }
            )

            idx += 1

            if idx < self.pop_size:
                offspring[idx] = child2
                offspring_fitness[idx] = fit2

                success2 = fit2 < best_parent_fit
                improvement2 = best_parent_fit - fit2 if success2 else 0.0

                self.logger.log_operator_application(
                    operator_name=selection_op.name,
                    parent_indices=[parent1_idx, parent2_idx],
                    offspring_fitness=[fit2],
                    success=success2,
                    details={
                        "best_parent_fit": best_parent_fit,
                        "improvement": improvement2,
                    }
                )

                self.logger.log_operator_application(
                    operator_name=crossover_op.name,
                    parent_indices=[parent1_idx, parent2_idx],
                    offspring_fitness=[fit2],
                    success=success2,
                    details={
                        "best_parent_fit": best_parent_fit,
                        "improvement": improvement2,
                    }
                )

                self.logger.log_operator_application(
                    operator_name=mutation_op.name,
                    parent_indices=[parent2_idx],
                    offspring_fitness=[fit2],
                    success=success2,
                    details={
                        "best_parent_fit": best_parent_fit,
                        "improvement": improvement2,
                    }
                )

                idx += 1

        return offspring, offspring_fitness

    # ==================== SELECTION METHODS ====================

    def _selection_tournament(
        self,
        fitness: np.ndarray,
        tournament_size: int,
        rng: np.random.Generator,
        **kwargs
    ) -> int:
        """Tournament selection."""
        candidates = rng.choice(len(fitness), tournament_size, replace=False)
        winner = candidates[np.argmin(fitness[candidates])]
        return winner

    def _selection_roulette(
        self,
        fitness: np.ndarray,
        rng: np.random.Generator,
        **kwargs
    ) -> int:
        """Roulette wheel selection (fitness proportionate)."""
        # Convert to maximization (lower fitness = higher selection prob)
        max_fit = np.max(fitness)
        adjusted = max_fit - fitness + 1e-10
        probs = adjusted / np.sum(adjusted)
        return rng.choice(len(fitness), p=probs)

    def _selection_rank(
        self,
        fitness: np.ndarray,
        rng: np.random.Generator,
        **kwargs
    ) -> int:
        """Rank-based selection."""
        ranks = np.argsort(np.argsort(fitness))  # Lower fitness = lower rank
        # Invert ranks for minimization (best = highest rank)
        ranks = len(fitness) - ranks
        probs = ranks / np.sum(ranks)
        return rng.choice(len(fitness), p=probs)

    def _selection_random(
        self,
        fitness: np.ndarray,
        rng: np.random.Generator,
        **kwargs
    ) -> int:
        """Random selection."""
        return rng.integers(0, len(fitness))

    # ==================== CROSSOVER OPERATORS ====================

    def _crossover_sbx(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
        eta: float,
        lower: np.ndarray,
        upper: np.ndarray,
        rng: np.random.Generator,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulated Binary Crossover (SBX)."""
        child1 = np.empty_like(parent1)
        child2 = np.empty_like(parent2)

        for i in range(len(parent1)):
            if rng.random() <= 0.5:
                if abs(parent1[i] - parent2[i]) > 1e-14:
                    if parent1[i] < parent2[i]:
                        y1, y2 = parent1[i], parent2[i]
                    else:
                        y1, y2 = parent2[i], parent1[i]

                    yl, yu = lower[i], upper[i]
                    rand = rng.random()

                    # Beta calculation
                    beta = 1.0 + (2.0 * (y1 - yl) / (y2 - y1))
                    alpha = 2.0 - beta ** (-(eta + 1.0))
                    if rand <= 1.0 / alpha:
                        betaq = (rand * alpha) ** (1.0 / (eta + 1.0))
                    else:
                        betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))

                    c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))

                    beta = 1.0 + (2.0 * (yu - y2) / (y2 - y1))
                    alpha = 2.0 - beta ** (-(eta + 1.0))
                    if rand <= 1.0 / alpha:
                        betaq = (rand * alpha) ** (1.0 / (eta + 1.0))
                    else:
                        betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))

                    c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1))

                    child1[i] = np.clip(c1, yl, yu)
                    child2[i] = np.clip(c2, yl, yu)
                else:
                    child1[i] = parent1[i]
                    child2[i] = parent2[i]
            else:
                child1[i] = parent1[i]
                child2[i] = parent2[i]

        return child1, child2

    def _crossover_blx_alpha(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
        alpha: float,
        lower: np.ndarray,
        upper: np.ndarray,
        rng: np.random.Generator,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """BLX-alpha crossover."""
        child1 = np.empty_like(parent1)
        child2 = np.empty_like(parent2)

        for i in range(len(parent1)):
            d = abs(parent1[i] - parent2[i])
            low = min(parent1[i], parent2[i]) - alpha * d
            high = max(parent1[i], parent2[i]) + alpha * d

            low = max(low, lower[i])
            high = min(high, upper[i])

            child1[i] = rng.uniform(low, high)
            child2[i] = rng.uniform(low, high)

        return child1, child2

    def _crossover_uniform(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
        rng: np.random.Generator,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Uniform crossover."""
        mask = rng.random(len(parent1)) < 0.5
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        return child1, child2

    # ==================== MUTATION OPERATORS ====================

    def _mutation_polynomial(
        self,
        individual: np.ndarray,
        prob: float,
        eta: float,
        lower: np.ndarray,
        upper: np.ndarray,
        rng: np.random.Generator,
        **kwargs
    ) -> np.ndarray:
        """Polynomial mutation."""
        mutant = individual.copy()

        for i in range(len(individual)):
            if rng.random() < prob:
                y = individual[i]
                yl, yu = lower[i], upper[i]
                delta1 = (y - yl) / (yu - yl)
                delta2 = (yu - y) / (yu - yl)

                rand = rng.random()
                mut_pow = 1.0 / (eta + 1.0)

                if rand < 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1.0))
                    deltaq = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1.0))
                    deltaq = 1.0 - val ** mut_pow

                mutant[i] = np.clip(y + deltaq * (yu - yl), yl, yu)

        return mutant

    def _mutation_gaussian(
        self,
        individual: np.ndarray,
        prob: float,
        sigma: float,
        lower: np.ndarray,
        upper: np.ndarray,
        rng: np.random.Generator,
        **kwargs
    ) -> np.ndarray:
        """Gaussian mutation."""
        mutant = individual.copy()
        ranges = upper - lower

        for i in range(len(individual)):
            if rng.random() < prob:
                mutant[i] += rng.normal(0, sigma * ranges[i])
                mutant[i] = np.clip(mutant[i], lower[i], upper[i])

        return mutant

    def _mutation_uniform(
        self,
        individual: np.ndarray,
        prob: float,
        lower: np.ndarray,
        upper: np.ndarray,
        rng: np.random.Generator,
        **kwargs
    ) -> np.ndarray:
        """Uniform mutation."""
        mutant = individual.copy()

        for i in range(len(individual)):
            if rng.random() < prob:
                mutant[i] = rng.uniform(lower[i], upper[i])

        return mutant


class SteadyStateGA(InstrumentedGA):
    """
    Steady-state GA variant.

    Instead of replacing entire population, replaces only worst individuals.
    """

    def __init__(
        self,
        dim: int,
        bounds: Tuple[np.ndarray, np.ndarray],
        pop_size: int = 50,
        max_generations: int = 100,
        replacement_size: int = 2,
        **kwargs
    ):
        super().__init__(dim, bounds, pop_size, max_generations, **kwargs)
        self.replacement_size = replacement_size
        self.config["variant"] = "steady_state"
        self.config["replacement_size"] = replacement_size

    def _generation_step(self) -> Tuple[np.ndarray, np.ndarray]:
        """Perform steady-state generation."""
        new_population = self.population.copy()
        new_fitness = self.fitness.copy()

        selection_op = self.get_operator(f"selection_{self.selection_method}")
        crossover_op = self.get_operator(f"crossover_{self.crossover_type}")
        mutation_op = self.get_operator(f"mutation_{self.mutation_type}")

        for _ in range(self.replacement_size // 2):
            # Select parents
            p1_idx = selection_op.apply(fitness=self.fitness, tournament_size=self.tournament_size, rng=self.rng)
            p2_idx = selection_op.apply(fitness=self.fitness, tournament_size=self.tournament_size, rng=self.rng)

            parent1 = self.population[p1_idx]
            parent2 = self.population[p2_idx]

            # Crossover
            if self.rng.random() < self.crossover_prob:
                child1, child2 = crossover_op.apply(
                    parent1=parent1, parent2=parent2,
                    eta=self.sbx_eta, alpha=self.blx_alpha,
                    lower=self.lower_bounds, upper=self.upper_bounds,
                    rng=self.rng
                )
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # Mutation
            child1 = mutation_op.apply(
                individual=child1, prob=self.mutation_prob,
                eta=self.mutation_eta, sigma=self.gaussian_sigma,
                lower=self.lower_bounds, upper=self.upper_bounds,
                rng=self.rng
            )
            child2 = mutation_op.apply(
                individual=child2, prob=self.mutation_prob,
                eta=self.mutation_eta, sigma=self.gaussian_sigma,
                lower=self.lower_bounds, upper=self.upper_bounds,
                rng=self.rng
            )

            child1 = self.clip_to_bounds(child1)
            child2 = self.clip_to_bounds(child2)

            # Evaluate
            fit1 = self._current_objective(child1)
            fit2 = self._current_objective(child2)
            self.evaluations += 2

            # Replace worst individuals if children are better
            worst_indices = np.argsort(new_fitness)[-2:]

            # Success criterion: offspring better than best parent
            best_parent_fit = min(self.fitness[p1_idx], self.fitness[p2_idx])
            success1 = fit1 < best_parent_fit
            success2 = fit2 < best_parent_fit
            improvement1 = best_parent_fit - fit1 if success1 else 0.0
            improvement2 = best_parent_fit - fit2 if success2 else 0.0

            # Log all operators for child1
            self.logger.log_operator_application(
                operator_name=selection_op.name,
                parent_indices=[p1_idx, p2_idx],
                offspring_fitness=[fit1],
                success=success1,
                details={"best_parent_fit": best_parent_fit, "improvement": improvement1}
            )
            self.logger.log_operator_application(
                operator_name=crossover_op.name,
                parent_indices=[p1_idx, p2_idx],
                offspring_fitness=[fit1],
                success=success1,
                details={"best_parent_fit": best_parent_fit, "improvement": improvement1}
            )
            self.logger.log_operator_application(
                operator_name=mutation_op.name,
                parent_indices=[p1_idx],
                offspring_fitness=[fit1],
                success=success1,
                details={"best_parent_fit": best_parent_fit, "improvement": improvement1}
            )

            # Log all operators for child2
            self.logger.log_operator_application(
                operator_name=selection_op.name,
                parent_indices=[p1_idx, p2_idx],
                offspring_fitness=[fit2],
                success=success2,
                details={"best_parent_fit": best_parent_fit, "improvement": improvement2}
            )
            self.logger.log_operator_application(
                operator_name=crossover_op.name,
                parent_indices=[p1_idx, p2_idx],
                offspring_fitness=[fit2],
                success=success2,
                details={"best_parent_fit": best_parent_fit, "improvement": improvement2}
            )
            self.logger.log_operator_application(
                operator_name=mutation_op.name,
                parent_indices=[p2_idx],
                offspring_fitness=[fit2],
                success=success2,
                details={"best_parent_fit": best_parent_fit, "improvement": improvement2}
            )

            # Replace worst individuals if children are better
            if fit1 < new_fitness[worst_indices[1]]:
                new_population[worst_indices[1]] = child1
                new_fitness[worst_indices[1]] = fit1

            if fit2 < new_fitness[worst_indices[0]]:
                new_population[worst_indices[0]] = child2
                new_fitness[worst_indices[0]] = fit2

        return new_population, new_fitness
