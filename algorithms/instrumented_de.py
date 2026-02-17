"""
Instrumented Differential Evolution (DE)

A fully instrumented implementation of DE with detailed logging
of all operator decisions and outcomes.
"""

from typing import Callable, Dict, List, Optional, Tuple, Any
import numpy as np

from ..core.instrumentation import InstrumentedAlgorithm, Operator
from ..core.trace_logger import InstrumentationLevel


class InstrumentedDE(InstrumentedAlgorithm):
    """
    Instrumented Differential Evolution algorithm.

    Supports multiple mutation strategies and crossover types,
    with full logging of operator selections and outcomes.
    """

    # Available mutation strategies
    MUTATION_STRATEGIES = [
        "rand/1",
        "best/1",
        "current-to-best/1",
        "rand/2",
        "best/2"
    ]

    # Available crossover types
    CROSSOVER_TYPES = ["binomial", "exponential"]

    def __init__(
        self,
        dim: int,
        bounds: Tuple[np.ndarray, np.ndarray],
        pop_size: int = 50,
        max_generations: int = 100,
        F: float = 0.5,
        CR: float = 0.9,
        mutation_strategy: str = "rand/1",
        crossover_type: str = "binomial",
        instrumentation_level: InstrumentationLevel = InstrumentationLevel.STANDARD,
        seed: Optional[int] = None
    ):
        """
        Initialize Instrumented DE.

        Args:
            dim: Problem dimensionality
            bounds: Tuple of (lower_bounds, upper_bounds)
            pop_size: Population size
            max_generations: Maximum generations
            F: Mutation scale factor
            CR: Crossover probability
            mutation_strategy: One of MUTATION_STRATEGIES
            crossover_type: "binomial" or "exponential"
            instrumentation_level: Logging detail level
            seed: Random seed
        """
        super().__init__(
            dim=dim,
            bounds=bounds,
            pop_size=pop_size,
            max_generations=max_generations,
            instrumentation_level=instrumentation_level,
            seed=seed
        )

        # DE-specific parameters
        self.F = F
        self.CR = CR
        self.mutation_strategy = mutation_strategy
        self.crossover_type = crossover_type

        # Store configuration
        self.config = {
            "F": F,
            "CR": CR,
            "mutation_strategy": mutation_strategy,
            "crossover_type": crossover_type,
            "pop_size": pop_size,
            "max_generations": max_generations
        }

        # Validate parameters
        if mutation_strategy not in self.MUTATION_STRATEGIES:
            raise ValueError(f"Unknown mutation strategy: {mutation_strategy}")
        if crossover_type not in self.CROSSOVER_TYPES:
            raise ValueError(f"Unknown crossover type: {crossover_type}")

    def _initialize_operators(self):
        """Initialize DE operators."""

        # Mutation operator (parameters passed explicitly in apply)
        mutation_func = self._get_mutation_function(self.mutation_strategy)
        self.register_operator(Operator(
            name=f"mutation_{self.mutation_strategy}",
            operator_type="mutation",
            function=mutation_func,
            parameters={},  # F passed explicitly
            description=f"DE mutation: {self.mutation_strategy}"
        ))

        # Crossover operator (parameters passed explicitly in apply)
        crossover_func = self._get_crossover_function(self.crossover_type)
        self.register_operator(Operator(
            name=f"crossover_{self.crossover_type}",
            operator_type="crossover",
            function=crossover_func,
            parameters={},  # CR passed explicitly
            description=f"DE crossover: {self.crossover_type}"
        ))

        # Selection operator (greedy)
        self.register_operator(Operator(
            name="selection_greedy",
            operator_type="selection",
            function=self._greedy_selection,
            parameters={},
            description="DE greedy selection"
        ))

    def _get_mutation_function(self, strategy: str) -> Callable:
        """Get mutation function for strategy."""
        strategies = {
            "rand/1": self._mutation_rand_1,
            "best/1": self._mutation_best_1,
            "current-to-best/1": self._mutation_current_to_best_1,
            "rand/2": self._mutation_rand_2,
            "best/2": self._mutation_best_2,
        }
        return strategies[strategy]

    def _get_crossover_function(self, crossover_type: str) -> Callable:
        """Get crossover function."""
        types = {
            "binomial": self._crossover_binomial,
            "exponential": self._crossover_exponential,
        }
        return types[crossover_type]

    def _generation_step(self) -> Tuple[np.ndarray, np.ndarray]:
        """Perform one generation of DE."""

        new_population = np.empty_like(self.population)
        new_fitness = np.empty_like(self.fitness)

        # Get operators
        mutation_op = self.get_operator(f"mutation_{self.mutation_strategy}")
        crossover_op = self.get_operator(f"crossover_{self.crossover_type}")
        selection_op = self.get_operator("selection_greedy")

        # Find best individual for strategies that need it
        best_idx = np.argmin(self.fitness)
        best_individual = self.population[best_idx]

        for i in range(self.pop_size):
            # Log operator selection
            self.logger.log_operator_selection(
                operator_name=mutation_op.name,
                parameters={"F": self.F, "target_idx": i},
                rationale=f"Applying {self.mutation_strategy} mutation to individual {i}"
            )

            # Apply mutation
            mutant = mutation_op.apply(
                population=self.population,
                target_idx=i,
                best=best_individual,
                F=self.F,
                rng=self.rng
            )
            mutant = self.clip_to_bounds(mutant)

            # Log crossover selection
            self.logger.log_operator_selection(
                operator_name=crossover_op.name,
                parameters={"CR": self.CR, "target_idx": i},
                rationale=f"Applying {self.crossover_type} crossover"
            )

            # Apply crossover
            trial = crossover_op.apply(
                target=self.population[i],
                mutant=mutant,
                CR=self.CR,
                rng=self.rng
            )

            # Evaluate trial
            trial_fitness = self._current_objective(trial)
            self.evaluations += 1

            # Selection
            if trial_fitness <= self.fitness[i]:
                new_population[i] = trial
                new_fitness[i] = trial_fitness
                success = True
            else:
                new_population[i] = self.population[i]
                new_fitness[i] = self.fitness[i]
                success = False

            # Log operator application results (both mutation and crossover contribute to trial success)
            self.logger.log_operator_application(
                operator_name=mutation_op.name,
                parent_indices=[i],
                offspring_fitness=[trial_fitness],
                success=success,
                details={
                    "target_fitness": self.fitness[i],
                    "trial_fitness": trial_fitness,
                    "improvement": self.fitness[i] - trial_fitness if success else 0
                }
            )

            # Crossover also contributed to the trial - log its success too
            self.logger.log_operator_application(
                operator_name=crossover_op.name,
                parent_indices=[i],
                offspring_fitness=[trial_fitness],
                success=success,
                details={
                    "target_fitness": self.fitness[i],
                    "trial_fitness": trial_fitness,
                    "improvement": self.fitness[i] - trial_fitness if success else 0,
                    "CR": self.CR
                }
            )

        return new_population, new_fitness

    # ==================== MUTATION STRATEGIES ====================

    def _mutation_rand_1(
        self,
        population: np.ndarray,
        target_idx: int,
        F: float,
        rng: np.random.Generator,
        **kwargs
    ) -> np.ndarray:
        """DE/rand/1 mutation: v = x_r1 + F * (x_r2 - x_r3)"""
        idxs = [i for i in range(len(population)) if i != target_idx]
        r1, r2, r3 = rng.choice(idxs, 3, replace=False)
        return population[r1] + F * (population[r2] - population[r3])

    def _mutation_best_1(
        self,
        population: np.ndarray,
        target_idx: int,
        best: np.ndarray,
        F: float,
        rng: np.random.Generator,
        **kwargs
    ) -> np.ndarray:
        """DE/best/1 mutation: v = x_best + F * (x_r1 - x_r2)"""
        idxs = [i for i in range(len(population)) if i != target_idx]
        r1, r2 = rng.choice(idxs, 2, replace=False)
        return best + F * (population[r1] - population[r2])

    def _mutation_current_to_best_1(
        self,
        population: np.ndarray,
        target_idx: int,
        best: np.ndarray,
        F: float,
        rng: np.random.Generator,
        **kwargs
    ) -> np.ndarray:
        """DE/current-to-best/1: v = x_i + F * (x_best - x_i) + F * (x_r1 - x_r2)"""
        idxs = [i for i in range(len(population)) if i != target_idx]
        r1, r2 = rng.choice(idxs, 2, replace=False)
        return (population[target_idx] +
                F * (best - population[target_idx]) +
                F * (population[r1] - population[r2]))

    def _mutation_rand_2(
        self,
        population: np.ndarray,
        target_idx: int,
        F: float,
        rng: np.random.Generator,
        **kwargs
    ) -> np.ndarray:
        """DE/rand/2 mutation: v = x_r1 + F * (x_r2 - x_r3) + F * (x_r4 - x_r5)"""
        idxs = [i for i in range(len(population)) if i != target_idx]
        r1, r2, r3, r4, r5 = rng.choice(idxs, 5, replace=False)
        return (population[r1] +
                F * (population[r2] - population[r3]) +
                F * (population[r4] - population[r5]))

    def _mutation_best_2(
        self,
        population: np.ndarray,
        target_idx: int,
        best: np.ndarray,
        F: float,
        rng: np.random.Generator,
        **kwargs
    ) -> np.ndarray:
        """DE/best/2 mutation: v = x_best + F * (x_r1 - x_r2) + F * (x_r3 - x_r4)"""
        idxs = [i for i in range(len(population)) if i != target_idx]
        r1, r2, r3, r4 = rng.choice(idxs, 4, replace=False)
        return (best +
                F * (population[r1] - population[r2]) +
                F * (population[r3] - population[r4]))

    # ==================== CROSSOVER TYPES ====================

    def _crossover_binomial(
        self,
        target: np.ndarray,
        mutant: np.ndarray,
        CR: float,
        rng: np.random.Generator,
        **kwargs
    ) -> np.ndarray:
        """Binomial (uniform) crossover."""
        trial = target.copy()
        j_rand = rng.integers(0, len(target))

        for j in range(len(target)):
            if rng.random() < CR or j == j_rand:
                trial[j] = mutant[j]

        return trial

    def _crossover_exponential(
        self,
        target: np.ndarray,
        mutant: np.ndarray,
        CR: float,
        rng: np.random.Generator,
        **kwargs
    ) -> np.ndarray:
        """Exponential crossover."""
        trial = target.copy()
        n = len(target)
        j = rng.integers(0, n)
        L = 0

        while True:
            trial[j] = mutant[j]
            j = (j + 1) % n
            L += 1
            if rng.random() >= CR or L >= n:
                break

        return trial

    # ==================== SELECTION ====================

    def _greedy_selection(
        self,
        target: np.ndarray,
        target_fitness: float,
        trial: np.ndarray,
        trial_fitness: float,
        **kwargs
    ) -> Tuple[np.ndarray, float]:
        """Greedy selection: keep better solution."""
        if trial_fitness <= target_fitness:
            return trial, trial_fitness
        return target, target_fitness


class AdaptiveDE(InstrumentedDE):
    """
    Self-adaptive DE with parameter adaptation.

    Implements jDE-style adaptation where F and CR are self-adapted.
    """

    def __init__(
        self,
        dim: int,
        bounds: Tuple[np.ndarray, np.ndarray],
        pop_size: int = 50,
        max_generations: int = 100,
        F_init: float = 0.5,
        CR_init: float = 0.9,
        tau_F: float = 0.1,
        tau_CR: float = 0.1,
        mutation_strategy: str = "rand/1",
        crossover_type: str = "binomial",
        instrumentation_level: InstrumentationLevel = InstrumentationLevel.STANDARD,
        seed: Optional[int] = None
    ):
        super().__init__(
            dim=dim,
            bounds=bounds,
            pop_size=pop_size,
            max_generations=max_generations,
            F=F_init,
            CR=CR_init,
            mutation_strategy=mutation_strategy,
            crossover_type=crossover_type,
            instrumentation_level=instrumentation_level,
            seed=seed
        )

        self.tau_F = tau_F
        self.tau_CR = tau_CR
        self.F_values = np.full(pop_size, F_init)
        self.CR_values = np.full(pop_size, CR_init)

        self.config.update({
            "adaptive": True,
            "tau_F": tau_F,
            "tau_CR": tau_CR
        })

    def _generation_step(self) -> Tuple[np.ndarray, np.ndarray]:
        """Perform one generation with parameter adaptation."""

        new_population = np.empty_like(self.population)
        new_fitness = np.empty_like(self.fitness)
        new_F = np.empty(self.pop_size)
        new_CR = np.empty(self.pop_size)

        # Get operators
        mutation_op = self.get_operator(f"mutation_{self.mutation_strategy}")
        crossover_op = self.get_operator(f"crossover_{self.crossover_type}")

        best_idx = np.argmin(self.fitness)
        best_individual = self.population[best_idx]

        for i in range(self.pop_size):
            # Adapt F
            if self.rng.random() < self.tau_F:
                F_i = 0.1 + 0.9 * self.rng.random()
            else:
                F_i = self.F_values[i]

            # Adapt CR
            if self.rng.random() < self.tau_CR:
                CR_i = self.rng.random()
            else:
                CR_i = self.CR_values[i]

            # Log parameter adaptation
            if F_i != self.F_values[i] or CR_i != self.CR_values[i]:
                from ..core.trace_logger import EventType
                self.logger._log_event(
                    EventType.PARAMETER_ADAPTATION,
                    generation=self.generation,
                    data={
                        "individual": i,
                        "old_F": self.F_values[i],
                        "new_F": F_i,
                        "old_CR": self.CR_values[i],
                        "new_CR": CR_i
                    }
                )

            # Apply mutation with adapted F
            mutant = mutation_op.apply(
                population=self.population,
                target_idx=i,
                best=best_individual,
                F=F_i,
                rng=self.rng
            )
            mutant = self.clip_to_bounds(mutant)

            # Apply crossover with adapted CR
            trial = crossover_op.apply(
                target=self.population[i],
                mutant=mutant,
                CR=CR_i,
                rng=self.rng
            )

            # Evaluate and select
            trial_fitness = self._current_objective(trial)
            self.evaluations += 1

            if trial_fitness <= self.fitness[i]:
                new_population[i] = trial
                new_fitness[i] = trial_fitness
                new_F[i] = F_i
                new_CR[i] = CR_i
                success = True
            else:
                new_population[i] = self.population[i]
                new_fitness[i] = self.fitness[i]
                new_F[i] = self.F_values[i]
                new_CR[i] = self.CR_values[i]
                success = False

            self.logger.log_operator_application(
                operator_name=mutation_op.name,
                parent_indices=[i],
                offspring_fitness=[trial_fitness],
                success=success,
                details={"F": F_i, "CR": CR_i}
            )

            # Log crossover success too
            self.logger.log_operator_application(
                operator_name=crossover_op.name,
                parent_indices=[i],
                offspring_fitness=[trial_fitness],
                success=success,
                details={"CR": CR_i}
            )

        self.F_values = new_F
        self.CR_values = new_CR

        return new_population, new_fitness
