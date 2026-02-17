"""
Instrumentation Module: Base classes for creating explainable metaheuristics.

Provides the infrastructure to wrap metaheuristic algorithms and capture
their decision-making process.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from .trace_logger import TraceLogger, InstrumentationLevel


@dataclass
class Operator:
    """
    Represents a metaheuristic operator (mutation, crossover, selection, etc.).

    Attributes:
        name: Unique identifier for the operator
        operator_type: Category (mutation, crossover, selection, etc.)
        function: The actual operator function
        parameters: Operator-specific parameters
        description: Human-readable description
    """
    name: str
    operator_type: str
    function: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def apply(self, *args, **kwargs) -> Any:
        """Apply the operator with current parameters."""
        return self.function(*args, **self.parameters, **kwargs)

    def __repr__(self) -> str:
        return f"Operator({self.name}, type={self.operator_type})"


class NeutralOperator(Operator):
    """
    A neutral operator that performs no transformation.

    Used in ablation studies to measure the contribution of specific operators.
    """

    def __init__(self, operator_type: str = "neutral"):
        super().__init__(
            name=f"neutral_{operator_type}",
            operator_type=operator_type,
            function=lambda x, **kwargs: x,
            description="Neutral operator (identity transformation)"
        )

    def apply(self, population: np.ndarray = None, **kwargs) -> np.ndarray:
        """Return appropriate neutral output based on operator type."""
        # For mutation: return target individual unchanged
        if 'target_idx' in kwargs and population is not None:
            return population[kwargs['target_idx']].copy()

        # For crossover: return target unchanged (no recombination)
        if 'target' in kwargs:
            return kwargs['target'].copy()

        # For selection: return original individual
        if population is not None:
            if population.ndim == 1:
                return population.copy()
            return population.copy()

        return None


class NeutralMutationOperator(Operator):
    """Neutral mutation that returns target unchanged (no perturbation)."""

    def __init__(self):
        super().__init__(
            name="neutral_mutation",
            operator_type="mutation",
            function=self._neutral_mutation,
            description="Neutral mutation (returns target unchanged)"
        )

    def _neutral_mutation(self, population=None, target_idx=None, individual=None, **kwargs):
        """Return the individual unchanged."""
        if individual is not None:
            return individual.copy()
        if population is not None and target_idx is not None:
            return population[target_idx].copy()
        return None

    def apply(self, population: np.ndarray = None, target_idx: int = None,
              individual: np.ndarray = None, **kwargs) -> np.ndarray:
        """Flexible neutral mutation supporting both DE and GA interfaces."""
        # GA-style: individual parameter
        if individual is not None:
            return individual.copy()

        # DE-style: population + target_idx
        if population is not None and target_idx is not None:
            return population[target_idx].copy()

        return None


class NeutralCrossoverOperator(Operator):
    """Neutral crossover that returns parents unchanged (no recombination)."""

    def __init__(self):
        super().__init__(
            name="neutral_crossover",
            operator_type="crossover",
            function=self._neutral_crossover,
            description="Neutral crossover (returns parents unchanged)"
        )

    def _neutral_crossover(self, target=None, mutant=None, parent1=None, parent2=None, **kwargs):
        """Return parents/target unchanged - no mixing."""
        if parent1 is not None:
            return parent1.copy(), parent2.copy() if parent2 is not None else parent1.copy()
        if target is not None:
            return target.copy()
        return None

    def apply(self, target: np.ndarray = None, mutant: np.ndarray = None,
              parent1: np.ndarray = None, parent2: np.ndarray = None, **kwargs):
        """Flexible neutral crossover supporting both DE and GA interfaces."""
        # GA-style: return parents unchanged (no recombination)
        if parent1 is not None:
            child1 = parent1.copy()
            child2 = parent2.copy() if parent2 is not None else parent1.copy()
            return child1, child2

        # DE-style: return target unchanged
        if target is not None:
            return target.copy()

        return None


class NeutralSelectionOperator(Operator):
    """Neutral selection that returns random/uniform selection (no fitness pressure)."""

    def __init__(self):
        super().__init__(
            name="neutral_selection",
            operator_type="selection",
            function=self._neutral_selection,
            description="Neutral selection (random selection, no fitness pressure)"
        )
        self._rng = np.random.default_rng()

    def _neutral_selection(self, original, candidate, original_fitness, candidate_fitness, **kwargs):
        """Always return original - never accept new solutions."""
        return original.copy(), original_fitness

    def apply(self, population: np.ndarray = None, fitness: np.ndarray = None,
              n_parents: int = None, original: np.ndarray = None, candidate: np.ndarray = None,
              original_fitness: float = None, candidate_fitness: float = None, **kwargs):
        """Flexible neutral selection supporting multiple interfaces."""

        # GA-style tournament selection: returns random INDEX (no fitness pressure)
        # When called with just fitness (no population), return random index
        if fitness is not None and population is None and original is None:
            return int(self._rng.integers(0, len(fitness)))

        # GA-style selection: select n_parents from population randomly
        if population is not None and fitness is not None and n_parents is not None:
            indices = self._rng.choice(len(population), size=n_parents, replace=True)
            return population[indices].copy()

        # DE-style selection: always keep original
        if original is not None and candidate is not None:
            return original.copy(), original_fitness if original_fitness is not None else 0.0

        # Default: return random index if fitness available
        if fitness is not None:
            return int(self._rng.integers(0, len(fitness)))

        # Default: return first input unchanged
        if population is not None:
            return population.copy()

        return 0


class NeutralVelocityOperator(Operator):
    """Neutral velocity update that returns zero velocity (no movement)."""

    def __init__(self):
        super().__init__(
            name="neutral_velocity",
            operator_type="velocity",
            function=self._neutral_velocity,
            description="Neutral velocity (returns zero velocity)"
        )

    def _neutral_velocity(self, velocity, **kwargs):
        """Return zero velocity - particles don't move."""
        return np.zeros_like(velocity)

    def apply(self, velocity: np.ndarray = None, **kwargs) -> np.ndarray:
        if velocity is not None:
            return np.zeros_like(velocity)
        return None


class NeutralPositionOperator(Operator):
    """Neutral position update that keeps position unchanged."""

    def __init__(self):
        super().__init__(
            name="neutral_position",
            operator_type="position",
            function=self._neutral_position,
            description="Neutral position (keeps position unchanged)"
        )

    def _neutral_position(self, position, velocity, **kwargs):
        """Return position unchanged - ignore velocity."""
        return position.copy()

    def apply(self, position: np.ndarray = None, velocity: np.ndarray = None, **kwargs) -> np.ndarray:
        if position is not None:
            return position.copy()
        return None


class NeutralTopologyOperator(Operator):
    """Neutral topology that returns particle's own position (no social influence)."""

    def __init__(self):
        super().__init__(
            name="neutral_topology",
            operator_type="topology",
            function=self._neutral_topology,
            description="Neutral topology (no social influence)"
        )

    def _neutral_topology(self, particle_idx, positions, **kwargs):
        """Return particle's own position - no social learning."""
        return positions[particle_idx].copy()

    def apply(self, particle_idx: int = 0, positions: np.ndarray = None,
              personal_best: np.ndarray = None, **kwargs) -> np.ndarray:
        # Return personal best if available, otherwise current position
        if personal_best is not None:
            if personal_best.ndim == 2:
                return personal_best[particle_idx].copy()
            return personal_best.copy()
        if positions is not None:
            return positions[particle_idx].copy()
        return None


def get_neutral_operator(operator_type: str) -> Operator:
    """Factory function to get appropriate neutral operator by type."""
    neutral_operators = {
        'mutation': NeutralMutationOperator,
        'crossover': NeutralCrossoverOperator,
        'selection': NeutralSelectionOperator,
        'velocity': NeutralVelocityOperator,
        'position': NeutralPositionOperator,
        'topology': NeutralTopologyOperator,
    }

    if operator_type in neutral_operators:
        return neutral_operators[operator_type]()

    return NeutralOperator(operator_type)


@dataclass
class OperatorResult:
    """Result of applying an operator."""
    offspring: np.ndarray
    offspring_fitness: np.ndarray
    success_mask: np.ndarray  # Boolean mask indicating successful applications
    metadata: Dict[str, Any] = field(default_factory=dict)


class InstrumentedAlgorithm(ABC):
    """
    Abstract base class for instrumented metaheuristics.

    Subclasses implement specific algorithms while this class handles
    instrumentation and explanation infrastructure.
    """

    def __init__(
        self,
        dim: int,
        bounds: Tuple[np.ndarray, np.ndarray],
        pop_size: int = 50,
        max_generations: int = 100,
        instrumentation_level: InstrumentationLevel = InstrumentationLevel.STANDARD,
        seed: Optional[int] = None
    ):
        """
        Initialize the instrumented algorithm.

        Args:
            dim: Problem dimensionality
            bounds: Tuple of (lower_bounds, upper_bounds) arrays
            pop_size: Population size
            max_generations: Maximum number of generations
            instrumentation_level: Level of detail for logging
            seed: Random seed for reproducibility
        """
        self.dim = dim
        self.bounds = bounds
        self.lower_bounds = np.array(bounds[0])
        self.upper_bounds = np.array(bounds[1])
        self.pop_size = pop_size
        self.max_generations = max_generations

        # Random state
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Operators registry
        self.operators: Dict[str, Operator] = {}

        # Logging
        self.logger = TraceLogger(
            level=instrumentation_level,
            checkpoint_interval=max(1, max_generations // 10)
        )

        # State
        self.population: Optional[np.ndarray] = None
        self.fitness: Optional[np.ndarray] = None
        self.generation: int = 0
        self.evaluations: int = 0

        # Best solution tracking
        self.best_solution: Optional[np.ndarray] = None
        self.best_fitness: float = float('inf')

        # Configuration
        self.config: Dict[str, Any] = {}

    @abstractmethod
    def _initialize_operators(self):
        """Initialize algorithm-specific operators. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _generation_step(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform one generation of the algorithm.

        Returns:
            Tuple of (new_population, new_fitness)
        """
        pass

    def register_operator(self, operator: Operator):
        """Register an operator for use and tracking."""
        self.operators[operator.name] = operator

    def get_operator(self, name: str) -> Optional[Operator]:
        """Get a registered operator by name."""
        return self.operators.get(name)

    def replace_operator(self, name: str, new_operator: Operator):
        """Replace an operator (useful for ablation studies)."""
        if name in self.operators:
            self.operators[name] = new_operator

    def disable_operator(self, name: str):
        """Disable an operator by replacing with appropriate neutral version."""
        if name in self.operators:
            op_type = self.operators[name].operator_type
            self.operators[name] = get_neutral_operator(op_type)

    def get_operators(self) -> List[str]:
        """Get list of operator names."""
        return list(self.operators.keys())

    def initialize_population(self) -> np.ndarray:
        """Initialize population within bounds."""
        population = self.rng.uniform(
            self.lower_bounds,
            self.upper_bounds,
            size=(self.pop_size, self.dim)
        )
        return population

    def evaluate(self, solutions: np.ndarray, objective_func: Callable) -> np.ndarray:
        """Evaluate solutions and track evaluation count."""
        if solutions.ndim == 1:
            solutions = solutions.reshape(1, -1)

        fitness = np.array([objective_func(sol) for sol in solutions])
        self.evaluations += len(fitness)

        return fitness

    def clip_to_bounds(self, solutions: np.ndarray) -> np.ndarray:
        """Clip solutions to stay within bounds."""
        return np.clip(solutions, self.lower_bounds, self.upper_bounds)

    def run(
        self,
        objective_func: Callable,
        problem_name: str = "unknown",
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Execute the algorithm.

        Args:
            objective_func: Function to minimize
            problem_name: Name for logging
            verbose: Whether to print progress

        Returns:
            Dictionary with results and trace
        """
        # Setup
        self._initialize_operators()
        self.logger.start_run(
            algorithm_name=self.__class__.__name__,
            problem_name=problem_name,
            dimension=self.dim,
            config=self.config
        )

        # Initialize population
        self.population = self.initialize_population()
        self.fitness = self.evaluate(self.population, objective_func)

        # Track best
        best_idx = np.argmin(self.fitness)
        self.best_solution = self.population[best_idx].copy()
        self.best_fitness = self.fitness[best_idx]

        # Log initial state
        self.logger.log_generation_end(self.population, self.fitness)

        # Main loop
        for self.generation in range(1, self.max_generations + 1):
            # Store objective function for use in generation step
            self._current_objective = objective_func

            # Perform generation
            self.population, self.fitness = self._generation_step()

            # Update best
            best_idx = np.argmin(self.fitness)
            if self.fitness[best_idx] < self.best_fitness:
                self.best_solution = self.population[best_idx].copy()
                self.best_fitness = self.fitness[best_idx]

            # Log generation
            self.logger.log_generation_end(
                self.population,
                self.fitness,
                additional_metrics={"evaluations": self.evaluations}
            )

            if verbose and self.generation % 10 == 0:
                print(f"Gen {self.generation}: Best = {self.best_fitness:.6e}")

        # Finalize
        self.logger.end_run(self.best_solution, self.best_fitness)

        if verbose:
            print(self.logger.summary())

        return {
            "best_solution": self.best_solution,
            "best_fitness": self.best_fitness,
            "evaluations": self.evaluations,
            "generations": self.generation,
            "trace": self.logger
        }

    def get_trace(self) -> TraceLogger:
        """Get the trace logger."""
        return self.logger

    def get_state(self) -> Dict[str, Any]:
        """Get current algorithm state for checkpointing."""
        return {
            "population": self.population.copy() if self.population is not None else None,
            "fitness": self.fitness.copy() if self.fitness is not None else None,
            "generation": self.generation,
            "evaluations": self.evaluations,
            "best_solution": self.best_solution.copy() if self.best_solution is not None else None,
            "best_fitness": self.best_fitness,
            "rng_state": self.rng.bit_generator.state
        }

    def set_state(self, state: Dict[str, Any]):
        """Restore algorithm state from checkpoint."""
        self.population = state["population"]
        self.fitness = state["fitness"]
        self.generation = state["generation"]
        self.evaluations = state["evaluations"]
        self.best_solution = state["best_solution"]
        self.best_fitness = state["best_fitness"]
        self.rng.bit_generator.state = state["rng_state"]

    def copy(self) -> "InstrumentedAlgorithm":
        """Create a copy of the algorithm with same configuration."""
        new_alg = self.__class__(
            dim=self.dim,
            bounds=self.bounds,
            pop_size=self.pop_size,
            max_generations=self.max_generations,
            instrumentation_level=self.logger.level,
            seed=None  # New random state
        )
        new_alg.config = self.config.copy()
        return new_alg
