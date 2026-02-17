"""
Instrumented Particle Swarm Optimization (PSO)

A fully instrumented implementation of PSO with detailed logging
of velocity updates, topology effects, and swarm dynamics.
"""

from typing import Callable, Dict, List, Optional, Tuple, Any
import numpy as np

from ..core.instrumentation import InstrumentedAlgorithm, Operator
from ..core.trace_logger import InstrumentationLevel, EventType


class InstrumentedPSO(InstrumentedAlgorithm):
    """
    Instrumented Particle Swarm Optimization algorithm.

    Supports multiple topologies and velocity update strategies
    with full logging of particle dynamics and decisions.
    """

    # Available topologies
    TOPOLOGIES = ["gbest", "lbest", "ring", "star", "random"]

    # Available velocity update strategies
    VELOCITY_STRATEGIES = ["standard", "constriction", "inertia_linear", "inertia_nonlinear"]

    def __init__(
        self,
        dim: int,
        bounds: Tuple[np.ndarray, np.ndarray],
        pop_size: int = 50,
        max_generations: int = 100,
        topology: str = "gbest",
        velocity_strategy: str = "constriction",
        w: float = 0.729,           # Inertia weight
        c1: float = 1.49445,        # Cognitive coefficient
        c2: float = 1.49445,        # Social coefficient
        w_min: float = 0.4,         # Min inertia (for adaptive)
        w_max: float = 0.9,         # Max inertia (for adaptive)
        v_max_ratio: float = 0.5,   # Max velocity as ratio of range
        neighborhood_size: int = 3,  # For lbest topology
        instrumentation_level: InstrumentationLevel = InstrumentationLevel.STANDARD,
        seed: Optional[int] = None
    ):
        """
        Initialize Instrumented PSO.

        Args:
            dim: Problem dimensionality
            bounds: Tuple of (lower_bounds, upper_bounds)
            pop_size: Number of particles (swarm size)
            max_generations: Maximum iterations
            topology: One of TOPOLOGIES
            velocity_strategy: One of VELOCITY_STRATEGIES
            w: Inertia weight
            c1: Cognitive (personal best) coefficient
            c2: Social (global/local best) coefficient
            w_min, w_max: Bounds for adaptive inertia
            v_max_ratio: Maximum velocity as fraction of search range
            neighborhood_size: Size of neighborhood for lbest
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

        # PSO-specific parameters
        self.topology = topology
        self.velocity_strategy = velocity_strategy
        self.w = w
        self.w_initial = w
        self.c1 = c1
        self.c2 = c2
        self.w_min = w_min
        self.w_max = w_max
        self.neighborhood_size = neighborhood_size

        # Calculate velocity bounds
        ranges = self.upper_bounds - self.lower_bounds
        self.v_max = v_max_ratio * ranges
        self.v_min = -self.v_max

        # Particle state
        self.velocities: Optional[np.ndarray] = None
        self.personal_best: Optional[np.ndarray] = None
        self.personal_best_fitness: Optional[np.ndarray] = None
        self.global_best: Optional[np.ndarray] = None
        self.global_best_fitness: float = float('inf')

        # Neighborhood structure (for lbest)
        self.neighbors: Optional[List[List[int]]] = None

        # Store configuration
        self.config = {
            "topology": topology,
            "velocity_strategy": velocity_strategy,
            "w": w,
            "c1": c1,
            "c2": c2,
            "w_min": w_min,
            "w_max": w_max,
            "v_max_ratio": v_max_ratio,
            "neighborhood_size": neighborhood_size,
            "pop_size": pop_size
        }

        # Validate parameters
        if topology not in self.TOPOLOGIES:
            raise ValueError(f"Unknown topology: {topology}")
        if velocity_strategy not in self.VELOCITY_STRATEGIES:
            raise ValueError(f"Unknown velocity strategy: {velocity_strategy}")

    def _initialize_operators(self):
        """Initialize PSO operators."""

        # Velocity update operator
        velocity_func = self._get_velocity_function(self.velocity_strategy)
        self.register_operator(Operator(
            name=f"velocity_{self.velocity_strategy}",
            operator_type="velocity_update",
            function=velocity_func,
            parameters={},
            description=f"PSO velocity update: {self.velocity_strategy}"
        ))

        # Position update operator
        self.register_operator(Operator(
            name="position_update",
            operator_type="position_update",
            function=self._position_update,
            parameters={},
            description="PSO position update"
        ))

        # Topology operator (determines social influence)
        topology_func = self._get_topology_function(self.topology)
        self.register_operator(Operator(
            name=f"topology_{self.topology}",
            operator_type="topology",
            function=topology_func,
            parameters={},
            description=f"PSO topology: {self.topology}"
        ))

    def _get_velocity_function(self, strategy: str) -> Callable:
        """Get velocity update function."""
        strategies = {
            "standard": self._velocity_standard,
            "constriction": self._velocity_constriction,
            "inertia_linear": self._velocity_inertia_linear,
            "inertia_nonlinear": self._velocity_inertia_nonlinear,
        }
        return strategies[strategy]

    def _get_topology_function(self, topology: str) -> Callable:
        """Get topology function."""
        topologies = {
            "gbest": self._topology_gbest,
            "lbest": self._topology_lbest,
            "ring": self._topology_ring,
            "star": self._topology_star,
            "random": self._topology_random,
        }
        return topologies[topology]

    def _initialize_swarm(self):
        """Initialize swarm-specific state."""
        # Initialize velocities
        ranges = self.upper_bounds - self.lower_bounds
        self.velocities = self.rng.uniform(
            -0.1 * ranges,
            0.1 * ranges,
            size=(self.pop_size, self.dim)
        )

        # Initialize personal bests
        self.personal_best = self.population.copy()
        self.personal_best_fitness = self.fitness.copy()

        # Initialize global best
        best_idx = np.argmin(self.fitness)
        self.global_best = self.population[best_idx].copy()
        self.global_best_fitness = self.fitness[best_idx]

        # Initialize neighborhood structure
        self._initialize_topology()

    def _initialize_topology(self):
        """Initialize neighborhood structure based on topology."""
        if self.topology == "gbest" or self.topology == "star":
            # All particles connected to best
            self.neighbors = None
        elif self.topology == "ring":
            # Ring topology
            self.neighbors = []
            for i in range(self.pop_size):
                left = (i - 1) % self.pop_size
                right = (i + 1) % self.pop_size
                self.neighbors.append([left, i, right])
        elif self.topology == "lbest":
            # Local best with k neighbors
            self.neighbors = []
            k = self.neighborhood_size // 2
            for i in range(self.pop_size):
                neighborhood = [(i + j) % self.pop_size for j in range(-k, k + 1)]
                self.neighbors.append(neighborhood)
        elif self.topology == "random":
            # Random topology (regenerated each iteration)
            self.neighbors = None

    def run(
        self,
        objective_func: Callable,
        problem_name: str = "unknown",
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Execute PSO with swarm initialization."""
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

        # Initialize swarm-specific state
        self._initialize_swarm()

        # Track best
        self.best_solution = self.global_best.copy()
        self.best_fitness = self.global_best_fitness

        # Log initial state
        self.logger.log_generation_end(
            self.population, self.fitness,
            additional_metrics={
                "global_best_fitness": self.global_best_fitness,
                "avg_velocity": np.mean(np.abs(self.velocities))
            }
        )

        # Main loop
        for self.generation in range(1, self.max_generations + 1):
            self._current_objective = objective_func

            # Perform iteration
            self.population, self.fitness = self._generation_step()

            # Update best
            if self.global_best_fitness < self.best_fitness:
                self.best_solution = self.global_best.copy()
                self.best_fitness = self.global_best_fitness

            # Log generation
            self.logger.log_generation_end(
                self.population, self.fitness,
                additional_metrics={
                    "evaluations": self.evaluations,
                    "global_best_fitness": self.global_best_fitness,
                    "avg_velocity": np.mean(np.abs(self.velocities)),
                    "inertia_weight": self.w
                }
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

    def _generation_step(self) -> Tuple[np.ndarray, np.ndarray]:
        """Perform one iteration of PSO."""

        velocity_op = self.get_operator(f"velocity_{self.velocity_strategy}")
        position_op = self.get_operator("position_update")
        topology_op = self.get_operator(f"topology_{self.topology}")

        # Update inertia weight if using adaptive strategy
        if self.velocity_strategy in ["inertia_linear", "inertia_nonlinear"]:
            self._update_inertia()

        new_population = np.empty_like(self.population)
        new_fitness = np.empty(self.pop_size)

        for i in range(self.pop_size):
            old_fitness = self.personal_best_fitness[i]

            # Get social best based on topology
            self.logger.log_operator_selection(
                operator_name=topology_op.name,
                parameters={"particle": i},
                rationale=f"Determining social influence for particle {i}"
            )

            social_best = topology_op.apply(
                particle_idx=i,
                positions=self.population,
                fitness=self.fitness,
                personal_best=self.personal_best,
                personal_best_fitness=self.personal_best_fitness,
                global_best=self.global_best,
                neighbors=self.neighbors,
                rng=self.rng
            )

            # Update velocity
            self.logger.log_operator_selection(
                operator_name=velocity_op.name,
                parameters={"w": self.w, "c1": self.c1, "c2": self.c2},
                rationale=f"Updating velocity for particle {i}"
            )

            self.velocities[i] = velocity_op.apply(
                velocity=self.velocities[i],
                position=self.population[i],
                personal_best=self.personal_best[i],
                social_best=social_best,
                w=self.w,
                c1=self.c1,
                c2=self.c2,
                v_max=self.v_max,
                v_min=self.v_min,
                generation=self.generation,
                max_generations=self.max_generations,
                rng=self.rng
            )

            # Update position (deterministic: x = x + v, not part of the
            # cooperative game but kept as an operator for ablation flexibility)
            new_population[i] = position_op.apply(
                position=self.population[i],
                velocity=self.velocities[i],
                lower=self.lower_bounds,
                upper=self.upper_bounds
            )

            # Evaluate
            new_fitness[i] = self._current_objective(new_population[i])
            self.evaluations += 1

            # Determine success
            success = new_fitness[i] < old_fitness
            improvement = old_fitness - new_fitness[i] if success else 0.0

            # Update personal best
            if new_fitness[i] < self.personal_best_fitness[i]:
                self.personal_best[i] = new_population[i].copy()
                self.personal_best_fitness[i] = new_fitness[i]

                # Update global best
                if new_fitness[i] < self.global_best_fitness:
                    self.global_best = new_population[i].copy()
                    self.global_best_fitness = new_fitness[i]

            # Log velocity operator (main search operator)
            self.logger.log_operator_application(
                operator_name=velocity_op.name,
                parent_indices=[i],
                offspring_fitness=[new_fitness[i]],
                success=success,
                details={
                    "improvement": improvement,
                    "velocity_magnitude": np.linalg.norm(self.velocities[i])
                }
            )

            # Log topology operator (social influence)
            self.logger.log_operator_application(
                operator_name=topology_op.name,
                parent_indices=[i],
                offspring_fitness=[new_fitness[i]],
                success=success,
                details={
                    "improvement": improvement,
                    "social_best_distance": np.linalg.norm(
                        social_best - self.population[i]
                    )
                }
            )

        return new_population, new_fitness

    def _update_inertia(self):
        """Update inertia weight based on strategy."""
        if self.velocity_strategy == "inertia_linear":
            # Linear decrease
            self.w = self.w_max - (self.w_max - self.w_min) * (self.generation / self.max_generations)
        elif self.velocity_strategy == "inertia_nonlinear":
            # Nonlinear decrease
            self.w = self.w_min + (self.w_max - self.w_min) * ((self.max_generations - self.generation) / self.max_generations) ** 2

    # ==================== VELOCITY UPDATE STRATEGIES ====================

    def _velocity_standard(
        self,
        velocity: np.ndarray,
        position: np.ndarray,
        personal_best: np.ndarray,
        social_best: np.ndarray,
        w: float,
        c1: float,
        c2: float,
        v_max: np.ndarray,
        v_min: np.ndarray,
        rng: np.random.Generator,
        **kwargs
    ) -> np.ndarray:
        """Standard PSO velocity update."""
        r1 = rng.random(len(velocity))
        r2 = rng.random(len(velocity))

        cognitive = c1 * r1 * (personal_best - position)
        social = c2 * r2 * (social_best - position)

        new_velocity = w * velocity + cognitive + social

        # Clip velocity
        return np.clip(new_velocity, v_min, v_max)

    def _velocity_constriction(
        self,
        velocity: np.ndarray,
        position: np.ndarray,
        personal_best: np.ndarray,
        social_best: np.ndarray,
        c1: float,
        c2: float,
        v_max: np.ndarray,
        v_min: np.ndarray,
        rng: np.random.Generator,
        **kwargs
    ) -> np.ndarray:
        """Constriction coefficient velocity update (Clerc & Kennedy)."""
        phi = c1 + c2
        if phi > 4:
            chi = 2.0 / abs(2.0 - phi - np.sqrt(phi * phi - 4.0 * phi))
        else:
            chi = 1.0

        r1 = rng.random(len(velocity))
        r2 = rng.random(len(velocity))

        cognitive = c1 * r1 * (personal_best - position)
        social = c2 * r2 * (social_best - position)

        new_velocity = chi * (velocity + cognitive + social)

        return np.clip(new_velocity, v_min, v_max)

    def _velocity_inertia_linear(
        self,
        velocity: np.ndarray,
        position: np.ndarray,
        personal_best: np.ndarray,
        social_best: np.ndarray,
        w: float,
        c1: float,
        c2: float,
        v_max: np.ndarray,
        v_min: np.ndarray,
        rng: np.random.Generator,
        **kwargs
    ) -> np.ndarray:
        """Linear inertia weight decrease."""
        return self._velocity_standard(
            velocity, position, personal_best, social_best,
            w, c1, c2, v_max, v_min, rng
        )

    def _velocity_inertia_nonlinear(
        self,
        velocity: np.ndarray,
        position: np.ndarray,
        personal_best: np.ndarray,
        social_best: np.ndarray,
        w: float,
        c1: float,
        c2: float,
        v_max: np.ndarray,
        v_min: np.ndarray,
        rng: np.random.Generator,
        **kwargs
    ) -> np.ndarray:
        """Nonlinear inertia weight decrease."""
        return self._velocity_standard(
            velocity, position, personal_best, social_best,
            w, c1, c2, v_max, v_min, rng
        )

    # ==================== POSITION UPDATE ====================

    def _position_update(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """Update position and handle boundary constraints."""
        new_position = position + velocity

        # Reflecting boundary handling
        for i in range(len(position)):
            if new_position[i] < lower[i]:
                new_position[i] = lower[i]
                # Optionally reverse velocity: self.velocities[particle_idx][i] *= -0.5
            elif new_position[i] > upper[i]:
                new_position[i] = upper[i]
                # Optionally reverse velocity

        return new_position

    # ==================== TOPOLOGY FUNCTIONS ====================

    def _topology_gbest(
        self,
        particle_idx: int,
        global_best: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """Global best topology - all particles connected to global best."""
        return global_best

    def _topology_lbest(
        self,
        particle_idx: int,
        personal_best: np.ndarray,
        personal_best_fitness: np.ndarray,
        neighbors: List[List[int]],
        **kwargs
    ) -> np.ndarray:
        """Local best topology - particles connected to neighborhood best."""
        neighborhood = neighbors[particle_idx]
        best_neighbor_idx = neighborhood[np.argmin(personal_best_fitness[neighborhood])]
        return personal_best[best_neighbor_idx]

    def _topology_ring(
        self,
        particle_idx: int,
        personal_best: np.ndarray,
        personal_best_fitness: np.ndarray,
        neighbors: List[List[int]],
        **kwargs
    ) -> np.ndarray:
        """Ring topology - each particle connected to two neighbors."""
        return self._topology_lbest(
            particle_idx, personal_best, personal_best_fitness, neighbors
        )

    def _topology_star(
        self,
        particle_idx: int,
        global_best: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """Star topology - all connected through central particle."""
        return global_best

    def _topology_random(
        self,
        particle_idx: int,
        personal_best: np.ndarray,
        personal_best_fitness: np.ndarray,
        rng: np.random.Generator,
        **kwargs
    ) -> np.ndarray:
        """Random topology - random neighbor each iteration."""
        random_neighbor = rng.integers(0, len(personal_best))
        return personal_best[random_neighbor]


class CPSO(InstrumentedPSO):
    """
    Comprehensive Learning PSO (CLPSO).

    Each dimension learns from different exemplars.
    """

    def __init__(
        self,
        dim: int,
        bounds: Tuple[np.ndarray, np.ndarray],
        pop_size: int = 50,
        max_generations: int = 100,
        learning_prob: float = 0.05,
        refresh_gap: int = 7,
        **kwargs
    ):
        # Force gbest topology for CLPSO
        kwargs['topology'] = 'gbest'
        super().__init__(dim, bounds, pop_size, max_generations, **kwargs)

        self.learning_prob = learning_prob
        self.refresh_gap = refresh_gap
        self.exemplars: Optional[np.ndarray] = None
        self.stagnation_counters: Optional[np.ndarray] = None

        self.config["variant"] = "CLPSO"
        self.config["learning_prob"] = learning_prob
        self.config["refresh_gap"] = refresh_gap

    def _initialize_swarm(self):
        """Initialize CLPSO-specific state."""
        super()._initialize_swarm()

        # Initialize exemplar matrix (which particle to learn from per dimension)
        self.exemplars = np.zeros((self.pop_size, self.dim), dtype=int)
        self.stagnation_counters = np.zeros(self.pop_size, dtype=int)

        # Initialize exemplars for each particle
        for i in range(self.pop_size):
            self._update_exemplars(i)

    def _update_exemplars(self, particle_idx: int):
        """Update learning exemplars for a particle."""
        pc = self.learning_prob

        for d in range(self.dim):
            if self.rng.random() < pc:
                # Learn from tournament winner
                candidates = self.rng.choice(self.pop_size, 2, replace=False)
                if self.personal_best_fitness[candidates[0]] < self.personal_best_fitness[candidates[1]]:
                    self.exemplars[particle_idx, d] = candidates[0]
                else:
                    self.exemplars[particle_idx, d] = candidates[1]
            else:
                # Learn from own personal best
                self.exemplars[particle_idx, d] = particle_idx

    def _generation_step(self) -> Tuple[np.ndarray, np.ndarray]:
        """CLPSO iteration."""
        new_population = np.empty_like(self.population)
        new_fitness = np.empty(self.pop_size)

        velocity_op = self.get_operator(f"velocity_{self.velocity_strategy}")

        for i in range(self.pop_size):
            # Check for stagnation and refresh exemplars
            if self.stagnation_counters[i] >= self.refresh_gap:
                self._update_exemplars(i)
                self.stagnation_counters[i] = 0

            # Build comprehensive learning exemplar
            exemplar = np.array([
                self.personal_best[self.exemplars[i, d], d]
                for d in range(self.dim)
            ])

            # Velocity update using comprehensive exemplar
            r1 = self.rng.random(self.dim)
            cognitive = self.c1 * r1 * (exemplar - self.population[i])
            self.velocities[i] = self.w * self.velocities[i] + cognitive
            self.velocities[i] = np.clip(self.velocities[i], self.v_min, self.v_max)

            # Position update
            new_population[i] = self.population[i] + self.velocities[i]
            new_population[i] = np.clip(new_population[i], self.lower_bounds, self.upper_bounds)

            # Evaluate
            new_fitness[i] = self._current_objective(new_population[i])
            self.evaluations += 1

            # Update personal best
            if new_fitness[i] < self.personal_best_fitness[i]:
                self.personal_best[i] = new_population[i].copy()
                self.personal_best_fitness[i] = new_fitness[i]
                self.stagnation_counters[i] = 0

                if new_fitness[i] < self.global_best_fitness:
                    self.global_best = new_population[i].copy()
                    self.global_best_fitness = new_fitness[i]

                self.logger.log_operator_application(
                    operator_name="clpso_update",
                    parent_indices=[i],
                    offspring_fitness=[new_fitness[i]],
                    success=True,
                    details={}
                )
            else:
                self.stagnation_counters[i] += 1

        return new_population, new_fitness
