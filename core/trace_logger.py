"""
TraceLogger: Records all decisions and events during metaheuristic execution.

This module provides fine-grained logging of the search process for
subsequent explanation generation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum, auto
import numpy as np
import json
import time
from datetime import datetime


class EventType(Enum):
    """Types of events that can occur during search."""
    INITIALIZATION = auto()
    OPERATOR_SELECTION = auto()
    OPERATOR_APPLICATION = auto()
    FITNESS_EVALUATION = auto()
    SELECTION = auto()
    GENERATION_END = auto()
    IMPROVEMENT = auto()
    STAGNATION = auto()
    DIVERSITY_CHANGE = auto()
    PARAMETER_ADAPTATION = auto()
    CHECKPOINT = auto()
    TERMINATION = auto()


@dataclass
class SearchEvent:
    """Represents a single event during the search process."""

    event_type: EventType
    generation: int
    timestamp: float
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert event to dictionary for serialization."""
        return {
            "event_type": self.event_type.name,
            "generation": self.generation,
            "timestamp": self.timestamp,
            "data": self._serialize_data(self.data)
        }

    @staticmethod
    def _serialize_data(data: Dict) -> Dict:
        """Serialize numpy arrays and other non-JSON types."""
        result = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                result[key] = value.tolist()
            elif isinstance(value, (np.float32, np.float64)):
                result[key] = float(value)
            elif isinstance(value, (np.int32, np.int64)):
                result[key] = int(value)
            else:
                result[key] = value
        return result


@dataclass
class GenerationSnapshot:
    """Complete snapshot of the population state at a generation."""

    generation: int
    population: np.ndarray  # Shape: (pop_size, dim)
    fitness: np.ndarray     # Shape: (pop_size,)
    best_solution: np.ndarray
    best_fitness: float
    worst_fitness: float
    mean_fitness: float
    std_fitness: float
    diversity: float        # Population diversity metric

    # Operator statistics for this generation
    operators_used: Dict[str, int] = field(default_factory=dict)
    operators_success: Dict[str, int] = field(default_factory=dict)
    operators_improvement: Dict[str, float] = field(default_factory=dict)

    # Additional metrics
    improvement_count: int = 0
    stagnation_counter: int = 0

    def to_dict(self) -> Dict:
        """Convert snapshot to dictionary."""
        return {
            "generation": self.generation,
            "best_fitness": float(self.best_fitness),
            "worst_fitness": float(self.worst_fitness),
            "mean_fitness": float(self.mean_fitness),
            "std_fitness": float(self.std_fitness),
            "diversity": float(self.diversity),
            "operators_used": self.operators_used,
            "operators_success": self.operators_success,
            "operators_improvement": self.operators_improvement,
            "improvement_count": self.improvement_count,
            "stagnation_counter": self.stagnation_counter
        }


class InstrumentationLevel(Enum):
    """Level of detail for instrumentation."""
    NONE = 0      # No logging
    LIGHT = 1     # Only best fitness per generation
    STANDARD = 2  # + Operators used, basic stats
    FULL = 3      # + Complete population state
    DEBUG = 4     # + All intermediate solutions


class TraceLogger:
    """
    Main logger for recording metaheuristic search process.

    Captures fine-grained information about decisions and outcomes
    for subsequent explanation generation.
    """

    def __init__(
        self,
        level: InstrumentationLevel = InstrumentationLevel.STANDARD,
        checkpoint_interval: int = 50,
        store_populations: bool = False
    ):
        """
        Initialize the trace logger.

        Args:
            level: Level of instrumentation detail
            checkpoint_interval: Generations between full checkpoints
            store_populations: Whether to store full population arrays
        """
        self.level = level
        self.checkpoint_interval = checkpoint_interval
        self.store_populations = store_populations

        # Storage
        self.events: List[SearchEvent] = []
        self.snapshots: List[GenerationSnapshot] = []
        self.checkpoints: Dict[int, Dict] = {}

        # Metadata
        self.algorithm_name: str = ""
        self.problem_name: str = ""
        self.dimension: int = 0
        self.start_time: float = 0
        self.end_time: float = 0

        # Running statistics
        self._current_generation: int = 0
        self._operator_usage: Dict[str, int] = {}
        self._operator_success: Dict[str, int] = {}
        self._operator_improvement: Dict[str, float] = {}
        self._improvements_this_gen: int = 0
        self._stagnation_counter: int = 0
        self._last_best_fitness: float = float('inf')

    def start_run(
        self,
        algorithm_name: str,
        problem_name: str,
        dimension: int,
        config: Dict[str, Any] = None
    ):
        """Start logging a new run."""
        self.algorithm_name = algorithm_name
        self.problem_name = problem_name
        self.dimension = dimension
        self.start_time = time.time()

        self._log_event(
            EventType.INITIALIZATION,
            generation=0,
            data={
                "algorithm": algorithm_name,
                "problem": problem_name,
                "dimension": dimension,
                "config": config or {}
            }
        )

    def end_run(self, final_solution: np.ndarray, final_fitness: float):
        """End logging for current run."""
        self.end_time = time.time()

        self._log_event(
            EventType.TERMINATION,
            generation=self._current_generation,
            data={
                "final_fitness": final_fitness,
                "total_time": self.end_time - self.start_time,
                "total_generations": self._current_generation,
                "total_events": len(self.events)
            }
        )

    def log_operator_selection(
        self,
        operator_name: str,
        parameters: Dict[str, Any],
        rationale: str = ""
    ):
        """Log when an operator is selected."""
        if self.level.value < InstrumentationLevel.STANDARD.value:
            return

        self._log_event(
            EventType.OPERATOR_SELECTION,
            generation=self._current_generation,
            data={
                "operator": operator_name,
                "parameters": parameters,
                "rationale": rationale
            }
        )

        # Track usage
        self._operator_usage[operator_name] = \
            self._operator_usage.get(operator_name, 0) + 1

    def log_operator_application(
        self,
        operator_name: str,
        parent_indices: List[int],
        offspring_fitness: List[float],
        success: bool,
        details: Dict[str, Any] = None
    ):
        """Log the result of applying an operator."""
        if self.level.value < InstrumentationLevel.STANDARD.value:
            return

        self._log_event(
            EventType.OPERATOR_APPLICATION,
            generation=self._current_generation,
            data={
                "operator": operator_name,
                "parent_indices": parent_indices,
                "offspring_fitness": offspring_fitness,
                "success": success,
                "details": details or {}
            }
        )

        if success:
            self._operator_success[operator_name] = \
                self._operator_success.get(operator_name, 0) + 1
            self._improvements_this_gen += 1

            # Track improvement magnitude
            improvement = 0.0
            if details:
                improvement = details.get("improvement", 0.0)
            self._operator_improvement[operator_name] = \
                self._operator_improvement.get(operator_name, 0.0) + improvement

    def log_generation_end(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        additional_metrics: Dict[str, Any] = None
    ):
        """Log end of generation with population statistics."""
        self._current_generation += 1

        # Calculate statistics
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_solution = population[best_idx].copy()

        # Check for improvement
        if best_fitness < self._last_best_fitness:
            improvement = self._last_best_fitness - best_fitness
            self._stagnation_counter = 0

            self._log_event(
                EventType.IMPROVEMENT,
                generation=self._current_generation,
                data={
                    "previous_best": self._last_best_fitness,
                    "new_best": best_fitness,
                    "improvement": improvement
                }
            )
            self._last_best_fitness = best_fitness
        else:
            self._stagnation_counter += 1
            if self._stagnation_counter % 10 == 0:  # Log every 10 stagnant gens
                self._log_event(
                    EventType.STAGNATION,
                    generation=self._current_generation,
                    data={"stagnation_count": self._stagnation_counter}
                )

        # Calculate diversity
        diversity = self._calculate_diversity(population)

        # Create snapshot
        snapshot = GenerationSnapshot(
            generation=self._current_generation,
            population=population.copy() if self.store_populations else np.array([]),
            fitness=fitness.copy(),
            best_solution=best_solution,
            best_fitness=best_fitness,
            worst_fitness=float(np.max(fitness)),
            mean_fitness=float(np.mean(fitness)),
            std_fitness=float(np.std(fitness)),
            diversity=diversity,
            operators_used=self._operator_usage.copy(),
            operators_success=self._operator_success.copy(),
            operators_improvement=self._operator_improvement.copy(),
            improvement_count=self._improvements_this_gen,
            stagnation_counter=self._stagnation_counter
        )

        self.snapshots.append(snapshot)

        # Log generation end event
        if self.level.value >= InstrumentationLevel.LIGHT.value:
            self._log_event(
                EventType.GENERATION_END,
                generation=self._current_generation,
                data={
                    "best_fitness": best_fitness,
                    "mean_fitness": float(np.mean(fitness)),
                    "diversity": diversity,
                    "improvements": self._improvements_this_gen,
                    **(additional_metrics or {})
                }
            )

        # Create checkpoint if needed
        if self._current_generation % self.checkpoint_interval == 0:
            self._create_checkpoint(population, fitness)

        # Reset per-generation counters
        self._improvements_this_gen = 0
        self._operator_usage.clear()
        self._operator_success.clear()
        self._operator_improvement.clear()

    def log_diversity_change(self, old_diversity: float, new_diversity: float):
        """Log significant changes in population diversity."""
        if self.level.value < InstrumentationLevel.STANDARD.value:
            return

        change_ratio = abs(new_diversity - old_diversity) / (old_diversity + 1e-10)
        if change_ratio > 0.1:  # Log if >10% change
            self._log_event(
                EventType.DIVERSITY_CHANGE,
                generation=self._current_generation,
                data={
                    "old_diversity": old_diversity,
                    "new_diversity": new_diversity,
                    "change_ratio": change_ratio
                }
            )

    def _log_event(self, event_type: EventType, generation: int, data: Dict):
        """Internal method to log an event."""
        event = SearchEvent(
            event_type=event_type,
            generation=generation,
            timestamp=time.time() - self.start_time,
            data=data
        )
        self.events.append(event)

    def _calculate_diversity(self, population: np.ndarray) -> float:
        """Calculate population diversity using average pairwise distance."""
        if len(population) < 2:
            return 0.0

        # Use subset for efficiency in large populations
        n = min(50, len(population))
        indices = np.random.choice(len(population), n, replace=False)
        subset = population[indices]

        # Calculate average pairwise Euclidean distance
        total_dist = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_dist += np.linalg.norm(subset[i] - subset[j])
                count += 1

        return total_dist / count if count > 0 else 0.0

    def _create_checkpoint(self, population: np.ndarray, fitness: np.ndarray):
        """Create a full checkpoint for potential replay."""
        self.checkpoints[self._current_generation] = {
            "generation": self._current_generation,
            "population": population.copy(),
            "fitness": fitness.copy(),
            "timestamp": time.time() - self.start_time,
            "operator_usage_cumulative": self._get_cumulative_operator_stats()
        }

        self._log_event(
            EventType.CHECKPOINT,
            generation=self._current_generation,
            data={"checkpoint_created": True}
        )

    def _get_cumulative_operator_stats(self) -> Dict[str, Dict[str, int]]:
        """Get cumulative operator statistics up to current generation."""
        cumulative = {}
        for snapshot in self.snapshots:
            for op, count in snapshot.operators_used.items():
                if op not in cumulative:
                    cumulative[op] = {"used": 0, "success": 0}
                cumulative[op]["used"] += count
            for op, count in snapshot.operators_success.items():
                if op in cumulative:
                    cumulative[op]["success"] += count
        return cumulative

    def get_checkpoint(self, generation: int) -> Optional[Dict]:
        """Get checkpoint at or before specified generation."""
        available = [g for g in self.checkpoints.keys() if g <= generation]
        if not available:
            return None
        closest = max(available)
        return self.checkpoints[closest]

    def get_operator_contributions(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate operator contribution statistics.

        Returns dict with usage counts, success rates, and improvement
        magnitudes per operator.
        """
        contributions = {}

        for snapshot in self.snapshots:
            for op, count in snapshot.operators_used.items():
                if op not in contributions:
                    contributions[op] = {
                        "total_used": 0,
                        "total_success": 0,
                        "cumulative_improvement": 0.0,
                    }
                contributions[op]["total_used"] += count

            for op, count in snapshot.operators_success.items():
                if op in contributions:
                    contributions[op]["total_success"] += count

            for op, improvement in snapshot.operators_improvement.items():
                if op in contributions:
                    contributions[op]["cumulative_improvement"] += improvement

        # Calculate derived metrics
        for op in contributions:
            used = contributions[op]["total_used"]
            success = contributions[op]["total_success"]
            cumulative = contributions[op]["cumulative_improvement"]
            contributions[op]["success_rate"] = success / used if used > 0 else 0
            contributions[op]["avg_improvement"] = (
                cumulative / success if success > 0 else 0.0
            )

        return contributions

    def get_fitness_history(self) -> Tuple[List[int], List[float], List[float]]:
        """Get fitness history as lists of generations, best, and mean fitness."""
        generations = [s.generation for s in self.snapshots]
        best_fitness = [s.best_fitness for s in self.snapshots]
        mean_fitness = [s.mean_fitness for s in self.snapshots]
        return generations, best_fitness, mean_fitness

    def get_diversity_history(self) -> Tuple[List[int], List[float]]:
        """Get diversity history."""
        generations = [s.generation for s in self.snapshots]
        diversity = [s.diversity for s in self.snapshots]
        return generations, diversity

    def get_critical_generations(self, top_k: int = 5) -> List[int]:
        """
        Identify generations with most significant events.

        Returns generations with largest improvements or other critical events.
        """
        improvements = []
        for event in self.events:
            if event.event_type == EventType.IMPROVEMENT:
                improvements.append((
                    event.generation,
                    event.data.get("improvement", 0)
                ))

        # Sort by improvement magnitude
        improvements.sort(key=lambda x: x[1], reverse=True)
        return [gen for gen, _ in improvements[:top_k]]

    def to_dict(self) -> Dict:
        """Convert entire trace to dictionary for serialization."""
        return {
            "metadata": {
                "algorithm": self.algorithm_name,
                "problem": self.problem_name,
                "dimension": self.dimension,
                "start_time": self.start_time,
                "end_time": self.end_time,
                "total_generations": self._current_generation,
                "instrumentation_level": self.level.name
            },
            "events": [e.to_dict() for e in self.events],
            "snapshots": [s.to_dict() for s in self.snapshots],
            "operator_contributions": self.get_operator_contributions()
        }

    def save(self, filepath: str):
        """Save trace to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def summary(self) -> str:
        """Generate a text summary of the search process."""
        if not self.snapshots:
            return "No data recorded."

        final = self.snapshots[-1]
        contributions = self.get_operator_contributions()
        critical_gens = self.get_critical_generations(3)

        lines = [
            "=" * 60,
            "XMH TRACE SUMMARY",
            "=" * 60,
            f"Algorithm: {self.algorithm_name}",
            f"Problem: {self.problem_name} (D={self.dimension})",
            f"Total generations: {self._current_generation}",
            f"Total events logged: {len(self.events)}",
            "",
            "FINAL RESULTS:",
            f"  Best fitness: {final.best_fitness:.6e}",
            f"  Mean fitness: {final.mean_fitness:.6e}",
            f"  Final diversity: {final.diversity:.4f}",
            "",
            "OPERATOR CONTRIBUTIONS:"
        ]

        for op, stats in contributions.items():
            lines.append(
                f"  {op}: {stats['total_used']} uses, "
                f"{stats['success_rate']*100:.1f}% success rate"
            )

        lines.extend([
            "",
            f"CRITICAL GENERATIONS: {critical_gens}",
            "=" * 60
        ])

        return "\n".join(lines)
