"""Instrumented SHADE (Success-History based Adaptive Differential Evolution).

Implements SHADE [Tanabe & Fukunaga, 2013] as a four-operator cooperative
game suitable for exact Shapley attribution:

    1. mutation           (current-to-pbest/1 with archive)
    2. crossover          (binomial)
    3. selection          (greedy with archive update)
    4. parameter_adaptation  (history-based F/CR memory)

The first three are the standard DE triplet; the fourth is the distinctive
SHADE component (success-history memory of (F, CR) pairs weighted by the
fitness improvement they produced). With n = 4 operators, the cooperative
game has 2^4 = 16 coalitions, the upper end of the n <= 4 regime in which
exact Shapley remains tractable.

Neutral counterparts (used when an operator is removed from a coalition):

    mutation:               trial vector = target (no perturbation)
    crossover:              trial vector = mutant (no recombination)
    selection:              random replacement with prob 0.5
    parameter_adaptation:   fixed F = 0.5, CR = 0.5 (no memory update)

This design makes SHADE's adaptation explicit as a player whose
contribution can be measured against the same axiomatic baseline as the
underlying DE operators.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from xmh.core.instrumentation import InstrumentedAlgorithm, Operator


class InstrumentedSHADE(InstrumentedAlgorithm):
    """SHADE with operator-level instrumentation."""

    def __init__(
        self,
        dim: int,
        bounds: Tuple[np.ndarray, np.ndarray],
        pop_size: int = 100,
        max_generations: int = 100,
        h_memory_size: int = 5,
        p_best_rate: float = 0.1,
        archive_size_factor: float = 1.0,
        seed: Optional[int] = None,
    ):
        super().__init__(dim=dim, bounds=bounds, pop_size=pop_size,
                         max_generations=max_generations, seed=seed)
        self.h_memory_size = h_memory_size
        self.p_best_rate = p_best_rate
        self.archive_size = max(1, int(archive_size_factor * pop_size))

        # SHADE state — initialized lazily in _initialize_state
        self.M_CR = np.full(h_memory_size, 0.5)
        self.M_F = np.full(h_memory_size, 0.5)
        self.memory_idx = 0
        self.archive: list[np.ndarray] = []

    def _initialize_state(self) -> None:
        """Reset SHADE-specific state at the start of a run."""
        self.M_CR = np.full(self.h_memory_size, 0.5)
        self.M_F = np.full(self.h_memory_size, 0.5)
        self.memory_idx = 0
        self.archive = []

    def _initialize_operators(self) -> None:
        self.register_operator(Operator(
            name="mutation_current_to_pbest_archive",
            operator_type="mutation",
            function=self._mutation_apply,
            parameters={},
            description="SHADE current-to-pbest/1 mutation with archive",
        ))
        self.register_operator(Operator(
            name="crossover_binomial",
            operator_type="crossover",
            function=self._crossover_apply,
            parameters={},
            description="DE binomial crossover with per-individual CR",
        ))
        self.register_operator(Operator(
            name="selection_greedy",
            operator_type="selection",
            function=self._selection_apply,
            parameters={},
            description="Greedy selection with archive update",
        ))
        self.register_operator(Operator(
            name="parameter_adaptation",
            operator_type="parameter_adaptation",
            function=self._parameter_adapt_apply,
            parameters={},
            description="History-based (F, CR) memory adaptation (SHADE)",
        ))

    def disable_operator(self, name: str):
        """Override base implementation to use SHADE-specific neutrals.

        The base class's ``get_neutral_operator`` factory only handles the
        standard EC operator types (mutation/crossover/selection/velocity/
        position/topology) and uses neutral signatures that do not match the
        SHADE-side ``apply`` calls. SHADE provides its own neutrals via
        ``get_neutral_operator`` (instance method) below.
        """
        if name in self.operators:
            op_type = self.operators[name].operator_type
            self.operators[name] = self.get_neutral_operator(op_type)

    def get_neutral_operator(self, operator_type: str):
        """Return a no-op-style neutral counterpart for each operator type.

        These are used when the cooperative-game machinery disables an
        operator: the neutral preserves the algorithm's data-generating
        process but removes the operator's contribution.
        """
        if operator_type == "mutation":
            return Operator(
                name="neutral_mutation",
                operator_type="mutation",
                function=lambda population, target_idx, **kw: population[target_idx].copy(),
                description="Neutral mutation (identity)",
            )
        if operator_type == "crossover":
            return Operator(
                name="neutral_crossover",
                operator_type="crossover",
                function=lambda target, mutant, **kw: mutant.copy(),
                description="Neutral crossover (mutant passes through)",
            )
        if operator_type == "selection":
            def neutral_select(target, trial, target_fitness, trial_fitness, rng, **kw):
                if rng.random() < 0.5:
                    return trial, trial_fitness, True
                return target, target_fitness, False
            return Operator(
                name="neutral_selection",
                operator_type="selection",
                function=neutral_select,
                description="Neutral selection (random replacement, p=0.5)",
            )
        if operator_type == "parameter_adaptation":
            return Operator(
                name="neutral_parameter_adaptation",
                operator_type="parameter_adaptation",
                function=lambda *a, **kw: None,
                description="Neutral parameter adaptation (no memory update)",
            )
        raise ValueError(f"Unknown operator_type: {operator_type}")

    # ============================================================
    # Operator implementations
    # ============================================================

    def _mutation_apply(self, population, target_idx, F, fitness, rng, **kwargs):
        """current-to-pbest/1 with archive."""
        n = len(population)
        # p-best candidates: top p_best_rate × n by fitness (lowest)
        p_count = max(2, int(self.p_best_rate * n))
        pbest_idxs = np.argsort(fitness)[:p_count]
        pbest_idx = rng.choice(pbest_idxs)
        x_pbest = population[pbest_idx]

        # r1 from population, r2 from population ∪ archive
        candidates_r1 = [i for i in range(n) if i != target_idx and i != pbest_idx]
        r1 = rng.choice(candidates_r1)
        x_r1 = population[r1]

        archive_n = len(self.archive)
        pool_size = n + archive_n
        r2 = rng.integers(0, pool_size)
        while r2 == target_idx or r2 == pbest_idx or r2 == r1:
            r2 = rng.integers(0, pool_size)
        if r2 < n:
            x_r2 = population[r2]
        else:
            x_r2 = self.archive[r2 - n]

        x_i = population[target_idx]
        return x_i + F * (x_pbest - x_i) + F * (x_r1 - x_r2)

    def _crossover_apply(self, target, mutant, CR, rng, **kwargs):
        """Binomial crossover with per-individual CR."""
        d = len(target)
        mask = rng.random(d) <= CR
        jrand = rng.integers(0, d)
        mask[jrand] = True
        return np.where(mask, mutant, target)

    def _selection_apply(self, target, trial, target_fitness, trial_fitness,
                         rng, target_idx=None, **kwargs):
        """Greedy selection with archive update on success."""
        if trial_fitness <= target_fitness:
            if target_idx is not None:
                self._record_archive_candidate(target)
            return trial, trial_fitness, True
        return target, target_fitness, False

    def _record_archive_candidate(self, individual: np.ndarray):
        """Add an inferior individual to the archive, evicting at random if full."""
        if len(self.archive) < self.archive_size:
            self.archive.append(individual.copy())
        else:
            idx = self.rng.integers(0, len(self.archive))
            self.archive[idx] = individual.copy()

    def _parameter_adapt_apply(self, S_F, S_CR, S_delta, **kwargs):
        """Update memory (M_F[k], M_CR[k]) using weighted means of successful (F, CR).

        Uses the SHADE weighting scheme: w_i = delta_i / sum(delta).
        M_F update uses the Lehmer (mean-of-squares / mean) form; M_CR uses
        the weighted arithmetic mean.
        """
        if len(S_F) == 0:
            return
        w = np.array(S_delta) / max(sum(S_delta), 1e-12)
        m_cr_new = float(np.sum(w * np.array(S_CR)))
        sf = np.array(S_F)
        m_f_new = float(np.sum(w * sf * sf) / max(np.sum(w * sf), 1e-12))
        self.M_CR[self.memory_idx] = m_cr_new
        self.M_F[self.memory_idx] = m_f_new
        self.memory_idx = (self.memory_idx + 1) % self.h_memory_size

    # ============================================================
    # Generation step
    # ============================================================

    def _generation_step(self) -> Tuple[np.ndarray, np.ndarray]:
        mutation_op = self.get_operator("mutation_current_to_pbest_archive")
        crossover_op = self.get_operator("crossover_binomial")
        selection_op = self.get_operator("selection_greedy")
        adapt_op = self.get_operator("parameter_adaptation")
        adapt_active = adapt_op.name == "parameter_adaptation"

        new_population = np.empty_like(self.population)
        new_fitness = np.empty_like(self.fitness)

        S_F: list[float] = []
        S_CR: list[float] = []
        S_delta: list[float] = []

        for i in range(self.pop_size):
            # Sample per-individual F, CR from memory (active adaptation)
            # or fixed values (neutral adaptation)
            if adapt_active:
                r_i = self.rng.integers(0, self.h_memory_size)
                CR_i = float(np.clip(self.rng.normal(self.M_CR[r_i], 0.1), 0.0, 1.0))
                F_i = -1.0
                while F_i <= 0:
                    F_i = float(self.rng.standard_cauchy() * 0.1 + self.M_F[r_i])
                F_i = float(min(F_i, 1.0))
            else:
                CR_i = 0.5
                F_i = 0.5

            self.logger.log_operator_selection(
                operator_name=mutation_op.name,
                parameters={"F": F_i, "target_idx": i},
                rationale=f"SHADE mutation on individual {i}",
            )
            mutant = mutation_op.apply(
                population=self.population, target_idx=i,
                F=F_i, fitness=self.fitness, rng=self.rng,
            )
            mutant = self.clip_to_bounds(mutant)

            self.logger.log_operator_selection(
                operator_name=crossover_op.name,
                parameters={"CR": CR_i, "target_idx": i},
                rationale="SHADE binomial crossover",
            )
            trial = crossover_op.apply(
                target=self.population[i], mutant=mutant,
                CR=CR_i, rng=self.rng,
            )
            trial = self.clip_to_bounds(trial)

            trial_fitness = self._current_objective(trial)
            self.evaluations += 1

            new_x, new_f, success = selection_op.apply(
                target=self.population[i], trial=trial,
                target_fitness=self.fitness[i], trial_fitness=trial_fitness,
                rng=self.rng, target_idx=i,
            )
            new_population[i] = new_x
            new_fitness[i] = new_f

            improvement = (self.fitness[i] - trial_fitness) if success else 0.0
            if success and improvement > 0:
                S_F.append(F_i); S_CR.append(CR_i); S_delta.append(improvement)

            for op in (mutation_op, crossover_op, selection_op):
                self.logger.log_operator_application(
                    operator_name=op.name,
                    parent_indices=[i],
                    offspring_fitness=[trial_fitness],
                    success=success,
                    details={
                        "target_fitness": float(self.fitness[i]),
                        "trial_fitness": float(trial_fitness),
                        "improvement": float(improvement),
                        "F": F_i, "CR": CR_i,
                    },
                )

        # Parameter adaptation fires once per generation (or skipped if neutral)
        prev_M_CR = self.M_CR.copy()
        prev_M_F = self.M_F.copy()
        adapt_op.apply(S_F=S_F, S_CR=S_CR, S_delta=S_delta)
        adaptation_changed = (
            not np.allclose(prev_M_CR, self.M_CR) or
            not np.allclose(prev_M_F, self.M_F)
        )
        self.logger.log_operator_application(
            operator_name=adapt_op.name,
            parent_indices=[],
            offspring_fitness=[],
            success=adaptation_changed,
            details={
                "n_successes": len(S_F),
                "memory_idx": int(self.memory_idx),
                "improvement": float(sum(S_delta)),
            },
        )

        return new_population, new_fitness
