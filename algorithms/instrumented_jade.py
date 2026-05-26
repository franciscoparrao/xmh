"""Instrumented JADE (Adaptive Differential Evolution with Optional Archive).

Implements JADE [Zhang & Sanner, 2009] as a four-operator cooperative game,
parallel to SHADE but with a simpler parameter-adaptation mechanism:

    1. mutation              (current-to-pbest/1 with archive) — same as SHADE
    2. crossover             (binomial) — same as SHADE
    3. selection             (greedy with archive update) — same as SHADE
    4. parameter_adaptation  (scalar mu_F, mu_CR; Lehmer-mean update)

This is the second n = 4 case used to corroborate the SHADE results. SHADE
and JADE share the same first three operators by construction and differ
only in how F and CR are adapted from successful past applications, so
comparing their Shapley breakdowns isolates the contribution of the
adaptation mechanism itself.

Neutral counterparts follow the same convention as ``InstrumentedSHADE``
(see ``instrumented_shade.py``).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from xmh.algorithms.instrumented_shade import InstrumentedSHADE
from xmh.core.instrumentation import Operator


class InstrumentedJADE(InstrumentedSHADE):
    """JADE: adaptive DE with Lehmer-mean parameter adaptation.

    Inherits the three classical DE operators (current-to-pbest mutation,
    binomial crossover, greedy selection with archive) from
    :class:`InstrumentedSHADE` and only replaces the parameter-adaptation
    operator with JADE's two-scalar moving-average update.
    """

    def __init__(
        self,
        dim: int,
        bounds: Tuple[np.ndarray, np.ndarray],
        pop_size: int = 100,
        max_generations: int = 100,
        p_best_rate: float = 0.05,
        archive_size_factor: float = 1.0,
        c_jade: float = 0.1,
        seed: Optional[int] = None,
    ):
        super().__init__(
            dim=dim, bounds=bounds, pop_size=pop_size,
            max_generations=max_generations,
            h_memory_size=1,   # JADE keeps a single scalar pair, not history
            p_best_rate=p_best_rate,
            archive_size_factor=archive_size_factor,
            seed=seed,
        )
        self.c_jade = c_jade
        # JADE state: two scalar moving averages
        self.mu_F = 0.5
        self.mu_CR = 0.5

    def _initialize_state(self) -> None:
        super()._initialize_state()
        self.mu_F = 0.5
        self.mu_CR = 0.5

    def _initialize_operators(self) -> None:
        # Reuse three of SHADE's operators; override the parameter_adaptation one
        super()._initialize_operators()
        # Replace SHADE's history-based adapt with JADE's Lehmer/moving-average
        self.operators["parameter_adaptation"] = Operator(
            name="parameter_adaptation",
            operator_type="parameter_adaptation",
            function=self._parameter_adapt_apply,
            parameters={},
            description="JADE scalar Lehmer-mean (F, CR) adaptation",
        )

    def _parameter_adapt_apply(self, S_F, S_CR, S_delta, **kwargs):
        """JADE update::

            mu_CR <- (1 - c) * mu_CR + c * arithmeticMean(S_CR)
            mu_F  <- (1 - c) * mu_F  + c * LehmerMean(S_F)

        where the Lehmer mean is ``sum(F^2) / sum(F)``. ``c`` is the
        learning rate (typically 0.1).
        """
        if len(S_F) == 0:
            return
        sf = np.array(S_F, dtype=float)
        mean_arith_cr = float(np.mean(S_CR))
        sf_sum = float(np.sum(sf))
        mean_lehmer_f = float(np.sum(sf * sf) / sf_sum) if sf_sum > 1e-12 else 0.5
        self.mu_F = (1.0 - self.c_jade) * self.mu_F + self.c_jade * mean_lehmer_f
        self.mu_CR = (1.0 - self.c_jade) * self.mu_CR + self.c_jade * mean_arith_cr

    # Generation step: same structure as SHADE but samples F, CR from scalar
    # moving averages instead of memory indices.
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
            if adapt_active:
                CR_i = float(np.clip(self.rng.normal(self.mu_CR, 0.1), 0.0, 1.0))
                F_i = -1.0
                while F_i <= 0:
                    F_i = float(self.rng.standard_cauchy() * 0.1 + self.mu_F)
                F_i = float(min(F_i, 1.0))
            else:
                CR_i = 0.5
                F_i = 0.5

            self.logger.log_operator_selection(
                operator_name=mutation_op.name,
                parameters={"F": F_i, "target_idx": i},
                rationale=f"JADE mutation on individual {i}",
            )
            mutant = mutation_op.apply(
                population=self.population, target_idx=i,
                F=F_i, fitness=self.fitness, rng=self.rng,
            )
            mutant = self.clip_to_bounds(mutant)

            self.logger.log_operator_selection(
                operator_name=crossover_op.name,
                parameters={"CR": CR_i, "target_idx": i},
                rationale="JADE binomial crossover",
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

        prev_mu = (self.mu_F, self.mu_CR)
        adapt_op.apply(S_F=S_F, S_CR=S_CR, S_delta=S_delta)
        adaptation_changed = (prev_mu != (self.mu_F, self.mu_CR))
        self.logger.log_operator_application(
            operator_name=adapt_op.name,
            parent_indices=[],
            offspring_fitness=[],
            success=adaptation_changed,
            details={
                "n_successes": len(S_F),
                "mu_F": float(self.mu_F),
                "mu_CR": float(self.mu_CR),
                "improvement": float(sum(S_delta)),
            },
        )

        return new_population, new_fitness
