"""
XMH Algorithms Module

Contains instrumented implementations of popular metaheuristic algorithms.
"""

from .instrumented_de import InstrumentedDE, AdaptiveDE
from .instrumented_ga import InstrumentedGA, SteadyStateGA
from .instrumented_pso import InstrumentedPSO, CPSO
from .instrumented_cmaes import InstrumentedCMAES
from .instrumented_shade import InstrumentedSHADE
from .instrumented_jade import InstrumentedJADE

__all__ = [
    # Differential Evolution
    "InstrumentedDE",
    "AdaptiveDE",
    # Genetic Algorithm
    "InstrumentedGA",
    "SteadyStateGA",
    # Particle Swarm Optimization
    "InstrumentedPSO",
    "CPSO",
    # CMA-ES
    "InstrumentedCMAES",
    # SHADE (n=4)
    "InstrumentedSHADE",
    # JADE (n=4)
    "InstrumentedJADE",
]
