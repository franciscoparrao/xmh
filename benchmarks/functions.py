"""
Standard Benchmark Functions for Optimization

Includes classic test functions commonly used in metaheuristics research.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class BenchmarkFunction:
    """
    Represents a benchmark optimization function.

    Attributes:
        name: Function name
        function: The callable function
        dim: Dimensionality (None for any dimension)
        bounds: Default bounds as (lower, upper) tuple
        optimum: Known global optimum value
        optimum_position: Position of global optimum
        characteristics: List of function characteristics
    """
    name: str
    function: Callable[[np.ndarray], float]
    dim: Optional[int]
    bounds: Tuple[float, float]
    optimum: float
    optimum_position: Optional[np.ndarray]
    characteristics: List[str]

    def __call__(self, x: np.ndarray) -> float:
        """Evaluate the function."""
        return self.function(x)

    def get_bounds(self, dim: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get bounds for specified dimension."""
        lower = np.full(dim, self.bounds[0])
        upper = np.full(dim, self.bounds[1])
        return lower, upper

    def get_optimum_position(self, dim: int) -> np.ndarray:
        """Get optimum position for specified dimension."""
        if self.optimum_position is not None:
            if len(self.optimum_position) == dim:
                return self.optimum_position
            # Extend if needed
            return np.full(dim, self.optimum_position[0])
        return np.zeros(dim)


# ==================== BENCHMARK FUNCTIONS ====================

def sphere(x: np.ndarray) -> float:
    """
    Sphere function (De Jong's function 1).

    f(x) = sum(x_i^2)

    Properties:
    - Unimodal
    - Separable
    - Continuous
    - Convex
    """
    return np.sum(x ** 2)


def rosenbrock(x: np.ndarray) -> float:
    """
    Rosenbrock function (banana function).

    f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)

    Properties:
    - Unimodal (for D<=3), Multimodal (for D>3)
    - Non-separable
    - Has a narrow curved valley
    """
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


def rastrigin(x: np.ndarray) -> float:
    """
    Rastrigin function.

    f(x) = 10*D + sum(x_i^2 - 10*cos(2*pi*x_i))

    Properties:
    - Highly multimodal
    - Separable
    - Many local optima
    """
    D = len(x)
    return 10 * D + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


def schwefel(x: np.ndarray) -> float:
    """
    Schwefel function.

    f(x) = 418.9829*D - sum(x_i * sin(sqrt(|x_i|)))

    Properties:
    - Highly multimodal
    - Deceptive (global optimum far from other local optima)
    """
    D = len(x)
    return 418.9829 * D - np.sum(x * np.sin(np.sqrt(np.abs(x))))


def ackley(x: np.ndarray) -> float:
    """
    Ackley function.

    Properties:
    - Multimodal
    - Nearly flat outer region
    - Large hole at center
    """
    D = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))

    return (-20 * np.exp(-0.2 * np.sqrt(sum1 / D)) -
            np.exp(sum2 / D) + 20 + np.e)


def griewank(x: np.ndarray) -> float:
    """
    Griewank function.

    Properties:
    - Multimodal
    - Has many widespread local optima
    - Becomes easier with higher dimensions
    """
    sum_sq = np.sum(x**2) / 4000
    prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return sum_sq - prod_cos + 1


def levy(x: np.ndarray) -> float:
    """
    Levy function.

    Properties:
    - Multimodal
    - Non-separable
    """
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[0])**2
    term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
    return term1 + term2 + term3


def michalewicz(x: np.ndarray, m: float = 10) -> float:
    """
    Michalewicz function.

    Properties:
    - Multimodal
    - Has D! local optima
    - Steep valleys
    """
    D = len(x)
    i = np.arange(1, D + 1)
    return -np.sum(np.sin(x) * np.sin(i * x**2 / np.pi)**(2 * m))


def zakharov(x: np.ndarray) -> float:
    """
    Zakharov function.

    Properties:
    - Unimodal
    - Non-separable
    """
    D = len(x)
    i = np.arange(1, D + 1)
    sum1 = np.sum(x**2)
    sum2 = np.sum(0.5 * i * x)
    return sum1 + sum2**2 + sum2**4


def dixon_price(x: np.ndarray) -> float:
    """
    Dixon-Price function.

    Properties:
    - Unimodal
    - Non-separable
    """
    D = len(x)
    term1 = (x[0] - 1)**2
    i = np.arange(2, D + 1)
    term2 = np.sum(i * (2 * x[1:]**2 - x[:-1])**2)
    return term1 + term2


# ==================== BENCHMARK REGISTRY ====================

BENCHMARKS: Dict[str, BenchmarkFunction] = {
    "sphere": BenchmarkFunction(
        name="Sphere",
        function=sphere,
        dim=None,
        bounds=(-5.12, 5.12),
        optimum=0.0,
        optimum_position=None,  # Origin
        characteristics=["unimodal", "separable", "convex"]
    ),
    "rosenbrock": BenchmarkFunction(
        name="Rosenbrock",
        function=rosenbrock,
        dim=None,
        bounds=(-5.0, 10.0),
        optimum=0.0,
        optimum_position=np.array([1.0]),  # All ones
        characteristics=["unimodal", "non-separable", "valley"]
    ),
    "rastrigin": BenchmarkFunction(
        name="Rastrigin",
        function=rastrigin,
        dim=None,
        bounds=(-5.12, 5.12),
        optimum=0.0,
        optimum_position=None,  # Origin
        characteristics=["multimodal", "separable", "many_local_optima"]
    ),
    "schwefel": BenchmarkFunction(
        name="Schwefel",
        function=schwefel,
        dim=None,
        bounds=(-500.0, 500.0),
        optimum=0.0,
        optimum_position=np.array([420.9687]),
        characteristics=["multimodal", "deceptive"]
    ),
    "ackley": BenchmarkFunction(
        name="Ackley",
        function=ackley,
        dim=None,
        bounds=(-32.768, 32.768),
        optimum=0.0,
        optimum_position=None,  # Origin
        characteristics=["multimodal", "non-separable"]
    ),
    "griewank": BenchmarkFunction(
        name="Griewank",
        function=griewank,
        dim=None,
        bounds=(-600.0, 600.0),
        optimum=0.0,
        optimum_position=None,  # Origin
        characteristics=["multimodal", "non-separable"]
    ),
    "levy": BenchmarkFunction(
        name="Levy",
        function=levy,
        dim=None,
        bounds=(-10.0, 10.0),
        optimum=0.0,
        optimum_position=np.array([1.0]),  # All ones
        characteristics=["multimodal", "non-separable"]
    ),
    "zakharov": BenchmarkFunction(
        name="Zakharov",
        function=zakharov,
        dim=None,
        bounds=(-5.0, 10.0),
        optimum=0.0,
        optimum_position=None,  # Origin
        characteristics=["unimodal", "non-separable"]
    ),
    "dixon_price": BenchmarkFunction(
        name="Dixon-Price",
        function=dixon_price,
        dim=None,
        bounds=(-10.0, 10.0),
        optimum=0.0,
        optimum_position=None,
        characteristics=["unimodal", "non-separable"]
    ),
}


def get_benchmark(name: str) -> BenchmarkFunction:
    """Get a benchmark function by name."""
    if name.lower() not in BENCHMARKS:
        raise ValueError(
            f"Unknown benchmark: {name}. "
            f"Available: {list(BENCHMARKS.keys())}"
        )
    return BENCHMARKS[name.lower()]


def list_benchmarks() -> List[str]:
    """List all available benchmark functions."""
    return list(BENCHMARKS.keys())


def get_benchmarks_by_characteristic(characteristic: str) -> List[BenchmarkFunction]:
    """Get all benchmarks with a specific characteristic."""
    return [
        b for b in BENCHMARKS.values()
        if characteristic in b.characteristics
    ]
