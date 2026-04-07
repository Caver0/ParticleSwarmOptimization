from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .base import ObjectiveFunction

@dataclass(slots=True)
class SphereObjective(ObjectiveFunction):
    def __call__(self, x: np.ndarray) -> float:
        return float(np.sum(np.square(x)))
    

@dataclass(slots=True)
class RosenbrockObjective(ObjectiveFunction):
    def __call__(self, x: np.ndarray) -> float:
        return float(
            np.sum(
                100.0 * np.square(x[1:] - np.square(x[:-1])) + np.square(1.0 - x[:-1])
            )
        )
    

@dataclass(slots=True)
class RastriginObjective(ObjectiveFunction):
    def __call__(self, x: np.ndarray) -> float:
        # Use the algebraically equivalent form x^2 + 10 * (1 - cos(.))
        # to avoid catastrophic cancellation near the global optimum.
        oscillation = 1.0 - np.cos(2.0 * np.pi * x)
        return float(np.sum(np.square(x) + 10.0 * oscillation))
    
@dataclass(slots=True)
class AckleyObjective(ObjectiveFunction):
    def __call__(self, x: np.ndarray) -> float:
        a = 20.0
        b = 0.2
        c = 2.0 * np.pi
        d =x.size
        sum_sq = np.sum(np.square(x))
        cos_sum = np.sum(np.cos(c * x))
        term1 = -a * np.exp(-b * np.sqrt(sum_sq/d))
        term2 = -np.exp(cos_sum/d)
        return float(term1 + term2 + a + np.e)

def _uniform_bounds(dimensions: int, lower:float, upper:float) -> list[tuple[float, float]]:
    return [(lower ,upper) for _ in range(dimensions)]

def build_objective(name: str, dimensions: int) -> ObjectiveFunction:
    normalized_name = name.strip().lower()
    aliases = {
        "sphere": "sphere",
        "rosenbrock": "rosenbrock",
        "rosenbrok": "rosenbrock",
        "rastrigin": "rastrigin",
        "ackley": "ackley",
    }
    canonical_name = aliases.get(normalized_name)
    if canonical_name is None:
        raise ValueError(f"Unknown objective '{name}'")

    if canonical_name == "sphere":
        return SphereObjective(
            name="sphere",
            dimensions=dimensions,
            bounds=_uniform_bounds(dimensions, -5.12, 5.12),
        )
    if canonical_name == "rosenbrock":
        return RosenbrockObjective(
            name="rosenbrock",
            dimensions=dimensions,
            bounds=_uniform_bounds(dimensions, -2.048, 2.048),
        )
    if canonical_name == "rastrigin":
        return RastriginObjective(
            name="rastrigin",
            dimensions=dimensions,
            bounds=_uniform_bounds(dimensions, -5.12, 5.12),
        )
    return AckleyObjective(
        name="ackley",
        dimensions=dimensions,
        bounds=_uniform_bounds(dimensions, -32.768, 32.768),
    )
