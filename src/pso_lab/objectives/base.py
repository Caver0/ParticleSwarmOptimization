from __future__ import annotations

from dataclasses import dataclass

import numpy as np

@dataclass(slots=True)
class ObjectiveFunction:
    """Callable minimization objective with default box constrains."""

    name: str
    dimensions: int
    bounds: list[tuple[float, float]]
    # Evaluate the objective function at a single point.
    # This method must be implemented by all subclasses.
    # It takes a position vector x and returns a scalar fitness value.
    # Lower values indicate better solutions, as the problem is formulated as minimization.
    def __call__(self, x: np.ndarray) -> float:
        raise NotImplementedError
    
    # Evaluate the objective function for multiple positions at once.
    # Each row in the input array represents a particle position.
    # Internally, this method applies the objective function to each position
    # and returns a NumPy array of fitness values.
    def evaluate_many(self, positions: np.ndarray) -> np.ndarray:
        return np.asarray([self(position) for position in positions], dtype=float)