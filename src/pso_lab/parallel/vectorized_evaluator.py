from __future__ import annotations

import numpy as np

from pso_lab.objectives import ObjectiveFunction
from pso_lab.parallel.evaluators import FitnessEvaluator


class VectorizedEvaluator(FitnessEvaluator):
    """NumPy-based evaluator used in V4."""

    def evaluate(
        self,
        objective_function: ObjectiveFunction,
        positions: np.ndarray,
    ) -> np.ndarray:
        positions_array = np.asarray(positions, dtype=float)
        n_particles = positions_array.shape[0]

        evaluate_batch = getattr(objective_function, "evaluate_batch", None)
        if callable(evaluate_batch):
            values = evaluate_batch(positions_array)
        else:
            values = [objective_function(position) for position in positions_array]

        values_array = np.asarray(values, dtype=float)
        expected_shape = (n_particles,)
        if values_array.shape != expected_shape:
            raise ValueError(
                "VectorizedEvaluator expected fitness values with shape "
                f"{expected_shape}, got {values_array.shape}"
            )

        return values_array
