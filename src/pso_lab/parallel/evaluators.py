from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from pso_lab.objectives import ObjectiveFunction
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Iterable


def _evaluate_batch(
        objective_function: ObjectiveFunction,
        positions: np.ndarray,
) -> np.ndarray:
    """Evaluate a batch of particle positions sequentially."""
    return objective_function.evaluate_many(positions)


class FitnessEvaluator(ABC):
    @abstractmethod
    def evaluate(
        self,
        objective_function: ObjectiveFunction,
        positions: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError

    def shutdown(self) -> None:
        pass

class SequentialEvaluator(FitnessEvaluator):
    """Sequential-based evaluator used in V0."""
    def evaluate(self, objective_function: ObjectiveFunction, positions: np.ndarray) -> np.ndarray:
        return objective_function.evaluate_many(positions)
    

class ThreadPoolEvaluator(FitnessEvaluator):
    """Thread-based evaluator used in V1."""
    def __init__(self, max_workers: int | None = None) -> None:
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def evaluate(
        self,
        objective_function: ObjectiveFunction,
        positions: np.ndarray,
    ) -> np.ndarray:
        values = list(self._executor.map(objective_function, positions))
        return np.asarray(values, dtype=float)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)

class ProcessPoolEvaluator(FitnessEvaluator):
    """Process-based evaluator used in V2."""
    def __init__(self, max_workers:int | None = None,
                 batch_size: int | None = None) -> None:
        self.max_workers = max_workers
        self.batch_size = batch_size
        self._executor = ProcessPoolExecutor(max_workers=max_workers)
    
    def _split_batches(self, positions: np.ndarray) -> list[np.ndarray]:
        n_particles = positions.shape[0]

        if self.batch_size is None or self.batch_size <= 0 or self.batch_size >= n_particles:
            return [positions]

        batches = []
        for start in range(0, n_particles, self.batch_size):
            end = start + self.batch_size
            batches.append(positions[start:end])
        return batches

    def evaluate(self, objective_function: ObjectiveFunction, positions: np.ndarray) -> np.ndarray:
        batches = self._split_batches(positions)
        mapped_results = self._executor.map(
            _evaluate_batch,
            [objective_function] * len(batches),
            batches
        )

        values = [np.asarray(batch_values, dtype=float) for batch_values in mapped_results]
        return np.concatenate(values)
    

    def shutdown(self) -> None:
        self._executor.shutdown(wait = True)