from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from pso_lab.objectives import ObjectiveFunction
from concurrent.futures import ThreadPoolExecutor

class FitnessEvaluator(ABC):
    @abstractmethod
    def evaluate(
        self,
        objective_function: ObjectiveFunction,
        positions: np.ndarray,
    ) -> np.ndarray:
        pass

    def shutdown(self) -> None:
        pass

class SequentialEvaluator(FitnessEvaluator):
    def evaluate(self, objective_function: ObjectiveFunction, positions: np.ndarray) -> np.ndarray:
        return objective_function.evaluate_many(positions)
    

class ThreadPoolEvaluator(FitnessEvaluator):
    def __init__(self, max_workers: int | None = None):
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