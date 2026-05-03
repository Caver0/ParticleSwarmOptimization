from __future__ import annotations

import asyncio

import numpy as np

from pso_lab.objectives import ObjectiveFunction
from pso_lab.parallel.evaluators import FitnessEvaluator


class UniformDelaySampler:
    """Samples reproducible per-particle delays for async fitness evaluation."""

    def __init__(
        self,
        *,
        min_delay: float = 0.0,
        max_delay: float = 0.0,
        seed: int | None = None,
    ) -> None:
        if min_delay < 0.0:
            raise ValueError("min_delay must be >= 0.0")
        if max_delay < 0.0:
            raise ValueError("max_delay must be >= 0.0")
        if min_delay > max_delay:
            raise ValueError("min_delay must be <= max_delay")

        self.min_delay = float(min_delay)
        self.max_delay = float(max_delay)
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def sample(self, count: int) -> np.ndarray:
        if count < 0:
            raise ValueError("count must be >= 0")
        if count == 0:
            return np.zeros(0, dtype=float)
        if self.max_delay == 0.0:
            return np.zeros(count, dtype=float)
        if self.min_delay == self.max_delay:
            return np.full(count, self.min_delay, dtype=float)
        return self._rng.uniform(self.min_delay, self.max_delay, size=count)


class AsyncEvaluator(FitnessEvaluator):
    """Asyncio-based evaluator used in V3 for latency-bound objectives."""

    def __init__(
        self,
        *,
        min_delay: float = 0.0,
        max_delay: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self.min_delay = float(min_delay)
        self.max_delay = float(max_delay)
        self.seed = seed
        self.delay_sampler = UniformDelaySampler(
            min_delay=min_delay,
            max_delay=max_delay,
            seed=seed,
        )
        self.last_delays = np.zeros(0, dtype=float)

    async def _evaluate_one(
        self,
        objective_function: ObjectiveFunction,
        position: np.ndarray,
        delay_s: float,
    ) -> float:
        if delay_s > 0.0:
            await asyncio.sleep(delay_s)
        return float(objective_function(position))

    async def _evaluate_all(
        self,
        objective_function: ObjectiveFunction,
        positions: np.ndarray,
    ) -> np.ndarray:
        delays = self.delay_sampler.sample(len(positions))
        self.last_delays = delays.copy()
        tasks = [
            self._evaluate_one(objective_function, position, delay_s)
            for position, delay_s in zip(positions, delays)
        ]
        if not tasks:
            return np.zeros(0, dtype=float)
        results = await asyncio.gather(*tasks)
        return np.asarray(results, dtype=float)

    def evaluate(
        self,
        objective_function: ObjectiveFunction,
        positions: np.ndarray,
    ) -> np.ndarray:
        positions = np.asarray(positions, dtype=float)
        if positions.size == 0:
            self.last_delays = np.zeros(positions.shape[0], dtype=float)
            return np.zeros(positions.shape[0], dtype=float)
        return asyncio.run(self._evaluate_all(objective_function, positions))
