from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

from pso_lab.core.config import PSOConfig
from pso_lab.core.optimizer import PSOOptimizer
from pso_lab.objectives import build_objective
from pso_lab.parallel.async_evaluator import AsyncEvaluator
from pso_lab.parallel.evaluators import (
    FitnessEvaluator,
    ProcessPoolEvaluator,
    SequentialEvaluator,
    ThreadPoolEvaluator,
)
from pso_lab.parallel.vectorized_evaluator import VectorizedEvaluator


@dataclass(slots=True)
class ExperimentResult:
    """Stores the result of a single PSO run."""

    objective_name: str
    evaluation_mode: str
    seed: int | None
    best_position: list[float]
    best_value: float
    iterations_completed: int
    elapsed_time_s: float
    best_value_history: list[float]
    timing_stats: dict
    config: dict
    swarm_position_history: list[list[list[float]]] | None = None


def build_evaluator(
    mode: str = "sequential",
    max_workers: int | None = None,
    batch_size: int | None = None,
    async_min_delay: float = 0.0,
    async_max_delay: float = 0.0,
    async_seed: int | None = None,
) -> FitnessEvaluator:
    """Build the evaluator strategy used to compute particle fitness."""
    normalized_mode = mode.strip().lower()

    if normalized_mode in {"sequential", "v0"}:
        return SequentialEvaluator()
    if normalized_mode in {"threading", "threads", "v1"}:
        return ThreadPoolEvaluator(max_workers=max_workers)
    if normalized_mode in {"multiprocessing", "processes", "v2"}:
        return ProcessPoolEvaluator(max_workers=max_workers, batch_size=batch_size)
    if normalized_mode in {"asyncio", "async", "v3"}:
        return AsyncEvaluator(
            min_delay=async_min_delay,
            max_delay=async_max_delay,
            seed=async_seed,
        )
    if normalized_mode in {"vectorized", "vectorised", "numpy", "v4"}:
        return VectorizedEvaluator()

    raise ValueError(f"Modo de evaluación desconocido: {mode}")


def run_single_experiment(
    objective_name: str,
    config: PSOConfig,
    evaluation_mode: str = "sequential",
    max_workers: int | None = None,
    batch_size: int | None = None,
    async_min_delay: float = 0.0,
    async_max_delay: float = 0.0,
    async_seed: int | None = None,
) -> ExperimentResult:
    """Run a single PSO experiment and returns its results."""

    objective = build_objective(objective_name, dimensions=config.dimensions)
    evaluator = build_evaluator(
        mode=evaluation_mode,
        max_workers=max_workers,
        batch_size=batch_size,
        async_min_delay=async_min_delay,
        async_max_delay=async_max_delay,
        async_seed=config.seed if async_seed is None else async_seed,
    )

    try:
        optimizer = PSOOptimizer(
            config=config,
            objective_function=objective,
            evaluator=evaluator,
        )
        start = perf_counter()
        optimization_result = optimizer.optimize()
        elapsed_time_s = perf_counter() - start
    finally:
        evaluator.shutdown()

    return ExperimentResult(
        objective_name=objective.name,
        evaluation_mode=evaluation_mode,
        seed=config.seed,
        best_position=optimization_result.best_position.tolist(),
        best_value=float(optimization_result.best_value),
        iterations_completed=optimization_result.iterations_completed,
        elapsed_time_s=elapsed_time_s,
        best_value_history=optimization_result.best_value_history,
        timing_stats=asdict(optimization_result.timing_stats),
        config=asdict(config),
        swarm_position_history=(
            [positions.tolist() for positions in optimization_result.swarm_position_history]
            if optimization_result.swarm_position_history is not None
            else None
        ),
    )
