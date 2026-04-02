from __future__ import annotations
from dataclasses import asdict, dataclass
from time import perf_counter

from pso_lab.core.config import PSOConfig
from pso_lab.core.optimizer import PSOOptimizer
from pso_lab.objectives import build_objective
from pso_lab.parallel.evaluators import FitnessEvaluator, SequentialEvaluator, ThreadPoolEvaluator
@dataclass(slots= True)
class ExperimentResult:
    """Stores the result of a single PSO run."""

    objective_name: str
    seed: int|None
    best_position: list[float]
    best_value: float
    iterations_completed: int
    elapsed_time_s: float
    best_value_history: list[float]
    config: dict


def build_evaluator(mode:str = "sequential", max_workers: int | None = None,) -> FitnessEvaluator:
    """Build the evaluator strategy used to compute particle fitness."""
    normalized_mode = mode.strip().lower()

    if normalized_mode in {"sequential", "v0"}:
        return SequentialEvaluator()
    if normalized_mode in {"threading", "threads", "v1"}:
        return ThreadPoolEvaluator(max_workers=max_workers)
    
    raise ValueError(f"Modo de evaluación desconocido: {mode}")

def run_single_experiment(
        objective_name: str, 
        config: PSOConfig,
        evaluation_mode: str = "sequential",
        max_workers: int | None = None,
) -> ExperimentResult:
    """Run a single PSO experiment and returns its results."""

    objective = build_objective(objective_name, dimensions=config.dimensions,)
    evaluator = build_evaluator(mode = evaluation_mode, max_workers=max_workers,)
    try:

        optimizer = PSOOptimizer(config=config, objective_function=objective, evaluator=evaluator,)
        start = perf_counter()
        optimizaiton_result = optimizer.optimize()
        elapsed_time_s = perf_counter() - start
    finally:
        evaluator.shutdown()

    return ExperimentResult(
        objective_name=objective.name,
        seed = config.seed,
        best_position=optimizaiton_result.best_position.tolist(),
        best_value=float(optimizaiton_result.best_value),
        iterations_completed=optimizaiton_result.iterations_completed,
        elapsed_time_s=elapsed_time_s,
        best_value_history=optimizaiton_result.best_value_history,
        config=asdict(config),
    )

