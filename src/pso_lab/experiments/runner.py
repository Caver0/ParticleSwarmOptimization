from __future__ import annotations
from dataclasses import asdict, dataclass
import numpy as np
from pso_lab.core.config import PSOConfig
from pso_lab.core.optimizer import PSOOptimizer
from pso_lab.objectives import build_objective

@dataclass(slots= True)
class ExperimentResult:
    """Stores the result of a single PSO run."""

    objective_name: str
    seed: int|None
    best_position: list[float]
    best_value: float
    config: dict

def run_single_experiment(
        objective_name: str, 
        config: PSOConfig,
) -> ExperimentResult:
    """Run a single PSO experiment and returns its results."""

    objective = build_objective(objective_name, dimensions=config.dimensions)
    optimizer = PSOOptimizer(config=config, objective_function=objective)

    best_position, best_value = optimizer.optimize()

    return ExperimentResult(
        objective_name=objective.name,
        seed = config.seed,
        best_position=best_position.tolist(),
        best_value=float(best_value),
        config=asdict(config),
    )

