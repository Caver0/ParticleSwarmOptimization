from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from pso_lab.experiments.runner import ExperimentResult

@dataclass(slots=True)
class ExperimentSummary:
    """Aggregated statistics for multiple PSO runs."""

    objective_name: str
    num_runs: int
    mean_best_value: float
    std_best_value: float
    min_best_value: float
    max_best_value: float


def summarize_experiments(results: list[ExperimentResult]) -> ExperimentSummary:
    """Compute summary statistics from a list of experiment results."""
    if not results:
        raise ValueError("Cannot summarize an empty list of experiment results.")
    
    values = np.asarray([result.best_value for result in results], dtype=float)
    objective_name = results[0].objective_name

    return ExperimentSummary(
        objective_name=objective_name,
        num_runs=len(results),
        mean_best_value= float(np.mean(values)),
        std_best_value= float(np.std(values)),
        min_best_value= float(np.min(values)),
        max_best_value= float(np.max(values)),
    )
