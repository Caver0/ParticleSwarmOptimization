from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import asdict
import io
import inspect
import logging
from time import perf_counter

import numpy as np

from pso_lab.core.config import PSOConfig
from pso_lab.experiments.runner import ExperimentResult
from pso_lab.objectives import build_objective


def _import_pyswarm_pso():
    try:
        from pyswarm import pso
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pyswarm is required to run the baseline comparison. "
            "Install it with 'pip install pyswarm' and rerun the command."
        ) from exc
    return pso


def ensure_pyswarm_available() -> None:
    _import_pyswarm_pso()


def _filter_supported_pyswarm_kwargs(pso_callable, kwargs: dict[str, object]) -> dict[str, object]:
    try:
        signature = inspect.signature(pso_callable)
    except (TypeError, ValueError):
        filtered_kwargs = dict(kwargs)
        filtered_kwargs.pop("processes", None)
        return filtered_kwargs

    return {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }


def _build_best_value_history(
    evaluation_values: list[float],
    *,
    swarmsize: int,
    iterations_completed: int,
) -> list[float]:
    if iterations_completed <= 0:
        return []

    block_mins = []
    for block_start in range(0, len(evaluation_values), swarmsize):
        block = evaluation_values[block_start:block_start + swarmsize]
        if len(block) != swarmsize:
            break
        block_mins.append(float(np.min(block)))

    if not block_mins:
        return []

    history = []
    running_best = float(block_mins[0])
    for iteration in range(1, min(iterations_completed + 1, len(block_mins))):
        running_best = min(running_best, float(block_mins[iteration]))
        history.append(running_best)
    return history


def _build_swarm_position_history(
    evaluated_positions: list[np.ndarray],
    *,
    swarmsize: int,
) -> list[list[list[float]]]:
    swarm_history: list[list[list[float]]] = []
    for block_start in range(0, len(evaluated_positions), swarmsize):
        block = evaluated_positions[block_start:block_start + swarmsize]
        if len(block) != swarmsize:
            break
        swarm_history.append(np.asarray(block, dtype=float).tolist())
    return swarm_history


def run_pyswarm_experiment(
    objective_name: str,
    config: PSOConfig,
    *,
    logger: logging.Logger | None = None,
) -> ExperimentResult:
    pso = _import_pyswarm_pso()
    objective = build_objective(objective_name, dimensions=config.dimensions)

    bounds = np.asarray(objective.bounds, dtype=float)
    lower_bounds = bounds[:, 0].tolist()
    upper_bounds = bounds[:, 1].tolist()

    evaluation_values: list[float] = []
    evaluated_positions: list[np.ndarray] = []

    def tracked_objective(x: np.ndarray) -> float:
        position = np.asarray(x, dtype=float)
        if config.track_swarm_history:
            evaluated_positions.append(position.copy())

        value = float(objective(position))
        evaluation_values.append(value)
        return value

    if config.seed is not None:
        np.random.seed(config.seed)

    stdout_buffer = io.StringIO()
    pso_kwargs = _filter_supported_pyswarm_kwargs(
        pso,
        {
            "swarmsize": config.num_particles,
            "omega": config.inertia_weight,
            "phip": config.cognitive_coefficient,
            "phig": config.social_coefficient,
            "maxiter": config.max_iterations,
            "minstep": config.tolerance if config.tolerance > 0.0 else 0.0,
            "minfunc": config.tolerance if config.tolerance > 0.0 else 0.0,
            "debug": False,
            "processes": 1,
        },
    )
    start = perf_counter()
    with redirect_stdout(stdout_buffer):
        best_position, best_value = pso(
            tracked_objective,
            lower_bounds,
            upper_bounds,
            **pso_kwargs,
        )
    elapsed_time_s = perf_counter() - start

    solver_messages = [
        line.strip()
        for line in stdout_buffer.getvalue().splitlines()
        if line.strip()
    ]
    if logger is not None:
        for message in solver_messages:
            logger.info("pyswarm | %s", message)

    iterations_completed = max(
        0,
        min(
            config.max_iterations,
            len(evaluation_values) // config.num_particles - 1,
        ),
    )
    best_value_history = (
        _build_best_value_history(
            evaluation_values,
            swarmsize=config.num_particles,
            iterations_completed=iterations_completed,
        )
        if config.track_history
        else []
    )
    swarm_position_history = (
        _build_swarm_position_history(
            evaluated_positions,
            swarmsize=config.num_particles,
        )
        if config.track_swarm_history
        else None
    )

    timing_stats = {
        "total_time_s": elapsed_time_s,
        "fitness_time_s": None,
        "velocity_update_time_s": None,
        "position_update_time_s": None,
        "solver": "pyswarm",
        "solver_messages": solver_messages,
    }

    return ExperimentResult(
        objective_name=objective.name,
        evaluation_mode="pyswarm",
        seed=config.seed,
        best_position=np.asarray(best_position, dtype=float).tolist(),
        best_value=float(best_value),
        iterations_completed=iterations_completed,
        elapsed_time_s=elapsed_time_s,
        best_value_history=best_value_history,
        timing_stats=timing_stats,
        config=asdict(config),
        swarm_position_history=swarm_position_history,
    )
