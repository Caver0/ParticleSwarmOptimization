from __future__ import annotations
import numpy as np

from pso_lab.core.config import PSOConfig
from pso_lab.experiments.runner import build_evaluator, run_single_experiment
from pso_lab.objectives import build_objective
from pso_lab.parallel.evaluators import SequentialEvaluator, ThreadPoolEvaluator, ProcessPoolEvaluator

def test_sequential_evaluator_matches_objective_evaluate_many() -> None:
    objective = build_objective("sphere", dimensions=2)
    positions = np.array([
        [0.0, 0.0],
        [1.0, 2.0],
        [-3.0, 4.0],
    ], dtype=float)
    evaluator = SequentialEvaluator()

    expected = objective.evaluate_many(positions)
    obtained = evaluator.evaluate(objective, positions)

    assert np.allclose(obtained, expected)


def test_threadpool_evaluator_matches_sequential_evaluator() -> None: 
    objective = build_objective("rastrigin", dimensions=2)
    positions = np.array([
        [0.0, 0.0],
        [1.0, 2.0],
        [-1.5, 3.5],
        [2.2, -0.7],
        [4.0, -4.0]
    ], dtype=float)

    sequential = SequentialEvaluator()
    threaded = ThreadPoolEvaluator(max_workers=4)

    try: 
        expected = sequential.evaluate(objective, positions)
        obtained = threaded.evaluate(objective, positions)

        assert np.allclose(expected, obtained)
    finally:
        threaded.shutdown()
    
def test_build_evaluator_returns_sequential() -> None:
    evaluator = build_evaluator(mode="sequential")
    assert isinstance(evaluator, SequentialEvaluator)

def test_build_evaluator_returns_threadpool() -> None:
    evaluator = build_evaluator(mode = "threading",  max_workers=4)

    try:
        assert isinstance(evaluator, ThreadPoolEvaluator)
    finally:
        evaluator.shutdown()

def test_run_single_experiment_threading_returns_valid_result() -> None:
    config = PSOConfig(
        num_particles=20,
        dimensions=2,
        max_iterations=30,
        inertia_weight=0.7,
        cognitive_coefficient=1.5,
        social_coefficient=1.5,
        seed=0,
        tolerance=0.0,
        stagnation_patience=None,
        track_history=True,
    )
    result = run_single_experiment(
        objective_name="sphere",
        config=config,
        evaluation_mode="threading",
        max_workers=4
    )

    assert result.objective_name == "sphere"
    assert result.evaluation_mode == "threading"
    assert isinstance(result.best_value, float)
    assert isinstance(result.best_position, list)
    assert len(result.best_position) == config.dimensions
    assert result.iterations_completed > 0
    assert len(result.best_value_history) == result.iterations_completed

def test_run_single_experiment_multiprocessing_returns_valid_result() -> None:
    config = PSOConfig(
        num_particles=20,
        dimensions=2,
        max_iterations=30,
        inertia_weight=0.7,
        cognitive_coefficient=1.5,
        social_coefficient=1.5,
        seed=0,
        tolerance=0.0,
        stagnation_patience=None,
        track_history=True,
    )

    result = run_single_experiment(
        objective_name="sphere",
        config=config,
        evaluation_mode="multiprocessing",
        max_workers=2, 
        batch_size=5,
    )

    assert result.objective_name == "sphere"
    assert result.evaluation_mode == "multiprocessing"
    assert isinstance(result.best_value, float)
    assert isinstance(result.best_position, list)
    assert len(result.best_value_history) == result.iterations_completed
    assert len(result.best_position) == config.dimensions
    assert result.iterations_completed > 0


def test_run_single_experiment_can_return_swarm_history() -> None:
    config = PSOConfig(
        num_particles=10,
        dimensions=3,
        max_iterations=12,
        inertia_weight=0.7,
        cognitive_coefficient=1.5,
        social_coefficient=1.5,
        seed=3,
        tolerance=0.0,
        stagnation_patience=None,
        track_history=True,
        track_swarm_history=True,
    )

    result = run_single_experiment(
        objective_name="sphere",
        config=config,
        evaluation_mode="sequential",
    )

    assert result.swarm_position_history is not None
    assert len(result.swarm_position_history) == result.iterations_completed + 1
    assert len(result.swarm_position_history[0]) == config.num_particles
    assert len(result.swarm_position_history[0][0]) == config.dimensions
