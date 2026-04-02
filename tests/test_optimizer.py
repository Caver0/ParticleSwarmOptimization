import numpy as np

from pso_lab.core.config import PSOConfig
from pso_lab.core.optimizer import PSOOptimizer
from pso_lab.objectives import build_objective

def build_default_config(seed: int |None = 42) -> PSOConfig:
    return PSOConfig(
        num_particles=30,
        dimensions=2,
        max_iterations=100,
        inertia_weight=0.7,
        cognitive_coefficient=1.5,
        social_coefficient=1.5,
        seed=seed,
        tolerance=0.0,
        stagnation_patience=None,
        track_history=True,
    )

def test_optimizer_returns_valid_shapes():
    config = build_default_config()

    objective = build_objective("sphere", dimensions=config.dimensions)
    optimizer = PSOOptimizer(config=config, objective_function=objective)

    result = optimizer.optimize()

    assert isinstance(result.best_position, np.ndarray)
    assert result.best_position.shape == (2,)
    assert isinstance(result.best_value, float)


def test_optimizer_improves_on_sphere():
    config = build_default_config()

    objective = build_objective("sphere", dimensions=config.dimensions)
    optimizer = PSOOptimizer(config=config, objective_function=objective)

    result = optimizer.optimize()

    assert result.best_value < 1e-6


def test_optimizer_respects_bounds():
    config = build_default_config()

    objective = build_objective("sphere", dimensions=config.dimensions)
    optimizer = PSOOptimizer(config=config, objective_function=objective)

    result = optimizer.optimize()

    bounds = np.array(objective.bounds)
    lower = bounds[:, 0]
    upper = bounds[:, 1]

    assert np.all(result.best_position >= lower)
    assert np.all(result.best_position <= upper)

def test_optimizer_is_reproducible_with_Same_seed():
    config1 = build_default_config(seed=123)
    config2 = build_default_config(seed=123)

    objective1 = build_objective("sphere", dimensions=config1.dimensions)
    objective2 = build_objective("sphere", dimensions=config2.dimensions)

    optimizer1 = PSOOptimizer(config=config1, objective_function=objective1)
    optimizer2 = PSOOptimizer(config=config2, objective_function=objective2)

    result1 = optimizer1.optimize()
    result2 = optimizer2.optimize()

    assert np.allclose(result1.best_position, result2.best_position)
    assert np.isclose(result1.best_value, result2.best_value)
    assert result1.best_value_history == result2.best_value_history


def test_global_best_history_is_monotone_non_increasing():
    config = build_default_config()
    objective = build_objective("sphere", dimensions=config.dimensions)
    optimizer = PSOOptimizer(config=config, objective_function=objective)

    result = optimizer.optimize()
    history = result.best_value_history
    assert len(history) > 0

    for i in range(1, len(history)):
        assert history[i] <= history[i -1]

def test_optimizer_can_stop_early_by_tolerance():
    config = PSOConfig(
        num_particles=30,
        dimensions=2,
        max_iterations=100,
        inertia_weight=0.7,
        cognitive_coefficient=1.5,
        social_coefficient=1.5,
        seed=42,
        tolerance=1e-2,
        stagnation_patience=None,
        track_history=True,
    )

    objective = build_objective("sphere", dimensions=config.dimensions)
    optimizer = PSOOptimizer(config=config, objective_function=objective)

    result = optimizer.optimize()

    assert result.best_value <= 1e-2
    assert result.iterations_completed <= config.max_iterations