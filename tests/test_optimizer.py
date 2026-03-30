import numpy as np

from pso_lab.core.config import PSOConfig
from pso_lab.core.optimizer import PSOOptimizer
from pso_lab.objectives import build_objective


def test_optimizer_returns_valid_shapes():
    config = PSOConfig(
        num_particles=20,
        dimensions=2,
        max_iterations=50,
        inertia_weight=0.7,
        cognitive_coefficient=1.5,
        social_coefficent=1.5,
        seed=42,
    )

    objective = build_objective("sphere", dimensions=config.dimensions)
    optimizer = PSOOptimizer(config=config, objective_function=objective)

    best_position, best_value = optimizer.optimize()

    assert isinstance(best_position, np.ndarray)
    assert best_position.shape == (2,)
    assert isinstance(best_value, float)


def test_optimizer_improves_on_sphere():
    config = PSOConfig(
        num_particles=30,
        dimensions=2,
        max_iterations=100,
        inertia_weight=0.7,
        cognitive_coefficient=1.5,
        social_coefficent=1.5,
        seed=42,
    )

    objective = build_objective("sphere", dimensions=config.dimensions)
    optimizer = PSOOptimizer(config=config, objective_function=objective)

    best_position, best_value = optimizer.optimize()

    assert best_value < 1e-6


def test_optimizer_respects_bounds():
    config = PSOConfig(
        num_particles=20,
        dimensions=2,
        max_iterations=30,
        inertia_weight=0.7,
        cognitive_coefficient=1.5,
        social_coefficent=1.5,
        seed=42,
    )

    objective = build_objective("sphere", dimensions=config.dimensions)
    optimizer = PSOOptimizer(config=config, objective_function=objective)

    best_position, _ = optimizer.optimize()

    bounds = np.array(objective.bounds)
    lower = bounds[:, 0]
    upper = bounds[:, 1]

    assert np.all(best_position >= lower)
    assert np.all(best_position <= upper)