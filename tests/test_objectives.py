import numpy as np

from pso_lab.objectives import build_objective


def test_rastrigin_is_zero_at_global_optimum() -> None:
    objective = build_objective("rastrigin", dimensions=2)

    assert objective(np.zeros(2, dtype=float)) == 0.0


def test_rastrigin_stays_positive_for_tiny_non_zero_inputs() -> None:
    objective = build_objective("rastrigin", dimensions=2)
    x = np.array([3.6213239405120304e-09, 7.695237604505083e-10], dtype=float)

    value = objective(x)

    assert value > 0.0
    assert value < 1e-12


def test_benchmark_objectives_evaluate_batch_matches_individual_evaluation() -> None:
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, -2.0, 0.5],
            [-1.5, 0.25, 3.0],
        ],
        dtype=float,
    )

    for objective_name in ("sphere", "rosenbrock", "rastrigin", "ackley"):
        objective = build_objective(objective_name, dimensions=3)
        individual = np.array([objective(position) for position in positions], dtype=float)
        batch = objective.evaluate_batch(positions)

        np.testing.assert_allclose(batch, individual)
