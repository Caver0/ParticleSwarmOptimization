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
