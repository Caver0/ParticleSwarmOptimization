from __future__ import annotations

from pso_lab.core.config import PSOConfig
from pso_lab.core.optimizer import PSOOptimizer
from pso_lab.objectives import build_objective
from pso_lab.io.results import save_result

def main() -> None:
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

    optimizer = PSOOptimizer(config= config, objective_function=objective)

    best_position, best_value = optimizer.optimize()

    save_result(
        output_path = "results/sphere_run.json",
        best_position = best_position,
        best_value = best_value,
        config = config,
        objective_name = objective.name,
    )

    print("Optimization finished")
    print(f"Objective: {objective.name}")
    print(f"Best position: {best_position}")
    print(f"Best value: {best_value}")



if __name__ == "__main__":
    main()