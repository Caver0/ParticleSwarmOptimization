from __future__ import annotations
from time import perf_counter
from pso_lab.core.config import PSOConfig
from pso_lab.core.optimizer import PSOOptimizer
from pso_lab.io.logging_utils import setup_logger
from pso_lab.objectives import build_objective
from pso_lab.io.results import save_result

def main() -> None:
    logger = setup_logger("pso_run")
    config = PSOConfig(
        num_particles=30,
        dimensions=2,
        max_iterations=100,
        inertia_weight=0.7,
        cognitive_coefficient=1.5,
        social_coefficient=1.5,
        seed=42,
        tolerance=1e-8,
        stagnation_patience=None,
        track_history=True,
    )

    objective = build_objective("sphere", dimensions=config.dimensions)

    optimizer = PSOOptimizer(config= config, objective_function=objective)

    start = perf_counter()
    result = optimizer.optimize()
    elapsed_time_s = perf_counter() - start

    output_path = "results/sphere_run.json"

    save_result(
        output_path = output_path,
        best_position = result.best_position,
        best_value = result.best_value,
        config = config,
        objective_name = objective.name,
        elapsed_time_s=elapsed_time_s,
        iterations_completed=result.iterations_completed,
        best_value_history=result.best_value_history,
        swarm_position_history=result.swarm_position_history,
    )

    logger.info("Optimization finished")
    logger.info("Objective: %s", objective.name)
    logger.info("Best position: %s", result.best_position.tolist())
    logger.info("Best value: %.6e", result.best_value)
    logger.info("Iterations completed: %d", result.iterations_completed)
    logger.info("Elapsed time (s): %.6f", elapsed_time_s)
    logger.info("Result saved to %s", output_path)




if __name__ == "__main__":
    main()
