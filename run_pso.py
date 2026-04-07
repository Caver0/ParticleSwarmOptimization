from __future__ import annotations
import sys
from collections.abc import Sequence
from time import perf_counter

from _repo_bootstrap import bootstrap_src_path

bootstrap_src_path()

from pso_lab.cli import parse_single_run_args
from pso_lab.core.config import PSOConfig
from pso_lab.core.optimizer import PSOOptimizer
from pso_lab.io.logging_utils import setup_logger
from pso_lab.io.results import save_result
from pso_lab.objectives import build_objective

def main(argv: Sequence[str] | None = None) -> None:
    logger = setup_logger("pso_run")
    args = parse_single_run_args(argv)
    config = PSOConfig(
        num_particles=args.particles,
        dimensions=args.dimension,
        max_iterations=args.iterations,
        inertia_weight=args.inertia,
        cognitive_coefficient=args.c1,
        social_coefficient=args.c2,
        seed=args.seed,
        tolerance=args.tolerance,
        stagnation_patience=None,
        track_history=True,
    )

    objective = build_objective(args.objective, dimensions=config.dimensions)

    optimizer = PSOOptimizer(config=config, objective_function=objective)

    start = perf_counter()
    result = optimizer.optimize()
    elapsed_time_s = perf_counter() - start

    output_path = args.output_path

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
    # Edit these values and press Run in VS Code.
    vscode_argv = [
        "--objective", "sphere",
        "--dimension", "2",
        "--particles", "30",
        "--iterations", "100",
        "--inertia", "0.7",
        "--c1", "1.5",
        "--c2", "1.5",
        "--seed", "42",
        "--tolerance", "1e-8",
        "--output-path", "results/sphere_run.json",
    ]
    main(sys.argv[1:] or vscode_argv)
