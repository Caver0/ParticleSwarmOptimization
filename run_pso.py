from __future__ import annotations

import sys
from collections.abc import Sequence

import numpy as np

from _repo_bootstrap import bootstrap_src_path

bootstrap_src_path()

from pso_lab.cli import parse_single_run_args
from pso_lab.core.config import PSOConfig
from pso_lab.experiments.runner import run_single_experiment
from pso_lab.io.logging_utils import setup_logger
from pso_lab.io.results import save_result


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

    result = run_single_experiment(
        objective_name=args.objective,
        config=config,
        evaluation_mode=args.mode,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        async_min_delay=args.async_min_delay,
        async_max_delay=args.async_max_delay,
    )

    output_path = args.output_path
    save_result(
        output_path=output_path,
        best_position=np.asarray(result.best_position, dtype=float),
        best_value=result.best_value,
        config=config,
        objective_name=result.objective_name,
        evaluation_mode=result.evaluation_mode,
        elapsed_time_s=result.elapsed_time_s,
        iterations_completed=result.iterations_completed,
        best_value_history=result.best_value_history,
        swarm_position_history=result.swarm_position_history,
        timing_stats=result.timing_stats,
    )

    logger.info("Optimization finished")
    logger.info("Objective: %s", result.objective_name)
    logger.info("Evaluation mode: %s", result.evaluation_mode)
    if args.mode == "asyncio":
        logger.info(
            "Async delay window (s): min=%.6f | max=%.6f",
            args.async_min_delay,
            args.async_max_delay,
        )
    logger.info("Best position: %s", result.best_position)
    logger.info("Best value: %.6e", result.best_value)
    logger.info("Iterations completed: %d", result.iterations_completed)
    logger.info("Elapsed time (s): %.6f", result.elapsed_time_s)
    logger.info("Result saved to %s", output_path)


if __name__ == "__main__":
    # Edit these values and press Run in VS Code.
    vscode_argv = [
        "--mode", "sequential",
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
        "--max-workers", "4",
        "--batch-size", "8",
        "--async-min-delay", "0.0",
        "--async-max-delay", "0.0",
    ]
    main(sys.argv[1:] or vscode_argv)
