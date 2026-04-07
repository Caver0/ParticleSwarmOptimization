from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tabulate import tabulate

from pso_lab.core.config import PSOConfig
from pso_lab.experiments.runner import run_single_experiment
from pso_lab.io.logging_utils import setup_logger
from pso_lab.io.results import save_result
from pso_lab.viz.plots import save_particle_motion_plot


METHOD_TO_MODE = {
    "v1": "sequential",
    "v2": "threading",
    "v3": "multiprocessing",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 2D particle-motion plots for 3D PSO runs."
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["v1", "v2", "v3"],
        choices=["v1", "v2", "v3"],
        help="Visualization method folders to generate.",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=3,
        help="Problem dimension for the visualization run.",
    )
    parser.add_argument(
        "--objectives",
        nargs="+",
        default=["sphere", "rosenbrock", "rastrigin", "ackley"],
        choices=["sphere", "rosenbrock", "rastrigin", "ackley"],
        help="Objective functions to visualize.",
    )
    parser.add_argument(
        "--particles",
        type=int,
        default=30,
        help="Number of particles in the swarm.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Maximum number of PSO iterations.",
    )
    parser.add_argument(
        "--inertia",
        type=float,
        default=0.7,
        help="Inertia weight (w).",
    )
    parser.add_argument(
        "--c1",
        type=float,
        default=1.5,
        help="Cognitive coefficient (c1).",
    )
    parser.add_argument(
        "--c2",
        type=float,
        default=1.5,
        help="Social coefficient (c2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used in the visualization run.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum workers for threading and multiprocessing evaluators.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for the multiprocessing evaluator.",
    )
    parser.add_argument(
        "--results-dir",
        default="results/visualization",
        help="Directory where raw visualization runs will be stored.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/plots",
        help="Directory where plot images will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger("pso_visualizations")

    if args.dimension < 3:
        raise ValueError("Particle-motion visualizations require dimension >= 3.")

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    summary_rows: list[dict[str, str | float | int]] = []

    for method_name in args.methods:
        evaluation_mode = METHOD_TO_MODE[method_name]
        method_results_dir = results_dir / method_name
        method_output_dir = output_dir / method_name

        for objective_name in args.objectives:
            logger.info(
                "Generating particle-motion plot | method=%s | objective=%s | d=%d | mode=%s | seed=%d",
                method_name,
                objective_name,
                args.dimension,
                evaluation_mode,
                args.seed,
            )

            config = PSOConfig(
                num_particles=args.particles,
                dimensions=args.dimension,
                max_iterations=args.iterations,
                inertia_weight=args.inertia,
                cognitive_coefficient=args.c1,
                social_coefficient=args.c2,
                seed=args.seed,
                tolerance=0.0,
                stagnation_patience=None,
                track_history=True,
                track_swarm_history=True,
            )

            result = run_single_experiment(
                objective_name=objective_name,
                config=config,
                evaluation_mode=evaluation_mode,
                max_workers=args.max_workers if evaluation_mode in {"threading", "multiprocessing"} else None,
                batch_size=args.batch_size if evaluation_mode == "multiprocessing" else None,
            )

            result_path = method_results_dir / f"d{args.dimension}" / objective_name / f"seed_{args.seed}.json"
            save_result(
                output_path=result_path,
                best_position=np.array(result.best_position, dtype=float),
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

            plot_path = save_particle_motion_plot(
                swarm_position_history=result.swarm_position_history,
                objective_name=result.objective_name,
                output_dir=method_output_dir,
                dimensions=config.dimensions,
                evaluation_mode=result.evaluation_mode,
                method_label=method_name,
                seed=config.seed,
            )

            logger.info(
                "Visualization generated | method=%s | objective=%s | best=%.6e | file=%s",
                method_name,
                objective_name,
                result.best_value,
                plot_path,
            )

            summary_rows.append(
                {
                    "Method": method_name,
                    "Mode": evaluation_mode,
                    "Objective": result.objective_name,
                    "Dimension": config.dimensions,
                    "Seed": config.seed,
                    "Best Value": f"{result.best_value:.6e}",
                    "Iterations": result.iterations_completed,
                    "Plot": str(plot_path) if plot_path is not None else "not generated",
                }
            )

    print("\n=== PARTICLE MOTION VISUALIZATIONS ===")
    print(tabulate(summary_rows, headers="keys", tablefmt="grid"))


if __name__ == "__main__":
    main()
