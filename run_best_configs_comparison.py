from __future__ import annotations
import argparse
import numpy as np
from tabulate import tabulate

from pso_lab.core.config import PSOConfig
from pso_lab.experiments.runner import run_single_experiment
from pso_lab.experiments.summary import summarize_experiments
from pso_lab.io.results import save_result, save_summary
from pso_lab.io.logging_utils import setup_logger

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the best PSO configurations across evaluation modes."
    )

    parser.add_argument(
        "--modes",
        nargs="+",
        default=["sequential", "threading", "multiprocessing"],
        choices=["sequential", "threading", "multiprocessing"],
        help="Evaluation modes to compare.",
    )
    parser.add_argument(
        "--dimensions",
        nargs="+",
        type=int,
        default=[2],
        help="Problem dimensions to compare.",
    )
    parser.add_argument(
        "--objectives",
        nargs="+",
        default=["sphere", "rosenbrock", "rastrigin", "ackley"],
        choices=["sphere", "rosenbrock", "rastrigin", "ackley"],
        help="Objective functions to compare.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="Random seeds for reproducible runs.",
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
        "--max-workers",
        type=int,
        default=4,
        help="Maximum workers for threading/multiprocessing evaluators.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for multiprocessing evaluator.",
    )

    return parser.parse_args()


def get_best_configs() -> dict[int, dict[str, dict[str, float]]]:
    return {
        2: {
            "sphere": {"w": 0.4, "c1": 1.0, "c2": 1.0},
            "rosenbrock": {"w": 0.4, "c1": 1.5, "c2": 1.5},
            "rastrigin": {"w": 0.4, "c1": 1.5, "c2": 1.0},
            "ackley": {"w": 0.4, "c1": 1.0, "c2": 1.0},
        },
        10: {
            "sphere": {"w": 0.4, "c1": 1.5, "c2": 2.0},
            "rosenbrock": {"w": 0.7, "c1": 1.5, "c2": 1.5},
            "rastrigin": {"w": 0.7, "c1": 1.5, "c2": 1.5},
            "ackley": {"w": 0.7, "c1": 2.0, "c2": 1.0},
        },
        30: {
            "sphere": {"w": 0.7, "c1": 2.0, "c2": 1.0},
            "rosenbrock": {"w": 0.7, "c1": 2.0, "c2": 1.0},
            "rastrigin": {"w": 0.4, "c1": 2.0, "c2": 1.0},
            "ackley": {"w": 0.7, "c1": 2.0, "c2": 1.0},
        },
    }

def main() -> None:
    args = parse_args()
    logger = setup_logger("pso_best_config_comparison")
    evaluation_modes = args.modes
    seeds = args.seeds
    dimensions = args.dimensions
    objective_names = args.objectives

    best_configs = get_best_configs()
    all_summaries = []
    for dimension in dimensions:
        if dimension not in best_configs:
            logger.warning("Skipping dimension %d - no best configuration registrated.", dimension)
            continue
        for objective_name in objective_names:
            if objective_name not in best_configs[dimension]:
                logger.warning("Skipping objective=%s for dimension=%d - no best configuration registred", objective_name, dimension)
                continue
            params = best_configs[dimension][objective_name]
            logger.info(
                "Comparing best config | d=%d | objective=%s | w=%.1f | c1=%.1f | c2=%.1f",
                dimension,
                objective_name,
                params["w"],
                params["c1"],
                params["c2"],
            )
            for evaluation_mode in evaluation_modes:
                results = []

                for seed in seeds:
                    config = PSOConfig(
                        num_particles=args.particles,
                        dimensions=dimension,
                        max_iterations=args.iterations,
                        inertia_weight=params["w"],
                        cognitive_coefficient=params["c1"],
                        social_coefficient=params["c2"],
                        seed=seed,
                        tolerance=0.0,
                        stagnation_patience=None,
                        track_history=True,
                    )

                    result = run_single_experiment(
                        objective_name=objective_name,
                        config=config,
                        evaluation_mode=evaluation_mode,
                        max_workers=args.max_workers if evaluation_mode in {"threading", "multiprocessing"} else None,
                        batch_size=args.batch_size if evaluation_mode == "multiprocessing" else None,
                    )

                    results.append(result)

                    save_result(
                        output_path=(
                            f"results/best_config_comparison/{evaluation_mode}/"
                            f"d{dimension}/{objective_name}/seed_{seed}.json"
                        ),
                        best_position=np.array(result.best_position, dtype=float),
                        best_value=result.best_value,
                        config=config,
                        objective_name=result.objective_name,
                        evaluation_mode=result.evaluation_mode,
                        elapsed_time_s=result.elapsed_time_s,
                        iterations_completed=result.iterations_completed,
                        best_value_history=result.best_value_history,
                        timing_stats=result.timing_stats,
                    )
                    logger.info(
                        "result | d=%d | objective=%s | mode=%s | seed=%d | best=%.6e | time=%.6f | iterations=%d",
                        dimension,
                        objective_name,
                        evaluation_mode,
                        seed,
                        result.best_value,
                        result.elapsed_time_s,
                        result.iterations_completed,
                    )
                summary = summarize_experiments(results)
                all_summaries.append((dimension, objective_name, params, evaluation_mode, summary))

                save_summary(
                    output_path=(
                        f"results/best_config_comparison/{evaluation_mode}/"
                        f"d{dimension}/{objective_name}_summary.json"
                    ),
                    summary=summary,
                    evaluation_mode=evaluation_mode,
                )
                logger.info(
                    "summary | d=%d | objective=%s | mode=%s | mean_best=%.6e | mean_time=%.6f",
                    dimension,
                    objective_name,
                    evaluation_mode,
                    summary.mean_best_value,
                    summary.mean_elapsed_time_s,
                )
                print("\n")
    global_table = []

    for dimension, objective_name, params, evaluation_mode, summary in all_summaries:
        global_table.append(
            {
                "Dimension": dimension,
                "Objective": objective_name,
                "Mode": evaluation_mode,
                "w": f"{params['w']:.1f}",
                "c1": f"{params['c1']:.1f}",
                "c2": f"{params['c2']:.1f}",
                "Runs": summary.num_runs,
                "Mean Best": f"{summary.mean_best_value:.6e}",
                "Std Best": f"{summary.std_best_value:.6e}",
                "Min Best": f"{summary.min_best_value:.6e}",
                "Max Best": f"{summary.max_best_value:.6e}",
                "Mean Time (s)": f"{summary.mean_elapsed_time_s:.6f}",
                "Std Time (s)": f"{summary.std_elapsed_time_s:.6f}",
                "Mean Iter": f"{summary.mean_iterations:.1f}",
            }
        )
    logger.info("Best config comparison summary generated")
    print("\n=== BEST CONFIG COMPARISON SUMMARY ===")
    print(tabulate(global_table, headers="keys", tablefmt="grid"))


if __name__ == "__main__":
    main()