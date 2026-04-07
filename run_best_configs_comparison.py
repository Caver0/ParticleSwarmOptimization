from __future__ import annotations
import sys
from collections.abc import Sequence
from time import perf_counter

import numpy as np
from tabulate import tabulate

from _repo_bootstrap import bootstrap_src_path

bootstrap_src_path()

from pso_lab.cli import parse_best_configs_comparison_args
from pso_lab.core.config import PSOConfig
from pso_lab.experiments.runner import run_single_experiment
from pso_lab.experiments.summary import summarize_experiments
from pso_lab.io.results import save_result, save_summary
from pso_lab.io.logging_utils import setup_logger


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

def main(argv: Sequence[str] | None = None) -> None:
    args = parse_best_configs_comparison_args(argv)
    logger = setup_logger("pso_best_config_comparison")
    evaluation_modes = args.modes
    seeds = args.seeds
    dimensions = args.dimensions
    objective_names = args.objectives

    best_configs = get_best_configs()
    all_summaries = []
    mode_elapsed_times = {mode: 0.0 for mode in evaluation_modes}
    total_start = perf_counter()
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
                mode_start = perf_counter()
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
                        swarm_position_history=result.swarm_position_history,
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
                mode_elapsed_times[evaluation_mode] += perf_counter() - mode_start
                print("\n")
    total_elapsed_time = perf_counter() - total_start
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
    comparison_table = tabulate(global_table, headers="keys", tablefmt="grid")
    print("\n=== BEST CONFIG COMPARISON SUMMARY ===")
    print(comparison_table)
    logger.info("BEST CONFIG COMPARISON SUMMARY printed")

    runtime_rows = [
        {
            "Mode": mode,
            "Total Time (s)": f"{elapsed_time:.6f}",
        }
        for mode, elapsed_time in mode_elapsed_times.items()
    ]
    runtime_rows.append(
        {
            "Mode": "total",
            "Total Time (s)": f"{total_elapsed_time:.6f}",
        }
    )

    runtime_table = tabulate(runtime_rows, headers="keys", tablefmt="grid")
    print("\n=== BEST CONFIG COMPARISON EXECUTION TIMES ===")
    print(runtime_table)
    logger.info("BEST CONFIG COMPARISON EXECUTION TIMES printed")


if __name__ == "__main__":
    # Edit these values and press Run in VS Code.
    vscode_argv = [
        "--modes", "sequential", "threading", "multiprocessing",
        "--dimensions", "2",
        "--objectives", "sphere", "rosenbrock", "rastrigin", "ackley",
        "--seeds", "0", "1", "2", "3", "4",
        "--particles", "30",
        "--iterations", "100",
        "--max-workers", "4",
        "--batch-size", "8",
    ]
    main(sys.argv[1:] or vscode_argv)
