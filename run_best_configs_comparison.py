from __future__ import annotations
import argparse
import numpy as np
from tabulate import tabulate

from pso_lab.core.config import PSOConfig
from pso_lab.experiments.runner import run_single_experiment
from pso_lab.experiments.summary import summarize_experiments
from pso_lab.io.results import save_result, save_summary


def main() -> None:
    evaluation_modes = ["sequential", "threading", "multiprocessing"]
    seeds = [0, 1, 2, 3, 4]

    best_configs = {
        "sphere": {"w": 0.4, "c1": 1.0, "c2": 1.0},
        "rosenbrock": {"w": 0.4, "c1": 1.5, "c2": 1.5},
        "rastrigin": {"w": 0.4, "c1": 1.5, "c2": 1.0},
        "ackley": {"w": 0.4, "c1": 1.5, "c2": 1.0},
    }

    all_summaries = []

    for objective_name, params in best_configs.items():
        print("\n========================================")
        print(f"COMPARING BEST CONFIG FOR: {objective_name}")
        print(
            f"w={params['w']:.1f}, c1={params['c1']:.1f}, c2={params['c2']:.1f}"
        )
        print("========================================")

        for evaluation_mode in evaluation_modes:
            results = []

            for seed in seeds:
                config = PSOConfig(
                    num_particles=30,
                    dimensions=2,
                    max_iterations=100,
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
                    max_workers=4 if evaluation_mode in {"threading", "multiprocessing"} else None,
                    batch_size=8 if evaluation_mode == "multiprocessing" else None,
                )

                results.append(result)

                save_result(
                    output_path=(
                        f"results/best_config_comparison/{evaluation_mode}/"
                        f"{objective_name}/seed_{seed}.json"
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

                print(
                    f"{objective_name} | {evaluation_mode} | seed={seed} | "
                    f"best={result.best_value:.6e} | "
                    f"time={result.elapsed_time_s:.6f} | "
                    f"iter={result.iterations_completed}"
                )

            summary = summarize_experiments(results)
            all_summaries.append((objective_name, params, evaluation_mode, summary))

            save_summary(
                output_path=(
                    f"results/best_config_comparison/{evaluation_mode}/"
                    f"{objective_name}_summary.json"
                ),
                summary=summary,
                evaluation_mode=evaluation_mode,
            )

    global_table = []

    for objective_name, params, evaluation_mode, summary in all_summaries:
        global_table.append(
            {
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

    print("\n=== BEST CONFIG COMPARISON SUMMARY ===")
    print(tabulate(global_table, headers="keys", tablefmt="grid"))


if __name__ == "__main__":
    main()