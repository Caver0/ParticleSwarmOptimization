from __future__ import annotations
import sys
from collections.abc import Sequence
from itertools import product
from time import perf_counter

from tabulate import tabulate

from _repo_bootstrap import bootstrap_src_path

bootstrap_src_path()

from pso_lab.cli import parse_grid_search_args
from pso_lab.core.config import PSOConfig
from pso_lab.experiments.runner import run_single_experiment
from pso_lab.experiments.summary import summarize_experiments
from pso_lab.io.logging_utils import setup_logger
from pso_lab.io.results import save_summary

def main(argv: Sequence[str] | None = None) -> None:
    args = parse_grid_search_args(argv)
    logger = setup_logger("pso_grid_search")
    evaluation_modes = ["v0", "v1", "v2"] if args.mode == "all" else [args.mode]
    objective_names = args.objectives
    dimensions = args.dimensions
    seeds = args.seeds

    inertia_values =args.inertia_values
    cognitive_values = args.c1_values
    social_values=  args.c2_values

    all_rows = []
    mode_elapsed_times = {}
    total_start = perf_counter()

    for evaluation_mode in evaluation_modes:
        mode_start = perf_counter()
        logger.info("Grid search for mode=%s", evaluation_mode)
        for dimension in dimensions:
            logger.info("Grid search for dimension d=%d", dimension)
            for objective_name in objective_names:
                logger.info("Grid search for objective=%s", objective_name)
                for inertia_weight, cognitive_coefficient, social_coefficient in product(
                    inertia_values,
                    cognitive_values,
                    social_values,
                ):
                    results = []
                    for seed in seeds:
                        config = PSOConfig(
                            num_particles=args.particles,
                            dimensions=dimension,
                            max_iterations=args.iterations,
                            inertia_weight=inertia_weight,
                            cognitive_coefficient=cognitive_coefficient,
                            social_coefficient=social_coefficient,
                            seed=seed,
                            tolerance=0.0,
                            stagnation_patience=None,
                            track_history=True,
                        )
                        result = run_single_experiment(
                            objective_name=objective_name,
                            config=config,
                            evaluation_mode=evaluation_mode,
                            max_workers=args.max_workers if evaluation_mode in {"v1", "v2", "threading", "multiprocessing"} else None,
                            batch_size=args.batch_size if evaluation_mode in {"v2", "multiprocessing"} else None,
                        )
                        results.append(result)

                    summary = summarize_experiments(results)
                    save_summary(
                        output_path=(
                            "results/grid_search/"
                            f"{evaluation_mode}/d{dimension}/{objective_name}/"
                            f"w_{inertia_weight}_c1_{cognitive_coefficient}_c2_{social_coefficient}.json"
                        ),
                        summary=summary,
                        evaluation_mode=evaluation_mode,
                    )

                    row = {
                        "Dimension": dimension,
                        "Objective": objective_name,
                        "Mode": evaluation_mode,
                        "w": inertia_weight,
                        "c1": cognitive_coefficient,
                        "c2": social_coefficient,
                        "Mean Best": summary.mean_best_value,
                        "Std Best": summary.std_best_value,
                        "Mean Time (s)": summary.mean_elapsed_time_s,
                        "Mean Iter": summary.mean_iterations,
                    }
                    all_rows.append(row)

                    logger.info(
                        "mode=%s | d=%d | objective=%s | w=%.1f | c1=%.1f | c2=%.1f | mean_best=%.6e | mean_time=%.6f",
                        evaluation_mode,
                        dimension,
                        objective_name,
                        inertia_weight,
                        cognitive_coefficient,
                        social_coefficient,
                        summary.mean_best_value,
                        summary.mean_elapsed_time_s,
                    )
        mode_elapsed_times[evaluation_mode] = perf_counter() - mode_start

    total_elapsed_time = perf_counter() - total_start

            
    sorted_rows = sorted(
        all_rows,
        key=lambda row: (
            row["Mode"],
            row["Dimension"],
            row["Objective"],
            row["Mean Best"],
            row["Mean Time (s)"],
        ),
    )

    printable_rows = []
    for row in sorted_rows:
        printable_rows.append(
            {
                "Dimension": row["Dimension"],
                "Objective": row["Objective"],
                "Mode": row["Mode"],
                "w": f"{row['w']:.1f}",
                "c1": f"{row['c1']:.1f}",
                "c2": f"{row['c2']:.1f}",
                "Mean Best": f"{row['Mean Best']:.6e}",
                "Std Best": f"{row['Std Best']:.6e}",
                "Mean Time (s)": f"{row['Mean Time (s)']:.6f}",
                "Mean Iter": f"{row['Mean Iter']:.1f}",
            }
        )

    grid_summary_table = tabulate(printable_rows, headers="keys", tablefmt="grid")
    print("\n=== GRID SEARCH SUMMARY ===")
    print(grid_summary_table)
    logger.info("GRID SEARCH SUMMARY printed")
    best_rows = []
    top3_rows = []
    for evaluation_mode in evaluation_modes:
        for dimension in dimensions:
            for objective_name in objective_names:
                objective_rows = [
                    row for row in sorted_rows
                    if row["Objective"] == objective_name
                    and row["Dimension"] == dimension
                    and row["Mode"] == evaluation_mode
                ]

                if not objective_rows:
                    continue

                best = objective_rows[0]
                best_rows.append(
                    {
                        "Dimension": best["Dimension"],
                        "Objective": best["Objective"],
                        "Mode": best["Mode"],
                        "Best w": f"{best['w']:.1f}",
                        "Best c1": f"{best['c1']:.1f}",
                        "Best c2": f"{best['c2']:.1f}",
                        "Mean Best": f"{best['Mean Best']:.6e}",
                        "Std Best": f"{best['Std Best']:.6e}",
                        "Mean Time (s)": f"{best['Mean Time (s)']:.6f}",
                        "Mean Iter": f"{best['Mean Iter']:.1f}",
                    }
                )

                for rank, row in enumerate(objective_rows[:3], start=1):
                    top3_rows.append(
                        {
                            "Dimension": row["Dimension"],
                            "Objective": row["Objective"],
                            "Mode": row["Mode"],
                            "Rank": rank,
                            "w": f"{row['w']:.1f}",
                            "c1": f"{row['c1']:.1f}",
                            "c2": f"{row['c2']:.1f}",
                            "Mean Best": f"{row['Mean Best']:.6e}",
                            "Std Best": f"{row['Std Best']:.6e}",
                            "Mean Time (s)": f"{row['Mean Time (s)']:.6f}",
                        }
                    )
    logger.info("Grid search summary generated")
    best_config_table = tabulate(best_rows, headers="keys", tablefmt="grid")
    print("\n=== BEST CONFIGURATION PER MODE, OBJECTIVE AND DIMENSION ===")
    print(best_config_table)
    logger.info("BEST CONFIGURATION PER MODE, OBJECTIVE AND DIMENSION printed")

    top3_table = tabulate(top3_rows, headers="keys", tablefmt="grid")
    print("\n=== TOP 3 CONFIGURATIONS PER MODE, OBJECTIVE AND DIMENSION ===")
    print(top3_table)
    logger.info("TOP 3 CONFIGURATIONS PER MODE, OBJECTIVE AND DIMENSION printed")

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
    print("\n=== GRID SEARCH EXECUTION TIMES ===")
    print(runtime_table)
    logger.info("GRID SEARCH EXECUTION TIMES printed")


if __name__ == "__main__":
    # Edit these values and press Run in VS Code.
    vscode_argv = [
        "--mode", "all",
        "--dimensions", "2",
        "--objectives", "sphere", "rosenbrock", "rastrigin", "ackley",
        "--seeds", "0", "1", "2", "3", "4",
        "--particles", "30",
        "--iterations", "100",
        "--inertia-values", "0.4", "0.7", "0.9",
        "--c1-values", "1.0", "1.5", "2.0",
        "--c2-values", "1.0", "1.5", "2.0",
        "--max-workers", "4",
        "--batch-size", "8",
    ]
    main(sys.argv[1:] or vscode_argv)
