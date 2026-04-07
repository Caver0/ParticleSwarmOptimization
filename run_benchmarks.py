from __future__ import annotations
from time import perf_counter
import numpy as np
from tabulate import tabulate

from pso_lab.cli import parse_benchmarks_args
from pso_lab.core.config import PSOConfig
from pso_lab.experiments.runner import run_single_experiment
from pso_lab.experiments.summary import summarize_experiments
from pso_lab.io.results import save_summary, save_result
from pso_lab.io.logging_utils import setup_logger



def main() -> None: 
    logger = setup_logger("pso_benchmarks")
    args = parse_benchmarks_args()
    all_summaries =[]
    mode_elapsed_times = {}
    total_start = perf_counter()
    evaluation_modes = args.modes
    dimensions = args.dimensions
    objectives = args.objectives
    seeds = args.seeds
    for evaluation_mode in evaluation_modes:
        mode_start = perf_counter()
        logger.info("Running mode=%s", evaluation_mode)
        for dimension in dimensions:
            logger.info("Running dimension d=%d", dimension)
            for objective_name in objectives:
                
                logger.info("Running benchmarks for objective=%s", objective_name)
                results = []
                for seed in seeds:
                    config = PSOConfig(
                        num_particles=args.particles,
                        dimensions=dimension, 
                        max_iterations=args.iterations,
                        inertia_weight=args.inertia, 
                        cognitive_coefficient=args.c1,
                        social_coefficient=args.c2,
                        seed=seed,
                        tolerance=0.0,
                        stagnation_patience=None, 
                        track_history=True,
                    )

                    result = run_single_experiment(objective_name=objective_name, 
                                                config=config, 
                                                evaluation_mode=evaluation_mode, 
                                                max_workers=args.max_workers if evaluation_mode in {"threading", "multiprocessing"} else None,
                                                batch_size=args.batch_size if evaluation_mode == "multiprocessing" else None,
                                                )
                    results.append(result)
                    save_result(
                        output_path=f"results/{evaluation_mode}/d{dimension}/{objective_name}/seed_{seed}.json",
                        best_position=np.array(result.best_position, dtype=float),
                        best_value=result.best_value,
                        config=config,
                        objective_name=result.objective_name,
                        elapsed_time_s=result.elapsed_time_s,
                        iterations_completed=result.iterations_completed,
                        best_value_history=result.best_value_history,
                        swarm_position_history=result.swarm_position_history,
                        timing_stats=result.timing_stats,
                    )
                    logger.info(
                        "mode=%s | d=%d  | objective=%s | seed=%d | best=%.6e | time=%.6f | iterations=%d",
                        evaluation_mode,
                        dimension,
                        objective_name,
                        seed,
                        result.best_value,
                        result.elapsed_time_s,
                        result.iterations_completed,
                    )
                summary = summarize_experiments(results)
                all_summaries.append((evaluation_mode, dimension, summary))
                save_summary(
                    output_path=f"results/{evaluation_mode}/d{dimension}/{objective_name}_benchmark_summary.json",
                    summary=summary,
                    evaluation_mode = evaluation_mode,
                )
                logger.info(
                    "summary | mode=%s | d=%d | objective=%s | mean_best=%.6e | mean_time=%.6f",
                    evaluation_mode,
                    dimension,
                    objective_name,
                    summary.mean_best_value,
                    summary.mean_elapsed_time_s,
                )
        mode_elapsed_times[evaluation_mode] = perf_counter() - mode_start

    total_elapsed_time = perf_counter() - total_start

    global_table = []

    for evaluation_mode, dimension, s in all_summaries:
        global_table.append({
            "Mode": evaluation_mode,
            "Dimension": dimension,
            "Objective": s.objective_name,
            "Runs": s.num_runs,
            "Mean Best": f"{s.mean_best_value:.6e}",
            "Std Best": f"{s.std_best_value:.6e}",
            "Min Best": f"{s.min_best_value:.6e}",
            "Max Best": f"{s.max_best_value:.6e}",
            "Mean Time (s)": f"{s.mean_elapsed_time_s:.6f}",
            "Std Time (s)": f"{s.std_elapsed_time_s:.6f}",
            "Mean Iter": f"{s.mean_iterations:.1f}",
            "Min Iter": s.min_iterations,
            "Max Iter": s.max_iterations,
        })

    logger.info("Global benchmark summary generated")
    global_summary_table = tabulate(global_table, headers="keys", tablefmt="grid")
    print("\n=== GLOBAL BENCHMARK SUMMARY ===")
    print(global_summary_table)
    logger.info("GLOBAL BENCHMARK SUMMARY printed")

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
    print("\n=== BENCHMARK EXECUTION TIMES ===")
    print(runtime_table)
    logger.info("BENCHMARK EXECUTION TIMES printed")
    

if __name__ == "__main__":
    main()
