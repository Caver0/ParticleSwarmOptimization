from __future__ import annotations
import argparse
from pso_lab.core.config import PSOConfig
from pso_lab.experiments.runner import run_single_experiment
from pso_lab.experiments.summary import summarize_experiments
from pso_lab.io.results import save_summary, save_result
from pso_lab.io.logging_utils import setup_logger
from tabulate import tabulate
import numpy as np

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PSO benchmarks across evaluation modes, dimensions and objectives.")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["sequential", "threading", "multiprocessing"],
        choices=["sequential", "threading", "multiprocessing"],
        help="Evaluation modes to benchmark",

    )
    parser.add_argument(
        "--dimensions",
        nargs="+",
        type=int,
        default=[2, 10, 30],
        help="Dimensions to benchmark",
    )
    parser.add_argument(
        "--objectives",
        nargs="+",
        default=["sphere", "rosenbrock", "rastrigin", "ackley"],
        choices=["sphere", "rosenbrock", "rastrigin", "ackley"],
        help="Objective functions to benchmark",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="Random seeds for benchmarking",
    )
    parser.add_argument(
        "--particles",
        type=int,
        default=30,
        help="Number of particles in the swarm",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Maximum number of iterations",
    )
    parser.add_argument(
        "--inertia",
        type=float,
        default=0.7,
        help="Inertia weight (w)",
    )
    parser.add_argument(
        "--c1",
        type=float,
        default=1.5,
        help="Cognitive coefficient (c1)",
    )
    parser.add_argument(
        "--c2",
        type=float,
        default=1.5,
        help="Social coefficient (c2)",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Max workers for parallel modes (threading and multiprocessing)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for multiprocessing evaluator",
    )
    return parser.parse_args()
def main() -> None: 
    logger = setup_logger("pso_benchmarks")
    args = parse_args()
    all_summaries =[]
    evaluation_modes = args.modes
    dimensions = args.dimensions
    objectives = args.objectives
    seeds = args.seeds
    for evaluation_mode in evaluation_modes:
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
    print("\n=== GLOBAL BENCHMARK SUMMARY ===")
    print(tabulate(global_table, headers="keys", tablefmt="grid"))
    

if __name__ == "__main__":
    main()