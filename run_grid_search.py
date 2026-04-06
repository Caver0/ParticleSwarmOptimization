from __future__ import annotations
from itertools import product
from tabulate import tabulate
import argparse
from pso_lab.core.config import PSOConfig
from pso_lab.experiments.runner import run_single_experiment
from pso_lab.experiments.summary import summarize_experiments
from pso_lab.io.results import save_summary

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PSO grid search over hyperparameters."
    )

    parser.add_argument(
        "--mode",
        default="sequential",
        choices=["sequential", "threading", "multiprocessing"],
        help="Evaluation mode used during grid search.",
    )
    parser.add_argument(
        "--dimensions",
        nargs="+",
        type=int,
        default=[2],
        help="Problem dimensions to evaluate.",
    )
    parser.add_argument(
        "--objectives",
        nargs="+",
        default=["sphere", "rosenbrock", "rastrigin", "ackley"],
        choices=["sphere", "rosenbrock", "rastrigin", "ackley"],
        help="Objective functions to include in the grid search.",
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
        "--inertia-values",
        nargs="+",
        type=float,
        default=[0.4, 0.7, 0.9],
        help="Grid values for inertia weight w.",
    )
    parser.add_argument(
        "--c1-values",
        nargs="+",
        type=float,
        default=[1.0, 1.5, 2.0],
        help="Grid values for cognitive coefficient c1.",
    )
    parser.add_argument(
        "--c2-values",
        nargs="+",
        type=float,
        default=[1.0, 1.5, 2.0],
        help="Grid values for social coefficient c2.",
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

def main() -> None:
    args = parse_args()

    evaluation_mode = args.mode
    objective_names = args.objectives
    dimensions = args.dimensions
    seeds = args.seeds

    inertia_values =args.inertia_values
    cognitive_values = args.c1_values
    social_values=  args.c2_values

    all_rows = []

    for dimension in dimensions:
        print("\n========================================")
        print(f"GRID SEARCH for dimension d={dimension}")
        print("========================================")
        for objective_name in objective_names:
            print("\n==============================")
            print(f"GRID SEARCH for: {objective_name}")
            print("==============================")

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
                        max_workers=args.max_workers if evaluation_mode in {"threading", "multiprocessing"} else None,
                        batch_size=args.batch_size if evaluation_mode == "multiprocessing" else None,
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

                print(
                    f"d = {dimension} | {objective_name} |"
                    f"w={inertia_weight:.1f}, "
                    f"c1={cognitive_coefficient:.1f}, "
                    f"c2={social_coefficient:.1f} | "
                    f"Mean Best={summary.mean_best_value:.6e} | "
                    f"Mean Time={summary.mean_elapsed_time_s:.6f}"
                )
            
    sorted_rows = sorted(
        all_rows,
        key=lambda row: (
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

    print("\n=== GRID SEARCH SUMMARY ===")
    print(tabulate(printable_rows, headers="keys", tablefmt="grid"))
    best_rows = []
    top3_rows = []
    for dimension in dimensions:
        for objective_name in objective_names:
            objective_rows = [row for row in sorted_rows if row["Objective"] == objective_name and row["Dimension"] == dimension]

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
                        "Rank": rank,
                        "w": f"{row['w']:.1f}",
                        "c1": f"{row['c1']:.1f}",
                        "c2": f"{row['c2']:.1f}",
                        "Mean Best": f"{row['Mean Best']:.6e}",
                        "Std Best": f"{row['Std Best']:.6e}",
                        "Mean Time (s)": f"{row['Mean Time (s)']:.6f}",
                    }
                )

    print("\n=== BEST CONFIGURATION PER OBJECTIVE AND DIMENSION ===")
    print(tabulate(best_rows, headers="keys", tablefmt="grid"))

    print("\n=== TOP 3 CONFIGURATIONS PER OBJECTIVE AND DIMENSION ===")
    print(tabulate(top3_rows, headers="keys", tablefmt="grid"))
if __name__ == "__main__":
    main()