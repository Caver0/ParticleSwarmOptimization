from __future__ import annotations
from pso_lab.core.config import PSOConfig
from pso_lab.experiments.runner import run_single_experiment
from pso_lab.experiments.summary import summarize_experiments
from pso_lab.io.results import save_summary, save_result
from tabulate import tabulate
import numpy as np
def main() -> None: 
    all_summaries =[]
    objectives = ["sphere", "rosenbrock", "rastrigin", "ackley"]
    seeds = [0, 1, 2, 3, 4]
    for objective_name in objectives:
        print(f"\nRunning benchmarks for: {objective_name}")
        results = []
        for seed in seeds:
            config = PSOConfig(
                num_particles=30,
                dimensions=2, 
                max_iterations=100,
                inertia_weight=0.7, 
                cognitive_coefficient=1.5,
                social_coefficient=1.5,
                seed=seed,
                tolerance=0.0,
                stagnation_patience=None, 
                track_history=True,
            )

            result = run_single_experiment(objective_name=objective_name, config=config)
            results.append(result)
            save_result(
                output_path=f"results/{objective_name}/seed_{seed}.json",
                best_position=np.array(result.best_position, dtype=float),
                best_value=result.best_value,
                config=config,
                objective_name=result.objective_name,
                elapsed_time_s=result.elapsed_time_s,
                iterations_completed=result.iterations_completed,
                best_value_history=result.best_value_history,
            )
            print(
                f"Seed {seed} | Best value: {result.best_value:.6e} | Time: {result.elapsed_time_s:.6f} | Iterations: {result.iterations_completed}"
            )
        summary = summarize_experiments(results)
        all_summaries.append(summary)
        save_summary(
            output_path=f"results/{objective_name}_benchmark_summary.json",
            summary=summary,
        )
    global_table = []

    for s in all_summaries:
        global_table.append({
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

    print("\n=== GLOBAL BENCHMARK SUMMARY ===")
    print(tabulate(global_table, headers="keys", tablefmt="grid"))

if __name__ == "__main__":
    main()