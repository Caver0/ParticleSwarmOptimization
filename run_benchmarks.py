from pso_lab.core.config import PSOConfig
from pso_lab.experiments.runner import run_single_experiment
from pso_lab.experiments.summary import summarize_experiments
from pso_lab.io.results import save_summarry
from tabulate import tabulate

def main() -> None: 
    all_summaries =[]
    objectives = ["sphere", "rosenbrok", "rastrigin", "ackley"]
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
                social_coefficent=1.5,
                seed=seed,
            )

            result = run_single_experiment(objective_name=objective_name, config=config)
            results.append(result)

            print(
                f"Seed {seed} | Best value: {result.best_value:.6e}"
            )
        summary = summarize_experiments(results)
        all_summaries.append(summary)
        save_summarry(
            output_path=f"results/{objective_name}_benchmark_summary.json",
            summmary=summary,
        )
    global_table = []

    for s in all_summaries:
        global_table.append([
            s.objective_name,
            s.num_runs, 
            f"{s.mean_best_value:.6e}",
            f"{s.std_best_value:.6e}",
            f"{s.min_best_value:.6e}",
            f"{s.max_best_value:.6e}",
        ])
    headers = ["Objective", "Runs", "Mean", "Std", "Min", "Max"]

    print("\n=== GLOBAL BENCHMAR SUMMARY ===")
    print(tabulate(global_table, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    main()