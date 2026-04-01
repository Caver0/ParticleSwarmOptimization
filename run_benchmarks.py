from pso_lab.core.config import PSOConfig
from pso_lab.experiments.runner import run_single_experiment
from pso_lab.experiments.summary import summarize_experiments

def main() -> None:
    objective_name = "sphere"
    seeds = [0, 1, 2, 3, 4]

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
            f"Seed {seed} | Objective: {result.objective_name} | "
            f"Best value: {result.best_value:.6e}"
        )
    summary = summarize_experiments(results)

    print("\nBenchmark summary:")
    print(f"Objective: {summary.objective_name}")
    print(f"Runs: {summary.num_runs}")
    print(f"Mean best value: {summary.mean_best_value:.6e}")
    print(f"Std best value: {summary.std_best_value:.6e}")
    print(f"Min best value: {summary.min_best_value:.6e}")
    print(f"Max best value: {summary.max_best_value:.6e}")


if __name__ == "__main__":
    main()