from __future__ import annotations
import sys
from collections.abc import Sequence
from time import perf_counter

import numpy as np
from tabulate import tabulate

from _repo_bootstrap import bootstrap_src_path

bootstrap_src_path()

from pso_lab.cli import parse_pyswarm_baseline_args
from pso_lab.core.config import PSOConfig
from pso_lab.experiments.pyswarm_runner import (
    ensure_pyswarm_available,
    run_pyswarm_experiment,
)
from pso_lab.experiments.runner import run_single_experiment
from pso_lab.experiments.summary import ExperimentSummary, summarize_experiments
from pso_lab.io.logging_utils import setup_logger
from pso_lab.io.results import save_result, save_summary


RESULTS_ROOT = "results/pyswarm_baseline"
PYSWARM_SOLVER = "pyswarm"


def _build_config(args, *, dimension: int, seed: int) -> PSOConfig:
    return PSOConfig(
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


def _run_solver_experiment(
    solver_name: str,
    *,
    objective_name: str,
    config: PSOConfig,
    args,
    logger,
):
    if solver_name == PYSWARM_SOLVER:
        return run_pyswarm_experiment(
            objective_name=objective_name,
            config=config,
            logger=logger,
        )

    return run_single_experiment(
        objective_name=objective_name,
        config=config,
        evaluation_mode=solver_name,
        max_workers=args.max_workers if solver_name in {"threading", "multiprocessing"} else None,
        batch_size=args.batch_size if solver_name == "multiprocessing" else None,
    )


def _save_solver_result(
    *,
    solver_name: str,
    dimension: int,
    objective_name: str,
    seed: int,
    config: PSOConfig,
    result,
) -> None:
    save_result(
        output_path=(
            f"{RESULTS_ROOT}/{solver_name}/d{dimension}/{objective_name}/seed_{seed}.json"
        ),
        best_position=np.asarray(result.best_position, dtype=float),
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


def _build_summary_row(
    *,
    dimension: int,
    objective_name: str,
    solver_name: str,
    summary: ExperimentSummary,
) -> dict[str, object]:
    return {
        "Dimension": dimension,
        "Objective": objective_name,
        "Solver": solver_name,
        "Runs": summary.num_runs,
        "Mean Best": f"{summary.mean_best_value:.6e}",
        "Std Best": f"{summary.std_best_value:.6e}",
        "Min Best": f"{summary.min_best_value:.6e}",
        "Max Best": f"{summary.max_best_value:.6e}",
        "Mean Time (s)": f"{summary.mean_elapsed_time_s:.6f}",
        "Std Time (s)": f"{summary.std_elapsed_time_s:.6f}",
        "Mean Iter": f"{summary.mean_iterations:.1f}",
        "Min Iter": summary.min_iterations,
        "Max Iter": summary.max_iterations,
    }


def _build_delta_rows(
    summary_by_key: dict[tuple[int, str, str], ExperimentSummary],
    *,
    dimensions: list[int],
    objectives: list[str],
    custom_solvers: list[str],
) -> list[dict[str, object]]:
    delta_rows: list[dict[str, object]] = []

    for dimension in dimensions:
        for objective_name in objectives:
            baseline = summary_by_key.get((dimension, objective_name, PYSWARM_SOLVER))
            if baseline is None:
                continue

            for solver_name in custom_solvers:
                summary = summary_by_key.get((dimension, objective_name, solver_name))
                if summary is None:
                    continue

                delta_rows.append(
                    {
                        "Dimension": dimension,
                        "Objective": objective_name,
                        "Solver": solver_name,
                        "Mean Best": f"{summary.mean_best_value:.6e}",
                        "pyswarm Mean Best": f"{baseline.mean_best_value:.6e}",
                        "Delta Best": f"{summary.mean_best_value - baseline.mean_best_value:.6e}",
                        "Mean Time (s)": f"{summary.mean_elapsed_time_s:.6f}",
                        "pyswarm Mean Time (s)": f"{baseline.mean_elapsed_time_s:.6f}",
                        "Delta Time (s)": f"{summary.mean_elapsed_time_s - baseline.mean_elapsed_time_s:.6f}",
                        "Mean Iter": f"{summary.mean_iterations:.1f}",
                        "pyswarm Mean Iter": f"{baseline.mean_iterations:.1f}",
                        "Delta Iter": f"{summary.mean_iterations - baseline.mean_iterations:.1f}",
                    }
                )
    return delta_rows


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_pyswarm_baseline_args(argv)
    logger = setup_logger("pso_pyswarm_baseline")
    try:
        ensure_pyswarm_available()
    except ModuleNotFoundError as exc:
        logger.error("%s", exc)
        return

    custom_solvers = list(args.modes)
    solvers = custom_solvers + [PYSWARM_SOLVER]
    dimensions = list(args.dimensions)
    objectives = list(args.objectives)
    seeds = list(args.seeds)

    summary_rows: list[dict[str, object]] = []
    summary_by_key: dict[tuple[int, str, str], ExperimentSummary] = {}
    solver_elapsed_times = {solver_name: 0.0 for solver_name in solvers}
    total_start = perf_counter()

    for dimension in dimensions:
        logger.info("Baseline comparison for dimension d=%d", dimension)
        for objective_name in objectives:
            logger.info("Baseline comparison for objective=%s", objective_name)
            for solver_name in solvers:
                logger.info("Running solver=%s", solver_name)
                solver_start = perf_counter()
                results = []

                for seed in seeds:
                    config = _build_config(args, dimension=dimension, seed=seed)
                    result = _run_solver_experiment(
                        solver_name,
                        objective_name=objective_name,
                        config=config,
                        args=args,
                        logger=logger,
                    )
                    results.append(result)
                    _save_solver_result(
                        solver_name=solver_name,
                        dimension=dimension,
                        objective_name=objective_name,
                        seed=seed,
                        config=config,
                        result=result,
                    )
                    logger.info(
                        "result | d=%d | objective=%s | solver=%s | seed=%d | best=%.6e | time=%.6f | iterations=%d",
                        dimension,
                        objective_name,
                        solver_name,
                        seed,
                        result.best_value,
                        result.elapsed_time_s,
                        result.iterations_completed,
                    )

                summary = summarize_experiments(results)
                summary_by_key[(dimension, objective_name, solver_name)] = summary
                summary_rows.append(
                    _build_summary_row(
                        dimension=dimension,
                        objective_name=objective_name,
                        solver_name=solver_name,
                        summary=summary,
                    )
                )
                save_summary(
                    output_path=(
                        f"{RESULTS_ROOT}/{solver_name}/d{dimension}/{objective_name}_summary.json"
                    ),
                    summary=summary,
                    evaluation_mode=solver_name,
                )
                logger.info(
                    "summary | d=%d | objective=%s | solver=%s | mean_best=%.6e | mean_time=%.6f",
                    dimension,
                    objective_name,
                    solver_name,
                    summary.mean_best_value,
                    summary.mean_elapsed_time_s,
                )
                solver_elapsed_times[solver_name] += perf_counter() - solver_start

    total_elapsed_time = perf_counter() - total_start

    print("\n=== PYSWARM BASELINE SUMMARY ===")
    print(tabulate(summary_rows, headers="keys", tablefmt="grid"))
    logger.info("PYSWARM BASELINE SUMMARY printed")

    delta_rows = _build_delta_rows(
        summary_by_key,
        dimensions=dimensions,
        objectives=objectives,
        custom_solvers=custom_solvers,
    )
    if delta_rows:
        print("\n=== DELTA VS PYSWARM ===")
        print(tabulate(delta_rows, headers="keys", tablefmt="grid"))
        logger.info("DELTA VS PYSWARM printed")

    runtime_rows = [
        {
            "Solver": solver_name,
            "Total Time (s)": f"{elapsed_time:.6f}",
        }
        for solver_name, elapsed_time in solver_elapsed_times.items()
    ]
    runtime_rows.append(
        {
            "Solver": "total",
            "Total Time (s)": f"{total_elapsed_time:.6f}",
        }
    )

    print("\n=== PYSWARM BASELINE EXECUTION TIMES ===")
    print(tabulate(runtime_rows, headers="keys", tablefmt="grid"))
    logger.info("PYSWARM BASELINE EXECUTION TIMES printed")


if __name__ == "__main__":
    # Edit these values and press Run in VS Code.
    vscode_argv = [
        "--modes", "sequential", "threading", "multiprocessing",
        "--dimensions", "2", "10", "30",
        "--objectives", "sphere", "rosenbrock", "rastrigin", "ackley",
        "--seeds", "0", "1", "2", "3", "4",
        "--particles", "30",
        "--iterations", "100",
        "--inertia", "0.7",
        "--c1", "1.5",
        "--c2", "1.5",
        "--max-workers", "4",
        "--batch-size", "8",
    ]
    main(sys.argv[1:] or vscode_argv)
