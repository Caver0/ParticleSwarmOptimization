from __future__ import annotations

import argparse
from collections.abc import Sequence


EVALUATION_MODE_CHOICES = ("sequential", "threading", "multiprocessing")
GRID_SEARCH_MODE_CHOICES = (
    "all",
    "v0",
    "v1",
    "v2",
    "sequential",
    "threading",
    "multiprocessing",
)
OBJECTIVE_CHOICES = ("sphere", "rosenbrock", "rastrigin", "ackley")
VISUALIZATION_METHOD_CHOICES = ("v1", "v2", "v3")

DEFAULT_BENCHMARK_DIMENSIONS = (2, 10, 30)
DEFAULT_SINGLE_DIMENSION_LIST = (2,)
DEFAULT_OBJECTIVES = OBJECTIVE_CHOICES
DEFAULT_SINGLE_RUN_OBJECTIVE = "sphere"
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_PARTICLES = 30
DEFAULT_ITERATIONS = 100
DEFAULT_INERTIA = 0.7
DEFAULT_C1 = 1.5
DEFAULT_C2 = 1.5
DEFAULT_SINGLE_RUN_TOLERANCE = 1e-8
DEFAULT_MAX_WORKERS = 4
DEFAULT_BATCH_SIZE = 8
DEFAULT_GRID_INERTIA_VALUES = (0.4, 0.7, 0.9)
DEFAULT_GRID_C1_VALUES = (1.0, 1.5, 2.0)
DEFAULT_GRID_C2_VALUES = (1.0, 1.5, 2.0)
DEFAULT_VISUALIZATION_METHODS = VISUALIZATION_METHOD_CHOICES
DEFAULT_VISUALIZATION_DIMENSION = 3
DEFAULT_VISUALIZATION_SEED = 42
DEFAULT_SINGLE_RUN_OUTPUT_PATH = "results/sphere_run.json"
DEFAULT_ANALYSIS_RESULTS_DIR = "results"
DEFAULT_ANALYSIS_PLOTS_DIR = "reports/plots"
DEFAULT_RESULTS_DIR = "results/visualization"
DEFAULT_OUTPUT_DIR = "reports/plots"


def _build_parser(description: str) -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description=description)


def _add_modes_argument(
    parser: argparse.ArgumentParser,
    *,
    default: Sequence[str],
    choices: Sequence[str] = EVALUATION_MODE_CHOICES,
    help_text: str,
) -> None:
    parser.add_argument(
        "--modes",
        nargs="+",
        default=list(default),
        choices=choices,
        help=help_text,
    )


def _add_dimensions_argument(
    parser: argparse.ArgumentParser,
    *,
    default: Sequence[int],
    help_text: str,
) -> None:
    parser.add_argument(
        "--dimensions",
        nargs="+",
        type=int,
        default=list(default),
        help=help_text,
    )


def _add_dimension_argument(
    parser: argparse.ArgumentParser,
    *,
    default: int,
    help_text: str,
) -> None:
    parser.add_argument(
        "--dimension",
        type=int,
        default=default,
        help=help_text,
    )


def _add_objectives_argument(
    parser: argparse.ArgumentParser,
    *,
    default: Sequence[str] = DEFAULT_OBJECTIVES,
    help_text: str,
) -> None:
    parser.add_argument(
        "--objectives",
        nargs="+",
        default=list(default),
        choices=OBJECTIVE_CHOICES,
        help=help_text,
    )


def _add_objective_argument(
    parser: argparse.ArgumentParser,
    *,
    default: str = DEFAULT_SINGLE_RUN_OBJECTIVE,
    help_text: str,
) -> None:
    parser.add_argument(
        "--objective",
        default=default,
        choices=OBJECTIVE_CHOICES,
        help=help_text,
    )


def _add_seeds_argument(
    parser: argparse.ArgumentParser,
    *,
    default: Sequence[int] = DEFAULT_SEEDS,
    help_text: str,
) -> None:
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(default),
        help=help_text,
    )


def _add_seed_argument(
    parser: argparse.ArgumentParser,
    *,
    default: int = DEFAULT_VISUALIZATION_SEED,
    help_text: str,
) -> None:
    parser.add_argument(
        "--seed",
        type=int,
        default=default,
        help=help_text,
    )


def _add_particles_argument(parser: argparse.ArgumentParser, *, help_text: str) -> None:
    parser.add_argument(
        "--particles",
        type=int,
        default=DEFAULT_PARTICLES,
        help=help_text,
    )


def _add_iterations_argument(parser: argparse.ArgumentParser, *, help_text: str) -> None:
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help=help_text,
    )


def _add_inertia_argument(parser: argparse.ArgumentParser, *, help_text: str) -> None:
    parser.add_argument(
        "--inertia",
        type=float,
        default=DEFAULT_INERTIA,
        help=help_text,
    )


def _add_c1_argument(parser: argparse.ArgumentParser, *, help_text: str) -> None:
    parser.add_argument(
        "--c1",
        type=float,
        default=DEFAULT_C1,
        help=help_text,
    )


def _add_c2_argument(parser: argparse.ArgumentParser, *, help_text: str) -> None:
    parser.add_argument(
        "--c2",
        type=float,
        default=DEFAULT_C2,
        help=help_text,
    )


def _add_tolerance_argument(
    parser: argparse.ArgumentParser,
    *,
    default: float = DEFAULT_SINGLE_RUN_TOLERANCE,
    help_text: str,
) -> None:
    parser.add_argument(
        "--tolerance",
        type=float,
        default=default,
        help=help_text,
    )


def _add_inertia_values_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--inertia-values",
        nargs="+",
        type=float,
        default=list(DEFAULT_GRID_INERTIA_VALUES),
        help="Grid values for inertia weight w.",
    )


def _add_c1_values_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--c1-values",
        nargs="+",
        type=float,
        default=list(DEFAULT_GRID_C1_VALUES),
        help="Grid values for cognitive coefficient c1.",
    )


def _add_c2_values_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--c2-values",
        nargs="+",
        type=float,
        default=list(DEFAULT_GRID_C2_VALUES),
        help="Grid values for social coefficient c2.",
    )


def _add_max_workers_argument(
    parser: argparse.ArgumentParser,
    *,
    flags: Sequence[str] = ("--max-workers",),
    help_text: str,
) -> None:
    parser.add_argument(
        *flags,
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=help_text,
    )


def _add_batch_size_argument(
    parser: argparse.ArgumentParser,
    *,
    flags: Sequence[str] = ("--batch-size",),
    help_text: str,
) -> None:
    parser.add_argument(
        *flags,
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=help_text,
    )


def _add_methods_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(DEFAULT_VISUALIZATION_METHODS),
        choices=VISUALIZATION_METHOD_CHOICES,
        help="Visualization method folders to generate.",
    )


def _add_results_dir_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help="Directory where raw visualization runs will be stored.",
    )


def _add_output_dir_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where plot images will be saved.",
    )


def _add_output_path_argument(
    parser: argparse.ArgumentParser,
    *,
    default: str = DEFAULT_SINGLE_RUN_OUTPUT_PATH,
    help_text: str,
) -> None:
    parser.add_argument(
        "--output-path",
        default=default,
        help=help_text,
    )


def build_single_run_parser() -> argparse.ArgumentParser:
    parser = _build_parser("Run a single PSO optimization instance.")
    _add_objective_argument(
        parser,
        help_text="Objective function to optimize.",
    )
    _add_dimension_argument(
        parser,
        default=DEFAULT_SINGLE_DIMENSION_LIST[0],
        help_text="Problem dimension.",
    )
    _add_particles_argument(parser, help_text="Number of particles in the swarm.")
    _add_iterations_argument(parser, help_text="Maximum number of PSO iterations.")
    _add_inertia_argument(parser, help_text="Inertia weight (w).")
    _add_c1_argument(parser, help_text="Cognitive coefficient (c1).")
    _add_c2_argument(parser, help_text="Social coefficient (c2).")
    _add_seed_argument(
        parser,
        help_text="Random seed used for the run.",
    )
    _add_tolerance_argument(
        parser,
        help_text="Stopping tolerance for the best objective value.",
    )
    _add_output_path_argument(
        parser,
        help_text="Output JSON file for the optimization result.",
    )
    return parser


def parse_single_run_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_single_run_parser().parse_args(argv)


def build_benchmarks_parser() -> argparse.ArgumentParser:
    parser = _build_parser(
        "Run PSO benchmarks across evaluation modes, dimensions and objectives."
    )
    _add_modes_argument(
        parser,
        default=EVALUATION_MODE_CHOICES,
        help_text="Evaluation modes to benchmark",
    )
    _add_dimensions_argument(
        parser,
        default=DEFAULT_BENCHMARK_DIMENSIONS,
        help_text="Dimensions to benchmark",
    )
    _add_objectives_argument(
        parser,
        help_text="Objective functions to benchmark",
    )
    _add_seeds_argument(
        parser,
        help_text="Random seeds for benchmarking",
    )
    _add_particles_argument(parser, help_text="Number of particles in the swarm")
    _add_iterations_argument(parser, help_text="Maximum number of iterations")
    _add_inertia_argument(parser, help_text="Inertia weight (w)")
    _add_c1_argument(parser, help_text="Cognitive coefficient (c1)")
    _add_c2_argument(parser, help_text="Social coefficient (c2)")
    _add_max_workers_argument(
        parser,
        flags=("--max_workers", "--max-workers"),
        help_text="Max workers for parallel modes (threading and multiprocessing)",
    )
    _add_batch_size_argument(
        parser,
        flags=("--batch_size", "--batch-size"),
        help_text="Batch size for multiprocessing evaluator",
    )
    return parser


def parse_benchmarks_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_benchmarks_parser().parse_args(argv)


def build_grid_search_parser() -> argparse.ArgumentParser:
    parser = _build_parser("Run PSO grid search over hyperparameters.")
    parser.add_argument(
        "--mode",
        default="all",
        choices=GRID_SEARCH_MODE_CHOICES,
        help="Evaluation mode used during grid search. Use 'all' to run v0, v1 and v2.",
    )
    _add_dimensions_argument(
        parser,
        default=DEFAULT_SINGLE_DIMENSION_LIST,
        help_text="Problem dimensions to evaluate.",
    )
    _add_objectives_argument(
        parser,
        help_text="Objective functions to include in the grid search.",
    )
    _add_seeds_argument(
        parser,
        help_text="Random seeds for reproducible runs.",
    )
    _add_particles_argument(parser, help_text="Number of particles in the swarm.")
    _add_iterations_argument(parser, help_text="Maximum number of PSO iterations.")
    _add_inertia_values_argument(parser)
    _add_c1_values_argument(parser)
    _add_c2_values_argument(parser)
    _add_max_workers_argument(
        parser,
        help_text="Maximum workers for threading/multiprocessing evaluators.",
    )
    _add_batch_size_argument(
        parser,
        help_text="Batch size for multiprocessing evaluator.",
    )
    return parser


def parse_grid_search_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_grid_search_parser().parse_args(argv)


def build_best_configs_comparison_parser() -> argparse.ArgumentParser:
    parser = _build_parser("Compare the best PSO configurations across evaluation modes.")
    _add_modes_argument(
        parser,
        default=EVALUATION_MODE_CHOICES,
        help_text="Evaluation modes to compare.",
    )
    _add_dimensions_argument(
        parser,
        default=DEFAULT_SINGLE_DIMENSION_LIST,
        help_text="Problem dimensions to compare.",
    )
    _add_objectives_argument(
        parser,
        help_text="Objective functions to compare.",
    )
    _add_seeds_argument(
        parser,
        help_text="Random seeds for reproducible runs.",
    )
    _add_particles_argument(parser, help_text="Number of particles in the swarm.")
    _add_iterations_argument(parser, help_text="Maximum number of PSO iterations.")
    _add_max_workers_argument(
        parser,
        help_text="Maximum workers for threading/multiprocessing evaluators.",
    )
    _add_batch_size_argument(
        parser,
        help_text="Batch size for multiprocessing evaluator.",
    )
    return parser


def parse_best_configs_comparison_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    return build_best_configs_comparison_parser().parse_args(argv)


def build_pyswarm_baseline_parser() -> argparse.ArgumentParser:
    parser = _build_parser("Compare our PSO implementations against a pyswarm baseline.")
    _add_modes_argument(
        parser,
        default=EVALUATION_MODE_CHOICES,
        help_text="Our PSO evaluation modes to compare against pyswarm.",
    )
    _add_dimensions_argument(
        parser,
        default=DEFAULT_BENCHMARK_DIMENSIONS,
        help_text="Problem dimensions to compare.",
    )
    _add_objectives_argument(
        parser,
        help_text="Objective functions to compare.",
    )
    _add_seeds_argument(
        parser,
        help_text="Random seeds for reproducible runs.",
    )
    _add_particles_argument(parser, help_text="Number of particles in the swarm.")
    _add_iterations_argument(parser, help_text="Maximum number of PSO iterations.")
    _add_inertia_argument(parser, help_text="Inertia weight (w).")
    _add_c1_argument(parser, help_text="Cognitive coefficient (c1).")
    _add_c2_argument(parser, help_text="Social coefficient (c2).")
    _add_max_workers_argument(
        parser,
        flags=("--max_workers", "--max-workers"),
        help_text="Maximum workers for threading/multiprocessing evaluators.",
    )
    _add_batch_size_argument(
        parser,
        flags=("--batch_size", "--batch-size"),
        help_text="Batch size for the multiprocessing evaluator.",
    )
    return parser


def parse_pyswarm_baseline_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_pyswarm_baseline_parser().parse_args(argv)


def build_visualization_parser() -> argparse.ArgumentParser:
    parser = _build_parser("Generate 2D particle-motion plots for 3D PSO runs.")
    _add_methods_argument(parser)
    _add_dimension_argument(
        parser,
        default=DEFAULT_VISUALIZATION_DIMENSION,
        help_text="Problem dimension for the visualization run.",
    )
    _add_objectives_argument(
        parser,
        help_text="Objective functions to visualize.",
    )
    _add_particles_argument(parser, help_text="Number of particles in the swarm.")
    _add_iterations_argument(parser, help_text="Maximum number of PSO iterations.")
    _add_inertia_argument(parser, help_text="Inertia weight (w).")
    _add_c1_argument(parser, help_text="Cognitive coefficient (c1).")
    _add_c2_argument(parser, help_text="Social coefficient (c2).")
    _add_seed_argument(
        parser,
        help_text="Random seed used in the visualization run.",
    )
    _add_max_workers_argument(
        parser,
        help_text="Maximum workers for threading and multiprocessing evaluators.",
    )
    _add_batch_size_argument(
        parser,
        help_text="Batch size for the multiprocessing evaluator.",
    )
    _add_results_dir_argument(parser)
    _add_output_dir_argument(parser)
    return parser


def parse_visualization_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_visualization_parser().parse_args(argv)


def build_analysis_parser() -> argparse.ArgumentParser:
    parser = _build_parser("Analyze generated PSO results and export plots.")
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_ANALYSIS_RESULTS_DIR,
        help="Directory containing generated result files.",
    )
    parser.add_argument(
        "--plots-dir",
        default=DEFAULT_ANALYSIS_PLOTS_DIR,
        help="Directory where analysis plots will be saved.",
    )
    return parser


def parse_analysis_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_analysis_parser().parse_args(argv)
