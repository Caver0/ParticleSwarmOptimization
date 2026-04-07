from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from tabulate import tabulate

from pso_lab.io.logging_utils import setup_logger
from pso_lab.viz.plots import (
    save_best_value_plot,
    save_convergence_plots,
    save_time_plot,
    save_time_vs_quality_scatter,
    save_grid_search_heatmaps,
)


logger = setup_logger("pso_analysis")


def _parse_result_metadata(result_file: Path, *, is_summary: bool) -> dict[str, object | None]:
    path_parts = result_file.parts

    mode = None
    objective = None
    dimension = None
    seed = None
    source = "unknown"

    if "grid_search" in path_parts:
        source = "grid_search"
        try:
            idx = path_parts.index("grid_search")
            mode = path_parts[idx + 1]
            dim_token = path_parts[idx + 2]
            if dim_token.startswith("d"):
                dimension = int(dim_token[1:])
            objective = path_parts[idx + 3]
        except (ValueError, IndexError):
            pass

    elif "best_config_comparison" in path_parts:
        source = "best_config_comparison"
        try:
            idx = path_parts.index("best_config_comparison")
            mode = path_parts[idx + 1]

            next_token = path_parts[idx + 2]
            if next_token.startswith("d"):
                dimension = int(next_token[1:])
                objective_index = idx + 3
            else:
                objective_index = idx + 2

            if is_summary:
                objective = result_file.stem.replace("_summary", "")
            else:
                objective = path_parts[objective_index]
                if result_file.stem.startswith("seed_"):
                    seed = int(result_file.stem.replace("seed_", ""))
        except (ValueError, IndexError):
            pass

    else:
        source = "benchmarks"
        try:
            idx = path_parts.index("results")
            mode = path_parts[idx + 1]
            dim_token = path_parts[idx + 2]
            if dim_token.startswith("d"):
                dimension = int(dim_token[1:])
            if is_summary:
                objective = result_file.stem.replace("_benchmark_summary", "")
            else:
                objective = path_parts[idx + 3]
                if result_file.stem.startswith("seed_"):
                    seed = int(result_file.stem.replace("seed_", ""))
        except (ValueError, IndexError):
            pass

    return {
        "Source": source,
        "Mode": mode,
        "Dimension": dimension,
        "Objective": objective,
        "Seed": seed,
    }


def load_summary_files(results_root: str | Path = "results") -> pd.DataFrame:
    results_root = Path(results_root)

    rows: list[dict] = []

    if not results_root.exists():
        logger.warning("Results directory does not exist: %s", results_root)
        return pd.DataFrame()

    summary_files = list(results_root.rglob("*summary.json"))
    logger.info("Found %d summary files", len(summary_files))

    for summary_file in summary_files:
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            metadata = _parse_result_metadata(summary_file, is_summary=True)
            mode = metadata["Mode"] or data.get("evaluation_mode")
            objective = metadata["Objective"] or data.get("objective_name")
            dimension = metadata["Dimension"]
            source = metadata["Source"]

            row = {
                "Source": source,
                "Mode": mode,
                "Dimension": dimension,
                "Objective": objective,
                "Runs": data.get("num_runs"),
                "Mean Best": data.get("mean_best_value"),
                "Std Best": data.get("std_best_value"),
                "Min Best": data.get("min_best_value"),
                "Max Best": data.get("max_best_value"),
                "Mean Time (s)": data.get("mean_elapsed_time_s"),
                "Std Time (s)": data.get("std_elapsed_time_s"),
                "Mean Iter": data.get("mean_iterations"),
                "Min Iter": data.get("min_iterations"),
                "Max Iter": data.get("max_iterations"),
                "File": str(summary_file),
            }
            rows.append(row)

        except Exception as exc:
            logger.warning("Could not read summary file %s: %s", summary_file, exc)

    df = pd.DataFrame(rows)
    return df


def load_history_files(
    results_root: str | Path = "results",
    source: str | None = None,
) -> pd.DataFrame:
    results_root = Path(results_root)

    rows: list[dict] = []

    if not results_root.exists():
        logger.warning("Results directory does not exist: %s", results_root)
        return pd.DataFrame()

    result_files = [
        result_file
        for result_file in results_root.rglob("*.json")
        if "summary" not in result_file.stem
    ]
    logger.info("Found %d result files with potential histories", len(result_files))

    for result_file in result_files:
        try:
            with open(result_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            history = data.get("best_value_history")
            if not isinstance(history, list) or not history:
                continue

            metadata = _parse_result_metadata(result_file, is_summary=False)
            record_source = metadata["Source"]
            if source is not None and record_source != source:
                continue

            config = data.get("config", {})
            row = {
                "Source": record_source,
                "Mode": metadata["Mode"] or data.get("evaluation_mode"),
                "Dimension": metadata["Dimension"] or config.get("dimensions"),
                "Objective": metadata["Objective"] or data.get("objective"),
                "Seed": metadata["Seed"] if metadata["Seed"] is not None else config.get("seed"),
                "Iterations": data.get("iterations_completed", len(history)),
                "Best Value History": history,
                "File": str(result_file),
            }
            rows.append(row)
        except Exception as exc:
            logger.warning("Could not read result file %s: %s", result_file, exc)

    return pd.DataFrame(rows)


def print_summary_table(df: pd.DataFrame) -> None:
    if df.empty:
        print("No summary data found.")
        return

    printable = df.copy()

    numeric_cols = [
        "Mean Best",
        "Std Best",
        "Min Best",
        "Max Best",
        "Mean Time (s)",
        "Std Time (s)",
    ]

    for col in numeric_cols:
        printable[col] = printable[col].map(
            lambda x: f"{x:.6e}" if isinstance(x, (int, float)) and col != "Mean Time (s)" and col != "Std Time (s)"
            else (f"{x:.6f}" if isinstance(x, (int, float)) else x)
        )

    if "Mean Iter" in printable.columns:
        printable["Mean Iter"] = printable["Mean Iter"].map(
            lambda x: f"{x:.1f}" if isinstance(x, (int, float)) else x
        )

    columns = [
        "Source",
        "Mode",
        "Dimension",
        "Objective",
        "Runs",
        "Mean Best",
        "Std Best",
        "Min Best",
        "Max Best",
        "Mean Time (s)",
        "Std Time (s)",
        "Mean Iter",
        "Min Iter",
        "Max Iter",
    ]

    print("\n=== ANALYSIS SUMMARY TABLE ===")
    print(tabulate(printable[columns], headers="keys", tablefmt="grid", showindex=False))


def main() -> None:
    logger.info("Starting results analysis")

    df = load_summary_files("results")

    if df.empty:
        logger.warning("No summary data available for analysis")
        print("No summary data found in results/.")
        return

    logger.info("Loaded %d summary rows", len(df))

    print_summary_table(df)

    plots_dir = Path("reports") / "plots"
    save_time_plot(df, plots_dir)
    save_best_value_plot(df, plots_dir)
    save_time_vs_quality_scatter(df, plots_dir)
    save_grid_search_heatmaps(df, plots_dir)
    history_df = load_history_files("results", source="best_config_comparison")
    save_convergence_plots(history_df, plots_dir)
    logger.info("Plots saved in %s", plots_dir)
    print(f"\nPlots generated in: {plots_dir}")


if __name__ == "__main__":
    main()
