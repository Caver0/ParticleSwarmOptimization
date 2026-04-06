from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _ensure_output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _dimension_color_map(dimensions: list[int]) -> dict[int, str]:
    palette = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]
    return {dim: palette[i % len(palette)] for i, dim in enumerate(sorted(dimensions))}


def save_time_plot(df: pd.DataFrame, output_dir: str | Path) -> None:
    output_path = _ensure_output_dir(output_dir) / "mean_time_by_mode_dimension_objective.png"

    if df.empty:
        return

    plot_df = (
        df.groupby(["Objective", "Mode", "Dimension"], as_index=False)["Mean Time (s)"]
        .mean()
        .sort_values(["Objective", "Mode", "Dimension"])
    )

    if plot_df.empty:
        return

    dimensions = sorted(plot_df["Dimension"].dropna().unique().tolist())
    color_map = _dimension_color_map(dimensions)

    labels = []
    values = []
    colors = []

    objective_order = ["sphere", "rosenbrock", "rastrigin", "ackley"]
    mode_order = ["sequential", "threading", "multiprocessing"]

    for objective in objective_order:
        objective_df = plot_df[plot_df["Objective"] == objective]
        if objective_df.empty:
            continue

        for mode in mode_order:
            mode_df = objective_df[objective_df["Mode"] == mode].sort_values("Dimension")
            if mode_df.empty:
                continue

            for row in mode_df.itertuples(index=False):
                labels.append(f"{objective} | {mode} | d={row.Dimension}")
                values.append(row[3])
                colors.append(color_map[row.Dimension])

    fig_height = max(8, 0.32 * len(labels))
    plt.figure(figsize=(14, fig_height))
    plt.barh(labels, values, color=colors, height=0.55)
    plt.xlabel("Mean Time (s)")
    plt.ylabel("Objective | Mode | Dimension")
    plt.title("Mean execution time by objective, mode and dimension")

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, label=f"d={dim}", color=color_map[dim])
        for dim in dimensions
    ]
    plt.legend(handles=legend_handles, title="Dimension")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_best_value_plot(df: pd.DataFrame, output_dir: str | Path) -> None:
    output_path = _ensure_output_dir(output_dir) / "mean_best_by_mode_dimension_objective.png"

    if df.empty:
        return

    plot_df = (
        df.groupby(["Objective", "Mode", "Dimension"], as_index=False)["Mean Best"]
        .mean()
        .sort_values(["Objective", "Mode", "Dimension"])
    )

    if plot_df.empty:
        return

    dimensions = sorted(plot_df["Dimension"].dropna().unique().tolist())
    color_map = _dimension_color_map(dimensions)

    labels = []
    values = []
    colors = []

    objective_order = ["sphere", "rosenbrock", "rastrigin", "ackley"]
    mode_order = ["sequential", "threading", "multiprocessing"]

    for objective in objective_order:
        objective_df = plot_df[plot_df["Objective"] == objective]
        if objective_df.empty:
            continue

        for mode in mode_order:
            mode_df = objective_df[objective_df["Mode"] == mode].sort_values("Dimension")
            if mode_df.empty:
                continue

            for row in mode_df.itertuples(index=False):
                labels.append(f"{objective} | {mode} | d={row.Dimension}")
                values.append(row[3])
                colors.append(color_map[row.Dimension])

    fig_height = max(8, 0.32 * len(labels))
    plt.figure(figsize=(14, fig_height))
    plt.barh(labels, values, color=colors, height=0.55)
    plt.xlabel("Mean Best Value")
    plt.ylabel("Objective | Mode | Dimension")
    plt.title("Mean best value by objective, mode and dimension")

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, label=f"d={dim}", color=color_map[dim])
        for dim in dimensions
    ]
    plt.legend(handles=legend_handles, title="Dimension")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_time_vs_quality_scatter(df: pd.DataFrame, output_dir: str | Path) -> None:
    output_path = _ensure_output_dir(output_dir) / "time_vs_quality_scatter.png"

    if df.empty:
        return

    plot_df = df.dropna(subset=["Mean Time (s)", "Mean Best", "Mode", "Dimension", "Objective"]).copy()
    if plot_df.empty:
        return

    dimensions = sorted(plot_df["Dimension"].dropna().unique().tolist())
    color_map = _dimension_color_map(dimensions)

    objective_markers = {
        "sphere": "o",
        "rosenbrock": "s",
        "rastrigin": "^",
        "ackley": "D",
    }

    mode_sizes = {
        "sequential": 60,
        "threading": 90,
        "multiprocessing": 120,
    }

    plt.figure(figsize=(12, 8))

    for objective in sorted(plot_df["Objective"].unique()):
        objective_df = plot_df[plot_df["Objective"] == objective]

        for mode in sorted(objective_df["Mode"].unique()):
            mode_df = objective_df[objective_df["Mode"] == mode]

            for dim in sorted(mode_df["Dimension"].unique()):
                subset = mode_df[mode_df["Dimension"] == dim]
                if subset.empty:
                    continue

                plt.scatter(
                    subset["Mean Time (s)"],
                    subset["Mean Best"],
                    marker=objective_markers.get(objective, "o"),
                    s=mode_sizes.get(mode, 80),
                    color=color_map[dim],
                    alpha=0.8,
                )

    dimension_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            label=f"d={dim}",
            markerfacecolor=color_map[dim],
            markeredgecolor=color_map[dim],
            markersize=8,
        )
        for dim in dimensions
    ]

    objective_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="",
            label=objective,
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=8,
        )
        for objective, marker in objective_markers.items()
        if objective in plot_df["Objective"].unique()
    ]

    mode_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            label=mode,
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=size / 15,
        )
        for mode, size in mode_sizes.items()
        if mode in plot_df["Mode"].unique()
    ]

    legend1 = plt.legend(handles=dimension_handles, title="Dimension", loc="upper left")
    plt.gca().add_artist(legend1)

    legend2 = plt.legend(handles=objective_handles, title="Objective", loc="upper center")
    plt.gca().add_artist(legend2)

    plt.legend(handles=mode_handles, title="Mode", loc="upper right")

    plt.xlabel("Mean Time (s)")
    plt.ylabel("Mean Best Value")
    plt.title("Trade-off between execution time and solution quality")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_grid_search_heatmaps(df: pd.DataFrame, output_dir: str | Path) -> None:
    output_dir = _ensure_output_dir(output_dir)

    if df.empty:
        return

    if "Source" not in df.columns:
        return

    plot_df = df[df["Source"] == "grid_search"].copy()
    required_cols = {"Objective", "Dimension", "w", "c1", "c2", "Mean Best"}
    if plot_df.empty or not required_cols.issubset(plot_df.columns):
        return

    plot_df = plot_df.dropna(subset=["Objective", "Dimension", "w", "c1", "c2", "Mean Best"])
    if plot_df.empty:
        return

    for objective in sorted(plot_df["Objective"].unique()):
        objective_df = plot_df[plot_df["Objective"] == objective]

        for dimension in sorted(objective_df["Dimension"].unique()):
            dim_df = objective_df[objective_df["Dimension"] == dimension]

            for w_value in sorted(dim_df["w"].unique()):
                heat_df = dim_df[dim_df["w"] == w_value]

                pivot = heat_df.pivot_table(
                    index="c1",
                    columns="c2",
                    values="Mean Best",
                    aggfunc="mean",
                )

                if pivot.empty:
                    continue

                plt.figure(figsize=(8, 6))
                plt.imshow(pivot.values, aspect="auto", origin="lower")
                plt.colorbar(label="Mean Best")
                plt.xticks(range(len(pivot.columns)), [str(v) for v in pivot.columns])
                plt.yticks(range(len(pivot.index)), [str(v) for v in pivot.index])
                plt.xlabel("c2")
                plt.ylabel("c1")
                plt.title(f"Grid Search Heatmap | {objective} | d={dimension} | w={w_value}")
                plt.tight_layout()

                filename = f"heatmap_{objective}_d{dimension}_w{w_value}.png"
                plt.savefig(output_dir / filename, dpi=200, bbox_inches="tight")
                plt.close()