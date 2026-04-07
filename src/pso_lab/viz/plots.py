from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pso_lab.objectives import build_objective


MODE_ORDER = ["sequential", "threading", "multiprocessing"]
OBJECTIVE_ORDER = ["sphere", "rosenbrock", "rastrigin", "ackley"]
METHOD_STYLES = {
    "sequential": {
        "label": "v0",
        "color": "tab:blue",
        "linestyle": "-",
        "marker": "o",
        "markevery": (0, 10),
    },
    "threading": {
        "label": "v1",
        "color": "tab:orange",
        "linestyle": "--",
        "marker": "s",
        "markevery": (3, 10),
    },
    "multiprocessing": {
        "label": "v2",
        "color": "tab:green",
        "linestyle": "-.",
        "marker": "^",
        "markevery": (6, 10),
    },
}


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


def _normalize_mode_name(mode: str | None) -> str | None:
    if mode is None:
        return None

    normalized = str(mode).strip().lower()
    aliases = {
        "v0": "sequential",
        "v1": "threading",
        "v2": "multiprocessing",
        "threads": "threading",
        "processes": "multiprocessing",
    }
    return aliases.get(normalized, normalized)


def _safe_positive(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    positive_values = values[values > 0]
    floor = positive_values.min() * 0.5 if positive_values.size else 1e-16
    floor = max(floor, 1e-300)
    return np.where(values > 0, values, floor)


def _history_to_curve_df(history_df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"Mode", "Dimension", "Objective", "Best Value History"}
    if history_df.empty or not required_columns.issubset(history_df.columns):
        return pd.DataFrame()

    plot_df = history_df.copy()
    plot_df["Mode"] = plot_df["Mode"].map(_normalize_mode_name)
    plot_df = plot_df[plot_df["Mode"].isin(METHOD_STYLES)]
    plot_df = plot_df.dropna(subset=["Dimension", "Objective", "Best Value History"])
    plot_df = plot_df[
        plot_df["Best Value History"].map(lambda values: isinstance(values, list) and len(values) > 0)
    ]

    rows: list[dict[str, float | int | str | None]] = []

    for record in plot_df.to_dict(orient="records"):
        for iteration, best_value in enumerate(record["Best Value History"], start=1):
            rows.append(
                {
                    "Dimension": int(record["Dimension"]),
                    "Objective": str(record["Objective"]),
                    "Mode": str(record["Mode"]),
                    "Seed": record.get("Seed"),
                    "Iteration": iteration,
                    "Best Value": float(best_value),
                }
            )

    if not rows:
        return pd.DataFrame()

    curve_df = pd.DataFrame(rows)
    return (
        curve_df.groupby(["Dimension", "Objective", "Mode", "Iteration"], as_index=False)["Best Value"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "Mean Best Value", "std": "Std Best Value"})
        .fillna({"Std Best Value": 0.0})
    )


def _reference_point_for_objective(objective_name: str, dimensions: int) -> np.ndarray:
    reference = np.zeros(dimensions, dtype=float)
    if objective_name.strip().lower() in {"rosenbrock", "rosenbrok"}:
        reference.fill(1.0)
    return reference


def _sample_snapshot_indices(total_steps: int, max_snapshots: int) -> list[int]:
    if total_steps <= 0:
        return []
    if total_steps <= max_snapshots:
        return list(range(total_steps))
    return np.unique(
        np.round(np.linspace(0, total_steps - 1, num=max_snapshots)).astype(int)
    ).tolist()


def _build_objective_projection_surface(
    objective_name: str,
    dimensions: int,
    resolution: int = 150,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    objective = build_objective(objective_name, dimensions=dimensions)
    x_bounds = objective.bounds[0]
    y_bounds = objective.bounds[1]
    x_values = np.linspace(x_bounds[0], x_bounds[1], resolution)
    y_values = np.linspace(y_bounds[0], y_bounds[1], resolution)
    grid_x, grid_y = np.meshgrid(x_values, y_values)

    reference_point = _reference_point_for_objective(objective_name, dimensions)
    sample_points = np.repeat(reference_point[np.newaxis, :], grid_x.size, axis=0)
    sample_points[:, 0] = grid_x.ravel()
    sample_points[:, 1] = grid_y.ravel()
    surface_values = objective.evaluate_many(sample_points).reshape(grid_x.shape)
    projected_surface = np.log10(1.0 + np.maximum(surface_values, 0.0))

    bounds = np.array([x_bounds, y_bounds], dtype=float)
    return grid_x, grid_y, projected_surface, reference_point, bounds


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

    for objective in OBJECTIVE_ORDER:
        objective_df = plot_df[plot_df["Objective"] == objective]
        if objective_df.empty:
            continue

        for mode in MODE_ORDER:
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

    for objective in OBJECTIVE_ORDER:
        objective_df = plot_df[plot_df["Objective"] == objective]
        if objective_df.empty:
            continue

        for mode in MODE_ORDER:
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


def save_convergence_plots(
    history_df: pd.DataFrame,
    output_dir: str | Path,
    source: str = "best_config_comparison",
) -> None:
    output_dir = _ensure_output_dir(output_dir)

    if history_df.empty:
        return

    plot_df = history_df.copy()
    if "Source" in plot_df.columns:
        plot_df = plot_df[plot_df["Source"] == source]

    curve_df = _history_to_curve_df(plot_df)
    if curve_df.empty:
        return

    for dimension in sorted(curve_df["Dimension"].unique()):
        dimension_df = curve_df[curve_df["Dimension"] == dimension]
        objectives = [objective for objective in OBJECTIVE_ORDER if objective in dimension_df["Objective"].unique()]

        if not objectives:
            objectives = sorted(dimension_df["Objective"].unique().tolist())

        if not objectives:
            continue

        ncols = 2 if len(objectives) > 1 else 1
        nrows = math.ceil(len(objectives) / ncols)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 4.8 * nrows), squeeze=False)
        axes_flat = axes.flatten()

        for index, objective in enumerate(objectives):
            ax = axes_flat[index]
            objective_df = dimension_df[dimension_df["Objective"] == objective]

            for mode_index, mode in enumerate(MODE_ORDER, start=1):
                mode_df = objective_df[objective_df["Mode"] == mode].sort_values("Iteration")
                if mode_df.empty:
                    continue

                style = METHOD_STYLES[mode]
                x_values = mode_df["Iteration"].to_numpy(dtype=int)
                mean_values = _safe_positive(mode_df["Mean Best Value"].to_numpy(dtype=float))
                std_values = mode_df["Std Best Value"].to_numpy(dtype=float)
                lower = _safe_positive(mode_df["Mean Best Value"].to_numpy(dtype=float) - std_values)
                upper = np.maximum(
                    mode_df["Mean Best Value"].to_numpy(dtype=float) + std_values,
                    lower,
                )

                ax.plot(
                    x_values,
                    mean_values,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=2.2,
                    marker=style["marker"],
                    markersize=4,
                    markevery=style["markevery"],
                    label=style["label"],
                    zorder=mode_index + 1,
                )
                ax.fill_between(
                    x_values,
                    lower,
                    upper,
                    color=style["color"],
                    alpha=0.12,
                    linewidth=0,
                    zorder=1,
                )

            ax.set_title(objective.capitalize())
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Mean best value")
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.25)

        for index in range(len(objectives), len(axes_flat)):
            fig.delaxes(axes_flat[index])

        handles = [
            plt.Line2D(
                [0],
                [0],
                color=METHOD_STYLES[mode]["color"],
                linestyle=METHOD_STYLES[mode]["linestyle"],
                marker=METHOD_STYLES[mode]["marker"],
                linewidth=2.2,
                markersize=5,
                label=METHOD_STYLES[mode]["label"],
            )
            for mode in MODE_ORDER
            if mode in dimension_df["Mode"].unique()
        ]

        fig.legend(handles=handles, title="Method", loc="upper center", ncol=max(1, len(handles)))
        fig.suptitle(
            f"PSO convergence by objective and method | d={dimension}",
            y=0.995,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(output_dir / f"convergence_d{dimension}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def save_particle_motion_plot(
    swarm_position_history: list[list[list[float]]] | list[np.ndarray] | None,
    objective_name: str,
    output_dir: str | Path,
    dimensions: int,
    evaluation_mode: str = "sequential",
    method_label: str | None = None,
    seed: int | None = None,
    max_snapshots: int = 6,
    surface_resolution: int = 150,
) -> Path | None:
    output_dir = _ensure_output_dir(output_dir)

    if not swarm_position_history or dimensions < 2:
        return None

    history = [np.asarray(positions, dtype=float) for positions in swarm_position_history]
    history = [
        positions
        for positions in history
        if positions.ndim == 2 and positions.shape[1] >= 2 and positions.shape[0] > 0
    ]
    if not history:
        return None

    normalized_mode = _normalize_mode_name(evaluation_mode) or "sequential"
    method_label = method_label or METHOD_STYLES.get(normalized_mode, {}).get("label", normalized_mode)

    grid_x, grid_y, surface, reference_point, bounds = _build_objective_projection_surface(
        objective_name=objective_name,
        dimensions=dimensions,
        resolution=surface_resolution,
    )
    snapshot_indices = _sample_snapshot_indices(len(history), max_snapshots=max_snapshots)
    if not snapshot_indices:
        return None

    particle_count = history[0].shape[0]
    particle_colors = plt.cm.viridis(np.linspace(0.08, 0.92, particle_count))
    contour_levels = np.linspace(surface.min(), surface.max(), 18)
    if np.isclose(surface.min(), surface.max()):
        contour_levels = 12

    ncols = min(3, len(snapshot_indices))
    nrows = math.ceil(len(snapshot_indices) / ncols)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.4 * ncols, 4.8 * nrows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for subplot_index, history_index in enumerate(snapshot_indices):
        ax = axes_flat[subplot_index]
        ax.contourf(grid_x, grid_y, surface, levels=contour_levels, cmap="Greys", alpha=0.88)
        ax.contour(grid_x, grid_y, surface, levels=8, colors="white", alpha=0.25, linewidths=0.55)

        trajectory = np.stack(history[: history_index + 1], axis=0)
        for particle_index in range(particle_count):
            particle_path = trajectory[:, particle_index, :]
            ax.plot(
                particle_path[:, 0],
                particle_path[:, 1],
                color=particle_colors[particle_index],
                alpha=0.22,
                linewidth=0.85,
            )

        current_positions = history[history_index]
        ax.scatter(
            current_positions[:, 0],
            current_positions[:, 1],
            c=particle_colors,
            s=42,
            edgecolors="black",
            linewidths=0.3,
            zorder=3,
        )
        ax.scatter(
            reference_point[0],
            reference_point[1],
            marker="*",
            s=175,
            color="gold",
            edgecolors="black",
            linewidths=0.8,
            zorder=4,
        )

        snapshot_title = "Initial state" if history_index == 0 else f"Iteration {history_index}"
        ax.set_title(snapshot_title)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_xlim(bounds[0, 0], bounds[0, 1])
        ax.set_ylim(bounds[1, 0], bounds[1, 1])
        ax.grid(True, alpha=0.18)
        ax.set_aspect("equal", adjustable="box")

    for subplot_index in range(len(snapshot_indices), len(axes_flat)):
        fig.delaxes(axes_flat[subplot_index])

    hidden_axis_note = ""
    if dimensions >= 3:
        hidden_axis_note = f" | background slice fixes x3={reference_point[2]:.1f}"

    title_parts = [
        f"Particle motion projected on x1-x2 | {objective_name} | d={dimensions}",
        f"method={method_label}",
    ]
    if seed is not None:
        title_parts.append(f"seed={seed}")

    fig.suptitle(" | ".join(title_parts), y=0.995)
    fig.text(
        0.5,
        0.015,
        f"Each point is one particle; faint trails show its path{hidden_axis_note}.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))

    output_path = output_dir / f"particle_motion_{objective_name}_d{dimensions}_{method_label}.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path
