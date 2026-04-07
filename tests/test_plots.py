from pathlib import Path

import pandas as pd

from pso_lab.viz.plots import save_convergence_plots, save_particle_motion_plot


def test_save_convergence_plots_creates_one_file_per_dimension(tmp_path: Path) -> None:
    history_df = pd.DataFrame(
        [
            {
                "Source": "best_config_comparison",
                "Mode": "v0",
                "Dimension": 2,
                "Objective": "sphere",
                "Seed": 0,
                "Best Value History": [1.0, 0.5, 0.1],
            },
            {
                "Source": "best_config_comparison",
                "Mode": "v1",
                "Dimension": 2,
                "Objective": "sphere",
                "Seed": 1,
                "Best Value History": [0.9, 0.45, 0.09],
            },
            {
                "Source": "best_config_comparison",
                "Mode": "v2",
                "Dimension": 2,
                "Objective": "sphere",
                "Seed": 2,
                "Best Value History": [0.95, 0.4, 0.08],
            },
            {
                "Source": "best_config_comparison",
                "Mode": "sequential",
                "Dimension": 10,
                "Objective": "ackley",
                "Seed": 0,
                "Best Value History": [5.0, 1.0, 0.2],
            },
            {
                "Source": "best_config_comparison",
                "Mode": "threading",
                "Dimension": 10,
                "Objective": "ackley",
                "Seed": 1,
                "Best Value History": [4.5, 0.9, 0.18],
            },
            {
                "Source": "best_config_comparison",
                "Mode": "multiprocessing",
                "Dimension": 10,
                "Objective": "ackley",
                "Seed": 2,
                "Best Value History": [4.8, 0.95, 0.19],
            },
        ]
    )

    save_convergence_plots(history_df, tmp_path)

    assert (tmp_path / "convergence_d2.png").exists()
    assert (tmp_path / "convergence_d10.png").exists()


def test_save_particle_motion_plot_creates_image(tmp_path: Path) -> None:
    swarm_history = [
        [
            [-3.0, -2.5, 1.0],
            [2.5, 2.0, -1.5],
            [1.0, -1.5, 0.4],
        ],
        [
            [-1.5, -1.0, 0.8],
            [1.5, 1.2, -1.0],
            [0.5, -0.5, 0.2],
        ],
        [
            [-0.2, -0.1, 0.2],
            [0.4, 0.3, -0.2],
            [0.1, -0.1, 0.1],
        ],
    ]

    output_path = save_particle_motion_plot(
        swarm_position_history=swarm_history,
        objective_name="sphere",
        output_dir=tmp_path,
        dimensions=3,
        evaluation_mode="v0",
        method_label="v3",
        seed=42,
        max_snapshots=3,
        surface_resolution=40,
    )

    assert output_path is not None
    assert output_path.exists()
    assert output_path.name.endswith("_v3.png")
