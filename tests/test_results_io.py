from __future__ import annotations

import json

from pso_lab.experiments.summary import ExperimentSummary
from pso_lab.io.results import save_summary


def test_save_summary_persists_evaluation_mode(tmp_path) -> None:
    summary = ExperimentSummary(
        objective_name="sphere",
        num_runs=3,
        mean_best_value=0.1,
        std_best_value=0.01,
        min_best_value=0.05,
        max_best_value=0.2,
        mean_elapsed_time_s=1.5,
        std_elapsed_time_s=0.2,
        mean_iterations=25.0,
        min_iterations=20,
        max_iterations=30,
    )

    output_path = tmp_path / "sphere_summary.json"
    save_summary(output_path, summary, evaluation_mode="pyswarm")

    data = json.loads(output_path.read_text(encoding="utf-8"))

    assert data["objective_name"] == "sphere"
    assert data["evaluation_mode"] == "pyswarm"
