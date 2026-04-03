from __future__ import annotations
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any
import numpy as np
from pso_lab.core.config import PSOConfig
from pso_lab.experiments.summary import ExperimentSummary

def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if is_dataclass(obj):
        return asdict(obj)
    return obj

def save_result(
        output_path: str|Path,
        best_position: np.ndarray,
        best_value: float, 
        config: PSOConfig,
        objective_name: str,
        evaluation_mode: str | None = None,
        elapsed_time_s: float | None = None,
        iterations_completed: int | None = None,
        best_value_history: list[float] | None = None,
) -> None:
    """Save PSO optimization result to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents= True, exist_ok=True)
    result = {
        "objective": objective_name,
        "evaluation_mode": evaluation_mode,
        "best_value": float(best_value),
        "best_position": best_position.tolist(),
        "elapsed_time_s": elapsed_time_s,
        "iterations_completed": iterations_completed,
        "best_value_history": best_value_history,
        "config":asdict(config),
    }

    with open(output_path, "w", encoding = "utf-8") as f:
        json.dump(result, f, indent = 4, default= _to_serializable)


def save_summary(output_path: str|Path, summary: ExperimentSummary, evaluation_mode:str | None = None) -> None:
    """Save benchmark summary statistics to a JSON file."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary_data =asdict(summary)
    summary_data["evaluation_mode"] = evaluation_mode

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=4, default= _to_serializable)