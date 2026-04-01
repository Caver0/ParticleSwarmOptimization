from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from pso_lab.core.config import PSOConfig


def save_result(
        output_path: str|Path,
        best_position: np.ndarray,
        best_value: float, 
        config: PSOConfig,
        objective_name: str
) -> None:
    """Save PSO optimization result to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents= True, exist_ok=True)
    result = {
        "objective": objective_name,
        "best_value": float(best_value),
        "best_position": best_position.tolist(),
        "config":{
            "num_particles": config.num_particles,
            "dimensions": config.dimensions,
            "max_iterations": config.max_iterations,
            "cognitive_coefficient":config.cognitive_coefficient,
            "social_coefficient" : config.social_coefficent,
            "seed" : config.seed,
        },
    }

    with open(output_path, "w", encoding = "utf-8") as f:
        json.dump(result, f, indent = 4)
        