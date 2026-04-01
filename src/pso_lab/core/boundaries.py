from __future__ import annotations

import numpy as np

def apply_clamp_bounds(
        positions: np.ndarray,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
) -> np.ndarray:
    """Clamp particle positions to the search box"""
    return np.clip(positions, lower_bounds, upper_bounds)