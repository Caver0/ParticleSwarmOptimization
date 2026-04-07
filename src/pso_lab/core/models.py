from __future__ import annotations
from dataclasses import dataclass

import numpy as np

@dataclass(slots=True)
class TimingStats:
    total_time_s: float
    fitness_time_s: float
    velocity_update_time_s: float
    position_update_time_s: float

@dataclass(slots=True)
class Particle:
    """Represents a partcle in the swarn."""

    position: np.ndarray
    velocity: np.ndarray

    best_position: np.ndarray
    best_value: float

@dataclass(slots=True)
class SwarmState:
    """Represents the state of the swarm"""
    
    positions: np.ndarray
    velocities: np.ndarray

    personal_best_positions: np.ndarray
    personal_best_values: np.ndarray

    global_best_position: np.ndarray
    global_best_value: float
    
@dataclass(slots = True)
class OptimizationResult:
    """Stores the result of a PSO optimization run."""

    best_position: np.ndarray
    best_value: float
    iterations_completed: int
    best_value_history: list[float]
    timing_stats: TimingStats
    swarm_position_history: list[np.ndarray] | None = None

