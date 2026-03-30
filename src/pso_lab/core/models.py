from dataclasses import dataclass

import numpy as np

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
    