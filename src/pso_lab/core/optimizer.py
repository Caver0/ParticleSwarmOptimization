from __future__ import annotations

import numpy as np

from pso_lab.core.config import PSOConfig
from pso_lab.core.models import SwarmState
from pso_lab.objectives import ObjectiveFunction

class PSOOptimizer:
    """Basic sequential PSO implementation (V0)"""

    def __init__(self, config: PSOConfig, objective_function: ObjectiveFunction):
        self.config = config
        self.objective_function = objective_function

        if config.seed is not None:
            np.random.seed(config.seed)
        
    def _initialize_swarm(self)-> SwarmState:
        n = self.config.num_particles
        d = self.config.dimensions

        bounds = np.array(self.objective_function.bounds)
        lower = bounds[:, 0]
        upper = bounds[:, 1]
        positions = np.random.uniform(lower, upper, (n, d))
        velocities = np.zeros((n, d))
        
        values = self.objective_function.evaluate_many(positions)

        personal_best_positions = positions.copy()
        personal_best_values = values.copy()

        best_idx = np.argmin(values)

        global_best_position = positions[best_idx].copy()
        global_best_value = values[best_idx]

        return SwarmState(
            positions=positions,    
            velocities=velocities,
            personal_best_positions=personal_best_positions,
            personal_best_values=personal_best_values,
            global_best_position=global_best_position,
            global_best_value=global_best_value
        )
    
    def _update_velocity(self, state: SwarmState) -> None:
        w = self.config.inertia_weight
        c1 = self.config.cognitive_coefficient
        c2 = self.config.social_coefficient

        r1 = np.random.rand(*state.positions.shape)
        r2 = np.random.rand(*state.positions.shape)

        cognitive = c1 * r1 * (state.personal_best_positions - state.positions)
        social = c2 * r2 * (state.global_best_position - state.positions)
        state.velocities = w * state.velocities + cognitive + social

    def _update_position(self, state: SwarmState) -> None:
        state.positions += state.velocities

        bounds = np.array(self.objective_function.bounds)
        lower = bounds[:, 0]
        upper = bounds[:, 1]

        state.positions = np.clip(state.positions, lower, upper)

    def _update_best(self, state: SwarmState, values: np.ndarray) -> None:
        improved = values < state.personal_best_values

        state.personal_best_positions[improved] = state.positions[improved]
        state.personal_best_values[improved] = values[improved]

        best_idx = np.argmin(state.personal_best_values)

        state.global_best_position = state.personal_best_positions[best_idx].copy()
        state.global_best_value = state.personal_best_values[best_idx]
    
    def optimize(self) -> tuple[np.ndarray, float]:
        state = self._initialize_swarm()

        for _ in range(self.config.max_iterations):
            self._update_velocity(state)
            self._update_position(state)

            values = self.objective_function.evaluate_many(state.positions)
            self._update_best(state, values)

        return state.global_best_position, state.global_best_value