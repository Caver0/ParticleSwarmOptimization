from __future__ import annotations

import numpy as np

from pso_lab.core.boundaries import apply_clamp_bounds
from pso_lab.core.config import PSOConfig
from pso_lab.core.models import SwarmState, OptimizationResult
from pso_lab.objectives import ObjectiveFunction

class PSOOptimizer:
    """Basic sequential PSO implementation (V0)"""

    def __init__(self, config: PSOConfig, objective_function: ObjectiveFunction):
        self.config = config
        self.objective_function = objective_function
        self.rng = np.random.default_rng(config.seed)
        
        bounds = np.asarray(self.objective_function, dtype=float)
        self.lower_bounds = bounds[:, 0]
        self.upper_bounds = bounds[:, 1]
        
    def _initialize_swarm(self)-> SwarmState:
        n = self.config.num_particles
        d = self.config.dimensions

        bounds = np.array(self.objective_function.bounds)
        lower = bounds[:, 0]
        upper = bounds[:, 1]
        positions = self.rng.uniform(
            self.lower_bounds,
            self.upper_bounds,
            size = (n, d),
        )
        velocities = np.zeros((n, d), dtype=float)
        
        values = self.objective_function.evaluate_many(positions)

        personal_best_positions = positions.copy()
        personal_best_values = values.copy()

        best_idx = np.argmin(values)

        global_best_position = positions[best_idx].copy()
        global_best_value = float(values[best_idx])

        return SwarmState(
            positions=positions,    
            velocities=velocities,
            personal_best_positions=personal_best_positions,
            personal_best_values=personal_best_values,
            global_best_position=global_best_position,
            global_best_value=global_best_value,
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
        state.positions = state.positions + state.velocities
        state.positions = apply_clamp_bounds(
            state.positions,
            self.lower_bounds,
            self.upper_bounds,
        )
    
    def _update_best(self, state: SwarmState, values: np.ndarray) -> bool:
        improved = values < state.personal_best_values

        state.personal_best_positions[improved] = state.positions[improved]
        state.personal_best_values[improved] = values[improved]

        previous_global_best = state.global_best_value

        best_idx = np.argmin(state.personal_best_values)
        state.global_best_position = state.personal_best_positions[best_idx].copy()
        state.global_best_value = float(state.personal_best_values[best_idx])
    
    def optimize(self) -> OptimizationResult:
        state = self._initialize_swarm()
        best_value_history: list[float] = []
        iterations_without_improvement = 0

        for iteration in range(self.config.max_iterations):
            self._update_velocity(state)
            self._update_position(state)
            
            values = self.objective_function.evaluate_many(state.positions)
            global_improved = self._update_best(state, values)

            if self.config.track_history:
                best_value_history.append(state.global_best_value)
            
            if global_improved:
                iterations_without_improvement = 0
            else: iterations_without_improvement += 1

            if self.config.tolerance > 0.0 and state.global_best_value <= self.config.tolerance:
                return OptimizationResult(
                    best_position=state.global_best_position.copy(),
                    best_value=float(state.global_best_value),
                    iterations_completed= iteration + 1,
                    best_value_history=best_value_history,
                )
            if(self.config.stagnation_patience is not None and iterations_without_improvement >= self.config.stagnation_patience):
                return OptimizationResult(
                    best_position=state.global_best_position.copy(),
                    best_value=float(state.global_best_value),
                    iterations_completed=iteration+1,
                    best_value_history=best_value_history,
                )
        return OptimizationResult(
            best_position=state.global_best_position.copy(),
            best_value=float(state.global_best_value),
            iterations_completed=self.config.max_iterations,
            best_value_history=best_value_history,
        )