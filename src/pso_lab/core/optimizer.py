from __future__ import annotations
from time import perf_counter
import numpy as np

from pso_lab.core.boundaries import apply_clamp_bounds
from pso_lab.core.config import PSOConfig
from pso_lab.core.models import SwarmState, OptimizationResult, TimingStats
from pso_lab.objectives import ObjectiveFunction
from pso_lab.parallel.evaluators import FitnessEvaluator, SequentialEvaluator


class PSOOptimizer:
    """Basic sequential PSO implementation (V0)"""

    def __init__(self, config: PSOConfig, objective_function: ObjectiveFunction, evaluator: FitnessEvaluator | None = None):
        self.config = config
        self.objective_function = objective_function
        self.rng = np.random.default_rng(config.seed)
        self.evaluator = evaluator or SequentialEvaluator()
        bounds = np.asarray(self.objective_function.bounds, dtype=float)
        self.lower_bounds = bounds[:, 0]
        self.upper_bounds = bounds[:, 1]
        
    def _initialize_swarm(self)-> SwarmState:
        n = self.config.num_particles
        d = self.config.dimensions
        positions = self.rng.uniform(
            self.lower_bounds,
            self.upper_bounds,
            size = (n, d),
        )
        velocities = np.zeros((n, d), dtype=float)
        
        values = self.evaluator.evaluate(self.objective_function, positions)

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

        r1 = self.rng.random(state.positions.shape)
        r2 = self.rng.random(state.positions.shape)

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
        return state.global_best_value < previous_global_best
    
    def _build_result(
            self,
            state: SwarmState,
            iterations_completed: int,
            best_value_history: list[float],
            swarm_position_history: list[np.ndarray] | None,
            total_time_s: float,
            fitness_time_s: float,
            velocity_update_time_s: float,
            position_update_time_s: float,
    ) -> OptimizationResult:
        return OptimizationResult(
            best_position=state.global_best_position.copy(),
            best_value=float(state.global_best_value),
            iterations_completed=iterations_completed,
            best_value_history=best_value_history,
            timing_stats=TimingStats(
                total_time_s=total_time_s,
                fitness_time_s=fitness_time_s,
                velocity_update_time_s=velocity_update_time_s,
                position_update_time_s=position_update_time_s,
            ),
            swarm_position_history=swarm_position_history,
        )

    def optimize(self) -> OptimizationResult:
        total_start = perf_counter()
        fitness_time_s = 0.0
        velocity_update_time_s = 0.0
        position_update_time_s = 0.0
        init_fitness_start = perf_counter()
        state = self._initialize_swarm()
        fitness_time_s += perf_counter() - init_fitness_start
        best_value_history: list[float] = []
        swarm_position_history: list[np.ndarray] | None = [] if self.config.track_swarm_history else None
        if swarm_position_history is not None:
            swarm_position_history.append(state.positions.copy())
        iterations_without_improvement = 0

        for iteration in range(self.config.max_iterations):
            velocity_start = perf_counter()
            self._update_velocity(state)
            velocity_update_time_s += perf_counter() - velocity_start

            position_start = perf_counter()
            self._update_position(state)
            position_update_time_s += perf_counter() - position_start

            fitness_start = perf_counter()
            values = self.evaluator.evaluate(self.objective_function, state.positions)
            fitness_time_s += perf_counter() - fitness_start

            global_improved = self._update_best(state, values)

            if self.config.track_history:
                best_value_history.append(state.global_best_value)
            if swarm_position_history is not None:
                swarm_position_history.append(state.positions.copy())
            
            if global_improved:
                iterations_without_improvement = 0
            else: iterations_without_improvement += 1

            if self.config.tolerance > 0.0 and state.global_best_value <= self.config.tolerance:
                total_time_s = perf_counter() - total_start
                return self._build_result(
                    state=state,
                    iterations_completed=iteration+1,
                    best_value_history=best_value_history,
                    swarm_position_history=swarm_position_history,
                    total_time_s=total_time_s,
                    fitness_time_s=fitness_time_s,
                    velocity_update_time_s=velocity_update_time_s,
                    position_update_time_s=position_update_time_s,
                )
            if(self.config.stagnation_patience is not None and iterations_without_improvement >= self.config.stagnation_patience):
                total_time_s = perf_counter() - total_start
                return self._build_result(
                    state=state,
                    iterations_completed=iteration+1,
                    best_value_history=best_value_history,
                    swarm_position_history=swarm_position_history,
                    total_time_s=total_time_s,
                    fitness_time_s=fitness_time_s,
                    velocity_update_time_s=velocity_update_time_s,
                    position_update_time_s=position_update_time_s,
                )
        total_time_s = perf_counter() - total_start    
        return self._build_result(
            state=state,
            iterations_completed=self.config.max_iterations,
            best_value_history=best_value_history,
            swarm_position_history=swarm_position_history,
            total_time_s=total_time_s,
            fitness_time_s=fitness_time_s,
            velocity_update_time_s=velocity_update_time_s,
            position_update_time_s=position_update_time_s,
        )
