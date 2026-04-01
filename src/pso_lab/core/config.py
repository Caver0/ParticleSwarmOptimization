from dataclasses import dataclass

@dataclass(slots=True)
class PSOConfig:
    """Configuration parameters for the PSO algorithm"""

    num_particles: int
    dimensions: int
    max_iterations: int

    inertia_weight: float
    cognitive_coefficient: float
    social_coefficient: float

    seed: int | None = None

    tolerance: float = 0.0
    stagnation_patience: int | None = None
    track_history: bool = True
    