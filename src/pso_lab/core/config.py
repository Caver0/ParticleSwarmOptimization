from dataclasses import dataclass

@dataclass(slots=True)
class PSOConfig:
    """Configuration parameters for the PSO algorithm"""

    num_particles: int
    dimensions: int
    max_iterations: int

    inertia_weight: float
    cognitive_coefficient: float
    social_coefficent: float

    seed: int | None = None

    