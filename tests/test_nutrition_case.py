from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from run_nutrition_case import (
    DEFAULT_FOODS,
    DEFAULT_TARGETS,
    MODE_LABELS,
    build_meal_tables,
    build_mode_payload,
    build_output_path,
    build_scenario_definition,
    flatten_optimized_foods,
    normalize_dimensions,
    should_show_meal_details_in_all,
)
from pso_lab.core.config import PSOConfig
from pso_lab.experiments.runner import ExperimentResult, run_objective_experiment
from pso_lab.objectives import NutritionObjective


def _sample_foods() -> list[dict[str, float | str]]:
    return [
        {
            "name": "chicken_breast",
            "kcal": 165.0,
            "protein": 31.0,
            "carbs": 0.0,
            "fat": 3.6,
            "fiber": 0.0,
            "sugar": 0.0,
            "salt": 0.18,
            "cost": 1.35,
            "min_g": 0.0,
            "max_g": 250.0,
        },
        {
            "name": "cooked_rice",
            "kcal": 130.0,
            "protein": 2.7,
            "carbs": 28.0,
            "fat": 0.3,
            "fiber": 0.4,
            "sugar": 0.1,
            "salt": 0.00,
            "cost": 0.18,
            "min_g": 0.0,
            "max_g": 350.0,
        },
        {
            "name": "broccoli",
            "kcal": 35.0,
            "protein": 2.8,
            "carbs": 7.0,
            "fat": 0.4,
            "fiber": 3.3,
            "sugar": 1.7,
            "salt": 0.04,
            "cost": 0.20,
            "min_g": 0.0,
            "max_g": 300.0,
        },
    ]


def _sample_args(**overrides) -> SimpleNamespace:
    data = {
        "particles": 12,
        "iterations": 20,
        "inertia": 0.7,
        "c1": 1.5,
        "c2": 1.5,
        "tolerance": 0.0,
        "max_workers": 2,
        "batch_size": 4,
        "async_min_delay": 0.0,
        "async_max_delay": 0.0,
        "max_active_foods": 5,
        "active_threshold_g": 5.0,
        "tiny_threshold": None,
        "active_penalty_weight": 0.10,
        "tiny_penalty_weight": 0.05,
        "fiber_penalty_weight": 1.0,
        "sugar_penalty_weight": 1.0,
        "salt_penalty_weight": 1.0,
        "cost_penalty_weight": 1.0,
        "meal_distribution_penalty_weight": 1.0,
        "output_dir": "results/nutrition",
        "output_path": None,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def _sample_objective() -> NutritionObjective:
    return NutritionObjective(
        foods=_sample_foods(),
        targets={
            "kcal": 700.0,
            "protein": 50.0,
            "carbs": 75.0,
            "fat": 15.0,
        },
        min_fiber=10.0,
        max_sugar=30.0,
        max_salt=4.0,
        max_cost=8.0,
        active_threshold_g=5.0,
        tiny_threshold_g=5.0,
        max_active_foods=2,
    )


def test_nutrition_objective_evaluate_returns_float() -> None:
    objective = _sample_objective()

    value = objective(np.array([150.0, 200.0, 100.0], dtype=float))

    assert isinstance(value, float)


def test_nutrition_objective_evaluate_batch_returns_expected_shape() -> None:
    objective = _sample_objective()
    positions = np.array(
        [
            [150.0, 200.0, 100.0],
            [100.0, 150.0, 50.0],
            [50.0, 100.0, 150.0],
        ],
        dtype=float,
    )

    values = objective.evaluate_batch(positions)

    assert values.shape == (3,)


def test_evaluate_components_sum_to_total_fitness() -> None:
    objective = _sample_objective()
    quantities = np.array([150.0, 200.0, 100.0], dtype=float)

    components = objective.evaluate_components(quantities)

    assert components["total_fitness"] == pytest.approx(
        components["macro_error"]
        + components["fiber_penalty"]
        + components["sugar_penalty"]
        + components["salt_penalty"]
        + components["cost_penalty"]
        + components["active_penalty"]
        + components["tiny_penalty"]
        + components["meal_distribution_penalty"]
    )
    assert objective(quantities) == pytest.approx(components["total_fitness"])


def test_decoded_solution_contains_food_quantities() -> None:
    objective = _sample_objective()

    decoded = objective.decode_solution(np.array([120.0, 180.0, 60.0], dtype=float))

    assert decoded["meal_names"] == ["meal"]
    assert len(decoded["meals"]) == 1
    assert decoded["meals"][0]["foods"][0]["quantity_g"] == pytest.approx(120.0)
    assert decoded["meals"][0]["foods"][1]["quantity_g"] == pytest.approx(180.0)


def test_flatten_optimized_foods_flattens_all_meals() -> None:
    objective = _sample_objective()
    decoded = objective.decode_solution(np.array([120.0, 180.0, 60.0], dtype=float))

    optimized_foods = flatten_optimized_foods(decoded)

    assert len(optimized_foods) == 3
    assert optimized_foods[0]["name"] == "chicken_breast"


def test_full_day_dimension_is_foods_times_meals() -> None:
    args = _sample_args()
    scenario = build_scenario_definition(
        "full_day",
        food_dimension=6,
        food_overrides={"*": _sample_foods() * 2},
        target_overrides={},
        args=args,
    )

    assert scenario.food_dimension == 6
    assert scenario.dimension == 24
    assert scenario.meal_names == ("breakfast", "lunch", "snack", "dinner")


def test_build_output_path_uses_scenario_mode_and_dimension_folders() -> None:
    output_path = build_output_path(
        "results/nutrition",
        "simple_meal",
        MODE_LABELS["vectorized"],
        10,
        42,
    )

    assert (
        str(output_path).replace("\\", "/")
        == "results/nutrition/simple_meal/v4_vectorized/dim_10/result_seed_42.json"
    )


def test_build_mode_payload_includes_main_json_fields() -> None:
    args = _sample_args()
    scenario = build_scenario_definition(
        "simple_meal",
        food_dimension=3,
        food_overrides={"*": _sample_foods()},
        target_overrides={},
        args=args,
    )
    result = ExperimentResult(
        objective_name=scenario.objective.name,
        evaluation_mode="vectorized",
        seed=42,
        best_position=[120.0, 180.0, 60.0],
        best_value=0.1,
        iterations_completed=12,
        elapsed_time_s=0.025,
        best_value_history=[1.0, 0.3, 0.1],
        timing_stats={"fitness_time_s": 0.01},
        config={},
    )

    payload = build_mode_payload(
        result,
        scenario,
        seed=42,
        mode_label=MODE_LABELS["vectorized"],
        config_payload={
            "particles": 12,
            "iterations": 20,
            "food_dimension": scenario.food_dimension,
            "dimension": scenario.dimension,
        },
        async_min_delay=0.0,
        async_max_delay=0.0,
    )

    assert payload["scenario"] == "simple_meal"
    assert payload["mode"] == "vectorized"
    assert payload["mode_label"] == "v4_vectorized"
    assert payload["dimension"] == 3
    assert payload["food_dimension"] == 3
    assert "fitness_components" in payload
    assert "obtained_macros" in payload
    assert "obtained_constraints" in payload
    assert "best_position" in payload
    assert "decoded_solution" in payload
    assert "optimized_foods" in payload
    assert payload["optimized_foods"][0]["quantity_g"] == pytest.approx(120.0)


def test_build_meal_tables_only_show_active_foods_by_default() -> None:
    args = _sample_args()
    scenario = build_scenario_definition(
        "simple_meal",
        food_dimension=3,
        food_overrides={"*": _sample_foods()},
        target_overrides={},
        args=args,
    )
    result = ExperimentResult(
        objective_name=scenario.objective.name,
        evaluation_mode="vectorized",
        seed=42,
        best_position=[80.0, 2.0, 0.0],
        best_value=0.1,
        iterations_completed=10,
        elapsed_time_s=0.01,
        best_value_history=[0.5, 0.1],
        timing_stats={"fitness_time_s": 0.001},
        config={},
    )
    payload = build_mode_payload(
        result,
        scenario,
        seed=42,
        mode_label=MODE_LABELS["vectorized"],
        config_payload={"particles": 12, "iterations": 20},
        async_min_delay=0.0,
        async_max_delay=0.0,
    )

    tables = build_meal_tables(
        payload,
        active_threshold_g=5.0,
        show_all_foods=False,
    )
    table_string = tables[0].get_string()

    assert "chicken_breast" in table_string
    assert "cooked_rice" not in table_string


def test_should_show_meal_details_in_all_defaults_to_true_for_single_run() -> None:
    assert should_show_meal_details_in_all(
        force_show=False,
        scenarios=["simple_meal"],
        dimensions=[10],
        seeds=[42],
    ) is True


def test_should_show_meal_details_in_all_defaults_to_false_for_multiple_runs() -> None:
    assert should_show_meal_details_in_all(
        force_show=False,
        scenarios=["simple_meal", "complex_meal"],
        dimensions=[10],
        seeds=[42],
    ) is False
    assert should_show_meal_details_in_all(
        force_show=True,
        scenarios=["simple_meal", "complex_meal"],
        dimensions=[10, 20],
        seeds=[0, 1],
    ) is True


def test_normalize_dimensions_raises_when_dimension_exceeds_food_count() -> None:
    with pytest.raises(ValueError, match="exceeds available foods"):
        normalize_dimensions([11], available_foods=10)


def test_vectorized_mode_produces_compatible_result() -> None:
    objective = NutritionObjective(
        foods=DEFAULT_FOODS[:4],
        targets=DEFAULT_TARGETS,
        active_threshold_g=5.0,
        tiny_threshold_g=5.0,
        max_active_foods=4,
    )
    config = PSOConfig(
        num_particles=6,
        dimensions=objective.dimensions,
        max_iterations=6,
        inertia_weight=0.7,
        cognitive_coefficient=1.5,
        social_coefficient=1.5,
        seed=7,
        tolerance=0.0,
        stagnation_patience=None,
        track_history=True,
        track_swarm_history=False,
    )

    result = run_objective_experiment(
        objective=objective,
        config=config,
        evaluation_mode="vectorized",
        max_workers=2,
        batch_size=2,
    )

    assert isinstance(result.best_value, float)
    assert len(result.best_position) == objective.dimensions
    assert objective(np.asarray(result.best_position, dtype=float)) == pytest.approx(result.best_value)
