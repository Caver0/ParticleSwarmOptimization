from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from prettytable import PrettyTable

from _repo_bootstrap import bootstrap_src_path

bootstrap_src_path()

from pso_lab.cli import parse_nutrition_case_args
from pso_lab.core.config import PSOConfig
from pso_lab.experiments.runner import ExperimentResult, run_objective_experiment
from pso_lab.io.logging_utils import setup_logger
from pso_lab.io.results import save_json_document
from pso_lab.objectives import NutritionObjective


CASE_NAME = "nutrition_optimization"
MACRO_NAMES = ("kcal", "protein", "carbs", "fat")
CONSTRAINT_LIMIT_KEYS = ("min_fiber", "max_sugar", "max_salt", "max_cost")
DAY_MEAL_NAMES = ("breakfast", "lunch", "snack", "dinner")
MODE_LABELS = {
    "sequential": "v0_sequential",
    "threading": "v1_threading",
    "multiprocessing": "v2_multiprocessing",
    "asyncio": "v3_asyncio",
    "vectorized": "v4_vectorized",
}
SUPPORTED_MODES = tuple(MODE_LABELS)
SUPPORTED_SCENARIOS = (
    "simple_meal",
    "complex_meal",
    "full_day",
    "full_day_constrained",
)


def _food(
    name: str,
    kcal: float,
    protein: float,
    carbs: float,
    fat: float,
    fiber: float,
    sugar: float,
    salt: float,
    cost: float,
    min_g: float,
    max_g: float,
) -> dict[str, float | str]:
    return {
        "name": name,
        "kcal": kcal,
        "protein": protein,
        "carbs": carbs,
        "fat": fat,
        "fiber": fiber,
        "sugar": sugar,
        "salt": salt,
        "cost": cost,
        "min_g": min_g,
        "max_g": max_g,
    }


EXPANDED_FOODS = [
    _food("chicken_breast", 165.0, 31.0, 0.0, 3.6, 0.0, 0.0, 0.18, 1.35, 0.0, 250.0),
    _food("cooked_rice", 130.0, 2.7, 28.0, 0.3, 0.4, 0.1, 0.00, 0.18, 0.0, 350.0),
    _food("olive_oil", 884.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.00, 0.55, 0.0, 30.0),
    _food("banana", 89.0, 1.1, 22.8, 0.3, 2.6, 12.2, 0.00, 0.22, 0.0, 250.0),
    _food("greek_yogurt", 59.0, 10.0, 3.6, 0.4, 0.0, 3.2, 0.09, 0.55, 0.0, 250.0),
    _food("tuna", 132.0, 29.0, 0.0, 1.0, 0.0, 0.0, 0.30, 1.20, 0.0, 160.0),
    _food("cooked_pasta", 158.0, 5.8, 30.9, 0.9, 1.8, 0.6, 0.00, 0.20, 0.0, 350.0),
    _food("whole_wheat_bread", 247.0, 13.0, 41.0, 4.2, 6.0, 5.2, 0.47, 0.30, 0.0, 150.0),
    _food("egg", 143.0, 13.0, 1.1, 9.5, 0.0, 1.1, 0.35, 0.42, 0.0, 150.0),
    _food("avocado", 160.0, 2.0, 8.5, 14.7, 6.7, 0.7, 0.02, 0.75, 0.0, 150.0),
    _food("oats", 389.0, 16.9, 66.3, 6.9, 10.6, 1.0, 0.01, 0.22, 0.0, 120.0),
    _food("salmon", 208.0, 20.4, 0.0, 13.4, 0.0, 0.0, 0.12, 1.80, 0.0, 220.0),
    _food("broccoli", 35.0, 2.8, 7.0, 0.4, 3.3, 1.7, 0.04, 0.20, 0.0, 300.0),
    _food("apple", 52.0, 0.3, 13.8, 0.2, 2.4, 10.4, 0.00, 0.24, 0.0, 250.0),
    _food("peanut_butter", 588.0, 25.0, 20.0, 50.0, 6.0, 9.0, 0.60, 0.70, 0.0, 60.0),
    _food("cottage_cheese", 98.0, 11.1, 3.4, 4.3, 0.0, 2.7, 0.36, 0.45, 0.0, 250.0),
    _food("sweet_potato", 86.0, 1.6, 20.1, 0.1, 3.0, 4.2, 0.14, 0.18, 0.0, 300.0),
    _food("black_beans", 132.0, 8.9, 23.7, 0.5, 8.7, 0.3, 0.24, 0.20, 0.0, 250.0),
    _food("almonds", 579.0, 21.2, 21.6, 49.9, 12.5, 4.4, 0.01, 1.10, 0.0, 80.0),
    _food("spinach", 23.0, 2.9, 3.6, 0.4, 2.2, 0.4, 0.20, 0.22, 0.0, 200.0),
    _food("tofu", 76.0, 8.0, 1.9, 4.8, 0.3, 0.6, 0.02, 0.35, 0.0, 250.0),
    _food("milk", 61.0, 3.2, 4.8, 3.3, 0.0, 5.0, 0.10, 0.10, 0.0, 400.0),
    _food("quinoa_cooked", 120.0, 4.4, 21.3, 1.9, 2.8, 0.9, 0.01, 0.35, 0.0, 300.0),
    _food("potato_boiled", 87.0, 1.9, 20.1, 0.1, 1.8, 0.9, 0.02, 0.10, 0.0, 350.0),
    _food("strawberries", 32.0, 0.7, 7.7, 0.3, 2.0, 4.9, 0.00, 0.45, 0.0, 250.0),
    _food("turkey_breast", 135.0, 29.0, 0.0, 1.6, 0.0, 0.0, 0.12, 1.30, 0.0, 250.0),
    _food("chickpeas_cooked", 164.0, 8.9, 27.4, 2.6, 7.6, 4.8, 0.02, 0.25, 0.0, 250.0),
    _food("cheddar_cheese", 403.0, 25.0, 1.3, 33.1, 0.0, 0.5, 1.62, 1.20, 0.0, 80.0),
    _food("tomato", 18.0, 0.9, 3.9, 0.2, 1.2, 2.6, 0.01, 0.15, 0.0, 250.0),
    _food("cucumber", 15.0, 0.7, 3.6, 0.1, 0.5, 1.7, 0.00, 0.12, 0.0, 250.0),
    _food("honey", 304.0, 0.3, 82.4, 0.0, 0.0, 82.1, 0.01, 0.65, 0.0, 40.0),
    _food("lentils_cooked", 116.0, 9.0, 20.1, 0.4, 7.9, 1.8, 0.01, 0.18, 0.0, 250.0),
]

DEFAULT_FOODS = [dict(food) for food in EXPANDED_FOODS[:10]]
DEFAULT_TARGETS = {
    "kcal": 720.0,
    "protein": 45.0,
    "carbs": 90.0,
    "fat": 20.0,
}
COMPLEX_MEAL_TARGETS = {
    "kcal": 950.0,
    "protein": 60.0,
    "carbs": 110.0,
    "fat": 28.0,
}
FULL_DAY_TARGETS = {
    "kcal": 2400.0,
    "protein": 160.0,
    "carbs": 260.0,
    "fat": 75.0,
}
FULL_DAY_CONSTRAINED_TARGETS = {
    "kcal": 2300.0,
    "protein": 165.0,
    "carbs": 240.0,
    "fat": 70.0,
}
FULL_DAY_KCAL_RANGES = {
    "breakfast": (450.0, 650.0),
    "lunch": (700.0, 900.0),
    "snack": (250.0, 400.0),
    "dinner": (600.0, 800.0),
}
FULL_DAY_CONSTRAINED_KCAL_RANGES = {
    "breakfast": (450.0, 600.0),
    "lunch": (700.0, 850.0),
    "snack": (250.0, 350.0),
    "dinner": (600.0, 750.0),
}


@dataclass(slots=True)
class ScenarioDefinition:
    name: str
    foods: list[dict[str, object]]
    food_dimension: int
    objective: NutritionObjective
    target_macros: dict[str, float]
    target_constraints: dict[str, object]
    meal_names: tuple[str, ...]

    @property
    def dimension(self) -> int:
        return self.objective.dimensions


def _copy_foods(foods: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    copied_foods: list[dict[str, object]] = []
    for index, food in enumerate(foods):
        if not isinstance(food, Mapping):
            raise ValueError(f"Food at index {index} must be a JSON object")
        copied_foods.append(dict(food))
    return copied_foods


def _copy_mapping(mapping: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(mapping, Mapping):
        raise ValueError("Expected a JSON object")
    return dict(mapping)


def _load_json_file(path: str | None, *, label: str) -> object | None:
    if path is None:
        return None

    file_path = Path(path)
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if label == "foods" and not isinstance(data, (list, dict)):
        raise ValueError("foods JSON must contain a top-level list or scenario mapping")
    if label == "targets" and not isinstance(data, dict):
        raise ValueError("targets JSON must contain a top-level object")
    return data


def _scenario_templates() -> dict[str, dict[str, object]]:
    return {
        "simple_meal": {
            "foods": _copy_foods(DEFAULT_FOODS),
            "targets": dict(DEFAULT_TARGETS),
            "meal_names": ("meal",),
            "objective_name": "nutrition_simple_meal",
            "target_constraints": {},
        },
        "complex_meal": {
            "foods": _copy_foods(EXPANDED_FOODS),
            "targets": dict(COMPLEX_MEAL_TARGETS),
            "meal_names": ("meal",),
            "objective_name": "nutrition_complex_meal",
            "target_constraints": {},
        },
        "full_day": {
            "foods": _copy_foods(EXPANDED_FOODS),
            "targets": dict(FULL_DAY_TARGETS),
            "meal_names": DAY_MEAL_NAMES,
            "objective_name": "nutrition_full_day",
            "target_constraints": {
                "meal_kcal_ranges": dict(FULL_DAY_KCAL_RANGES),
            },
        },
        "full_day_constrained": {
            "foods": _copy_foods(EXPANDED_FOODS),
            "targets": dict(FULL_DAY_CONSTRAINED_TARGETS),
            "meal_names": DAY_MEAL_NAMES,
            "objective_name": "nutrition_full_day_constrained",
            "target_constraints": {
                "min_fiber": 32.0,
                "max_sugar": 90.0,
                "max_salt": 6.0,
                "max_cost": 13.0,
                "meal_kcal_ranges": dict(FULL_DAY_CONSTRAINED_KCAL_RANGES),
            },
        },
    }


def _normalize_food_overrides(
    data: object | None,
) -> dict[str, list[dict[str, object]]]:
    if data is None:
        return {}
    if isinstance(data, list):
        return {"*": _copy_foods(data)}
    if not isinstance(data, Mapping):
        raise ValueError("foods JSON must contain a top-level list or scenario mapping")

    normalized: dict[str, list[dict[str, object]]] = {}
    for scenario_name, foods in data.items():
        if scenario_name not in SUPPORTED_SCENARIOS:
            raise ValueError(
                f"Unknown scenario '{scenario_name}' in foods JSON. "
                f"Expected one of {SUPPORTED_SCENARIOS}"
            )
        if not isinstance(foods, list):
            raise ValueError(
                f"foods JSON for scenario '{scenario_name}' must be a list"
            )
        normalized[str(scenario_name)] = _copy_foods(foods)
    return normalized


def _normalize_target_overrides(
    data: object | None,
) -> dict[str, dict[str, object]]:
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ValueError("targets JSON must contain a top-level object")

    if set(data).issubset(set(SUPPORTED_SCENARIOS)):
        normalized: dict[str, dict[str, object]] = {}
        for scenario_name, scenario_config in data.items():
            if not isinstance(scenario_config, Mapping):
                raise ValueError(
                    f"targets JSON for scenario '{scenario_name}' must be an object"
                )
            normalized[str(scenario_name)] = dict(scenario_config)
        return normalized

    return {"*": dict(data)}


def _resolve_food_catalog(
    scenario_name: str,
    food_overrides: Mapping[str, list[dict[str, object]]],
) -> list[dict[str, object]]:
    templates = _scenario_templates()
    if scenario_name in food_overrides:
        return _copy_foods(food_overrides[scenario_name])
    if "*" in food_overrides:
        return _copy_foods(food_overrides["*"])
    return _copy_foods(templates[scenario_name]["foods"])


def _resolve_target_override(
    scenario_name: str,
    target_overrides: Mapping[str, dict[str, object]],
) -> dict[str, object]:
    if scenario_name in target_overrides:
        return dict(target_overrides[scenario_name])
    if "*" in target_overrides:
        return dict(target_overrides["*"])
    return {}


def load_case_data(
    foods_path: str | None,
    targets_path: str | None,
) -> tuple[dict[str, list[dict[str, object]]], dict[str, dict[str, object]]]:
    foods_data = _load_json_file(foods_path, label="foods")
    targets_data = _load_json_file(targets_path, label="targets")
    return _normalize_food_overrides(foods_data), _normalize_target_overrides(targets_data)


def normalize_dimensions(
    dimensions: Sequence[int],
    available_foods: int,
) -> list[int]:
    if not dimensions:
        raise ValueError("At least one nutrition dimension must be provided")

    normalized_dimensions: list[int] = []
    for dimension in dimensions:
        normalized_dimension = int(dimension)
        if normalized_dimension < 1:
            raise ValueError("Nutrition dimensions must be >= 1")
        if normalized_dimension > available_foods:
            raise ValueError(
                "Nutrition dimension "
                f"{normalized_dimension} exceeds available foods ({available_foods})"
            )
        normalized_dimensions.append(normalized_dimension)
    return normalized_dimensions


def select_foods_for_dimension(
    foods: Sequence[Mapping[str, object]],
    dimension: int,
) -> list[dict[str, object]]:
    if dimension < 1:
        raise ValueError("Nutrition dimensions must be >= 1")
    if dimension > len(foods):
        raise ValueError(
            f"Nutrition dimension {dimension} exceeds available foods ({len(foods)})"
        )
    return _copy_foods(foods[:dimension])


def build_output_path(
    output_dir: str | Path,
    scenario_name: str,
    mode_label: str,
    dimension: int,
    seed: int | None,
) -> Path:
    seed_label = "none" if seed is None else str(seed)
    return (
        Path(output_dir)
        / scenario_name
        / mode_label
        / f"dim_{dimension}"
        / f"result_seed_{seed_label}.json"
    )


def build_global_summary_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / "global_summary.json"


def build_pso_config(args, *, dimension: int, seed: int | None) -> PSOConfig:
    return PSOConfig(
        num_particles=args.particles,
        dimensions=dimension,
        max_iterations=args.iterations,
        inertia_weight=args.inertia,
        cognitive_coefficient=args.c1,
        social_coefficient=args.c2,
        seed=seed,
        tolerance=args.tolerance,
        stagnation_patience=None,
        track_history=True,
        track_swarm_history=False,
    )


def build_penalty_weights(args) -> dict[str, float]:
    return {
        "fiber_penalty": float(args.fiber_penalty_weight),
        "sugar_penalty": float(args.sugar_penalty_weight),
        "salt_penalty": float(args.salt_penalty_weight),
        "cost_penalty": float(args.cost_penalty_weight),
        "active_penalty": float(args.active_penalty_weight),
        "tiny_penalty": float(args.tiny_penalty_weight),
        "meal_distribution_penalty": float(args.meal_distribution_penalty_weight),
    }


def _merge_target_macros(
    base_macros: Mapping[str, object],
    override: Mapping[str, object],
) -> dict[str, float]:
    merged = {macro_name: float(base_macros[macro_name]) for macro_name in MACRO_NAMES}
    if "targets" in override:
        nested_targets = override["targets"]
        if not isinstance(nested_targets, Mapping):
            raise ValueError("targets override field 'targets' must be a JSON object")
        for macro_name in MACRO_NAMES:
            if macro_name in nested_targets:
                merged[macro_name] = float(nested_targets[macro_name])
    for macro_name in MACRO_NAMES:
        if macro_name in override:
            merged[macro_name] = float(override[macro_name])
    return merged


def _merge_target_constraints(
    base_constraints: Mapping[str, object],
    override: Mapping[str, object],
) -> dict[str, object]:
    merged = dict(base_constraints)
    for key in CONSTRAINT_LIMIT_KEYS:
        if key in override:
            value = override[key]
            merged[key] = None if value is None else float(value)

    if "meal_kcal_ranges" in override:
        raw_ranges = override["meal_kcal_ranges"]
        if not isinstance(raw_ranges, Mapping):
            raise ValueError("meal_kcal_ranges override must be a JSON object")
        normalized_ranges: dict[str, tuple[float, float]] = {}
        for meal_name, limits in raw_ranges.items():
            if not isinstance(limits, Sequence) or len(limits) != 2:
                raise ValueError(
                    f"meal_kcal_ranges['{meal_name}'] must contain two numeric values"
                )
            normalized_ranges[str(meal_name)] = (float(limits[0]), float(limits[1]))
        merged["meal_kcal_ranges"] = normalized_ranges
    return merged


def build_scenario_definition(
    scenario_name: str,
    *,
    food_dimension: int,
    food_overrides: Mapping[str, list[dict[str, object]]],
    target_overrides: Mapping[str, dict[str, object]],
    args,
) -> ScenarioDefinition:
    templates = _scenario_templates()
    template = templates[scenario_name]
    available_foods = _resolve_food_catalog(scenario_name, food_overrides)
    selected_foods = select_foods_for_dimension(available_foods, food_dimension)
    target_override = _resolve_target_override(scenario_name, target_overrides)

    target_macros = _merge_target_macros(template["targets"], target_override)
    target_constraints = _merge_target_constraints(
        _copy_mapping(template["target_constraints"]),
        target_override,
    )
    meal_names = tuple(template["meal_names"])

    objective = NutritionObjective(
        foods=selected_foods,
        targets=target_macros,
        meal_names=meal_names,
        penalty_weights=build_penalty_weights(args),
        active_threshold_g=float(args.active_threshold_g),
        tiny_threshold_g=(
            float(args.active_threshold_g)
            if args.tiny_threshold is None
            else float(args.tiny_threshold)
        ),
        max_active_foods=int(args.max_active_foods),
        min_fiber=target_constraints.get("min_fiber"),
        max_sugar=target_constraints.get("max_sugar"),
        max_salt=target_constraints.get("max_salt"),
        max_cost=target_constraints.get("max_cost"),
        meal_kcal_ranges=target_constraints.get("meal_kcal_ranges"),
        objective_name=str(template["objective_name"]),
    )

    return ScenarioDefinition(
        name=scenario_name,
        foods=selected_foods,
        food_dimension=food_dimension,
        objective=objective,
        target_macros=target_macros,
        target_constraints=target_constraints,
        meal_names=meal_names,
    )


def resolve_modes(args) -> list[str]:
    requested_modes = list(args.modes) if args.modes else [args.mode]
    if "all" in requested_modes:
        return list(SUPPORTED_MODES)

    resolved_modes: list[str] = []
    for mode in requested_modes:
        if mode not in SUPPORTED_MODES:
            raise ValueError(f"Unsupported nutrition mode: {mode}")
        if mode not in resolved_modes:
            resolved_modes.append(mode)
    return resolved_modes


def resolve_scenarios(args) -> list[str]:
    requested_scenarios = list(args.scenarios) if args.scenarios else [args.scenario]
    resolved_scenarios: list[str] = []
    for scenario_name in requested_scenarios:
        if scenario_name not in SUPPORTED_SCENARIOS:
            raise ValueError(f"Unsupported nutrition scenario: {scenario_name}")
        if scenario_name not in resolved_scenarios:
            resolved_scenarios.append(scenario_name)
    return resolved_scenarios


def resolve_seeds(args) -> list[int]:
    if args.seeds:
        return [int(seed) for seed in args.seeds]
    return [int(args.seed)]


def should_show_meal_details_in_all(
    *,
    force_show: bool,
    scenarios: Sequence[str],
    dimensions: Sequence[int],
    seeds: Sequence[int],
) -> bool:
    return force_show or (
        len(scenarios) == 1 and len(dimensions) == 1 and len(seeds) == 1
    )


def flatten_optimized_foods(decoded_solution: Mapping[str, object]) -> list[dict[str, object]]:
    optimized_foods: list[dict[str, object]] = []
    for meal in decoded_solution["meals"]:
        for food in meal["foods"]:
            optimized_foods.append(dict(food))
    return optimized_foods


def build_case_config(args, scenario: ScenarioDefinition) -> dict[str, object]:
    return {
        "particles": int(args.particles),
        "iterations": int(args.iterations),
        "inertia": float(args.inertia),
        "c1": float(args.c1),
        "c2": float(args.c2),
        "food_dimension": int(scenario.food_dimension),
        "dimension": int(scenario.dimension),
        "meals": list(scenario.meal_names),
        "tolerance": float(args.tolerance),
        "max_workers": args.max_workers,
        "batch_size": args.batch_size,
        "async_min_delay": float(args.async_min_delay),
        "async_max_delay": float(args.async_max_delay),
        "max_active_foods": int(args.max_active_foods),
        "active_threshold_g": float(args.active_threshold_g),
        "tiny_threshold_g": (
            float(args.active_threshold_g)
            if args.tiny_threshold is None
            else float(args.tiny_threshold)
        ),
        "penalty_weights": build_penalty_weights(args),
        "output_dir": args.output_dir,
        "output_path": args.output_path,
    }


def build_mode_payload(
    result: ExperimentResult,
    scenario: ScenarioDefinition,
    *,
    seed: int | None,
    mode_label: str,
    config_payload: Mapping[str, object],
    async_min_delay: float,
    async_max_delay: float,
) -> dict[str, object]:
    best_position = np.asarray(result.best_position, dtype=float)
    obtained_macros = scenario.objective.compute_totals(best_position)
    obtained_constraints = scenario.objective.compute_obtained_constraints(best_position)
    relative_errors = scenario.objective.compute_relative_errors(best_position)
    fitness_components = scenario.objective.evaluate_components(best_position)
    decoded_solution = scenario.objective.decode_solution(best_position)
    optimized_foods = flatten_optimized_foods(decoded_solution)

    payload = {
        "case": CASE_NAME,
        "scenario": scenario.name,
        "objective_name": result.objective_name,
        "mode": result.evaluation_mode,
        "mode_label": mode_label,
        "dimension": int(scenario.dimension),
        "food_dimension": int(scenario.food_dimension),
        "meals": list(scenario.meal_names),
        "seed": seed,
        "particles": int(config_payload["particles"]),
        "iterations": int(config_payload["iterations"]),
        "config": dict(config_payload),
        "foods": _copy_foods(scenario.foods),
        "target_macros": dict(scenario.target_macros),
        "target_constraints": dict(scenario.target_constraints),
        "best_fitness": float(fitness_components["total_fitness"]),
        "best_position": best_position.tolist(),
        "optimized_foods": optimized_foods,
        "decoded_solution": decoded_solution,
        "obtained_macros": obtained_macros,
        "obtained_constraints": obtained_constraints,
        "relative_errors": relative_errors,
        "fitness_components": fitness_components,
        "active_foods": int(decoded_solution["active_foods_total"]),
        "elapsed_time": float(result.elapsed_time_s),
        "elapsed_time_s": float(result.elapsed_time_s),
        "iterations_completed": int(result.iterations_completed),
        "convergence_history": list(result.best_value_history),
        "best_value_history": list(result.best_value_history),
        "timing_stats": dict(result.timing_stats),
    }

    if result.evaluation_mode == "asyncio":
        payload["async_delay_window_s"] = {
            "min": float(async_min_delay),
            "max": float(async_max_delay),
        }

    return payload


def build_single_output_document(*, mode_payload: Mapping[str, object]) -> dict[str, object]:
    return dict(mode_payload)


def build_global_output_document(
    *,
    output_dir: str | Path,
    modes: Sequence[str],
    scenarios: Sequence[str],
    food_dimensions: Sequence[int],
    seeds: Sequence[int],
    results: Sequence[Mapping[str, object]],
    mode_summaries: Sequence[Mapping[str, object]],
    scenario_summaries: Sequence[Mapping[str, object]],
    scenario_mode_summaries: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "case": CASE_NAME,
        "output_dir": str(output_dir),
        "modes": list(modes),
        "scenarios": list(scenarios),
        "food_dimensions": list(food_dimensions),
        "seeds": list(seeds),
        "results": [dict(payload) for payload in results],
        "mode_summaries": [dict(summary) for summary in mode_summaries],
        "scenario_summaries": [dict(summary) for summary in scenario_summaries],
        "scenario_mode_summaries": [dict(summary) for summary in scenario_mode_summaries],
    }


def summarize_payloads(
    payloads: Sequence[Mapping[str, object]],
    *,
    group_label: str,
    group_type: str,
) -> dict[str, object]:
    if not payloads:
        raise ValueError("Cannot summarize an empty list of nutrition results")

    fitness_values = np.asarray([float(payload["best_fitness"]) for payload in payloads], dtype=float)
    macro_errors = np.asarray(
        [float(payload["fitness_components"]["macro_error"]) for payload in payloads],
        dtype=float,
    )
    elapsed_times = np.asarray([float(payload["elapsed_time"]) for payload in payloads], dtype=float)

    return {
        "group_type": group_type,
        "label": group_label,
        "runs": len(payloads),
        "modes": sorted({str(payload["mode_label"]) for payload in payloads}),
        "scenarios": sorted({str(payload["scenario"]) for payload in payloads}),
        "food_dimensions": sorted({int(payload["food_dimension"]) for payload in payloads}),
        "dimensions": sorted({int(payload["dimension"]) for payload in payloads}),
        "seeds": sorted({int(payload["seed"]) for payload in payloads if payload["seed"] is not None}),
        "mean_fitness": float(np.mean(fitness_values)),
        "best_fitness": float(np.min(fitness_values)),
        "mean_macro_error": float(np.mean(macro_errors)),
        "mean_time_s": float(np.mean(elapsed_times)),
        "total_time_s": float(np.sum(elapsed_times)),
    }


def _format_fitness(value: float) -> str:
    return f"{value:.6e}"


def _format_percent(value: float) -> str:
    return f"{value * 100.0:.2f}"


def _format_list(values: Iterable[object]) -> str:
    return ", ".join(str(value) for value in values)


def _format_limit(prefix: str, value: object | None) -> str:
    if value is None:
        return "n/a"
    return f"{prefix}{float(value):.2f}"


def _print_table(table: PrettyTable) -> None:
    print()
    print(table)


def _build_configuration_table(
    *,
    modes: Sequence[str],
    scenarios: Sequence[str],
    dimensions: Sequence[int],
    seeds: Sequence[int],
    args,
) -> PrettyTable:
    table = PrettyTable()
    table.title = "Nutrition Configuration"
    table.field_names = ["Setting", "Value"]
    table.align["Setting"] = "l"
    table.align["Value"] = "l"
    rows = [
        ("Modes", _format_list(MODE_LABELS[mode] for mode in modes)),
        ("Scenarios", _format_list(scenarios)),
        ("Food dimensions", _format_list(dimensions)),
        ("Seeds", _format_list(seeds)),
        ("Particles", args.particles),
        ("Iterations", args.iterations),
        ("Inertia", f"{args.inertia:.2f}"),
        ("c1", f"{args.c1:.2f}"),
        ("c2", f"{args.c2:.2f}"),
        ("Max active foods / meal", args.max_active_foods),
        ("Active threshold (g)", f"{args.active_threshold_g:.2f}"),
        (
            "Tiny threshold (g)",
            f"{(args.active_threshold_g if args.tiny_threshold is None else args.tiny_threshold):.2f}",
        ),
        ("Output dir", args.output_dir),
    ]
    if args.foods_path:
        rows.append(("Foods override", args.foods_path))
    if args.targets_path:
        rows.append(("Targets override", args.targets_path))
    for setting, value in rows:
        table.add_row([setting, value])
    return table


def _build_scenario_table(scenario: ScenarioDefinition) -> PrettyTable:
    table = PrettyTable()
    table.title = f"Scenario Setup | {scenario.name}"
    table.field_names = ["Setting", "Value"]
    table.align["Setting"] = "l"
    table.align["Value"] = "l"
    rows = [
        ("Scenario", scenario.name),
        ("Selected foods", scenario.food_dimension),
        ("Meals", _format_list(scenario.meal_names)),
        ("Real dimension", scenario.dimension),
    ]
    for setting, value in rows:
        table.add_row([setting, value])
    return table


def _build_targets_table(scenario: ScenarioDefinition) -> PrettyTable:
    table = PrettyTable()
    table.title = f"Target Macros | {scenario.name}"
    table.field_names = ["Macro", "Target"]
    table.align["Macro"] = "l"
    for macro_name in MACRO_NAMES:
        table.add_row([macro_name, f"{float(scenario.target_macros[macro_name]):.2f}"])
    return table


def _active_food_entries(
    food_entries: Sequence[Mapping[str, object]],
    *,
    active_threshold_g: float,
    show_all_foods: bool,
) -> list[Mapping[str, object]]:
    if show_all_foods:
        return list(food_entries)

    active_foods = [
        food
        for food in food_entries
        if float(food["quantity_g"]) > active_threshold_g
    ]
    return active_foods or list(food_entries)


def build_meal_tables(
    payload: Mapping[str, object],
    *,
    active_threshold_g: float,
    show_all_foods: bool,
) -> list[PrettyTable]:
    tables: list[PrettyTable] = []
    decoded_solution = payload["decoded_solution"]
    meals = list(decoded_solution["meals"])

    for meal in meals:
        table = PrettyTable()
        title = (
            "OPTIMIZED MEAL | "
            f"scenario={payload['scenario']} | mode={payload['mode_label']} | "
            f"dimension={int(payload['dimension'])} | seed={payload['seed']}"
        )
        if len(meals) > 1:
            title += f" | meal={meal['meal']}"
        table.title = title
        table.field_names = [
            "Food",
            "Quantity (g)",
            "kcal",
            "protein",
            "carbs",
            "fat",
        ]
        table.align["Food"] = "l"

        for food in _active_food_entries(
            meal["foods"],
            active_threshold_g=active_threshold_g,
            show_all_foods=show_all_foods,
        ):
            table.add_row(
                [
                    food["name"],
                    f"{float(food['quantity_g']):.2f}",
                    f"{float(food['kcal']):.2f}",
                    f"{float(food['protein']):.2f}",
                    f"{float(food['carbs']):.2f}",
                    f"{float(food['fat']):.2f}",
                ]
            )
        tables.append(table)

    return tables


def _build_macros_table(payload: Mapping[str, object]) -> PrettyTable:
    table = PrettyTable()
    table.title = (
        "OBTAINED MACROS | "
        f"scenario={payload['scenario']} | mode={payload['mode_label']} | "
        f"dimension={int(payload['dimension'])} | seed={payload['seed']}"
    )
    table.field_names = ["Macro", "Target", "Obtained", "Abs Error", "Rel Error (%)"]
    table.align["Macro"] = "l"
    obtained_macros = payload["obtained_macros"]
    relative_errors = payload["relative_errors"]

    for macro_name in MACRO_NAMES:
        target_value = float(payload["target_macros"][macro_name])
        obtained_value = float(obtained_macros[macro_name])
        table.add_row(
            [
                macro_name,
                f"{target_value:.2f}",
                f"{obtained_value:.2f}",
                f"{abs(obtained_value - target_value):.2f}",
                f"{float(relative_errors[macro_name]) * 100.0:.2f}",
            ]
        )
    return table


def _meal_breakdown_string(values: Mapping[str, object]) -> str:
    return ", ".join(f"{meal}={float(value):.2f}" for meal, value in values.items())


def _build_constraints_table(payload: Mapping[str, object]) -> PrettyTable:
    constraints = payload["obtained_constraints"]
    target_constraints = payload["target_constraints"]
    table = PrettyTable()
    table.title = (
        "CONSTRAINTS | "
        f"scenario={payload['scenario']} | mode={payload['mode_label']} | "
        f"dimension={int(payload['dimension'])} | seed={payload['seed']}"
    )
    table.field_names = ["Constraint", "Target", "Obtained"]
    table.align["Constraint"] = "l"
    table.align["Target"] = "l"
    table.align["Obtained"] = "l"

    table.add_row(
        [
            "Fiber",
            _format_limit(">=", target_constraints.get("min_fiber")),
            f"{float(constraints['fiber']['value']):.2f}",
        ]
    )
    table.add_row(
        [
            "Sugar",
            _format_limit("<=", target_constraints.get("max_sugar")),
            f"{float(constraints['sugar']['value']):.2f}",
        ]
    )
    table.add_row(
        [
            "Salt",
            _format_limit("<=", target_constraints.get("max_salt")),
            f"{float(constraints['salt']['value']):.2f}",
        ]
    )
    table.add_row(
        [
            "Cost",
            _format_limit("<=", target_constraints.get("max_cost")),
            f"{float(constraints['cost']['value']):.2f}",
        ]
    )
    table.add_row(
        [
            "Active foods / meal",
            f"<= {constraints['max_active_foods_per_meal']}",
            (
                _meal_breakdown_string(constraints["active_foods_by_meal"])
                + f" | total={int(constraints['active_foods_total'])}"
            ),
        ]
    )
    table.add_row(
        [
            "Tiny quantities",
            f"< {float(constraints['tiny_threshold_g']):.2f} g",
            (
                _meal_breakdown_string(constraints["tiny_foods_by_meal"])
                + f" | total={int(constraints['tiny_foods_total'])}"
            ),
        ]
    )
    meal_ranges = target_constraints.get("meal_kcal_ranges")
    table.add_row(
        [
            "Meal kcal distribution",
            (
                ", ".join(
                    f"{meal}={limits[0]:.0f}-{limits[1]:.0f}"
                    for meal, limits in meal_ranges.items()
                )
                if meal_ranges
                else "n/a"
            ),
            _meal_breakdown_string(constraints["meal_kcal_totals"]),
        ]
    )
    return table


def _build_fitness_components_table(payload: Mapping[str, object]) -> PrettyTable:
    components = payload["fitness_components"]
    table = PrettyTable()
    table.title = (
        "FITNESS COMPONENTS | "
        f"scenario={payload['scenario']} | mode={payload['mode_label']} | "
        f"dimension={int(payload['dimension'])} | seed={payload['seed']}"
    )
    table.field_names = ["Component", "Value"]
    table.align["Component"] = "l"
    rows = [
        ("Macro Error", _format_fitness(float(components["macro_error"]))),
        ("kcal Error", _format_fitness(float(components["kcal_error"]))),
        ("protein Error", _format_fitness(float(components["protein_error"]))),
        ("carbs Error", _format_fitness(float(components["carbs_error"]))),
        ("fat Error", _format_fitness(float(components["fat_error"]))),
        (
            "Meal Distribution Penalty",
            _format_fitness(float(components["meal_distribution_penalty"])),
        ),
        ("Total Fitness", _format_fitness(float(components["total_fitness"]))),
    ]
    for label, value in rows:
        table.add_row([label, value])
    return table


def _build_execution_summary_table(payload: Mapping[str, object]) -> PrettyTable:
    components = payload["fitness_components"]
    table = PrettyTable()
    table.field_names = [
        "Scenario",
        "Mode",
        "Foods",
        "Dimension",
        "Seed",
        "Best Fitness",
        "Macro Error",
        "Elapsed Time (s)",
        "Active Foods",
        "Iterations",
    ]
    table.add_row(
        [
            payload["scenario"],
            payload["mode_label"],
            int(payload["food_dimension"]),
            int(payload["dimension"]),
            payload["seed"],
            _format_fitness(float(payload["best_fitness"])),
            _format_fitness(float(components["macro_error"])),
            f"{float(payload['elapsed_time']):.6f}",
            int(payload["active_foods"]),
            int(payload["iterations_completed"]),
        ]
    )
    return table


def _build_group_summary_table(
    summary: Mapping[str, object],
    *,
    title: str,
    group_header: str,
) -> PrettyTable:
    table = PrettyTable()
    table.title = title
    table.field_names = [
        group_header,
        "Runs",
        "Scenarios",
        "Modes",
        "Food Dims",
        "Real Dims",
        "Seeds",
        "Mean Fitness",
        "Best Fitness",
        "Mean Macro Error",
        "Mean Time (s)",
        "Total Time (s)",
    ]
    table.align[group_header] = "l"
    table.align["Scenarios"] = "l"
    table.align["Modes"] = "l"
    table.add_row(
        [
            summary["label"],
            int(summary["runs"]),
            _format_list(summary["scenarios"]),
            _format_list(summary["modes"]),
            _format_list(summary["food_dimensions"]),
            _format_list(summary["dimensions"]),
            _format_list(summary["seeds"]) if summary["seeds"] else "n/a",
            _format_fitness(float(summary["mean_fitness"])),
            _format_fitness(float(summary["best_fitness"])),
            _format_fitness(float(summary["mean_macro_error"])),
            f"{float(summary['mean_time_s']):.6f}",
            f"{float(summary['total_time_s']):.6f}",
        ]
    )
    return table


def _build_global_summary_table(payloads: Sequence[Mapping[str, object]]) -> PrettyTable:
    table = PrettyTable()
    table.title = "NUTRITION GLOBAL SUMMARY"
    table.field_names = [
        "Scenario",
        "Mode",
        "Foods",
        "Dimension",
        "Seed",
        "Best Fitness",
        "Macro Error",
        "Elapsed Time (s)",
        "Active Foods",
        "kcal error %",
        "protein error %",
        "carbs error %",
        "fat error %",
    ]

    def _sort_key(payload: Mapping[str, object]) -> tuple[int, str, int, int]:
        return (
            SUPPORTED_MODES.index(str(payload["mode"])),
            str(payload["scenario"]),
            int(payload["food_dimension"]),
            int(payload["seed"]) if payload["seed"] is not None else -1,
        )

    for payload in sorted(payloads, key=_sort_key):
        components = payload["fitness_components"]
        relative_errors = payload["relative_errors"]
        table.add_row(
            [
                payload["scenario"],
                payload["mode_label"],
                int(payload["food_dimension"]),
                int(payload["dimension"]),
                payload["seed"],
                _format_fitness(float(payload["best_fitness"])),
                _format_fitness(float(components["macro_error"])),
                f"{float(payload['elapsed_time']):.6f}",
                int(payload["active_foods"]),
                f"{float(relative_errors['kcal']) * 100.0:.2f}",
                f"{float(relative_errors['protein']) * 100.0:.2f}",
                f"{float(relative_errors['carbs']) * 100.0:.2f}",
                f"{float(relative_errors['fat']) * 100.0:.2f}",
            ]
        )
    return table


def _build_mode_summary_table(mode_summaries: Sequence[Mapping[str, object]]) -> PrettyTable:
    table = PrettyTable()
    table.title = "NUTRITION MODE SUMMARY"
    table.field_names = [
        "Mode",
        "Runs",
        "Scenarios",
        "Food Dims",
        "Real Dims",
        "Mean Fitness",
        "Best Fitness",
        "Mean Macro Error",
        "Mean Time (s)",
        "Total Time (s)",
    ]

    def _sort_key(summary: Mapping[str, object]) -> int:
        mode_label = str(summary["label"])
        mode = next(mode for mode, label in MODE_LABELS.items() if label == mode_label)
        return SUPPORTED_MODES.index(mode)

    for summary in sorted(mode_summaries, key=_sort_key):
        table.add_row(
            [
                summary["label"],
                int(summary["runs"]),
                _format_list(summary["scenarios"]),
                _format_list(summary["food_dimensions"]),
                _format_list(summary["dimensions"]),
                _format_fitness(float(summary["mean_fitness"])),
                _format_fitness(float(summary["best_fitness"])),
                _format_fitness(float(summary["mean_macro_error"])),
                f"{float(summary['mean_time_s']):.6f}",
                f"{float(summary['total_time_s']):.6f}",
            ]
        )
    return table


def _build_scenario_summary_table(
    scenario_summaries: Sequence[Mapping[str, object]],
) -> PrettyTable:
    table = PrettyTable()
    table.title = "NUTRITION SCENARIO SUMMARY"
    table.field_names = [
        "Scenario",
        "Runs",
        "Modes",
        "Food Dims",
        "Real Dims",
        "Mean Fitness",
        "Best Fitness",
        "Mean Macro Error",
        "Mean Time (s)",
        "Total Time (s)",
    ]
    for summary in scenario_summaries:
        table.add_row(
            [
                summary["label"],
                int(summary["runs"]),
                _format_list(summary["modes"]),
                _format_list(summary["food_dimensions"]),
                _format_list(summary["dimensions"]),
                _format_fitness(float(summary["mean_fitness"])),
                _format_fitness(float(summary["best_fitness"])),
                _format_fitness(float(summary["mean_macro_error"])),
                f"{float(summary['mean_time_s']):.6f}",
                f"{float(summary['total_time_s']):.6f}",
            ]
        )
    return table


def display_run_details(
    payload: Mapping[str, object],
    *,
    active_threshold_g: float,
    show_all_foods: bool,
) -> None:
    for table in build_meal_tables(
        payload,
        active_threshold_g=active_threshold_g,
        show_all_foods=show_all_foods,
    ):
        _print_table(table)
    _print_table(_build_macros_table(payload))
    _print_table(_build_constraints_table(payload))
    _print_table(_build_fitness_components_table(payload))


def print_execution_banner(
    logger,
    *,
    scenario_name: str,
    mode_label: str,
    food_dimension: int,
    dimension: int,
    seed: int | None,
    particles: int,
    iterations: int,
) -> None:
    logger.info(
        "[Nutrition] Running scenario=%s | mode=%s | foods=%d | dimension=%d | seed=%s | particles=%d | iterations=%d",
        scenario_name,
        mode_label,
        food_dimension,
        dimension,
        seed,
        particles,
        iterations,
    )


def log_interpretation(logger, payload: Mapping[str, object]) -> None:
    relative_errors = payload["relative_errors"]
    logger.info(
        "[Nutrition] Best solution uses %d active foods. Main error: "
        "kcal=%s%%, protein=%s%%, carbs=%s%%, fat=%s%%",
        int(payload["active_foods"]),
        _format_percent(float(relative_errors["kcal"])),
        _format_percent(float(relative_errors["protein"])),
        _format_percent(float(relative_errors["carbs"])),
        _format_percent(float(relative_errors["fat"])),
    )


def _save_payload(
    output_path: Path,
    payload: Mapping[str, object],
) -> None:
    save_json_document(output_path, build_single_output_document(mode_payload=payload))


def run_case(args, logger) -> dict[str, object]:
    food_overrides, target_overrides = load_case_data(args.foods_path, args.targets_path)
    modes = resolve_modes(args)
    scenarios = resolve_scenarios(args)
    seeds = resolve_seeds(args)
    requested_dimensions = [int(value) for value in args.dimensions]
    show_details_in_all = should_show_meal_details_in_all(
        force_show=args.show_meals_in_all,
        scenarios=scenarios,
        dimensions=requested_dimensions,
        seeds=seeds,
    )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    _print_table(
        _build_configuration_table(
            modes=modes,
            scenarios=scenarios,
            dimensions=requested_dimensions,
            seeds=seeds,
            args=args,
        )
    )

    all_payloads: list[dict[str, object]] = []
    mode_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    scenario_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    scenario_mode_summaries: list[dict[str, object]] = []

    for scenario_name in scenarios:
        available_foods = _resolve_food_catalog(scenario_name, food_overrides)
        normalized_dimensions = normalize_dimensions(requested_dimensions, len(available_foods))
        logger.info(
            "[Nutrition] Scenario=%s | available_foods=%d | food_dimensions=%s",
            scenario_name,
            len(available_foods),
            normalized_dimensions,
        )

        scenario_dimension_definitions = [
            build_scenario_definition(
                scenario_name,
                food_dimension=food_dimension,
                food_overrides=food_overrides,
                target_overrides=target_overrides,
                args=args,
            )
            for food_dimension in normalized_dimensions
        ]
        for scenario_definition in scenario_dimension_definitions:
            _print_table(_build_scenario_table(scenario_definition))
            _print_table(_build_targets_table(scenario_definition))

        for mode in modes:
            mode_label = MODE_LABELS[mode]
            scenario_mode_payloads: list[dict[str, object]] = []
            for scenario_definition in scenario_dimension_definitions:
                for seed in seeds:
                    print_execution_banner(
                        logger,
                        scenario_name=scenario_name,
                        mode_label=mode_label,
                        food_dimension=scenario_definition.food_dimension,
                        dimension=scenario_definition.dimension,
                        seed=seed,
                        particles=args.particles,
                        iterations=args.iterations,
                    )
                    result = run_objective_experiment(
                        objective=scenario_definition.objective,
                        config=build_pso_config(
                            args,
                            dimension=scenario_definition.dimension,
                            seed=seed,
                        ),
                        evaluation_mode=mode,
                        max_workers=args.max_workers,
                        batch_size=args.batch_size,
                        async_min_delay=args.async_min_delay,
                        async_max_delay=args.async_max_delay,
                    )
                    payload = build_mode_payload(
                        result,
                        scenario_definition,
                        seed=seed,
                        mode_label=mode_label,
                        config_payload=build_case_config(args, scenario_definition),
                        async_min_delay=args.async_min_delay,
                        async_max_delay=args.async_max_delay,
                    )
                    output_path = build_output_path(
                        args.output_dir,
                        scenario_name,
                        mode_label,
                        scenario_definition.dimension,
                        seed,
                    )
                    _save_payload(output_path, payload)
                    log_interpretation(logger, payload)
                    logger.info("[Nutrition] Saved run result to %s", output_path)

                    all_payloads.append(payload)
                    mode_groups[mode_label].append(payload)
                    scenario_groups[scenario_name].append(payload)
                    scenario_mode_payloads.append(payload)

                    _print_table(_build_execution_summary_table(payload))
                    if len(modes) == 1 or show_details_in_all:
                        display_run_details(
                            payload,
                            active_threshold_g=float(args.active_threshold_g),
                            show_all_foods=args.show_all_foods,
                        )

            summary = summarize_payloads(
                scenario_mode_payloads,
                group_label=f"{scenario_name} | {mode_label}",
                group_type="scenario_mode",
            )
            scenario_mode_summaries.append(summary)
            _print_table(
                _build_group_summary_table(
                    summary,
                    title=f"SCENARIO MODE SUMMARY | {scenario_name} | {mode_label}",
                    group_header="Scenario | Mode",
                )
            )

        scenario_summary = summarize_payloads(
            scenario_groups[scenario_name],
            group_label=scenario_name,
            group_type="scenario",
        )
        _print_table(
            _build_group_summary_table(
                scenario_summary,
                title=f"SCENARIO SUMMARY | {scenario_name}",
                group_header="Scenario",
            )
        )

    mode_summaries = [
        summarize_payloads(payloads, group_label=mode_label, group_type="mode")
        for mode_label, payloads in mode_groups.items()
    ]
    scenario_summaries = [
        summarize_payloads(payloads, group_label=scenario_name, group_type="scenario")
        for scenario_name, payloads in scenario_groups.items()
    ]

    global_summary_document = build_global_output_document(
        output_dir=args.output_dir,
        modes=modes,
        scenarios=scenarios,
        food_dimensions=requested_dimensions,
        seeds=seeds,
        results=all_payloads,
        mode_summaries=mode_summaries,
        scenario_summaries=scenario_summaries,
        scenario_mode_summaries=scenario_mode_summaries,
    )
    global_summary_path = build_global_summary_path(args.output_dir)
    save_json_document(global_summary_path, global_summary_document)
    logger.info("[Nutrition] Saved global summary to %s", global_summary_path)

    if args.output_path is not None:
        save_json_document(args.output_path, global_summary_document)
        logger.info("[Nutrition] Saved compatibility output to %s", args.output_path)

    return {
        "payloads": all_payloads,
        "mode_summaries": mode_summaries,
        "scenario_summaries": scenario_summaries,
        "scenario_mode_summaries": scenario_mode_summaries,
        "modes": modes,
        "scenarios": scenarios,
        "seeds": seeds,
        "dimensions": requested_dimensions,
    }


def display_global_results(result_bundle: Mapping[str, object]) -> None:
    payloads = result_bundle["payloads"]
    mode_summaries = result_bundle["mode_summaries"]
    scenario_summaries = result_bundle["scenario_summaries"]
    _print_table(_build_global_summary_table(payloads))
    _print_table(_build_mode_summary_table(mode_summaries))
    _print_table(_build_scenario_summary_table(scenario_summaries))


def main(argv: Sequence[str] | None = None) -> None:
    logger = setup_logger("pso_nutrition_case")
    args = parse_nutrition_case_args(argv)
    result_bundle = run_case(args, logger)
    display_global_results(result_bundle)


if __name__ == "__main__":
    # Default run profile used when the script is launched without CLI arguments.
    # This makes the VS Code "Run Python File" button behave like the curated
    # multi-scenario benchmark profile below.
    default_argv = [
        "--modes", "all",
        "--scenarios", "simple_meal", "complex_meal", "full_day_constrained",
        "--dimensions", "10",
        "--seeds", "0", "1", "2",
        "--particles", "800",
        "--iterations", "400",
        "--inertia", "0.7",
        "--c1", "1.5",
        "--c2", "1.5",
        "--max-active-foods", "5",
        "--active-threshold", "5.0",
        "--tiny-threshold", "5.0",
        "--active-penalty-weight", "0.10",
        "--tiny-penalty-weight", "0.05",
        "--output-dir", "results/nutrition",
        "--async-min-delay", "0.0",
        "--async-max-delay", "0.0",
        "--show-meals-in-all",
    ]
    effective_argv = sys.argv[1:]
    if not effective_argv:
        effective_argv = default_argv
    main(effective_argv or None)
