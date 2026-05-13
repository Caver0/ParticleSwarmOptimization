from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .base import ObjectiveFunction


class NutritionObjective(ObjectiveFunction):
    """Nutrition optimization objective for single meals and multi-meal plans."""

    __slots__ = (
        "foods",
        "targets",
        "meal_names",
        "macro_names",
        "constraint_names",
        "nutrition_matrix",
        "constraint_matrix",
        "target_vector",
        "macro_weights",
        "macro_weight_vector",
        "penalty_weights",
        "active_threshold_g",
        "tiny_threshold_g",
        "max_active_foods",
        "min_fiber",
        "max_sugar",
        "max_salt",
        "max_cost",
        "meal_kcal_ranges",
        "n_foods",
        "n_meals",
    )

    def __init__(
        self,
        foods: Sequence[Mapping[str, object]],
        targets: Mapping[str, object],
        *,
        meal_names: Sequence[str] | None = None,
        macro_weights: Mapping[str, float] | None = None,
        penalty_weights: Mapping[str, float] | None = None,
        active_threshold_g: float = 5.0,
        tiny_threshold_g: float | None = None,
        max_active_foods: int | None = 5,
        min_fiber: float | None = None,
        max_sugar: float | None = None,
        max_salt: float | None = None,
        max_cost: float | None = None,
        meal_kcal_ranges: Mapping[str, Sequence[float]] | None = None,
        objective_name: str = "nutrition_meal_optimization",
    ) -> None:
        self.macro_names = ("kcal", "protein", "carbs", "fat")
        self.constraint_names = ("fiber", "sugar", "salt", "cost")
        self.foods = self._normalize_foods(foods)
        self.targets = self._normalize_targets(targets)
        self.meal_names = self._normalize_meal_names(meal_names)
        self.n_foods = len(self.foods)
        self.n_meals = len(self.meal_names)
        self.macro_weights, self.macro_weight_vector = self._normalize_macro_weights(
            macro_weights
        )
        self.penalty_weights = self._normalize_penalty_weights(penalty_weights)
        self.active_threshold_g = float(active_threshold_g)
        self.tiny_threshold_g = (
            float(active_threshold_g)
            if tiny_threshold_g is None
            else float(tiny_threshold_g)
        )
        self.max_active_foods = None if max_active_foods is None else int(max_active_foods)
        self.min_fiber = None if min_fiber is None else float(min_fiber)
        self.max_sugar = None if max_sugar is None else float(max_sugar)
        self.max_salt = None if max_salt is None else float(max_salt)
        self.max_cost = None if max_cost is None else float(max_cost)
        self.meal_kcal_ranges = self._normalize_meal_kcal_ranges(meal_kcal_ranges)

        if self.active_threshold_g < 0.0:
            raise ValueError("active_threshold_g must be >= 0")
        if self.tiny_threshold_g < 0.0:
            raise ValueError("tiny_threshold_g must be >= 0")
        if self.max_active_foods is not None and self.max_active_foods < 0:
            raise ValueError("max_active_foods must be >= 0")
        for name, value in (
            ("min_fiber", self.min_fiber),
            ("max_sugar", self.max_sugar),
            ("max_salt", self.max_salt),
            ("max_cost", self.max_cost),
        ):
            if value is not None and value <= 0.0:
                raise ValueError(f"{name} must be > 0 when provided")

        self.nutrition_matrix = np.asarray(
            [
                [float(food[macro_name]) for macro_name in self.macro_names]
                for food in self.foods
            ],
            dtype=float,
        )
        self.constraint_matrix = np.asarray(
            [
                [float(food[name]) for name in self.constraint_names]
                for food in self.foods
            ],
            dtype=float,
        )
        self.target_vector = np.asarray(
            [float(self.targets[macro_name]) for macro_name in self.macro_names],
            dtype=float,
        )

        super().__init__(
            name=objective_name,
            dimensions=self.n_foods * self.n_meals,
            bounds=self.get_bounds(),
        )

    def _normalize_foods(
        self,
        foods: Sequence[Mapping[str, object]],
    ) -> list[dict[str, float | str]]:
        if not foods:
            raise ValueError("foods must contain at least one food item")

        required_keys = ("name", "kcal", "protein", "carbs", "fat", "min_g", "max_g")
        optional_numeric_keys = ("fiber", "sugar", "salt", "cost")
        normalized_foods: list[dict[str, float | str]] = []

        for index, raw_food in enumerate(foods):
            missing_keys = [key for key in required_keys if key not in raw_food]
            if missing_keys:
                raise ValueError(
                    f"Food at index {index} is missing required keys: {missing_keys}"
                )

            name = str(raw_food["name"]).strip()
            if not name:
                raise ValueError(f"Food at index {index} must have a non-empty name")

            normalized_food: dict[str, float | str] = {
                "name": name,
                "kcal": float(raw_food["kcal"]),
                "protein": float(raw_food["protein"]),
                "carbs": float(raw_food["carbs"]),
                "fat": float(raw_food["fat"]),
                "min_g": float(raw_food["min_g"]),
                "max_g": float(raw_food["max_g"]),
            }
            for key in optional_numeric_keys:
                normalized_food[key] = float(raw_food.get(key, 0.0))

            if float(normalized_food["min_g"]) > float(normalized_food["max_g"]):
                raise ValueError(
                    f"Food '{name}' has min_g > max_g: "
                    f"{normalized_food['min_g']} > {normalized_food['max_g']}"
                )

            normalized_foods.append(normalized_food)

        return normalized_foods

    def _normalize_targets(self, targets: Mapping[str, object]) -> dict[str, float]:
        missing_keys = [key for key in self.macro_names if key not in targets]
        if missing_keys:
            raise ValueError(f"targets is missing required keys: {missing_keys}")

        normalized_targets = {
            macro_name: float(targets[macro_name]) for macro_name in self.macro_names
        }
        invalid_targets = [
            macro_name
            for macro_name, value in normalized_targets.items()
            if value <= 0.0
        ]
        if invalid_targets:
            raise ValueError(
                "All target values must be > 0. Invalid targets: "
                f"{invalid_targets}"
            )
        return normalized_targets

    def _normalize_meal_names(
        self,
        meal_names: Sequence[str] | None,
    ) -> tuple[str, ...]:
        if meal_names is None:
            return ("meal",)

        normalized_meal_names = tuple(str(name).strip() for name in meal_names)
        if not normalized_meal_names:
            raise ValueError("meal_names must contain at least one meal")
        if any(not name for name in normalized_meal_names):
            raise ValueError("meal_names cannot contain empty names")
        return normalized_meal_names

    def _normalize_macro_weights(
        self,
        macro_weights: Mapping[str, float] | None,
    ) -> tuple[dict[str, float], np.ndarray]:
        default_weights = {
            "kcal": 1.5,
            "protein": 1.0,
            "carbs": 1.0,
            "fat": 1.0,
        }
        if macro_weights is None:
            normalized_weights = default_weights
        else:
            unknown_keys = sorted(set(macro_weights) - set(self.macro_names))
            if unknown_keys:
                raise ValueError(
                    f"macro_weights contains unknown macro keys: {unknown_keys}"
                )
            normalized_weights = {
                macro_name: float(
                    macro_weights.get(macro_name, default_weights[macro_name])
                )
                for macro_name in self.macro_names
            }

        weight_vector = np.asarray(
            [normalized_weights[macro_name] for macro_name in self.macro_names],
            dtype=float,
        )
        return normalized_weights, weight_vector

    def _normalize_penalty_weights(
        self,
        penalty_weights: Mapping[str, float] | None,
    ) -> dict[str, float]:
        default_weights = {
            "fiber_penalty": 1.0,
            "sugar_penalty": 1.0,
            "salt_penalty": 1.0,
            "cost_penalty": 1.0,
            "active_penalty": 0.10,
            "tiny_penalty": 0.05,
            "meal_distribution_penalty": 1.0,
        }
        if penalty_weights is None:
            return default_weights

        unknown_keys = sorted(set(penalty_weights) - set(default_weights))
        if unknown_keys:
            raise ValueError(
                f"penalty_weights contains unknown penalty keys: {unknown_keys}"
            )
        return {
            key: float(penalty_weights.get(key, default_weights[key]))
            for key in default_weights
        }

    def _normalize_meal_kcal_ranges(
        self,
        meal_kcal_ranges: Mapping[str, Sequence[float]] | None,
    ) -> dict[str, tuple[float, float]]:
        if meal_kcal_ranges is None:
            return {}

        normalized_ranges: dict[str, tuple[float, float]] = {}
        for meal_name in self.meal_names:
            if meal_name not in meal_kcal_ranges:
                continue
            raw_range = meal_kcal_ranges[meal_name]
            if len(raw_range) != 2:
                raise ValueError(
                    f"Meal kcal range for '{meal_name}' must contain exactly two values"
                )
            lower, upper = float(raw_range[0]), float(raw_range[1])
            if lower < 0.0 or upper < 0.0 or lower > upper:
                raise ValueError(
                    f"Invalid meal kcal range for '{meal_name}': {(lower, upper)}"
                )
            normalized_ranges[meal_name] = (lower, upper)
        return normalized_ranges

    def _validate_quantities(self, quantities: Sequence[float] | np.ndarray) -> np.ndarray:
        quantities_array = np.asarray(quantities, dtype=float)
        expected_shape = (self.dimensions,)
        if quantities_array.shape != expected_shape:
            raise ValueError(
                "Expected quantities with shape "
                f"{expected_shape}, got {quantities_array.shape}"
            )
        return quantities_array

    def _validate_positions(
        self,
        positions: Sequence[Sequence[float]] | np.ndarray,
    ) -> np.ndarray:
        positions_array = np.asarray(positions, dtype=float)
        if positions_array.ndim != 2:
            raise ValueError(
                "Expected positions with shape (n_particles, dimensions), got "
                f"{positions_array.shape}"
            )
        if positions_array.shape[1] != self.dimensions:
            raise ValueError(
                "Expected positions with "
                f"{self.dimensions} columns, got {positions_array.shape[1]}"
            )
        return positions_array

    def _reshape_quantities(self, quantities: Sequence[float] | np.ndarray) -> np.ndarray:
        quantities_array = self._validate_quantities(quantities)
        return quantities_array.reshape(self.n_meals, self.n_foods)

    def _reshape_positions(
        self,
        positions: Sequence[Sequence[float]] | np.ndarray,
    ) -> np.ndarray:
        positions_array = self._validate_positions(positions)
        return positions_array.reshape(positions_array.shape[0], self.n_meals, self.n_foods)

    def _compute_macro_totals_batch_from_reshaped(
        self,
        positions_reshaped: np.ndarray,
    ) -> np.ndarray:
        meal_totals = positions_reshaped @ self.nutrition_matrix / 100.0
        return np.sum(meal_totals, axis=1, dtype=float)

    def _compute_constraint_totals_batch_from_reshaped(
        self,
        positions_reshaped: np.ndarray,
    ) -> np.ndarray:
        meal_totals = positions_reshaped @ self.constraint_matrix / 100.0
        return np.sum(meal_totals, axis=1, dtype=float)

    def compute_totals_array(self, quantities: Sequence[float] | np.ndarray) -> np.ndarray:
        quantities_matrix = self._reshape_quantities(quantities)
        meal_totals = quantities_matrix @ self.nutrition_matrix / 100.0
        return np.sum(meal_totals, axis=0, dtype=float)

    def compute_totals(self, quantities: Sequence[float] | np.ndarray) -> dict[str, float]:
        totals_array = self.compute_totals_array(quantities)
        return {
            macro_name: float(total_value)
            for macro_name, total_value in zip(self.macro_names, totals_array, strict=True)
        }

    def compute_totals_batch(
        self,
        positions: Sequence[Sequence[float]] | np.ndarray,
    ) -> np.ndarray:
        positions_reshaped = self._reshape_positions(positions)
        return self._compute_macro_totals_batch_from_reshaped(positions_reshaped)

    def compute_constraint_totals_array(
        self,
        quantities: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        quantities_matrix = self._reshape_quantities(quantities)
        meal_totals = quantities_matrix @ self.constraint_matrix / 100.0
        return np.sum(meal_totals, axis=0, dtype=float)

    def compute_constraint_totals(
        self,
        quantities: Sequence[float] | np.ndarray,
    ) -> dict[str, float]:
        totals_array = self.compute_constraint_totals_array(quantities)
        return {
            name: float(total_value)
            for name, total_value in zip(self.constraint_names, totals_array, strict=True)
        }

    def compute_relative_errors(
        self,
        quantities: Sequence[float] | np.ndarray,
    ) -> dict[str, float]:
        totals_array = self.compute_totals_array(quantities)
        relative_errors = np.abs(totals_array - self.target_vector) / self.target_vector
        return {
            macro_name: float(error_value)
            for macro_name, error_value in zip(
                self.macro_names,
                relative_errors,
                strict=True,
            )
        }

    def compute_meal_macro_totals(
        self,
        quantities: Sequence[float] | np.ndarray,
    ) -> dict[str, dict[str, float]]:
        quantities_matrix = self._reshape_quantities(quantities)
        meal_totals = quantities_matrix @ self.nutrition_matrix / 100.0
        return {
            meal_name: {
                macro_name: float(meal_totals[meal_idx, macro_idx])
                for macro_idx, macro_name in enumerate(self.macro_names)
            }
            for meal_idx, meal_name in enumerate(self.meal_names)
        }

    def get_food_names(self) -> list[str]:
        return [str(food["name"]) for food in self.foods]

    def get_bounds(self) -> list[tuple[float, float]]:
        per_food_bounds = [
            (float(food["min_g"]), float(food["max_g"]))
            for food in self.foods
        ]
        return per_food_bounds * self.n_meals

    def _component_arrays_from_reshaped(
        self,
        positions_reshaped: np.ndarray,
    ) -> dict[str, np.ndarray]:
        macro_totals = self._compute_macro_totals_batch_from_reshaped(positions_reshaped)
        relative_errors = np.abs(macro_totals - self.target_vector) / self.target_vector
        macro_errors = np.sum(relative_errors * self.macro_weight_vector, axis=1, dtype=float)

        constraint_totals = self._compute_constraint_totals_batch_from_reshaped(
            positions_reshaped
        )
        fiber_totals = constraint_totals[:, 0]
        sugar_totals = constraint_totals[:, 1]
        salt_totals = constraint_totals[:, 2]
        cost_totals = constraint_totals[:, 3]

        zeros = np.zeros(positions_reshaped.shape[0], dtype=float)
        fiber_penalties = zeros.copy()
        sugar_penalties = zeros.copy()
        salt_penalties = zeros.copy()
        cost_penalties = zeros.copy()
        if self.min_fiber is not None:
            fiber_penalties = self.penalty_weights["fiber_penalty"] * np.maximum(
                self.min_fiber - fiber_totals,
                0.0,
            ) / self.min_fiber
        if self.max_sugar is not None:
            sugar_penalties = self.penalty_weights["sugar_penalty"] * np.maximum(
                sugar_totals - self.max_sugar,
                0.0,
            ) / self.max_sugar
        if self.max_salt is not None:
            salt_penalties = self.penalty_weights["salt_penalty"] * np.maximum(
                salt_totals - self.max_salt,
                0.0,
            ) / self.max_salt
        if self.max_cost is not None:
            cost_penalties = self.penalty_weights["cost_penalty"] * np.maximum(
                cost_totals - self.max_cost,
                0.0,
            ) / self.max_cost

        active_counts_by_meal = np.sum(
            positions_reshaped > self.active_threshold_g,
            axis=2,
            dtype=int,
        )
        active_foods_total = np.sum(active_counts_by_meal, axis=1, dtype=int)
        active_penalties = zeros.copy()
        if self.max_active_foods is not None:
            active_excess = np.maximum(active_counts_by_meal - self.max_active_foods, 0)
            active_penalties = (
                self.penalty_weights["active_penalty"]
                * np.sum(active_excess, axis=1, dtype=float)
            )

        tiny_counts_total = np.sum(
            (positions_reshaped > 0.0) & (positions_reshaped < self.tiny_threshold_g),
            axis=(1, 2),
            dtype=int,
        )
        tiny_penalties = self.penalty_weights["tiny_penalty"] * tiny_counts_total.astype(
            float
        )

        meal_distribution_penalties = zeros.copy()
        if self.meal_kcal_ranges:
            meal_kcal_totals = positions_reshaped @ self.nutrition_matrix[:, :1] / 100.0
            meal_kcal_totals = meal_kcal_totals[:, :, 0]
            lower_bounds = np.asarray(
                [
                    self.meal_kcal_ranges.get(meal_name, (0.0, 0.0))[0]
                    for meal_name in self.meal_names
                ],
                dtype=float,
            )
            upper_bounds = np.asarray(
                [
                    self.meal_kcal_ranges.get(meal_name, (np.inf, np.inf))[1]
                    for meal_name in self.meal_names
                ],
                dtype=float,
            )
            below_penalty = np.where(
                lower_bounds > 0.0,
                np.maximum(lower_bounds - meal_kcal_totals, 0.0)
                / np.where(lower_bounds > 0.0, lower_bounds, 1.0),
                0.0,
            )
            finite_upper = np.isfinite(upper_bounds)
            above_penalty = np.where(
                finite_upper,
                np.maximum(meal_kcal_totals - upper_bounds, 0.0)
                / np.where(upper_bounds > 0.0, upper_bounds, 1.0),
                0.0,
            )
            meal_distribution_penalties = (
                self.penalty_weights["meal_distribution_penalty"]
                * np.sum(below_penalty + above_penalty, axis=1, dtype=float)
            )

        total_fitness = (
            macro_errors
            + fiber_penalties
            + sugar_penalties
            + salt_penalties
            + cost_penalties
            + active_penalties
            + tiny_penalties
            + meal_distribution_penalties
        )

        return {
            "kcal_error": relative_errors[:, 0],
            "protein_error": relative_errors[:, 1],
            "carbs_error": relative_errors[:, 2],
            "fat_error": relative_errors[:, 3],
            "macro_error": macro_errors,
            "fiber_penalty": fiber_penalties,
            "sugar_penalty": sugar_penalties,
            "salt_penalty": salt_penalties,
            "cost_penalty": cost_penalties,
            "active_penalty": active_penalties,
            "tiny_penalty": tiny_penalties,
            "meal_distribution_penalty": meal_distribution_penalties,
            "total_fitness": total_fitness,
            "fitness": total_fitness,
            "fiber_total": fiber_totals,
            "sugar_total": sugar_totals,
            "salt_total": salt_totals,
            "cost_total": cost_totals,
            "active_foods_total": active_foods_total.astype(float),
            "tiny_foods_total": tiny_counts_total.astype(float),
        }

    def evaluate_components(
        self,
        quantities: Sequence[float] | np.ndarray,
    ) -> dict[str, float]:
        quantities_matrix = self._reshape_quantities(quantities)
        component_arrays = self._component_arrays_from_reshaped(
            quantities_matrix.reshape(1, self.n_meals, self.n_foods)
        )
        return {
            key: float(value_array[0])
            for key, value_array in component_arrays.items()
            if key not in {"fiber_total", "sugar_total", "salt_total", "cost_total", "active_foods_total", "tiny_foods_total"}
        } | {
            "fiber_total": float(component_arrays["fiber_total"][0]),
            "sugar_total": float(component_arrays["sugar_total"][0]),
            "salt_total": float(component_arrays["salt_total"][0]),
            "cost_total": float(component_arrays["cost_total"][0]),
            "active_foods_total": int(round(component_arrays["active_foods_total"][0])),
            "tiny_foods_total": int(round(component_arrays["tiny_foods_total"][0])),
        }

    def compute_obtained_constraints(
        self,
        quantities: Sequence[float] | np.ndarray,
    ) -> dict[str, object]:
        quantities_matrix = self._reshape_quantities(quantities)
        constraint_totals = self.compute_constraint_totals(quantities)
        macro_totals_by_meal = self.compute_meal_macro_totals(quantities)
        active_foods_by_meal = {
            meal_name: int(np.sum(quantities_matrix[meal_idx] > self.active_threshold_g))
            for meal_idx, meal_name in enumerate(self.meal_names)
        }
        tiny_foods_by_meal = {
            meal_name: int(
                np.sum(
                    (quantities_matrix[meal_idx] > 0.0)
                    & (quantities_matrix[meal_idx] < self.tiny_threshold_g)
                )
            )
            for meal_idx, meal_name in enumerate(self.meal_names)
        }
        components = self.evaluate_components(quantities)

        return {
            "fiber": {
                "value": float(constraint_totals["fiber"]),
                "minimum": self.min_fiber,
                "penalty": float(components["fiber_penalty"]),
            },
            "sugar": {
                "value": float(constraint_totals["sugar"]),
                "maximum": self.max_sugar,
                "penalty": float(components["sugar_penalty"]),
            },
            "salt": {
                "value": float(constraint_totals["salt"]),
                "maximum": self.max_salt,
                "penalty": float(components["salt_penalty"]),
            },
            "cost": {
                "value": float(constraint_totals["cost"]),
                "maximum": self.max_cost,
                "penalty": float(components["cost_penalty"]),
            },
            "active_foods_total": int(components["active_foods_total"]),
            "active_foods_by_meal": active_foods_by_meal,
            "max_active_foods_per_meal": self.max_active_foods,
            "active_penalty": float(components["active_penalty"]),
            "tiny_foods_total": int(components["tiny_foods_total"]),
            "tiny_foods_by_meal": tiny_foods_by_meal,
            "tiny_threshold_g": float(self.tiny_threshold_g),
            "tiny_penalty": float(components["tiny_penalty"]),
            "meal_kcal_totals": {
                meal_name: float(macro_totals_by_meal[meal_name]["kcal"])
                for meal_name in self.meal_names
            },
            "meal_kcal_ranges": {
                meal_name: {
                    "min": self.meal_kcal_ranges[meal_name][0],
                    "max": self.meal_kcal_ranges[meal_name][1],
                }
                for meal_name in self.meal_kcal_ranges
            },
            "meal_distribution_penalty": float(components["meal_distribution_penalty"]),
        }

    def decode_solution(
        self,
        quantities: Sequence[float] | np.ndarray,
    ) -> dict[str, object]:
        quantities_matrix = self._reshape_quantities(quantities)
        meal_macro_totals = quantities_matrix @ self.nutrition_matrix / 100.0
        meal_constraint_totals = quantities_matrix @ self.constraint_matrix / 100.0

        meals: list[dict[str, object]] = []
        for meal_idx, meal_name in enumerate(self.meal_names):
            food_entries: list[dict[str, object]] = []
            for food, quantity_g in zip(self.foods, quantities_matrix[meal_idx], strict=True):
                quantity_value = float(quantity_g)
                food_entries.append(
                    {
                        "meal": meal_name,
                        "name": str(food["name"]),
                        "quantity_g": quantity_value,
                        "kcal": quantity_value * float(food["kcal"]) / 100.0,
                        "protein": quantity_value * float(food["protein"]) / 100.0,
                        "carbs": quantity_value * float(food["carbs"]) / 100.0,
                        "fat": quantity_value * float(food["fat"]) / 100.0,
                        "fiber": quantity_value * float(food["fiber"]) / 100.0,
                        "sugar": quantity_value * float(food["sugar"]) / 100.0,
                        "salt": quantity_value * float(food["salt"]) / 100.0,
                        "cost": quantity_value * float(food["cost"]) / 100.0,
                        "is_active": quantity_value > self.active_threshold_g,
                    }
                )

            meals.append(
                {
                    "meal": meal_name,
                    "foods": food_entries,
                    "totals": {
                        "kcal": float(meal_macro_totals[meal_idx, 0]),
                        "protein": float(meal_macro_totals[meal_idx, 1]),
                        "carbs": float(meal_macro_totals[meal_idx, 2]),
                        "fat": float(meal_macro_totals[meal_idx, 3]),
                        "fiber": float(meal_constraint_totals[meal_idx, 0]),
                        "sugar": float(meal_constraint_totals[meal_idx, 1]),
                        "salt": float(meal_constraint_totals[meal_idx, 2]),
                        "cost": float(meal_constraint_totals[meal_idx, 3]),
                    },
                    "active_foods": int(np.sum(quantities_matrix[meal_idx] > self.active_threshold_g)),
                    "tiny_foods": int(
                        np.sum(
                            (quantities_matrix[meal_idx] > 0.0)
                            & (quantities_matrix[meal_idx] < self.tiny_threshold_g)
                        )
                    ),
                }
            )

        return {
            "meal_names": list(self.meal_names),
            "meals": meals,
            "day_totals": self.compute_totals(quantities) | self.compute_constraint_totals(quantities),
            "active_foods_total": int(np.sum(quantities_matrix > self.active_threshold_g)),
            "tiny_foods_total": int(
                np.sum((quantities_matrix > 0.0) & (quantities_matrix < self.tiny_threshold_g))
            ),
        }

    def __call__(self, quantities: np.ndarray) -> float:
        return float(self.evaluate_components(quantities)["total_fitness"])

    def evaluate_batch(self, positions: np.ndarray) -> np.ndarray:
        positions_reshaped = self._reshape_positions(positions)
        component_arrays = self._component_arrays_from_reshaped(positions_reshaped)
        return np.asarray(component_arrays["total_fitness"], dtype=float)
