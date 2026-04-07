from __future__ import annotations

import sys
import types

import numpy as np

from pso_lab.core.config import PSOConfig
from pso_lab.experiments.pyswarm_runner import run_pyswarm_experiment


def _build_config(*, track_swarm_history: bool = False) -> PSOConfig:
    return PSOConfig(
        num_particles=2,
        dimensions=2,
        max_iterations=2,
        inertia_weight=0.7,
        cognitive_coefficient=1.5,
        social_coefficient=1.5,
        seed=7,
        tolerance=0.0,
        stagnation_patience=None,
        track_history=True,
        track_swarm_history=track_swarm_history,
    )


def test_run_pyswarm_experiment_adapts_pyswarm_output(monkeypatch) -> None:
    fake_module = types.ModuleType("pyswarm")

    def fake_pso(
        func,
        lb,
        ub,
        *,
        swarmsize,
        omega,
        phip,
        phig,
        maxiter,
        minstep,
        minfunc,
        debug,
        processes,
    ):
        assert lb == [-5.12, -5.12]
        assert ub == [5.12, 5.12]
        assert swarmsize == 2
        assert omega == 0.7
        assert phip == 1.5
        assert phig == 1.5
        assert maxiter == 2
        assert minstep == 0.0
        assert minfunc == 0.0
        assert debug is False
        assert processes == 1

        positions = [
            np.array([1.0, 0.0]),
            np.array([2.0, 1.0]),
            np.array([0.5, 0.0]),
            np.array([1.0, 1.0]),
            np.array([0.0, 0.0]),
            np.array([1.0, 0.0]),
        ]
        for position in positions:
            func(position)

        print("Stopping search: maximum iterations reached --> 2")
        return np.array([0.0, 0.0]), 0.0

    fake_module.pso = fake_pso
    monkeypatch.setitem(sys.modules, "pyswarm", fake_module)

    result = run_pyswarm_experiment("sphere", _build_config())

    assert result.objective_name == "sphere"
    assert result.evaluation_mode == "pyswarm"
    assert result.best_position == [0.0, 0.0]
    assert result.best_value == 0.0
    assert result.iterations_completed == 2
    assert result.best_value_history == [0.25, 0.0]
    assert result.swarm_position_history is None
    assert result.timing_stats["solver"] == "pyswarm"
    assert result.timing_stats["solver_messages"] == [
        "Stopping search: maximum iterations reached --> 2"
    ]


def test_run_pyswarm_experiment_can_rebuild_swarm_history(monkeypatch) -> None:
    fake_module = types.ModuleType("pyswarm")

    def fake_pso(
        func,
        lb,
        ub,
        *,
        swarmsize,
        omega,
        phip,
        phig,
        maxiter,
        minstep,
        minfunc,
        debug,
        processes,
    ):
        positions = [
            np.array([1.0, 1.0]),
            np.array([2.0, 2.0]),
            np.array([0.5, 0.5]),
            np.array([1.0, 0.5]),
            np.array([0.0, 0.0]),
            np.array([0.5, 0.0]),
        ]
        for position in positions:
            func(position)

        return np.array([0.0, 0.0]), 0.0

    fake_module.pso = fake_pso
    monkeypatch.setitem(sys.modules, "pyswarm", fake_module)

    result = run_pyswarm_experiment(
        "sphere",
        _build_config(track_swarm_history=True),
    )

    assert result.swarm_position_history is not None
    assert len(result.swarm_position_history) == 3
    assert result.swarm_position_history[0] == [[1.0, 1.0], [2.0, 2.0]]
    assert result.swarm_position_history[-1] == [[0.0, 0.0], [0.5, 0.0]]


def test_run_pyswarm_experiment_supports_older_signature_without_processes(monkeypatch) -> None:
    fake_module = types.ModuleType("pyswarm")

    def fake_pso(
        func,
        lb,
        ub,
        *,
        swarmsize,
        omega,
        phip,
        phig,
        maxiter,
        minstep,
        minfunc,
        debug,
    ):
        assert lb == [-5.12, -5.12]
        assert ub == [5.12, 5.12]
        assert swarmsize == 2
        assert omega == 0.7
        assert phip == 1.5
        assert phig == 1.5
        assert maxiter == 2
        assert minstep == 0.0
        assert minfunc == 0.0
        assert debug is False

        for position in (
            np.array([1.0, 1.0]),
            np.array([0.5, 0.5]),
            np.array([0.0, 0.0]),
            np.array([0.25, 0.25]),
            np.array([0.0, 0.0]),
            np.array([0.1, 0.1]),
        ):
            func(position)

        return np.array([0.0, 0.0]), 0.0

    fake_module.pso = fake_pso
    monkeypatch.setitem(sys.modules, "pyswarm", fake_module)

    result = run_pyswarm_experiment("sphere", _build_config())

    assert result.evaluation_mode == "pyswarm"
    assert result.best_position == [0.0, 0.0]
    assert result.iterations_completed == 2
