"""Regression tests for the Orbit Wars fixed-buffer engine."""

from __future__ import annotations

import math
import os
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace

import pytest

import main
import package_submission


def load_engine():
    assert main._ensure_native_engine()
    return main.orbit_engine


def obs(planets, fleets=None, **extra):
    data = {
        "player": extra.get("player", 0),
        "step": extra.get("step", 0),
        "angular_velocity": extra.get("angular_velocity", 0.0),
        "remainingOverageTime": extra.get("remainingOverageTime", 0.0),
        "planets": planets,
        "fleets": fleets or [],
        "initial_planets": extra.get("initial_planets", planets),
        "comets": extra.get("comets", []),
        "comet_planet_ids": extra.get("comet_planet_ids", []),
    }
    return data


def make_engine(observation):
    orbit_engine = load_engine()
    engine = orbit_engine.Engine(int(observation.get("player", 0)))
    engine.update_observation(observation)
    return engine


def planet_by_id(state, planet_id):
    for planet in state["planets"]:
        if planet["id"] == planet_id:
            return planet
    raise AssertionError(f"missing planet {planet_id}")


def total_launched_from(actions, planet_id):
    return sum(int(row[2]) for row in actions if int(row[0]) == planet_id)


def test_speed_formula_matches_rules():
    orbit_engine = load_engine()
    assert orbit_engine.speed_for_ships(1) == pytest.approx(1.0)
    assert orbit_engine.speed_for_ships(1000) == pytest.approx(6.0)
    assert 4.0 < orbit_engine.speed_for_ships(500) < 6.0


def test_observation_parses_orbiting_static_and_comets():
    observation = obs(
        [
            [0, 0, 60.0, 50.0, 1.0, 20, 2],
            [1, -1, 99.0, 50.0, 2.0, 10, 3],
            [2, -1, 10.0, 10.0, 1.0, 4, 1],
        ],
        angular_velocity=0.05,
        comet_planet_ids=[2],
    )
    engine = make_engine(observation)
    state = engine.debug_state()
    assert planet_by_id(state, 0)["is_orbiting"]
    assert not planet_by_id(state, 1)["is_orbiting"]
    assert planet_by_id(state, 2)["is_comet"]


def test_launch_spend_legality_and_production_ordering():
    engine = make_engine(
        obs(
            [
                [0, 0, 10.0, 10.0, 2.0, 10, 2],
                [1, -1, 30.0, 10.0, 2.0, 0, 1],
            ]
        )
    )
    engine.step([[0, 0.0, 10], [0, 0.0, 10]])
    state = engine.debug_state()
    assert planet_by_id(state, 0)["ships"] == 2
    assert len(state["fleets"]) == 1
    assert state["fleets"][0]["ships"] == 10


def test_sun_occlusion_removes_crossing_fleet():
    engine = make_engine(
        obs(
            [[0, 0, 20.0, 20.0, 2.0, 10, 1]],
            fleets=[[9, 0, 35.0, 50.0, 0.0, 0, 1000]],
        )
    )
    engine.step([])
    assert engine.debug_state()["fleets"] == []


def test_planet_collision_resolves_capture():
    engine = make_engine(
        obs(
            [[0, -1, 21.0, 50.0, 1.0, 0, 1]],
            fleets=[[7, 0, 20.0, 50.0, 0.0, -1, 1]],
        )
    )
    engine.step([])
    planet = planet_by_id(engine.debug_state(), 0)
    assert planet["owner"] == 0
    assert planet["ships"] == 1


def test_moving_comet_sweeps_stationary_fleet_position():
    comet_path = [[[10.0, 10.0], [12.0, 10.0]]]
    engine = make_engine(
        obs(
            [[5, -1, 10.0, 10.0, 1.0, 0, 1]],
            fleets=[[11, 0, 11.0, 10.0, math.pi / 2.0, -1, 5]],
            comet_planet_ids=[5],
            comets=[{"planet_ids": [5], "paths": comet_path, "path_index": 0}],
        )
    )
    engine.step([])
    planet = planet_by_id(engine.debug_state(), 5)
    assert planet["owner"] == 0
    assert planet["ships"] == 5


def test_tied_attackers_destroy_each_other_without_capture():
    engine = make_engine(
        obs(
            [[3, -1, 80.0, 80.0, 2.0, 10, 1]],
            fleets=[
                [1, 0, 78.0, 80.0, 0.0, -1, 5],
                [2, 1, 82.0, 80.0, math.pi, -1, 5],
            ],
        )
    )
    engine.step([])
    planet = planet_by_id(engine.debug_state(), 3)
    assert planet["owner"] == -1
    assert planet["ships"] == 10


def test_static_interception_action_format_and_angle():
    engine = make_engine(
        obs(
            [
                [0, 0, 10.0, 10.0, 2.0, 30, 2],
                [1, -1, 25.0, 10.0, 2.0, 5, 3],
            ]
        )
    )
    actions = engine.choose_actions(50, seed=1)
    assert actions
    first = actions[0]
    assert isinstance(first[0], int)
    assert isinstance(first[1], float)
    assert isinstance(first[2], int)
    assert first[0] == 0
    assert abs(first[1]) < 0.05
    assert 1 <= total_launched_from(actions, 0) <= 30


def test_macro_packer_does_not_overspend_source():
    engine = make_engine(
        obs(
            [
                [0, 0, 10.0, 10.0, 2.0, 8, 1],
                [1, -1, 25.0, 10.0, 2.0, 2, 3],
                [2, -1, 10.0, 25.0, 2.0, 2, 3],
            ]
        )
    )
    actions = engine.choose_actions(100, seed=2)
    assert total_launched_from(actions, 0) <= 8


def test_main_agent_returns_orbit_wars_action_list():
    observation = obs(
        [
            [0, 0, 10.0, 10.0, 2.0, 30, 2],
            [1, -1, 25.0, 10.0, 2.0, 5, 3],
        ]
    )
    actions = main.agent(observation)
    assert isinstance(actions, list)
    assert all(isinstance(row, list) and len(row) == 3 for row in actions)


def test_hot_sources_avoid_dynamic_allocation_primitives():
    hot_files = [
        "src/orbit_engine_sim.cpp",
        "src/orbit_engine_candidate.cpp",
        "src/orbit_engine_search.cpp",
        "src/orbit_engine_eval.cpp",
        "src/orbit_engine_geometry.cpp",
    ]
    forbidden = ("std::vector", "std::set", "make_unique", "make_shared", "std::function", "std::async")
    root = os.path.dirname(__file__)
    for rel in hot_files:
        with open(os.path.join(root, rel), encoding="utf-8") as handle:
            source = handle.read()
        for token in forbidden:
            assert token not in source


def test_packaged_submission_jit_compiles_in_extracted_directory():
    pytest.importorskip("pybind11")
    package_path = package_submission.build_package()
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(package_path, "r:gz") as tar:
            try:
                tar.extractall(tmp, filter="data")
            except TypeError:
                tar.extractall(tmp)
        smoke = subprocess.run(
            [sys.executable, "main.py"],
            cwd=tmp,
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert smoke.returncode == 0, smoke.stdout + smoke.stderr
        assert "orbit_engine native available: True" in smoke.stdout

        call = subprocess.run(
            [
                sys.executable,
                "-c",
                "from types import SimpleNamespace\n"
                "from main import agent\n"
                "obs = SimpleNamespace(player=0, step=0, angular_velocity=0.0, "
                "planets=[[0,0,10.0,10.0,2.0,30,2],[1,-1,25.0,10.0,2.0,5,3]], "
                "fleets=[], initial_planets=[[0,0,10.0,10.0,2.0,30,2],[1,-1,25.0,10.0,2.0,5,3]], "
                "comets=[], comet_planet_ids=[])\n"
                "actions = agent(obs)\n"
                "assert isinstance(actions, list)\n"
                "print(actions)\n",
            ],
            cwd=tmp,
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert call.returncode == 0, call.stdout + call.stderr


def test_kaggle_environment_smoke_when_available():
    kaggle_environments = pytest.importorskip("kaggle_environments")
    env = kaggle_environments.make("orbit_wars", configuration={"seed": 1}, debug=True)
    env.run(["main.py", "random"])
    assert env.steps[-1][0].status in {"DONE", "ACTIVE", "TIMEOUT", "ERROR"}


if __name__ == "__main__":
    observation = SimpleNamespace(
        player=0,
        step=0,
        angular_velocity=0.0,
        planets=[[0, 0, 10.0, 10.0, 2.0, 30, 2], [1, -1, 25.0, 10.0, 2.0, 5, 3]],
        fleets=[],
        initial_planets=[[0, 0, 10.0, 10.0, 2.0, 30, 2], [1, -1, 25.0, 10.0, 2.0, 5, 3]],
        comets=[],
        comet_planet_ids=[],
    )
    assert isinstance(main.agent(observation), list)
    print("orbit_engine smoke tests passed")
