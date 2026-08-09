"""Simulator tests: launch legality, collisions, comet sweeps, and combat."""

from __future__ import annotations

import math

from _fixtures import make_engine, obs, planet_by_id


def test_launch_spend_legality_and_production_ordering():
    """Reject overspending launches and apply production after launch spending."""

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
    """Remove a fleet whose continuous segment crosses the sun radius."""

    engine = make_engine(
        obs(
            [[0, 0, 20.0, 20.0, 2.0, 10, 1]],
            fleets=[[9, 0, 35.0, 50.0, 0.0, 0, 1000]],
        )
    )
    engine.step([])
    assert engine.debug_state()["fleets"] == []


def test_planet_collision_resolves_capture():
    """Queue a fleet-planet collision and resolve the capture."""

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
    """Capture fleets swept by a moving comet between path samples."""

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
    """Keep ownership unchanged when top attacking forces tie."""

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
