"""Pybind bridge tests: geometry, observation parsing, entrypoint, hyperparameters."""

from __future__ import annotations

import pytest

import main
from _fixtures import load_engine, make_engine, obs, planet_by_id


def test_speed_formula_matches_rules():
    """Verify the native speed curve matches key Orbit Wars rule anchors."""

    orbit_engine = load_engine()
    assert orbit_engine.speed_for_ships(1) == pytest.approx(1.0)
    assert orbit_engine.speed_for_ships(1000) == pytest.approx(6.0)
    assert 4.0 < orbit_engine.speed_for_ships(500) < 6.0


def test_observation_parses_orbiting_static_and_comets():
    """Verify observation loading classifies orbiting, static, and comet planets."""

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


def test_main_agent_returns_orbit_wars_action_list():
    """Verify the Kaggle entrypoint returns launch rows of the required shape."""

    observation = obs(
        [
            [0, 0, 10.0, 10.0, 2.0, 30, 2],
            [1, -1, 25.0, 10.0, 2.0, 5, 3],
        ]
    )
    actions = main.agent(observation)
    assert isinstance(actions, list)
    assert all(isinstance(row, list) and len(row) == 3 for row in actions)


def test_set_hyperparameters_round_trips_through_engine():
    """set_hyperparameters must accept known keys, return them, and survive agent() calls."""

    main._ENGINES.clear()
    main._HYPERPARAMS.clear()

    observation = obs(
        [
            [0, 0, 10.0, 10.0, 2.0, 30, 2],
            [1, -1, 25.0, 10.0, 2.0, 5, 3],
        ]
    )
    baseline_actions = main.agent(observation)
    assert isinstance(baseline_actions, list)

    main.set_hyperparameters(
        beam_width=64,
        search_depth=2,
        rollout_horizon=16,
        hard_stop_ms=300,
        ship=1.5,
        production=12.0,
        territory_own=0.5,
        territory_opp=0.25,
        threat=2.0,
        comet_owned=0.0,
        comet_enemy=0.0,
        comet_neutral=0.0,
        owner_enemy=10.0,
        owner_neutral=5.0,
        owner_self=-1.0,
        comet_bonus=0.0,
        prod_per_unit=5.0,
        kind_exact=0.0,
        kind_over=0.0,
        kind_all_safe=0.0,
        kind_harass=0.0,
        eta_discount=0.1,
        ship_cost=0.01,
        search_threads=1,
    )
    stored = main.get_hyperparameters()
    assert stored["beam_width"] == 64
    assert stored["production"] == 12.0
    assert stored["ship_cost"] == 0.01
    assert stored["search_threads"] == 1

    tuned_actions = main.agent(observation)
    assert isinstance(tuned_actions, list)
    assert all(isinstance(row, list) and len(row) == 3 for row in tuned_actions)

    with pytest.raises(TypeError):
        main.set_hyperparameters(definitely_not_a_real_key=1.0)
