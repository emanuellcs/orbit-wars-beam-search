"""Opponent loader and archetype registration tests."""

from __future__ import annotations

import random as _random

import pytest


def test_opponents_loader_lists_available_policies():
    """The opponents loader must expose all registered policies."""

    from opponents import list_opponents, load_opponent

    names = list_opponents()
    assert "baseline" in names
    assert "mirror" in names
    assert "greedy" in names
    for name in names:
        module = load_opponent(name)
        assert callable(module.agent)


def test_opponents_random_set_ffa_and_1v1():
    """random_opponent_set must return the requested number of opponents."""

    from opponents import random_opponent_set

    rng = _random.Random(42)
    one = random_opponent_set(1, rng)
    assert len(one) == 1
    three = random_opponent_set(3, rng)
    assert len(three) == 3
    for name in one + three:
        assert isinstance(name, str)


@pytest.mark.parametrize("name", ["baseline", "greedy", "mirror", "reinforcer", "all_in_rush", "economizer"])
def test_archetypes_import_and_return_action_list(name):
    """Every archetype must expose a callable returning Orbit Wars launch rows."""

    from opponents import load_opponent

    module = load_opponent(name)
    obs = {
        "player": 0,
        "step": 0,
        "angular_velocity": 0.05,
        "planets": [[0, 0, 10.0, 10.0, 2.0, 30, 2], [1, -1, 25.0, 10.0, 2.0, 5, 3]],
        "fleets": [],
        "initial_planets": [[0, 0, 10.0, 10.0, 2.0, 30, 2], [1, -1, 25.0, 10.0, 2.0, 5, 3]],
        "comets": [],
        "comet_planet_ids": [],
    }
    actions = module.agent(obs)
    assert isinstance(actions, list)
    for row in actions:
        assert len(row) == 3
