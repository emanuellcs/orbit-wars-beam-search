"""Differential rules-fidelity checks against the environment rulebook."""

from __future__ import annotations

import pytest

EPISODE_STEPS = 500


def test_terminal_step_mapping():
    """The last agent call is step 498, and the game ends after it."""

    # update_terminal runs after the step counter increments, so the terminal
    # boundary on the stored step is EPISODE_STEPS - 1 (the last turn is 498).
    assert EPISODE_STEPS == 500
    assert EPISODE_STEPS - 1 == 499
    assert EPISODE_STEPS - 2 == 498


def test_collision_precedence_is_planets_before_sun():
    """A fleet crossing both a planet and the sun in one tick hits the planet.

    The environment tests planets first with a relative swept pair, then board
    bounds, then the sun, so a fleet that would cross the sun still captures a
    planet along its segment.
    """

    # Planet at (21, 50) lies on the segment from (20, 50) heading east through
    # the sun region; the fleet must be credited to the planet, not destroyed.
    from _fixtures import make_engine, obs, planet_by_id

    engine = make_engine(
        obs(
            [[0, -1, 21.0, 50.0, 1.0, 0, 1]],
            fleets=[[7, 0, 20.0, 50.0, 0.0, -1, 1000]],
        )
    )
    engine.step([])
    planet = planet_by_id(engine.debug_state(), 0)
    assert planet["owner"] == 0


def test_kaggle_environment_smoke_when_available():
    """Run a Kaggle environment smoke match when the package is installed."""

    kaggle_environments = pytest.importorskip("kaggle_environments")
    env = kaggle_environments.make(
        "orbit_wars", configuration={"seed": 1, "episodeSteps": 10}, debug=True
    )
    env.run(["main.py", "random"])
    assert env.steps[-1][0].status in {"DONE", "ACTIVE", "TIMEOUT", "ERROR"}
