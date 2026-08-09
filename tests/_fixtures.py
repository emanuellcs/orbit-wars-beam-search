"""Shared fixtures for the Orbit Wars test modules.

The engine is exercised through the pybind bridge with compact dict-style
observations matching the ``main.agent`` contract.
"""

from __future__ import annotations

from pathlib import Path

import main

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_engine():
    """Ensure the native extension is available and return the module."""

    assert main._ensure_native_engine()
    return main.orbit_engine


def obs(planets, fleets=None, **extra):
    """Build a dict-style Orbit Wars observation for tests."""

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
    """Create and initialize a native Engine from a test observation."""

    orbit_engine = load_engine()
    engine = orbit_engine.Engine(int(observation.get("player", 0)))
    engine.update_observation(observation)
    return engine


def planet_by_id(state, planet_id):
    """Return a debug-state planet dict by id or fail the test."""

    for planet in state["planets"]:
        if planet["id"] == planet_id:
            return planet
    raise AssertionError(f"missing planet {planet_id}")


def total_launched_from(actions, planet_id):
    """Sum ships launched from one planet id in an action list."""

    return sum(int(row[2]) for row in actions if int(row[0]) == planet_id)
