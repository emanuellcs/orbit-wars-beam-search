"""Loader and helpers for the ``opponents/`` package.

The tuning harness uses :func:`list_opponents` to enumerate all available
Python opponent policies, :func:`load_opponent` to import a specific policy as
a callable, and :func:`random_opponent_set` to sample a set of opponents for
either 1v1 (two players) or FFA (four players) matches. The string sentinel
``"random"`` is always available as a shortcut for the built-in
``kaggle_environments`` random agent.
"""

from __future__ import annotations

import importlib
import os
import random
from types import ModuleType
from typing import Iterable, List, Sequence


_HERE = os.path.dirname(os.path.abspath(__file__))
_EXCLUDE = {"__init__.py", "__pycache__"}


def list_opponents() -> List[str]:
    """Return the sorted list of available opponent module names.

    Returns:
        list[str]: Module stems (no ``.py`` suffix) under the ``opponents/``
            directory, excluding this loader and any ``__pycache__`` artifact.
    """

    names: List[str] = []
    for entry in sorted(os.listdir(_HERE)):
        if entry in _EXCLUDE:
            continue
        if entry.endswith(".py"):
            names.append(entry[:-3])
    return names


def load_opponent(name: str) -> ModuleType:
    """Import a single opponent module by stem name.

    Args:
        name: Opponent module stem under ``opponents/``.

    Returns:
        module: The imported module. The module is expected to expose a
        callable named ``agent`` (and an alias ``act``) accepting the same
        observation shape as Kaggle's Orbit Wars environment.

    Raises:
        FileNotFoundError: When ``name`` is not present in the directory.
        ImportError: When the module exists but fails to import.
    """

    if name not in list_opponents():
        raise FileNotFoundError(
            f"opponent '{name}' is not in opponents/; available: {list_opponents()}"
        )
    return importlib.import_module(f"opponents.{name}")


def make_callable(name: str):
    """Return the ``agent`` callable of an opponent module.

    Args:
        name: Opponent module stem or the literal ``"random"`` to reference
            the Kaggle built-in random agent by string (no Python import).

    Returns:
        Callable or str: The ``agent`` function from the module, or the
        string ``"random"`` when ``name == "random"``.

    Raises:
        FileNotFoundError: When ``name`` is not a built-in sentinel and is
            also not present in the opponents directory.
    """

    if name == "random":
        return "random"
    module = load_opponent(name)
    return module.agent


def random_opponent_set(n: int, rng: random.Random,
                        exclude: Sequence[str] = ()) -> List[str]:
    """Sample ``n`` opponent module names (or ``'random'``) for a match.

    Sampling is with replacement so the same policy can be picked twice in a
    4-player FFA. The ``exclude`` argument drops specific names from the
    candidate pool before sampling.

    Args:
        n: Number of opponent slots to fill. ``1`` for a 1v1 opponent;
            ``3`` to fill the FFA slots around the agent.
        rng: ``random.Random`` instance used for sampling (caller-controlled
            so the tuning harness can seed its own RNG).
        exclude: Iterable of opponent names to skip.

    Returns:
        list[str]: A list of length ``n`` containing module stems or the
        ``"random"`` sentinel.
    """

    pool = [name for name in list_opponents() + ["random"] if name not in set(exclude)]
    if not pool:
        raise ValueError("opponent pool is empty after exclusions")
    return [rng.choice(pool) for _ in range(n)]


def resolve_agent_list(controlled_callable, opponent_names: Iterable[str]) -> list:
    """Build a Kaggle agent list starting with ``controlled_callable``.

    Args:
        controlled_callable: The tuned agent callable, placed at index 0.
        opponent_names: Names from :func:`random_opponent_set` or any caller
            that wants a deterministic ordering.

    Returns:
        list: A list ``[controlled_callable, *opponents]`` ready to pass to
        :func:`kaggle_environments.evaluate` or ``env.run``.
    """

    agents = [controlled_callable]
    for name in opponent_names:
        agents.append(make_callable(name))
    return agents
