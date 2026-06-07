"""Aggressive greedy Orbit Wars opponent.

Same nearest-target heuristic as ``baseline`` but commits ships at one third
of the production-based reserve. This produces relentless pressure without
caring about counter-attacks, which is the policy our tuned agent must learn
to repel. Because it overruns itself late-game, a tuned agent can usually
outlast it through better pacing.
"""

from __future__ import annotations

import math


def _get(obj, name, default=None):
    """Read a field from a dict-like or attribute-style observation."""

    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


_RESERVE_DIVISOR = 3  # baseline uses *4; this policy keeps a much thinner reserve


def agent(obs, config=None):
    """Return greedy nearest-target launch actions with a thin defensive reserve.

    Args:
        obs: Dict-like or attribute-style Orbit Wars observation.
        config: Optional Kaggle configuration object, accepted for compatibility.

    Returns:
        list[list[int | float]]: Launch rows ``[from_planet_id, angle, ships]``.
    """

    del config
    player = int(_get(obs, "player", 0))
    planets = _get(obs, "planets", []) or []
    moves = []
    my_planets = [p for p in planets if int(p[1]) == player]
    targets = [p for p in planets if int(p[1]) != player]
    for mine in my_planets:
        if not targets:
            break
        target = min(
            targets,
            key=lambda p: math.hypot(float(p[2]) - float(mine[2]), float(p[3]) - float(mine[3])),
        )
        ships = int(target[5]) + 1
        reserve = max(2, int(mine[6]) * _RESERVE_DIVISOR)
        if int(mine[5]) - reserve >= ships:
            moves.append(
                [
                    int(mine[0]),
                    math.atan2(float(target[3]) - float(mine[3]), float(target[2]) - float(mine[2])),
                    ships,
                ]
            )
    return moves


def act(obs, config=None):
    """Safe wrapper matching alternative local runner conventions."""

    try:
        return agent(obs, config)
    except Exception:
        return []
