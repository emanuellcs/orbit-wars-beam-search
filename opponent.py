"""Simple Orbit Wars benchmark opponent used for local smoke matches."""

from __future__ import annotations

import math


def _get(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def agent(obs, config=None):
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
        reserve = max(5, int(mine[6]) * 4)
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
    try:
        return agent(obs, config)
    except Exception:
        return []
