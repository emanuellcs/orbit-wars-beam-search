"""All-in rush archetype: aggressive early capture waves, minimal reserve.

Prioritises tempo: every source planet launches the largest affordable packet at
the nearest non-owned target whenever it can keep a thin reserve.  Exposes the
weakness of passive openers and forces the opponent to respond or lose the map.
"""

from __future__ import annotations

import math


def _get(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def agent(obs, config=None):
    del config
    player = int(_get(obs, "player", 0))
    planets = _get(obs, "planets", []) or []
    moves = []

    for planet in planets:
        pid = int(planet[0])
        owner = int(planet[1])
        ships = int(planet[5])
        if owner != player or ships <= 0:
            continue
        targets = [p for p in planets if int(p[1]) != player]
        if not targets:
            break
        target = min(targets, key=lambda p: math.hypot(p[2] - planet[2], p[3] - planet[3]))
        # All-in: keep only a thin reserve (3x production).
        reserve = max(3, int(planet[6]) * 3)
        send = ships - reserve
        if send > 0:
            moves.append([
                pid,
                math.atan2(float(target[3]) - float(planet[3]), float(target[2]) - float(planet[2])),
                send,
            ])
    return moves


def act(obs, config=None):
    try:
        return agent(obs, config)
    except Exception:
        return []
