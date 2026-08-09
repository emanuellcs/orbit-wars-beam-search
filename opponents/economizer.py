"""Economizer archetype: production-first, defensive, plays the ship-count endgame.

Captures high-production planets with exact-sized packets, keeps a fat reserve,
and avoids harassment.  Wins by out-producing the opponent and converting that
into a ship-count lead at the time limit.
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
        prod = int(planet[6])
        if owner != player or ships <= 0:
            continue
        # Prefer the highest-production reachable target.
        targets = [p for p in planets if int(p[1]) != player]
        if not targets:
            break
        target = max(targets, key=lambda p: int(p[6]) - math.hypot(p[2] - planet[2], p[3] - planet[3]) / 40.0)
        capture = int(target[5]) + 1
        reserve = max(10, prod * 6)  # fat reserve protects the economy
        if ships - reserve >= capture:
            moves.append([
                pid,
                math.atan2(float(target[3]) - float(planet[3]), float(target[2]) - float(planet[2])),
                capture,
            ])
    return moves


def act(obs, config=None):
    try:
        return agent(obs, config)
    except Exception:
        return []
