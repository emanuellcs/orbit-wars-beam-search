"""Reinforcer archetype: prioritises holding owned planets over capturing.

The #1 skill gap against top-tier play is reinforcement — sending ships to your
own threatened planets before they fall.  This opponent computes, for each owned
planet, the incoming enemy mass projected over the next ``WINDOW`` ticks and
reinforces whenever it would otherwise be overwhelmed.  It still captures
lightly, but defense comes first.
"""

from __future__ import annotations

import math

WINDOW = 20.0


def _get(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def agent(obs, config=None):
    del config
    player = int(_get(obs, "player", 0))
    planets = _get(obs, "planets", []) or []
    fleets = _get(obs, "fleets", []) or []
    moves = []

    by_id = {int(p[0]): p for p in planets}

    def project_arrivals(planet_id):
        """Sum enemy ships expected to reach ``planet_id`` within WINDOW ticks."""
        target = by_id[planet_id]
        mass = 0.0
        for f in fleets:
            if int(f[1]) == player:
                continue
            fx, fy, angle = float(f[2]), float(f[3]), float(f[4])
            speed = 1.0 + 5.0 * (math.log(max(1, int(f[6]))) / math.log(1000.0)) ** 1.5
            dx = target[2] - fx
            dy = target[3] - fy
            dot = math.cos(angle) * dx + math.sin(angle) * dy
            dist = math.hypot(dx, dy)
            if dist <= 0:
                continue
            if dot > dist * 0.85 and dist / speed <= WINDOW:
                mass += int(f[6])
        return mass

    for planet in planets:
        pid = int(planet[0])
        owner = int(planet[1])
        ships = int(planet[5])
        prod = int(planet[6])
        if owner != player or ships <= 0:
            continue
        incoming = project_arrivals(pid)
        needed = incoming + 1
        surplus = ships - needed
        if surplus >= 0:
            continue  # planet is safe; no reinforcement needed
        shortfall = needed - ships
        # Pull reinforcement from the nearest allied source with spare ships.
        best = None
        best_dist = float("inf")
        for src in planets:
            sid = int(src[0])
            if int(src[1]) != player or sid == pid:
                continue
            if int(src[5]) > 10 + int(src[6]) * 4:
                d = math.hypot(src[2] - planet[2], src[3] - planet[3])
                if d < best_dist:
                    best_dist = d
                    best = src
        if best is not None and shortfall > 0:
            send = min(shortfall, int(best[5]) - (10 + int(best[6]) * 4))
            if send > 0:
                moves.append([
                    int(best[0]),
                    math.atan2(float(planet[3]) - float(best[3]), float(planet[2]) - float(best[2])),
                    send,
                ])

    # Light capture after defense.
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
        capture = int(target[5]) + 1
        reserve = max(10, int(planet[6]) * 4)
        if int(planet[5]) - reserve >= capture:
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
