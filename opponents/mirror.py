"""Mirror-symmetric Orbit Wars opponent.

Identical tactic to ``baseline`` (nearest-target with a defensive reserve), but
flips coordinates through the board's center before aiming. This forces the
tuned agent to defend against flanking pressure that is not on the straight
line to the nearest enemy, which the symmetric ``baseline`` policy never
produces on its own side of the board.

The mirroring deliberately targets the opponent half of the map so the policy
plays aggressively even when the agent is on the same side of the board.
"""

from __future__ import annotations

import math


BOARD_SIZE = 100.0


def _get(obj, name, default=None):
    """Read a field from a dict-like or attribute-style observation."""

    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _mirror(planet):
    """Return the planet's mirror image across both board axes."""

    return (
        int(planet[0]),
        int(planet[1]),
        BOARD_SIZE - float(planet[2]),
        BOARD_SIZE - float(planet[3]),
        float(planet[4]),
        int(planet[5]),
        int(planet[6]),
    )


def agent(obs, config=None):
    """Return mirrored-coord nearest-target launch actions.

    Args:
        obs: Dict-like or attribute-style Orbit Wars observation.
        config: Optional Kaggle configuration object, accepted for compatibility.

    Returns:
        list[list[int | float]]: Launch rows ``[from_planet_id, angle, ships]``.
    """

    del config
    player = int(_get(obs, "player", 0))
    planets = _get(obs, "planets", []) or []
    if not planets:
        return []
    mirrored = [_mirror(p) for p in planets]
    moves = []
    my_planets = [p for p in planets if int(p[1]) == player]
    if not my_planets:
        return []
    targets = [p for p in mirrored if int(p[1]) != player]
    for mine in my_planets:
        if not targets:
            break
        target = min(
            targets,
            key=lambda p: math.hypot(float(p[2]) - float(mine[2]), float(p[3]) - float(mine[3])),
        )
        # Map the mirrored target back to the real planet by id.
        real_target = next((q for q in planets if int(q[0]) == int(target[0])), None)
        if real_target is None:
            continue
        ships = int(real_target[5]) + 1
        reserve = max(5, int(mine[6]) * 4)
        if int(mine[5]) - reserve >= ships:
            moves.append(
                [
                    int(mine[0]),
                    math.atan2(float(real_target[3]) - float(mine[3]), float(real_target[2]) - float(mine[2])),
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
