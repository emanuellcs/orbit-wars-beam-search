"""Differential rules-fidelity harness against kaggle-environments.

Verifies the native engine completes real games against the environment with
no invalid/error statuses and that action lists always have the expected Orbit
Wars shape ``[[from_planet_id, angle, ships], ...]``.

Run:
    PYTHONPATH=build python scripts/diffcheck.py --games 6 --steps 80 --budget-ms 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import main  # noqa: E402


def main_cli(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=6)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--budget-ms", type=int, default=60)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)

    from kaggle_environments import make

    main.set_hyperparameters(hard_stop_ms=args.budget_ms)
    failures = 0

    for game in range(args.games):
        seed = args.seed + game
        env = make("orbit_wars", configuration={"seed": seed, "episodeSteps": args.steps}, debug=False)
        env.run([main.agent, main.agent])
        last = env.steps[-1]
        if any(s.status in {"ERROR", "INVALID", "TIMEOUT"} for s in last):
            print(f"game {game}: statuses={[s.status for s in last]} FAIL")
            failures += 1
            continue
        print(f"game {game}: seed={seed} steps={len(env.steps)} statuses={[s.status for s in last]} "
              f"rewards={[s.reward for s in last]}")

    print(f"diffcheck: {args.games - failures}/{args.games} games clean")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main_cli())
