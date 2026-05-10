"""Local timing harness for Orbit Wars beam-search smoke experiments.

The script builds deterministic synthetic observations and repeatedly calls the
production ``agent`` entrypoint, giving quick latency and action-count feedback
without running full Kaggle matches.
"""

from __future__ import annotations

import argparse
import statistics
import time

from main import agent


def synthetic_obs(seed: int):
    """Build a deterministic four-planet observation for a smoke run.

    Args:
        seed: Integer used to vary planet offsets and starting ships.

    Returns:
        dict: Observation with the same row shapes expected by ``main.agent``.
    """

    offset = float(seed % 7)
    return {
        "player": 0,
        "step": 0,
        "angular_velocity": 0.035,
        "planets": [
            [0, 0, 12.0, 12.0, 2.0, 30 + seed % 5, 2],
            [1, -1, 28.0 + offset, 12.0, 2.0, 5, 3],
            [2, 1, 84.0, 84.0, 2.0, 30, 2],
            [3, -1, 70.0 - offset, 80.0, 2.0, 8, 4],
        ],
        "fleets": [],
        "initial_planets": [
            [0, 0, 12.0, 12.0, 2.0, 30 + seed % 5, 2],
            [1, -1, 28.0 + offset, 12.0, 2.0, 5, 3],
            [2, 1, 84.0, 84.0, 2.0, 30, 2],
            [3, -1, 70.0 - offset, 80.0, 2.0, 8, 4],
        ],
        "comets": [],
        "comet_planet_ids": [],
    }


def main() -> int:
    """Run repeated synthetic observations and print aggregate latency metrics.

    Returns:
        int: Process exit status.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=16)
    args = parser.parse_args()
    elapsed = []
    action_counts = []
    for seed in range(args.seeds):
        obs = synthetic_obs(seed)
        start = time.perf_counter()
        actions = agent(obs)
        elapsed.append((time.perf_counter() - start) * 1000.0)
        action_counts.append(len(actions))
    print(f"runs={args.seeds}")
    print(f"mean_ms={statistics.mean(elapsed):.2f}")
    print(f"max_ms={max(elapsed):.2f}")
    print(f"mean_actions={statistics.mean(action_counts):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
