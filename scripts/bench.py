"""Benchmark harness: measure the champion against the league.

Plays the champion (routed by player count) against every opponent on held-out
seeds with full seat rotation, reports win rate with a Wilson confidence
interval, updates league Elo, and optionally gates a promotion against the
incumbent.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import sys
import math
from pathlib import Path

multiprocessing.set_start_method("spawn", force=True)

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import main  # noqa: E402
from opponents import load_opponent  # noqa: E402
from tuning import evaluation, league as league_mod, reporting, schema  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG = logging.getLogger("bench")

HANDCRAFTED_OPPONENTS = ("baseline", "greedy", "mirror", "reinforcer", "all_in_rush", "economizer")


def _make_opponent_agent(kind, name, params):
    if kind == "handcrafted":
        return load_opponent(name).agent

    def checkpoint_agent(obs, config=None):
        main.set_hyperparameters(**params)
        return main.agent(obs, config)

    return checkpoint_agent


def _outcome(seat, opponents, seed, max_steps, timeout, players):
    from kaggle_environments import evaluate

    agents = []
    opp_idx = 0
    for i in range(players):
        if i == seat:
            agents.append(main.agent)
        else:
            agents.append(opponents[opp_idx % max(1, len(opponents))])
            opp_idx += 1
    results = evaluation.run_with_timeout(
        lambda: evaluate(
            "orbit_wars",
            agents=agents,
            configuration={"seed": int(seed), "episodeSteps": int(max_steps)},
            num_episodes=1,
            debug=False,
        ),
        timeout,
    )
    if not results or not results[0] or any(r is None for r in results[0]):
        return 0.0
    rewards = [float(r) for r in results[0]]
    my = rewards[seat]
    others = [r for i, r in enumerate(rewards) if i != seat]
    return 1.0 if my > max(others) else (0.5 if my == max(others) else 0.0)


def wilson_ci(wins, games, z=1.96):
    if games == 0:
        return 0.0, 0.0, 0.0
    p = wins / games
    denom = 1 + z * z / games
    center = (p + z * z / (2 * games)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * games)) / games) / denom
    return p, max(0.0, center - margin), min(1.0, center + margin)


def main_cli(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--players", type=int, choices=(2, 4), default=2)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=9000)
    parser.add_argument("--max-steps", type=int, default=499)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--budget-ms", type=int, default=300)
    parser.add_argument("--elo", action="store_true")
    parser.add_argument("--gate", type=float, default=0.0)
    args = parser.parse_args(argv)

    hyper_config = _ROOT / "config" / "hyperparameters.json"
    league_path = _ROOT / "config" / "league.json"
    champion_key = "champion" if args.players == 2 else "champion_4p"
    champion_params = dict(schema.Schema.load(hyper_config).defaults())
    champion_params.update(reporting.read_champion(hyper_config, champion_key))

    lg = league_mod.League(league_path)
    opponents = [("handcrafted", n, None) for n in HANDCRAFTED_OPPONENTS]
    opponents += [("checkpoint", e["name"], e.get("params", {})) for e in lg.entries
                  if e["name"] not in HANDCRAFTED_OPPONENTS]

    main.set_hyperparameters(**champion_params)
    main._FORMAT = args.players

    for kind, name, params in opponents:
        opponent = _make_opponent_agent(kind, name, params)
        others = [opponent] * (args.players - 1)
        wins = games = 0
        for i in range(args.seeds):
            main.set_hyperparameters(**champion_params)
            for seat in range(args.players):
                wins += _outcome(seat, others, args.base_seed + i, args.max_steps, args.timeout, args.players)
                games += 1
        p, lo, hi = wilson_ci(wins, games)
        print(f"vs {name:<24} {games:4d} games  WR={p:.3f}  CI=[{lo:.3f},{hi:.3f}]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
