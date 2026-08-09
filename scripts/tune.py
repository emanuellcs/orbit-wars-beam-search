"""Optuna tuning for the Orbit Wars beam-search engine.

Orbit Wars is scored separately for 2-player and 4-player games, and the two
formats reward different play, so tuning runs a dedicated study per format:

- ``--players 2`` scores head-to-head win rate against the opponent pool with
  full seat rotation over fixed seeds.
- ``--players 4`` scores 1st-place rate in a four-player seat-rotated FFA.

Each study writes its champion back to ``config/hyperparameters.json``
(``champion`` for 2p, ``champion_4p`` for 4p) and registers it in
``config/league.json``.  ``main.py`` routes by the observed player count.
"""

from __future__ import annotations

import argparse
import functools
import logging
import multiprocessing
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

multiprocessing.set_start_method("spawn", force=True)

import optuna  # noqa: E402
from kaggle_environments import evaluate  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import main  # noqa: E402
from opponents import load_opponent  # noqa: E402
from tuning import evaluation, league as league_mod, optuna_runner, reporting, schema  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG = logging.getLogger("tune")

HANDCRAFTED_OPPONENTS = ("baseline", "greedy", "mirror", "reinforcer", "all_in_rush", "economizer")
FAILURE_SCORE = evaluation.FAILURE_SCORE


@dataclass(frozen=True)
class EvalConfig:
    """Serializable match settings."""

    players: int
    seeds: int
    base_seed: int
    max_steps: int
    timeout: float
    opponent_pool: tuple  # (kind, name, params)


def _make_opponent_agent(kind, name, params):
    if kind == "handcrafted":
        return load_opponent(name).agent

    def checkpoint_agent(obs, config=None):
        main.set_hyperparameters(**params)
        return main.agent(obs, config)

    return checkpoint_agent


def _play_match(agents, seed, max_steps, timeout):
    """Return per-player rewards or None on crash/timeout."""
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
        return None
    return [float(r) for r in results[0]]


def _rotate_agents(candidate, opponents, seat, players):
    """Build the Kaggle agent list with the candidate seated at ``seat``."""
    agents = []
    opp_idx = 0
    for i in range(players):
        if i == seat:
            agents.append(candidate)
        else:
            agents.append(opponents[opp_idx % max(1, len(opponents))])
            opp_idx += 1
    return agents


def _score_2p(rewards, seat):
    my = rewards[seat]
    opp = rewards[1 - seat]
    return 1.0 if my > opp else (0.5 if my == opp else 0.0)


def _score_4p(rewards, seat):
    my = rewards[seat]
    others = [r for i, r in enumerate(rewards) if i != seat]
    return 1.0 if my > max(others) else (0.5 if my == max(others) else 0.0)


def evaluate_hp(hp, config: EvalConfig) -> float:
    """Score one hyperparameter set in a worker process."""
    try:
        main.set_hyperparameters(**hp)
        opponents = [_make_opponent_agent(*o) for o in config.opponent_pool]
        seats = range(config.players)
        total = 0.0
        games = 0
        for i in range(config.seeds):
            seed = config.base_seed + i
            for seat in seats:
                main.set_hyperparameters(**hp)
                agents = _rotate_agents(main.agent, opponents, seat, config.players)
                rewards = _play_match(agents, seed, config.max_steps, config.timeout)
                if rewards is None or len(rewards) != config.players:
                    continue
                score = _score_2p(rewards, seat) if config.players == 2 else _score_4p(rewards, seat)
                total += score
                games += 1
        rate = total / max(1, games)
        _LOG.info("hp rate=%.3f games=%d", rate, games)
        return rate
    except Exception:  # noqa: BLE001
        _LOG.error("evaluation failed:\n%s", traceback.format_exc())
        return FAILURE_SCORE


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--players", type=int, choices=(2, 4), default=2)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--time-budget", type=int, default=3600)
    parser.add_argument("--storage", default="sqlite:///tune.db")
    parser.add_argument("--study-name")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-steps", type=int, default=499)
    parser.add_argument("--budget-ms", type=int, default=300, help="Per-turn hard stop")
    parser.add_argument("--promote", action="store_true")
    return parser


def main_cli(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.study_name = args.study_name or f"orbit-vs-league-{args.players}p"

    hyper_config = _ROOT / "config" / "hyperparameters.json"
    league_path = _ROOT / "config" / "league.json"
    sch = schema.Schema.load(hyper_config)

    lg = league_mod.League(league_path)
    pool = [("handcrafted", n, None) for n in HANDCRAFTED_OPPONENTS]
    pool += [("checkpoint", e["name"], e.get("params", {})) for e in lg.checkpoints()
             if e["name"] not in HANDCRAFTED_OPPONENTS]

    config = EvalConfig(
        players=args.players,
        seeds=args.seeds,
        base_seed=args.base_seed,
        max_steps=args.max_steps,
        timeout=args.timeout,
        opponent_pool=tuple(pool),
    )

    champion = dict(sch.defaults())
    champion.update(reporting.read_champion(hyper_config))

    best = optuna_runner.run_study(
        objective=functools.partial(evaluate_hp, config=config),
        sample=sch.sample_trial,
        trials=args.trials,
        n_jobs=args.n_jobs,
        storage=args.storage,
        study_name=args.study_name,
        time_budget_s=args.time_budget,
        champion=champion,
        seed_offset=args.base_seed,
    )
    if best is None:
        print("No successful trials; nothing to promote.")
        return 1

    if args.promote:
        reporting.promote_champion(
            hyper_config,
            league_path,
            name=f"champion-{args.players}p",
            params=best,
            note=f"promoted from {args.study_name}",
            champion_key="champion" if args.players == 2 else "champion_4p",
        )
        print(f"Promoted {args.players}p champion -> {hyper_config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
