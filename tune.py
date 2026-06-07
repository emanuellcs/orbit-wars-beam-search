"""Optuna-based tuning harness for the Orbit Wars beam-search engine.

This script runs a parallelized Bayesian Optimization loop against the local
``kaggle_environments`` Orbit Wars environment. The objective function:

* Samples search, evaluator, and candidate-prior hyperparameters via Optuna's
  TPE sampler (continuous, smooth gradients).
* Applies the hyperparameters to the native ``orbit_engine`` through
  ``main.set_hyperparameters`` so subsequent ``agent()`` calls see them.
* For each requested seed, plays one ``kaggle_environments`` match against a
  randomly chosen opponent set drawn from the ``opponents/`` directory. Both
  1v1 and 4-player FFA matches are sampled according to ``--ffa-prob``.
* Aggregates win rate, draw rate, normalized ship margin, and reward z-score
  into a single composite objective dominated by win rate.

The whole evaluation is wrapped in ``try/except`` so any single trial that
crashes (killed by a timeout, environment bug, pybind11 fault) only loses
its own reward, not the rest of the study.

Notes on parallelization:

* ``multiprocessing.set_start_method("spawn", force=True)`` is configured at
  import time so the Optuna process workers do not fork a half-constructed
  C++ ``std::thread`` pool (which deadlocks on Linux). Spawn is mandatory
  whenever the parent process has already imported and initialized the
  pybind11 module.
* Each spawned worker calls ``set_hyperparameters(..., search_threads=1)``
  before its first ``agent()`` call so the C++ search never spawns more than
  one worker thread per process. With ``--n-jobs=16`` on a 20-core host that
  caps total active search threads at 16, leaving the remaining 4 cores for
  the OS scheduler.

CLI surface (required flags):

* ``--trials`` (default 1000): total Optuna trials to execute.
* ``--n-jobs`` (default 16): parallel process workers; honored only when >1.
* ``--seeds`` (default 5): episodes per trial.
* ``--time-budget`` (default 300): wall-clock seconds for the study.
* ``--storage`` (default ``sqlite:///tune.db``): Optuna RDB URL.
* ``--study-name`` (default ``orbit-vs-opponent``): Optuna study identifier.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import random
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence

# ----------------------------------------------------------------------------
# Multiprocessing bootstrap. MUST happen before any optuna/kaggle import that
# might internally fork, since pybind11 + std::thread in the parent process
# cannot be safely fork()-cloned.
# ----------------------------------------------------------------------------
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    # start method already locked by a prior call; that is fine.
    pass

import optuna
from optuna.samplers import TPESampler

# Cap the C++ search worker thread pool BEFORE main.py imports the native
# module. With the default MAX_SEARCH_THREADS=20 the search thrashes a
# 20-core host to death (each ``agent()`` call takes 10+ seconds) and the
# Kaggle environment's 1-second actTimeout then deadlocks on the slow
# response. The Kaggle runtime is unaffected because it loads main.py as
# a fresh module without this guard.
import orbit_engine  # noqa: E402  (must come after the spawn policy lock)
orbit_engine.set_search_thread_limit(1)

# Configure quiet logging: optuna is chatty by default.
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
_LOG = logging.getLogger("tune")

# Make the project root importable for ``main`` and ``opponents``.
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Defer heavy imports until after the spawn policy is locked and the C++
# thread cap is reduced.
import main  # noqa: E402
from opponents import (  # noqa: E402
    list_opponents,
    make_callable,
    random_opponent_set,
    resolve_agent_list,
)


# ---------------------------------------------------------------------------
# Hyperparameter space definition. Defaults are placed in the middle of each
# range so the TPE prior stays close to the "Grandmaster" baseline.
# ---------------------------------------------------------------------------
_INT_RANGES = {
    "beam_width":      (64,  512),
    "search_depth":    (2,   10),
    "rollout_horizon": (8,   96),
    "hard_stop_ms":    (400, 900),
}

_EVAL_RANGES = {
    "ship":           (0.0, 3.0),
    "production":     (5.0, 60.0),
    "territory_own":  (0.0, 2.0),
    "territory_opp":  (0.0, 1.5),
    "threat":         (0.0, 3.0),
    "comet_owned":    (0.0, 3.0),
    "comet_enemy":    (0.0, 3.0),
    "comet_neutral":  (0.0, 3.0),
}

_CANDIDATE_RANGES = {
    "owner_enemy":   (0.0,   90.0),
    "owner_neutral": (0.0,   60.0),
    "owner_self":    (-90.0, 0.0),
    "comet_bonus":   (0.0,   40.0),
    "prod_per_unit": (0.0,   50.0),
    "kind_exact":    (0.0,   30.0),
    "kind_over":     (0.0,   40.0),
    "kind_all_safe": (0.0,   25.0),
    "kind_harass":   (0.0,   10.0),
    "eta_discount":  (0.0,   2.0),
    "ship_cost":     (0.0,   0.3),
}

# Trials that crash return this heavy penalty so the TPE posterior collapses
# away from the failing region of the search space.
_FAILURE_SCORE = -1.0e6


# ---------------------------------------------------------------------------
# Sample helpers.
# ---------------------------------------------------------------------------
def _sample_hyperparameters(trial: optuna.Trial) -> Dict[str, Any]:
    """Sample one set of hyperparameters from the Optuna trial.

    Args:
        trial: Optuna trial handle from inside the objective.

    Returns:
        dict: ``{name: value}`` keyword arguments ready for
        :func:`main.set_hyperparameters`.
    """

    hp: Dict[str, Any] = {}
    for name, (lo, hi) in _INT_RANGES.items():
        hp[name] = trial.suggest_int(name, lo, hi, log=(name == "beam_width"))
    for name, (lo, hi) in _EVAL_RANGES.items():
        hp[name] = trial.suggest_float(name, lo, hi)
    for name, (lo, hi) in _CANDIDATE_RANGES.items():
        hp[name] = trial.suggest_float(name, lo, hi)
    return hp


# ---------------------------------------------------------------------------
# Match execution.
# ---------------------------------------------------------------------------
def _select_match_shape(rng: random.Random, ffa_prob: float) -> str:
    """Return ``"ffa"`` with probability ``ffa_prob`` else ``"1v1"``."""

    return "ffa" if rng.random() < ffa_prob else "1v1"


def _build_agents(controlled: Any, shape: str, opponent_names: Sequence[str]) -> List[Any]:
    """Build a Kaggle agent list with the controlled agent at index 0."""

    return resolve_agent_list(controlled, opponent_names) if shape == "1v1" \
        else resolve_agent_list(controlled, opponent_names)


def _play_match(agents: List[Any], seed: int, episode_steps: int,
                timeout: float) -> Optional[List[Optional[float]]]:
    """Run a single episode and return the reward list, or ``None`` on crash.

    Args:
        agents: Agent list in player order (index 0 is the controlled bot).
        seed: Deterministic map seed.
        episode_steps: Maximum turns before the environment declares a winner.
        timeout: Wall-clock cap in seconds for this episode.

    Returns:
        list[Optional[float]] | None: Reward per player, or ``None`` when the
        episode raised any exception (the objective treats that as a failure).
    """

    # Local imports keep the cold start cheap in the spawn workers.
    from kaggle_environments import evaluate

    try:
        results = evaluate(
            "orbit_wars",
            agents=agents,
            configuration={
                "seed": int(seed),
                "episodeSteps": int(episode_steps),
            },
            num_episodes=1,
            debug=False,
        )
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("kaggle env crashed: %s", exc)
        return None
    if not results or not results[0]:
        return None
    rewards = list(results[0])
    if any(r is None for r in rewards):
        return None
    return rewards


def _score_match(my_reward: float, opponent_rewards: Sequence[float]) -> Dict[str, float]:
    """Compute per-match metrics for the objective composite.

    Args:
        my_reward: Final reward of the controlled agent (player 0).
        opponent_rewards: Final reward of every other player in the match.

    Returns:
        dict: ``win``, ``draw``, ``ship_margin``, ``score_z`` floats.
    """

    if not opponent_rewards:
        return {"win": 0.0, "draw": 0.0, "ship_margin": 0.0, "score_z": 0.0}
    best_opp = max(opponent_rewards)
    mean_opp = sum(opponent_rewards) / len(opponent_rewards)
    std_opp = max(1.0, (sum((r - mean_opp) ** 2 for r in opponent_rewards) /
                        max(1, len(opponent_rewards) - 1)) ** 0.5)
    return {
        "win":       1.0 if my_reward > best_opp else 0.0,
        "draw":      1.0 if my_reward == best_opp else 0.0,
        "ship_margin": (my_reward - mean_opp) / (abs(mean_opp) + 1.0),
        "score_z":   (my_reward - mean_opp) / std_opp,
    }


def _play_one_episode(rng: random.Random, ffa_prob: float, episode_steps: int,
                      timeout: float) -> Optional[Dict[str, float]]:
    """Run a single 1v1 or FFA episode and return the per-match metrics.

    Args:
        rng: Per-episode RNG shared with the trial so opponent picks stay
            deterministic per trial.
        ffa_prob: Probability of an FFA match.
        episode_steps: Maximum turns for the episode.
        timeout: Wall-clock cap per episode.

    Returns:
        dict | None: ``_score_match`` output, or ``None`` on crash.
    """

    shape = _select_match_shape(rng, ffa_prob)
    if shape == "1v1":
        opponents = random_opponent_set(1, rng)
    else:
        opponents = random_opponent_set(3, rng)
    agents = _build_agents(_tune_agent_callable, shape, opponents)
    if len(agents) < 2:
        return None
    seed = rng.randint(0, 2 ** 31 - 1)
    rewards = _play_match(agents, seed, episode_steps, timeout)
    if rewards is None or len(rewards) < 2:
        return None
    my_reward = float(rewards[0])
    opponent_rewards = [float(r) for r in rewards[1:]]
    return _score_match(my_reward, opponent_rewards)


# ---------------------------------------------------------------------------
# The controlled agent callable, patched in by the parent process so each
# spawn worker actually runs the native engine with the trial's config.
# ---------------------------------------------------------------------------
def _tune_agent_callable(obs, config=None):
    """Forward to the native ``main.agent`` for a single observation.

    Args:
        obs: Kaggle Orbit Wars observation.
        config: Unused; accepted for Kaggle environment compatibility.

    Returns:
        list: Launch rows from the tuned native engine.
    """

    del config
    return main.agent(obs)


# ---------------------------------------------------------------------------
# Objective and study entry points.
# ---------------------------------------------------------------------------
class _Deadline:
    """Wall-clock deadline holder for cooperative trial-level timeouts."""

    def __init__(self, seconds: float) -> None:
        self._end = time.monotonic() + max(0.0, seconds)

    def exceeded(self) -> bool:
        return time.monotonic() >= self._end


def _evaluate_trial(trial: optuna.Trial, args: argparse.Namespace,
                    deadline: _Deadline) -> float:
    """Score one Optuna trial over ``args.seeds`` episodes.

    Args:
        trial: Optuna trial handle.
        args: Parsed CLI arguments.
        deadline: Wall-clock deadline checked before every match.

    Returns:
        float: Composite objective score. ``_FAILURE_SCORE`` for crashes.
    """

    if deadline.exceeded():
        raise optuna.TrialPruned("wall-clock budget exhausted before trial started")

    hp = _sample_hyperparameters(trial)
    # Force the C++ search to use a single worker thread per process so the
    # ``n_jobs`` parallel trials don't oversubscribe the host machine. The
    # ``search_threads`` kwarg is special: it does not change the SearchConfig
    # but flips the runtime thread cap.
    try:
        main.set_hyperparameters(search_threads=1, **hp)
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("set_hyperparameters failed: %s", exc)
        return _FAILURE_SCORE

    rng = random.Random(trial.number + args.seed_offset)
    metrics: List[Dict[str, float]] = []
    for seed_idx in range(args.seeds):
        if deadline.exceeded():
            raise optuna.TrialPruned("wall-clock budget exhausted mid-trial")
        try:
            result = _play_one_episode(rng, args.ffa_prob, args.max_steps, args.timeout)
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("trial %d seed %d crashed: %s", trial.number, seed_idx, exc)
            continue
        if result is not None:
            metrics.append(result)

    if not metrics:
        trial.set_user_attr("failure", "all seeds crashed")
        return _FAILURE_SCORE

    win_rate      = sum(m["win"] for m in metrics) / len(metrics)
    draw_rate     = sum(m["draw"] for m in metrics) / len(metrics)
    ship_margin   = sum(m["ship_margin"] for m in metrics) / len(metrics)
    score_z       = sum(m["score_z"] for m in metrics) / len(metrics)

    composite = (100.0 * win_rate
                 + 20.0 * draw_rate
                 + 1.0 * ship_margin
                 + 0.5 * score_z)

    trial.set_user_attr("win_rate", win_rate)
    trial.set_user_attr("draw_rate", draw_rate)
    trial.set_user_attr("ship_margin", ship_margin)
    trial.set_user_attr("score_z", score_z)
    trial.set_user_attr("n_matches", len(metrics))
    return composite


def _build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser with all required flags and reasonable defaults."""

    parser = argparse.ArgumentParser(
        description="Optuna-based tuning harness for the Orbit Wars beam-search engine.",
    )
    parser.add_argument("--trials",       type=int,   default=1000)
    parser.add_argument("--n-jobs",       type=int,   default=16)
    parser.add_argument("--seeds",        type=int,   default=5)
    parser.add_argument("--time-budget",  type=int,   default=300)
    parser.add_argument("--storage",      type=str,   default="sqlite:///tune.db")
    parser.add_argument("--study-name",   type=str,   default="orbit-vs-opponent")
    parser.add_argument("--max-steps",    type=int,   default=499,
                        help="Episode step cap passed to kaggle_environments.")
    parser.add_argument("--ffa-prob",     type=float, default=0.5,
                        help="Probability that a trial's match is 4-player FFA.")
    parser.add_argument("--timeout",      type=float, default=60.0,
                        help="Per-episode wall-clock cap in seconds.")
    parser.add_argument("--seed-offset",  type=int,   default=0,
                        help="Stable offset added to trial number when seeding RNGs.")
    return parser


def main_cli(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point: configure the study, run trials, and print a summary.

    Args:
        argv: Optional override for ``sys.argv[1:]``; used by unit tests.

    Returns:
        int: Process exit status.
    """

    args = _build_parser().parse_args(argv)
    if not list_opponents():
        print("error: no opponents found under opponents/", file=sys.stderr)
        return 2

    sampler = TPESampler(
        multivariate=True,
        group=True,
        n_startup_trials=min(20, max(1, args.trials // 50)),
    )
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        sampler=sampler,
        direction="maximize",
        load_if_exists=True,
    )

    deadline = _Deadline(args.time_budget)
    print(
        f"[tune] starting study={args.study_name!r} trials={args.trials} "
        f"n_jobs={args.n_jobs} seeds={args.seeds} time_budget={args.time_budget}s "
        f"storage={args.storage!r} opponents={list_opponents()}",
        flush=True,
    )

    def _objective(trial: optuna.Trial) -> float:
        return _evaluate_trial(trial, args, deadline)

    try:
        study.optimize(
            _objective,
            n_trials=args.trials,
            n_jobs=args.n_jobs,
            timeout=args.time_budget,
            show_progress_bar=False,
        )
    except KeyboardInterrupt:
        print("[tune] interrupted; partial study preserved", flush=True)
    except Exception:
        print("[tune] optimization failed:", flush=True)
        traceback.print_exc()

    print("[tune] best value:", study.best_value)
    print("[tune] best params:")
    for key, value in study.best_params.items():
        print(f"  {key} = {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
