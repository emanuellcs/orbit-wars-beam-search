# Tuning System

This document describes the tuning pipeline for the Orbit Wars engine: the hyperparameter schema, the Optuna runner, the evaluation protocol, the opponent league, champion promotion, and the 2-player and 4-player split.

## Overview

Orbit Wars is scored separately for 2-player and 4-player games, and the two formats reward different play. Tuning therefore runs a dedicated study per format, and `main.py` routes the champion by the detected player count. The system otherwise follows the same design as the tuning package described in this document:

- A single source of truth for tuned values and search-space bounds.
- A reliable evaluation signal based on fixed seeds and full seat rotation.
- A promotion step that writes the best configuration back to the config file and records it as a self-play checkpoint.

The harness is `scripts/tune.py`. It uses the reusable package under `tuning/`.

## Single Source of Truth

`config/hyperparameters.json` holds three blocks:

- `champion`: the current best 2-player values.
- `champion_4p`: the current best 4-player values.
- `schema`: one entry per tunable parameter with its type, bounds, step, and default.

The `tuning/schema.py` module maps these definitions onto Optuna `Trial` suggestions. `main.py` loads both champions and selects one after detecting the player count on the first observation.

### Parameter Groups

| Group | Parameters |
| --- | --- |
| search | `beam_width`, `search_depth`, `rollout_horizon`, `hard_stop_ms` |
| eval | `ship`, `production`, `territory_own`, `territory_opp`, `threat`, `comet_owned`, `comet_enemy`, `comet_neutral`, `timeline` |
| candidate | `owner_enemy`, `owner_neutral`, `owner_self`, `comet_bonus`, `prod_per_unit`, `kind_exact`, `kind_over`, `kind_all_safe`, `kind_harass`, `kind_reinforce`, `eta_discount`, `ship_cost` |

Numeric values are not duplicated in documentation. See `config/hyperparameters.json` for the authoritative champion values and bounds.

## Optuna Runner

`tuning/optuna_runner.py` implements the ask-and-tell loop:

- A multivariate TPE sampler explores the search space.
- Trials run in spawned worker processes so the native engine state stays clean.
- The incumbent champion is enqueued as the first trial when the study is new.
- Studies persist to SQLite and resume when rerun with the same `--storage` and `--study-name`.
- A wall-clock deadline bounds the whole study.
- Worker crashes, timeouts, and `BrokenProcessPool` are handled without killing the study.
- Trials that fail or return a sentinel failure score are marked `FAIL`.

## Evaluation Protocol

Each trial scores one hyperparameter set in a worker process. The protocol controls variance:

- A fixed seed set is shared across all trials.
- Every match rotates the candidate through every seat (both seats in 2-player, all four in 4-player).
- The opponent pool is fixed for the study: every hand-crafted archetype plus the self-play checkpoints in `config/league.json`.
- The per-turn search budget is a runtime knob set by `--budget-ms`, applied through `main.set_hyperparameters(hard_stop_ms=...)`. It is not part of the Optuna search space.

For 2-player games the trial score is the head-to-head win rate. For 4-player games the trial score is the first-place rate, defined as the share of matches in which the candidate owns the top reward.

## Opponent League

`config/league.json` records named opponents with an Elo rating. Two kinds exist:

- `handcrafted`: deterministic Python policies in `opponents/`.
- `checkpoint`: saved engine hyperparameter sets, which make self-play opponents.

Promoting a champion registers it as a checkpoint. The league is never pruned of hand-crafted archetypes, which keeps the pool diverse and prevents mode collapse.

## Champion Promotion

`scripts/tune.py --promote` runs after a successful study and:

1. Writes the best parameters into `champion` (2-player) or `champion_4p` (4-player) in `config/hyperparameters.json`.
2. Registers the champion as a checkpoint in `config/league.json`.
3. Records a promotion note and timestamp.

`main.py` detects the player count on the first observation of a game and applies the matching champion.

## Budget Knob

`main.set_hyperparameters(hard_stop_ms=N)` caps the per-turn native search time. Tuning uses a reduced budget for cheap exploration and the full budget for final studies. The Kaggle submission keeps the full budget.

## CLI Reference

```text
usage: scripts/tune.py [options]

  --players {2,4}        Format to tune (default 2)
  --trials N             Total Optuna trials (default 200)
  --n-jobs N             Parallel worker processes (default 8)
  --seeds N              Seed count per trial (default 3)
  --time-budget S        Study wall-clock seconds (default 3600)
  --storage URL          Optuna storage URL (default sqlite:///tune.db)
  --study-name NAME      Stable study identifier
  --base-seed N          Seed set base offset (default 42)
  --timeout S            Per-match wall-clock cap (default 60)
  --max-steps N          Episode step cap for matches (default 499)
  --budget-ms N          Per-turn native search budget (default 300)
  --promote              Write the champion back on success
```

## Worked Examples

Smoke study:

```bash
PYTHONPATH=build python scripts/tune.py \
  --players 2 --trials 4 --n-jobs 2 --seeds 1 --max-steps 40 \
  --time-budget 300 --budget-ms 50 --timeout 30 \
  --storage sqlite:////tmp/orbit-smoke.db --study-name smoke
```

Production 2-player study:

```bash
PYTHONPATH=build python scripts/tune.py \
  --players 2 --trials 1000 --n-jobs 16 --seeds 5 \
  --time-budget 43200 --budget-ms 900 --timeout 90 \
  --storage sqlite:///tune.db --study-name orbit-2p-vs-league --promote
```

Production 4-player study:

```bash
PYTHONPATH=build python scripts/tune.py \
  --players 4 --trials 1000 --n-jobs 16 --seeds 5 \
  --time-budget 43200 --budget-ms 900 --timeout 90 \
  --storage sqlite:///tune.db --study-name orbit-4p-vs-league --promote
```

Use `make tune` to run the defaults.
