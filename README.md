# Orbit Wars Beam Search

This repository contains a C++20 agent core for Kaggle Orbit Wars, exposed through a thin Python `main.py` entrypoint. The native module is named `orbit_engine`.

The engine is built around fixed-buffer continuous physics, geometric launch generation, and a heuristic-guided beam-search action selector. It does not use grid bitboards or hidden-state search.

## Current Architecture

- `src/orbit_engine.hpp` defines fixed capacities, observation buffers, SoA state, launch buffers, and the public `Engine` facade.
- `src/orbit_engine_state.cpp` maps Kaggle observations into `PlanetSoA`, `FleetSoA`, and comet path storage.
- `src/orbit_engine_sim.cpp` implements the turn-order simulator with swept fleet collisions and moving planet/comet catches.
- `src/geometry.hpp` and `src/orbit_engine_geometry.cpp` implement fleet speed, segment-circle hits, moving-body sweep helpers, and interception solving.
- `src/candidate.hpp` and `src/orbit_engine_candidate.cpp` generate solved `AtomicLaunch` candidates and pack legal macro-actions.
- `src/search.hpp` and `src/orbit_engine_search.cpp` evaluate macro-actions with root-parallel deterministic rollouts.
- `src/eval.hpp` and `src/orbit_engine_eval.cpp` implement the dense state evaluator.
- `src/orbit_engine_bindings.cpp` exposes the pybind11 bridge.

## Fixed Capacities

The hot path uses fixed-size `std::array` buffers:

```text
MAX_PLAYERS = 4
MAX_PLANETS = 96
MAX_FLEETS = 4096
MAX_COMET_GROUPS = 8
MAX_COMET_PATH_POINTS = 512
MAX_ATOMIC_LAUNCHES = 1536
MAX_MACRO_ACTIONS = 512
MAX_BEAM_WIDTH = 512
```

Search and simulator hot modules are tested to avoid `std::vector`, `std::set`, `new`, `make_unique`, `make_shared`, `std::function`, and `std::async`.

## Simulator

`OrbitSim::step()` follows the Orbit Wars order:

1. comet expiration
2. comet spawn phase as a no-op unless the comet is already present in observation data
3. fleet launch
4. production
5. fleet movement and swept collision checks
6. planet rotation and comet movement, including moving-body catches
7. combat resolution

Fleet movement checks the whole segment, not only endpoints. If a fleet intersects multiple hazards at the same earliest time, the simulator resolves ties as sun first, then lowest planet/comet id.

The simulator models active observed comet paths exactly. It does not fabricate future comet groups that are not present in the current observation.

## Search

The action generator solves interception geometry instead of sampling random angles.

For each owned source and target planet/comet, it builds tactical packets:

- `capture_exact`
- `capture_over`
- `harass`
- `all_safe`

Orbiting and comet targets use bounded root solving over `tau in [1, 120]`. Macro-actions are packed from atomic launches with per-source spend accounting, so a planet cannot overspend its garrison.

The root action selector evaluates packed macro-actions with deterministic opponent heuristics, a depth-8 search prefix, and 64-tick rollouts under a 900ms budget. It keeps a legal 1-ply fallback ready before deeper evaluation.

The evaluator is:

```text
ship_delta + 25 * production_delta + territory_centrality - incoming_threat + comet_value
```

## Python Entrypoint

Kaggle calls:

```python
from main import agent
```

The returned action format is:

```python
[[from_planet_id, angle_radians, ships], ...]
```

If a compatible native extension is not already importable, `main.py` JIT-compiles `orbit_engine` from `src/*.cpp` using local or vendored pybind11 headers.

## Development

Install development dependencies:

```bash
python -m pip install -r requirements-dev.txt
```

Build locally:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
```

Run tests:

```bash
PYTHONPATH=build python test.py
PYTHONPATH=build python -m pytest -q test.py
```

Run the Kaggle-style JIT smoke:

```bash
python main.py
```

Package a source submission:

```bash
python package_submission.py
```

## Assumptions

- Future comet spawns are introduced by future observations, not predicted internally.
- The observation’s `initial_planets` and `angular_velocity` are trusted for orbit prediction.
- Opponent rollouts use deterministic geometric heuristics rather than stochastic policy sampling.
- The engine keeps the `orbit_engine` module name throughout CMake, pybind11, Python imports, tests, and packaging.
