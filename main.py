"""Kaggle entrypoint for the Orbit Wars C++ beam-search engine."""

from __future__ import annotations

import importlib
import math
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import traceback

try:
    import orbit_engine

    _ENGINE_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    orbit_engine = None
    _ENGINE_IMPORT_ERROR = exc


_ENGINES = {}
_JIT_ATTEMPTED = False
try:
    _ROOT = Path(__file__).resolve().parent
except NameError:
    _ROOT = Path(os.getcwd()).resolve()


def _jit_log(message):
    print(f"[orbit_engine jit] {message}", file=sys.stderr, flush=True)


def _pybind11_include_dir():
    candidates = [_ROOT / "vendor" / "pybind11" / "include"]
    try:
        import pybind11

        candidates.append(Path(pybind11.get_include()))
    except Exception:
        pass
    for path in candidates:
        if (path / "pybind11" / "pybind11.h").exists():
            return path
    return None


def _python_include_dirs():
    paths = sysconfig.get_paths()
    include_dirs = []
    for key in ("include", "platinclude"):
        value = paths.get(key)
        if value and Path(value).exists() and value not in include_dirs:
            include_dirs.append(value)
    return include_dirs


def _compile_native_engine():
    sources = sorted((_ROOT / "src").glob("*.cpp"))
    if not sources:
        _jit_log("no C++ sources found under src/")
        return False

    pybind_include = _pybind11_include_dir()
    if pybind_include is None:
        _jit_log("pybind11 headers not found")
        return False

    python_includes = _python_include_dirs()
    if not python_includes:
        _jit_log("Python development headers not found")
        return False

    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    output = _ROOT / f"orbit_engine{ext_suffix}"
    compiler = os.environ.get("CXX", "g++")
    command = [
        compiler,
        "-std=c++20",
        "-O3",
        "-DNDEBUG",
        "-fPIC",
        "-shared",
        "-ffast-math",
        "-march=native",
        "-pthread",
        "-Isrc",
        f"-I{pybind_include}",
    ]
    command.extend(f"-I{path}" for path in python_includes)
    command.extend(str(path.relative_to(_ROOT)) for path in sources)
    command.extend(["-o", str(output)])

    _jit_log("compiling native engine")
    try:
        result = subprocess.run(
            command, cwd=_ROOT, capture_output=True, text=True, timeout=180
        )
    except Exception:
        _jit_log("compiler invocation failed")
        _jit_log(traceback.format_exc())
        return False

    if result.stdout:
        _jit_log("compiler stdout:\n" + result.stdout)
    if result.stderr:
        _jit_log("compiler stderr:\n" + result.stderr)
    if result.returncode != 0:
        _jit_log(f"compiler exited with status {result.returncode}")
        return False
    _jit_log(f"native engine built at {output.name}")
    return True


def _ensure_native_engine():
    global orbit_engine, _JIT_ATTEMPTED
    if orbit_engine is not None:
        return True
    if _JIT_ATTEMPTED:
        return False
    _JIT_ATTEMPTED = True
    if _ENGINE_IMPORT_ERROR is not None:
        _jit_log(f"initial import failed: {_ENGINE_IMPORT_ERROR}")
    if not _compile_native_engine():
        return False
    try:
        importlib.invalidate_caches()
        if str(_ROOT) not in sys.path:
            sys.path.insert(0, str(_ROOT))
        orbit_engine = importlib.import_module("orbit_engine")
        _jit_log("native engine import succeeded after JIT compile")
        return True
    except Exception:
        _jit_log("native engine import failed after JIT compile")
        _jit_log(traceback.format_exc())
        orbit_engine = None
        return False


def _get(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _fallback_agent(obs):
    player = int(_get(obs, "player", 0))
    planets = _get(obs, "planets", []) or []
    moves = []
    my_planets = [p for p in planets if int(p[1]) == player]
    targets = [p for p in planets if int(p[1]) != player]
    for mine in my_planets:
        best = None
        best_dist = 1.0e100
        for target in targets:
            dist = math.hypot(float(target[2]) - float(mine[2]), float(target[3]) - float(mine[3]))
            if dist < best_dist:
                best_dist = dist
                best = target
        if best is None:
            continue
        ships = int(best[5]) + 1
        if int(mine[5]) >= ships:
            moves.append([
                int(mine[0]),
                math.atan2(float(best[3]) - float(mine[3]), float(best[2]) - float(mine[2])),
                ships,
            ])
    return moves


def agent(obs, config=None):
    _ensure_native_engine()
    if orbit_engine is None:
        return _fallback_agent(obs)

    player = int(_get(obs, "player", 0))
    engine = _ENGINES.get(player)
    if engine is None:
        engine = orbit_engine.Engine(player)
        _ENGINES[player] = engine
    step = int(_get(obs, "step", 0))
    engine.update_observation(obs)
    return engine.choose_actions(900, seed=(step + 1) * 1315423911 + player)


if __name__ == "__main__":
    ok = _ensure_native_engine()
    print(f"orbit_engine native available: {ok}")
