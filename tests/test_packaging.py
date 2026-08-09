"""Packaging and hot-path allocation-guard tests."""

from __future__ import annotations

import os
import subprocess
import sys
import tarfile
import tempfile

import pytest

import package_submission
from _fixtures import REPO_ROOT


def test_hot_sources_avoid_dynamic_allocation_primitives():
    """Guard search/simulator hot files against allocation-heavy primitives."""

    hot_files = [
        "src/orbit_engine_sim.cpp",
        "src/orbit_engine_candidate.cpp",
        "src/orbit_engine_search.cpp",
        "src/orbit_engine_eval.cpp",
        "src/orbit_engine_geometry.cpp",
    ]
    forbidden = ("std::vector", "std::set", "make_unique", "make_shared", "std::function", "std::async")
    for rel in hot_files:
        with open(os.path.join(REPO_ROOT, rel), encoding="utf-8") as handle:
            source = handle.read()
        for token in forbidden:
            assert token not in source


def test_packaged_submission_jit_compiles_in_extracted_directory():
    """Build the Kaggle source package and smoke-test in-place JIT compilation."""

    pytest.importorskip("pybind11")
    package_path = package_submission.build_package()
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(package_path, "r:gz") as tar:
            try:
                tar.extractall(tmp, filter="data")
            except TypeError:
                tar.extractall(tmp)
        smoke = subprocess.run(
            [sys.executable, "main.py"],
            cwd=tmp,
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert smoke.returncode == 0, smoke.stdout + smoke.stderr
        assert "orbit_engine native available: True" in smoke.stdout

        call = subprocess.run(
            [
                sys.executable,
                "-c",
                "from types import SimpleNamespace\n"
                "from main import agent\n"
                "obs = SimpleNamespace(player=0, step=0, angular_velocity=0.0, "
                "planets=[[0,0,10.0,10.0,2.0,30,2],[1,-1,25.0,10.0,2.0,5,3]], "
                "fleets=[], initial_planets=[[0,0,10.0,10.0,2.0,30,2],[1,-1,25.0,10.0,2.0,5,3]], "
                "comets=[], comet_planet_ids=[])\n"
                "actions = agent(obs)\n"
                "assert isinstance(actions, list)\n"
                "print(actions)\n",
            ],
            cwd=tmp,
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert call.returncode == 0, call.stdout + call.stderr
