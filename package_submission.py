"""Create a Kaggle submission bundle with engine sources and pybind11 headers.

The produced archive contains ``main.py``, all native C++ source/header files,
and vendored pybind11 headers so Kaggle can JIT-compile the extension without a
separate package installation step.
"""

from __future__ import annotations

from pathlib import Path
import tarfile

# Repository root used as the packaging base directory.
ROOT = Path(__file__).resolve().parent

# Default Kaggle submission archive path.
OUT = ROOT / "submission.tar.gz"


def _pybind11_include_dir() -> Path:
    """Find installed pybind11 headers to vendor into the source bundle.

    Returns:
        Path: Include directory containing ``pybind11/pybind11.h``.

    Raises:
        RuntimeError: If pybind11 is missing or does not expose headers.
    """

    try:
        import pybind11
    except Exception as exc:  # pragma: no cover - packaging requires dev deps.
        raise RuntimeError(
            "pybind11 is required to vendor headers into the submission"
        ) from exc

    include = Path(pybind11.get_include())
    if not (include / "pybind11" / "pybind11.h").exists():
        raise RuntimeError(f"pybind11 headers not found under {include}")
    return include


def _add_file(tar: tarfile.TarFile, path: Path, arcname: Path) -> None:
    """Add one file with a stable archive name and no recursive surprises.

    Args:
        tar: Open tarfile handle.
        path: Source file on disk.
        arcname: Archive-relative destination path.
    """

    tar.add(path, arcname=str(arcname), recursive=False)


def build_package() -> Path:
    """Create ``submission.tar.gz`` with main.py, engine sources, and pybind11.

    Returns:
        Path: Path to the written archive.
    """

    pybind_include = _pybind11_include_dir()
    with tarfile.open(OUT, "w:gz") as tar:
        _add_file(tar, ROOT / "main.py", Path("main.py"))

        for path in sorted((ROOT / "src").glob("*.cpp")) + sorted(
            (ROOT / "src").glob("*.hpp")
        ):
            _add_file(tar, path, Path("src") / path.name)

        for path in sorted(pybind_include.rglob("*")):
            if path.is_file():
                _add_file(
                    tar,
                    path,
                    Path("vendor")
                    / "pybind11"
                    / "include"
                    / path.relative_to(pybind_include),
                )

    return OUT


if __name__ == "__main__":
    package = build_package()
    print(f"wrote {package}")
