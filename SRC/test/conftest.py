"""Shared pytest configuration for SRC/test.

- Registers the ``smoke`` and ``flow`` markers (there is no pytest.ini).
- ``artifact_dir``: session-scoped ``<repo>/OUTPUTs/<YYYYMMDD_HHMMSS>-test-aero2d/``
  with a ``manifest.json``; only created when a test actually requests it, so
  unit tests never touch the disk.
- ``store``: ``artifact_dir / "store"`` (the simblock content-hash cache root).
- ``has_openfoam``: whether ``openfoam2512`` is on PATH.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Experiment-level fixtures live in SRC/test/exp as plain modules (not conftest files)
# so that unit/, smoke/ and flow/ can share them.
pytest_plugins = ("test.exp.agentloop_fixtures",)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "smoke: minute-scale end-to-end tests; need openfoam2512 on PATH (skipped otherwise)",
    )
    config.addinivalue_line(
        "markers",
        "flow: ten-minute-scale cross-stage tests; need openfoam2512 on PATH (skipped otherwise)",
    )


def _git_head() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else "unknown"


@pytest.fixture(scope="session")
def artifact_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = REPO_ROOT / "OUTPUTs"  / f"{stamp}-test-aero2d"
    out.mkdir(parents=True, exist_ok=False)
    manifest = {
        "kind": "test",
        "name": "simblock",
        "date": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_head": _git_head(),
        "uname": platform.uname()._asdict(),
        "python": sys.version,
        "repo_root": str(REPO_ROOT),
        "openfoam2512": shutil.which("openfoam2512"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return out


@pytest.fixture(scope="session")
def store(artifact_dir: Path) -> Path:
    return artifact_dir / "store"


@pytest.fixture(scope="session")
def has_openfoam() -> bool:
    return shutil.which("openfoam2512") is not None


# ---------------------------------------------------------------------------
# agentloop additions (appended by the agentloop test sub-agent; the block
# above is owned by the simblock tests and must not be edited).
#
# - ``artifact_dir_agentloop``: session-scoped
#   ``<repo>/OUTPUTs/<YYYYMMDD_HHMMSS>-test-agentloop/`` with a ``manifest.json``;
#   only created when a test actually requests it.
# - ``has_claude_cli``: whether the ``claude`` CLI is on PATH.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def artifact_dir_agentloop() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = REPO_ROOT / "OUTPUTs"  / f"{stamp}-test-agentloop"
    out.mkdir(parents=True, exist_ok=False)
    manifest = {
        "kind": "test",
        "name": "agentloop",
        "date": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_head": _git_head(),
        "uname": platform.uname()._asdict(),
        "python": sys.version,
        "repo_root": str(REPO_ROOT),
        "openfoam2512": shutil.which("openfoam2512"),
        "claude_cli": shutil.which("claude"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return out


@pytest.fixture(scope="session")
def has_claude_cli() -> bool:
    return shutil.which("claude") is not None
