"""Smoke tests for ``pipeline.agentloop`` v2 -- REQUIREMENTS (agentloop v2) section 3, S3-S4.

Minute-scale. Needs ``openfoam2512`` on PATH (skipped otherwise); does not need
the ``claude`` CLI. Artifacts go to ``OUTPUTs/<ts>-test-agentloop/exp_v2/``.

S3: run_line line A, seed 1, run_tag "s1", budget 3, memory None.
S4: a second run (seed 0, run_tag "s0") into the same exp_dir, then
    ``python -m pipeline.agentloop.evaluation.finding_report <exp_dir>``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from pipeline.agentloop import WING_SCENARIO, perturb_x0, run_line

pytestmark = pytest.mark.smoke

V1_SUMMARY_KEYS = {
    "line",
    "n_cases",
    "n_unique_hash",
    "best_metric",
    "best_params",
    "best_case_id",
    "stop_reason",
    "n_failed",
    "n_untrusted_accepted",
    "holdout",
    "predictions",
}
V2_SUMMARY_KEYS = {"seed", "run_tag", "memory_used", "x0_used"}


@pytest.fixture(autouse=True)
def _need_openfoam(has_openfoam):
    if not has_openfoam:
        pytest.skip("openfoam2512 not on PATH")


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _close(a: dict, b: dict, keys, tol: float = 1e-9) -> bool:
    return all(abs(float(a[k]) - float(b[k])) < tol for k in keys)


def _first_case_params(tree_dir: Path, branch: str) -> dict:
    """params.json of the first case commit on ``branch`` (oldest first)."""
    for sha in _git("log", branch, "--reverse", "--format=%H", cwd=tree_dir).split():
        added = _git(
            "show", "--pretty=format:", "--name-only", "--diff-filter=A", sha, cwd=tree_dir
        ).splitlines()
        for path in added:
            parts = path.split("/")
            if len(parts) == 3 and parts[0] == "cases" and parts[2] == "params.json":
                return json.loads(_git("show", f"{sha}:{path}", cwd=tree_dir))
    raise AssertionError(f"no case commit with params.json on {branch}")


def _line_cfg(exp_dir: Path, store: Path, *, seed: int, run_tag: str) -> dict:
    return {
        "line": "A",
        "scenario": WING_SCENARIO,
        "budget": 3,
        "verify_every": 100,
        "exp_dir": exp_dir,
        "store": store,
        "model": None,
        "tau": 0.4,
        "k_backtrack": 4,
        "k_stagnation": 8,
        "seed": seed,
        "run_tag": run_tag,
        "memory": None,
    }


@pytest.fixture(scope="module")
def exp_dir(artifact_dir_agentloop: Path) -> Path:
    return artifact_dir_agentloop / "exp_v2"


@pytest.fixture(scope="module")
def store(artifact_dir_agentloop: Path) -> Path:
    return artifact_dir_agentloop / "store"


@pytest.fixture(scope="module")
def summary_s1(exp_dir: Path, store: Path) -> dict:
    return run_line(_line_cfg(exp_dir, store, seed=1, run_tag="s1"))


@pytest.fixture(scope="module")
def summary_s0(summary_s1: dict, exp_dir: Path, store: Path) -> dict:
    # runs after S3's seed-1 run, into the same exp_dir
    return run_line(_line_cfg(exp_dir, store, seed=0, run_tag="s0"))


@pytest.fixture(scope="module")
def finding_dir(summary_s0: dict, exp_dir: Path) -> Path:
    proc = subprocess.run(
        [sys.executable, "-m", "pipeline.agentloop.evaluation.finding_report", str(exp_dir)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"finding_report failed:\n{proc.stdout}\n{proc.stderr}"
    hits = sorted(exp_dir.rglob("finding.json"))
    assert hits, f"finding.json not written under {exp_dir}"
    return hits[0].parent


# ---------------------------------------------------------------------------
# S3  run_line line A, seed 1, run_tag "s1", budget 3, memory None
# ---------------------------------------------------------------------------


def test_s3_summary_fields(summary_s1):
    s = summary_s1
    assert isinstance(s, dict)
    assert V1_SUMMARY_KEYS <= set(s)
    assert V2_SUMMARY_KEYS <= set(s)
    assert s["line"] == "A"
    assert s["seed"] == 1
    assert s["run_tag"] == "s1"
    assert s["memory_used"] is False
    assert s["n_cases"] == 3
    assert s["stop_reason"] == "budget"
    assert s["predictions"] == []


def test_s3_x0_used_is_perturbed_seed1(summary_s1):
    expected = perturb_x0(WING_SCENARIO, 1)
    x0_used = summary_s1["x0_used"]
    assert isinstance(x0_used, dict)
    assert set(WING_SCENARIO.param_keys) <= set(x0_used)
    assert _close(x0_used, expected, WING_SCENARIO.param_keys)
    # seed 1 really moved away from the plain x0
    assert not _close(x0_used, WING_SCENARIO.x0, WING_SCENARIO.param_keys)


def test_s3_tagged_dir_and_branch(summary_s1, exp_dir: Path):
    line_dir = exp_dir / "lines" / "A-s1"
    assert line_dir.is_dir()
    assert not (exp_dir / "lines" / "A").exists()
    tree = exp_dir / "tree"
    assert tree.is_dir()
    branches = _git("branch", "--list", "run/A-s1/main", cwd=tree)
    assert "run/A-s1/main" in branches
    assert "run/A/main" not in _git("branch", "--list", "run/A/main", cwd=tree)
    # summary.json / holdout.json live under the tagged dir
    assert (line_dir / "summary.json").is_file()
    assert (line_dir / "holdout.json").is_file()
    on_disk = json.loads((line_dir / "summary.json").read_text())
    assert on_disk == json.loads(json.dumps(summary_s1))


def test_s3_first_case_params_equal_x0_used(summary_s1, exp_dir: Path):
    tree = exp_dir / "tree"
    params = _first_case_params(tree, "run/A-s1/main")
    assert set(WING_SCENARIO.param_keys) <= set(params)
    assert _close(params, summary_s1["x0_used"], WING_SCENARIO.param_keys)


# ---------------------------------------------------------------------------
# S4  second run (seed 0, run_tag "s0") + finding_report over the exp_dir
# ---------------------------------------------------------------------------


def test_s4_second_run_seed0(summary_s0, exp_dir: Path):
    s = summary_s0
    assert s["seed"] == 0
    assert s["run_tag"] == "s0"
    assert s["memory_used"] is False
    assert s["n_cases"] == 3
    # seed 0 -> x0 itself (v1-compatible first step)
    assert _close(s["x0_used"], WING_SCENARIO.x0, WING_SCENARIO.param_keys)
    assert (exp_dir / "lines" / "A-s0").is_dir()
    assert (exp_dir / "lines" / "A-s1").is_dir()
    tree = exp_dir / "tree"
    assert "run/A-s0/main" in _git("branch", "--list", "run/A-s0/main", cwd=tree)
    assert _close(
        _first_case_params(tree, "run/A-s0/main"), WING_SCENARIO.x0, WING_SCENARIO.param_keys
    )


def test_s4_finding_json_aggregate(finding_dir: Path):
    finding = json.loads((finding_dir / "finding.json").read_text())
    assert "aggregate" in finding
    assert "walk_best" in finding  # null allowed: no walk sweep in this exp_dir
    agg = finding["aggregate"]
    assert isinstance(agg, dict)
    assert "A|nomem" in agg, sorted(agg)
    group = agg["A|nomem"]
    assert group["n_runs"] == 2
    for k in (
        "best_metric_mean",
        "best_metric_std",
        "best_metric_min",
        "best_metric_max",
        "holdout_robust_frac",
        "n_failed_mean",
        "n_cases_mean",
        "stop_reasons",
    ):
        assert k in group, k
    assert abs(group["n_cases_mean"] - 3.0) < 1e-9
    assert group["stop_reasons"] == {"budget": 2}
    assert "A|mem" not in agg


def test_s4_finding_md_has_aggregate_table(finding_dir: Path):
    md_path = finding_dir / "finding.md"
    assert md_path.is_file()
    md = md_path.read_text()
    low = md.lower()
    assert ("aggregate" in low) or ("n_runs" in low) or ("聚合" in md)
    # per-run rows name both tagged runs
    assert "s0" in md and "s1" in md
    # a markdown table is present
    assert "|" in md
