"""Smoke tests for ``pipeline.agentloop`` -- REQUIREMENTS (agentloop v1) section 3.2, S1-S2.

Minute-scale. Needs ``openfoam2512`` on PATH (skipped otherwise); does not need
the ``claude`` CLI. Artifacts go to ``OUTPUTs/<ts>-test-agentloop/``.
"""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
from pathlib import Path

import pytest

from pipeline.agentloop import WING_SCENARIO, make_exec_code, objective, parse_term_out, run_line

pytestmark = pytest.mark.smoke

DESIGN_CODE = 'PARAMS = {"h_c": 0.2, "alpha_deg": 3, "camber": 0.04}'
CASE_FILES = {"params.json", "result.json", "reading.md", "proposal.json"}
SUMMARY_KEYS = {
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


@pytest.fixture(autouse=True)
def _need_openfoam(has_openfoam):
    if not has_openfoam:
        pytest.skip("openfoam2512 not on PATH")


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _case_files(tree_dir: Path) -> dict[str, dict[str, str]]:
    """{case_id: {filename: content}} for every ``cases/<id>/<file>`` on any branch."""
    out: dict[str, dict[str, str]] = {}
    for sha in _git("log", "--all", "--format=%H", cwd=tree_dir).split():
        for path in _git("ls-tree", "-r", "--name-only", sha, cwd=tree_dir).splitlines():
            parts = path.split("/")
            if len(parts) == 3 and parts[0] == "cases":
                out.setdefault(parts[1], {}).setdefault(
                    parts[2], _git("show", f"{sha}:{path}", cwd=tree_dir)
                )
    return out


@pytest.fixture(scope="module")
def line_a_summary(artifact_dir_agentloop: Path) -> dict:
    return run_line(
        {
            "line": "A",
            "scenario": WING_SCENARIO,
            "budget": 3,
            "verify_every": 100,
            "exp_dir": artifact_dir_agentloop / "expA",
            "store": artifact_dir_agentloop / "store",
            "model": None,
            "tau": 0.4,
            "k_backtrack": 4,
            "k_stagnation": 8,
            "seed": 0,
        }
    )


# ---------------------------------------------------------------------------
# S1  run_line line A, budget 3
# ---------------------------------------------------------------------------


def test_s1_summary(line_a_summary):
    s = line_a_summary
    assert isinstance(s, dict)
    assert SUMMARY_KEYS <= set(s)
    assert s["line"] == "A"
    assert s["n_cases"] == 3
    assert s["stop_reason"] == "budget"
    assert s["predictions"] == []
    assert isinstance(s["n_unique_hash"], int) and 1 <= s["n_unique_hash"] <= 3
    assert isinstance(s["n_failed"], int) and 0 <= s["n_failed"] <= 3
    assert isinstance(s["n_untrusted_accepted"], int)
    if s["best_metric"] is not None:
        assert isinstance(s["best_params"], dict)
        assert set(WING_SCENARIO.param_keys) <= set(s["best_params"])
        assert isinstance(s["best_case_id"], str) and s["best_case_id"]


def test_s1_tree_has_three_case_commits(line_a_summary, artifact_dir_agentloop: Path):
    exp_dir = artifact_dir_agentloop / "expA"
    tree = exp_dir / "tree"
    assert tree.is_dir()
    assert _git("rev-parse", "--is-inside-work-tree", cwd=tree) == "true"
    assert (exp_dir / "lines" / "A").is_dir()

    log = _git("log", "--all", "--oneline", cwd=tree).splitlines()
    assert len(log) >= 3, log

    branches = _git("branch", "--list", "run/A/main", cwd=tree)
    assert "run/A/main" in branches

    cases = _case_files(tree)
    assert len(cases) == 3, sorted(cases)
    for case_id, files in cases.items():
        assert set(files) == CASE_FILES, (case_id, sorted(files))
        params = json.loads(files["params.json"])
        assert set(WING_SCENARIO.param_keys) <= set(params)
        result = json.loads(files["result.json"])
        assert result["status"] in {"ok", "failed"}
        json.loads(files["proposal.json"])
        assert isinstance(files["reading.md"], str)

    # each case commit is a distinct commit adding cases/<id>/ (3 case commits)
    case_commits = 0
    for sha in _git("log", "--all", "--format=%H", cwd=tree).split():
        added = _git(
            "show", "--pretty=format:", "--name-only", "--diff-filter=A", sha, cwd=tree
        ).splitlines()
        if any(p.startswith("cases/") for p in added):
            case_commits += 1
    assert case_commits == 3

    assert _git("tag", cwd=tree) == ""


def test_s1_summary_and_holdout_files(line_a_summary, artifact_dir_agentloop: Path):
    exp_dir = artifact_dir_agentloop / "expA"
    summaries = list(exp_dir.rglob("summary.json"))
    holdouts = list(exp_dir.rglob("holdout.json"))
    assert summaries, "summary.json not written under exp_dir"
    assert holdouts, "holdout.json not written under exp_dir"
    on_disk = json.loads(summaries[0].read_text())
    assert on_disk == json.loads(json.dumps(line_a_summary))
    holdout = json.loads(holdouts[0].read_text())
    assert holdout == json.loads(json.dumps(line_a_summary["holdout"]))


# ---------------------------------------------------------------------------
# S2  exec(make_exec_code(...)) -> parse_term_out -> objective consistency
# ---------------------------------------------------------------------------


def test_s2_exec_template_line_c(artifact_dir_agentloop: Path):
    store = artifact_dir_agentloop / "store"
    script = make_exec_code(DESIGN_CODE, WING_SCENARIO, "C", store)
    compile(script, "<exec_line_C>", "exec")

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(script, "<exec_line_C>", "exec"), {"__name__": "__main__"})
    term_out = buf.getvalue()

    d = parse_term_out(term_out)
    assert isinstance(d, dict), term_out[-2000:]
    for k in ("metric", "is_bug", "summary", "result"):
        assert k in d
    assert isinstance(d["is_bug"], bool)
    assert isinstance(d["summary"], str) and d["summary"]
    result = d["result"]
    assert isinstance(result, dict)
    assert result["status"] in {"ok", "failed"}

    expected = objective(result, WING_SCENARIO)
    if expected is None:
        assert d["metric"] is None
        assert d["is_bug"] is True
    else:
        assert d["metric"] is not None
        assert abs(d["metric"] - expected) < 1e-9
        assert d["is_bug"] is False
    if result["status"] == "ok":
        assert isinstance(result["trust"], dict) and all(
            isinstance(v, bool) for v in result["trust"].values()
        )
