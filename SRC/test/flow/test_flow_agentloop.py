"""Flow tests for ``pipeline.agentloop`` -- REQUIREMENTS (agentloop v1) section 3.3, F1-F3.

Ten-minute-scale. Needs ``openfoam2512`` AND the ``claude`` CLI on PATH; skipped
if either is missing. Model from ``POC_LLM_MODEL`` (default
``claude-cli:claude-sonnet-5``). Artifacts go to ``OUTPUTs/<ts>-test-agentloop/``.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from pipeline.agentloop import PLAN_KEYS, WING_SCENARIO, run_line, summarize_sensitivity

pytestmark = pytest.mark.flow

MODEL = os.environ.get("POC_LLM_MODEL", "claude-cli:claude-sonnet-5")
BUDGET = 3
CASE_FILES = {"params.json", "result.json", "reading.md", "proposal.json"}
LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+\S")


@pytest.fixture(autouse=True)
def _need_openfoam_and_cli(has_openfoam, has_claude_cli):
    if not has_openfoam:
        pytest.skip("openfoam2512 not on PATH")
    if not has_claude_cli:
        pytest.skip("claude CLI not on PATH")


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


def _line_cfg(line: str, artifact_dir: Path) -> dict:
    return {
        "line": line,
        "scenario": WING_SCENARIO,
        "budget": BUDGET,
        "verify_every": 100,
        "exp_dir": artifact_dir / f"exp{line}",
        "store": artifact_dir / "store",
        "model": MODEL,
        "tau": 0.4,
        "k_backtrack": 4,
        "k_stagnation": 8,
        "seed": 0,
    }


def _check_llm_line(summary: dict, exp_dir: Path, line: str) -> dict[str, dict[str, str]]:
    assert isinstance(summary, dict)
    assert summary["line"] == line
    assert isinstance(summary["n_cases"], int)
    assert 1 <= summary["n_cases"] <= BUDGET
    assert isinstance(summary["n_untrusted_accepted"], int)
    assert summary["stop_reason"] in {"budget", "target", "stagnation"}

    tree = exp_dir / "tree"
    assert _git("rev-parse", "--is-inside-work-tree", cwd=tree) == "true"
    assert f"run/{line}/main" in _git("branch", "--list", f"run/{line}/main", cwd=tree)
    assert _git("tag", cwd=tree) == ""

    cases = _case_files(tree)
    assert len(cases) == summary["n_cases"], sorted(cases)
    n_valid = 0
    for case_id, files in cases.items():
        assert set(files) == CASE_FILES, (case_id, sorted(files))
        proposal = json.loads(files["proposal.json"])
        assert isinstance(proposal, dict)
        assert isinstance(proposal["valid"], bool)
        assert isinstance(proposal["missing"], list)
        if proposal["valid"]:
            assert set(PLAN_KEYS) <= set(proposal)
            assert proposal["intent"] in {"continue", "branch", "backtrack"}
            assert 0.0 <= float(proposal["confidence"]) <= 1.0
            float(proposal["prediction"])
            n_valid += 1
        else:
            assert set(proposal["missing"]) <= set(PLAN_KEYS)

    preds = summary["predictions"]
    assert isinstance(preds, list)
    assert len(preds) == n_valid
    for p in preds:
        assert {"step", "prediction", "outcome"} <= set(p)
        assert isinstance(p["step"], int)
        float(p["prediction"])
    return cases


# ---------------------------------------------------------------------------
# F1  happy path, line C
# ---------------------------------------------------------------------------


def test_f1_line_c_happy(artifact_dir_agentloop: Path):
    exp_dir = artifact_dir_agentloop / "expC"
    summary = run_line(_line_cfg("C", artifact_dir_agentloop))
    cases = _check_llm_line(summary, exp_dir, "C")
    for files in cases.values():
        result = json.loads(files["result.json"])
        assert result["status"] in {"ok", "failed"}
    assert list(exp_dir.rglob("summary.json"))
    assert list(exp_dir.rglob("holdout.json"))


# ---------------------------------------------------------------------------
# F2  line B: scalar only, field_summary is None
# ---------------------------------------------------------------------------


def test_f2_line_b_scalar_only(artifact_dir_agentloop: Path):
    exp_dir = artifact_dir_agentloop / "expB"
    summary = run_line(_line_cfg("B", artifact_dir_agentloop))
    cases = _check_llm_line(summary, exp_dir, "B")
    for case_id, files in cases.items():
        result = json.loads(files["result.json"])
        assert result["field_summary"] is None, case_id


# ---------------------------------------------------------------------------
# F3  summarize_sensitivity on synthetic sweep rows
# ---------------------------------------------------------------------------

SWEEP_ROWS = [
    {"h_c": 0.10, "alpha_deg": 2.0, "camber": 0.04, "Cl": -1.62, "Cd": 0.041, "trusted": True, "failure_type": None, "cp_peak_val": -4.1, "sep_x": 0.72},
    {"h_c": 0.15, "alpha_deg": 2.0, "camber": 0.04, "Cl": -1.48, "Cd": 0.037, "trusted": True, "failure_type": None, "cp_peak_val": -3.6, "sep_x": 0.78},
    {"h_c": 0.20, "alpha_deg": 2.0, "camber": 0.04, "Cl": -1.35, "Cd": 0.034, "trusted": True, "failure_type": None, "cp_peak_val": -3.1, "sep_x": 0.83},
    {"h_c": 0.20, "alpha_deg": 6.0, "camber": 0.04, "Cl": -1.71, "Cd": 0.058, "trusted": False, "failure_type": None, "cp_peak_val": -4.8, "sep_x": 0.55},
    {"h_c": 0.20, "alpha_deg": 2.0, "camber": 0.08, "Cl": -1.52, "Cd": 0.039, "trusted": True, "failure_type": None, "cp_peak_val": -3.4, "sep_x": 0.80},
    {"h_c": 0.05, "alpha_deg": 8.0, "camber": 0.08, "Cl": None, "Cd": None, "trusted": False, "failure_type": "mesh_invalid", "cp_peak_val": None, "sep_x": None},
]


def test_f3_summarize_sensitivity():
    md = summarize_sensitivity(SWEEP_ROWS, MODEL)
    assert isinstance(md, str)
    assert md.strip()
    items = [ln for ln in md.splitlines() if LIST_ITEM_RE.match(ln)]
    assert len(items) >= 5, md
    assert re.search(r"\d", md)
