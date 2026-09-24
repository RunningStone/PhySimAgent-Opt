"""U11/U14: official TopK behavior independent of implementation details."""
from __future__ import annotations

import copy
import importlib
import json
import os
import subprocess
import sys

import pytest


@pytest.fixture
def select_topk():
    return importlib.import_module("pipeline.agentloop.harness").select_topk


def assessment(design, score, **changes):
    result = {"candidate_id": "candidate-" + design, "design_hash": design,
              "stage_id": "M0", "protocol_hash": "protocol-1", "replicate_id": 0,
              "objective_value": score, "completeness": "complete", "numerical_status": "pass",
              "feasibility": "feasible", "rank_eligible": True, "role": "formal",
              "evidence_refs": ["obs-" + design]}
    result.update(changes)
    return result


def test_topk_prefers_larger_target_score_and_deduplicates_design(select_topk):
    rows = [assessment("a", -0.3), assessment("b", -0.1), assessment("a", -0.2),
            assessment("c", -0.4)]
    result = select_topk(rows, 3)
    assert [row["design_hash"] for row in result] == ["b", "a", "c"]
    assert len({row["design_hash"] for row in result}) == 3


def test_topk_never_uses_exploration_failed_or_infeasible_items(select_topk):
    rows = [assessment("valid", -0.5),
            assessment("explore", 100, role="exploration", rank_eligible=False),
            assessment("infeasible", 100, feasibility="infeasible", rank_eligible=False),
            assessment("failed", 100, numerical_status="fail", rank_eligible=False),
            assessment("unknown", None, completeness="incomplete", rank_eligible=False)]
    assert [row["design_hash"] for row in select_topk(rows, 5)] == ["valid"]


@pytest.mark.parametrize("field,value", [("stage_id", "M1"), ("protocol_hash", "other")])
def test_mixed_stage_or_protocol_cannot_be_ranked_together(select_topk, field, value):
    rows = [assessment("a", -0.1), assessment("b", -0.2, **{field: value})]
    with pytest.raises(ValueError):
        select_topk(rows, 2)


def test_no_eligible_designs_has_empty_topk(select_topk):
    assert select_topk([], 3) == []
    assert select_topk([assessment("a", -0.1, rank_eligible=False)], 3) == []
    assert select_topk([assessment("a", -0.1)], 0) == []


def test_ties_are_stable_across_input_order_and_input_not_mutated(select_topk):
    rows = [assessment("z", -0.1), assessment("a", -0.1), assessment("m", -0.1)]
    before = copy.deepcopy(rows)
    first = select_topk(rows, 2)
    assert [row["design_hash"] for row in first] == [row["design_hash"] for row in select_topk(rows[::-1], 2)]
    assert rows == before


def test_negative_k_is_invalid(select_topk):
    with pytest.raises(ValueError):
        select_topk([assessment("a", -0.1)], -1)


def test_isolated_live_backend_invokes_cli_without_tools_sessions_or_project_memory(tmp_path, monkeypatch):
    """Observe the executable boundary with a stand-in CLI, not private helpers."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    record_path = tmp_path / "cli-invocation.json"
    executable = bin_dir / "claude"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json,os,sys\n"
        f"with open({str(record_path)!r}, 'w') as stream:\n"
        "    json.dump({'argv':sys.argv[1:],'cwd':os.getcwd()},stream)\n"
        "print(json.dumps({'result':'{\"action\":\"stop\",\"stop_reason\":\"test\"}',"
        "'is_error':False,'usage':{'input_tokens':3,'output_tokens':2}}))\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ.get("PATH", ""))
    cwd = tmp_path / "isolated-agent-view"
    cwd.mkdir()
    query = importlib.import_module("aide.backend.backend_claude_cli").query
    response = query(system_message="Static task instructions", user_message='{"observations":[]}',
                     model="claude-sonnet-4-6", isolated=True, cwd=cwd, timeout=10)
    invocation = json.loads(record_path.read_text())
    argv = invocation["argv"]
    assert {"--safe-mode", "--no-session-persistence", "--strict-mcp-config", "--tools"} <= set(argv)
    assert argv[argv.index("--tools") + 1] == ""
    assert not ({"--resume", "--continue", "-r", "-c"} & set(argv))
    assert invocation["cwd"] == str(cwd)
    assert response[2:4] == (3, 2), "available actual token usage must not be reported unavailable"


def test_live_backend_rejects_cli_error_envelope_even_with_exit_zero(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    executable = bin_dir / "claude"
    executable.write_text(
        f"#!{sys.executable}\nimport json\n"
        "print(json.dumps({'is_error':True,'result':'model quota failure',"
        "'usage':{'input_tokens':3,'output_tokens':0}}))\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ.get("PATH", ""))
    query = importlib.import_module("aide.backend.backend_claude_cli").query
    with pytest.raises((RuntimeError, ValueError)):
        query(system_message="Static task instructions", user_message="{}", model="claude-sonnet-4-6",
              isolated=True, cwd=tmp_path, timeout=5)


def test_live_backend_public_timeout_limits_actual_subprocess(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    executable = bin_dir / "claude"
    executable.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(10)\n")
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ.get("PATH", ""))
    query = importlib.import_module("aide.backend.backend_claude_cli").query
    import time
    started = time.monotonic()
    with pytest.raises((RuntimeError, TimeoutError, subprocess.TimeoutExpired)):
        query(system_message="Static task instructions", user_message="{}", model="claude-sonnet-4-6",
              isolated=True, cwd=tmp_path, timeout=0.25)
    assert time.monotonic() - started < 3
