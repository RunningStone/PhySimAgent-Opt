"""Unit tests for ``pipeline.agentloop`` -- REQUIREMENTS (agentloop v1) section 3.1, U1-U11.

Second-scale. No OpenFOAM, no ``claude`` CLI, no LLM calls, nothing written to
disk except under ``tmp_path``. Only names listed in REQUIREMENTS section 2 are
imported from ``pipeline.agentloop``.
"""

from __future__ import annotations

import dataclasses
import json
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline.agentloop import (
    LINES,
    PLAN_KEYS,
    WING_SCENARIO,
    DesignAgent,
    Scenario,
    Tree,
    make_exec_code,
    nelder_mead_propose,
    objective,
    parse_plan,
    parse_term_out,
    promote_tier,
    run_line,
    select_parent,
    should_stop,
)

# ---------------------------------------------------------------------------
# shared fixtures / helpers
# ---------------------------------------------------------------------------

DESIGN_CODE = 'PARAMS = {"h_c": 0.2, "alpha_deg": 3, "camber": 0.04}'
PARAM_KEYS = ("h_c", "alpha_deg", "camber")
TRUST_KEYS = ("converged", "residual_ok", "force_settled", "yplus_ok", "mesh_ok")
EXPECTED_PLAN_KEYS = {
    "reading",
    "hypothesis",
    "change",
    "expected_direction",
    "kill_criterion",
    "prediction",
    "intent",
    "confidence",
}
GOOD_PLAN = {
    "reading": "Cl=-1.5 at h/c=0.2; suction peak near the leading edge",
    "hypothesis": "lower ride height strengthens ground effect",
    "change": {"h_c": 0.15},
    "expected_direction": "Cl more negative, Cd slightly up",
    "kill_criterion": "Cd > cd_max or trust.converged False",
    "prediction": 1.7,
    "intent": "continue",
    "confidence": 0.6,
}


def _fence(obj: dict) -> str:
    return "Some reasoning first.\n```json\n" + json.dumps(obj, indent=2) + "\n```\nTrailing prose."


def _rec(step, metric=None, *, parent=None, is_bug=False, confidence=None, intent=None):
    """One 'records' entry as defined in REQUIREMENTS section 2.1."""
    return {
        "step": step,
        "parent_step": parent,
        "metric": metric,
        "is_bug": is_bug,
        "confidence": confidence,
        "intent": intent,
    }


def _result(status="ok", Cl=-1.5, Cd=0.03, trust_false=(), field_summary=None):
    """A CaseResult JSON dict (simblock shape: forces={Cl,Cd}, trust=5 bools)."""
    base = {"case_id": "0123456789ab", "input_hash": "0123456789ab" * 5 + "cdef"}
    if status != "ok":
        return {
            **base,
            "status": "failed",
            "forces": None,
            "trust": {k: False for k in TRUST_KEYS},
            "field_summary": None,
            "failure": {"type": "mesh_invalid", "stage": "mesh", "evidence": "Failed 1 mesh checks"},
        }
    trust = {k: True for k in TRUST_KEYS}
    for k in trust_false:
        trust[k] = False
    return {
        **base,
        "status": "ok",
        "forces": {"Cl": Cl, "Cd": Cd},
        "trust": trust,
        "field_summary": field_summary,
        "failure": None,
    }


def _eval_line(metric, is_bug, summary, result) -> str:
    return "EVAL_JSON: " + json.dumps(
        {"metric": metric, "is_bug": is_bug, "summary": summary, "result": result}
    )


# ---------------------------------------------------------------------------
# section 2.1 constants (support for U1 / U4 / U11)
# ---------------------------------------------------------------------------


def test_constants_lines_and_plan_keys():
    assert LINES == ("A", "B", "C")
    assert set(PLAN_KEYS) == EXPECTED_PLAN_KEYS
    assert len(PLAN_KEYS) == 8


def test_wing_scenario_shape():
    assert isinstance(WING_SCENARIO, Scenario)
    assert dataclasses.is_dataclass(WING_SCENARIO)
    assert isinstance(WING_SCENARIO.name, str) and WING_SCENARIO.name
    assert isinstance(WING_SCENARIO.task_desc, str) and WING_SCENARIO.task_desc.strip()
    assert WING_SCENARIO.param_keys == PARAM_KEYS
    assert set(WING_SCENARIO.bounds) == set(PARAM_KEYS)
    assert set(WING_SCENARIO.x0) == set(PARAM_KEYS)
    for k, (lo, hi) in WING_SCENARIO.bounds.items():
        assert lo < hi
        assert lo <= WING_SCENARIO.x0[k] <= hi
    assert WING_SCENARIO.cd_max > 0
    assert WING_SCENARIO.holdout_delta == {"h_c": 0.01}
    assert WING_SCENARIO.tiers == (0,)
    assert callable(WING_SCENARIO.objective_fn)
    assert isinstance(WING_SCENARIO.recovery_menu, str) and WING_SCENARIO.recovery_menu.strip()
    for placeholder in ("{design_code}", "{scalar_only}", "{store}", "{cd_max}"):
        assert placeholder in WING_SCENARIO.eval_template
    assert "EVAL_JSON" in WING_SCENARIO.eval_template
    assert "pipeline.exp_layer.aero2d" in WING_SCENARIO.eval_template
    # frozen
    with pytest.raises(dataclasses.FrozenInstanceError):
        WING_SCENARIO.name = "x"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# U1  make_exec_code
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "line,want,forbid",
    [
        ("B", r"scalar_only\s*=\s*True\b", r"scalar_only\s*=\s*False\b"),
        ("C", r"scalar_only\s*=\s*False\b", r"scalar_only\s*=\s*True\b"),
    ],
)
def test_u1_make_exec_code_compiles_and_sets_scalar_only(tmp_path, line, want, forbid):
    script = make_exec_code(DESIGN_CODE, WING_SCENARIO, line, tmp_path)
    assert isinstance(script, str)
    compile(script, f"<exec_line_{line}>", "exec")
    assert "EVAL_JSON" in script
    assert re.search(want, script), f"line {line}: expected {want!r} in script"
    assert not re.search(forbid, script), f"line {line}: unexpected {forbid!r} in script"
    assert DESIGN_CODE in script
    assert str(tmp_path) in script
    assert "pipeline.exp_layer.aero2d" in script
    for placeholder in ("{design_code}", "{scalar_only}", "{store}", "{cd_max}"):
        assert placeholder not in script


@pytest.mark.parametrize(
    "bad",
    [
        "import os\n" + DESIGN_CODE,
        DESIGN_CODE + "\nimport subprocess",
        DESIGN_CODE + '\nsubprocess.run(["ls"])',
        DESIGN_CODE + '\nopen("holdout.json").read()',
        "x = os.environ\n" + DESIGN_CODE,
        DESIGN_CODE + "\nsys.exit(0)",
    ],
)
def test_u1_make_exec_code_rejects_non_parameter_code(tmp_path, bad):
    with pytest.raises(ValueError, match=re.escape("design code may only assign PARAMS/OPTIONS")):
        make_exec_code(bad, WING_SCENARIO, "C", tmp_path)


# ---------------------------------------------------------------------------
# U2  parse_term_out
# ---------------------------------------------------------------------------


def test_u2_parse_term_out_uses_last_eval_json_line():
    result = _result()
    stale = _eval_line(None, True, "stale line", {})
    final = _eval_line(1.5, False, "Cl=-1.5 Cd=0.03 all trusted", result)
    text = "\n".join(["gmsh: meshing...", stale, "simpleFoam: converged", final, ""])
    d = parse_term_out(text)
    assert isinstance(d, dict)
    assert d["metric"] == 1.5
    assert d["is_bug"] is False
    assert d["summary"] == "Cl=-1.5 Cd=0.03 all trusted"
    assert d["result"] == result


def test_u2_parse_term_out_without_line_is_none():
    assert parse_term_out("gmsh: meshing...\nsimpleFoam: done\n") is None
    assert parse_term_out("") is None


def test_u2_parse_term_out_bad_json_is_none():
    assert parse_term_out("foo\nEVAL_JSON: {not valid json\n") is None
    assert parse_term_out("EVAL_JSON:") is None


# ---------------------------------------------------------------------------
# U3  objective
# ---------------------------------------------------------------------------

CD_MAX_U3 = 0.05


def _wing_objective_005(r: dict) -> float:
    return -r["forces"]["Cl"] - 10.0 * max(0.0, r["forces"]["Cd"] - CD_MAX_U3)


@pytest.fixture
def scenario_u3() -> Scenario:
    return dataclasses.replace(WING_SCENARIO, cd_max=CD_MAX_U3, objective_fn=_wing_objective_005)


def test_u3_objective_trusted_ok(scenario_u3):
    v = objective(_result(Cl=-1.5, Cd=0.03), scenario_u3)
    assert v is not None
    assert abs(v - 1.5) < 1e-9


def test_u3_objective_drag_penalty(scenario_u3):
    v = objective(_result(Cl=-1.5, Cd=0.07), scenario_u3)
    assert v is not None
    assert abs(v - (1.5 - 0.2)) < 1e-9


@pytest.mark.parametrize("bad_key", TRUST_KEYS)
def test_u3_objective_untrusted_ok_is_none(scenario_u3, bad_key):
    assert objective(_result(Cl=-1.5, Cd=0.03, trust_false=(bad_key,)), scenario_u3) is None


def test_u3_objective_failed_is_none(scenario_u3):
    assert objective(_result(status="failed"), scenario_u3) is None


def test_u3_wing_objective_fn_formula():
    cd_max = WING_SCENARIO.cd_max
    ok = _result(Cl=-1.5, Cd=cd_max - 0.02)
    assert abs(WING_SCENARIO.objective_fn(ok) - 1.5) < 1e-9
    over = _result(Cl=-1.5, Cd=cd_max + 0.02)
    assert abs(WING_SCENARIO.objective_fn(over) - (1.5 - 0.2)) < 1e-9
    assert abs(objective(ok, WING_SCENARIO) - 1.5) < 1e-9


# ---------------------------------------------------------------------------
# U4  parse_plan
# ---------------------------------------------------------------------------


def test_u4_parse_plan_full_fenced():
    p = parse_plan(_fence(GOOD_PLAN))
    assert p["valid"] is True
    assert p["missing"] == []
    for k in EXPECTED_PLAN_KEYS:
        assert p[k] == GOOD_PLAN[k]


def test_u4_parse_plan_bare_json_object():
    p = parse_plan(json.dumps(GOOD_PLAN))
    assert p["valid"] is True
    assert p["missing"] == []


def test_u4_parse_plan_missing_kill_criterion():
    plan = {k: v for k, v in GOOD_PLAN.items() if k != "kill_criterion"}
    p = parse_plan(_fence(plan))
    assert p["valid"] is False
    assert p["missing"] == ["kill_criterion"]


def test_u4_parse_plan_bad_intent():
    p = parse_plan(_fence({**GOOD_PLAN, "intent": "jump"}))
    assert p["valid"] is False


@pytest.mark.parametrize("conf", [1.5, -0.1, "high"])
def test_u4_parse_plan_bad_confidence(conf):
    p = parse_plan(_fence({**GOOD_PLAN, "confidence": conf}))
    assert p["valid"] is False


def test_u4_parse_plan_non_numeric_prediction():
    p = parse_plan(_fence({**GOOD_PLAN, "prediction": "better"}))
    assert p["valid"] is False


@pytest.mark.parametrize("text", ["", "no json here at all", "```json\n{ broken\n```", "[1, 2, 3]"])
def test_u4_parse_plan_no_json(text):
    p = parse_plan(text)
    assert p["valid"] is False
    assert p["missing"] == sorted(PLAN_KEYS)


# ---------------------------------------------------------------------------
# U5  select_parent
# ---------------------------------------------------------------------------


def test_u5a_empty_is_draft():
    assert select_parent([], 0.4, 4) == (None, "draft")
    assert select_parent([]) == (None, "draft")


def test_u5b_last_bug_is_debug():
    recs = [_rec(0, 1.0), _rec(1, None, parent=0, is_bug=True)]
    assert select_parent(recs, 0.4, 4) == (1, "debug")
    all_bug = [_rec(0, None, is_bug=True), _rec(1, None, parent=0, is_bug=True)]
    assert select_parent(all_bug, 0.4, 4) == (1, "debug")


def test_u5c_monotone_improvement_is_improve_on_last():
    recs = [_rec(i, 1.0 + 0.1 * i, parent=(i - 1 if i else None), confidence=0.9) for i in range(5)]
    assert select_parent(recs, 0.4, 4) == (4, "improve")
    assert select_parent(recs) == (4, "improve")


def test_u5d_backtrack_to_global_best():
    recs = [
        _rec(0, 1.0, confidence=0.9),
        _rec(1, 0.5, parent=0, confidence=0.9),
        _rec(2, 0.6, parent=1, confidence=0.9),
        _rec(3, 0.7, parent=2, confidence=0.9),
    ]
    assert select_parent(recs, 0.4, 3) == (0, "backtrack")


def test_u5e_low_confidence_branches_from_best():
    recs = [
        _rec(0, 1.0, confidence=0.9),
        _rec(1, 1.1, parent=0, confidence=0.9),
        _rec(2, 1.2, parent=1, confidence=0.9),
        _rec(3, 1.3, parent=2, confidence=0.9),
        _rec(4, 1.25, parent=3, confidence=0.2),
    ]
    assert select_parent(recs, 0.4, 4) == (3, "branch")


def test_u5_backtrack_has_priority_over_branch():
    recs = [
        _rec(0, 1.0, confidence=0.9),
        _rec(1, 0.5, parent=0, confidence=0.9),
        _rec(2, 0.6, parent=1, confidence=0.9),
        _rec(3, 0.7, parent=2, confidence=0.2),  # also < tau
    ]
    assert select_parent(recs, 0.4, 3) == (0, "backtrack")


def test_u5_confidence_none_or_at_tau_does_not_branch():
    recs = [
        _rec(0, 1.0, confidence=0.9),
        _rec(1, 1.1, parent=0, confidence=0.9),
        _rec(2, 1.2, parent=1, confidence=0.9),
        _rec(3, 1.3, parent=2, confidence=0.9),
        _rec(4, 1.25, parent=3, confidence=None),
    ]
    assert select_parent(recs, 0.4, 4) == (4, "improve")
    recs[-1] = _rec(4, 1.25, parent=3, confidence=0.4)  # not strictly below tau
    assert select_parent(recs, 0.4, 4) == (4, "improve")


def test_u5_no_best_degrades_to_improve():
    recs = [
        _rec(0, None, is_bug=True),
        _rec(1, None, parent=0, is_bug=True),
        _rec(2, None, parent=1, is_bug=False, confidence=0.1),
    ]
    assert select_parent(recs, 0.4, 2) == (2, "improve")


# ---------------------------------------------------------------------------
# U6  should_stop
# ---------------------------------------------------------------------------


def test_u6_budget():
    recs = [_rec(i, 1.0 + i) for i in range(3)]
    assert should_stop(recs, 3) == "budget"
    assert should_stop(recs, 2) == "budget"
    assert should_stop(recs, 4) is None


def test_u6_target():
    recs = [_rec(0, 1.0), _rec(1, 1.5), _rec(2, 1.2)]
    assert should_stop(recs, 10, target=1.5) == "target"  # best >= target (equality)
    assert should_stop(recs, 10, target=1.4) == "target"
    assert should_stop(recs, 10, target=1.6) is None
    assert should_stop(recs, 10, target=None) is None


def test_u6_stagnation():
    recs = [_rec(0, 1.0), _rec(1, 0.9), _rec(2, 0.8), _rec(3, 0.7)]
    assert should_stop(recs, 10, None, 3) == "stagnation"
    assert should_stop(recs, 10, None, k_stagnation=3) == "stagnation"
    # fewer records than k_stagnation -> not stagnation
    assert should_stop(recs, 10, None, 8) is None
    # improvement inside the window -> not stagnation
    improving = [_rec(0, 1.0), _rec(1, 0.9), _rec(2, 1.1), _rec(3, 1.0)]
    assert should_stop(improving, 10, None, 3) is None


def test_u6_none():
    recs = [_rec(0, 1.0), _rec(1, 1.1)]
    assert should_stop(recs, 10, None, 8) is None
    assert should_stop([], 10) is None


def test_u6_priority_budget_then_target_then_stagnation():
    recs = [_rec(0, 1.0), _rec(1, 0.9), _rec(2, 0.8), _rec(3, 0.7)]
    # all three satisfied -> budget
    assert should_stop(recs, 4, target=0.5, k_stagnation=3) == "budget"
    # target + stagnation -> target
    assert should_stop(recs, 10, target=0.5, k_stagnation=3) == "target"
    # stagnation only
    assert should_stop(recs, 10, target=5.0, k_stagnation=3) == "stagnation"


# ---------------------------------------------------------------------------
# U7  Tree (real git in tmp_path)
# ---------------------------------------------------------------------------

SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def test_u7_tree_lifecycle(tmp_path):
    root = tmp_path / "tree"
    t = Tree.init(root)
    assert isinstance(t, Tree)
    assert _git("rev-parse", "--is-inside-work-tree", cwd=root) == "true"

    sha1 = t.commit_case("abc", {"params.json": "{}"}, "m")
    assert SHA_RE.match(sha1), sha1
    assert (root / "cases" / "abc" / "params.json").read_text() == "{}"
    assert t.head() == sha1

    t.branch("run/C/b1")
    sha2 = t.commit_case("def", {"params.json": "{}", "result.json": "{}"}, "m2")
    assert SHA_RE.match(sha2) and sha2 != sha1
    assert t.head() == sha2
    assert (root / "cases" / "def" / "result.json").exists()

    t.checkout("main")
    assert t.head() == sha1
    assert not (root / "cases" / "def").exists()

    graph = t.log_graph()
    assert isinstance(graph, str)
    assert "run/C/b1" in graph
    assert "main" in graph
    assert "m2" in graph
    assert len([ln for ln in graph.splitlines() if ln.strip()]) >= 2

    wt = t.worktree(tmp_path / "wt", "run/A/main")
    assert isinstance(wt, Tree)
    assert wt.head() == t.head() == sha1
    assert (tmp_path / "wt" / "cases" / "abc" / "params.json").exists()
    sha3 = wt.commit_case("ghi", {"params.json": "{}"}, "m3")
    assert SHA_RE.match(sha3) and sha3 not in (sha1, sha2)
    assert wt.head() == sha3
    assert _git("rev-parse", "run/A/main", cwd=root) == sha3
    assert t.head() == sha1  # source checkout untouched

    # no tags, ever
    assert _git("tag", cwd=root) == ""
    # fixed author for reproducibility
    assert _git("log", "-1", "--format=%an|%ae", sha1, cwd=root) == "poc|poc@local"
    assert _git("log", "-1", "--format=%cn|%ce", sha1, cwd=root) == "poc|poc@local"
    # all three commits visible from the shared repo
    assert len(_git("log", "--all", "--oneline", cwd=root).splitlines()) == 3

    # init on an existing repo opens it instead of failing
    t2 = Tree.init(root)
    assert t2.head() == sha1


def test_u7_tree_branch_from_sha(tmp_path):
    root = tmp_path / "tree"
    t = Tree.init(root)
    sha1 = t.commit_case("a", {"params.json": "{}"}, "one")
    sha2 = t.commit_case("b", {"params.json": "{}"}, "two")
    t.branch("run/C/b2", at_sha=sha1)
    assert t.head() == sha1
    assert not (root / "cases" / "b").exists()
    sha3 = t.commit_case("c", {"params.json": "{}"}, "three")
    assert _git("rev-parse", "run/C/b2", cwd=root) == sha3
    assert _git("rev-parse", "main", cwd=root) == sha2
    assert _git("tag", cwd=root) == ""


# ---------------------------------------------------------------------------
# U8  nelder_mead_propose
# ---------------------------------------------------------------------------

OPT_U8 = 0.7  # optimum inside unit bounds, 0.2 away from x0=0.5 in every coordinate


@pytest.fixture
def scenario_u8() -> Scenario:
    return dataclasses.replace(
        WING_SCENARIO,
        bounds={k: (0.0, 1.0) for k in PARAM_KEYS},
        x0={k: 0.5 for k in PARAM_KEYS},
    )


def _err(p: dict) -> float:
    return max(abs(float(p[k]) - OPT_U8) for k in PARAM_KEYS)


def test_u8_nelder_mead_quadratic(scenario_u8):
    seen: list[dict] = []

    def evaluate(params: dict):
        assert set(params) == set(PARAM_KEYS)
        seen.append({k: float(params[k]) for k in PARAM_KEYS})
        return -sum(((float(params[k]) - OPT_U8) / 0.5) ** 2 for k in PARAM_KEYS)

    out = nelder_mead_propose(scenario_u8, evaluate, 25)
    assert isinstance(out, list)
    assert 1 <= len(out) <= 25
    assert len(seen) <= 25
    assert [{k: float(p[k]) for k in PARAM_KEYS} for p in out] == seen  # in evaluation order
    for p in out:
        for k in PARAM_KEYS:
            assert 0.0 <= float(p[k]) <= 1.0
    assert min(_err(p) for p in out[-5:]) < 0.1
    assert min(_err(p) for p in out) < 0.1


def test_u8_nelder_mead_deterministic(scenario_u8):
    def evaluate(params: dict):
        return -sum((float(params[k]) - OPT_U8) ** 2 for k in PARAM_KEYS)

    a = nelder_mead_propose(scenario_u8, evaluate, 25)
    b = nelder_mead_propose(scenario_u8, evaluate, 25)
    assert [{k: float(p[k]) for k in PARAM_KEYS} for p in a] == [
        {k: float(p[k]) for k in PARAM_KEYS} for p in b
    ]


def test_u8_nelder_mead_tolerates_none(scenario_u8):
    def evaluate_partial(params: dict):
        if float(params["h_c"]) > 0.6:
            return None
        return -sum((float(params[k]) - OPT_U8) ** 2 for k in PARAM_KEYS)

    out = nelder_mead_propose(scenario_u8, evaluate_partial, 25)
    assert 1 <= len(out) <= 25
    for p in out:
        for k in PARAM_KEYS:
            assert 0.0 <= float(p[k]) <= 1.0

    out_all_none = nelder_mead_propose(scenario_u8, lambda params: None, 10)
    assert 1 <= len(out_all_none) <= 10


def test_u8_nelder_mead_respects_budget(scenario_u8):
    n = 0

    def evaluate(params: dict):
        nonlocal n
        n += 1
        return -sum((float(params[k]) - OPT_U8) ** 2 for k in PARAM_KEYS)

    out = nelder_mead_propose(scenario_u8, evaluate, 5)
    assert len(out) <= 5
    assert n <= 5


# ---------------------------------------------------------------------------
# U9  DesignAgent.parse_exec_result (no LLM)
# ---------------------------------------------------------------------------


@pytest.fixture
def aide_env(tmp_path, monkeypatch):
    """Import AIDE, block every LLM path, and build a minimal AIDE Config."""
    aide = pytest.importorskip("aide")
    import aide.agent
    import aide.backend
    from aide.interpreter import ExecutionResult
    from aide.journal import Journal, Node
    from aide.utils.config import Config
    from aide.utils.metric import MetricValue, WorstMetricValue

    omegaconf = pytest.importorskip("omegaconf")

    def _no_llm(*args, **kwargs):
        raise AssertionError("LLM query must not be called by DesignAgent.parse_exec_result")

    monkeypatch.setattr(aide.backend, "query", _no_llm)
    monkeypatch.setattr(aide.agent, "query", _no_llm)
    for name in (
        "backend_claude_cli",
        "backend_anthropic",
        "backend_openai",
        "backend_openrouter",
        "backend_gemini",
    ):
        mod = getattr(aide.backend, name, None)
        if mod is not None and hasattr(mod, "query"):
            monkeypatch.setattr(mod, "query", _no_llm)

    real_run, real_popen = subprocess.run, subprocess.Popen

    def _guard(fn):
        def wrapped(args, *a, **kw):
            argv = args if isinstance(args, (list, tuple)) else [str(args)]
            if argv and Path(str(argv[0])).name == "claude":
                raise AssertionError("claude CLI must not be spawned by DesignAgent.parse_exec_result")
            return fn(args, *a, **kw)

        return wrapped

    monkeypatch.setattr(subprocess, "run", _guard(real_run))
    monkeypatch.setattr(subprocess, "Popen", _guard(real_popen))

    default_yaml = Path(aide.__file__).resolve().parent / "utils" / "config.yaml"
    cfg = omegaconf.OmegaConf.merge(
        omegaconf.OmegaConf.structured(Config), omegaconf.OmegaConf.load(default_yaml)
    )
    cfg.exp_name = "unit-agentloop"
    cfg.log_dir = str(tmp_path / "logs")
    cfg.workspace_dir = str(tmp_path / "ws")
    cfg.agent.code.model = "claude-cli:none"
    cfg.agent.feedback.model = "claude-cli:none"
    cfg.agent.steps = 3
    cfg.agent.data_preview = False

    return SimpleNamespace(
        cfg=cfg,
        Agent=aide.agent.Agent,
        ExecutionResult=ExecutionResult,
        Journal=Journal,
        Node=Node,
        MetricValue=MetricValue,
        WorstMetricValue=WorstMetricValue,
    )


def _plan_valid(plan_json) -> bool:
    return plan_json["valid"]


def _plan_missing(plan_json) -> list:
    return plan_json["missing"]


def test_u9_parse_exec_result_with_eval_json(aide_env):
    journal = aide_env.Journal()
    agent = DesignAgent("task description", aide_env.cfg, journal, WING_SCENARIO, "C")
    assert isinstance(agent, aide_env.Agent)

    result = _result(Cl=-1.5, Cd=0.03)
    node = aide_env.Node(code=DESIGN_CODE, plan=_fence(GOOD_PLAN))
    exec_result = aide_env.ExecutionResult(
        term_out=["gmsh: ok\n", "simpleFoam: converged\n", _eval_line(1.5, False, "Cl=-1.5 Cd=0.03 all trusted", result) + "\n"],
        exec_time=1.25,
        exc_type=None,
    )
    agent.parse_exec_result(node, exec_result)

    assert node.is_buggy is False
    assert isinstance(node.metric, aide_env.MetricValue)
    assert not isinstance(node.metric, aide_env.WorstMetricValue)
    assert abs(node.metric.value - 1.5) < 1e-9
    assert node.metric.maximize is True
    assert node.analysis == "Cl=-1.5 Cd=0.03 all trusted"
    assert _plan_valid(node.plan_json) is True
    assert _plan_missing(node.plan_json) == []
    assert node.plan_json["intent"] == "continue"
    assert node.plan_json["prediction"] == 1.7


def test_u9_parse_exec_result_without_eval_json(aide_env):
    agent = DesignAgent("task description", aide_env.cfg, aide_env.Journal(), WING_SCENARIO, "B")
    node = aide_env.Node(code=DESIGN_CODE, plan="I will just try something.")
    exec_result = aide_env.ExecutionResult(
        term_out=["Traceback (most recent call last):\n", "KeyError: 'h_c'\n"],
        exec_time=0.1,
        exc_type="KeyError",
    )
    agent.parse_exec_result(node, exec_result)

    assert node.is_buggy is True
    assert isinstance(node.metric, aide_env.WorstMetricValue)
    assert _plan_valid(node.plan_json) is False
    assert _plan_missing(node.plan_json) == sorted(PLAN_KEYS)


def test_u9_parse_exec_result_block_bug_or_null_metric(aide_env):
    agent = DesignAgent("task description", aide_env.cfg, aide_env.Journal(), WING_SCENARIO, "C")

    node_bug = aide_env.Node(code=DESIGN_CODE, plan=_fence(GOOD_PLAN))
    agent.parse_exec_result(
        node_bug,
        aide_env.ExecutionResult(
            term_out=[_eval_line(None, True, "failed: mesh_invalid", _result(status="failed")) + "\n"],
            exec_time=0.5,
            exc_type=None,
        ),
    )
    assert node_bug.is_buggy is True
    assert isinstance(node_bug.metric, aide_env.WorstMetricValue)
    assert node_bug.analysis == "failed: mesh_invalid"
    assert _plan_valid(node_bug.plan_json) is True

    node_untrusted = aide_env.Node(code=DESIGN_CODE, plan=_fence(GOOD_PLAN))
    agent.parse_exec_result(
        node_untrusted,
        aide_env.ExecutionResult(
            term_out=[_eval_line(None, False, "ok but converged=False", _result(trust_false=("converged",))) + "\n"],
            exec_time=0.5,
            exc_type=None,
        ),
    )
    assert node_untrusted.is_buggy is True  # metric None -> buggy
    assert isinstance(node_untrusted.metric, aide_env.WorstMetricValue)


def test_u9_search_policy_and_data_preview_need_no_llm(aide_env):
    journal = aide_env.Journal()
    agent = DesignAgent("task description", aide_env.cfg, journal, WING_SCENARIO, "C")

    assert agent.update_data_preview() is None

    assert agent.search_policy() is None
    assert agent.last_action == "draft"

    node = aide_env.Node(code=DESIGN_CODE, plan=_fence(GOOD_PLAN))
    agent.parse_exec_result(
        node,
        aide_env.ExecutionResult(
            term_out=[_eval_line(1.5, False, "ok", _result()) + "\n"], exec_time=1.0, exc_type=None
        ),
    )
    journal.append(node)
    picked = agent.search_policy()
    assert picked is node
    assert agent.last_action == "improve"


# ---------------------------------------------------------------------------
# U10  run_line refuses an existing line directory
# ---------------------------------------------------------------------------


def test_u10_run_line_existing_line_dir_raises(tmp_path):
    exp_dir = tmp_path / "exp"
    line_dir = exp_dir / "lines" / "A"
    line_dir.mkdir(parents=True)
    store = tmp_path / "store"
    verify_calls: list = []

    def verify_fn(*args, **kwargs):
        verify_calls.append((args, kwargs))
        return {}

    cfg = {
        "line": "A",
        "scenario": WING_SCENARIO,
        "budget": 3,
        "verify_every": 1,
        "exp_dir": exp_dir,
        "store": store,
        "model": None,
        "tau": 0.4,
        "k_backtrack": 4,
        "k_stagnation": 8,
        "seed": 0,
        "verify_fn": verify_fn,
    }
    with pytest.raises(FileExistsError):
        run_line(cfg)

    # nothing ran: no case files, no store content, no summary, no verify calls
    assert list(line_dir.iterdir()) == []
    assert not store.exists() or not any(store.iterdir())
    assert list(exp_dir.rglob("summary.json")) == []
    assert list(exp_dir.rglob("holdout.json")) == []
    assert verify_calls == []


# ---------------------------------------------------------------------------
# U11  promote_tier
# ---------------------------------------------------------------------------

PARAMS_U11 = {"h_c": 0.2, "alpha_deg": 3.0, "camber": 0.04}


def _params_of(tier):
    if tier is None:
        return lambda step: dict(PARAMS_U11)
    return lambda step: {**PARAMS_U11, "tier": tier}


def test_u11_single_tier_always_none():
    recs = [_rec(0, 1.0), _rec(1, 1.5)]
    assert promote_tier(recs, _params_of(None), (0,)) is None
    assert promote_tier(recs, _params_of(0), (0,)) is None
    assert promote_tier([_rec(0, 1.0)], _params_of(0), (0,)) is None
    assert promote_tier([], _params_of(0), (0,)) is None


def test_u11_new_best_at_low_tier_is_promoted():
    recs = [_rec(0, 1.0), _rec(1, 1.5)]
    out = promote_tier(recs, _params_of(0), (0, 2))
    assert out == {**PARAMS_U11, "tier": 2}
    # three tiers: middle -> top
    out3 = promote_tier(recs, _params_of(1), (0, 1, 2))
    assert out3 == {**PARAMS_U11, "tier": 2}
    # a single first record is trivially the new best
    assert promote_tier([_rec(0, 1.0)], _params_of(0), (0, 2)) == {**PARAMS_U11, "tier": 2}


def test_u11_already_top_tier_is_none():
    recs = [_rec(0, 1.0), _rec(1, 1.5)]
    assert promote_tier(recs, _params_of(2), (0, 2)) is None


def test_u11_not_best_is_none():
    recs = [_rec(0, 1.5), _rec(1, 1.0)]
    assert promote_tier(recs, _params_of(0), (0, 2)) is None
    tie = [_rec(0, 1.5), _rec(1, 1.5)]
    assert promote_tier(tie, _params_of(0), (0, 2)) is None


def test_u11_bug_is_none():
    recs = [_rec(0, 1.0), _rec(1, None, is_bug=True)]
    assert promote_tier(recs, _params_of(0), (0, 2)) is None
    assert promote_tier([], _params_of(0), (0, 2)) is None
