"""U25/U27: complete six-arm scheduling contract; injected runs are synthetic."""
from __future__ import annotations

import copy
import importlib
import json
import os
from pathlib import Path

import pytest


@pytest.fixture
def run_suite():
    return importlib.import_module("pipeline.agentloop.driver").run_suite


def _forbidden_backend(prompt):
    raise AssertionError("invalid suite must not invoke any language model")


@pytest.mark.parametrize("omission", range(6))
def test_suite_refuses_missing_arm_before_tool_or_llm(run_suite, level_suite, synthetic_invoker, tmp_path, omission):
    del level_suite["runs"][omission]
    invoke, calls = synthetic_invoker
    llm_calls = []

    def backend(*args, **kwargs):
        llm_calls.append((args, kwargs))
        raise AssertionError("invalid suite must not reach LLM")

    with pytest.raises(ValueError):
        run_suite(level_suite, tmp_path / "suite", backend=backend, invoke_fn=invoke)
    assert calls == [] and llm_calls == []


def test_suite_refuses_missing_physical_node_before_execution(run_suite, level_suite, synthetic_invoker, tmp_path):
    del level_suite["profiles"]["M1"]
    invoke, calls = synthetic_invoker
    with pytest.raises(ValueError):
        run_suite(level_suite, tmp_path / "suite", backend=_forbidden_backend, invoke_fn=invoke)
    assert calls == []


def test_suite_refuses_unknown_keys_before_execution(run_suite, level_suite, synthetic_invoker, tmp_path):
    level_suite["silent_skip_missing"] = True
    invoke, calls = synthetic_invoker
    with pytest.raises(ValueError):
        run_suite(level_suite, tmp_path / "suite", backend=_forbidden_backend, invoke_fn=invoke)
    assert calls == []


def _stage(summary, level, arm, stage_id):
    run = next(row for row in summary["runs"] if row["level"] == level and row["arm"] == arm)
    return next(row for row in run["stages"] if row["stage_id"] == stage_id)


def _m1_target_stop_is_covered(stage, loaded, seed_hashes, tolerance):
    """Acceptance rule only; its unit fixtures do not claim physical execution."""
    if stage["stage_id"] != "M1" or not seed_hashes or stage["cost"].get("llm_calls", 0) < 1:
        return False
    formal = [row for row in loaded["observations"] if row["stage_id"] == "M1"
              and row["protocol_hash"] == stage["protocol_hash"] and row["source"] == "live"
              and row["role"] == "formal" and row["status"] == "completed"
              and row["cost"].get("solver_calls", 0) > 0]
    if not set(seed_hashes) <= {row["design_hash"] for row in formal}:
        return False
    best = stage.get("best_confirmed")
    if not best or not best.get("rank_eligible") or best.get("completeness") != "complete" \
            or best.get("numerical_status") != "pass" or best.get("feasibility") != "feasible":
        return False
    value = best.get("objective_value")
    if value is None or not -tolerance <= value <= 0:
        return False
    best_refs = set(best.get("evidence_refs", []))
    formal_refs = {row["observation_id"] for row in formal}
    if not best_refs or not best_refs <= formal_refs:
        return False
    for proposal in loaded["proposals"]:
        refs = set(proposal.get("evidence_refs", []))
        reason = proposal.get("stop_reason", "").lower()
        target_reason = any(word in reason for word in ("target", "目标", "tolerance", "容差", "resolution", "分辨率"))
        if proposal.get("action") == "stop" and target_reason and best_refs <= refs <= formal_refs:
            return True
    return False


@pytest.fixture
def m1_stop_evidence():
    stage = {"stage_id": "M1", "protocol_hash": "p1", "cost": {"llm_calls": 1},
             "best_confirmed": {"rank_eligible": True, "completeness": "complete",
                                "numerical_status": "pass", "feasibility": "feasible",
                                "objective_value": -0.005, "evidence_refs": ["new-stage-best"]}}
    observations = [{"stage_id": "M1", "protocol_hash": "p1", "source": "live", "role": "formal",
                     "status": "completed", "cost": {"solver_calls": 1}, "design_hash": design,
                     "observation_id": identity}
                    for design, identity in (("seed-a", "new-stage-best"), ("seed-b", "new-stage-other"))]
    loaded = {"observations": observations, "proposals": [{"action": "stop",
               "evidence_refs": ["new-stage-best"], "stop_reason": "Target error is below the frozen numerical resolution."}]}
    return stage, loaded


def test_m1_paid_reassessment_and_evidence_based_target_stop_is_a_closed_loop(m1_stop_evidence):
    stage, loaded = m1_stop_evidence
    assert _m1_target_stop_is_covered(stage, loaded, {"seed-a", "seed-b"}, 0.01)


@pytest.mark.parametrize("violation", ["M0", "missing_seed", "unpaid", "cached", "ineligible",
                                       "outside_resolution", "no_best_reference", "foreign_reference", "budget_stop", "no_llm"])
def test_m1_stop_exception_cannot_admit_missing_scientific_or_decision_evidence(m1_stop_evidence, violation):
    stage, loaded = m1_stop_evidence
    if violation == "M0":
        stage["stage_id"] = "M0"
    elif violation == "missing_seed":
        loaded["observations"].pop()
    elif violation == "unpaid":
        loaded["observations"][0]["cost"]["solver_calls"] = 0
    elif violation == "cached":
        loaded["observations"][0]["source"] = "cache"
    elif violation == "ineligible":
        stage["best_confirmed"]["rank_eligible"] = False
    elif violation == "outside_resolution":
        stage["best_confirmed"]["objective_value"] = -0.0101
    elif violation == "no_best_reference":
        loaded["proposals"][0]["evidence_refs"] = ["new-stage-other"]
    elif violation == "foreign_reference":
        loaded["proposals"][0]["evidence_refs"] = ["new-stage-best", "parent-stage-best"]
    elif violation == "budget_stop":
        loaded["proposals"][0]["stop_reason"] = "Budget exhausted."
    else:
        stage["cost"]["llm_calls"] = 0
    assert not _m1_target_stop_is_covered(stage, loaded, {"seed-a", "seed-b"}, 0.01)


def _scripted_backend(level_candidate, explicit_candidate):
    """One feedback-derived formal design per fixed stage, then stop."""
    prompts = []
    counts = {}

    def backend(prompt):
        prompts.append(copy.deepcopy(prompt))
        experiment = prompt["experiment"]
        key = (experiment["level"], experiment["arm"], experiment["stage_id"])
        step = counts.get(key, 0)
        counts[key] = step + 1
        candidate = copy.deepcopy(explicit_candidate if experiment["level"] >= 2 else level_candidate)
        if step:
            return json.dumps({"action": "stop", "stop_reason": "Scripted coverage completed."})
        already_at_target = any(abs(row.get("raw_metrics", {}).get("Cl", 0) - 0.5) < 1e-12
                                for row in prompt["observations"])
        candidate["parameters"]["alpha_deg"] = 4.0 if already_at_target else 5.0
        refs = [row["observation_id"] for row in prompt["observations"]]
        action = "submit_for_evaluation" if key[:2] == (2, "b") else "evaluate_design"
        return json.dumps({"action": action, "candidate": candidate,
                           "question": "Move toward the target using observed lift.", "evidence_refs": refs})

    return backend, prompts


@pytest.mark.flow
def test_complete_suite_has_six_arms_every_node_shared_m0_and_truthful_source(
    run_suite, level_suite, synthetic_invoker, level_candidate, explicit_candidate, tmp_path,
):
    invoke, calls = synthetic_invoker
    backend, prompts = _scripted_backend(level_candidate, explicit_candidate)
    summary = run_suite(level_suite, tmp_path / "suite", backend=backend, invoke_fn=invoke)
    assert {(row["level"], row["arm"]) for row in summary["runs"]} == {
        (0, "base"), (1, "a"), (1, "b"), (2, "a"), (2, "b"), (3, "topk")}
    for run in summary["runs"]:
        expected = {"M0", "M1"} if run["level"] in (1, 3) else {"M0"}
        assert {row["stage_id"] for row in run["stages"]} == expected
        for stage in run["stages"]:
            assert stage["source"] == "synthetic"
            assert stage["formal_results"], "script submits a feasible formal design in every stage"
            assert stage["cost"]["solver_calls"] > 0
            assert Path(stage["output_dir"]).is_dir()
            assert stage["status"] not in {"skipped", "unsupported", "failed", "pending"}
    baseline = _stage(summary, 0, "base", "M0")
    for arm in ("a", "b"):
        shared = _stage(summary, 1, arm, "M0")
        assert shared["output_dir"] == baseline["output_dir"]
        assert shared["cost"] == baseline["cost"]
    unique_stages = {stage["output_dir"]: stage for run in summary["runs"] for stage in run["stages"]}
    assert len(unique_stages) == 7
    assert sum(stage["cost"]["solver_calls"] for stage in unique_stages.values()) == sum(row["tool"] == "solve" for row in calls)
    assert prompts and all({"task", "experiment", "observations", "best_confirmed", "unconfirmed", "budget", "initialization"} <= set(prompt) for prompt in prompts)


@pytest.mark.flow
def test_completed_suite_resume_uses_commits_without_tool_or_llm_reexecution(
    run_suite, level_suite, synthetic_invoker, level_candidate, explicit_candidate, tmp_path,
):
    invoke, calls = synthetic_invoker
    backend, prompts = _scripted_backend(level_candidate, explicit_candidate)
    path = tmp_path / "suite"
    first = run_suite(level_suite, path, backend=backend, invoke_fn=invoke)
    counts = (len(calls), len(prompts))
    second = run_suite(level_suite, path, backend=backend, invoke_fn=invoke, resume=True)
    assert (len(calls), len(prompts)) == counts
    assert second["runs"] == first["runs"]


@pytest.mark.flow
def test_suite_snapshot_change_refuses_resume_before_new_execution(
    run_suite, level_suite, synthetic_invoker, level_candidate, explicit_candidate, tmp_path,
):
    invoke, calls = synthetic_invoker
    backend, prompts = _scripted_backend(level_candidate, explicit_candidate)
    path = tmp_path / "suite"
    run_suite(level_suite, path, backend=backend, invoke_fn=invoke)
    counts = (len(calls), len(prompts))
    level_suite["profiles"]["M1"]["flow"]["Re"] = 200.0
    with pytest.raises(ValueError):
        run_suite(level_suite, path, backend=backend, invoke_fn=invoke, resume=True)
    assert (len(calls), len(prompts)) == counts


@pytest.mark.flow
def test_topk_transfers_designs_while_cold_start_has_only_public_initialization(
    run_suite, level_suite, synthetic_invoker, level_candidate, explicit_candidate, tmp_path,
):
    invoke, calls = synthetic_invoker
    backend, prompts = _scripted_backend(level_candidate, explicit_candidate)
    summary = run_suite(level_suite, tmp_path / "suite", backend=backend, invoke_fn=invoke)
    first = {}
    for prompt in prompts:
        experiment = prompt["experiment"]
        first.setdefault((experiment["level"], experiment["arm"], experiment["stage_id"]), prompt)
    inherited = first[(1, "a", "M1")]
    cold = first[(1, "b", "M1")]
    assert '5.0' in json.dumps(inherited["initialization"])
    assert '5.0' not in json.dumps(cold["initialization"])
    assert all(row["stage_id"] == "M1" for row in inherited["observations"])
    assert all(row["stage_id"] == "M1" for row in cold["observations"])
    parent_path = _stage(summary, 0, "base", "M0")["output_dir"]
    assert parent_path not in json.dumps(cold)
    cold_path = Path(_stage(summary, 1, "b", "M1")["output_dir"])
    assert cold_path != Path(parent_path)
    assert _stage(summary, 1, "a", "M1")["cost"]["solver_calls"] > 0
    assert _stage(summary, 1, "b", "M1")["cost"]["solver_calls"] > 0


@pytest.mark.flow
def test_two_successful_explorations_supply_context_without_formal_score_then_confirm(
    run_suite, level_suite, synthetic_invoker, level_candidate, explicit_candidate, tmp_path,
):
    invoke, calls = synthetic_invoker
    normal_backend, _ = _scripted_backend(level_candidate, explicit_candidate)
    exploration_prompts = []

    def explore_invoke(tool, inputs, context):
        result = invoke(tool, inputs, context)
        if tool == "post":
            result["raw_metrics"]["Cl"] = 0.5 if context["role"] == "exploration" else 0.2
        return result

    def backend(prompt):
        experiment = prompt["experiment"]
        if (experiment["level"], experiment["arm"]) != (2, "b"):
            return normal_backend(prompt)
        exploration_prompts.append(copy.deepcopy(prompt))
        step = len(exploration_prompts) - 1
        refs = [row["observation_id"] for row in prompt["observations"]]
        candidate = copy.deepcopy(explicit_candidate)
        candidate["parameters"]["alpha_deg"] = 2.0 + min(step, 1)
        if step < 2:
            return json.dumps({"action": "explore", "candidate": candidate, "evidence_refs": refs,
                               "requested_numerics": {"mesh_scale": [0.7, 1.3][step]},
                               "question": "Use exploration feedback before formal submission."})
        if step == 2:
            return json.dumps({"action": "submit_for_evaluation", "candidate": candidate, "evidence_refs": refs})
        return json.dumps({"action": "stop", "stop_reason": "Confirmed under frozen settings."})

    summary = run_suite(level_suite, tmp_path / "suite", backend=backend, invoke_fn=explore_invoke)
    assert len(exploration_prompts) >= 3
    third = exploration_prompts[2]
    assert third["best_confirmed"] is None
    assert len(third["unconfirmed"]) >= 2
    assert len(third["observations"]) >= 2
    assert all(row["role"] == "exploration" for row in third["observations"])
    assert all(row.get("is_buggy") is not True for row in third["observations"])
    stage = _stage(summary, 2, "b", "M0")
    assert stage["formal_results"]
    assert all(row["objective_value"] == pytest.approx(-0.3) for row in stage["formal_results"])
    explored = [row for row in calls if row["tool"] == "solve" and row["context"]["role"] == "exploration"]
    assert {row["inputs"]["settings"]["mesh"]["mesh_scale"] for row in explored} == {0.7, 1.3}


@pytest.mark.flow
def test_only_exploration_can_finish_with_no_formal_solution(
    run_suite, level_suite, synthetic_invoker, level_candidate, explicit_candidate, tmp_path,
):
    invoke, calls = synthetic_invoker
    normal_backend, _ = _scripted_backend(level_candidate, explicit_candidate)
    prompts = []

    def backend(prompt):
        if (prompt["experiment"]["level"], prompt["experiment"]["arm"]) != (2, "b"):
            return normal_backend(prompt)
        prompts.append(copy.deepcopy(prompt))
        if len(prompts) == 1:
            return json.dumps({"action": "explore", "candidate": explicit_candidate,
                               "requested_numerics": {"mesh_scale": 0.7}})
        return json.dumps({"action": "stop", "stop_reason": "Remain explicitly unconfirmed."})

    summary = run_suite(level_suite, tmp_path / "suite", backend=backend, invoke_fn=invoke)
    stage = _stage(summary, 2, "b", "M0")
    assert stage["formal_results"] == []
    assert len(prompts) == 2 and prompts[-1]["best_confirmed"] is None
    assert prompts[-1]["unconfirmed"]
    assert stage["cost"]["solver_calls"] > 0


@pytest.mark.parametrize("bad_value", [None, -1, True, "1"])
def test_suite_requires_frozen_valid_topk(run_suite, level_suite, synthetic_invoker, tmp_path, bad_value):
    level_suite["topk"] = bad_value
    invoke, calls = synthetic_invoker
    with pytest.raises(ValueError):
        run_suite(level_suite, tmp_path / "suite", backend=_forbidden_backend, invoke_fn=invoke)
    assert calls == []


def test_suite_cannot_silently_shorten_frozen_path(run_suite, level_suite, synthetic_invoker, tmp_path):
    level_suite["runs"][-1]["stages"] = ["M0"]
    invoke, calls = synthetic_invoker
    with pytest.raises(ValueError):
        run_suite(level_suite, tmp_path / "suite", backend=_forbidden_backend, invoke_fn=invoke)
    assert calls == []


@pytest.mark.flow
def test_l2b_reserves_confirmation_budget_before_exploration(
    run_suite, level_suite, synthetic_invoker, level_candidate, explicit_candidate, tmp_path,
):
    for run in level_suite["runs"][3:5]:
        run["limits"] = {"solve_slots": 1, "validation_reserve": 4}
    invoke, calls = synthetic_invoker
    normal_backend, _ = _scripted_backend(level_candidate, explicit_candidate)

    def backend(prompt):
        if (prompt["experiment"]["level"], prompt["experiment"]["arm"]) != (2, "b"):
            return normal_backend(prompt)
        return json.dumps({"action": "explore", "candidate": explicit_candidate,
                           "requested_numerics": {"mesh_scale": 0.7}})

    summary = run_suite(level_suite, tmp_path / "suite", backend=backend, invoke_fn=invoke)
    stage = _stage(summary, 2, "b", "M0")
    assert stage["formal_results"] == []
    assert stage["cost"]["solver_calls"] == 0
    assert not [row for row in calls if row["context"]["role"] == "exploration"]


@pytest.mark.flow
def test_stage_interruption_resume_retains_commits_without_repeating_baseline(
    run_suite, level_suite, synthetic_invoker, level_candidate, explicit_candidate, tmp_path,
):
    invoke, calls = synthetic_invoker
    normal_backend, _ = _scripted_backend(level_candidate, explicit_candidate)
    interrupted = False

    def backend(prompt):
        nonlocal interrupted
        exp = prompt["experiment"]
        if not interrupted and (exp["level"], exp["arm"], exp["stage_id"]) == (1, "a", "M1"):
            interrupted = True
            raise KeyboardInterrupt("simulated process interruption after committed seed")
        return normal_backend(prompt)

    path = tmp_path / "suite"
    with pytest.raises(KeyboardInterrupt):
        run_suite(level_suite, path, backend=backend, invoke_fn=invoke)
    before = [(row["context"]["stage_id"], row["context"]["design_hash"], row["inputs"]["case_dir"])
              for row in calls if row["tool"] == "solve"]
    split = len(calls)
    summary = run_suite(level_suite, path, backend=backend, invoke_fn=invoke, resume=True)
    later_cases = {row["inputs"]["case_dir"] for row in calls[split:] if row["tool"] == "solve"}
    assert not ({case for _, _, case in before} & later_cases)
    assert len(summary["runs"]) == 6
    assert _stage(summary, 0, "base", "M0")["cost"]["solver_calls"] == 2
    assert _stage(summary, 1, "a", "M1")["cost"]["solver_calls"] == 2


@pytest.mark.flow
def test_failed_parent_llm_blocks_child_until_parent_recovers_and_refreezes_topk(
    run_suite, level_suite, synthetic_invoker, level_candidate, explicit_candidate, tmp_path,
):
    """U27: a normal infrastructure error cannot freeze a child from unfinished TopK."""
    invoke, calls = synthetic_invoker
    normal_backend, _ = _scripted_backend(level_candidate, explicit_candidate)
    failure_boundary = None
    failed_once = False
    requested_stages = []

    def backend(prompt):
        nonlocal failure_boundary, failed_once
        exp = prompt["experiment"]
        requested_stages.append((exp["level"], exp["arm"], exp["stage_id"]))
        if not failed_once and (exp["level"], exp["arm"], exp["stage_id"]) == (3, "topk", "M0"):
            failed_once = True
            failure_boundary = len(calls)
            raise RuntimeError("synthetic transient LLM infrastructure error")
        return normal_backend(prompt)

    directory = tmp_path / "suite"
    try:
        failed_summary = run_suite(level_suite, directory, backend=backend, invoke_fn=invoke)
    except RuntimeError:
        failed_summary = None
    assert failed_once and failure_boundary is not None
    assert not [row for row in calls[failure_boundary:] if row["context"]["stage_id"] == "M1"], (
        "a failed parent must block its dependent child, not freeze incomplete inherited seeds")
    assert (3, "topk", "M1") not in requested_stages
    if failed_summary is not None:
        assert _stage(failed_summary, 3, "topk", "M0")["status"] != "completed"
        child = _stage(failed_summary, 3, "topk", "M1")
        assert child["status"] == "blocked"
        assert child["cost"].get("solver_calls", 0) == child["cost"].get("llm_calls", 0) == 0
        build = importlib.import_module("pipeline.agentloop.evaluation.finding_report").build
        assert build(directory)["schema_version"] == 1
    recovered = run_suite(level_suite, directory, backend=backend, invoke_fn=invoke, resume=True)
    parent = _stage(recovered, 3, "topk", "M0")
    child = _stage(recovered, 3, "topk", "M1")
    assert parent["formal_results"] and child["formal_results"]
    assert '"alpha_deg": 5.0' in json.dumps(child["initialization"])
    assert child["cost"]["solver_calls"] > 0


@pytest.mark.flow
def test_completed_parent_without_feasible_designs_uses_public_fallback_in_child(
    run_suite, level_suite, synthetic_invoker, level_candidate, explicit_candidate, tmp_path,
):
    """An honest infeasible research result is not an infrastructure dependency failure."""
    invoke, calls = synthetic_invoker
    backend, _ = _scripted_backend(level_candidate, explicit_candidate)

    def infeasible_m0(tool, inputs, context):
        result = invoke(tool, inputs, context)
        if tool == "post" and context["stage_id"] == "M0":
            result["raw_metrics"]["Cd"] = 0.3  # Valid measurement, frozen hard bound is 0.1.
        return result

    summary = run_suite(level_suite, tmp_path / "suite", backend=backend, invoke_fn=infeasible_m0)
    for level, arm in ((1, "a"), (3, "topk")):
        parent = _stage(summary, level, arm, "M0")
        child = _stage(summary, level, arm, "M1")
        assert parent["status"] == "completed" and parent["formal_results"] == []
        assert child["status"] == "completed" and child["formal_results"]
        assert child["cost"]["solver_calls"] > 0
        assert '"alpha_deg": 2.0' in json.dumps(child["initialization"])
        assert '"alpha_deg": 5.0' not in json.dumps(child["initialization"])


@pytest.mark.flow
def test_all_frozen_public_seeds_are_preserved_and_l2b_can_see_them_without_confirmation(
    run_suite, level_suite, synthetic_invoker, level_candidate, explicit_candidate, tmp_path,
):
    for representation, prototype in (("parameterized", level_candidate), ("explicit_shape", explicit_candidate)):
        second = copy.deepcopy(prototype)
        second["parameters"]["alpha_deg"] = 0.0
        level_suite["initialization"][representation].append(second)
    invoke, calls = synthetic_invoker
    prompts = []

    def backend(prompt):
        prompts.append(copy.deepcopy(prompt))
        return json.dumps({"action": "stop", "stop_reason": "Inspect frozen initialization only."})

    run_suite(level_suite, tmp_path / "suite", backend=backend, invoke_fn=invoke)
    first = {(row["experiment"]["level"], row["experiment"]["arm"], row["experiment"]["stage_id"]): row
             for row in prompts}
    for key in ((0, "base", "M0"), (2, "a", "M0"), (3, "topk", "M0"), (1, "b", "M1")):
        assert len(first[key]["observations"]) == 2
        assert len({row["design_hash"] for row in first[key]["observations"]}) == 2
    exploration = first[(2, "b", "M0")]
    initialization = exploration["initialization"]
    # Public initialization may include metadata, but every frozen design is visible.
    serialized = json.dumps(initialization, sort_keys=True)
    assert '"alpha_deg": 0.0' in serialized and '"alpha_deg": 2.0' in serialized
    assert exploration["observations"] == [] and exploration["best_confirmed"] is None


@pytest.mark.flow
def test_report_groups_formal_curves_by_stage_and_protocol_with_real_costs(
    run_suite, level_suite, synthetic_invoker, level_candidate, explicit_candidate, tmp_path,
):
    invoke, calls = synthetic_invoker
    backend, _ = _scripted_backend(level_candidate, explicit_candidate)
    directory = tmp_path / "suite"
    summary = run_suite(level_suite, directory, backend=backend, invoke_fn=invoke)
    build = importlib.import_module("pipeline.agentloop.evaluation.finding_report").build
    report = build(directory)
    assert report["schema_version"] == 1
    groups = report["protocol_groups"]
    assert groups and any(key.startswith("M0|") for key in groups) and any(key.startswith("M1|") for key in groups)
    for identity, group in groups.items():
        assert {"runs", "curves", "cost"} <= set(group)
        for curve in group["curves"].values():
            prior_cost = 0
            for point in curve:
                assert {"objective_value", "cumulative_cost", "design_hash", "evidence_refs"} <= set(point)
                assert point["objective_value"] is not None
                assert point["evidence_refs"]
                assert point["cumulative_cost"]["solver_calls"] >= prior_cost
                prior_cost = point["cumulative_cost"]["solver_calls"]
    assert summary["runs"]


@pytest.mark.flow
def test_report_provenance_graph_has_resolvable_unique_nodes_and_cross_stage_origins(
    run_suite, level_suite, synthetic_invoker, level_candidate, explicit_candidate, tmp_path,
):
    invoke, calls = synthetic_invoker
    backend, _ = _scripted_backend(level_candidate, explicit_candidate)
    directory = tmp_path / "suite"
    run_suite(level_suite, directory, backend=backend, invoke_fn=invoke)
    build = importlib.import_module("pipeline.agentloop.evaluation.finding_report").build
    graph = build(directory)["provenance_graph"]
    nodes = graph["nodes"]
    ids = {node["id"] for node in nodes}
    assert len(ids) == len(nodes) and nodes
    assert all({"id", "type", "stage_id", "run_id"} <= set(node) for node in nodes)
    assert graph["edges"]
    for edge in graph["edges"]:
        assert {"source", "target", "type"} <= set(edge)
        assert edge["source"] in ids and edge["target"] in ids
    by_id = {node["id"]: node for node in nodes}
    assert any(by_id[edge["source"]]["stage_id"] == "M0" and by_id[edge["target"]]["stage_id"] == "M1"
               for edge in graph["edges"])


@pytest.mark.smoke
@pytest.mark.flow
def test_real_six_arm_suite_committed_evidence_is_complete_and_cost_reconciles():
    """Read-only final acceptance of separately executed live LLM/OpenFOAM suite.

    A missing suite is a visible skip during development, and must be rerun with
    a real summary for final acceptance. Synthetic fixtures cannot satisfy it.
    """
    root = Path(__file__).resolve().parents[3]
    suite_dir = Path(os.environ.get("INTERVIEW_REAL_SUITE_DIR", str(
        root / "OUTPUTs/20260924_133000-produce-level0-3-smokerun/suite")))
    summary_path = suite_dir / "summary.json"
    if not summary_path.is_file():
        pytest.skip("live six-arm suite has not been executed; this is not acceptance")
    summary = json.loads(summary_path.read_text())
    frozen = json.loads((root / "OUTPUTs/20260924_133000-produce-level0-3-smokerun/pilot/frozen_profiles.json").read_text())
    assert {(row["level"], row["arm"]) for row in summary["runs"]} == {
        (0, "base"), (1, "a"), (1, "b"), (2, "a"), (2, "b"), (3, "topk")}
    RunStore = importlib.import_module("pipeline.agentloop.records").RunStore
    canonical_design = importlib.import_module("pipeline.agentloop.environment").canonical_design

    def candidate_hash(candidate, stage_id):
        try:
            public = {key: value for key, value in candidate.items()
                      if key in {"representation", "parameters", "shape", "candidate_id", "parent_candidate_ids"}}
            return canonical_design(public, frozen["profiles"][stage_id]["geometry"])["design_hash"]
        except ValueError:
            return None  # Rejected proposals stay in the audit, but cannot prove a live design.

    unique = {}
    shape_search = set()
    for run in summary["runs"]:
        level, arm = run["level"], run["arm"]
        expected = {"M0", "M1"} if level in (1, 3) else {"M0"}
        assert {stage["stage_id"] for stage in run["stages"]} == expected
        for stage in run["stages"]:
            stage_id = stage["stage_id"]
            assert stage["source"] == "live"
            assert stage["status"] not in {"failed", "skipped", "pending", "unsupported", "blocked"}
            directory = Path(stage["output_dir"])
            loaded = RunStore(directory).load_committed()
            observations = loaded["observations"]
            assert observations and all(row["source"] == "live" for row in observations)
            assert stage["cost"]["solver_calls"] > 0 and stage["cost"]["llm_calls"] > 0
            assert all(row["stage_id"] == stage_id for row in observations)
            assert all(Path(filename).is_file() for row in observations for filename in row["artifacts"].values())
            by_id = {row["observation_id"]: row for row in observations}
            proposals = [row for row in loaded["proposals"]
                         if row.get("action") in {"evaluate_design", "explore", "submit_for_evaluation", "stop"}]
            decisions = [proposal for proposal in proposals if proposal.get("evidence_refs") and proposal.get("candidate")]
            seed_hashes = {candidate_hash(row.get("candidate", row), stage_id) for row in stage["initialization"]}
            target_stop = _m1_target_stop_is_covered(stage, loaded, seed_hashes, frozen["numerical_resolution"]["objective_tolerance"])
            assert decisions or target_stop, f"{level}/{arm}/{stage_id} needs a feedback-driven subsequent proposal or qualified M1 target-stop"
            for decision in decisions:
                assert all(ref in by_id for ref in decision["evidence_refs"])
            if (level, arm) != (2, "b") and not target_stop:
                formally_executed = {row["design_hash"] for row in observations
                                     if row["role"] == "formal" and row["status"] == "completed"}
                formal_decisions = [row for row in decisions if row["action"] == "evaluate_design"
                                    and candidate_hash(row["candidate"], stage_id) in formally_executed]
                assert formal_decisions, "fixed stages need at least one LLM decision after initial feedback"
                assert any(
                    candidate_hash(row["candidate"], stage_id)
                    not in {by_id[ref]["design_hash"] for ref in row["evidence_refs"]}
                    for row in formal_decisions
                ), "a copied baseline is not a subsequent new design"
            for proposal in proposals:
                candidate = proposal.get("candidate", {})
                if candidate.get("representation") != "explicit_shape":
                    continue
                submitted = candidate.get("shape")
                original_shapes = [row["shape"] for row in frozen["initialization"]["explicit_shape"]]
                executed = [row for row in observations if row["status"] == "completed"
                            and row["design_hash"] == candidate_hash(candidate, stage_id)]
                if submitted and executed and all(submitted != initial for initial in original_shapes):
                    actual_geometry = json.loads(Path(executed[0]["artifacts"]["geometry"]).read_text())
                    assert actual_geometry["shape"] == submitted
                    shape_search.add((level, arm))
            if (level, arm) == (2, "b"):
                explored = [row for row in observations if row["role"] == "exploration"]
                confirmed = [row for row in observations if row["role"] == "formal"]
                assert explored and confirmed
                exploration_ids = {row["observation_id"] for row in explored}
                assert any(row["action"] == "submit_for_evaluation" and exploration_ids.intersection(row.get("evidence_refs", []))
                           for row in proposals)
                first_formal = next(index for index, row in enumerate(observations) if row["role"] == "formal")
                assert any(row["role"] == "exploration" for row in observations[:first_formal])
            unique[str(directory)] = stage
    assert {(2, "a"), (2, "b"), (3, "topk")} <= shape_search
    assert len(unique) == 7
    shared = _stage(summary, 0, "base", "M0")
    for arm in ("a", "b"):
        assert _stage(summary, 1, arm, "M0")["output_dir"] == shared["output_dir"]
        assert _stage(summary, 1, arm, "M0")["cost"] == shared["cost"]
    assert _stage(summary, 1, "a", "M1")["protocol_hash"] == _stage(summary, 1, "b", "M1")["protocol_hash"]
    physical_cost = summary["physical_execution_cost"]
    for field in ("solver_calls", "llm_calls", "tool_seconds"):
        assert physical_cost[field] == pytest.approx(sum(stage["cost"][field] for stage in unique.values()))
