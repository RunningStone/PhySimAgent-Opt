"""U02-U05/U12/U20/U23: public proposal and fixed runtime boundary tests."""
from __future__ import annotations

import copy
import importlib
import os
import signal
import time

import pytest


@pytest.fixture
def environment():
    return importlib.import_module("pipeline.agentloop.environment")


def test_experiment_is_retained_in_task_spec(environment, level_config):
    task = environment.resolve_task(level_config)
    assert task.experiment == level_config["experiment"]


def test_fixed_level_rejects_numerics_before_execution(environment, level_config, level_candidate):
    task = environment.resolve_task(level_config)
    proposal = {"action": "submit_for_evaluation", "candidate": level_candidate,
                "requested_numerics": {"mesh_scale": 0.7}}
    with pytest.raises(environment.ProtocolError):
        environment.validate_proposal(proposal, task)


def test_proposal_cannot_self_certify_score_or_formal_status(environment, level_config, level_candidate):
    task = environment.resolve_task(level_config)
    proposal = {"action": "submit_for_evaluation", "candidate": level_candidate, "score": 100.0, "is_formal": True}
    with pytest.raises(environment.ProtocolError):
        environment.validate_proposal(proposal, task)


def test_design_hash_ignores_candidate_labels_and_parents(environment, level_config, level_candidate):
    first = environment.canonical_design(level_candidate, level_config["task"]["geometry"])
    other = copy.deepcopy(level_candidate)
    other.update(candidate_id="renamed", parent_candidate_ids=["prior-1", "prior-2"])
    second = environment.canonical_design(other, level_config["task"]["geometry"])
    assert first["design_hash"] == second["design_hash"]
    other["parameters"]["alpha_deg"] = 3.0
    assert environment.canonical_design(other, level_config["task"]["geometry"])["design_hash"] != first["design_hash"]


def test_final_runtime_executes_all_required_stages(environment, level_config, level_candidate, synthetic_invoker, tmp_path):
    ToolRuntime = importlib.import_module("pipeline.agentloop.workflow").ToolRuntime
    invoke, calls = synthetic_invoker
    runtime = ToolRuntime(environment.resolve_task(level_config), invoke_fn=invoke, output_dir=tmp_path / "runtime")
    result = runtime.validate_final(level_candidate)
    assert [row["tool"] for row in calls] == ["geometry", "mesh", "solve", "post"]
    assert result["assessment"]["rank_eligible"] is True
    assert result["observations"][0]["source"] == "synthetic"
    assert result["cost"]["solver_calls"] == 1


def test_final_runtime_rejects_agent_options_before_any_stage(environment, level_config, level_candidate, synthetic_invoker, tmp_path):
    ToolRuntime = importlib.import_module("pipeline.agentloop.workflow").ToolRuntime
    invoke, calls = synthetic_invoker
    runtime = ToolRuntime(environment.resolve_task(level_config), invoke_fn=invoke, output_dir=tmp_path / "runtime")
    with pytest.raises(environment.ProtocolError):
        runtime.validate_final(level_candidate, options={"end_time": 1})
    assert calls == []


@pytest.mark.parametrize("alpha", [None, "", True, float("nan"), float("inf"), -5.01, 10.01, "2.0"])
def test_invalid_literal_parameter_never_passes_proposal(environment, level_config, level_candidate, alpha):
    level_candidate["parameters"]["alpha_deg"] = alpha
    with pytest.raises(environment.ProtocolError):
        environment.validate_proposal({"action": "evaluate_design", "candidate": level_candidate},
                                      environment.resolve_task(level_config))


def test_task_bounds_are_stricter_than_physical_regression_probe(environment, level_config):
    level_config["task"]["params"] = {"h_c": {"type": "float", "bounds": [0.05, 1.0], "default": 0.1}}
    candidate = {"representation": "parameterized", "parameters": {"h_c": 0.04}}
    with pytest.raises(environment.ProtocolError):
        environment.validate_proposal({"action": "evaluate_design", "candidate": candidate},
                                      environment.resolve_task(level_config))


@pytest.mark.parametrize("parameters", [{}, {"alpha_deg": 2.0, "h_c": 0.1}, {"alpha_deg": 2.0, "mesh_scale": 1.0}])
def test_missing_or_unknown_design_parameter_rejected(environment, level_config, parameters):
    candidate = {"representation": "parameterized", "parameters": parameters}
    with pytest.raises(environment.ProtocolError):
        environment.validate_proposal({"action": "evaluate_design", "candidate": candidate},
                                      environment.resolve_task(level_config))


@pytest.mark.parametrize("field,value", [
    ("score_formula", "100"), ("stage_id", "M1"), ("protocol_hash", "mine"),
    ("is_converged", True), ("is_formal", True), ("code", "__import__('os').system('id')"),
])
def test_host_owned_or_program_fields_rejected(environment, level_config, level_candidate, field, value):
    proposal = {"action": "evaluate_design", "candidate": level_candidate, field: value}
    with pytest.raises(environment.ProtocolError):
        environment.validate_proposal(proposal, environment.resolve_task(level_config))


@pytest.mark.parametrize("level,arm", [(0, "base"), (1, "a"), (1, "b"), (2, "a"), (3, "topk")])
def test_fixed_arms_cannot_use_exploration_numerics(environment, level_config, level_candidate, explicit_candidate, level, arm):
    level_config["experiment"].update(level=level, arm=arm)
    candidate = explicit_candidate if level >= 2 else level_candidate
    with pytest.raises(environment.ProtocolError):
        environment.validate_proposal({"action": "explore", "candidate": candidate,
                                       "requested_numerics": {"mesh_scale": 0.8}},
                                      environment.resolve_task(level_config))


@pytest.mark.parametrize("mesh_scale", [0.7, 1.0, 1.3])
def test_l2b_accepts_only_frozen_exploration_values(environment, level_config, explicit_candidate, mesh_scale):
    level_config["experiment"].update(level=2, arm="b")
    proposal = {"action": "explore", "candidate": explicit_candidate, "requested_numerics": {"mesh_scale": mesh_scale}}
    assert isinstance(environment.validate_proposal(proposal, environment.resolve_task(level_config)), dict)


@pytest.mark.parametrize("numerics", [
    {"mesh_scale": 0.69}, {"mesh_scale": 1.31}, {"mesh_scale": True},
    {"mesh_scale": float("nan")}, {"first_layer_height": 0.001},
    {"delta_t": 0.01}, {"warm_start": "parent-flow"}, {"model": "M1"},
])
def test_l2b_rejects_unsupported_or_outside_exploration(environment, level_config, explicit_candidate, numerics):
    level_config["experiment"].update(level=2, arm="b")
    with pytest.raises(environment.ProtocolError):
        environment.validate_proposal({"action": "explore", "candidate": explicit_candidate,
                                       "requested_numerics": numerics}, environment.resolve_task(level_config))


def test_evidence_refs_are_capabilities_not_arbitrary_history(environment, level_config, level_candidate):
    task = environment.resolve_task(level_config)
    proposal = {"action": "evaluate_design", "candidate": level_candidate, "evidence_refs": ["allowed-observation"]}
    assert isinstance(environment.validate_proposal(proposal, task, allowed_evidence={"allowed-observation"}), dict)
    for forbidden in ("nonexistent-observation", "parent-stage-secret"):
        proposal["evidence_refs"] = [forbidden]
        with pytest.raises(environment.ProtocolError):
            environment.validate_proposal(proposal, task, allowed_evidence={"allowed-observation"})


def test_canonical_identity_normalizes_integer_and_float(environment, level_config, level_candidate):
    first = environment.canonical_design(level_candidate, level_config["task"]["geometry"])
    level_candidate["parameters"]["alpha_deg"] = 2
    assert environment.canonical_design(level_candidate, level_config["task"]["geometry"])["design_hash"] == first["design_hash"]


def test_canonical_design_refuses_numeric_authority(environment, level_config, level_candidate):
    level_candidate["numerics"] = {"mesh_scale": 0.8}
    with pytest.raises(ValueError):
        environment.canonical_design(level_candidate, level_config["task"]["geometry"])


def test_new_schema_unknown_experiment_key_is_rejected(environment, level_config):
    level_config["experiment"]["bypass_validation"] = True
    with pytest.raises(environment.ConfigurationError):
        environment.resolve_task(level_config)


def test_insufficient_solve_budget_stops_before_geometry(environment, level_config, level_candidate, synthetic_invoker, tmp_path):
    ToolRuntime = importlib.import_module("pipeline.agentloop.workflow").ToolRuntime
    level_config["limits"]["solve_slots"] = 0
    invoke, calls = synthetic_invoker
    runtime = ToolRuntime(environment.resolve_task(level_config), invoke_fn=invoke, output_dir=tmp_path / "runtime")
    with pytest.raises(environment.BudgetError):
        runtime.validate_final(level_candidate)
    assert calls == []


@pytest.mark.parametrize("timeout_owner", ["environment", "limits"])
def test_deadline_terminates_worker_and_child_that_created_new_session(environment, level_config, level_candidate, tmp_path, timeout_owner):
    ToolRuntime = importlib.import_module("pipeline.agentloop.workflow").ToolRuntime
    level_config["environment"]["entrypoint"] = "test.agentloop.timeout_probe:invoke"
    level_config["environment"]["settings"]["timeout_s"] = 0.5 if timeout_owner == "environment" else 20
    if timeout_owner == "limits":
        level_config["limits"]["tool_timeout_s"] = 0.5
    runtime = ToolRuntime(environment.resolve_task(level_config), output_dir=tmp_path / "runtime")
    started = time.monotonic()
    try:
        try:
            result = runtime.validate_final(level_candidate)
        except (TimeoutError, environment.ProtocolError, environment.BudgetError):
            result = None
        assert time.monotonic() - started < 4.0
        pids = list(tmp_path.rglob("child.pid"))
        assert pids, "the probe must actually start a child in a new session"
        time.sleep(2.1)
        assert not list(tmp_path.rglob("late-artifact.txt")), "timed-out descendants must not write later"
        if result is not None:
            assert not result.get("assessment", {}).get("rank_eligible", False)
            assert result["cost"]["tool_seconds"] > 0
    finally:
        for path in tmp_path.rglob("*.pid"):
            try:
                os.kill(int(path.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_opt_in_cache_reuses_same_namespace_across_runtime_instances(
    environment, level_config, level_candidate, synthetic_invoker, tmp_path,
):
    ToolRuntime = importlib.import_module("pipeline.agentloop.workflow").ToolRuntime
    level_config["task"]["evaluation"]["allowed_sources"] = ["synthetic", "cache"]
    invoke, calls = synthetic_invoker
    task = environment.resolve_task(level_config)
    directory = tmp_path / "runtime"
    first = ToolRuntime(task, invoke_fn=invoke, output_dir=directory, cache=True).validate_final(level_candidate)
    count = len(calls)
    second = ToolRuntime(task, invoke_fn=invoke, output_dir=directory, cache=True).validate_final(level_candidate)
    assert len(calls) == count
    original, cached = first["observations"][0], second["observations"][0]
    assert cached["source"] == "cache"
    assert cached["original_observation_id"] == original["observation_id"]
    assert cached["original_cost"] == original["cost"]
    assert cached["observation_id"] != original["observation_id"]
    assert second["cost"]["solver_calls"] == 0
    assert second["assessment"]["rank_eligible"] is True


def test_cache_disabled_preserves_charged_real_repetition(environment, level_config, level_candidate, synthetic_invoker, tmp_path):
    ToolRuntime = importlib.import_module("pipeline.agentloop.workflow").ToolRuntime
    invoke, calls = synthetic_invoker
    task = environment.resolve_task(level_config)
    first = ToolRuntime(task, invoke_fn=invoke, output_dir=tmp_path / "runtime", cache=False).validate_final(level_candidate)
    second = ToolRuntime(task, invoke_fn=invoke, output_dir=tmp_path / "runtime", cache=False).validate_final(level_candidate)
    assert sum(row["tool"] == "solve" for row in calls) == 2
    assert first["cost"]["solver_calls"] == second["cost"]["solver_calls"] == 1


@pytest.mark.parametrize("change", ["version", "unknown_version", "arm", "stage", "settings", "namespace"])
def test_persistent_cache_never_crosses_evidence_identity(
    environment, level_config, level_candidate, synthetic_invoker, tmp_path, change,
):
    ToolRuntime = importlib.import_module("pipeline.agentloop.workflow").ToolRuntime
    level_config["task"]["evaluation"]["allowed_sources"] = ["synthetic", "cache"]
    if change == "stage":
        level_config["experiment"].update(level=1, arm="a")
    invoke, calls = synthetic_invoker
    directory = tmp_path / "runtime"
    ToolRuntime(environment.resolve_task(level_config), invoke_fn=invoke, output_dir=directory, cache=True).validate_final(level_candidate)
    changed = copy.deepcopy(level_config)
    if change == "version":
        changed["environment"]["version"] = "test-v2"
    elif change == "unknown_version":
        changed["environment"]["version"] = "unknown"
    elif change == "arm":
        changed["experiment"].update(level=1, arm="a")
    elif change == "stage":
        changed["experiment"]["stage_id"] = "M1"
        changed["environment"]["settings"]["model"] = "M1"
    elif change == "settings":
        changed["environment"]["settings"]["mesh"]["mesh_scale"] = 1.1
    else:
        directory = tmp_path / "other-arm-namespace"
    second = ToolRuntime(environment.resolve_task(changed), invoke_fn=invoke, output_dir=directory, cache=True).validate_final(level_candidate)
    assert sum(row["tool"] == "solve" for row in calls) == 2
    assert second["observations"][0]["source"] == "synthetic"
    assert second["cost"]["solver_calls"] == 1


def test_unknown_version_does_not_enable_cache_even_for_identical_repetition(
    environment, level_config, level_candidate, synthetic_invoker, tmp_path,
):
    ToolRuntime = importlib.import_module("pipeline.agentloop.workflow").ToolRuntime
    level_config["environment"]["version"] = "unknown"
    invoke, calls = synthetic_invoker
    task = environment.resolve_task(level_config)
    for _ in range(2):
        ToolRuntime(task, invoke_fn=invoke, output_dir=tmp_path / "runtime", cache=True).validate_final(level_candidate)
    assert sum(row["tool"] == "solve" for row in calls) == 2


def test_cached_source_is_retained_when_formal_protocol_disallows_it(
    environment, level_config, level_candidate, synthetic_invoker, tmp_path,
):
    ToolRuntime = importlib.import_module("pipeline.agentloop.workflow").ToolRuntime
    invoke, calls = synthetic_invoker
    task = environment.resolve_task(level_config)
    for index in range(2):
        result = ToolRuntime(task, invoke_fn=invoke, output_dir=tmp_path / "runtime", cache=True).validate_final(level_candidate)
    assert result["observations"][0]["source"] == "cache"
    assert result["assessment"]["rank_eligible"] is False


def test_legacy_configuration_keeps_experiment_none(environment, level_config):
    del level_config["experiment"]
    assert environment.resolve_task(level_config).experiment is None


@pytest.mark.parametrize("missing", ["objective", "params"])
def test_incomplete_new_task_definition_is_configuration_error(environment, level_config, missing):
    del level_config["task"][missing]
    with pytest.raises(environment.ConfigurationError):
        environment.resolve_task(level_config)


@pytest.mark.parametrize("change", ["version", "mesh"])
def test_protocol_changes_preserve_design_identity_but_change_observation_protocol(
    environment, level_config, level_candidate, synthetic_invoker, tmp_path, change,
):
    ToolRuntime = importlib.import_module("pipeline.agentloop.workflow").ToolRuntime
    invoke, _ = synthetic_invoker
    first = ToolRuntime(environment.resolve_task(level_config), invoke_fn=invoke,
                        output_dir=tmp_path / "first").validate_final(level_candidate)["observations"][0]
    if change == "version":
        level_config["environment"]["version"] = "changed-adapter-version"
    else:
        level_config["environment"]["settings"]["mesh"]["mesh_scale"] = 1.1
    second = ToolRuntime(environment.resolve_task(level_config), invoke_fn=invoke,
                         output_dir=tmp_path / "second").validate_final(level_candidate)["observations"][0]
    assert first["design_hash"] == second["design_hash"]
    assert first["protocol_hash"] != second["protocol_hash"]
    assert first["observation_id"] != second["observation_id"]


@pytest.mark.parametrize("failure", ["missing_artifact", "failed_status"])
def test_stage_failure_stops_downstream_even_when_function_returns_normally(
    environment, level_config, level_candidate, synthetic_invoker, tmp_path, failure,
):
    ToolRuntime = importlib.import_module("pipeline.agentloop.workflow").ToolRuntime
    invoke, calls = synthetic_invoker

    def broken_mesh(tool, inputs, context):
        result = invoke(tool, inputs, context)
        if tool == "mesh":
            if failure == "missing_artifact":
                from pathlib import Path
                Path(result["artifacts"]["mesh"]).unlink()
            else:
                result["status"] = "failed"
        return result

    runtime = ToolRuntime(environment.resolve_task(level_config), invoke_fn=broken_mesh, output_dir=tmp_path / "runtime")
    try:
        result = runtime.validate_final(level_candidate)
    except environment.ProtocolError:
        result = None
    assert [row["tool"] for row in calls] == ["geometry", "mesh"]
    if result is not None:
        assert not result["assessment"]["rank_eligible"]
        assert result["cost"]["tool_seconds"] > 0


def test_explicit_shape_signed_zero_has_one_physical_design_identity(environment, level_config, explicit_candidate):
    positive = copy.deepcopy(explicit_candidate)
    negative = copy.deepcopy(explicit_candidate)
    negative["shape"]["lower"][0] = -0.0
    negative["shape"]["lower"][-1] = -0.0
    negative["shape"]["x"][0] = -0.0
    first = environment.canonical_design(positive, level_config["task"]["geometry"])
    second = environment.canonical_design(negative, level_config["task"]["geometry"])
    assert first["design_hash"] == second["design_hash"]


def test_signed_zero_canonicalization_does_not_round_nonzero_shape_changes(environment, level_config, explicit_candidate):
    other = copy.deepcopy(explicit_candidate)
    other["shape"]["upper"][2] += 1e-10
    first = environment.canonical_design(explicit_candidate, level_config["task"]["geometry"])
    second = environment.canonical_design(other, level_config["task"]["geometry"])
    assert first["design_hash"] != second["design_hash"]
