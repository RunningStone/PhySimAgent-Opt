"""Unit contract for serial tools, candidates, budgets, and termination."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import time

import pytest

from test.exp.agentloop_fixtures import (
    AuditedWorkflowEnvironment,
    DETERMINISTIC_ENV_YAML,
    RecordingEnvironment,
    error_kind,
    value,
)


PARAMS = {"h_c": 0.2, "alpha_deg": 3.0, "camber": 0.04}
OPTIONS = {"tier": 0}


def _agent_mapping(mapping):
    updated = copy.deepcopy(mapping)
    updated["workflow"]["mode"] = "agent"
    return updated


def test_fixed_mode_executes_yaml_sequence_exactly(make_runtime, fake_environment):
    result = make_runtime().run_candidate(PARAMS, OPTIONS)
    assert [event["tool"] for event in value(result, "tool_calls")] == [
        "simulate",
        "evaluate",
    ]
    assert value(result, "eval_json_count") == 1
    assert value(result, "journal_entries") == 1
    assert [call["tool"] for call in fake_environment.calls] == ["simulate", "evaluate"]
    candidate_ids = {call["context"]["candidate_id"] for call in fake_environment.calls}
    assert len(candidate_ids) == 1


@pytest.mark.parametrize(
    ("candidate_options", "explicit_args", "expected_tier"),
    [
        ({}, {}, 0),
        ({"tier": 2}, {}, 2),
        ({"tier": 2}, {"tier": 0}, 0),
    ],
)
def test_option_priority_is_default_then_candidate_then_approved_tool_arg(
    interface_contract,
    make_spec,
    wing_spec_mapping,
    fake_environment,
    candidate_options,
    explicit_args,
    expected_tier,
):
    mapping = _agent_mapping(wing_spec_mapping)
    mapping["tools"]["simulate"]["option_overrides"] = ["tier"]
    mapping["tools"]["simulate"]["args"] = {
        "probe": {"type": "string", "required": False}
    }
    task = make_spec(mapping)
    environment = RecordingEnvironment(copy.deepcopy(fake_environment.responses))
    calls = iter(
        [
            {
                "name": "simulate",
                "args": {"params": PARAMS, "probe": "kept-as-tool-arg", **explicit_args},
            },
            {"name": "evaluate", "args": {}},
        ]
    )
    result = interface_contract.runtime(task, environment).run_candidate(
        PARAMS,
        candidate_options,
        choose_next=lambda observations, allowed_tools, limits: next(calls),
    )
    received = environment.calls[0]["inputs"]
    assert received["options"] == candidate_options
    assert received["args"]["probe"] == "kept-as-tool-arg"
    assert "probe" not in received["options"]
    if explicit_args:
        assert received["args"]["tier"] == expected_tier
    else:
        assert "tier" not in received["args"]
    assert value(result, "options")["tier"] == expected_tier


def test_unapproved_tool_arg_cannot_override_candidate_option(
    interface_contract, make_spec, wing_spec_mapping
):
    task = make_spec(wing_spec_mapping)
    state = {
        "candidate_id": "candidate-options",
        "problem_hash": task.problem_hash,
        "artifacts": {},
    }
    with pytest.raises(Exception) as caught:
        interface_contract.validate_tool_call(
            {"name": "simulate", "args": {"params": PARAMS, "tier": 2}}, task, state
        )
    assert error_kind(caught.value) == "protocol_error"


@pytest.mark.parametrize(
    "call_args",
    [
        {"params": {**PARAMS, "h_c": 0.4}},
        {"options": {"tier": 2}},
    ],
)
def test_candidate_internal_call_cannot_replace_sealed_params_or_options(
    interface_contract,
    make_spec,
    wing_spec_mapping,
    fake_environment,
    call_args,
):
    task = make_spec(_agent_mapping(wing_spec_mapping))
    environment = RecordingEnvironment(copy.deepcopy(fake_environment.responses))
    result = interface_contract.runtime(
        task, environment, limits={"total_llm": 1, "total_tools": 1}
    ).run_candidate(
        PARAMS,
        OPTIONS,
        choose_next=lambda observations, allowed_tools, limits: {
            "name": "simulate",
            "args": call_args,
        },
    )
    assert value(result, "accepted") is False
    assert value(result, "failure")["kind"] == "protocol_error"
    assert environment.calls == []


def test_candidate_internal_call_may_repeat_identical_sealed_params_and_options(
    interface_contract, make_spec, wing_spec_mapping, fake_environment
):
    task = make_spec(_agent_mapping(wing_spec_mapping))
    environment = RecordingEnvironment(copy.deepcopy(fake_environment.responses))
    result = interface_contract.runtime(
        task, environment, limits={"max_tools_per_candidate": 1, "total_tools": 1}
    ).run_candidate(
        PARAMS,
        OPTIONS,
        choose_next=lambda observations, allowed_tools, limits: {
            "name": "simulate",
            "args": {"params": PARAMS, "options": OPTIONS},
        },
    )
    assert environment.calls
    failure = value(result, "failure")
    assert failure is None or failure.get("kind") != "protocol_error"


@pytest.mark.parametrize(
    ("trusted", "expected_second"), [(True, "evaluate"), (False, "diagnose")]
)
def test_agent_chooser_receives_new_observation_and_changes_next_tool(
    make_runtime, fake_environment, wing_spec_mapping, trusted, expected_second
):
    fake_environment.responses["simulate"]["trust"]["converged"] = trusted
    seen = []

    def choose_next(observations, allowed_tools, limits):
        seen.append(copy.deepcopy(observations))
        if not observations:
            return {"name": "simulate", "args": {"params": PARAMS}}
        assert observations[-1]["tool"] == "simulate"
        assert observations[-1]["result"]["trust"]["converged"] is trusted
        return {"name": expected_second, "args": {}}

    result = make_runtime(spec_mapping=_agent_mapping(wing_spec_mapping)).run_candidate(
        PARAMS, OPTIONS, choose_next=choose_next
    )
    assert [event["tool"] for event in value(result, "tool_calls")][:2] == [
        "simulate",
        expected_second,
    ]
    assert seen[0] == []
    assert len(seen[1]) == 1


def test_multiple_internal_calls_create_one_outer_candidate_and_one_journal_entry(
    make_runtime, wing_spec_mapping
):
    choices = iter(
        [
            {"name": "simulate", "args": {"params": PARAMS}},
            {"name": "diagnose", "args": {}},
            {"name": "evaluate", "args": {}},
        ]
    )
    result = make_runtime(spec_mapping=_agent_mapping(wing_spec_mapping)).run_candidate(
        PARAMS,
        OPTIONS,
        choose_next=lambda observations, allowed_tools, limits: next(choices),
    )
    assert value(result, "node_count") == 1
    assert value(result, "eval_json_count") == 1
    assert value(result, "journal_entries") == 1


def test_diagnosis_without_final_evaluation_is_failed_not_bad_design(
    make_runtime, wing_spec_mapping
):
    choices = iter(
        [
            {"name": "simulate", "args": {"params": PARAMS}},
            {"name": "diagnose", "args": {}},
        ]
    )
    result = make_runtime(
        spec_mapping=_agent_mapping(wing_spec_mapping),
        limits={"max_tools_per_candidate": 2, "total_tools": 2},
    ).run_candidate(
        PARAMS,
        OPTIONS,
        choose_next=lambda observations, allowed_tools, limits: next(choices),
    )
    assert value(result, "metric") is None
    assert value(result, "status") == "failed"
    assert value(result, "stop_reason") == "max_tools_per_candidate"
    assert value(result, "journal_entries") == 1
    assert value(result, "is_bad_design") is False


def test_missing_required_artifact_is_rejected_before_environment_invoke(
    interface_contract, make_spec, wing_spec_mapping, fake_environment
):
    task = make_spec(wing_spec_mapping)
    state = {"candidate_id": "generated-1", "problem_hash": task.problem_hash, "artifacts": {}}
    with pytest.raises(Exception) as caught:
        interface_contract.validate_tool_call(
            {"name": "evaluate", "args": {}}, task, state
        )
    assert error_kind(caught.value) == "dependency_error"
    assert fake_environment.calls == []


def test_recompute_invalidation_rejects_old_solution_during_candidate(
    make_runtime, wing_spec_mapping, fake_environment
):
    choices = iter(
        [
            {"name": "simulate", "args": {"params": PARAMS}},
            {"name": "remesh", "args": {}},
            {"name": "evaluate", "args": {}},
        ]
    )
    result = make_runtime(spec_mapping=_agent_mapping(wing_spec_mapping)).run_candidate(
        PARAMS,
        OPTIONS,
        choose_next=lambda observations, allowed_tools, limits: next(choices),
    )
    assert [call["tool"] for call in fake_environment.calls] == ["simulate", "remesh"]
    assert value(result, "status") == "failed"
    assert value(result, "metric") is None
    assert value(result, "failure")["kind"] == "dependency_error"


@pytest.mark.parametrize(
    "handle",
    [
        {
            "name": "solution",
            "candidate_id": "other-candidate",
            "problem_hash": "sha256:this-problem",
        },
        {
            "name": "solution",
            "candidate_id": "generated-1",
            "problem_hash": "sha256:other-problem",
        },
    ],
)
def test_cross_candidate_or_problem_artifact_handle_is_rejected(
    interface_contract, make_spec, wing_spec_mapping, handle
):
    task = make_spec(wing_spec_mapping)
    state = {
        "candidate_id": "generated-1",
        "problem_hash": "sha256:this-problem",
        "artifacts": {"solution": handle},
    }
    with pytest.raises(Exception) as caught:
        interface_contract.validate_tool_call(
            {"name": "evaluate", "args": {"solution": handle}}, task, state
        )
    assert error_kind(caught.value) == "dependency_error"


def test_validated_same_problem_warm_start_is_explicit_exception(
    interface_contract, make_spec, wing_spec_mapping
):
    task = make_spec(wing_spec_mapping)
    state = {
        "candidate_id": "generated-2",
        "problem_hash": task.problem_hash,
        "artifacts": {},
        "validated_warm_starts": {
            "archived-case-7": {"problem_hash": task.problem_hash, "validated": True}
        },
    }
    call = interface_contract.validate_tool_call(
        {
            "name": "simulate",
            "args": {"params": PARAMS, "warm_start_from": "archived-case-7"},
        },
        task,
        state,
    )
    assert call["args"]["warm_start_from"] == "archived-case-7"


@pytest.mark.parametrize(
    ("problem_hash", "validated"),
    [("sha256:other-problem", True), (None, False)],
)
def test_unvalidated_or_other_problem_warm_start_is_rejected(
    interface_contract, make_spec, wing_spec_mapping, problem_hash, validated
):
    task = make_spec(wing_spec_mapping)
    state = {
        "candidate_id": "generated-2",
        "problem_hash": task.problem_hash,
        "artifacts": {},
        "validated_warm_starts": {
            "archived-case-7": {
                "problem_hash": problem_hash or task.problem_hash,
                "validated": validated,
            }
        },
    }
    with pytest.raises(Exception) as caught:
        interface_contract.validate_tool_call(
            {
                "name": "simulate",
                "args": {"params": PARAMS, "warm_start_from": "archived-case-7"},
            },
            task,
            state,
        )
    assert error_kind(caught.value) == "dependency_error"


def test_repeated_diagnosis_stops_at_exact_per_candidate_limit(
    make_runtime, wing_spec_mapping
):
    def choose_next(observations, allowed_tools, limits):
        if not observations:
            return {"name": "simulate", "args": {"params": PARAMS}}
        return {"name": "diagnose", "args": {}}

    result = make_runtime(
        spec_mapping=_agent_mapping(wing_spec_mapping),
        limits={"max_tools_per_candidate": 3, "total_tools": 10},
    ).run_candidate(PARAMS, OPTIONS, choose_next=choose_next)
    assert len(value(result, "tool_calls")) == 3
    assert value(result, "stop_reason") == "max_tools_per_candidate"
    assert value(result, "metric") is None


def test_total_tool_budget_is_reserved_before_execution(make_runtime, fake_environment):
    runtime = make_runtime(limits={"max_tools_per_candidate": 4, "total_tools": 1})
    runtime.invoke("simulate", {"params": PARAMS})
    with pytest.raises(Exception) as caught:
        runtime.invoke("diagnose", {})
    assert error_kind(caught.value) == "budget_error"
    assert [call["tool"] for call in fake_environment.calls] == ["simulate"]
    assert value(runtime, "ledger")["tool_queries"] == 1


def test_cache_hit_still_counts_query_but_reports_actual_time(make_runtime):
    runtime = make_runtime()
    first = runtime.invoke("simulate", {"params": PARAMS})
    second = runtime.invoke("simulate", {"params": PARAMS})
    assert value(first, "cache_hit") is False
    assert value(second, "cache_hit") is True
    assert value(runtime, "ledger")["tool_queries"] == 2
    assert value(runtime, "ledger")["tool_executions"] == 1
    assert value(runtime, "ledger")["tool_seconds"] >= 0.0


def test_cached_candidate_rebinds_artifacts_conditions_and_provenance(
    interface_contract, make_spec, wing_spec_mapping
):
    mapping = copy.deepcopy(wing_spec_mapping)
    mapping["limits"]["total_tools"] = 5
    mapping["limits"]["solve_slots"] = 4
    environment = AuditedWorkflowEnvironment()
    runtime = interface_contract.runtime(make_spec(mapping), environment)
    first = runtime.run_candidate(PARAMS, OPTIONS)
    second = runtime.run_candidate(PARAMS, OPTIONS)
    assert value(first, "accepted") is True
    assert value(second, "accepted") is True
    calls = value(second, "tool_calls")
    assert [call["tool"] for call in calls] == ["simulate", "evaluate"]
    assert all(call["result"]["cache_hit"] is True for call in calls)
    candidate_ids = {
        call["result"]["provenance"]["candidate_id"] for call in calls
    }
    assert len(candidate_ids) == 1
    current_candidate = candidate_ids.pop()
    first_candidate = value(first, "tool_calls")[0]["result"]["provenance"][
        "candidate_id"
    ]
    assert current_candidate != first_candidate
    solution_id = calls[0]["result"]["artifact_handles"]["solution"]["artifact_id"]
    evaluation = calls[1]["result"]
    assert evaluation["provenance"]["call_id"] == calls[1]["result_id"]
    assert evaluation["provenance"]["input_artifact_ids"] == [solution_id]
    assert all(
        artifact["candidate_id"] == current_candidate
        for call in calls
        for artifact in call["result"]["artifacts"]
    )
    current_ids = {
        handle["artifact_id"]
        for call in calls
        for handle in call["result"]["artifact_handles"].values()
    }
    assert set(evaluation["conditions"]["design"]["artifact_ids"]) <= current_ids


@pytest.mark.parametrize(
    ("condition_uses_required_input", "accepted"), [(True, False), (False, True)]
)
def test_terminal_invalidation_rejects_evidence_from_invalidated_required_artifact(
    interface_contract,
    make_spec,
    wing_spec_mapping,
    condition_uses_required_input,
    accepted,
):
    mapping = copy.deepcopy(wing_spec_mapping)
    mapping["tools"]["evaluate"]["invalidates"] = ["solution"]
    environment = AuditedWorkflowEnvironment(
        condition_uses_required_input=condition_uses_required_input
    )
    result = interface_contract.runtime(make_spec(mapping), environment).run_candidate(
        PARAMS, OPTIONS
    )
    assert value(result, "accepted") is accepted
    if accepted:
        assert value(result, "metric") is not None
    else:
        assert value(result, "metric") is None
        assert value(result, "failure")["kind"] == "dependency_error"


def test_failed_tool_call_consumes_query_and_solve_slot(
    interface_contract, make_spec, wing_spec_mapping
):
    environment = RecordingEnvironment({"simulate": RuntimeError("solver failed")})
    runtime = interface_contract.runtime(make_spec(wing_spec_mapping), environment)
    observed = runtime.invoke("simulate", {"params": PARAMS})
    assert value(observed, "failure") is not None
    assert value(runtime, "ledger")["tool_queries"] == 1
    assert value(runtime, "ledger")["solve_slots"] == 1


def test_chooser_format_retry_consumes_llm_budget(make_runtime, wing_spec_mapping):
    calls = 0

    def choose_next(observations, allowed_tools, limits):
        nonlocal calls
        calls += 1
        if calls == 1:
            return "not a tool call"
        return {"name": "simulate", "args": {"params": PARAMS}}

    result = make_runtime(
        spec_mapping=_agent_mapping(wing_spec_mapping),
        limits={"max_tools_per_candidate": 1, "total_tools": 1, "total_llm": 2},
    ).run_candidate(PARAMS, OPTIONS, choose_next=choose_next)
    assert value(result, "llm_calls") == 2
    assert value(result, "stop_reason") == "max_tools_per_candidate"


def test_design_agent_chooser_hides_conflicting_design_response_format_only(
    make_design_agent
):
    task_desc = """## Problem
TASK_CONTENT_SENTINEL: optimise the public objective.

## Physics
PHYSICS_CONTENT_SENTINEL: preserve this mechanism description.

## Response format (mandatory)
DESIGN_FORMAT_SENTINEL: return planning JSON and fenced PARAMS.
"""
    agent = make_design_agent(task_desc)
    chooser_prompts = []

    def choose_backend(prompt):
        chooser_prompts.append(copy.deepcopy(prompt))
        return json.dumps({"name": "evaluate", "args": {}})

    agent.backend = choose_backend
    agent.choose_tool([], ["evaluate"], {"max_tools_per_candidate": 1})

    class CapturedOuterPrompt(Exception):
        pass

    outer_prompts = []

    def capture_outer(prompt):
        outer_prompts.append(copy.deepcopy(prompt))
        raise CapturedOuterPrompt

    agent.backend = capture_outer
    with pytest.raises(CapturedOuterPrompt):
        agent._ask(None, "draft")

    assert outer_prompts[0]["Task description"] == task_desc
    chooser_task = chooser_prompts[0]["Task description"]
    assert "TASK_CONTENT_SENTINEL" in chooser_task
    assert "PHYSICS_CONTENT_SENTINEL" in chooser_task
    assert "## Response format" not in chooser_task
    assert "DESIGN_FORMAT_SENTINEL" not in chooser_task


def test_design_agent_passes_structured_chooser_prompt_to_aide_backend(
    make_design_agent, monkeypatch
):
    import aide.agent
    import aide.backend

    captured = []

    def query(system_message, user_message, model, **kwargs):
        captured.append(
            {
                "system_message": copy.deepcopy(system_message),
                "user_message": copy.deepcopy(user_message),
                "model": model,
            }
        )
        return json.dumps({"name": "evaluate", "args": {}})

    monkeypatch.setattr(aide.backend, "query", query)
    monkeypatch.setattr(aide.agent, "query", query)
    agent = make_design_agent(
        "## Problem\nPreserve structured sections.\n\n"
        "## Response format (mandatory)\nReturn fenced PARAMS.\n"
    )
    agent.choose_tool([], ["evaluate"], {"max_tools_per_candidate": 1})
    assert captured
    assert isinstance(captured[0]["system_message"], dict)


def test_validation_reserve_cannot_be_spent_on_candidate_work(
    make_runtime, fake_environment, valid_wing_result
):
    runtime = make_runtime(
        limits={"max_tools_per_candidate": 4, "total_tools": 3, "validation_reserve": 1}
    )
    runtime.invoke("simulate", {"params": PARAMS})
    runtime.invoke("diagnose", {})
    with pytest.raises(Exception) as caught:
        runtime.invoke("evaluate", {})
    assert error_kind(caught.value) == "budget_error"
    fake_environment.responses["evaluate"] = valid_wing_result
    verification = runtime.validate_final(PARAMS, OPTIONS)
    assert value(verification, "failure") is None


def test_wall_clock_timeout_terminates_spawned_process_group(
    interface_contract, make_spec, wing_spec_mapping, tmp_path
):
    marker = tmp_path / "child-finished"

    def slow_tool(inputs, context):
        child = subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import pathlib,time; time.sleep(2); "
                    f"pathlib.Path({str(marker)!r}).write_text('orphan')"
                ),
            ],
            start_new_session=False,
        )
        time.sleep(5)
        return {"child_pid": child.pid}

    environment = RecordingEnvironment({"simulate": slow_tool})
    runtime = interface_contract.runtime(
        make_spec(wing_spec_mapping),
        environment,
        limits={"total_tools": 1, "wall_clock_s": 0.2},
    )
    observed = runtime.invoke("simulate", {"params": PARAMS})
    assert value(observed, "failure")["kind"] == "timeout"
    time.sleep(2.2)
    assert not marker.exists(), "timed-out tool left a live child process"


def _process_is_running(pid):
    completed = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)],
        capture_output=True,
        text=True,
        check=False,
    )
    state = completed.stdout.strip()
    return completed.returncode == 0 and bool(state) and not state.startswith("Z")


def test_real_yaml_entrypoint_executes_outside_the_host_process(interface_contract):
    task = interface_contract.resolve_yaml(DETERMINISTIC_ENV_YAML)
    observed = interface_contract.ToolRuntime(task).invoke("observe_worker", {})
    assert value(observed, "failure") is None
    assert value(observed, "worker_pid") != os.getpid()


def test_real_yaml_entrypoint_timeout_leaves_no_worker_or_child_process(
    interface_contract, tmp_path
):
    worker_pid_path = tmp_path / "worker.pid"
    child_pid_path = tmp_path / "child.pid"
    marker = tmp_path / "child-finished"
    task = interface_contract.resolve_yaml(DETERMINISTIC_ENV_YAML)
    runtime = interface_contract.ToolRuntime(
        task, limits={"total_tools": 1, "solve_slots": 1, "wall_clock_s": 0.3}
    )
    observed = runtime.invoke(
        "timeout_tree",
        {
            "worker_pid_path": str(worker_pid_path),
            "child_pid_path": str(child_pid_path),
            "marker_path": str(marker),
            "sleep_s": 10.0,
        },
    )
    assert value(observed, "failure")["kind"] == "timeout"
    assert worker_pid_path.is_file() and child_pid_path.is_file()
    worker_pid = int(worker_pid_path.read_text(encoding="ascii"))
    child_pid = int(child_pid_path.read_text(encoding="ascii"))
    assert worker_pid != os.getpid()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and (
        _process_is_running(worker_pid) or _process_is_running(child_pid)
    ):
        time.sleep(0.02)
    assert not _process_is_running(worker_pid)
    assert not _process_is_running(child_pid)
    time.sleep(1.1)
    assert not marker.exists(), "timed-out worker left a child able to finish"
