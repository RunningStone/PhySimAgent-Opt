"""Unit contract for TaskSpec, design input, calls, and final evaluation."""

from __future__ import annotations

import copy
import importlib
import math

import pytest

from test.exp.agentloop_fixtures import DETERMINISTIC_ENV_YAML, RecordingEnvironment, error_kind, value


def test_r2_public_environment_interface_is_exported(interface_contract):
    api = interface_contract.module
    assert api.TaskSpec is not None
    assert api.ToolRuntime is not None
    assert callable(api.resolve_task)
    assert callable(api.parse_design)
    assert callable(api.validate_tool_call)
    assert callable(api.normalize_evaluation)
    for method in ("run_candidate", "invoke", "validate_final"):
        assert callable(getattr(api.ToolRuntime, method))


def test_valid_merged_mapping_and_yaml_have_same_problem_hash(
    interface_contract, make_spec, wing_spec_mapping, write_task_yaml
):
    from_mapping = make_spec(wing_spec_mapping)
    from_yaml = interface_contract.resolve_yaml(write_task_yaml(wing_spec_mapping))
    assert value(from_mapping, "problem_hash") == value(from_yaml, "problem_hash")


def test_mode_does_not_change_physical_problem_hash(make_spec, wing_spec_mapping):
    fixed = make_spec(wing_spec_mapping)
    agent_mapping = copy.deepcopy(wing_spec_mapping)
    agent_mapping["workflow"]["mode"] = "agent"
    agent = make_spec(agent_mapping)
    assert value(fixed, "problem_hash") == value(agent, "problem_hash")


@pytest.mark.parametrize("mutation", ["constraint", "data", "tools"])
def test_physical_definition_changes_problem_hash(
    make_spec, wing_spec_mapping, mutation
):
    original = make_spec(wing_spec_mapping)
    changed_mapping = copy.deepcopy(wing_spec_mapping)
    if mutation == "constraint":
        changed_mapping["task"]["constraints"][0]["threshold"] = 0.07
    elif mutation == "data":
        changed_mapping["environment"]["data"]["version"] = "sha256:changed"
    else:
        changed_mapping["tools"]["simulate"]["produces"] = ["other"]
    changed = make_spec(changed_mapping)
    assert value(original, "problem_hash") != value(changed, "problem_hash")


@pytest.mark.parametrize("unknown", ["mystery", "unsupported_case", "undeclared_branch"])
def test_unknown_top_level_configuration_field_is_rejected(
    interface_contract, wing_spec_mapping, unknown
):
    mapping = copy.deepcopy(wing_spec_mapping)
    mapping[unknown] = True
    with pytest.raises(Exception) as caught:
        interface_contract.resolve_task(mapping)
    assert error_kind(caught.value) == "configuration_error"


def test_unknown_environment_entrypoint_is_rejected_before_execution(
    interface_contract, wing_spec_mapping
):
    mapping = copy.deepcopy(wing_spec_mapping)
    mapping["environment"]["entrypoint"] = "not-registered:environment"
    with pytest.raises(Exception) as caught:
        interface_contract.resolve_task(mapping)
    assert error_kind(caught.value) == "configuration_error"


def test_new_importable_environment_module_and_yaml_run_without_registration(
    interface_contract,
):
    task = interface_contract.resolve_yaml(DETERMINISTIC_ENV_YAML)
    runtime = interface_contract.ToolRuntime(task)
    result = runtime.run_candidate({"x": 0.25}, {"method": "alternate"})
    assert value(result, "accepted") is True
    assert value(result, "metric") == pytest.approx(0.25)


@pytest.mark.parametrize(
    "entrypoint",
    [
        "test.exp.deterministic_environment",
        "test.exp.does_not_exist:invoke",
        "test.exp.deterministic_environment:NON_CALLABLE",
    ],
)
def test_malformed_unimportable_or_noncallable_entrypoint_is_rejected_preexecution(
    interface_contract, entrypoint
):
    fixture_module = importlib.import_module(
        "test.exp.deterministic_environment"
    )
    fixture_module.INVOCATION_COUNT = 0
    mapping = interface_contract.driver.load_config(DETERMINISTIC_ENV_YAML)
    mapping["environment"]["entrypoint"] = entrypoint
    with pytest.raises(Exception) as caught:
        interface_contract.resolve_task(mapping)
    assert error_kind(caught.value) == "configuration_error"
    assert fixture_module.INVOCATION_COUNT == 0


@pytest.mark.parametrize("invalid_bound", [True, math.nan, math.inf, -math.inf])
def test_yaml_numeric_bounds_reject_bool_and_nonfinite_values(
    interface_contract, wing_spec_mapping, write_task_yaml, invalid_bound
):
    mapping = copy.deepcopy(wing_spec_mapping)
    mapping["task"]["params"]["h_c"]["bounds"][0] = invalid_bound
    with pytest.raises(Exception) as caught:
        interface_contract.resolve_yaml(write_task_yaml(mapping))
    assert error_kind(caught.value) == "configuration_error"


def test_float_design_parameter_must_be_bounded(interface_contract, wing_spec_mapping):
    mapping = copy.deepcopy(wing_spec_mapping)
    del mapping["task"]["params"]["h_c"]["bounds"]
    with pytest.raises(Exception) as caught:
        interface_contract.resolve_task(mapping)
    assert error_kind(caught.value) == "configuration_error"


@pytest.mark.parametrize(
    ("tool", "field", "names"),
    [
        ("simulate", "produces", ["../solution"]),
        ("simulate", "requires", ["undeclared-artifact"]),
        ("simulate", "invalidates", ["undeclared-artifact"]),
    ],
)
def test_requires_produces_invalidates_use_declared_finite_names(
    interface_contract, wing_spec_mapping, tool, field, names
):
    mapping = copy.deepcopy(wing_spec_mapping)
    mapping["tools"][tool][field] = names
    with pytest.raises(Exception) as caught:
        interface_contract.resolve_task(mapping)
    assert error_kind(caught.value) == "configuration_error"


@pytest.mark.parametrize(
    "code",
    [
        'PARAMS = {"h_c": __import__("os").system("true")}',
        "import os\nPARAMS = {}",
        "PARAMS = dict(h_c=0.2)",
        "X = 1\nPARAMS = {}",
        "PARAMS = {k: 1 for k in ['h_c']}",
        "PARAMS = {'h_c': (lambda: 0.2)()}",
    ],
)
def test_parse_design_rejects_non_literal_or_extra_statements(
    interface_contract, make_spec, wing_spec_mapping, code
):
    with pytest.raises(Exception) as caught:
        interface_contract.parse_design(code, make_spec(wing_spec_mapping))
    assert error_kind(caught.value) == "protocol_error"


@pytest.mark.parametrize(
    "code",
    [
        'PARAMS = {"h_c": True, "alpha_deg": 3.0, "camber": 0.04}',
        'PARAMS = {"h_c": 1e999, "alpha_deg": 3.0, "camber": 0.04}',
        'PARAMS = {"h_c": float("nan"), "alpha_deg": 3.0, "camber": 0.04}',
        'PARAMS = {"h_c": 0.2, "alpha_deg": 3.0, "camber": 0.04, "extra": 1.0}',
        'PARAMS = {"h_c": 0.2, "alpha_deg": 3.0, "camber": 0.04}\nOPTIONS = {"unknown": 1}',
    ],
)
def test_parse_design_rejects_bool_nonfinite_and_unknown_keys(
    interface_contract, make_spec, wing_spec_mapping, code
):
    with pytest.raises(Exception) as caught:
        interface_contract.parse_design(code, make_spec(wing_spec_mapping))
    assert error_kind(caught.value) == "protocol_error"


def test_parse_design_returns_declared_params_and_options_tuple(
    interface_contract, make_spec, wing_spec_mapping
):
    params, options = interface_contract.parse_design(
        'PARAMS = {"h_c": 0.2, "alpha_deg": 3.0, "camber": 0.04}\n'
        'OPTIONS = {"tier": 0}',
        make_spec(wing_spec_mapping),
    )
    assert params == {"h_c": 0.2, "alpha_deg": 3.0, "camber": 0.04}
    assert options == {"tier": 0}


def test_validate_tool_call_allows_only_declared_name_and_args(
    interface_contract, make_spec, wing_spec_mapping
):
    task = make_spec(wing_spec_mapping)
    state = {"candidate_id": "candidate-1", "artifacts": {}}
    with pytest.raises(Exception) as caught:
        interface_contract.validate_tool_call(
            {"name": "/tmp/arbitrary-script.py", "args": {}}, task, state
        )
    assert error_kind(caught.value) == "protocol_error"


def test_out_of_bounds_reaches_environment_and_returns_original_typed_failure(
    interface_contract, make_spec, wing_spec_mapping
):
    failure = {
        "case_id": "case-out-of-bounds",
        "failure": {"kind": "domain_failure", "field": "h_c"},
        "counted": True,
        "native_solver_invocations": 0,
    }
    environment = RecordingEnvironment({"simulate": failure})
    runtime = interface_contract.runtime(
        make_spec(wing_spec_mapping), environment, limits={"total_tools": 1}
    )
    observed = runtime.invoke(
        "simulate", {"params": {"h_c": 2.0, "alpha_deg": 3.0, "camber": 0.04}}
    )
    assert value(observed, "failure")["kind"] == "domain_failure"
    assert value(observed, "case_id") == "case-out-of-bounds"
    assert value(observed, "counted") is True
    assert value(observed, "native_solver_invocations") == 0
    assert len(environment.calls) == 1


def _audited_fixture_result(context):
    artifact_id = context["output_artifact_ids"]["evaluation"]
    return {
        "case_id": f"{context['candidate_id']}:{context['call_id']}",
        "input_hash": context["design_hash"],
        "status": "ok",
        "value": 0.5,
        "limit": 0.5,
        "fidelity": "fixture",
        "trust": {"complete": True},
        "conditions": {
            "fixture": {
                "status": "ok",
                "fidelity": "fixture",
                "artifact_ids": [artifact_id],
            }
        },
        "conditions_completed": ["fixture"],
        "artifacts": [
            {
                "name": "evaluation",
                "artifact_id": artifact_id,
                "candidate_id": context["candidate_id"],
                "call_id": context["call_id"],
                "problem_hash": context["problem_hash"],
            }
        ],
        "provenance": {
            key: context[key]
            for key in (
                "run_id",
                "candidate_id",
                "call_id",
                "problem_hash",
                "design_hash",
                "input_artifact_ids",
            )
        },
        "versions": {"fixture": "1"},
        "summary": "audited deterministic result",
        "failure": None,
    }


@pytest.mark.parametrize(
    "missing",
    [
        "provenance",
        "versions",
        "artifacts",
        "status_failed",
        "run_id",
        "candidate_id",
        "call_id",
        "problem_hash",
        "design_hash",
        "input_artifact_ids",
    ],
)
def test_nonlegacy_terminal_result_missing_audit_identity_is_not_accepted(
    interface_contract, missing
):
    task = interface_contract.resolve_yaml(DETERMINISTIC_ENV_YAML)

    def incomplete_result(inputs, context):
        raw = _audited_fixture_result(context)
        if missing == "status_failed":
            raw["status"] = "failed"
        elif missing in {"provenance", "versions", "artifacts"}:
            del raw[missing]
        else:
            del raw["provenance"][missing]
        return raw

    environment = RecordingEnvironment({"evaluate": incomplete_result})
    result = interface_contract.runtime(task, environment).run_candidate(
        {"x": 0.5}, {"method": "default"}
    )
    assert value(result, "accepted") is False
    assert value(result, "metric") is None


@pytest.mark.parametrize(
    ("params", "options"),
    [
        ({"x": -0.01}, {"method": "default"}),
        ({"x": 1.01}, {"method": "default"}),
        ({"x": math.nan}, {"method": "default"}),
        ({"x": True}, {"method": "default"}),
        ({"x": 0.5, "unknown": 1.0}, {"method": "default"}),
        ({"x": 0.5}, {"method": "default", "unknown": "value"}),
    ],
)
def test_nonlegacy_run_candidate_rejects_invalid_direct_inputs_before_environment(
    interface_contract, params, options
):
    task = interface_contract.resolve_yaml(DETERMINISTIC_ENV_YAML)
    environment = RecordingEnvironment(
        {"evaluate": lambda inputs, context: _audited_fixture_result(context)}
    )
    runtime = interface_contract.runtime(task, environment)
    try:
        result = runtime.run_candidate(params, options)
    except Exception as caught:
        assert error_kind(caught) == "protocol_error"
    else:
        assert value(result, "accepted") is False
        assert value(result, "metric") is None
        assert value(result, "failure")["kind"] == "protocol_error"
    assert environment.calls == []


def test_wing_metric_uses_configured_penalty(
    interface_contract, make_spec, wing_spec_mapping, valid_wing_result
):
    raw = {**valid_wing_result, "Cl": -1.4, "Cd": 0.08}
    scored = interface_contract.normalize_evaluation(raw, make_spec(wing_spec_mapping))
    assert value(scored, "accepted") is True
    assert value(scored, "metric") == pytest.approx(1.2)
    assert value(scored, "raw")["Cl"] == -1.4
    assert value(scored, "raw")["Cd"] == 0.08




@pytest.mark.parametrize(
    "trust",
    [{}, {"converged": False, "mesh_ok": True}, {"converged": True, "mesh_ok": False}],
)
def test_missing_or_false_trust_has_no_metric(
    interface_contract, make_spec, wing_spec_mapping, valid_wing_result, trust
):
    scored = interface_contract.normalize_evaluation(
        {**valid_wing_result, "trust": trust}, make_spec(wing_spec_mapping)
    )
    assert value(scored, "accepted") is False
    assert value(scored, "metric") is None


@pytest.mark.parametrize(
    "mutation",
    [
        lambda result: {k: v for k, v in result.items() if k != "Cl"},
        lambda result: {k: v for k, v in result.items() if k != "Cd"},
        lambda result: {**result, "Cl": math.nan},
        lambda result: {**result, "failure": {"kind": "solver_failed"}},
        lambda result: {**result, "conditions_completed": []},
        lambda result: {**result, "fidelity": 1},
    ],
)
def test_incomplete_nonfinite_failed_or_wrong_fidelity_has_no_metric(
    interface_contract, make_spec, wing_spec_mapping, valid_wing_result, mutation
):
    scored = interface_contract.normalize_evaluation(
        mutation(valid_wing_result), make_spec(wing_spec_mapping)
    )
    assert value(scored, "accepted") is False
    assert value(scored, "metric") is None


def test_hard_constraint_rejects_but_penalty_constraint_does_not(
    interface_contract, make_spec, wing_spec_mapping, valid_wing_result
):
    hard_mapping = copy.deepcopy(wing_spec_mapping)
    hard_mapping["task"]["constraints"][0]["kind"] = "hard"
    raw = {**valid_wing_result, "Cd": 0.08}
    hard = interface_contract.normalize_evaluation(raw, make_spec(hard_mapping))
    penalty = interface_contract.normalize_evaluation(raw, make_spec(wing_spec_mapping))
    assert value(hard, "accepted") is False
    assert value(hard, "metric") is None
    assert value(penalty, "accepted") is True
    assert value(penalty, "metric") is not None


def test_tier_zero_keeps_first_batch_acceptance_rule(
    interface_contract, make_spec, wing_spec_mapping, valid_wing_result
):
    scored = interface_contract.normalize_evaluation(
        {**valid_wing_result, "fidelity": 0}, make_spec(wing_spec_mapping)
    )
    assert value(scored, "accepted") is True
    assert value(scored, "metric") is not None


def test_fidelity_below_new_task_minimum_is_rejected(
    interface_contract, make_spec, wing_spec_mapping, valid_wing_result
):
    mapping = copy.deepcopy(wing_spec_mapping)
    mapping["task"]["evaluation"]["allowed_fidelity"] = [2]
    scored = interface_contract.normalize_evaluation(
        {**valid_wing_result, "fidelity": 0}, make_spec(mapping)
    )
    assert value(scored, "accepted") is False
    assert value(scored, "metric") is None


def test_caller_supplied_metric_is_ignored_and_recomputed(
    interface_contract, make_spec, wing_spec_mapping, valid_wing_result
):
    scored = interface_contract.normalize_evaluation(
        {**valid_wing_result, "Cl": -1.4, "Cd": 0.08, "metric": 99999.0},
        make_spec(wing_spec_mapping),
    )
    assert value(scored, "metric") == pytest.approx(1.2)


def test_runtime_validate_final_uses_best_params_and_options(
    interface_contract, make_spec, wing_spec_mapping, valid_wing_result
):
    environment = RecordingEnvironment({"evaluate": valid_wing_result})
    runtime = interface_contract.runtime(make_spec(wing_spec_mapping), environment)
    result = runtime.validate_final(
        {"h_c": 0.2, "alpha_deg": 3.0, "camber": 0.04}, {"tier": 2}
    )
    assert value(result, "metric") is not None
    assert environment.calls[-1]["tool"] == "evaluate"
    assert environment.calls[-1]["inputs"]["options"] == {"tier": 2}
