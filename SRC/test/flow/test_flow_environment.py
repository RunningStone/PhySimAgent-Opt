"""Small-budget fixed/agent flows through YAML -> line_cfg -> run_line."""

from __future__ import annotations

import copy
import importlib.util
import json
import shutil

import pytest

from test.exp.agentloop_fixtures import (
    DETERMINISTIC_ENV_YAML,
    OLD_RUN_CONFIGS,
    RecordingEnvironment,
    plain,
    value,
)


pytestmark = pytest.mark.flow


class _DeterministicBackend:
    def __init__(self):
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(copy.deepcopy(prompt))
        introduction = prompt.get("Introduction", "")
        if introduction.startswith("Choose the next simulation tool"):
            name = "inspect" if not prompt["Observations"] else "evaluate"
            return json.dumps({"name": name, "args": {}})
        return 'PARAMS = {"x": 0.25}\nOPTIONS = {"method": "default"}'


class _SharedBudgetBackend:
    def __init__(self):
        self.calls = 0

    def __call__(self, prompt):
        self.calls += 1
        assert self.calls <= 4, "backend was called after the shared total_llm limit"
        if self.calls == 1:
            return "invalid outer proposal"
        if self.calls == 2:
            return 'PARAMS = {"x": 0.25}\nOPTIONS = {"method": "default"}'
        if self.calls == 3:
            return "invalid chooser response"
        return json.dumps({"name": "inspect", "args": {}})


class _ExtraFieldChooserBackend:
    def __init__(self):
        self.calls = 0
        self.chooser_calls = 0

    def __call__(self, prompt):
        self.calls += 1
        introduction = prompt.get("Introduction", "")
        if not introduction.startswith("Choose the next simulation tool"):
            return 'PARAMS = {"x": 0.25}\nOPTIONS = {"method": "default"}'
        self.chooser_calls += 1
        if self.chooser_calls == 1:
            return json.dumps(
                {
                    "name": "inspect",
                    "args": {},
                    "reading": "undeclared",
                    "hypothesis": "undeclared",
                    "PARAMS": {"x": 0.25},
                }
            )
        name = "inspect" if not prompt["Observations"] else "evaluate"
        return json.dumps({"name": name, "args": {}})


def _deterministic_line(contract, tmp_path, mode):
    merged = contract.driver.load_config(DETERMINISTIC_ENV_YAML)
    merged = copy.deepcopy(merged)
    merged["workflow"] = {"mode": mode, "fixed": ["inspect", "evaluate"]}
    merged["tools"]["evaluate"]["requires"] = ["inspection"]
    merged.update(
        {
            "line": "B",
            "budget": 2,
            "seed": 0,
            "llm_model": "deterministic-fixture",
            "exp_name": f"deterministic-{mode}",
            "output_dir": str(tmp_path / mode),
        }
    )
    merged["limits"].update(
        {"total_tools": 12, "solve_slots": 12, "total_llm": 8, "validation_reserve": 0}
    )
    return merged, contract.driver.line_cfg_from_yaml(merged)


def _candidate_id(tool_call):
    return tool_call["result"]["provenance"]["candidate_id"]


def _assert_runtime_counts(trace):
    ledger = trace["ledger"]
    for key in ("tool_queries", "candidate_tool_queries", "validation_queries"):
        assert trace[key] == ledger[key]
    assert trace["chooser_calls"] == ledger["llm_calls"]
    assert trace["candidate_tool_queries"] == len(trace["tool_calls"])
    assert trace["validation_queries"] == len(trace["verification"]["holdout"]["cases"])


def _assert_choices_link_within_each_nonbootstrap_candidate(trace):
    calls = trace["tool_calls"]
    bootstrap_id = _candidate_id(calls[0])
    chosen_calls = [call for call in calls if _candidate_id(call) != bootstrap_id]
    assert len(trace["choices"]) == len(chosen_calls)
    previous = None
    previous_candidate = None
    for choice, call in zip(trace["choices"], chosen_calls):
        candidate = _candidate_id(call)
        assert choice["name"] == call["tool"]
        if candidate != previous_candidate:
            assert choice["observation_id"] is None
        else:
            assert choice["observation_id"] == previous["result_id"]
            assert _candidate_id(previous) == candidate
        previous = call
        previous_candidate = candidate


def _require_environment_and_llm(environment):
    if shutil.which("claude") is None:
        pytest.skip("blocked: claude CLI is unavailable; live LLM flow not run")
    if environment == "wing" and shutil.which("openfoam2512") is None:
        pytest.skip("blocked: openfoam2512 is unavailable; real wing flow not run")


def _overlay_for(environment, compat_overlays):
    version = "v1"
    old = next(
        path
        for path in OLD_RUN_CONFIGS
        if {"v1": "aero2d"}[version] in path.parts and path.name == "lineB.yaml"
    )
    assert old.resolve() in compat_overlays
    return compat_overlays[old.resolve()]


def _live_config(contract, overlays, environment, mode, output_dir, **extra):
    merged = contract.driver.load_config(_overlay_for(environment, overlays))
    merged = copy.deepcopy(merged)
    merged["workflow"]["mode"] = mode
    merged["budget"] = extra.pop("candidate_budget", 2)
    merged["limits"].update(
        {
            "max_tools_per_candidate": extra.pop("max_tools_per_candidate", 3),
            "total_tools": extra.pop("total_tools", 7),
            "validation_reserve": extra.pop("validation_reserve", 1),
        }
    )
    merged["run_id"] = f"flow-{environment}-{mode}"
    merged["output_dir"] = output_dir
    merged.update(extra)
    return merged, contract.driver.line_cfg_from_yaml(merged)


@pytest.mark.parametrize("environment", ["wing"])
@pytest.mark.parametrize("mode", ["fixed", "agent"])
def test_small_budget_live_flow_has_linked_trace_and_exact_accounting(
    interface_contract, compat_overlays, environment, mode, artifact_dir_interface
):
    _require_environment_and_llm(environment)
    _, line_cfg = _live_config(
        interface_contract,
        compat_overlays,
        environment,
        mode,
        artifact_dir_interface / f"live-{environment}-{mode}",
    )
    result = interface_contract.module.run_line(line_cfg)
    trace = value(result, "trace")
    assert trace["interface"] == "ToolRuntime"
    assert trace["mode"] == mode
    assert trace["candidate_count"] <= 2
    assert trace["tool_queries"] <= 7
    assert trace["candidate_tool_queries"] <= 6
    assert trace["candidate_count"] == len(trace["nodes"])
    assert set(event["tool"] for event in trace["tool_calls"]) <= set(
        trace["allowed_tools"]
    )
    _assert_runtime_counts(trace)
    assert "/private/" not in repr(trace["prompts"])
    if mode == "fixed":
        assert trace["chooser_calls"] == 0
        assert all(node["evaluate_calls"] == 1 for node in trace["nodes"])
    else:
        assert trace["chooser_calls"] >= trace["candidate_count"] - 1
        if trace["candidate_count"] > 1:
            assert trace["choices"]
            bootstrap_id = _candidate_id(trace["tool_calls"][0])
            assert any(
                _candidate_id(call) != bootstrap_id for call in trace["tool_calls"]
            )
        _assert_choices_link_within_each_nonbootstrap_candidate(trace)


@pytest.mark.parametrize("environment", ["wing"])
def test_controlled_failure_flow_has_no_stale_metric_and_no_orphan_work(
    interface_contract, compat_overlays, environment, artifact_dir_interface
):
    _require_environment_and_llm(environment)
    _, line_cfg = _live_config(
        interface_contract,
        compat_overlays,
        environment,
        "agent",
        artifact_dir_interface / f"failure-{environment}",
        candidate_budget=1,
        max_tools_per_candidate=2,
        total_tools=3,
    )
    failed_environment = RecordingEnvironment(
        {
            "simulate": {
                "case_id": "controlled-failure",
                "failure": {"kind": "solver_failed"},
            },
            "evaluate": {
                "case_id": "controlled-failure",
                "failure": {"kind": "solver_failed"},
            },
            "diagnose": {"diagnosis": "solver did not produce a solution"},
        }
    )
    line_cfg["invoke_fn"] = failed_environment.invoke
    result = interface_contract.module.run_line(line_cfg)
    trace = value(result, "trace")
    assert trace["nodes"][0]["status"] == "failed"
    assert trace["nodes"][0]["metric"] is None
    assert trace["nodes"][0]["journal_entries"] == 1
    assert trace["orphan_processes"] == []
    assert trace["ledger"]["failed_tool_queries"] >= 1


def test_modes_share_physics_bootstrap_tools_model_and_budget(
    interface_contract, compat_overlays, tmp_path
):
    fixed_merged, fixed_cfg = _live_config(
        interface_contract, compat_overlays, "wing", "fixed", tmp_path / "fixed"
    )
    agent_merged, agent_cfg = _live_config(
        interface_contract, compat_overlays, "wing", "agent", tmp_path / "agent"
    )
    fixed_task = interface_contract.resolve_task(fixed_merged)
    agent_task = interface_contract.resolve_task(agent_merged)
    assert fixed_task.problem_hash == agent_task.problem_hash
    fixed_task_data = plain(fixed_task)
    agent_task_data = plain(agent_task)
    for field in ("environment", "task", "tools", "limits"):
        assert fixed_task_data[field] == agent_task_data[field]
    assert fixed_cfg["llm_model"] == agent_cfg["llm_model"]
    assert fixed_cfg["budget"] == agent_cfg["budget"]
    assert fixed_cfg["run_id"] != agent_cfg["run_id"]
    assert fixed_cfg["exp_dir"] != agent_cfg["exp_dir"]


def test_mode_output_paths_and_aggregation_keys_remain_separate(
    interface_contract, compat_overlays, tmp_path
):
    _, fixed = _live_config(
        interface_contract, compat_overlays, "wing", "fixed", tmp_path / "fixed"
    )
    _, agent = _live_config(
        interface_contract, compat_overlays, "wing", "agent", tmp_path / "agent"
    )
    assert fixed["exp_dir"] != agent["exp_dir"]
    grouped = {fixed["workflow"]["mode"]: [], agent["workflow"]["mode"]: []}
    assert set(grouped) == {"fixed", "agent"}


def test_fixed_and_agent_share_fixed_step_zero_then_diverge_to_chooser(
    interface_contract, tmp_path
):
    traces = {}
    for mode in ("fixed", "agent"):
        merged, line_cfg = _deterministic_line(interface_contract, tmp_path, mode)
        backend = _DeterministicBackend()
        line_cfg["backend"] = backend
        trace = value(interface_contract.module.run_line(line_cfg), "trace")
        traces[mode] = trace
        first_candidate = _candidate_id(trace["tool_calls"][0])
        bootstrap_tools = [
            call["tool"]
            for call in trace["tool_calls"]
            if _candidate_id(call) == first_candidate
        ]
        assert bootstrap_tools == merged["workflow"]["fixed"]
        assert trace["ledger"]["tool_queries"] >= len(bootstrap_tools)
        assert trace["ledger"]["solve_slots"] >= len(bootstrap_tools)
        _assert_runtime_counts(trace)

    assert traces["fixed"]["chooser_calls"] == 0
    assert traces["fixed"]["choices"] == []
    assert traces["agent"]["chooser_calls"] > 0
    _assert_choices_link_within_each_nonbootstrap_candidate(traces["agent"])


def test_next_nonlegacy_proposal_sees_public_objective_and_constraint_only(
    interface_contract, tmp_path
):
    _, line_cfg = _deterministic_line(interface_contract, tmp_path, "fixed")
    backend = _DeterministicBackend()
    line_cfg["backend"] = backend
    interface_contract.module.run_line(line_cfg)
    proposals = [
        prompt
        for prompt in backend.prompts
        if prompt.get("Introduction", "").startswith("You are an engineering design agent")
    ]
    assert proposals
    previous_result = proposals[0]["Previous design"]["Result"]
    assert '"value"' in previous_result
    assert '"limit"' in previous_result
    prompt_text = json.dumps(proposals[0], sort_keys=True).lower()
    assert "private_validation" not in prompt_text
    assert "heldout" not in prompt_text and "held_out" not in prompt_text


def test_driver_agent_uses_one_total_llm_budget_for_proposals_and_chooser_retries(
    interface_contract, tmp_path
):
    merged, line_cfg = _deterministic_line(interface_contract, tmp_path, "agent")
    merged["limits"]["total_llm"] = 4
    line_cfg = interface_contract.driver.line_cfg_from_yaml(merged)
    backend = _SharedBudgetBackend()
    line_cfg["backend"] = backend
    trace = value(interface_contract.module.run_line(line_cfg), "trace")
    assert backend.calls == 4
    assert trace["llm_calls"] == 4
    assert trace["chooser_calls"] == 2
    assert trace["stop_reason"] in {"total_llm", "budget_error"}


def test_driver_retries_chooser_tool_call_with_unknown_top_level_fields(
    interface_contract, tmp_path
):
    _, line_cfg = _deterministic_line(interface_contract, tmp_path, "agent")
    backend = _ExtraFieldChooserBackend()
    line_cfg["backend"] = backend
    trace = value(interface_contract.module.run_line(line_cfg), "trace")
    assert backend.calls == 4
    assert backend.chooser_calls == 3
    assert trace["llm_calls"] == 4
    assert trace["chooser_calls"] == 3
    bootstrap_id = _candidate_id(trace["tool_calls"][0])
    candidate_tools = [
        call["tool"]
        for call in trace["tool_calls"]
        if _candidate_id(call) != bootstrap_id
    ]
    assert candidate_tools == ["inspect", "evaluate"]


def test_chooser_prompt_overrides_task_design_format_with_exact_plain_json_contract(
    interface_contract, tmp_path
):
    _, line_cfg = _deterministic_line(interface_contract, tmp_path, "agent")
    backend = _DeterministicBackend()
    line_cfg["backend"] = backend
    interface_contract.module.run_line(line_cfg)
    chooser_prompts = [
        prompt
        for prompt in backend.prompts
        if prompt.get("Introduction", "").startswith("Choose the next simulation tool")
    ]
    assert chooser_prompts
    instructions = chooser_prompts[0]["Instructions"].lower()
    assert "task description" in instructions
    assert "override" in instructions or "ignore" in instructions
    assert "exactly" in instructions
    assert '"name"' in instructions and '"args"' in instructions
    assert all(field in instructions for field in ("reading", "hypothesis", "params"))
    assert "fence" in instructions or "```" in instructions
    assert "do not" in instructions or "must not" in instructions
