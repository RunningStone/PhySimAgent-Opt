"""Real-solver smoke gate for direct block vs adapter ``invoke``."""

from __future__ import annotations

import importlib.util
import shutil

import pytest

from test.exp.agentloop_fixtures import (
    OLD_RUN_CONFIGS,
    RecordingEnvironment,
    default_design,
    direct_block_run,
    plain,
    value,
)


pytestmark = pytest.mark.smoke


def _require_real_backend(environment):
    if environment == "wing" and shutil.which("openfoam2512") is None:
        pytest.skip("blocked: openfoam2512 is unavailable; real simblock smoke not run")


def _overlay_for(environment, compat_overlays):
    version = "v1"
    old = next(
        path
        for path in OLD_RUN_CONFIGS
        if {"v1": "aero2d"}[version] in path.parts and path.name == "lineA.yaml"
    )
    assert old.resolve() in compat_overlays
    return compat_overlays[old.resolve()]


def _raw(result):
    if hasattr(result, "raw") or isinstance(result, dict) and "raw" in result:
        return value(result, "raw")
    return result


def _physical_projection(environment, result):
    raw = plain(_raw(result))
    physical_key = "forces"
    physical = raw.get(physical_key)
    if physical is None:
        names = ("Cl", "Cd")
        physical = {name: raw[name] for name in names}
    return {
        "case_id": raw["case_id"],
        "input_hash": raw["input_hash"],
        "status": raw.get("status"),
        physical_key: physical,
        "trust": raw["trust"],
        "failure": raw["failure"],
    }


def _legacy_metric(environment, result, task):
    raw = _physical_projection(environment, result)
    if raw["failure"] is not None or not raw["trust"] or not all(raw["trust"].values()):
        return None
    if environment == "wing":
        forces = raw["forces"]
        task_data = plain(task)
        task_body = task_data.get("task", task_data)
        cd_max = next(
            item["threshold"]
            for item in task_body["constraints"]
            if item["field"] == "Cd"
        )
        return -forces["Cl"] - 10 * max(0.0, forces["Cd"] - cd_max)
    raise ValueError(f"unsupported test environment {environment!r}")


@pytest.mark.parametrize("environment", ["wing"])
def test_direct_block_and_adapter_validate_final_match_semantically(
    interface_contract, compat_overlays, environment, artifact_dir_interface
):
    _require_real_backend(environment)
    overlay = _overlay_for(environment, compat_overlays)
    task = interface_contract.resolve_yaml(overlay)
    params, options = default_design(task)
    direct = direct_block_run(
        environment,
        params,
        options,
        artifact_dir_interface / f"{environment}-direct-store",
    )
    runtime = interface_contract.ToolRuntime(task)
    adapted = runtime.validate_final(params, options)
    assert _physical_projection(environment, adapted) == _physical_projection(
        environment, direct
    )
    assert value(adapted, "metric") == pytest.approx(
        _legacy_metric(environment, direct, task)
    )
    assert value(adapted, "trace")["interface"] == "ToolRuntime"
    assert value(adapted, "trace")["cache_hit"] is False
    assert value(adapted, "trace")["real_solver_invocations"] == 1


def test_real_adapter_result_has_trace_constraints_and_frozen_versions(
    interface_contract, compat_overlays, artifact_dir_interface
):
    _require_real_backend("wing")
    task = interface_contract.resolve_yaml(_overlay_for("wing", compat_overlays))
    params, options = default_design(task)
    result = interface_contract.ToolRuntime(task).validate_final(params, options)
    assert value(result, "trace")
    assert value(result, "constraints")
    versions = value(result, "versions")
    assert versions["config_hash"]
    assert versions["source_version"]
    assert versions["data_version"]
    assert artifact_dir_interface.is_dir()


def test_third_yaml_environment_runs_through_generic_runtime(
    interface_contract, wing_spec_mapping, write_task_yaml
):
    third = {
        **wing_spec_mapping,
        "environment": {
            "name": "deterministic-third-environment",
            "entrypoint": "test:third_environment",
            "version": "fixture-v1",
            "data": {"dataset": "fixture", "version": "sha256:fixture"},
        },
        "task": {
            **wing_spec_mapping["task"],
            "name": "deterministic-third-task",
            "prompt": "Maximise public quality.",
            "params": {"x": {"type": "float", "bounds": [0.0, 1.0]}},
            "options": {},
            "objective": {"field": "quality", "direction": "maximize", "scale": 1.0},
            "constraints": [],
            "trust": {"required": ["ok"]},
            "evaluation": {
                "tool": "evaluate",
                "required_conditions": ["only"],
                "allowed_fidelity": [0],
            },
            "private_validation": {},
        },
        "tools": {
            "evaluate": {
                "description": "Return deterministic quality.",
                "requires": [],
                "produces": ["evaluation"],
                "invalidates": [],
            }
        },
        "workflow": {"mode": "fixed", "fixed": ["evaluate"]},
    }
    environment = RecordingEnvironment(
        {
            "evaluate": lambda inputs, context: {
                "case_id": "third-1",
                "input_hash": "sha256:third",
                "quality": inputs["params"]["x"],
                "conditions_completed": ["only"],
                "fidelity": 0,
                "trust": {"ok": True},
                "failure": None,
                "summary": "deterministic",
            }
        }
    )
    task = interface_contract.resolve_yaml(write_task_yaml(third, "third.yaml"))
    runtime = interface_contract.runtime(task, environment)
    result = runtime.run_candidate({"x": 0.75}, {})
    assert value(result, "metric") == pytest.approx(0.75)
    assert value(result, "status") == "complete"
