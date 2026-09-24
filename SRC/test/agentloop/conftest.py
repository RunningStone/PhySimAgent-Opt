"""Literal public-contract inputs for the independent Level 0-3 tests."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def level_candidate():
    return {"representation": "parameterized", "parameters": {"alpha_deg": 2.0}}


@pytest.fixture
def explicit_candidate():
    return {"representation": "explicit_shape", "parameters": {"alpha_deg": 2.0},
            "shape": {"x": [0.0, 0.25, 0.5, 0.75, 1.0],
                      "upper": [0.0, 0.07, 0.09, 0.04, 0.0],
                      "lower": [0.0, -0.03, -0.04, -0.02, 0.0]}}


@pytest.fixture
def level_config():
    geometry = {"x": [0.0, 0.25, 0.5, 0.75, 1.0], "min_thickness": 0.001,
                "max_thickness": 0.3, "coordinate_bounds": [-0.3, 0.3]}
    settings = {"model": "M0", "flow": {"U_inf": 1.0, "chord": 1.0, "Re": 100.0},
                "domain": {"x_min": -5.0, "x_max": 10.0, "y_min": -5.0, "y_max": 5.0},
                "mesh": {"far": 0.8, "wing": 0.025, "mesh_scale": 1.0},
                "solver": {"end_time": 400, "delta_t": 1.0},
                "geometry": copy.deepcopy(geometry), "exploration": {"mesh_scale": [0.7, 1.3]}}
    return {
        "environment": {"name": "interview", "entrypoint": "pipeline.exp_layer.aero2d.interview:invoke",
                        "version": "test-v1", "settings": settings},
        "task": {"name": "interview", "prompt": "Find a design approaching target lift.",
                 "params": {"alpha_deg": {"type": "float", "bounds": [-5.0, 10.0], "default": 2.0}},
                 "options": {}, "geometry": geometry,
                 "objective": {"field": "Cl", "direction": "target", "target": 0.5},
                 "constraints": [{"field": "Cd", "kind": "hard", "relation": "max", "threshold": 0.1}],
                 "trust": {"required": ["mesh_ok", "residual_ok"]},
                 "evaluation": {"tool": "post", "required_conditions": ["design"],
                                "allowed_sources": ["synthetic"]}},
        "tools": {"geometry": {"requires": [], "produces": ["geometry"]},
                  "mesh": {"requires": ["geometry"], "produces": ["mesh"]},
                  "solve": {"requires": ["mesh"], "produces": ["solution"]},
                  "post": {"requires": ["solution"], "produces": ["post"]}},
        "workflow": {"mode": "fixed", "fixed": ["geometry", "mesh", "solve", "post"]},
        "limits": {"total_tools": 40, "solve_slots": 10, "total_llm": 8,
                   "wall_clock_s": 100, "validation_reserve": 4, "max_candidates": 4},
        "experiment": {"schema_version": 1, "level": 0, "arm": "base", "stage_id": "M0",
                       "parent_stage_id": None, "replicate_id": 0,
                       "initialization": "public", "transfer": {}},
    }


@pytest.fixture
def level_suite(level_config, level_candidate, explicit_candidate):
    common = copy.deepcopy(level_config)
    del common["experiment"]
    common.update(llm_model="synthetic-test", seed=17)
    m0 = copy.deepcopy(common["environment"]["settings"])
    m1 = copy.deepcopy(m0)
    m1["model"] = "M1"
    m1["solver"] = {"end_time": 5.0, "delta_t": 0.01}
    return {"schema_version": 1, "suite_id": "synthetic-level-contract",
            "profiles": {"M0": m0, "M1": m1}, "common": common,
            "initialization": {"parameterized": [level_candidate], "explicit_shape": [explicit_candidate]},
            "topk": 1,
            "runs": [{"level": 0, "arm": "base", "stages": ["M0"], "replicate_id": 0},
                     {"level": 1, "arm": "a", "stages": ["M0", "M1"], "replicate_id": 0},
                     {"level": 1, "arm": "b", "stages": ["M0", "M1"], "replicate_id": 0},
                     {"level": 2, "arm": "a", "stages": ["M0"], "replicate_id": 0},
                     {"level": 2, "arm": "b", "stages": ["M0"], "replicate_id": 0},
                     {"level": 3, "arm": "topk", "stages": ["M0", "M1"], "replicate_id": 0}]}


@pytest.fixture
def synthetic_invoker():
    calls = []

    def invoke(tool, inputs, context):
        calls.append({"tool": tool, "inputs": copy.deepcopy(inputs), "context": copy.deepcopy(context)})
        case = Path(inputs["case_dir"])
        case.mkdir(parents=True, exist_ok=True)
        artifact_name = "solution" if tool == "solve" else tool
        artifact = case / (artifact_name + ".json")
        artifact.write_text(json.dumps({"source": "synthetic", "tool": tool}))
        artifacts = {**inputs.get("artifacts", {}), artifact_name: str(artifact)}
        effective = copy.deepcopy(context["snapshot"]["settings"])
        result = {"status": "completed", "source": "synthetic", "artifacts": artifacts,
                  "diagnostics": {"mesh_ok": True, "residual_ok": True},
                  "cost": {"solver_calls": int(tool == "solve"), "tool_seconds": 0.25},
                  "effective_settings": effective,
                  "settings_hash": hashlib.sha256(json.dumps(effective, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
        if tool == "post":
            alpha = inputs["candidate"]["parameters"]["alpha_deg"]
            result["raw_metrics"] = {"Cl": float(alpha) / 10, "Cd": 0.03}
        return result

    return invoke, calls
