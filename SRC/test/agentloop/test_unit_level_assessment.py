"""U05-U07/U09/U11/U14: observable assessment behavior, synthetic only."""
from __future__ import annotations

import copy
import importlib

import pytest


@pytest.fixture
def assess():
    return importlib.import_module("pipeline.agentloop.evaluation.assessment").assess


@pytest.fixture
def protocol():
    return {"stage_id": "M0", "protocol_hash": "protocol-frozen-1",
            "objective": {"field": "Cl", "direction": "target", "target": 0.5},
            "constraints": [{"field": "Cd", "kind": "hard", "relation": "max", "threshold": 0.1}],
            "required_conditions": ["design"],
            "numerical_checks": ["mesh_ok", "residual_ok"],
            "allowed_sources": ["synthetic"]}


@pytest.fixture
def observation():
    return {"observation_id": "obs-1", "candidate_id": "cand-1", "design_hash": "design-1",
            "stage_id": "M0", "protocol_hash": "protocol-frozen-1", "condition": "design",
            "source": "synthetic", "status": "completed", "role": "formal",
            "raw_metrics": {"Cl": 0.4, "Cd": 0.03},
            "diagnostics": {"mesh_ok": True, "residual_ok": True},
            "artifacts": {"post": "synthetic://post-1"},
            "cost": {"solver_calls": 1, "tool_seconds": 2.0}}


def test_complete_feasible_observation_has_target_objective_and_trace(assess, protocol, observation):
    result = assess([observation], protocol)
    assert result["completeness"] == "complete"
    assert result["numerical_status"] == "pass"
    assert result["feasibility"] == "feasible"
    assert result["objective_value"] == pytest.approx(-0.1)
    assert result["rank_eligible"] is True
    assert result["evidence_refs"] == ["obs-1"]
    for key in ("candidate_id", "design_hash", "stage_id", "protocol_hash"):
        assert result[key] == observation[key]


@pytest.mark.parametrize("missing", ["Cl", "Cd"])
def test_missing_required_metric_is_unknown_not_zero(assess, protocol, observation, missing):
    del observation["raw_metrics"][missing]
    before = copy.deepcopy(observation)
    result = assess([observation], protocol)
    assert result["completeness"] == "incomplete"
    assert not result["rank_eligible"]
    assert result["feasibility"] == "unknown"
    if missing == "Cl":
        assert result["objective_value"] is None
    assert observation == before


def test_missing_condition_cannot_pass(assess, protocol, observation):
    protocol["required_conditions"] = ["design", "off_design"]
    result = assess([observation], protocol)
    assert result["completeness"] == "incomplete"
    assert not result["rank_eligible"]


@pytest.mark.parametrize("diagnostics", [
    {"mesh_ok": False, "residual_ok": True},
    {"mesh_ok": True, "residual_ok": False},
])
def test_numeric_failure_keeps_observed_metrics_but_cannot_rank(assess, protocol, observation, diagnostics):
    observation["diagnostics"] = diagnostics
    before = copy.deepcopy(observation)
    result = assess([observation], protocol)
    assert result["numerical_status"] == "fail"
    assert not result["rank_eligible"]
    assert observation == before


@pytest.mark.parametrize("value", [None, "true", 1])
def test_unknown_or_non_boolean_check_is_not_success(assess, protocol, observation, value):
    observation["diagnostics"]["residual_ok"] = value
    result = assess([observation], protocol)
    assert result["numerical_status"] != "pass"
    assert not result["rank_eligible"]


def test_absent_numerical_check_is_unknown(assess, protocol, observation):
    del observation["diagnostics"]["residual_ok"]
    result = assess([observation], protocol)
    assert result["numerical_status"] == "unknown"
    assert not result["rank_eligible"]


def test_failed_tool_status_cannot_be_overridden_by_good_diagnostics(assess, protocol, observation):
    observation["status"] = "failed"
    result = assess([observation], protocol)
    assert result["numerical_status"] == "fail"
    assert not result["rank_eligible"]


def test_trustworthy_infeasible_result_keeps_actual_objective(assess, protocol, observation):
    observation["raw_metrics"]["Cd"] = 0.3
    result = assess([observation], protocol)
    assert result["completeness"] == "complete"
    assert result["numerical_status"] == "pass"
    assert result["feasibility"] == "infeasible"
    assert result["objective_value"] == pytest.approx(-0.1)
    assert not result["rank_eligible"]


@pytest.mark.parametrize("field,value", [
    ("Cl", float("nan")), ("Cl", float("inf")), ("Cd", None), ("Cl", True),
])
def test_invalid_metric_never_enters_ranking(assess, protocol, observation, field, value):
    observation["raw_metrics"][field] = value
    result = assess([observation], protocol)
    assert not result["rank_eligible"]


def test_exploration_success_is_not_formal_evidence(assess, protocol, observation):
    observation["role"] = "exploration"
    observation["raw_metrics"]["Cl"] = 0.5
    result = assess([observation], protocol)
    assert result["numerical_status"] == "pass"
    assert result["feasibility"] == "feasible"
    assert not result["rank_eligible"]


@pytest.mark.parametrize("source", ["synthetic", "replay", "cache"])
def test_live_protocol_cannot_admit_unapproved_sources(assess, protocol, observation, source):
    protocol["allowed_sources"] = ["live"]
    observation["source"] = source
    result = assess([observation], protocol)
    assert not result["rank_eligible"]


@pytest.mark.parametrize("field,value", [("stage_id", "M1"), ("protocol_hash", "other")])
def test_wrong_frozen_identity_cannot_rank(assess, protocol, observation, field, value):
    observation[field] = value
    result = assess([observation], protocol)
    assert not result["rank_eligible"]


def test_assessment_does_not_mix_distinct_designs_into_one_score(assess, protocol, observation):
    second = copy.deepcopy(observation)
    second.update(observation_id="obs-2", candidate_id="cand-2", design_hash="design-2")
    second["raw_metrics"]["Cl"] = 0.5
    with pytest.raises(ValueError):
        assess([observation, second], protocol)
