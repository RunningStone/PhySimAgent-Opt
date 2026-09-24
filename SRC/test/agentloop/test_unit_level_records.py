"""U08/U10/U11/U26: durable publication and accounting public contract."""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def RunStore():
    return importlib.import_module("pipeline.agentloop.records").RunStore


@pytest.fixture
def snapshot():
    return {"schema_version": 1, "stage_id": "M0", "protocol_hash": "protocol-1",
            "seed": 17, "settings": {"model": "M0", "solver": {"end_time": 100}}}


@pytest.fixture
def proposal():
    return {"action": "submit_for_evaluation", "candidate": {"representation": "parameterized",
            "parameters": {"alpha_deg": 2.0}, "candidate_id": "candidate-1",
            "parent_candidate_ids": ["parent-a", "parent-b"]},
            "question": "Does the baseline approach target lift?", "evidence_refs": []}


@pytest.fixture
def observation(tmp_path):
    post = tmp_path / "post.json"
    post.write_text(json.dumps({"Cl": 0.4, "Cd": 0.03}))
    return {"observation_id": "obs-1", "candidate_id": "candidate-1", "design_hash": "design-1",
            "stage_id": "M0", "protocol_hash": "protocol-1", "replicate_id": 0,
            "condition": "design", "source": "synthetic", "status": "completed", "role": "formal",
            "raw_metrics": {"Cl": 0.4, "Cd": 0.03},
            "diagnostics": {"mesh_ok": True, "residual_ok": True}, "artifacts": {"post": str(post)},
            "cost": {"solver_calls": 1, "tool_seconds": 2.0}}


@pytest.fixture
def assessment():
    return {"candidate_id": "candidate-1", "design_hash": "design-1", "stage_id": "M0",
            "protocol_hash": "protocol-1", "replicate_id": 0, "completeness": "complete",
            "numerical_status": "pass", "feasibility": "feasible", "objective_value": -0.1,
            "rank_eligible": True, "evidence_refs": ["obs-1"], "role": "formal"}


@pytest.fixture
def cost():
    return {"solver_calls": 1, "tool_seconds": 2.0, "wall_seconds": 3.0,
            "llm_calls": 0, "llm_tokens": "unavailable"}


def test_begin_is_not_a_published_observation(RunStore, tmp_path, snapshot, proposal):
    store = RunStore(tmp_path / "run", snapshot)
    attempt = store.begin_attempt(proposal, {"solver_calls": 1, "tool_seconds": 5.0})
    assert isinstance(attempt, str) and attempt
    visible = store.load_committed()
    for key in ("proposals", "observations", "assessments", "attempts", "ledger"):
        assert isinstance(visible[key], list)
    assert visible["observations"] == []
    assert visible["assessments"] == []


def test_commit_roundtrip_preserves_evidence_parents_cost_and_checkpoint(
    RunStore, tmp_path, snapshot, proposal, observation, assessment, cost,
):
    run = tmp_path / "run"
    store = RunStore(run, snapshot)
    attempt = store.begin_attempt(proposal, {"solver_calls": 1, "tool_seconds": 5.0})
    checkpoint = {"rng_state": [1, 2, 3], "visible_evidence": ["obs-1"], "next_step": 1}
    store.commit_attempt(attempt, [observation], assessment, cost, checkpoint)
    loaded = RunStore(run, snapshot).load_committed()
    assert loaded["observations"] == [observation]
    assert loaded["assessments"] == [assessment]
    assert loaded["checkpoint"] == checkpoint
    assert len(loaded["proposals"]) == 1
    assert "parent-a" in json.dumps(loaded["proposals"]) and "parent-b" in json.dumps(loaded["proposals"])
    assert loaded["ledger"][0]["attempt_id"] == attempt
    assert loaded["ledger"][0]["cost"] == cost


def test_same_attempt_commit_is_idempotent(RunStore, tmp_path, snapshot, proposal, observation, assessment, cost):
    store = RunStore(tmp_path / "run", snapshot)
    attempt = store.begin_attempt(proposal, {"solver_calls": 1})
    store.commit_attempt(attempt, [observation], assessment, cost, {"step": 1})
    first = store.load_committed()
    store.commit_attempt(attempt, [observation], assessment, cost, {"step": 1})
    assert store.load_committed() == first


def test_repeated_formal_design_keeps_real_cost_but_publishes_one_rank_identity(
    RunStore, tmp_path, snapshot, proposal, observation, assessment, cost,
):
    store = RunStore(tmp_path / "run", snapshot)
    for step in (1, 2):
        obs = copy.deepcopy(observation)
        obs["observation_id"] = f"obs-{step}"
        score = copy.deepcopy(assessment)
        score["evidence_refs"] = [obs["observation_id"]]
        attempt = store.begin_attempt(proposal, {"solver_calls": 1})
        store.commit_attempt(attempt, [obs], score, cost, {"step": step})
    loaded = store.load_committed()
    assert len(loaded["observations"]) == 2
    assert len(loaded["ledger"]) == 2
    assert sum(row["cost"]["solver_calls"] for row in loaded["ledger"]) == 2
    assert len(loaded["assessments"]) == 1


def test_explicit_replicate_is_retained_as_measurement_not_new_design(
    RunStore, tmp_path, snapshot, proposal, observation, assessment, cost,
):
    store = RunStore(tmp_path / "run", snapshot)
    for replicate in (0, 1):
        obs = copy.deepcopy(observation)
        obs.update(observation_id=f"obs-{replicate}", replicate_id=replicate)
        score = copy.deepcopy(assessment)
        score.update(replicate_id=replicate, evidence_refs=[obs["observation_id"]])
        attempt = store.begin_attempt(proposal, {"solver_calls": 1})
        store.commit_attempt(attempt, [obs], score, cost, {"replicate": replicate})
    loaded = store.load_committed()
    assert {row["replicate_id"] for row in loaded["assessments"]} == {0, 1}
    assert {row["design_hash"] for row in loaded["assessments"]} == {"design-1"}


def test_inflight_recovery_preserves_reservation_and_unknown_actual_cost(RunStore, tmp_path, snapshot, proposal):
    run = tmp_path / "run"
    reservation = {"solver_calls": 1, "tool_seconds": 5.0}
    attempt = RunStore(run, snapshot).begin_attempt(proposal, reservation)
    store = RunStore(run, snapshot)
    interrupted = store.recover_inflight({})
    assert len(interrupted) == 1
    record = interrupted[0]
    assert record["attempt_id"] == attempt and record["status"] == "interrupted"
    assert record["reservation"] == reservation
    assert "unknown" in json.dumps(record)
    assert store.load_committed()["assessments"] == []
    assert store.recover_inflight({}) == [], "recovery must not charge one interruption twice"


def test_committed_attempt_is_not_recovered_or_reexecuted(RunStore, tmp_path, snapshot, proposal, observation, assessment, cost):
    run = tmp_path / "run"
    store = RunStore(run, snapshot)
    attempt = store.begin_attempt(proposal, {"solver_calls": 1})
    store.commit_attempt(attempt, [observation], assessment, cost, {"step": 1})
    recovered = RunStore(run, snapshot)
    assert recovered.recover_inflight({}) == []
    assert recovered.load_committed()["checkpoint"] == {"step": 1}


@pytest.mark.parametrize("field,value", [("seed", 18), ("protocol_hash", "protocol-2"), ("stage_id", "M1")])
def test_changed_snapshot_refuses_resume(RunStore, tmp_path, snapshot, field, value):
    run = tmp_path / "run"
    RunStore(run, snapshot)
    changed = copy.deepcopy(snapshot)
    changed[field] = value
    with pytest.raises(ValueError):
        RunStore(run, changed)


def test_exploration_can_commit_without_assessment_or_score(RunStore, tmp_path, snapshot, proposal, observation, cost):
    proposal["action"] = "explore"
    observation["role"] = "exploration"
    store = RunStore(tmp_path / "run", snapshot)
    attempt = store.begin_attempt(proposal, {"solver_calls": 1})
    store.commit_attempt(attempt, [observation], None, cost, {"visible_evidence": ["obs-1"]})
    loaded = RunStore(tmp_path / "run", snapshot).load_committed()
    assert loaded["assessments"] == []
    assert loaded["observations"][0]["raw_metrics"]["Cl"] == 0.4
    assert loaded["checkpoint"]["visible_evidence"] == ["obs-1"]


def test_committing_unknown_attempt_is_rejected(RunStore, tmp_path, snapshot, observation, assessment, cost):
    store = RunStore(tmp_path / "run", snapshot)
    with pytest.raises(ValueError):
        store.commit_attempt("never-begun", [observation], assessment, cost, {})
    assert store.load_committed()["observations"] == []


def test_complete_multicondition_batch_is_committed_atomically(
    RunStore, tmp_path, snapshot, proposal, observation, assessment, cost,
):
    store = RunStore(tmp_path / "run", snapshot)
    second = copy.deepcopy(observation)
    second.update(observation_id="obs-2", condition="off_design")
    assessment["evidence_refs"] = ["obs-1", "obs-2"]
    attempt = store.begin_attempt(proposal, {"solver_calls": 2})
    cost["solver_calls"] = 2
    store.commit_attempt(attempt, [observation, second], assessment, cost, {"step": 1})
    loaded = RunStore(tmp_path / "run", snapshot).load_committed()
    assert loaded["observations"] == [observation, second]
    assert loaded["assessments"][0]["evidence_refs"] == ["obs-1", "obs-2"]
    assert len(loaded["ledger"]) == 1 and loaded["ledger"][0]["cost"]["solver_calls"] == 2


def test_missing_artifact_in_batch_cannot_partially_publish_success(
    RunStore, tmp_path, snapshot, proposal, observation, assessment, cost,
):
    store = RunStore(tmp_path / "run", snapshot)
    second = copy.deepcopy(observation)
    second.update(observation_id="obs-2", condition="off_design")
    second["artifacts"] = {"post": str(tmp_path / "never-produced.json")}
    assessment["evidence_refs"] = ["obs-1", "obs-2"]
    attempt = store.begin_attempt(proposal, {"solver_calls": 2})
    with pytest.raises(ValueError):
        store.commit_attempt(attempt, [observation, second], assessment, cost, {"step": 1})
    loaded = RunStore(tmp_path / "run", snapshot).load_committed()
    assert loaded["observations"] == [] and loaded["assessments"] == []


def test_failed_formal_identity_can_be_promoted_by_successful_retry(
    RunStore, tmp_path, snapshot, proposal, observation, assessment, cost,
):
    store = RunStore(tmp_path / "run", snapshot)
    failed = copy.deepcopy(observation)
    failed.update(observation_id="failed-observation", status="failed")
    failed["diagnostics"]["residual_ok"] = False
    failed_score = copy.deepcopy(assessment)
    failed_score.update(numerical_status="fail", rank_eligible=False,
                        evidence_refs=["failed-observation"], objective_value=None)
    attempt = store.begin_attempt(proposal, {"solver_calls": 1})
    store.commit_attempt(attempt, [failed], failed_score, cost, {"step": 1})
    second = store.begin_attempt(proposal, {"solver_calls": 1})
    store.commit_attempt(second, [observation], assessment, cost, {"step": 2})
    loaded = RunStore(tmp_path / "run", snapshot).load_committed()
    assert sum(row["cost"]["solver_calls"] for row in loaded["ledger"]) == 2
    assert {row["status"] for row in loaded["observations"]} == {"failed", "completed"}
    eligible = [row for row in loaded["assessments"] if row["rank_eligible"]]
    assert len(eligible) == 1 and eligible[0]["evidence_refs"] == ["obs-1"]
    select_topk = importlib.import_module("pipeline.agentloop.harness").select_topk
    assert select_topk(loaded["assessments"], 1)[0]["design_hash"] == "design-1"


def test_cache_receipt_roundtrip_preserves_original_observation_and_cost(tmp_path, observation):
    records = importlib.import_module("pipeline.agentloop.records")
    cache_dir = tmp_path / "cache"
    records.save_observation_cache(cache_dir, "exact-cache-key", observation)
    assert records.load_observation_cache(cache_dir, "exact-cache-key") == observation
    assert records.load_observation_cache(cache_dir, "other-cache-key") is None


def test_cache_receipt_rejects_artifact_content_change(tmp_path, observation):
    records = importlib.import_module("pipeline.agentloop.records")
    cache_dir = tmp_path / "cache"
    records.save_observation_cache(cache_dir, "exact-cache-key", observation)
    Path(observation["artifacts"]["post"]).write_text('{"Cl":999}')
    assert records.load_observation_cache(cache_dir, "exact-cache-key") is None


@pytest.mark.parametrize("damage", ["delete", "change"])
def test_cache_verifies_underlying_files_named_by_artifact_manifest(tmp_path, observation, damage):
    records = importlib.import_module("pipeline.agentloop.records")
    raw = tmp_path / "raw-pressure.dat"
    raw.write_text("1 2 3\n")
    manifest = Path(observation["artifacts"]["post"])
    manifest.write_text(json.dumps({"files": {raw.name: hashlib.sha256(raw.read_bytes()).hexdigest()}}))
    cache_dir = tmp_path / "cache"
    records.save_observation_cache(cache_dir, "exact-cache-key", observation)
    assert records.load_observation_cache(cache_dir, "exact-cache-key") == observation
    if damage == "delete":
        raw.unlink()
    else:
        raw.write_text("changed\n")
    assert records.load_observation_cache(cache_dir, "exact-cache-key") is None


@pytest.mark.parametrize("replacement", [{}, {"schema_version": 0}, {"legacy": True}, "invalid-json"])
def test_old_or_corrupt_cache_receipt_is_not_new_formal_evidence(tmp_path, observation, replacement):
    records = importlib.import_module("pipeline.agentloop.records")
    cache_dir = tmp_path / "cache"
    records.save_observation_cache(cache_dir, "exact-cache-key", observation)
    receipts = list(cache_dir.glob("*.json"))
    assert len(receipts) == 1
    receipts[0].write_text(replacement if isinstance(replacement, str) else json.dumps(replacement))
    assert records.load_observation_cache(cache_dir, "exact-cache-key") is None


@pytest.mark.parametrize("field,value", [("status", "failed"), ("source", "cache"), ("source", "replay")])
def test_non_original_or_failed_observation_cannot_become_persistent_cache(tmp_path, observation, field, value):
    records = importlib.import_module("pipeline.agentloop.records")
    observation[field] = value
    cache_dir = tmp_path / "cache"
    try:
        records.save_observation_cache(cache_dir, "exact-cache-key", observation)
    except ValueError:
        pass
    assert records.load_observation_cache(cache_dir, "exact-cache-key") is None
