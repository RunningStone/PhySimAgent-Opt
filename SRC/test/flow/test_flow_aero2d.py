"""Flow tests F1-F5 for pipeline.exp_layer.aero2d (REQUIREMENTS.html §3.3).

Ten-minute-scale; require ``openfoam2512`` on PATH (skipped otherwise). The
session ``store`` is shared with the smoke tests, so the S1 case
(Params(0.3, 2, 0.04)) is a cache hit when both layers run in one session.
Observations that the spec asks to *record* (F2 branch, F4 iterations) are
printed and written to ``<artifact_dir>/flow_notes.json``.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

from pipeline.exp_layer.aero2d import FAILURE_TYPES, CaseResult, Params, RunOptions, input_hash, run_case, verify

pytestmark = pytest.mark.flow

HEX64 = re.compile(r"^[0-9a-f]{64}$")
TRUST_KEYS = {"converged", "residual_ok", "force_settled", "yplus_ok", "mesh_ok"}
FIELD_KEYS = {"cp_peak_x", "cp_peak_val", "sep_x", "pressure_recovery_slope"}
STAGES = {"geometry", "mesh", "solve", "post"}

P_S1 = Params(0.3, 2, 0.04)
P_F2 = Params(0.03, 10, 0.09)
P_F4 = Params(0.25, 2, 0.04)


@pytest.fixture(scope="module", autouse=True)
def _require_openfoam(has_openfoam: bool):
    if not has_openfoam:
        pytest.skip("openfoam2512 not on PATH; flow tests need OpenFOAM")


@pytest.fixture(scope="module")
def flow_store(store: Path) -> Path:
    return store


@pytest.fixture(scope="module")
def flow_notes(artifact_dir: Path):
    notes: dict = {}
    yield notes
    (artifact_dir / "flow_notes.json").write_text(json.dumps(notes, indent=2, ensure_ascii=False))


@pytest.fixture(scope="module")
def s1_result(flow_store: Path) -> CaseResult:
    return run_case(P_S1, store=flow_store)


def _assert_valid_case_result(r) -> None:
    """Structural contract of CaseResult, whichever branch (ok / typed failure) occurred."""
    assert isinstance(r, CaseResult)
    assert r.status in ("ok", "failed")
    assert HEX64.match(r.input_hash) and r.case_id == r.input_hash[:12]
    assert set(r.trust.keys()) == TRUST_KEYS
    assert all(isinstance(v, bool) for v in r.trust.values())
    assert set(r.provenance.keys()) == {"mesh_hash", "solver_build"}
    assert isinstance(r.iterations, int) and isinstance(r.runtime_s, float)
    if r.status == "ok":
        assert r.failure is None
        assert set(r.forces.keys()) == {"Cl", "Cd"}
        assert all(math.isfinite(v) for v in r.forces.values())
        assert r.field_summary is None or set(r.field_summary.keys()) == FIELD_KEYS
        assert HEX64.match(r.provenance["mesh_hash"])
    else:
        assert isinstance(r.failure, dict)
        assert r.failure["type"] in FAILURE_TYPES
        assert r.failure["stage"] in STAGES
        assert isinstance(r.failure["evidence"], str) and 0 < len(r.failure["evidence"]) <= 300
        assert r.forces is None and r.field_summary is None
        assert all(v is False for v in r.trust.values())
        assert r.trusted is False


# --------------------------------------------------------------------------- #
# F1  happy path: verify() reproduces the stored result byte-for-byte
# --------------------------------------------------------------------------- #
def test_f1_verify_happy(s1_result: CaseResult, flow_store: Path):
    assert s1_result.status == "ok", s1_result.failure
    same, diff = verify(P_S1, RunOptions(), store=flow_store)
    assert isinstance(same, bool) and isinstance(diff, str)
    assert same is True, diff
    assert diff == ""


# --------------------------------------------------------------------------- #
# F2  failure path: extreme geometry never raises, yields a typed failure or an untrusted ok
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def f2_result(flow_store: Path) -> CaseResult:
    return run_case(P_F2, store=flow_store)


def test_f2_failure_branch(f2_result: CaseResult, flow_notes: dict):
    r = f2_result
    _assert_valid_case_result(r)
    if r.status == "failed":
        branch = f"failed/{r.failure['type']}@{r.failure['stage']}"
        assert r.failure["type"] in FAILURE_TYPES
        assert len(r.failure["evidence"]) > 0
    else:
        branch = "ok/untrusted"
        assert not all(r.trust.values()), "an ok result for this extreme case must not be fully trusted"
        assert r.trusted is False
    flow_notes["F2"] = {
        "params": {"h_c": P_F2.h_c, "alpha_deg": P_F2.alpha_deg, "camber": P_F2.camber},
        "branch": branch,
        "status": r.status,
        "failure": r.failure,
        "trust": r.trust,
        "iterations": r.iterations,
        "runtime_s": r.runtime_s,
    }
    print(f"[F2] branch = {branch}; trust = {r.trust}; failure = {r.failure}")


# --------------------------------------------------------------------------- #
# F3  recovery menu: gap_refine changes the hash and still yields a valid result
# --------------------------------------------------------------------------- #
def test_f3_menu_gap_refine(f2_result: CaseResult, flow_store: Path, flow_notes: dict):
    opts = RunOptions(gap_refine=1)
    assert input_hash(P_F2, opts) != input_hash(P_F2, RunOptions())
    r3 = run_case(P_F2, opts, store=flow_store)
    _assert_valid_case_result(r3)
    assert r3.input_hash == input_hash(P_F2, opts)
    assert r3.input_hash != f2_result.input_hash
    assert r3.case_id != f2_result.case_id
    flow_notes["F3"] = {
        "gap_refine": 1,
        "status": r3.status,
        "failure": r3.failure,
        "trust": r3.trust,
        "iterations": r3.iterations,
    }
    print(f"[F3] gap_refine=1 -> status={r3.status} failure={r3.failure} trust={r3.trust}")


# --------------------------------------------------------------------------- #
# F4  warm start from the S1 case
# --------------------------------------------------------------------------- #
def test_f4_warm_start(s1_result: CaseResult, flow_store: Path, flow_notes: dict):
    assert s1_result.status == "ok", s1_result.failure
    warm_opts = RunOptions(warm_start_from=s1_result.case_id)
    assert input_hash(P_F4, warm_opts) != input_hash(P_F4, RunOptions())

    warm = run_case(P_F4, warm_opts, store=flow_store)
    _assert_valid_case_result(warm)
    assert warm.input_hash == input_hash(P_F4, warm_opts)

    cold = run_case(P_F4, RunOptions(), store=flow_store)
    _assert_valid_case_result(cold)
    assert cold.input_hash != warm.input_hash

    note = {
        "warm_start_from": s1_result.case_id,
        "warm": {"status": warm.status, "iterations": warm.iterations, "failure": warm.failure, "trust": warm.trust},
        "cold": {"status": cold.status, "iterations": cold.iterations, "failure": cold.failure, "trust": cold.trust},
    }
    if warm.status == "ok" and cold.status == "ok":
        note["warm_iterations_le_cold"] = warm.iterations <= cold.iterations
        # recorded, not enforced (REQUIREMENTS F4: "记录,不强制")
    flow_notes["F4"] = note
    print(f"[F4] warm iterations={warm.iterations} ({warm.status}); cold iterations={cold.iterations} ({cold.status})")


# --------------------------------------------------------------------------- #
# F5  scalar_only only strips field_summary from the returned value
# --------------------------------------------------------------------------- #
def test_f5_scalar_only(s1_result: CaseResult, flow_store: Path):
    assert s1_result.status == "ok", s1_result.failure
    r5 = run_case(P_S1, scalar_only=True, store=flow_store)
    assert isinstance(r5, CaseResult)
    assert r5.status == "ok"
    assert r5.field_summary is None
    assert r5.forces == s1_result.forces
    assert r5.input_hash == s1_result.input_hash
    assert r5.case_id == s1_result.case_id
    # scalar_only must not change what is on disk: the stored result keeps its field_summary
    on_disk = CaseResult.from_json((flow_store / s1_result.input_hash / "result.json").read_text())
    assert on_disk.field_summary is not None
    assert set(on_disk.field_summary.keys()) == FIELD_KEYS
