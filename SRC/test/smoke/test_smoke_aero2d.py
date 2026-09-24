"""Smoke tests S1-S3 for pipeline.exp_layer.aero2d (REQUIREMENTS.html §3.2).

Minute-scale; require ``openfoam2512`` on PATH (skipped otherwise). All disk
writes go to the session ``artifact_dir`` (OUTPUTs/<stamp>-test-aero2d/store).
"""

from __future__ import annotations

import math
import re
import time
from pathlib import Path

import pytest

from pipeline.exp_layer.aero2d import CaseResult, Params, RunOptions, input_hash, run_case, solver_build

pytestmark = pytest.mark.smoke

HEX64 = re.compile(r"^[0-9a-f]{64}$")
TRUST_KEYS = {"converged", "residual_ok", "force_settled", "yplus_ok", "mesh_ok"}
FIELD_KEYS = {"cp_peak_x", "cp_peak_val", "sep_x", "pressure_recovery_slope"}
BUILD_KEYS = {"app_version", "app_zip_sha256", "api_info_hash"}
REPO_ROOT = Path(__file__).resolve().parents[3]

P_S1 = Params(0.3, 2, 0.04)


@pytest.fixture(scope="module", autouse=True)
def _require_openfoam(has_openfoam: bool):
    if not has_openfoam:
        pytest.skip("openfoam2512 not on PATH; smoke tests need OpenFOAM")


@pytest.fixture(scope="module")
def s1_result(store: Path) -> CaseResult:
    return run_case(P_S1, store=store)


# --------------------------------------------------------------------------- #
# S1  one real case end-to-end
# --------------------------------------------------------------------------- #
def test_s1_status_forces_trust(s1_result: CaseResult):
    r = s1_result
    assert isinstance(r, CaseResult)
    assert r.status == "ok", r.failure
    assert r.failure is None
    assert set(r.forces.keys()) == {"Cl", "Cd"}
    assert all(isinstance(v, float) and math.isfinite(v) for v in r.forces.values())
    assert r.forces["Cl"] < 0.0, "inverted wing in ground effect must produce downforce (Cl < 0)"
    assert set(r.trust.keys()) == TRUST_KEYS
    assert all(isinstance(v, bool) for v in r.trust.values())
    assert isinstance(r.iterations, int) and r.iterations > 0
    assert isinstance(r.runtime_s, float) and 0.0 < r.runtime_s < 300.0


def test_s1_field_summary(s1_result: CaseResult):
    fs = s1_result.field_summary
    assert isinstance(fs, dict)
    assert set(fs.keys()) == FIELD_KEYS
    assert all(isinstance(v, float) and math.isfinite(v) for v in fs.values())
    assert -6.0 < fs["cp_peak_val"] < 0.0
    assert 0.0 <= fs["cp_peak_x"] <= 1.0
    assert 0.0 < fs["sep_x"] <= 1.0


def test_s1_provenance_and_ids(s1_result: CaseResult):
    r = s1_result
    assert HEX64.match(r.input_hash)
    assert r.input_hash == input_hash(P_S1, RunOptions())
    assert r.case_id == r.input_hash[:12]
    assert set(r.provenance.keys()) == {"mesh_hash", "solver_build"}
    assert "input_hash" not in r.provenance
    assert HEX64.match(r.provenance["mesh_hash"])
    assert set(r.provenance["solver_build"].keys()) == BUILD_KEYS


def test_s1_store_layout(s1_result: CaseResult, store: Path, artifact_dir: Path):
    case_dir = store / s1_result.input_hash
    assert case_dir.is_dir()
    assert (case_dir / "result.json").is_file()
    assert (case_dir / "mesh.msh").is_file()
    assert (case_dir / "case" / "log.simpleFoam").is_file()
    # on-disk result equals what run_case returned (deterministic form)
    on_disk = CaseResult.from_json((case_dir / "result.json").read_text())
    assert on_disk.to_json(deterministic=True) == s1_result.to_json(deterministic=True)
    # everything written stays inside the store: no OpenFOAM case leaks elsewhere
    for cd in artifact_dir.rglob("controlDict"):
        assert store in cd.parents, f"controlDict written outside store: {cd}"
    assert list((REPO_ROOT / "SRC").rglob("controlDict")) == []


# --------------------------------------------------------------------------- #
# S2  cache hit
# --------------------------------------------------------------------------- #
def test_s2_cache_hit_is_identical_and_fast(s1_result: CaseResult, store: Path):
    t0 = time.perf_counter()
    r2 = run_case(P_S1, store=store)
    dt = time.perf_counter() - t0
    assert dt < 2.0, f"cache hit took {dt:.2f}s"
    assert r2.to_json(deterministic=True) == s1_result.to_json(deterministic=True)
    assert r2.runtime_s == s1_result.runtime_s, "cache hit must return the stored runtime_s unchanged"
    assert r2.input_hash == s1_result.input_hash


# --------------------------------------------------------------------------- #
# S3  solver_build
# --------------------------------------------------------------------------- #
def test_s3_solver_build():
    sb = solver_build()
    assert isinstance(sb, dict)
    assert set(sb.keys()) == BUILD_KEYS
    assert re.fullmatch(r"2\.\d+\.\d+", sb["app_version"]), sb["app_version"]
    assert HEX64.match(sb["api_info_hash"]), sb["api_info_hash"]
    assert sb["app_zip_sha256"] == "unknown" or HEX64.match(sb["app_zip_sha256"])
    assert solver_build() == sb  # cached in-process
