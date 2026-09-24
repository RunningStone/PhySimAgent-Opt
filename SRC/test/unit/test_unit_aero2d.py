"""Unit tests U1-U7 for pipeline.exp_layer.aero2d (REQUIREMENTS.html §3.1).

Second-scale, no OpenFOAM / gmsh execution, nothing written to disk (U6 uses
pytest's tmp_path only to prove that *nothing* is created under the store).
Only names from REQUIREMENTS §2 are imported.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import re

import numpy as np
import pytest

from pipeline.exp_layer.aero2d import (
    FAILURE_TYPES,
    SETTINGS,
    CaseResult,
    Params,
    RunOptions,
    classify_failure,
    cp_features,
    input_hash,
    run_case,
    wing_coords,
)

HEX64 = re.compile(r"^[0-9a-f]{64}$")
TRUST_KEYS = {"converged", "residual_ok", "force_settled", "yplus_ok", "mesh_ok"}
FIELD_KEYS = {"cp_peak_x", "cp_peak_val", "sep_x", "pressure_recovery_slope"}
STAGES = {"geometry", "mesh", "solve", "post"}


# --------------------------------------------------------------------------- #
# public API surface (§2)
# --------------------------------------------------------------------------- #
def test_public_api_names_exist():
    import pipeline.exp_layer.aero2d as sb

    for name in (
        "Params", "RunOptions", "CaseResult", "input_hash", "wing_coords",
        "classify_failure", "cp_features", "run_case", "verify", "solver_build",
        "SETTINGS", "FAILURE_TYPES",
    ):
        assert hasattr(sb, name), f"pipeline.exp_layer.aero2d lacks public name {name!r}"


def test_data_models_are_frozen_dataclasses():
    p = Params(0.3, 2, 0.04)
    o = RunOptions()
    assert dataclasses.is_dataclass(p) and dataclasses.is_dataclass(o)
    assert dataclasses.is_dataclass(CaseResult)
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.h_c = 0.5  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        o.gap_refine = 1  # type: ignore[misc]


def test_params_field_order_and_run_options_defaults():
    p = Params(0.3, 2, 0.04)
    assert (p.h_c, p.alpha_deg, p.camber) == (0.3, 2, 0.04)
    o = RunOptions()
    assert o.gap_refine == 0
    assert o.warm_start_from is None
    assert o.relax_step == 0
    assert o.iter_mult == 1


# --------------------------------------------------------------------------- #
# U1  input_hash
# --------------------------------------------------------------------------- #
def test_u1_input_hash_is_64_hex_and_deterministic():
    h1 = input_hash(Params(0.3, 2, 0.04))
    h2 = input_hash(Params(0.3, 2, 0.04), RunOptions(), SETTINGS)
    assert isinstance(h1, str)
    assert HEX64.match(h1), h1
    assert h1 == h2


def test_u1_input_hash_independent_of_field_order():
    a = input_hash(Params(h_c=0.3, alpha_deg=2, camber=0.04), RunOptions(gap_refine=1, iter_mult=2))
    b = input_hash(Params(camber=0.04, alpha_deg=2, h_c=0.3), RunOptions(iter_mult=2, gap_refine=1))
    assert a == b
    # equivalent settings dict with keys inserted in a different order
    reordered = dict(reversed(list(SETTINGS.items())))
    assert list(reordered.keys()) != list(SETTINGS.keys())
    assert input_hash(Params(0.3, 2, 0.04), RunOptions(), reordered) == input_hash(Params(0.3, 2, 0.04))


def test_u1_input_hash_changes_with_any_field():
    base = input_hash(Params(0.3, 2, 0.04))
    assert input_hash(Params(0.3 + 1e-9, 2, 0.04)) != base
    assert input_hash(Params(0.3, 2 + 1e-9, 0.04)) != base
    assert input_hash(Params(0.3, 2, 0.04 + 1e-9)) != base
    assert input_hash(Params(0.3, 2, 0.04), RunOptions(gap_refine=1)) != base
    assert input_hash(Params(0.3, 2, 0.04), RunOptions(relax_step=1)) != base
    assert input_hash(Params(0.3, 2, 0.04), RunOptions(iter_mult=2)) != base
    assert input_hash(Params(0.3, 2, 0.04), RunOptions(warm_start_from="abcdef012345")) != base

    top = copy.deepcopy(SETTINGS)
    top["Re"] = top["Re"] * 1.01
    assert input_hash(Params(0.3, 2, 0.04), RunOptions(), top) != base

    nested = copy.deepcopy(SETTINGS)
    nested["mesh"]["wing"] = nested["mesh"]["wing"] * 1.01
    assert input_hash(Params(0.3, 2, 0.04), RunOptions(), nested) != base

    # settings must not be mutated by hashing
    assert input_hash(Params(0.3, 2, 0.04)) == base


def test_u1_input_hash_all_distinct_for_distinct_inputs():
    hs = {
        input_hash(Params(0.3, 2, 0.04)),
        input_hash(Params(0.3, 2, 0.04), RunOptions(gap_refine=1)),
        input_hash(Params(0.3, 2, 0.04), RunOptions(gap_refine=2)),
        input_hash(Params(0.25, 2, 0.04)),
        input_hash(Params(0.03, 10, 0.09)),
    }
    assert len(hs) == 5


# --------------------------------------------------------------------------- #
# U2  wing_coords
# --------------------------------------------------------------------------- #
def _coords(p: Params, **kw):
    x, y = wing_coords(p, **kw)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    assert x.ndim == 1 and y.ndim == 1 and x.shape == y.shape
    assert np.all(np.isfinite(x)) and np.all(np.isfinite(y))
    return x, y


def _split_surfaces(x: np.ndarray, y: np.ndarray):
    """Split the closed loop at the leading edge (min x) into the two surfaces,
    each returned sorted by x. 'upper' is the one with the larger mean y."""
    le = int(np.argmin(x))
    a = (x[: le + 1], y[: le + 1])
    b = (x[le:], y[le:])
    segs = []
    for sx, sy in (a, b):
        order = np.argsort(sx, kind="stable")
        segs.append((sx[order], sy[order]))
    segs.sort(key=lambda s: float(np.mean(s[1])))
    lower, upper = segs
    return upper, lower


def _y_at(seg, xq):
    sx, sy = seg
    return np.interp(xq, sx, sy)


@pytest.mark.parametrize("p", [Params(0.1, 0, 0), Params(0.1, 5, 0.04)])
def test_u2_wing_coords_closed_no_duplicates_positioned(p: Params):
    x, y = _coords(p)
    n = SETTINGS["n_surface"]
    assert len(x) >= 2 * n - 1 - n and len(x) <= 2 * n - 1 + n, "point count should be near 2n-1"
    # closed loop: first and last point coincide
    assert math.isclose(x[0], x[-1], abs_tol=1e-12) and math.isclose(y[0], y[-1], abs_tol=1e-12)
    # no repeated adjacent points
    seg = np.hypot(np.diff(x), np.diff(y))
    assert np.all(seg > 0.0), "found duplicate adjacent points"
    # x range [0, ~1]: leading edge at x = 0, chord 1
    assert x.min() >= -0.01
    assert np.min(np.abs(x)) < 1e-9, "no point at the leading edge x = 0"
    assert 0.95 <= x.max() <= 1.05
    # lowest point sits exactly at h_c above the ground (y = 0)
    assert abs(y.min() - p.h_c) < 1e-9
    assert y.min() > 0.0


def test_u2_wing_coords_symmetric_when_no_camber_no_alpha():
    x, y = _coords(Params(0.1, 0, 0))
    y_chord = 0.5 * (y.max() + y.min())
    upper, lower = _split_surfaces(x, y)
    xq = np.linspace(0.0, 1.0, 201)
    yu = _y_at(upper, xq)
    yl = _y_at(lower, xq)
    assert np.allclose(yu + yl, 2.0 * y_chord, atol=1e-6)
    # both TE and LE lie on the chord line
    assert abs(y[np.argmax(x)] - y_chord) < 1e-6
    assert abs(y[np.argmin(x)] - y_chord) < 1e-6
    # NACA 0012 thickness: max half-thickness = 0.06 (t = 0.12)
    assert abs((y.max() - y.min()) - SETTINGS["naca_t"]) < 5e-3


def test_u2_wing_coords_alpha_raises_trailing_edge():
    x0, y0 = _coords(Params(0.1, 0, 0))
    x5, y5 = _coords(Params(0.1, 5, 0))
    xc, yc = _coords(Params(0.1, 5, 0.04))
    te0 = y0[np.argmax(x0)]
    te5 = y5[np.argmax(x5)]
    tec = yc[np.argmax(xc)]
    assert te5 > te0 + 0.02, "positive alpha must raise the TE relative to the lowest point"
    assert tec > te0 + 0.02


def test_u2_wing_coords_camber_is_inverted_and_more_concave_below():
    xs, ys = _coords(Params(0.1, 0, 0))
    xc, yc = _coords(Params(0.1, 0, 0.04))
    ups, los = _split_surfaces(xs, ys)
    upc, loc = _split_surfaces(xc, yc)

    def chord_line_y(x, y):
        return 0.5 * (y[np.argmax(x)] + y[np.argmin(x)])

    cls = chord_line_y(xs, ys)
    clc = chord_line_y(xc, yc)
    xq = np.array([0.3, 0.5, 0.7])
    # inverted: the mean line of the cambered wing lies *below* its chord line
    mean_c = 0.5 * (_y_at(upc, xq) + _y_at(loc, xq))
    assert np.all(mean_c < clc - 0.01), "camber must point down (inverted wing)"
    # lower (suction) surface is deeper below the chord line than the symmetric wing
    depth_sym = cls - _y_at(los, xq)
    depth_cam = clc - _y_at(loc, xq)
    assert np.all(depth_cam > depth_sym + 0.01)
    # the cambered wing's total sag below the chord line is larger
    assert (clc - yc.min()) > (cls - ys.min()) + 0.02


def test_u2_wing_coords_respects_n():
    x_big, _ = _coords(Params(0.1, 0, 0), n=SETTINGS["n_surface"])
    x_small, _ = _coords(Params(0.1, 0, 0), n=40)
    assert len(x_small) < len(x_big)
    assert abs(len(x_small) - (2 * 40 - 1)) <= 40


# --------------------------------------------------------------------------- #
# U3  classify_failure
# --------------------------------------------------------------------------- #
CONVERGED_LOG = (
    "Time = 812\n"
    "smoothSolver:  Solving for Ux, Initial residual = 9.1e-07, Final residual = 3.2e-09, No Iterations 2\n"
    "GAMG:  Solving for p, Initial residual = 8.8e-05, Final residual = 4.1e-07, No Iterations 4\n"
    "\nSIMPLE solution converged in 812 iterations\n\nEnd\n"
)
UNSETTLED_LOG = (
    "Time = 1500\n"
    "smoothSolver:  Solving for Ux, Initial residual = 3.4e-03, Final residual = 1.1e-05, No Iterations 4\n"
    "GAMG:  Solving for p, Initial residual = 2.2e-02, Final residual = 9.5e-04, No Iterations 6\n"
    "\nEnd\n"
)
FPE_LOG = (
    "Time = 37\n"
    "GAMG:  Solving for p, Initial residual = 1, Final residual = 1e+30, No Iterations 1000\n"
    "#0  Foam::error::printStack(Foam::Ostream&)\n"
    "#1  Foam::sigFpe::sigHandler(int)\n"
    "Floating point exception\n"
)
MESH_FAILED_LOG = (
    "Checking geometry...\n"
    "    Max non-orthogonality = 89.7 degrees (average 12.3)\n"
    " ***Number of severely non-orthogonal (> 70 degrees) faces: 42.\n"
    "Failed 1 mesh checks.\n\nEnd\n"
)


def _check_failure_dict(d, expected_type, expected_stage):
    assert isinstance(d, dict)
    assert set(d.keys()) >= {"type", "stage", "evidence"}
    assert d["type"] == expected_type
    assert d["type"] in FAILURE_TYPES
    assert d["stage"] == expected_stage
    assert isinstance(d["evidence"], str)
    assert 0 < len(d["evidence"]) <= 300


def test_u3_timed_out_wins_over_everything():
    # even a converged solve log is a timeout if timed_out=True
    _check_failure_dict(classify_failure("solve", CONVERGED_LOG, 0, True), "timeout", "solve")
    _check_failure_dict(classify_failure("mesh", "", None, True), "timeout", "mesh")


def test_u3_geometry_stage():
    _check_failure_dict(
        classify_failure("geometry", "h_c=0.001 outside [0.03, 2.0]", 1, False),
        "geometry_invalid",
        "geometry",
    )


def test_u3_mesh_stage_checkmesh_failed():
    _check_failure_dict(classify_failure("mesh", MESH_FAILED_LOG, 0, False), "mesh_invalid", "mesh")


def test_u3_mesh_stage_nonzero_exit_or_negative_volume():
    _check_failure_dict(
        classify_failure("mesh", "gmshToFoam: cannot read mesh.msh", 1, False), "mesh_invalid", "mesh"
    )
    _check_failure_dict(
        classify_failure("mesh", "***Zero or negative volume cells: 3\nFailed 1 mesh checks.", 0, False),
        "mesh_invalid",
        "mesh",
    )


def test_u3_mesh_stage_clean_is_none():
    assert classify_failure("mesh", "Checking geometry...\nMesh OK.\n\nEnd\n", 0, False) is None


def test_u3_solve_diverged_variants():
    _check_failure_dict(classify_failure("solve", FPE_LOG, 0, False), "diverged", "solve")
    _check_failure_dict(classify_failure("solve", FPE_LOG, 136, False), "diverged", "solve")
    _check_failure_dict(
        classify_failure("solve", "Time = 12\nGAMG:  Solving for p, Initial residual = nan\n", 0, False),
        "diverged",
        "solve",
    )
    _check_failure_dict(
        classify_failure("solve", "--> FOAM FATAL ERROR:\nMaximum number of iterations exceeded\n", 1, False),
        "diverged",
        "solve",
    )
    # non-zero exit with an otherwise clean log is still diverged
    _check_failure_dict(classify_failure("solve", UNSETTLED_LOG, 1, False), "diverged", "solve")


def test_u3_solve_not_settled():
    _check_failure_dict(classify_failure("solve", UNSETTLED_LOG, 0, False), "not_settled", "solve")


def test_u3_solve_converged_is_none():
    assert classify_failure("solve", CONVERGED_LOG, 0, False) is None


def test_u3_post_parse_failure_is_diverged_with_post_evidence():
    d = classify_failure("post", "ValueError: cp_features: need >= 3 samples, got 2", 1, False)
    _check_failure_dict(d, "diverged", "post")
    assert "post" in d["evidence"].lower()


def test_u3_evidence_is_capped_at_300_chars():
    long_log = "x" * 50 + "\nFloating point exception\n" + ("y" * 5000)
    d = classify_failure("solve", long_log, 0, False)
    _check_failure_dict(d, "diverged", "solve")
    assert len(d["evidence"]) <= 300


def test_u3_never_raises_on_odd_inputs():
    for args in (
        ("solve", "", None, False),
        ("solve", "", 0, False),
        ("mesh", "", 0, False),
        ("post", "", None, False),
        ("geometry", "", 0, False),
        ("weird-stage", "", 0, False),
        ("solve", "\x00\xff binary junk", -11, False),
    ):
        out = classify_failure(*args)
        assert out is None or (isinstance(out, dict) and out["type"] in FAILURE_TYPES)


# --------------------------------------------------------------------------- #
# U4  cp_features
# --------------------------------------------------------------------------- #
def _synthetic_cp():
    x = np.arange(51) / 50.0  # 0, 0.02, ..., 1.0 ; x[15] == 0.3 exactly
    cp = np.interp(x, [0.0, 0.3, 1.0], [1.0, -2.5, 0.2])
    return x, cp


def test_u4_cp_features_peak_separation_slope():
    x, cp = _synthetic_cp()
    tau = 0.7 - x  # positive before x=0.7, negative after
    f = cp_features(x, cp, tau)
    assert set(f.keys()) == FIELD_KEYS
    assert f["cp_peak_x"] == pytest.approx(0.3, abs=1e-9)
    assert f["cp_peak_val"] == pytest.approx(-2.5, abs=1e-9)
    assert abs(f["sep_x"] - 0.7) <= 0.021
    assert f["pressure_recovery_slope"] > 0
    # cp is exactly linear from the peak to the TE: slope = (0.2 - (-2.5)) / (1 - 0.3)
    assert f["pressure_recovery_slope"] == pytest.approx(2.7 / 0.7, rel=1e-6)
    for v in f.values():
        assert isinstance(v, float) and math.isfinite(v)


def test_u4_cp_features_no_separation_gives_one():
    x, cp = _synthetic_cp()
    tau = np.full_like(x, 0.02)
    f = cp_features(x, cp, tau)
    assert f["sep_x"] == pytest.approx(1.0, abs=1e-12)
    assert f["cp_peak_x"] == pytest.approx(0.3, abs=1e-9)
    assert f["cp_peak_val"] == pytest.approx(-2.5, abs=1e-9)


def test_u4_cp_features_separation_before_peak_is_ignored():
    x, cp = _synthetic_cp()
    tau = np.full_like(x, 0.02)
    tau[(x > 0.05) & (x < 0.15)] = -0.02  # a negative patch *before* the peak
    f = cp_features(x, cp, tau)
    assert f["sep_x"] == pytest.approx(1.0, abs=1e-12)


def test_u4_cp_features_rejects_short_input():
    with pytest.raises(ValueError):
        cp_features(np.array([0.0, 1.0]), np.array([0.0, 0.0]), np.array([1.0, 1.0]))


@pytest.mark.parametrize("which", ["x", "cp", "tau"])
def test_u4_cp_features_rejects_nan(which):
    x, cp = _synthetic_cp()
    tau = 0.7 - x
    arrays = {"x": x.copy(), "cp": cp.copy(), "tau": tau.copy()}
    arrays[which][10] = np.nan
    with pytest.raises(ValueError):
        cp_features(arrays["x"], arrays["cp"], arrays["tau"])


# --------------------------------------------------------------------------- #
# U5  CaseResult
# --------------------------------------------------------------------------- #
def _hash_for(p: Params, o: RunOptions = RunOptions()) -> str:
    return input_hash(p, o)


def _ok_result(runtime_s: float = 45.2, **overrides) -> CaseResult:
    h = _hash_for(Params(0.3, 2, 0.04))
    fields = dict(
        case_id=h[:12],
        input_hash=h,
        status="ok",
        forces={"Cl": -0.9, "Cd": 0.1 + 0.2},  # 0.30000000000000004 -> exercises repr floats
        trust={k: True for k in sorted(TRUST_KEYS)},
        field_summary={"cp_peak_x": 0.31, "cp_peak_val": -2.4, "sep_x": 0.83, "pressure_recovery_slope": 3.1},
        failure=None,
        provenance={
            "mesh_hash": "ab" * 32,
            "solver_build": {"app_version": "2.5.0", "app_zip_sha256": "unknown", "api_info_hash": "cd" * 32},
        },
        iterations=812,
        runtime_s=runtime_s,
    )
    fields.update(overrides)
    return CaseResult(**fields)


def _failed_result() -> CaseResult:
    h = _hash_for(Params(0.001, 0, 0))
    return CaseResult(
        case_id=h[:12],
        input_hash=h,
        status="failed",
        forces=None,
        trust={k: False for k in sorted(TRUST_KEYS)},
        field_summary=None,
        failure={"type": "geometry_invalid", "stage": "geometry", "evidence": "h_c=0.001 outside [0.03, 2.0]"},
        provenance={
            "mesh_hash": None,
            "solver_build": {"app_version": "2.5.0", "app_zip_sha256": "unknown", "api_info_hash": "cd" * 32},
        },
        iterations=0,
        runtime_s=0.01,
    )


def test_u5_caseresult_json_roundtrip_ok():
    r = _ok_result()
    s = r.to_json()
    assert isinstance(s, str)
    json.loads(s)  # valid JSON
    assert CaseResult.from_json(s) == r


def test_u5_caseresult_json_roundtrip_failed():
    r = _failed_result()
    back = CaseResult.from_json(r.to_json())
    assert back == r
    assert back.status == "failed"
    assert back.forces is None
    assert back.field_summary is None
    assert set(back.trust.keys()) == TRUST_KEYS
    assert all(v is False for v in back.trust.values())
    assert back.failure["type"] in FAILURE_TYPES
    assert back.provenance["mesh_hash"] is None
    assert back.trusted is False


def test_u5_caseresult_trusted():
    assert _ok_result().trusted is True
    for k in sorted(TRUST_KEYS):
        trust = {kk: True for kk in sorted(TRUST_KEYS)}
        trust[k] = False
        assert _ok_result(trust=trust).trusted is False, k
    assert _failed_result().trusted is False
    # status failed with all-True flags is still not trusted
    weird = _ok_result(status="failed")
    assert weird.trusted is False


def _assert_keys_sorted(obj):
    if isinstance(obj, dict):
        keys = list(obj.keys())
        assert keys == sorted(keys), keys
        for v in obj.values():
            _assert_keys_sorted(v)
    elif isinstance(obj, list):
        for v in obj:
            _assert_keys_sorted(v)


def test_u5_caseresult_deterministic_json():
    r1 = _ok_result(runtime_s=45.2)
    r2 = _ok_result(runtime_s=99.9)
    d1 = r1.to_json(deterministic=True)
    d2 = r2.to_json(deterministic=True)
    assert isinstance(d1, str)
    assert d1 == d2, "runtime_s must not influence the deterministic form"
    obj = json.loads(d1)
    assert "runtime_s" not in obj
    _assert_keys_sorted(obj)
    assert "0.30000000000000004" in d1  # float written with repr precision
    # the non-deterministic form keeps runtime_s and differs between the two
    assert "runtime_s" in json.loads(r1.to_json())
    assert r1.to_json() != r2.to_json()


def test_u5_caseresult_contract_invariants():
    r = _ok_result()
    assert r.case_id == r.input_hash[:12]
    assert HEX64.match(r.input_hash)
    assert "input_hash" not in r.provenance
    assert set(r.provenance.keys()) == {"mesh_hash", "solver_build"}


# --------------------------------------------------------------------------- #
# U6  run_case with out-of-range inputs (no mesh, no store, no exception)
# --------------------------------------------------------------------------- #
def _assert_geometry_invalid(r, store_dir):
    assert isinstance(r, CaseResult)
    assert r.status == "failed"
    assert isinstance(r.failure, dict)
    assert r.failure["type"] == "geometry_invalid"
    assert r.failure["stage"] == "geometry"
    assert isinstance(r.failure["evidence"], str) and 0 < len(r.failure["evidence"]) <= 300
    assert r.provenance["mesh_hash"] is None
    assert r.forces is None
    assert r.field_summary is None
    assert set(r.trust.keys()) == TRUST_KEYS
    assert all(v is False for v in r.trust.values())
    assert r.trusted is False
    assert HEX64.match(r.input_hash) and r.case_id == r.input_hash[:12]
    assert isinstance(r.iterations, int)
    assert isinstance(r.runtime_s, float)
    # nothing may be created under the store for a rejected case
    if store_dir.exists():
        assert [p for p in store_dir.iterdir() if p.is_dir()] == []


def test_u6_run_case_params_out_of_range(tmp_path):
    store_dir = tmp_path / "store"
    r = run_case(Params(0.001, 0, 0), store=store_dir)
    _assert_geometry_invalid(r, store_dir)
    assert r.input_hash == input_hash(Params(0.001, 0, 0))


@pytest.mark.parametrize(
    "p",
    [
        Params(2.5, 0, 0),      # h_c > 2.0
        Params(0.3, 13, 0),     # alpha > 12
        Params(0.3, -3, 0),     # alpha < -2
        Params(0.3, 0, 0.1),    # camber > 0.09
        Params(0.3, 0, -0.01),  # camber < 0
        Params(float("nan"), 0, 0),
    ],
)
def test_u6_run_case_each_param_bound(tmp_path, p):
    store_dir = tmp_path / "store"
    _assert_geometry_invalid(run_case(p, store=store_dir), store_dir)


def test_u6_run_case_options_out_of_range(tmp_path):
    store_dir = tmp_path / "store"
    r = run_case(Params(0.3, 2, 0.04), RunOptions(gap_refine=9), store=store_dir)
    _assert_geometry_invalid(r, store_dir)
    ev = r.failure["evidence"].lower()
    assert "gap_refine" in ev or "option" in ev, ev


@pytest.mark.parametrize(
    "o",
    [RunOptions(gap_refine=-1), RunOptions(relax_step=3), RunOptions(iter_mult=0), RunOptions(iter_mult=5)],
)
def test_u6_run_case_each_option_bound(tmp_path, o):
    store_dir = tmp_path / "store"
    _assert_geometry_invalid(run_case(Params(0.3, 2, 0.04), o, store=store_dir), store_dir)


# --------------------------------------------------------------------------- #
# U7  SETTINGS / FAILURE_TYPES
# --------------------------------------------------------------------------- #
def test_u7_failure_types():
    assert tuple(FAILURE_TYPES) == ("geometry_invalid", "mesh_invalid", "diverged", "not_settled", "timeout")
    assert len(FAILURE_TYPES) == 5
    assert len(set(FAILURE_TYPES)) == 5


def test_u7_settings_keys_match_spec():
    assert isinstance(SETTINGS, dict)
    assert set(SETTINGS.keys()) == {
        "U_inf", "chord", "Re", "naca_p", "naca_t", "n_surface", "domain", "mesh", "solver", "trust",
    }
    assert set(SETTINGS["domain"].keys()) == {"x_min", "x_max", "y_max"}
    assert set(SETTINGS["mesh"].keys()) == {
        "far", "wing", "gap", "wake", "bl_layers", "bl_first", "bl_ratio", "algorithm", "gap_refine_factor",
    }
    assert set(SETTINGS["solver"].keys()) == {"end_time", "write_interval", "residual_control", "relax"}
    assert set(SETTINGS["solver"]["residual_control"].keys()) == {"p", "U", "k", "omega"}
    assert isinstance(SETTINGS["solver"]["relax"], list) and len(SETTINGS["solver"]["relax"]) == 3
    for lvl in SETTINGS["solver"]["relax"]:
        assert set(lvl.keys()) == {"p", "U", "turb"}
    assert set(SETTINGS["trust"].keys()) == {
        "yplus_min", "yplus_max", "max_nonortho", "max_skew", "force_settle_window", "force_settle_tol",
    }


def test_u7_settings_fixed_values():
    # values the spec pins explicitly (not subject to the "same order of magnitude" tuning clause)
    assert SETTINGS["naca_p"] == 0.4
    assert SETTINGS["naca_t"] == 0.12
    assert SETTINGS["chord"] == 1.0
    assert SETTINGS["mesh"]["gap_refine_factor"] == 0.6
    assert SETTINGS["domain"]["x_min"] < 0 < SETTINGS["domain"]["x_max"]
    assert SETTINGS["domain"]["y_max"] > 0
    assert SETTINGS["n_surface"] >= 3
    assert SETTINGS["Re"] > 0 and SETTINGS["U_inf"] > 0
    json.dumps(SETTINGS, sort_keys=True)  # must be JSON-serialisable (it participates in the hash)
