"""Interview adapter boundaries and real smoke evidence (U01/U05/U13/U17/U19/U20).

Unmarked tests exercise rejection before invoking physical tools. Real smoke
tests below use the frozen pilot profile and never substitute synthetic data.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
from pathlib import Path
import re

import numpy as np
import pytest


@pytest.fixture
def invoke():
    return importlib.import_module("pipeline.exp_layer.aero2d.interview").invoke


@pytest.fixture
def settings():
    return {"model": "M0", "flow": {"U_inf": 1.0, "chord": 1.0, "Re": 100.0},
            "domain": {"x_min": -5.0, "x_max": 10.0, "y_min": -5.0, "y_max": 5.0},
            "mesh": {"far": 0.8, "wing": 0.025, "mesh_scale": 1.0},
            "solver": {"end_time": 400, "delta_t": 1.0},
            "geometry": {"x": [0.0, 0.25, 0.5, 0.75, 1.0], "min_thickness": 0.001,
                         "max_thickness": 0.3, "coordinate_bounds": [-0.3, 0.3]},
            "exploration": {"mesh_scale": [0.7, 1.3]}}


@pytest.fixture
def inputs(tmp_path, settings):
    return {"candidate": {"representation": "parameterized", "parameters": {"alpha_deg": 2.0}},
            "settings": settings, "case_dir": str(tmp_path / "case"), "artifacts": {}}


@pytest.fixture
def context(settings):
    return {"snapshot": {"settings": copy.deepcopy(settings)}, "candidate_id": "candidate-1",
            "design_hash": "design-1", "stage_id": "M0", "protocol_hash": "protocol-1", "role": "formal"}


@pytest.mark.parametrize("tool", ["mesh", "solve", "post"])
def test_missing_real_upstream_artifact_stops_before_tool(invoke, inputs, context, tool):
    with pytest.raises(ValueError):
        invoke(tool, inputs, context)
    case = Path(inputs["case_dir"])
    assert not case.exists() or not list(case.rglob("*"))


@pytest.mark.parametrize("tool,artifact", [("mesh", "geometry"), ("solve", "mesh"), ("post", "solution")])
def test_fabricated_active_handle_is_not_an_upstream_artifact(invoke, inputs, context, tool, artifact):
    inputs["artifacts"][artifact] = str(Path(inputs["case_dir"]) / "nonexistent.json")
    with pytest.raises(ValueError):
        invoke(tool, inputs, context)


def test_agent_cannot_replace_host_formal_settings(invoke, inputs, context):
    inputs["settings"]["solver"]["end_time"] = 1
    with pytest.raises(ValueError):
        invoke("geometry", inputs, context)
    assert not Path(inputs["case_dir"]).exists()


def test_unknown_physical_model_is_rejected_before_case_creation(invoke, inputs, context):
    inputs["settings"]["model"] = "unsupported-M9"
    context["snapshot"]["settings"]["model"] = "unsupported-M9"
    context["stage_id"] = "M9"
    with pytest.raises(ValueError):
        invoke("geometry", inputs, context)
    assert not Path(inputs["case_dir"]).exists()


def test_external_artifact_path_is_not_a_stage_dependency(invoke, inputs, context, tmp_path):
    external = tmp_path / "external-geometry.json"
    external.write_text('{}')
    inputs["artifacts"]["geometry"] = str(external)
    with pytest.raises(ValueError):
        invoke("mesh", inputs, context)


def test_unknown_tool_is_rejected(invoke, inputs, context):
    with pytest.raises(ValueError):
        invoke("python", inputs, context)


def _point_to_shape_distance(x, y, shape):
    point = np.array([x, y])
    best = float("inf")
    for surface in ("upper", "lower"):
        coords = np.column_stack([shape["x"], shape[surface]])
        for start, end in zip(coords[:-1], coords[1:], strict=True):
            edge = end - start
            fraction = np.clip(np.dot(point - start, edge) / np.dot(edge, edge), 0, 1)
            best = min(best, float(np.linalg.norm(point - (start + fraction * edge))))
    return best


def _mesh_wing_points(case):
    """Read OpenFOAM ASCII mesh data independently of adapter manifests."""
    mesh = case / "constant" / "polyMesh"
    points = np.asarray([[float(v) for v in match.split()] for match in
                         re.findall(r"^\(([-+0-9.eE ]+)\)$", (mesh / "points").read_text(), re.MULTILINE)])
    faces = [[int(v) for v in match.split()] for match in
             re.findall(r"^\d+\(([0-9 ]+)\)$", (mesh / "faces").read_text(), re.MULTILINE)]
    boundary = (mesh / "boundary").read_text()
    wing = re.search(r"\bwing\s*\{([^}]+)\}", boundary)
    assert wing, "a real wing patch must exist"
    start = int(re.search(r"startFace\s+(\d+)", wing.group(1)).group(1))
    count = int(re.search(r"nFaces\s+(\d+)", wing.group(1)).group(1))
    vertices = {index for face in faces[start:start + count] for index in face}
    assert vertices
    return points[sorted(vertices)], len(points)


@pytest.fixture(scope="module")
def real_interview_cases(tmp_path_factory):
    """Real OpenFOAM stage execution; absence is failure, never a passing skip."""
    root = Path(__file__).resolve().parents[3]
    frozen_path = root / "OUTPUTs/20260924_133000-produce-level0-3-smokerun/pilot/frozen_profiles.json"
    assert frozen_path.is_file(), "a separately calibrated, frozen pilot profile is required"
    frozen = json.loads(frozen_path.read_text())
    invoke = importlib.import_module("pipeline.exp_layer.aero2d.interview").invoke
    directory = tmp_path_factory.mktemp("real-interview-smoke")
    param = copy.deepcopy(frozen["initialization"]["parameterized"][1])
    naca = copy.deepcopy(frozen["initialization"]["explicit_shape"][1])
    first, second = copy.deepcopy(naca), copy.deepcopy(naca)
    for index, x in enumerate(naca["shape"]["x"]):
        if index in (0, len(naca["shape"]["x"]) - 1):
            continue
        bump = 0.006 * math.sin(math.pi * x) ** 2
        first["shape"]["upper"][index] += bump
        second["shape"]["lower"][index] -= bump
    cases = [("m0-parameterized", "M0", param, 1.0), ("m1-parameterized", "M1", param, 1.0),
             ("naca-explicit", "M0", naca, 1.0), ("shape-a-fine", "M0", first, 0.7),
             ("shape-a-coarse", "M0", first, 1.3), ("shape-b", "M0", second, 1.0)]
    results = {}
    for name, stage, candidate, mesh_scale in cases:
        settings = copy.deepcopy(frozen["profiles"][stage])
        settings["mesh"]["mesh_scale"] = mesh_scale
        case = directory / name
        context = {"snapshot": {"settings": settings}, "candidate_id": name,
                   "design_hash": hashlib.sha256(json.dumps(candidate, sort_keys=True).encode()).hexdigest(),
                   "stage_id": stage, "protocol_hash": "independent-real-smoke",
                   "role": "exploration" if mesh_scale != 1.0 else "formal"}
        inputs = {"candidate": candidate, "settings": copy.deepcopy(settings),
                  "case_dir": str(case), "artifacts": {}}
        responses = []
        for tool in ("geometry", "mesh", "solve", "post"):
            response = invoke(tool, inputs, context)
            responses.append(response)
            assert response["source"] == "live"
            assert response["status"] not in {"failed", "skipped", "unsupported"}
            inputs["artifacts"].update(copy.deepcopy(response["artifacts"]))
        (case / "independent-test-responses.json").write_text(json.dumps(responses, indent=2))
        results[name] = {"case": case, "candidate": candidate, "settings": settings, "responses": responses}
    return results


@pytest.mark.smoke
def test_real_physical_stages_produce_mesh_solver_and_post_evidence(real_interview_cases):
    for item in real_interview_cases.values():
        case, responses = item["case"], item["responses"]
        for response in responses:
            for filename in response["artifacts"].values():
                artifact = Path(filename).resolve()
                assert artifact.is_file() and artifact.is_relative_to(case.resolve())
        for name in ("points", "faces", "boundary", "owner", "neighbour"):
            assert (case / "constant" / "polyMesh" / name).stat().st_size > 0
        app = item["settings"]["solver"]["application"]
        log = (case / ("log." + app)).read_text()
        assert "Solving for" in log and "End" in log
        assert list((case / "postProcessing").rglob("*coefficient*.dat"))
        post = responses[-1]
        assert math.isfinite(post["raw_metrics"]["Cl"]) and math.isfinite(post["raw_metrics"]["Cd"])
        assert all(post["diagnostics"].get(key) is True for key in item["settings"]["numerical_checks"])
        assert sum(response["cost"]["solver_calls"] for response in responses) == 1
        assert sum(response["cost"]["tool_seconds"] for response in responses) > 0


@pytest.mark.smoke
def test_m1_changes_actual_physics_to_time_evolution(real_interview_cases):
    m0, m1 = [real_interview_cases[name] for name in ("m0-parameterized", "m1-parameterized")]
    assert m0["settings"]["flow"] == m1["settings"]["flow"]
    control0 = (m0["case"] / "system/controlDict").read_text()
    control1 = (m1["case"] / "system/controlDict").read_text()
    assert re.search(r"application\s+simpleFoam\s*;", control0)
    assert re.search(r"application\s+(?:pimpleFoam|icoFoam)\s*;", control1)
    schemes0 = (m0["case"] / "system/fvSchemes").read_text()
    schemes1 = (m1["case"] / "system/fvSchemes").read_text()
    assert "steadyState" in schemes0 and "steadyState" not in schemes1
    assert any(name in schemes1 for name in ("Euler", "backward", "CrankNicolson"))
    for item in (m0, m1):
        turbulence = item["case"] / "constant/turbulenceProperties"
        assert "laminar" in turbulence.read_text()
        velocity = (item["case"] / "0/U").read_text()
        assert "freestreamVelocity" in velocity and "ground" not in velocity.lower()


@pytest.mark.smoke
def test_explicit_shapes_reach_real_mesh_boundary_and_pressure_samples(real_interview_cases):
    for name in ("shape-a-fine", "shape-a-coarse", "shape-b", "naca-explicit"):
        item = real_interview_cases[name]
        shape = item["candidate"]["shape"]
        manifest = json.loads((item["case"] / "geometry.json").read_text())
        for key in ("x", "upper", "lower"):
            assert np.allclose(manifest["shape"][key], shape[key], rtol=0, atol=1e-11)
        points, _ = _mesh_wing_points(item["case"])
        assert max(_point_to_shape_distance(point[0], point[1], shape) for point in points) < 1e-8
        surface = json.loads((item["case"] / "surface-samples.json").read_text())
        samples = surface["samples"]
        assert {row["surface"] for row in samples} == {"upper", "lower"}
        assert max(_point_to_shape_distance(row["x"], row["y"], shape) for row in samples) < 1e-8
        assert all(math.isfinite(row["p"]) and math.isfinite(row["Cp"]) for row in samples)
    first = json.loads((real_interview_cases["shape-a-fine"]["case"] / "geometry.json").read_text())
    second = json.loads((real_interview_cases["shape-b"]["case"] / "geometry.json").read_text())
    assert first["shape_hash"] != second["shape_hash"]


@pytest.mark.smoke
def test_authorized_mesh_controls_change_actual_mesh_not_only_logs(real_interview_cases):
    fine = real_interview_cases["shape-a-fine"]
    coarse = real_interview_cases["shape-a-coarse"]
    assert fine["candidate"] == coarse["candidate"]
    _, fine_count = _mesh_wing_points(fine["case"])
    _, coarse_count = _mesh_wing_points(coarse["case"])
    assert fine_count > coarse_count
    assert fine["responses"][-1]["effective_settings"]["mesh"]["mesh_scale"] == 0.7
    assert coarse["responses"][-1]["effective_settings"]["mesh"]["mesh_scale"] == 1.3


@pytest.mark.smoke
def test_explicit_exported_naca_matches_parameterized_baseline(real_interview_cases):
    parameterized = real_interview_cases["m0-parameterized"]
    explicit = real_interview_cases["naca-explicit"]
    first = json.loads((parameterized["case"] / "geometry.json").read_text())
    second = json.loads((explicit["case"] / "geometry.json").read_text())
    assert np.allclose(first["coordinates"], second["coordinates"], rtol=0, atol=1e-11)
    assert parameterized["responses"][-1]["raw_metrics"]["Cl"] == pytest.approx(explicit["responses"][-1]["raw_metrics"]["Cl"], abs=1e-7)
