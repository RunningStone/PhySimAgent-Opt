"""Real free-air laminar airfoil experiment, with four file-backed stages.

The caller owns the worker deadline and process group. External commands here
inherit that group: there is deliberately no second timeout/session owner.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import time

import numpy as np

from pipeline.agentloop.environment import ConfigurationError, ProtocolError
from .geometry import shape_coords, validate_shape

ADAPTER_VERSION = "interview-laminar-v1"
TOOLS = ("geometry", "mesh", "solve", "post")


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _number(value, name, *, positive=False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ConfigurationError(f"{name} must be a finite numeric literal")
    if positive and value <= 0:
        raise ConfigurationError(f"{name} must be positive")
    return float(value)


def _path(case: Path, value) -> Path:
    path = Path(value).resolve()
    if not path.is_relative_to(case) or not path.is_file():
        raise ProtocolError(f"missing or external stage artifact: {value}")
    return path


def _load(case, artifacts, stage, settings_hash, candidate_hash):
    if stage not in artifacts:
        raise ProtocolError(f"{stage} artifact is required")
    path = _path(case, artifacts[stage])
    try:
        data = json.loads(path.read_text())
    except (ValueError, OSError) as exc:
        raise ProtocolError(f"unreadable {stage} artifact") from exc
    if (data.get("stage") != stage or data.get("settings_hash") != settings_hash
            or data.get("candidate_hash") != candidate_hash or not data.get("files")):
        raise ProtocolError(f"{stage} manifest identity or file evidence mismatch")
    for relative, checksum in data["files"].items():
        source = _path(case, case / relative)
        if _sha(source) != checksum:
            raise ProtocolError(f"changed {stage} source file: {relative}")
    return data


def _manifest(case, stage, settings_hash, candidate_hash, files, **data):
    path = case / f"{stage}.json"
    path.write_text(json.dumps({"stage": stage, "settings_hash": settings_hash,
        "candidate_hash": candidate_hash, "adapter_version": ADAPTER_VERSION,
        "files": {str(p.relative_to(case)): _sha(p) for p in files}, **data}, indent=2, allow_nan=False))
    return str(path)


def _foam(cls, name, body):
    return f"FoamFile\n{{version 2.0; format ascii; class {cls}; object {name};}}\n{body}\n"


def _run(case: Path, app: str, *args):
    log = case / f"log.{app}"
    command = shlex.join([app, "-case", str(case), *map(str, args)])
    with log.open("a") as stream:
        stream.write(f"COMMAND {command}\n")
        stream.flush()
        result = subprocess.run(["openfoam2512", "-c", command], cwd=case, stdout=stream,
                                stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"{app} failed ({result.returncode}); see {log}")
    return log


def _mesh(case: Path, geometry: dict, settings: dict):
    import gmsh

    c = settings["flow"]["chord"]
    coords = np.asarray(geometry["coordinates"]) * c
    m, d = settings["mesh"], settings["domain"]
    scale = m["mesh_scale"]
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        for opt in ("MeshSizeExtendFromBoundary", "MeshSizeFromPoints", "MeshSizeFromCurvature"):
            gmsh.option.setNumber("Mesh." + opt, 0)
        geo = gmsh.model.geo
        points = [geo.addPoint(float(x), float(y), 0) for x, y in coords[:-1]]
        wing = [geo.addLine(a, b) for a, b in zip(points, points[1:] + points[:1])]
        corners = [geo.addPoint(x*c, y*c, 0) for x, y in (
            (d["x_min"], d["y_min"]), (d["x_max"], d["y_min"]),
            (d["x_max"], d["y_max"]), (d["x_min"], d["y_max"]))]
        outer = [geo.addLine(a, b) for a, b in zip(corners, corners[1:] + corners[:1])]
        face = geo.addPlaneSurface([geo.addCurveLoop(outer), geo.addCurveLoop(wing)])
        geo.synchronize()
        field = gmsh.model.mesh.field
        distance = field.add("Distance")
        field.setNumbers(distance, "CurvesList", wing)
        field.setNumber(distance, "Sampling", 200)
        near = field.add("Threshold")
        for key, value in {"InField": distance, "SizeMin": m["wing"]*scale*c,
                           "SizeMax": m["far"]*scale*c, "DistMin": 0.08*c, "DistMax": 1.5*c}.items():
            field.setNumber(near, key, value)
        field.setAsBackgroundMesh(near)
        ext = geo.extrude([(2, face)], 0, 0, 1, numElements=[1], recombine=True)
        geo.synchronize()
        sides = {"wing": [], "farfield": []}
        for dim, tag in ext[2:]:
            if dim == 2:
                boundary = {abs(t) for _, t in gmsh.model.getBoundary([(2, tag)], oriented=False)}
                if boundary.intersection(wing):
                    sides["wing"].append(tag)
                elif boundary.intersection(outer):
                    sides["farfield"].append(tag)
        for name, tags in sides.items():
            gmsh.model.addPhysicalGroup(2, tags, name=name)
        gmsh.model.addPhysicalGroup(2, [face, ext[0][1]], name="frontAndBack")
        gmsh.model.addPhysicalGroup(3, [ext[1][1]], name="internal")
        gmsh.model.mesh.generate(3)
        gmsh.write(str(case / "mesh.msh"))
    finally:
        gmsh.finalize()
    _run(case, "gmshToFoam", case / "mesh.msh")
    for patch, kind in (("wing", "wall"), ("frontAndBack", "empty")):
        _run(case, "foamDictionary", "constant/polyMesh/boundary", "-entry", f"entry0/{patch}/type", "-set", kind)
    log = _run(case, "checkMesh").read_text()
    if "Mesh OK." not in log or "Failed" in log:
        raise RuntimeError("checkMesh did not certify Mesh OK")
    count = re.search(r"cells:\s+(\d+)", log)
    return {"mesh_ok": True, "cell_count": int(count.group(1)) if count else None,
            "mesh_scale": scale, "geometry_hash": geometry["shape_hash"]}


def _write_case(case, candidate, settings):
    model, flow, solver = settings["model"], settings["flow"], settings["solver"]
    alpha = math.radians(candidate["parameters"]["alpha_deg"])
    ux, uy = flow["U_inf"]*math.cos(alpha), flow["U_inf"]*math.sin(alpha)
    velocity = f"({ux:.14g} {uy:.14g} 0)"
    for directory in ("0", "constant", "system"):
        (case / directory).mkdir(exist_ok=True)
    (case / "0/U").write_text(_foam("volVectorField", "U", f"""
dimensions [0 1 -1 0 0 0 0]; internalField uniform {velocity};
boundaryField {{ farfield {{type freestreamVelocity; freestreamValue uniform {velocity};}}
wing {{type noSlip;}} frontAndBack {{type empty;}} }}"""))
    (case / "0/p").write_text(_foam("volScalarField", "p", """
dimensions [0 2 -2 0 0 0 0]; internalField uniform 0;
boundaryField { farfield {type freestreamPressure; freestreamValue uniform 0;}
wing {type zeroGradient;} frontAndBack {type empty;} }"""))
    nu = flow["U_inf"]*flow["chord"]/flow["Re"]
    (case / "constant/transportProperties").write_text(_foam("dictionary", "transportProperties", f"transportModel Newtonian; nu {nu};"))
    (case / "constant/turbulenceProperties").write_text(_foam("dictionary", "turbulenceProperties", "simulationType laminar;"))
    app = "simpleFoam" if model == "M0" else "pimpleFoam"
    end, dt = solver["end_time"], solver["delta_t"]
    steps = int(round(end/dt))
    (case / "system/controlDict").write_text(_foam("dictionary", "controlDict", f"""
application {app}; startFrom startTime; startTime 0; stopAt endTime; endTime {end}; deltaT {dt};
writeControl timeStep; writeInterval {steps}; purgeWrite 0; writeFormat ascii;
writePrecision 12; writeCompression off; timeFormat general; timePrecision 10; runTimeModifiable false;
functions {{
forceCoeffs {{type forceCoeffs; libs (forces); writeControl timeStep; writeInterval 1; log no;
patches (wing); rho rhoInf; rhoInf 1; liftDir ({-math.sin(alpha):.14g} {math.cos(alpha):.14g} 0);
dragDir ({math.cos(alpha):.14g} {math.sin(alpha):.14g} 0); pitchAxis (0 0 1); CofR ({0.25*flow['chord']} 0 0);
magUInf {flow['U_inf']}; lRef {flow['chord']}; Aref {flow['chord']};}}
surfaces {{type surfaces; libs (sampling); writeControl writeTime; surfaceFormat raw;
interpolationScheme cell; fields (p); surfaces {{wing {{type patch; patches (wing); interpolate false;}}}}}}
}}"""))
    ddt = "steadyState" if model == "M0" else "Euler"
    convection = "bounded Gauss linearUpwind grad(U)" if model == "M0" else "Gauss linearUpwind grad(U)"
    (case / "system/fvSchemes").write_text(_foam("dictionary", "fvSchemes", f"""
ddtSchemes {{default {ddt};}} gradSchemes {{default Gauss linear;}}
divSchemes {{default none; div(phi,U) {convection}; div((nuEff*dev2(T(grad(U))))) Gauss linear;}}
laplacianSchemes {{default Gauss linear corrected;}} interpolationSchemes {{default linear;}}
snGradSchemes {{default corrected;}} fluxRequired {{default no; p;}}"""))
    algorithm = ("SIMPLE {nNonOrthogonalCorrectors 1; pRefCell 0; pRefValue 0;}\n"
                 "relaxationFactors {fields {p 0.3;} equations {U 0.7;}}") if model == "M0" else (
                 "PIMPLE {nOuterCorrectors 1; nCorrectors 2; nNonOrthogonalCorrectors 1; pRefCell 0; pRefValue 0;}")
    (case / "system/fvSolution").write_text(_foam("dictionary", "fvSolution", """
solvers {
p {solver GAMG; smoother GaussSeidel; tolerance 1e-9; relTol 0.01;}
pFinal {$p; relTol 0;}
U {solver smoothSolver; smoother symGaussSeidel; tolerance 1e-10; relTol 0.01;}
UFinal {$U; relTol 0;}
}
""" + algorithm))
    return app


def _post(case, geometry, mesh, solution, settings):
    coeff = _path(case, case / solution["coefficients"])
    values = np.loadtxt(coeff, comments="#", ndmin=2)
    if values.shape[1] < 5 or not np.isfinite(values).all():
        raise RuntimeError("force coefficient table is missing or nonfinite")
    solver = settings["solver"]
    if abs(float(values[-1, 0])-solver["end_time"]) > max(1e-8, solver["delta_t"]*1e-6):
        raise RuntimeError("force history does not reach the frozen end time")
    log = (case / solution["solver_log"]).read_text()
    blocks = re.split(r"^Time = ([0-9.eE+-]+)\s*$", log, flags=re.M)
    initial, final = {}, {}
    for field, a, b in re.findall(r"Solving for (\w+), Initial residual = ([0-9.eE+-]+), Final residual = ([0-9.eE+-]+)", blocks[-1]):
        initial.setdefault(field, float(a))
        final[field] = float(b)
    diagnostics = {**mesh["diagnostics"], "physics_model": settings["model"],
        "solver_application": solution["solver_application"], "initial_residuals": initial,
        "final_residuals": final, "initial_condition": "uniform freestream U; p=0; no restart"}
    if settings["model"] == "M0":
        window = int(solver["sample_window"])
        samples = values[-window:]
        drift = np.ptp(samples[:, [1, 4]], axis=0)
        diagnostics.update(residual_ok=bool(initial and all(initial.get(k, math.inf) <= solver["residual_tolerance"] for k in ("p", "Ux", "Uy"))),
            force_settled=bool(len(samples) == window and max(drift) <= solver["force_tolerance"]),
            force_window_range={"Cd": float(drift[0]), "Cl": float(drift[1])},
            sampling_window=[float(samples[0, 0]), float(samples[-1, 0])])
    else:
        begin, end = solver["averaging_window"]
        samples = values[(values[:, 0] >= begin-1e-8) & (values[:, 0] <= end+1e-8)]
        co = [float(v) for v in re.findall(r"Courant Number mean: [0-9.eE+-]+ max: ([0-9.eE+-]+)", log)]
        complete = len(samples) >= 2 and samples[0, 0] <= begin+solver["delta_t"]+1e-8 and samples[-1, 0] >= end-1e-8
        diagnostics.update(residual_ok=bool(final and all(final.get(k, math.inf) <= solver["linear_residual_tolerance"] for k in ("p", "Ux", "Uy"))),
            time_window_complete=bool(complete), sampling_window=[begin, end],
            max_courant=max(co) if co else None, courant_ok=bool(co and max(co) <= solver["max_courant"]),
            interpretation="finite-time mean from a uniform initial condition; not steady-state convergence")
    if len(samples) == 0:
        raise RuntimeError("no samples in the declared averaging window")
    metrics = {"Cl": float(samples[:, 4].mean()), "Cd": float(samples[:, 1].mean())}
    diagnostics["force_finite"] = bool(all(math.isfinite(v) for v in metrics.values()))
    raw = np.loadtxt(_path(case, case / solution["surface_pressure"]), comments="#", ndmin=2)
    if raw.shape[1] != 4 or not np.isfinite(raw).all():
        raise RuntimeError("missing physical surface pressure samples")
    contour = np.asarray(geometry["coordinates"])*settings["flow"]["chord"]
    starts, ends = contour[:-1], contour[1:]
    vectors = ends-starts
    q = raw[:, :2]
    t = np.clip(((q[:, None]-starts)*vectors).sum(axis=2)/(vectors*vectors).sum(axis=1), 0, 1)
    distance = np.linalg.norm(q[:, None]-(starts+t[:, :, None]*vectors), axis=2)
    indices = np.argmin(distance, axis=1)
    mismatch = float(distance[np.arange(len(q)), indices].max())
    if mismatch > 1e-6*settings["flow"]["chord"]:
        raise RuntimeError("solver pressure samples do not lie on submitted geometry")
    surface_path = case / "surface-samples.json"
    surface_path.write_text(json.dumps({"geometry_hash": geometry["shape_hash"], "max_boundary_distance": mismatch,
        "samples": [{"x": float(row[0]), "y": float(row[1]), "p": float(row[3]),
                     "Cp": float(2*row[3]/settings['flow']['U_inf']**2),
                     "surface": "upper" if index < len(geometry["shape"]["x"])-1 else "lower", "segment": int(index)}
                    for row, index in zip(raw, indices)]}, indent=2))
    diagnostics.update(surface_geometry_ok=True, surface_geometry_hash=geometry["shape_hash"],
                       sample_count=len(samples), coefficient_std={"Cl": float(samples[:, 4].std()), "Cd": float(samples[:, 1].std())})
    return metrics, diagnostics, surface_path


def invoke(tool: str, inputs: dict, context: dict) -> dict:
    """Execute one stage under the exact host snapshot; return actual cost/evidence."""
    start = time.monotonic()
    if tool not in TOOLS:
        raise ProtocolError(f"unsupported interview tool: {tool}")
    settings = context.get("snapshot", {}).get("settings")
    if not isinstance(settings, dict) or settings != inputs.get("settings"):
        raise ProtocolError("input settings differ from the host snapshot")
    if settings.get("model") not in ("M0", "M1"):
        raise ConfigurationError("unsupported physical model")
    if any(not isinstance(settings.get(key), dict) for key in ("flow", "domain", "mesh", "solver", "geometry")):
        raise ConfigurationError("incomplete physical/numerical profile")
    for key in ("U_inf", "chord", "Re"):
        _number(settings["flow"][key], key, positive=True)
    for key in ("far", "wing", "mesh_scale"):
        _number(settings["mesh"][key], key, positive=True)
    for key in ("end_time", "delta_t"):
        _number(settings["solver"][key], key, positive=True)
    if settings["model"] != context.get("stage_id", settings["model"]):
        raise ProtocolError("snapshot model differs from the current stage")
    app = "simpleFoam" if settings["model"] == "M0" else "pimpleFoam"
    if settings["solver"].get("application", app) != app:
        raise ConfigurationError("solver application does not implement the selected model")
    candidate = inputs["candidate"]
    parameters = candidate.get("parameters", {})
    if set(parameters) != {"alpha_deg"}:
        raise ProtocolError("the design parameters must contain only alpha_deg")
    _number(parameters["alpha_deg"], "alpha_deg")
    shape = candidate.get("shape")
    if candidate.get("representation") == "parameterized":
        if shape is not None:
            raise ProtocolError("parameterized candidates cannot replace their NACA0012 shape")
        x = np.asarray(settings["geometry"]["x"], dtype=float)
        y = 0.6*(0.2969*np.sqrt(x)-0.126*x-0.3516*x**2+0.2843*x**3-0.1036*x**4)
        y[0] = y[-1] = 0
        shape = {"x": x.tolist(), "upper": y.tolist(), "lower": (-y).tolist()}
    elif candidate.get("representation") != "explicit_shape":
        raise ProtocolError("unsupported geometry representation")
    shape = validate_shape(shape, settings["geometry"])
    settings_hash, candidate_hash = _hash(settings), _hash({"shape": shape, "parameters": parameters})
    case = Path(inputs["case_dir"]).resolve()
    artifacts = inputs.get("artifacts", {})
    previous = {}
    for upstream_tool in TOOLS[:TOOLS.index(tool)]:
        upstream = "solution" if upstream_tool == "solve" else upstream_tool
        previous[upstream] = _load(case, artifacts, upstream, settings_hash, candidate_hash)
    case.mkdir(parents=True, exist_ok=True)
    diagnostics, metrics, files = {}, None, []
    solver_calls = int(tool == "solve")
    try:
        if tool == "geometry":
            x, y = shape_coords(shape)
            coordinates = np.column_stack([x, y])
            source = case / "geometry-coordinates.csv"
            np.savetxt(source, coordinates, delimiter=",", header="x,y", comments="")
            output = _manifest(case, tool, settings_hash, candidate_hash, [source], shape=shape,
                shape_hash=_hash(shape), coordinates=coordinates.tolist(), diagnostics={"geometry_ok": True})
            diagnostics = {"geometry_ok": True}
        elif tool == "mesh":
            # gmshToFoam reads controlDict even before the solver begins.
            _write_case(case, candidate, settings)
            diagnostics = _mesh(case, previous["geometry"], settings)
            files = [case/"mesh.msh", case/"log.checkMesh", *sorted((case/"constant/polyMesh").iterdir())]
            output = _manifest(case, tool, settings_hash, candidate_hash, [p for p in files if p.is_file()],
                               diagnostics=diagnostics, geometry_artifact=str(case/"geometry.json"))
        elif tool == "solve":
            app = _write_case(case, candidate, settings)
            log = _run(case, app)
            text = log.read_text()
            if "End" not in text or "FOAM FATAL" in text or re.search(r"\bnan\b", text, re.I):
                raise RuntimeError("solver did not finish cleanly")
            coeff = next((case/"postProcessing/forceCoeffs").rglob("coefficient.dat"))
            pressure = next((case/"postProcessing/surfaces").rglob("p*.raw"))
            files = [log, coeff, pressure, *sorted((case/"system").iterdir()), case/"constant/transportProperties", case/"constant/turbulenceProperties", case/"0/U", case/"0/p"]
            diagnostics = {"solver_completed": True, "physics_model": settings["model"], "solver_application": app}
            output = _manifest(case, "solution", settings_hash, candidate_hash, files, coefficients=str(coeff.relative_to(case)),
                surface_pressure=str(pressure.relative_to(case)), solver_log=str(log.relative_to(case)),
                solver_application=app, diagnostics=diagnostics)
        else:
            metrics, diagnostics, surface = _post(case, previous["geometry"], previous["mesh"], previous["solution"], settings)
            output = _manifest(case, tool, settings_hash, candidate_hash, [surface], raw_metrics=metrics,
                               diagnostics=diagnostics, surface_samples=str(surface))
        result = {"status": "completed", "artifacts": {tool if tool != "solve" else "solution": output}, "diagnostics": diagnostics}
        if metrics is not None:
            result["raw_metrics"] = metrics
    except (OSError, RuntimeError, StopIteration, ValueError) as exc:
        result = {"status": "failed", "artifacts": {}, "diagnostics": {"failure_stage": tool, "reason": str(exc)}}
    elapsed = time.monotonic()-start
    result.update(source="live", settings_hash=settings_hash, effective_settings=settings,
        cost={"wall_seconds": elapsed, "tool_seconds": elapsed, "solver_calls": solver_calls})
    return result
