"""External-tool I/O layer of pipeline.exp_layer.aero2d (batch 1 case code): gmsh meshing, OpenFOAM case writing,
one ``openfoam2512 -c`` subprocess per command, and post-processing parsers.
Everything written lands inside the case directory passed by the caller.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import signal
import subprocess
import time
from pathlib import Path

import numpy as np

from pipeline.exp_layer.aero2d.contract import Params, RunOptions, cp_features, naca4, wing_coords

OPENFOAM = "openfoam2512"


# --------------------------------------------------------------------------- #
# mesh
# --------------------------------------------------------------------------- #
def build_mesh(params: Params, options: RunOptions, settings: dict, msh_path: Path) -> str:
    """Single-threaded gmsh mesh of the box domain around the wing, one-cell-thick
    extrusion, MSH 2.2 for gmshToFoam. Returns sha256 of the written file."""
    import gmsh

    m, d = settings["mesh"], settings["domain"]
    gap = m["gap"] * m["gap_refine_factor"] ** options.gap_refine
    x, y = wing_coords(params, settings["n_surface"])
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.option.setNumber("Mesh.Algorithm", m["algorithm"])
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        geo = gmsh.model.geo
        n = len(x) - 1  # closing point dropped
        pts = [geo.addPoint(float(x[i]), float(y[i]), 0.0) for i in range(n)]
        le = n // 2
        upper = geo.addSpline(pts[: le + 1])
        lower = geo.addSpline(pts[le:] + [pts[0]])
        c = [geo.addPoint(d["x_min"], 0.0, 0.0), geo.addPoint(d["x_max"], 0.0, 0.0),
             geo.addPoint(d["x_max"], d["y_max"], 0.0), geo.addPoint(d["x_min"], d["y_max"], 0.0)]
        curves = {"ground": geo.addLine(c[0], c[1]), "outlet": geo.addLine(c[1], c[2]),
                  "top": geo.addLine(c[2], c[3]), "inlet": geo.addLine(c[3], c[0]),
                  "wing": (upper, lower)}
        outer = geo.addCurveLoop([curves["ground"], curves["outlet"], curves["top"], curves["inlet"]])
        hole = geo.addCurveLoop([upper, lower])
        surf = geo.addPlaneSurface([outer, hole])
        geo.synchronize()

        f = gmsh.model.mesh.field
        dist = f.add("Distance")
        f.setNumbers(dist, "CurvesList", [upper, lower])
        f.setNumber(dist, "Sampling", 400)
        near = f.add("Threshold")
        f.setNumber(near, "InField", dist)
        f.setNumber(near, "SizeMin", m["wing"])
        f.setNumber(near, "SizeMax", m["far"])
        f.setNumber(near, "DistMin", 0.06)
        f.setNumber(near, "DistMax", 3.0)
        box_gap = f.add("Box")
        for k, v in dict(VIn=gap, VOut=m["far"], XMin=-0.2, XMax=1.2, YMin=0.0,
                         YMax=float(y.max()), Thickness=0.4).items():
            f.setNumber(box_gap, k, v)
        box_wake = f.add("Box")
        for k, v in dict(VIn=m["wake"], VOut=m["far"], XMin=0.7, XMax=5.0, YMin=0.0,
                         YMax=float(y.max()) + 0.5, Thickness=1.0).items():
            f.setNumber(box_wake, k, v)
        # ponytail: no gmsh BoundaryLayer field; isotropic wall refinement + wall functions
        # (y+ ~ 30-100 at wing size 0.006) keeps the gap meshable down to h_c = 0.03.
        bg = f.add("Min")
        f.setNumbers(bg, "FieldsList", [near, box_gap, box_wake])
        f.setAsBackgroundMesh(bg)

        ext = geo.extrude([(2, surf)], 0.0, 0.0, 1.0, numElements=[1], recombine=True)
        geo.synchronize()
        top_surf, vol = ext[0][1], ext[1][1]
        side = {}
        for dim, tag in ext[2:]:
            if dim != 2:
                continue
            bnd = {abs(t) for _, t in gmsh.model.getBoundary([(2, tag)], oriented=False)}
            for name, ct in curves.items():
                if bnd & set(ct if isinstance(ct, tuple) else (ct,)):
                    side.setdefault(name, []).append(tag)
        for name, tags in side.items():
            gmsh.model.addPhysicalGroup(2, tags, name=name)
        gmsh.model.addPhysicalGroup(2, [surf, top_surf], name="frontAndBack")
        gmsh.model.addPhysicalGroup(3, [vol], name="internal")
        gmsh.model.mesh.generate(3)
        gmsh.write(str(msh_path))
    finally:
        gmsh.finalize()
    return hashlib.sha256(Path(msh_path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# OpenFOAM case
# --------------------------------------------------------------------------- #
def _foam(cls: str, obj: str, body: str) -> str:
    return f"FoamFile\n{{\n    version 2.0;\n    format ascii;\n    class {cls};\n    object {obj};\n}}\n{body}\n"


def _field(cls: str, name: str, dims: str, internal: str, bcs: dict[str, str]) -> str:
    bf = "".join(f"    {p} {{ {bc} }}\n" for p, bc in bcs.items())
    return _foam(cls, name, f"dimensions {dims};\ninternalField uniform {internal};\nboundaryField\n{{\n{bf}}}")


def write_case(case: Path, params: Params, options: RunOptions, settings: dict) -> None:
    U, c, Re = settings["U_inf"], settings["chord"], settings["Re"]
    nu = U * c / Re
    k = 1.5 * (0.01 * U) ** 2                      # Tu = 1 %
    omega = math.sqrt(k) / (0.09 ** 0.25 * 0.07 * c)  # L = 0.07 c
    s = settings["solver"]
    end = int(s["end_time"] * options.iter_mult)
    relax = s["relax"][options.relax_step]
    rc = s["residual_control"]
    for sub in ("0", "constant", "system"):
        (case / sub).mkdir(parents=True, exist_ok=True)

    empty, slip = "type empty;", "type slip;"
    (case / "0" / "U").write_text(_field("volVectorField", "U", "[0 1 -1 0 0 0 0]", f"({U} 0 0)", {
        "inlet": f"type fixedValue; value uniform ({U} 0 0);",
        "outlet": f"type inletOutlet; inletValue uniform (0 0 0); value uniform ({U} 0 0);",
        "top": slip, "ground": f"type movingWallVelocity; value uniform ({U} 0 0);",
        "wing": "type noSlip;", "frontAndBack": empty}))
    (case / "0" / "p").write_text(_field("volScalarField", "p", "[0 2 -2 0 0 0 0]", "0", {
        "inlet": "type zeroGradient;", "outlet": "type fixedValue; value uniform 0;",
        "top": slip, "ground": "type zeroGradient;", "wing": "type zeroGradient;", "frontAndBack": empty}))
    (case / "0" / "k").write_text(_field("volScalarField", "k", "[0 2 -2 0 0 0 0]", f"{k:.6g}", {
        "inlet": f"type fixedValue; value uniform {k:.6g};",
        "outlet": f"type inletOutlet; inletValue uniform {k:.6g}; value uniform {k:.6g};",
        "top": slip, "ground": f"type kqRWallFunction; value uniform {k:.6g};",
        "wing": f"type kqRWallFunction; value uniform {k:.6g};", "frontAndBack": empty}))
    (case / "0" / "omega").write_text(_field("volScalarField", "omega", "[0 0 -1 0 0 0 0]", f"{omega:.6g}", {
        "inlet": f"type fixedValue; value uniform {omega:.6g};",
        "outlet": f"type inletOutlet; inletValue uniform {omega:.6g}; value uniform {omega:.6g};",
        "top": slip, "ground": f"type omegaWallFunction; value uniform {omega:.6g};",
        "wing": f"type omegaWallFunction; value uniform {omega:.6g};", "frontAndBack": empty}))
    (case / "0" / "nut").write_text(_field("volScalarField", "nut", "[0 2 -1 0 0 0 0]", "0", {
        "inlet": "type calculated; value uniform 0;", "outlet": "type calculated; value uniform 0;",
        "top": "type calculated; value uniform 0;", "ground": "type nutkWallFunction; value uniform 0;",
        "wing": "type nutkWallFunction; value uniform 0;", "frontAndBack": empty}))

    (case / "constant" / "transportProperties").write_text(
        _foam("dictionary", "transportProperties", f"transportModel Newtonian;\nnu {nu!r};"))
    (case / "constant" / "turbulenceProperties").write_text(
        _foam("dictionary", "turbulenceProperties",
              "simulationType RAS;\nRAS\n{\n    RASModel kOmegaSST;\n    turbulence on;\n    printCoeffs off;\n}"))

    functions = f"""functions
{{
    forceCoeffs
    {{
        type forceCoeffs; libs (forces); writeControl timeStep; writeInterval 1; log no;
        patches (wing); rho rhoInf; rhoInf 1;
        liftDir (0 1 0); dragDir (1 0 0); pitchAxis (0 0 1); CofR (0.25 {params.h_c!r} 0);
        magUInf {U!r}; lRef {c!r}; Aref {c * 1.0!r};
    }}
    residuals
    {{
        type solverInfo; libs (utilityFunctionObjects); fields (U p k omega); writeResidualFields no;
    }}
    yPlus
    {{
        type yPlus; libs (fieldFunctionObjects); executeControl writeTime; writeControl writeTime;
    }}
    wallShearStress
    {{
        type wallShearStress; libs (fieldFunctionObjects); patches (wing);
        executeControl writeTime; writeControl writeTime;
    }}
    surfaces
    {{
        type surfaces; libs (sampling); writeControl writeTime;
        surfaceFormat raw; interpolationScheme cell; fields (p wallShearStress);
        surfaces {{ wing {{ type patch; patches (wing); interpolate false; }} }}
    }}
}}"""
    (case / "system" / "controlDict").write_text(_foam("dictionary", "controlDict", f"""application simpleFoam;
startFrom startTime; startTime 0; stopAt endTime; endTime {end}; deltaT 1;
writeControl timeStep; writeInterval {end}; purgeWrite 0;
writeFormat ascii; writePrecision 8; writeCompression off; timeFormat general; timePrecision 6;
runTimeModifiable false;
{functions}"""))
    (case / "system" / "fvSchemes").write_text(_foam("dictionary", "fvSchemes", """ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes
{
    default none;
    div(phi,U) bounded Gauss linearUpwind grad(U);
    div(phi,k) bounded Gauss upwind;
    div(phi,omega) bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
wallDist { method meshWave; }"""))
    (case / "system" / "fvSolution").write_text(_foam("dictionary", "fvSolution", f"""solvers
{{
    p {{ solver GAMG; smoother GaussSeidel; tolerance 1e-7; relTol 0.05; }}
    "(U|k|omega)" {{ solver smoothSolver; smoother symGaussSeidel; nSweeps 2; tolerance 1e-9; relTol 0.1; }}
}}
SIMPLE
{{
    nNonOrthogonalCorrectors 0;
    residualControl {{ p {rc['p']!r}; U {rc['U']!r}; k {rc['k']!r}; omega {rc['omega']!r}; }}
}}
relaxationFactors
{{
    fields {{ p {relax['p']!r}; }}
    equations {{ U {relax['U']!r}; "(k|omega)" {relax['turb']!r}; }}
}}"""))
    (case / "system" / "mapFieldsDict").write_text(
        _foam("dictionary", "mapFieldsDict", "patchMap ( );\ncuttingPatches ( );"))


# --------------------------------------------------------------------------- #
# run OpenFOAM, one subprocess per command
# --------------------------------------------------------------------------- #
def _run(case: Path, app: str, cmd: str, deadline: float) -> dict:
    """Run one command through the openfoam2512 launcher; the log is case/log.<app>."""
    shell = f"cd '{case}' && {cmd} >> log.{app} 2>&1"
    timed_out, code = False, 127
    try:
        proc = subprocess.Popen([OPENFOAM, "-c", shell], start_new_session=True,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            code = proc.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    except OSError as e:
        (case / f"log.{app}").write_text(f"{app}: launcher failed: {e}\n")
    log_file = case / f"log.{app}"
    log = log_file.read_text(errors="replace") if log_file.is_file() else ""
    return {"app": app, "exit": code, "log": log, "timed_out": timed_out}


def run_foam(case: Path, msh: Path, warm_src: Path | None, deadline: float) -> list[dict]:
    """gmshToFoam -> patch types -> checkMesh -> [mapFields] -> simpleFoam. Stops at the
    first non-zero exit or timeout. Each entry: stage, app, exit, log, timed_out."""
    fix = "foamDictionary constant/polyMesh/boundary -entry entry0/{p}/type -set {t}"
    steps = [("mesh", "gmshToFoam", f"gmshToFoam '{msh}'")]
    steps += [("mesh", "foamDictionary", fix.format(p=p, t=t))
              for p, t in (("frontAndBack", "empty"), ("ground", "wall"), ("wing", "wall"))]
    steps.append(("mesh", "checkMesh", "checkMesh"))
    if warm_src is not None:
        steps.append(("solve", "mapFields", f"mapFields -sourceTime latestTime '{warm_src}'"))
    steps.append(("solve", "simpleFoam", "simpleFoam"))
    out = []
    for stage, app, cmd in steps:
        r = _run(case, app, cmd, deadline)
        r["stage"] = stage
        out.append(r)
        if r["exit"] != 0 or r["timed_out"]:
            break
    return out


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
_TIME_RE = re.compile(r"^Time = (\d+)", re.M)
_CONV_RE = re.compile(r"SIMPLE solution converged in (\d+) iterations")
_RES_RE = re.compile(r"Solving for (\w+), Initial residual = ([0-9.eE+-]+)")


def parse_iterations(log: str) -> int:
    m = _CONV_RE.search(log)
    if m:
        return int(m.group(1))
    times = _TIME_RE.findall(log)
    return int(times[-1]) if times else 0


def _last_initial_residuals(log: str) -> dict[str, float]:
    blocks = _TIME_RE.split(log)
    last = blocks[-1] if len(blocks) > 1 else ""
    res: dict[str, float] = {}
    for name, val in _RES_RE.findall(last):
        res.setdefault(name, float(val))  # first solve of each field in the final iteration
    return res


def _num(pattern: str, text: str, default: float) -> float:
    m = re.search(pattern, text)
    return float(m.group(1)) if m else default


def _dat_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    header, rows = [], []
    for ln in path.read_text().splitlines():
        if ln.startswith("#"):
            header = ln.lstrip("# ").split()
        elif ln.strip():
            rows.append(ln.split())
    return header, rows


def _latest_time_dir(root: Path) -> Path:
    dirs = [p for p in root.iterdir() if p.is_dir() and re.fullmatch(r"[0-9.e+-]+", p.name)]
    return max(dirs, key=lambda p: float(p.name))


def _read_raw(path: Path) -> np.ndarray:
    return np.loadtxt(path, comments="#", ndmin=2)


def force_tail(case: Path, settings: dict) -> tuple[float, float, str] | None:
    """(mean Cl, mean Cd over the last force_settle_window steps, kind) from coefficient.dat,
    kind = "force_oscillating" if that window still moves by more than force_settle_tol,
    else "residual_stall". None when the file is missing or unparsable (never raises)."""
    try:
        t = settings["trust"]
        hdr, rows = _dat_rows(case / "postProcessing" / "forceCoeffs" / "0" / "coefficient.dat")
        icd, icl = hdr.index("Cd"), hdr.index("Cl")
        cl = np.array([float(r[icl]) for r in rows])
        cd = np.array([float(r[icd]) for r in rows])
        w = int(t["force_settle_window"])
        if len(cl) == 0:
            return None
        moving = (len(cl) >= 2 * w
                  and abs(cl[-w:].mean() - cl[-2 * w:-w].mean()) > t["force_settle_tol"] * max(abs(cl[-1]), 0.1))
        return float(cl[-w:].mean()), float(cd[-w:].mean()), "force_oscillating" if moving else "residual_stall"
    except Exception:
        return None


def parse_case(case: Path, params: Params, settings: dict) -> tuple[dict, dict, dict, int]:
    """Read logs and postProcessing of a finished simpleFoam run.
    Returns (forces, trust, field_summary, iterations); raises on missing/unparsable data."""
    t = settings["trust"]
    log = (case / "log.simpleFoam").read_text(errors="replace")
    iterations = parse_iterations(log)

    # forces + settling
    hdr, rows = _dat_rows(case / "postProcessing" / "forceCoeffs" / "0" / "coefficient.dat")
    icd, icl = hdr.index("Cd"), hdr.index("Cl")
    cl = np.array([float(r[icl]) for r in rows])
    forces = {"Cl": float(cl[-1]), "Cd": float(rows[-1][icd])}
    w = int(t["force_settle_window"])
    settled = False
    if len(cl) >= 2 * w:
        settled = abs(cl[-w:].mean() - cl[-2 * w:-w].mean()) <= t["force_settle_tol"] * max(abs(cl[-1]), 0.1)

    # residuals of the final iteration vs 10x the control targets
    res = _last_initial_residuals(log)
    rc = settings["solver"]["residual_control"]
    want = {"p": rc["p"], "Ux": rc["U"], "Uy": rc["U"], "k": rc["k"], "omega": rc["omega"]}
    residual_ok = all(f in res and res[f] <= 10.0 * thr for f, thr in want.items())

    # y+ on the wing
    _, yrows = _dat_rows(case / "postProcessing" / "yPlus" / "0" / "yPlus.dat")
    ywing = [r for r in yrows if r[1] == "wing"]
    yplus = float(ywing[-1][4]) if ywing else float("nan")
    yplus_ok = bool(t["yplus_min"] <= yplus <= t["yplus_max"])

    # mesh quality from checkMesh
    cm = (case / "log.checkMesh").read_text(errors="replace")
    mesh_ok = ("Failed" not in cm
               and _num(r"non-orthogonality (?:Max:|=) ([0-9.eE+-]+)", cm, 1e9) <= t["max_nonortho"]
               and _num(r"Max skewness = ([0-9.eE+-]+)", cm, 1e9) <= t["max_skew"])

    trust = {"converged": bool(_CONV_RE.search(log)), "residual_ok": bool(residual_ok),
             "force_settled": bool(settled), "yplus_ok": yplus_ok, "mesh_ok": bool(mesh_ok)}

    # suction-surface Cp / tau_x from the raw patch samples of the last written time
    tdir = _latest_time_dir(case / "postProcessing" / "surfaces")
    praw = _read_raw(next(tdir.rglob("p*.raw")))
    traw = _read_raw(next(tdir.rglob("wallShearStress*.raw")))
    field_summary = suction_features(praw, traw, params, settings)
    return forces, trust, field_summary, iterations


def suction_features(praw: np.ndarray, traw: np.ndarray, params: Params, settings: dict) -> dict:
    """Map wing-patch face samples (x y z value...) to the airfoil chord coordinate via
    the nearest contour vertex and evaluate cp_features on the lower (suction) surface."""
    n = settings["n_surface"]
    xl, _ = naca4(float(params.camber), settings["naca_p"], settings["naca_t"], n)
    xw, yw = wing_coords(params, n)
    verts = np.column_stack([xw[:-1], yw[:-1]])
    if praw.shape[0] != traw.shape[0] or not np.allclose(praw[:, :3], traw[:, :3]):
        raise ValueError("p and wallShearStress samples do not share the same faces")
    idx = np.argmin(((praw[:, None, :2] - verts[None, :, :]) ** 2).sum(-1), axis=1)
    # naca4 orders TE -> "upper" -> LE (indices 0..n-1); after inversion that arc is the
    # lower = suction surface. ponytail: chord position = airfoil-x of the nearest vertex (~1/n).
    suction = idx <= n - 1
    xc = xl[idx[suction]]
    cp = praw[suction, 3] / (0.5 * settings["U_inf"] ** 2)
    tau_x = -traw[suction, 3]  # OpenFOAM wallShearStress points against the flow; flip so +x = attached
    o = np.argsort(xc, kind="stable")
    return cp_features(xc[o], cp[o], tau_x[o])
