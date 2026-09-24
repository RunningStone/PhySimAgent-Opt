"""pipeline.exp_layer.aero2d — deterministic, content-hash-cached, never-raising 2-D RANS
block for an inverted NACA-4 wing in ground effect (REQUIREMENTS §2).

Public API: Params, RunOptions, CaseResult, input_hash, wing_coords,
classify_failure, cp_features, run_case, verify, solver_build, SETTINGS, FAILURE_TYPES.
"""

from __future__ import annotations

import difflib
import functools
import hashlib
import json
import plistlib
import re
import shutil
import subprocess
import time
from dataclasses import replace
from pathlib import Path

from .contract import (
    FAILURE_TYPES,
    SETTINGS,
    CaseResult,
    Params,
    RunOptions,
    classify_failure,
    cp_features,
    failed_result,
    find_case_dir,
    input_hash,
    not_settled_evidence,
    validate,
    wing_coords,
)
from batch1.aero2d.code.solve import OPENFOAM, build_mesh, force_tail, parse_case, parse_iterations, run_foam, write_case

__all__ = [
    "Params", "RunOptions", "CaseResult", "input_hash", "wing_coords", "classify_failure",
    "cp_features", "run_case", "verify", "solver_build", "SETTINGS", "FAILURE_TYPES",
]

DEFAULT_STORE = Path(__file__).resolve().parents[4] / "OUTPUTs" / "batch1" / "aero2d" / "store"
_APP_PLIST = Path("/Applications/OpenFOAM-v2512.app/Contents/Info.plist")
_CASK = Path("/opt/homebrew/Library/Taps/gerlero/homebrew-openfoam/Casks/openfoam@2512.rb")


@functools.lru_cache(maxsize=1)
def solver_build() -> dict:
    """{app_version, app_zip_sha256, api_info_hash} of the native OpenFOAM build; "unknown" per missing source."""
    out = {"app_version": "unknown", "app_zip_sha256": "unknown", "api_info_hash": "unknown"}
    try:
        with _APP_PLIST.open("rb") as fh:
            out["app_version"] = str(plistlib.load(fh)["CFBundleShortVersionString"])
    except Exception:
        pass
    try:
        cask = _CASK
        if not cask.is_file():
            repo = subprocess.run(["brew", "--repository", "gerlero/openfoam"], capture_output=True, text=True, timeout=30)
            cask = Path(repo.stdout.strip()) / "Casks" / "openfoam@2512.rb"
        m = re.search(r'sha256\s+"([0-9a-f]{64})"', cask.read_text())
        if m:
            out["app_zip_sha256"] = m.group(1)
    except Exception:
        pass
    try:
        r = subprocess.run([OPENFOAM, "-c", 'cat "$WM_PROJECT_DIR/META-INFO/api-info"'],
                           capture_output=True, timeout=60)
        if r.returncode == 0 and r.stdout:
            out["api_info_hash"] = hashlib.sha256(r.stdout).hexdigest()
    except Exception:
        pass
    return out


def run_case(params: Params, options: RunOptions = RunOptions(), *, scalar_only: bool = False,
             cache: bool = True, store=None, timeout_s: float = 900) -> CaseResult:
    """Validate -> hash -> cache -> mesh -> write case -> OpenFOAM -> parse -> store/<hash>/result.json."""
    t0 = time.perf_counter()
    store = (Path(store) if store is not None else DEFAULT_STORE).resolve()
    h = input_hash(params, options)
    build = solver_build()
    evidence = validate(params, options, store)
    if evidence is not None:
        return failed_result(h, classify_failure("geometry", evidence, 1, False), None, build,
                             runtime_s=time.perf_counter() - t0)
    d = store / h
    if cache and (d / "result.json").is_file():
        r = CaseResult.from_json((d / "result.json").read_text())
    else:
        r = _compute(params, options, h, d, store, build, time.monotonic() + timeout_s)
        r = replace(r, runtime_s=time.perf_counter() - t0)
        d.mkdir(parents=True, exist_ok=True)
        (d / "result.json").write_text(r.to_json())
    return replace(r, field_summary=None) if scalar_only else r


def _compute(params: Params, options: RunOptions, h: str, d: Path, store: Path, build: dict,
             deadline: float) -> CaseResult:
    stage, mesh_hash, iterations = "mesh", None, 0
    case = d / "case"
    try:
        shutil.rmtree(d, ignore_errors=True)
        case.mkdir(parents=True)
        mesh_hash = build_mesh(params, options, SETTINGS, d / "mesh.msh")
        write_case(case, params, options, SETTINGS)
        src = find_case_dir(store, options.warm_start_from) if options.warm_start_from else None
        for step in run_foam(case, d / "mesh.msh", src / "case" if src else None, deadline):
            stage = step["stage"]
            if step["app"] == "simpleFoam":
                iterations = parse_iterations(step["log"])
            # helper commands are judged by exit/timeout only; checkMesh and simpleFoam by their log
            if step["exit"] != 0 or step["timed_out"] or step["app"] in ("checkMesh", "simpleFoam"):
                failure = classify_failure(stage, step["log"], step["exit"], step["timed_out"])
                if failure is not None:
                    if failure["type"] == "not_settled":  # cliff indicator: last-100 Cl/Cd + residuals
                        tail = force_tail(case, SETTINGS)
                        if tail is not None:
                            failure = {**failure, "evidence": not_settled_evidence(step["log"], *tail)}
                    return failed_result(h, failure, mesh_hash, build, iterations)
        stage = "post"
        forces, trust, field_summary, iterations = parse_case(case, params, SETTINGS)
    except Exception as e:  # never raise: any exception becomes a typed failure of the current stage
        failure = classify_failure(stage, f"{type(e).__name__}: {e}", 1, False)
        return failed_result(h, failure, mesh_hash, build, iterations)
    return CaseResult(case_id=h[:12], input_hash=h, status="ok", forces=forces, trust=trust,
                      field_summary=field_summary, failure=None,
                      provenance={"mesh_hash": mesh_hash, "solver_build": build},
                      iterations=iterations, runtime_s=0.0)


def verify(params: Params, options: RunOptions = RunOptions(), store=None) -> tuple[bool, str]:
    """Re-run without cache into store/_verify_tmp and diff the deterministic JSON forms."""
    store = Path(store) if store is not None else DEFAULT_STORE
    a = run_case(params, options, store=store, cache=True)
    # ponytail: the temp store does not see warm-start sources of the main store (F1 has none)
    b = run_case(params, options, store=store / "_verify_tmp", cache=False)
    ja, jb = (json.dumps(json.loads(r.to_json(deterministic=True)), indent=1, sort_keys=True).splitlines()
              for r in (a, b))
    diff = "\n".join(difflib.unified_diff(ja, jb, "store", "rerun", lineterm=""))
    return (diff == "", diff)
