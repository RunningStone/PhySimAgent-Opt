"""walk-stage orchestrator (POC PLAN §4.3, no LLM in the loop): h/c x alpha sweep,
report + cliff indicators, deterministic spot checks, failure-type reproduction,
SENSITIVITY.md (via pipeline.agentloop when available) and the experiment tree.

    uv --project ENV run python -m pipeline.exp_layer.aero2d.evaluation.walk CONFIGs/batch1/aero2d/matrix/walk.yaml [--dry-run]

Everything lands under OUTPUTs/<exp_name>/ (track two, idempotent: a re-run of the same
yaml only hits the simblock cache). Never raises on a per-case basis; exit code 0 unless
the config itself is unusable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import yaml

from pipeline.exp_layer.aero2d import FAILURE_TYPES, CaseResult, Params, RunOptions, run_case, solver_build, verify
from pipeline.exp_layer.aero2d.evaluation.sweep_report import write_report

REPO = Path(__file__).resolve().parents[5]
TREE_FILES = ("sweep.jsonl", "report.json", "sweep.png", "SENSITIVITY.md", "verify.json")


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path) -> dict:
    """YAML with a ``defaults:`` list merged in order (paths relative to the file), then the file itself."""
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text()) or {}
    merged: dict = {}
    for d in raw.pop("defaults", None) or []:
        merged = deep_merge(merged, load_config(path.parent / d))
    return deep_merge(merged, raw)


def _repo_path(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO / p


def _git_head() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _say(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# sweep
# --------------------------------------------------------------------------- #
def _fmt(v, spec: str = ".4f") -> str:
    return "-" if v is None else format(v, spec)


def _progress(i: int, n: int, p: Params, r: CaseResult, dt: float) -> str:
    ft = "" if r.failure is None else f"/{r.failure['type']}"
    cl = None if r.forces is None else r.forces["Cl"]
    cd = None if r.forces is None else r.forces["Cd"]
    return (f"[{i:>3}/{n}] h_c={p.h_c:<6g} alpha={p.alpha_deg:<4g} {r.status + ft:<20} "
            f"Cl={_fmt(cl):>8} Cd={_fmt(cd):>7} trusted={str(r.trusted):<5} it={r.iterations:<5} {dt:6.1f}s")


def sweep(cfg: dict, exp_dir: Path, store: Path) -> tuple[list[CaseResult], list[Params]]:
    combos = [Params(float(h), float(a), float(cfg["camber"])) for h in cfg["h_c"] for a in cfg["alpha_deg"]]
    results: list[CaseResult] = []
    with (exp_dir / "sweep.jsonl").open("w") as fh:
        for i, p in enumerate(combos, 1):
            t0 = time.perf_counter()
            r = run_case(p, store=store, timeout_s=float(cfg.get("timeout_s", 900)))
            results.append(r)
            fh.write(json.dumps({"params": asdict(p), "result": json.loads(r.to_json())}) + "\n")
            fh.flush()
            _say(_progress(i, len(combos), p, r, time.perf_counter() - t0))
    return results, combos


# --------------------------------------------------------------------------- #
# report + cliff indicators
# --------------------------------------------------------------------------- #
def cliff_indicators(results: list[CaseResult], params: list[Params]) -> tuple[dict, dict]:
    """critical_h_c[alpha] = h_c of max downforce (-Cl) among trusted cases;
    cliff_indicator[alpha] = first h_c, scanning from far field toward the ground (descending h_c),
    whose case is not_settled or ok-but-untrusted (None if the whole line is trusted)."""
    critical: dict[str, float | None] = {}
    cliff: dict[str, float | None] = {}
    for a in sorted({p.alpha_deg for p in params}):
        line = sorted(((p, r) for p, r in zip(params, results) if p.alpha_deg == a), key=lambda t: -t[0].h_c)
        trusted = [(p, r) for p, r in line if r.trusted]
        critical[str(a)] = max(trusted, key=lambda t: -t[1].forces["Cl"])[0].h_c if trusted else None
        hit = next((p.h_c for p, r in line
                    if (r.status == "ok" and not r.trusted)
                    or (r.failure is not None and r.failure["type"] == "not_settled")), None)
        cliff[str(a)] = hit
    return critical, cliff


def report(results, params, exp_dir: Path, literature) -> dict:
    rep = write_report(results, params, exp_dir, literature_json=literature)
    rep["critical_h_c"], rep["cliff_indicator"] = cliff_indicators(results, params)
    rep["cliff_indicator_rule"] = ("first h_c scanning from large to small h_c whose case is not_settled "
                                   "or ok-but-untrusted; null = none")
    (exp_dir / "report.json").write_text(json.dumps(rep, indent=2))
    return rep


# --------------------------------------------------------------------------- #
# deterministic spot checks
# --------------------------------------------------------------------------- #
def spot_verify(results, params, n: int, store: Path, exp_dir: Path) -> dict:
    ok = [(p, r) for p, r in zip(params, results) if r.status == "ok"]
    picks = random.Random(0).sample(ok, min(int(n), len(ok))) if n and ok else []
    out_file = exp_dir / "verify.json"
    want = sorted(r.case_id for _, r in picks)
    if out_file.is_file():
        try:
            prev = json.loads(out_file.read_text())
            if sorted(prev) == want:  # same picks already verified in this exp_dir -> idempotent re-run
                _say(f"verify: reusing {out_file.name} ({len(prev)} cases)")
                return prev
        except (OSError, ValueError):
            pass
    out: dict[str, dict] = {}
    for p, r in picks:
        t0 = time.perf_counter()
        same, diff = verify(p, store=store)
        out[r.case_id] = {"identical": bool(same), "diff_chars": len(diff)}
        _say(f"verify {r.case_id} h_c={p.h_c:g} alpha={p.alpha_deg:g}: identical={same} "
             f"diff_chars={len(diff)} {time.perf_counter() - t0:.1f}s")
    out_file.write_text(json.dumps(out, indent=2))
    return out


# --------------------------------------------------------------------------- #
# failure reproduction (one case per failure type, best effort)
# --------------------------------------------------------------------------- #
def _attempt(kind: str, label: str, fn, out: Path, notes: list[str]) -> bool:
    t0 = time.perf_counter()
    p, o, r = fn()
    got = "ok" if r.failure is None else r.failure["type"]
    _say(f"failure_repro {kind}: {label} -> {got} ({time.perf_counter() - t0:.1f}s)")
    if got == kind:
        (out / f"{kind}.json").write_text(json.dumps(
            {"params": asdict(p), "options": asdict(o), "attempt": label,
             "result": json.loads(r.to_json(deterministic=True))}, indent=2))  # no runtime_s -> stable tree
        return True
    notes.append(f"{label} -> {got}" + (f" ({r.failure['evidence']})" if r.failure else ""))
    return False


def failure_repro(cfg: dict, exp_dir: Path, store: Path, results, params) -> dict:
    out = exp_dir / "cases" / "failures"
    out.mkdir(parents=True, exist_ok=True)
    timeout_s = float(cfg.get("timeout_s", 900))
    status: dict[str, str] = {}
    notes: dict[str, list[str]] = {k: [] for k in FAILURE_TYPES}

    def run(p, o=RunOptions(), **kw):
        return p, o, run_case(p, o, store=store, timeout_s=timeout_s, **kw)

    # geometry_invalid: out-of-range parameter, never meshed, never stored
    done = _attempt("geometry_invalid", "Params(0.001, 0, 0.04)",
                    lambda: run(Params(0.001, 0.0, 0.04)), out, notes["geometry_invalid"])
    status["geometry_invalid"] = "reproduced" if done else "not reproduced"

    # timeout: 3 s deadline, no cache, private store so the main cache is not polluted
    if (out / "timeout.json").is_file():
        _say("failure_repro timeout: reusing timeout.json")
        status["timeout"] = "reproduced (cached file)"
    else:
        tmp = out / "store_tmp"
        p = Params(0.3, 2.0, 0.04)
        done = _attempt("timeout", "Params(0.3, 2, 0.04) timeout_s=3 cache=False store=store_tmp",
                        lambda: (p, RunOptions(), run_case(p, store=tmp, cache=False, timeout_s=3)),
                        out, notes["timeout"])
        shutil.rmtree(tmp, ignore_errors=True)
        status["timeout"] = "reproduced" if done else "not reproduced"

    # not_settled: prefer a case from the sweep, else the extreme corner of the domain
    ns = [(p, r) for p, r in zip(params, results) if r.failure and r.failure["type"] == "not_settled"]
    if ns:
        p, r = ns[0]
        done = _attempt("not_settled", f"sweep case {r.case_id} {asdict(p)}", lambda: (p, RunOptions(), r),
                        out, notes["not_settled"])
    else:
        done = _attempt("not_settled", "Params(0.03, 10, 0.09)", lambda: run(Params(0.03, 10.0, 0.09)),
                        out, notes["not_settled"])
    status["not_settled"] = "reproduced" if done else "not reproduced"
    ns_id = None
    if done:
        ns_id = json.loads((out / "not_settled.json").read_text())["result"]["case_id"]

    # mesh_invalid: thinnest gap, steepest wing, maximum gap refinement (not guaranteed)
    done = _attempt("mesh_invalid", "Params(0.03, 12, 0.09) gap_refine=3",
                    lambda: run(Params(0.03, 12.0, 0.09), RunOptions(gap_refine=3)), out, notes["mesh_invalid"])
    status["mesh_invalid"] = "reproduced" if done else "not reproduced"

    # diverged: warm start from an unconverged field, then the extreme corner with default options
    done = False
    if ns_id is not None:
        p_ns = Params(**json.loads((out / "not_settled.json").read_text())["params"])
        done = _attempt("diverged", f"{asdict(p_ns)} warm_start_from={ns_id}",
                        lambda: run(p_ns, RunOptions(warm_start_from=ns_id)), out, notes["diverged"])
    if not done:
        done = _attempt("diverged", "Params(0.03, 12, 0.09) relax_step=0 iter_mult=1",
                        lambda: run(Params(0.03, 12.0, 0.09)), out, notes["diverged"])
    status["diverged"] = "reproduced" if done else "not reproduced"

    lines = ["# cases/failures — one reproduction case per failure type (walk stage)", ""]
    for k in FAILURE_TYPES:
        lines.append(f"- `{k}`: {status[k]}" + (f" → `{k}.json`" if status[k].startswith("reproduced") else ""))
        if not status[k].startswith("reproduced"):
            lines.append(f"  - 未复现:" + "; ".join(notes[k]))
    (out / "README.md").write_text("\n".join(lines) + "\n")
    return status


# --------------------------------------------------------------------------- #
# SENSITIVITY.md
# --------------------------------------------------------------------------- #
def sensitivity_rows(results, params) -> list[dict]:
    rows = []
    for p, r in zip(params, results):
        fs = r.field_summary or {}
        rows.append({"h_c": p.h_c, "alpha_deg": p.alpha_deg, "camber": p.camber,
                     "Cl": None if r.forces is None else r.forces["Cl"],
                     "Cd": None if r.forces is None else r.forces["Cd"],
                     "trusted": r.trusted, "failure_type": None if r.failure is None else r.failure["type"],
                     "cp_peak_val": fs.get("cp_peak_val"), "sep_x": fs.get("sep_x")})
    return rows


def write_sensitivity(rows: list[dict], model: str, exp_dir: Path) -> bool:
    """LLM summary of the sweep rows. Idempotent: an existing summary for the same rows + model
    (stamped in .sensitivity_input.sha256) is kept instead of calling the LLM again."""
    out, stamp = exp_dir / "SENSITIVITY.md", exp_dir / ".sensitivity_input.sha256"
    digest = hashlib.sha256(json.dumps({"rows": rows, "model": model}, sort_keys=True).encode()).hexdigest()
    if out.is_file() and stamp.is_file() and stamp.read_text().strip() == digest \
            and not out.read_text().startswith("(未生成"):
        _say("SENSITIVITY.md: reusing (rows and model unchanged)")
        return True
    try:
        from pipeline.agentloop import summarize_sensitivity  # implemented by the agentloop sub-agent
        text = summarize_sensitivity(rows, model)
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("summarize_sensitivity returned empty text")
        out.write_text(text)
        stamp.write_text(digest + "\n")
        _say(f"SENSITIVITY.md written ({len(text)} chars, model={model})")
        return True
    except (ImportError, RuntimeError) as e:
        reason = f"{type(e).__name__}: {e}"
    except Exception as e:  # keep the walk stage alive whatever the LLM path does
        reason = f"{type(e).__name__}: {e}"
    out.write_text(f"(未生成:{reason})\n")
    stamp.unlink(missing_ok=True)
    _say(f"SENSITIVITY.md not generated: {reason}")
    return False


# --------------------------------------------------------------------------- #
# experiment tree + manifest
# --------------------------------------------------------------------------- #
def commit_tree(exp_dir: Path) -> str:
    tree = exp_dir / "tree"
    git = ["git", "-C", str(tree)]
    if not (tree / ".git").is_dir():
        tree.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", "-q", "-b", "main", str(tree)], check=True)
        subprocess.run(git + ["config", "user.name", "poc"], check=True)
        subprocess.run(git + ["config", "user.email", "poc@local"], check=True)
    for name in TREE_FILES:
        if (exp_dir / name).is_file():
            shutil.copy2(exp_dir / name, tree / name)
    fdir = exp_dir / "cases" / "failures"
    if fdir.is_dir():
        (tree / "cases" / "failures").mkdir(parents=True, exist_ok=True)
        for f in list(fdir.glob("*.json")) + [fdir / "README.md"]:
            if f.is_file():
                shutil.copy2(f, tree / "cases" / "failures" / f.name)
    subprocess.run(git + ["add", "-A"], check=True)
    r = subprocess.run(git + ["commit", "-q", "-m", "walk: sweep + sensitivity"], capture_output=True, text=True)
    head = subprocess.run(git + ["rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    _say(f"tree: {'committed' if r.returncode == 0 else 'no change'} @ {head or '?'}")
    return head


def write_manifest(exp_dir: Path, cfg: dict, n_cases: int, elapsed_s: float, extra: dict) -> None:
    manifest = {"kind": "produce", "name": cfg["exp_name"],
                "date": datetime.now().astimezone().isoformat(timespec="seconds"),
                "git_head": _git_head(), "config": cfg, "solver_build": solver_build(),
                "uname": platform.uname()._asdict(), "python": sys.version, "repo_root": str(REPO),
                "openfoam2512": shutil.which("openfoam2512"), "n_cases": n_cases,
                "elapsed_s": round(elapsed_s, 1), **extra}
    (exp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="simblock walk stage: sweep + report + verify + failure repro + SENSITIVITY")
    ap.add_argument("yaml", help="matrix yaml (with optional defaults: list)")
    ap.add_argument("--dry-run", action="store_true", help="print the number of combinations and exp_dir only")
    args = ap.parse_args(argv)

    cfg = load_config(args.yaml)
    for k in ("exp_name", "h_c", "alpha_deg", "camber"):
        if k not in cfg:
            ap.error(f"config missing key {k!r}")
    exp_dir = _repo_path(cfg["output_dir"]) if cfg.get("output_dir") else REPO / cfg.get("output_root", "OUTPUTs") / str(cfg["exp_name"])
    store = _repo_path(cfg.get("store", "OUTPUTs/batch1/aero2d/store"))
    n = len(cfg["h_c"]) * len(cfg["alpha_deg"])
    _say(f"walk: {len(cfg['h_c'])} h_c x {len(cfg['alpha_deg'])} alpha = {n} cases, camber={cfg['camber']}, "
         f"exp_dir={exp_dir}, store={store}")
    if args.dry_run:
        return 0

    t_start = time.perf_counter()
    exp_dir.mkdir(parents=True, exist_ok=True)
    store.mkdir(parents=True, exist_ok=True)

    results, params = sweep(cfg, exp_dir, store)
    literature = _repo_path(cfg["literature"]) if cfg.get("literature") else None
    rep = report(results, params, exp_dir, literature)
    _say(f"report: n_ok={rep['n_ok']} n_trusted={rep['n_trusted']} failures={rep['failures']} "
         f"critical_h_c={rep['critical_h_c']} cliff_indicator={rep['cliff_indicator']}")

    extra: dict = {}
    extra["verify"] = spot_verify(results, params, int(cfg.get("verify_cases", 0)), store, exp_dir)
    if cfg.get("failure_repro"):
        extra["failure_repro"] = failure_repro(cfg, exp_dir, store, results, params)
    if cfg.get("write_sensitivity"):
        extra["sensitivity_generated"] = write_sensitivity(sensitivity_rows(results, params),
                                                           str(cfg.get("llm_model", "")), exp_dir)
    extra["tree_head"] = commit_tree(exp_dir)
    write_manifest(exp_dir, cfg, n, time.perf_counter() - t_start, extra)
    _say(f"walk done in {time.perf_counter() - t_start:.1f}s -> {exp_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
