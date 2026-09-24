"""finding_report — four artefacts from the run_line outputs of an experiment dir
(CPLAN §2; one per POC CHECKER Phase-2 evidence item; v2 adds the multi-seed aggregate):

  finding.json               per run (lines/<line>[-<run_tag>]): summary keys + seed / memory_used
                             + cliff distance (best h/c - critical h/c); ``aggregate`` (per line x memory,
                             ``aggregate_lines``); ``walk_best`` (best trusted objective in the walk sweep)
  prediction_vs_outcome.csv  pre-registered predictions vs outcomes, all runs
  tree_graph.txt             git log --graph of the experiment tree
  finding.md                 the per-run table, the aggregate table, the walk_best line and three fill-in
                             lines (judgements go to PROGRESS, not here)

Usage: python -m pipeline.agentloop.evaluation.finding_report <exp_dir> [--walk <walk_dir>] [--out <dir>]
``<walk_dir>/report.json`` supplies ``critical_h_c`` (number, or {alpha_deg: h_c} dict as the walk writes it)
and ``<walk_dir>/sweep.jsonl`` the walk results for ``walk_best`` (default walk dir = exp_dir; absent -> null).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ..driver import Tree
from ..harness import aggregate_lines, objective
from ..scenario import WING_SCENARIO

_KEYS = ("line", "seed", "run_tag", "memory_used", "problem_hash", "workflow_mode",
         "environment_name", "n_cases", "n_unique_hash", "best_metric", "best_params",
         "best_raw_metrics", "stop_reason")
_AGG_COLS = ("group", "n_runs", "best_metric_mean", "best_metric_std", "best_metric_min", "best_metric_max",
             "holdout_robust_frac", "n_failed_mean", "n_cases_mean", "stop_reasons")


def _cliff_distance(best_params: dict, critical_h_c) -> float | None:
    """best h/c minus the walk's critical h/c; the walk reports it per alpha_deg ({"1.0": 0.5, ...}),
    so pick the alpha closest to the best design's."""
    if critical_h_c is None or "h_c" not in best_params:
        return None
    if isinstance(critical_h_c, dict):
        if not critical_h_c:
            return None
        a = float(best_params.get("alpha_deg", 0.0))
        critical_h_c = critical_h_c[min(critical_h_c, key=lambda k: abs(float(k) - a))]
    return None if critical_h_c is None else float(best_params["h_c"]) - float(critical_h_c)


def _scenario_named(name: str | None):
    return WING_SCENARIO


def walk_best(walk_dir: Path | str | None, scenario) -> dict | None:
    """Best trusted (status ok, all trust flags true) row of ``<walk_dir>/sweep.jsonl`` under the
    scenario's objective: {metric, params, case_id}; None when there is no sweep or no trusted row."""
    if walk_dir is None:
        return None
    sweep = Path(walk_dir) / "sweep.jsonl"
    if not sweep.is_file():
        return None
    best = None
    for raw in sweep.read_text().splitlines():
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        result = row.get("result") if isinstance(row, dict) else None
        if not isinstance(result, dict):
            continue
        try:
            m = objective(result, scenario)
        except (KeyError, TypeError, ValueError):
            m = None
        if m is not None and (best is None or m > best["metric"]):
            best = {"metric": m, "params": row.get("params"), "case_id": result.get("case_id")}
    return best


def _grouped_aggregate(summaries: list[dict]) -> dict:
    """Keep fixed and agent results separate for the same physical problem."""
    if not any(s.get("problem_hash") or s.get("workflow_mode") for s in summaries):
        return aggregate_lines(summaries)
    groups: dict[tuple[str, str], list[dict]] = {}
    for summary in summaries:
        key = (summary.get("problem_hash") or "legacy", summary.get("workflow_mode") or "legacy")
        groups.setdefault(key, []).append(summary)
    result = {}
    for (problem_hash, mode), members in groups.items():
        for old_key, aggregate in aggregate_lines(members).items():
            result[f"{problem_hash}|{mode}|{old_key}"] = aggregate
    return result


def build(exp_dir: Path | str, critical_h_c: float | None = None, out_dir: Path | str | None = None,
          walk_dir: Path | str | None = None) -> dict:
    exp_dir = Path(exp_dir)
    out = Path(out_dir) if out_dir else exp_dir / "finding"
    out.mkdir(parents=True, exist_ok=True)
    runs = [(p.parent.name, json.loads(p.read_text())) for p in sorted(exp_dir.glob("lines/*/summary.json"))]
    summaries = [s for _, s in runs]

    lines = {}
    for run_name, s in runs:  # run_name = <line> (v1) or <line>-<run_tag> (v2)
        bp = s.get("best_params") or {}
        holdout = s.get("holdout") or {}
        lines[run_name] = {**{k: s.get(k) for k in _KEYS},
                           "memory_used": bool(s.get("memory_used")),
                           "holdout_min_metric": holdout.get("min_metric"),
                           "holdout_robust": holdout.get("robust"),
                           "n_predictions": len(s.get("predictions") or []),
                           "cliff_distance": _cliff_distance(bp, critical_h_c)}
    scenario = _scenario_named(next((s.get("scenario") for s in summaries if s.get("scenario")), None))
    finding = {"exp_dir": str(exp_dir), "critical_h_c": critical_h_c, "lines": lines,
               "aggregate": _grouped_aggregate(summaries),
               "walk_best": walk_best(exp_dir if walk_dir is None else walk_dir, scenario)}
    (out / "finding.json").write_text(json.dumps(finding, indent=1))

    with (out / "prediction_vs_outcome.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["run", "line", "step", "prediction", "outcome"])
        w.writeheader()
        for run_name, s in runs:
            for row in s.get("predictions") or []:
                w.writerow({"run": run_name, "line": s["line"],
                            **{k: row.get(k) for k in ("step", "prediction", "outcome")}})

    tree = exp_dir / "tree"
    (out / "tree_graph.txt").write_text(Tree(tree).log_graph() if (tree / ".git").exists() else "")
    (out / "finding.md").write_text(_markdown(finding))
    return finding


def _cell(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return (json.dumps(v) if isinstance(v, dict) else str(v)).replace("|", "\\|")  # "A|mem" inside a table cell


def _table(cols: tuple[str, ...], rows: list[dict]) -> str:
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(_cell(r.get(c)) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def _markdown(finding: dict) -> str:
    cols = ("run", "line", "seed", "memory_used", "n_cases", "n_unique_hash", "best_metric", "best_params",
            "stop_reason", "holdout_min_metric", "holdout_robust", "n_predictions", "cliff_distance")
    run_rows = [{"run": name, **finding["lines"][name]} for name in sorted(finding["lines"])]
    agg_rows = [{"group": g, **finding["aggregate"][g]} for g in sorted(finding["aggregate"])]
    wb = finding.get("walk_best")
    wb_line = "null (no walk sweep.jsonl or no trusted row)" if wb is None else (
        f"objective {wb['metric']:.4g} at {json.dumps(wb['params'])} (case {wb['case_id']})")
    return (
        f"# Finding — {finding['exp_dir']}\n\ncritical h/c (walk): {finding['critical_h_c']}\n\n"
        f"walk_best (best trusted design of the walk sweep): {wb_line}\n\n"
        "## Runs (one row per run; run = <line>[-<run_tag>])\n\n"
        + _table(cols, run_rows)
        + "\n\n## Aggregate (per line x memory; std = population std over seeds)\n\n"
        + _table(_AGG_COLS, agg_rows)
        + "\n\n- Where the LLM lines beat Nelder-Mead (A): ___\n"
        "- Where they are worse: ___\n"
        "- Prediction calibration (see prediction_vs_outcome.csv): ___\n"
    )


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description="four finding artefacts from an experiment dir")
    ap.add_argument("exp_dir")
    ap.add_argument("--walk", default=None, help="walk dir; report.json may hold critical_h_c, sweep.jsonl -> walk_best")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    critical = None
    if args.walk and (Path(args.walk) / "report.json").exists():
        critical = json.loads((Path(args.walk) / "report.json").read_text()).get("critical_h_c")
    finding = build(args.exp_dir, critical, args.out, walk_dir=args.walk)
    print(json.dumps(finding, indent=1))
    return finding


if __name__ == "__main__":
    main()
