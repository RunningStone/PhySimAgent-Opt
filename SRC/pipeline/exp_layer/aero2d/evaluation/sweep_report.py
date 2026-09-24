"""Metrics-style evaluation of one simblock sweep: report.json (table, failure /
untrusted counters, per-feature monotonicity in h/c for each alpha) and sweep.png
(Cl and the four Cp features vs h/c, one line per alpha; Cl panel overlays the
Zerihan & Zhang 2000 points when a literature JSON is given).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

FEATURES = ("Cl", "cp_peak_x", "cp_peak_val", "sep_x", "pressure_recovery_slope")


def _value(r, key: str):
    if r.status != "ok":
        return None
    if key == "Cl":
        return r.forces["Cl"]
    return None if r.field_summary is None else r.field_summary.get(key)


def _monotone(v: list[float]) -> bool:
    a = np.asarray(v, dtype=float)
    return bool(len(a) >= 2 and (np.all(np.diff(a) >= 0) or np.all(np.diff(a) <= 0)))


def write_report(results: list, params: list, out_dir, literature_json=None) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for r, p in zip(results, params):
        rows.append({"case_id": r.case_id, "h_c": p.h_c, "alpha_deg": p.alpha_deg, "camber": p.camber,
                     "status": r.status, "trusted": r.trusted, "iterations": r.iterations,
                     "failure_type": None if r.failure is None else r.failure["type"],
                     **{k: _value(r, k) for k in FEATURES}})
    failures = Counter(x["failure_type"] for x in rows if x["failure_type"])
    untrusted = Counter(k for r in results if r.status == "ok" for k, v in r.trust.items() if not v)
    alphas = sorted({x["alpha_deg"] for x in rows})
    monotone = {}
    for a in alphas:
        line = sorted((x for x in rows if x["alpha_deg"] == a and x["status"] == "ok"), key=lambda x: x["h_c"])
        monotone[str(a)] = {k: _monotone([x[k] for x in line if x[k] is not None]) for k in FEATURES}

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.2))
    for ax, k in zip(axes, FEATURES):
        for a in alphas:
            line = sorted((x for x in rows if x["alpha_deg"] == a and x[k] is not None), key=lambda x: x["h_c"])
            ax.plot([x["h_c"] for x in line], [x[k] for x in line], "o-", ms=3, label=f"alpha={a}")
        ax.set_xlabel("h/c")
        ax.set_title(k)
        ax.grid(alpha=0.3)
    lit = None
    if literature_json is not None and Path(literature_json).is_file():
        lit = json.loads(Path(literature_json).read_text())
        pts = np.asarray(lit["CL_vs_h_c_alpha1"], dtype=float)
        axes[0].plot(pts[:, 0], -pts[:, 1], "k^", ms=5, label="Zerihan&Zhang 2000 (-CL, alpha=1)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "sweep.png", dpi=120)
    plt.close(fig)

    report = {"n_cases": len(rows), "n_ok": sum(x["status"] == "ok" for x in rows),
              "n_trusted": sum(x["trusted"] for x in rows), "failures": dict(failures),
              "untrusted_flags": dict(untrusted), "monotone_in_h_c": monotone,
              "literature": None if lit is None else lit.get("source"), "rows": rows}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":  # self-check on synthetic data (writes only under OUTPUTs/_dev)
    from pipeline.exp_layer.aero2d import CaseResult, Params, input_hash

    repo = Path(__file__).resolve().parents[5]
    res, prm = [], []
    for a in (0.0, 2.0):
        for h in (0.1, 0.2, 0.4, 0.8):
            p = Params(h, a, 0.04)
            hh = input_hash(p)
            ok = h > 0.1
            res.append(CaseResult(hh[:12], hh, "ok" if ok else "failed",
                                  {"Cl": -1.5 / (h + 0.5) - 0.1 * a, "Cd": 0.05} if ok else None,
                                  {k: ok for k in ("converged", "residual_ok", "force_settled", "yplus_ok", "mesh_ok")},
                                  {"cp_peak_x": 0.3, "cp_peak_val": -3 / (h + 0.5), "sep_x": 0.9, "pressure_recovery_slope": 2.0} if ok else None,
                                  None if ok else {"type": "mesh_invalid", "stage": "mesh", "evidence": "synthetic"},
                                  {"mesh_hash": None, "solver_build": {}}, 500, 1.0))
            prm.append(p)
    rep = write_report(res, prm, repo / "OUTPUTs" / "_dev" / "sweep_report_selfcheck",
                       repo / "CONFIGs" / "v1" / "literature_zerihan2000.json")
    assert rep["n_ok"] == 6 and rep["failures"] == {"mesh_invalid": 2}
    assert rep["monotone_in_h_c"]["0.0"]["Cl"] is True
    print("sweep_report self-check ok:", rep["n_cases"], "cases")
