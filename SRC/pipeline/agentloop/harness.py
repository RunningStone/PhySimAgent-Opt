"""Pure-function core of the loop (no aide import): constants, exec-script assembly,
stdout parsing, objective, forced-structure plan parsing, SHEPHERD, STOP, tier
promotion, the line-A Nelder-Mead proposer (v1 REQUIREMENTS §2.1–2.2) and the v2
seed perturbation / multi-seed aggregation (v2 REQUIREMENTS §2)."""

from __future__ import annotations

import json
import copy
import math
from functools import cmp_to_key
import random
import re
import statistics
from collections import Counter
from typing import Callable

from scipy.optimize import Bounds, minimize


def select_topk(assessments: list[dict], k: int) -> list[dict]:
    """Rank distinct, formally confirmed designs within one frozen protocol."""
    if not isinstance(k, int) or isinstance(k, bool) or k < 0:
        raise ValueError("k must be a nonnegative integer")
    identities = {(r.get("stage_id"), r.get("protocol_hash")) for r in assessments}
    if len(identities) > 1:
        raise ValueError("TopK cannot mix physical stages or formal protocols")
    groups = {}
    for row in assessments:
        value = row.get("objective_value")
        if (row.get("rank_eligible") is not True or row.get("role", "formal") != "formal"
                or row.get("completeness") != "complete" or row.get("numerical_status") != "pass"
                or row.get("feasibility") != "feasible" or not isinstance(value, (int, float))
                or isinstance(value, bool) or not math.isfinite(value)):
            continue
        # The store owns formal deduplication. Do not reward repeated attempts
        # if a caller passes raw rather than already committed assessments.
        groups.setdefault(row["design_hash"], {}).setdefault(row.get("replicate_id", 0), row)
    rows = []
    for replicas in groups.values():
        values = list(replicas.values())
        row = copy.deepcopy(values[0])
        row["objective_value"] = sum(r["objective_value"] for r in values) / len(values)
        row["replicate_count"] = len(values)
        rows.append(row)
    def compare(a, b):
        delta = a["objective_value"] - b["objective_value"]
        tolerance = max(float(a.get("ranking_tolerance", 1e-12)), float(b.get("ranking_tolerance", 1e-12)))
        if abs(delta) > tolerance:
            return -1 if delta > 0 else 1
        return (a["design_hash"] > b["design_hash"]) - (a["design_hash"] < b["design_hash"])
    return sorted(rows, key=cmp_to_key(compare))[:k]

PLAN_KEYS = frozenset(
    {"reading", "hypothesis", "change", "expected_direction", "kill_criterion", "prediction", "intent", "confidence"}
)
LINES = ("A", "B", "C")
INTENTS = ("continue", "branch", "backtrack")
EVAL_TAG = "EVAL_JSON:"
# R3 trust boundary: the agent may only assign PARAMS / OPTIONS (substring blacklist, comments stripped).
_FORBIDDEN = ("import", "open(", "subprocess", "os.", "sys.")


def make_exec_code(design_code: str, scenario, line: str, store) -> str:
    """Splice ``design_code`` into ``scenario.eval_template``; lines A/B evaluate scalar-only."""
    stripped = re.sub(r"#.*", "", design_code)  # ponytail: comments are inert; keeps 'ratios.' from tripping 'os.'
    if any(tok in stripped for tok in _FORBIDDEN):
        raise ValueError("design code may only assign PARAMS/OPTIONS")
    return (
        scenario.eval_template.replace("{design_code}", design_code)
        .replace("{scalar_only}", str(line in ("A", "B")))
        .replace("{store}", str(store))
        .replace("{cd_max}", repr(float(scenario.cd_max)))
    )


def parse_term_out(term_out: str) -> dict | None:
    """The last ``EVAL_JSON: {...}`` line of stdout as a dict; None if absent or malformed."""
    for line in reversed(term_out.splitlines()):
        line = line.strip()
        if line.startswith(EVAL_TAG):
            try:
                d = json.loads(line[len(EVAL_TAG):])
            except json.JSONDecodeError:
                return None
            return d if isinstance(d, dict) else None
    return None


def objective(result: dict, scenario) -> float | None:
    """Maximised objective for a trusted ok result, else None (D7: trust != converged)."""
    trust = result.get("trust") or {}
    if result.get("status") != "ok" or not trust or not all(v is True for v in trust.values()):
        return None
    return float(scenario.objective_fn(result))


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _first_json_object(text: str):
    m = re.search(r"```json\s*(.*?)```", text, re.S)
    dec = json.JSONDecoder()
    for cand in ([m.group(1)] if m else []) + [text]:
        for i, ch in enumerate(cand):
            if ch == "{":
                try:
                    return dec.raw_decode(cand, i)[0]
                except json.JSONDecodeError:
                    continue
    return None


def parse_plan(plan_text: str) -> dict:
    """Forced-structure proposal: {valid, missing, **fields}; never raises."""
    obj = _first_json_object(plan_text or "")
    if not isinstance(obj, dict):
        return {"valid": False, "missing": sorted(PLAN_KEYS)}
    missing = sorted(k for k in PLAN_KEYS if k not in obj)
    valid = (
        not missing
        and obj["intent"] in INTENTS
        and _is_num(obj["confidence"]) and 0.0 <= obj["confidence"] <= 1.0
        and _is_num(obj["prediction"])
    )
    return {**obj, "valid": bool(valid), "missing": missing}


def _no_gain(records: list[dict], k: int) -> bool:
    """True if the last k records never beat the best metric of the records before them."""
    if len(records) <= k:
        return False
    prior = [r["metric"] for r in records[:-k] if r["metric"] is not None]
    window = [r["metric"] for r in records[-k:] if r["metric"] is not None]
    if not prior:
        return not window
    return not any(m > max(prior) for m in window)


def _best(records: list[dict]) -> dict | None:
    scored = [r for r in records if not r["is_bug"] and r["metric"] is not None]
    return max(scored, key=lambda r: r["metric"]) if scored else None


def select_parent(records: list[dict], tau: float = 0.4, k: int = 4) -> tuple[int | None, str]:
    """SHEPHERD: draft / debug / backtrack / branch / improve (REQUIREMENTS §2.2)."""
    if not records:
        return None, "draft"
    last = records[-1]
    if last["is_bug"]:
        return last["step"], "debug"
    best = _best(records)
    if best is None:
        return last["step"], "improve"
    if _no_gain(records, k):
        return best["step"], "backtrack"
    conf = last.get("confidence")
    if conf is not None and conf < tau:
        return best["step"], "branch"
    return last["step"], "improve"


def should_stop(records: list[dict], budget: int, target: float | None = None, k_stagnation: int = 8) -> str | None:
    """STOP in fixed order: budget -> target -> stagnation -> None."""
    if len(records) >= budget:
        return "budget"
    best = _best(records)
    if target is not None and best is not None and best["metric"] >= target:
        return "target"
    if len(records) >= k_stagnation and _no_gain(records, k_stagnation):
        return "stagnation"
    return None


def promote_tier(records: list[dict], params_of: Callable[[int], dict], tiers: tuple[int, ...]) -> dict | None:
    """Fidelity ladder: a new global best evaluated below the top tier is re-evaluated at the top tier."""
    if len(tiers) <= 1 or not records:
        return None
    last = records[-1]
    if last["is_bug"] or last["metric"] is None:
        return None
    prior = [r["metric"] for r in records[:-1] if not r["is_bug"] and r["metric"] is not None]
    if prior and last["metric"] <= max(prior):
        return None
    params = dict(params_of(last["step"]))
    if params.get("tier", tiers[0]) == tiers[-1]:
        return None
    return {**params, "tier": tiers[-1]}


def nelder_mead_propose(
    scenario, evaluate: Callable[[dict], float | None], budget: int, x0: dict | None = None
) -> list[dict]:
    """Line A: minimise -objective with scipy Nelder-Mead inside ``scenario.bounds``;
    ``evaluate`` returning None counts as penalty +10. Returns the params in evaluation order.
    ``x0`` (v2) is the start point (default ``scenario.x0``); scipy evaluates it first, so the
    first evaluation is always ``x0`` itself (step 0 = seeded x0, no LLM)."""
    start = dict(scenario.x0) if x0 is None else dict(x0)
    # ponytail: "tier" is a discrete fidelity level, not a design variable; line A keeps it at x0
    fixed = {k: start[k] for k in scenario.param_keys if k == "tier"}
    keys = tuple(k for k in scenario.param_keys if k not in fixed)
    seen: list[dict] = []

    def f(x):
        p = {k: float(v) for k, v in zip(keys, x)} | fixed
        seen.append(p)
        m = evaluate(p)
        return 10.0 if m is None else -float(m)

    minimize(
        f,
        [start[k] for k in keys],
        method="Nelder-Mead",
        bounds=Bounds([scenario.bounds[k][0] for k in keys], [scenario.bounds[k][1] for k in keys]),
        options={"maxfev": budget, "xatol": 1e-4, "fatol": 1e-4},
    )
    return seen[:budget]


# --------------------------------------------------------------------------- #
# v2: seeded start point and multi-seed aggregation (pure functions)
# --------------------------------------------------------------------------- #
X0_PERTURB_FRAC = 0.15  # seed jitter as a fraction of each key's [lo, hi] span


def perturb_x0(scenario, seed: int) -> dict:
    """Seeded start design: seed 0 -> ``dict(scenario.x0)``; otherwise ``random.Random(seed)`` draws
    one ``uniform(-1, 1)`` per continuous key (in ``param_keys`` order; ``tier`` is kept as is and
    consumes no draw) and shifts ``x0[k]`` by ``u * 0.15 * (hi - lo)``, clipped to ``[lo, hi]``."""
    if seed == 0:
        return dict(scenario.x0)
    rng = random.Random(seed)
    out: dict = {}
    for k in scenario.param_keys:
        if k == "tier":
            out[k] = scenario.x0[k]
            continue
        lo, hi = scenario.bounds[k]
        v = float(scenario.x0[k]) + rng.uniform(-1.0, 1.0) * X0_PERTURB_FRAC * (hi - lo)
        out[k] = min(max(v, lo), hi)
    return out


def _mean(vals: list) -> float | None:
    vals = [float(v) for v in vals if v is not None]
    return statistics.fmean(vals) if vals else None


def aggregate_lines(summaries: list[dict]) -> dict:
    """Group run summaries by ``(line, memory_used)`` -> ``"<line>|mem"`` / ``"<line>|nomem"`` and report
    n_runs, best_metric mean / population std / min / max (None best_metric counts in n_runs only),
    holdout_robust_frac, n_failed_mean, n_cases_mean and a stop_reason histogram. Empty input -> {}."""
    groups: dict[str, list[dict]] = {}
    for s in summaries:
        key = f"{s['line']}|{'mem' if s.get('memory_used') else 'nomem'}"
        groups.setdefault(key, []).append(s)
    out: dict = {}
    for key, runs in groups.items():
        metrics = [float(s["best_metric"]) for s in runs if s.get("best_metric") is not None]
        robust = [bool((s.get("holdout") or {}).get("robust")) for s in runs]
        out[key] = {
            "n_runs": len(runs),
            "best_metric_mean": statistics.fmean(metrics) if metrics else None,
            "best_metric_std": statistics.pstdev(metrics) if metrics else None,
            "best_metric_min": min(metrics) if metrics else None,
            "best_metric_max": max(metrics) if metrics else None,
            "holdout_robust_frac": sum(robust) / len(runs),
            "n_failed_mean": _mean([s.get("n_failed") for s in runs]),
            "n_cases_mean": _mean([s.get("n_cases") for s in runs]),
            "stop_reasons": dict(Counter(str(s.get("stop_reason")) for s in runs)),
        }
    return out
