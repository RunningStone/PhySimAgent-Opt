"""Pure, frozen-protocol assessment of immutable observations."""

from __future__ import annotations

import copy
import math


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def assess(observations: list[dict], protocol: dict) -> dict:
    """Keep completeness, numerical trust, feasibility and ranking distinct.

    Every required condition contributes, using the least favourable objective.
    Failed retries remain evidence; the last observation of each condition is
    its final result. Neither observations nor the protocol are modified.
    """
    identities = {(o.get("candidate_id"), o.get("design_hash")) for o in observations}
    if len(identities) > 1:
        raise ValueError("one assessment cannot combine different designs")
    candidate_id, design_hash = next(iter(identities), (None, None))
    required = list(protocol.get("required_conditions", ["design"]))
    objective = protocol.get("objective", {})
    constraints = protocol.get("constraints", [])
    fields = [objective.get("field")] + [c.get("field") for c in constraints]
    selected = {o.get("condition"): o for o in observations}
    rows = [selected[c] for c in required if c in selected]
    complete = bool(required) and len(rows) == len(required) and all(
        bool(o.get("artifacts")) and all(_number(o.get("raw_metrics", {}).get(f)) for f in fields)
        for o in rows
    )
    checks = [o.get("diagnostics", {}).get(c) for o in rows for c in protocol.get("numerical_checks", [])]
    failed = any(o.get("status") == "failed" for o in rows) or any(c is False for c in checks)
    numerical = "fail" if failed else (
        "pass" if rows and len(rows) == len(required) and all(o.get("status") == "completed" for o in rows)
        and all(c is True for c in checks) else "unknown"
    )
    feasibility, value = "unknown", None
    violations = []
    if complete and numerical == "pass":
        values = []
        for o in rows:
            metrics = o["raw_metrics"]
            raw = float(metrics[objective["field"]])
            direction = objective.get("direction", "maximize")
            if direction == "target":
                score = -abs(raw - float(objective["target"]))
            elif direction in {"maximize", "minimize"}:
                score = raw if direction == "maximize" else -raw
            else:
                raise ValueError(f"unknown objective direction {direction!r}")
            score *= float(objective.get("scale", 1.0))
            for c in constraints:
                actual, threshold = float(metrics[c["field"]]), float(c["threshold"])
                violation = max(0.0, actual - threshold) if c.get("relation") == "max" else max(0.0, threshold - actual)
                violations.append({**copy.deepcopy(c), "condition": o["condition"], "value": actual, "violation": violation})
                if c.get("kind") == "penalty":
                    score -= float(c.get("weight", 0.0)) * violation
            values.append(score)
        value = min(values)
        feasibility = "infeasible" if any(c.get("kind") == "hard" and c["violation"] > 0 for c in violations) else "feasible"
    identity_ok = all(o.get("stage_id") == protocol.get("stage_id") and o.get("protocol_hash") == protocol.get("protocol_hash") for o in rows)
    formal = bool(rows) and all(o.get("role") == "formal" for o in rows)
    source_ok = all(o.get("source") in protocol.get("allowed_sources", ["live"]) for o in rows)
    eligible = complete and numerical == "pass" and feasibility == "feasible" and identity_ok and source_ok and formal
    return {
        "candidate_id": candidate_id, "design_hash": design_hash,
        "stage_id": protocol.get("stage_id"), "protocol_hash": protocol.get("protocol_hash"),
        "replicate_id": protocol.get("replicate_id", 0),
        "completeness": "complete" if complete else "incomplete", "numerical_status": numerical,
        "feasibility": feasibility, "objective_value": value if formal else None,
        "rank_eligible": bool(eligible), "role": "formal" if formal else "exploration",
        "evidence_refs": [o["observation_id"] for o in observations if o.get("observation_id")],
        "diagnostics": {"identity_ok": identity_ok, "source_ok": source_ok, "constraints": violations},
        "raw_metrics": {o["condition"]: copy.deepcopy(o.get("raw_metrics", {})) for o in rows},
    }
