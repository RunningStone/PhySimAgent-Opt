"""A small third-party-style environment loaded only through its entrypoint."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time


NON_CALLABLE = "this is deliberately not callable"
INVOCATION_COUNT = 0


def _provenance(context):
    return {
        key: context[key]
        for key in (
            "run_id",
            "candidate_id",
            "call_id",
            "problem_hash",
            "design_hash",
            "input_artifact_ids",
        )
    }


def _artifact(context, name):
    return {
        "name": name,
        "artifact_id": context["output_artifact_ids"][name],
        "candidate_id": context["candidate_id"],
        "call_id": context["call_id"],
        "problem_hash": context["problem_hash"],
    }


def invoke(tool, inputs, context):
    global INVOCATION_COUNT
    INVOCATION_COUNT += 1
    if tool == "observe_worker":
        return {"worker_pid": os.getpid(), "failure": None}
    if tool == "inspect":
        artifact = _artifact(context, "inspection")
        return {
            "observed": float(inputs["params"]["x"]),
            "artifacts": [artifact],
            "provenance": _provenance(context),
            "versions": {"fixture": "1"},
            "failure": None,
        }
    if tool == "timeout_tree":
        args = inputs["args"]
        worker_pid_path = Path(args["worker_pid_path"])
        child_pid_path = Path(args["child_pid_path"])
        marker_path = Path(args["marker_path"])
        worker_pid_path.write_text(str(os.getpid()), encoding="ascii")
        child = subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import os,pathlib,time; "
                    f"pathlib.Path({str(child_pid_path)!r}).write_text(str(os.getpid())); "
                    "time.sleep(1); "
                    f"pathlib.Path({str(marker_path)!r}).write_text('orphan')"
                ),
            ]
        )
        while not child_pid_path.exists():
            time.sleep(0.01)
        time.sleep(float(args.get("sleep_s", 10.0)))
        return {"worker_pid": os.getpid(), "child_pid": child.pid, "failure": None}
    if tool != "evaluate":
        raise ValueError(f"unknown fixture tool {tool!r}")
    artifact = _artifact(context, "evaluation")
    artifact_id = artifact["artifact_id"]
    return {
        "case_id": f"{context['candidate_id']}:{context['call_id']}",
        "input_hash": context["design_hash"],
        "status": "ok",
        "value": float(inputs["params"]["x"]),
        "limit": 0.5,
        "fidelity": "fixture",
        "trust": {"complete": True},
        "conditions": {
            "fixture": {
                "status": "ok",
                "fidelity": "fixture",
                "artifact_ids": [artifact_id],
            }
        },
        "conditions_completed": ["fixture"],
        "condition_evidence": [
            {
                "condition": "fixture",
                "artifact_id": artifact_id,
                "candidate_id": context["candidate_id"],
            }
        ],
        "artifacts": [artifact],
        "provenance": _provenance(context),
        "versions": {"fixture": "1"},
        "summary": "deterministic third environment",
        "failure": None,
    }
