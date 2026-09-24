"""Base template for the thin adapters (shared ToolRuntime contract side); stdlib only, no experiment knowledge."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

__all__ = ["Experiment", "failure", "provenance", "atomic_json", "sha256_file"]

_PROVENANCE_KEYS = ("run_id", "candidate_id", "call_id", "problem_hash", "design_hash", "input_artifact_ids")


class Experiment:
    def invoke(self, tool: str, inputs: dict, context: dict) -> dict:
        raise NotImplementedError


def failure(kind: str, evidence, *, cut: slice = slice(500)) -> dict:
    return {"status": "failed", "failure": {"kind": kind, "evidence": str(evidence)[cut]}}


def provenance(context: dict, **extra) -> dict:
    return {**{key: context[key] for key in _PROVENANCE_KEYS}, **extra}


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
