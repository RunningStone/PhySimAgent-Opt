"""One SQLite transaction publishes evidence, formal ranking, cost and state."""

from __future__ import annotations

import copy
import json
import sqlite3
import time
import uuid
import hashlib
from pathlib import Path

from .environment import ConfigurationError, ProtocolError


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sum_costs(rows: list[dict]) -> dict:
    """Sum numeric counters; unavailable/unknown quantities never become zero."""
    total = {}
    for row in rows:
        for key, value in row.get("cost", row).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                total[key] = total.get(key, 0) + value if isinstance(total.get(key, 0), (int, float)) else total[key]
            elif value is not None:
                total[key] = value
    return total


class RunStore:
    def __init__(self, run_dir: Path, snapshot: dict | None = None):
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.run_dir / "run.sqlite3", timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS attempts (
                id TEXT PRIMARY KEY, proposal TEXT NOT NULL, reservation TEXT NOT NULL,
                status TEXT NOT NULL, cost TEXT, created REAL NOT NULL, committed REAL);
            CREATE TABLE IF NOT EXISTS records (
                kind TEXT NOT NULL, identity TEXT NOT NULL, attempt_id TEXT NOT NULL,
                body TEXT NOT NULL, PRIMARY KEY(kind, identity));
        """)
        with self.db:
            previous = self.db.execute("SELECT value FROM state WHERE key='snapshot'").fetchone()
            if previous and snapshot is not None and previous[0] != _json(snapshot):
                raise ConfigurationError("resume snapshot differs from the frozen run snapshot")
            if not previous:
                self.db.execute("INSERT INTO state VALUES ('snapshot', ?)", (_json(snapshot or {}),))
        self.snapshot = json.loads(self.db.execute("SELECT value FROM state WHERE key='snapshot'").fetchone()[0])

    def begin_attempt(self, proposal: dict, reservation: dict) -> str:
        attempt_id = "attempt-" + uuid.uuid4().hex
        with self.db:
            self.db.execute("INSERT INTO attempts VALUES (?,?,?,'running',NULL,?,NULL)",
                            (attempt_id, _json(proposal), _json(reservation), time.time()))
        return attempt_id

    def commit_attempt(self, attempt_id: str, observations: list[dict], assessment: dict | None,
                       cost: dict, checkpoint: dict) -> None:
        # Serialize before publication: non-finite and malformed payloads cannot
        # leave a partly visible transaction.
        for observation in observations:
            for name, path in observation.get("artifacts", {}).items():
                artifact = Path(path)
                if not artifact.is_file():
                    raise ProtocolError("cannot commit an absent observation artifact")
                expected = observation.get("artifact_checksums", {}).get(name)
                if expected is not None and hashlib.sha256(artifact.read_bytes()).hexdigest() != expected:
                    raise ProtocolError("observation artifact changed before publication")
        observation_bodies = [(o["observation_id"], _json(o)) for o in observations]
        assessment_body = _json(assessment) if assessment is not None else None
        cost_body, checkpoint_body = _json(cost), _json(checkpoint)
        with self.db:
            row = self.db.execute("SELECT status FROM attempts WHERE id=?", (attempt_id,)).fetchone()
            if not row:
                raise ProtocolError("cannot commit an unknown attempt")
            if row[0] == "committed":
                return
            if row[0] != "running":
                raise ProtocolError("cannot publish an interrupted attempt")
            for identity, body in observation_bodies:
                existing = self.db.execute("SELECT body FROM records WHERE kind='observation' AND identity=?", (identity,)).fetchone()
                if existing and existing[0] != body:
                    raise ProtocolError("an observation id cannot change its immutable evidence")
                self.db.execute("INSERT OR IGNORE INTO records VALUES ('observation',?,?,?)", (identity, attempt_id, body))
            if assessment is not None:
                known = {r[0] for r in self.db.execute("SELECT identity FROM records WHERE kind='observation'")}
                if not set(assessment.get("evidence_refs", [])) <= known:
                    raise ProtocolError("assessment references unpublished evidence")
                identity = _json([assessment.get(k) for k in ("stage_id", "protocol_hash", "design_hash", "replicate_id")])
                self.db.execute("INSERT INTO records VALUES ('assessment_event',?,?,?)", (attempt_id, attempt_id, assessment_body))
                previous = self.db.execute("SELECT body FROM records WHERE kind='assessment' AND identity=?", (identity,)).fetchone()
                if previous and not json.loads(previous[0]).get("rank_eligible") and assessment.get("rank_eligible"):
                    self.db.execute("UPDATE records SET body=?,attempt_id=? WHERE kind='assessment' AND identity=?",
                                    (assessment_body, attempt_id, identity))
                self.db.execute("INSERT OR IGNORE INTO records VALUES ('assessment',?,?,?)", (identity, attempt_id, assessment_body))
            self.db.execute("UPDATE attempts SET status='committed',cost=?,committed=? WHERE id=?",
                            (cost_body, time.time(), attempt_id))
            self.db.execute("INSERT OR REPLACE INTO state VALUES ('checkpoint',?)", (checkpoint_body,))

    def load_committed(self) -> dict:
        attempts, proposals, ledger = [], [], []
        for identity, proposal, reservation, status, cost in self.db.execute(
                "SELECT id,proposal,reservation,status,cost FROM attempts WHERE status!='running' ORDER BY rowid"):
            item = {"attempt_id": identity, "proposal": json.loads(proposal), "reservation": json.loads(reservation),
                    "status": status, "cost": json.loads(cost) if cost else {"status": "unknown"}}
            attempts.append(item)
            proposals.append(copy.deepcopy(item["proposal"]))
            ledger.append({"attempt_id": identity, "cost": copy.deepcopy(item["cost"])})
        output = {"attempts": attempts, "proposals": proposals, "ledger": ledger}
        for kind in ("observation", "assessment"):
            output[kind + "s"] = [json.loads(row[0]) for row in self.db.execute(
                "SELECT body FROM records WHERE kind=? ORDER BY rowid", (kind,))]
        checkpoint = self.db.execute("SELECT value FROM state WHERE key='checkpoint'").fetchone()
        output["checkpoint"] = json.loads(checkpoint[0]) if checkpoint else {}
        output["assessment_history"] = [json.loads(row[0]) for row in self.db.execute("SELECT body FROM records WHERE kind='assessment_event' ORDER BY rowid")]
        return output

    def recover_inflight(self, policy: dict) -> list[dict]:
        recovered = []
        with self.db:
            for identity, proposal, reservation, created in self.db.execute(
                    "SELECT id,proposal,reservation,created FROM attempts WHERE status='running' ORDER BY rowid").fetchall():
                reserve = json.loads(reservation)
                cost = {**reserve, "actual_cost_status": "unknown", "accounting_basis": "reserved_upper_bound"}
                if "wall_seconds" in reserve:
                    # Even unknown work cannot have consumed more elapsed wall
                    # time than the persisted start-to-recovery interval.
                    cost["wall_seconds"] = min(reserve["wall_seconds"], max(0.0, time.time() - created))
                self.db.execute("UPDATE attempts SET status='interrupted',cost=?,committed=? WHERE id=?",
                                (_json(cost), time.time(), identity))
                recovered.append({"attempt_id": identity, "proposal": json.loads(proposal), "status": "interrupted",
                                  "reservation": reserve, "cost": cost})
        return recovered

    def export(self) -> dict:
        result = self.load_committed()
        (self.run_dir / "records.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        return result

    def close(self):
        self.db.close()


def load_observation_cache(cache_dir: Path, key: str) -> dict | None:
    """Read one exact receipt only while all original evidence remains intact."""
    if not isinstance(key, str):
        return None
    receipt_path = Path(cache_dir) / (hashlib.sha256(key.encode()).hexdigest() + ".json")
    try:
        receipt = json.loads(receipt_path.read_text())
        if (not isinstance(receipt, dict) or type(receipt.get("schema_version")) is not int
                or receipt["schema_version"] != 1 or receipt.get("key") != key):
            return None
        observation, recorded = receipt["observation"], receipt["files"]
        if (not isinstance(observation, dict) or observation.get("status") != "completed"
                or observation.get("source") not in {"live", "synthetic"}
                or not isinstance(recorded, dict) or not isinstance(observation.get("artifacts"), dict)):
            return None
        pending = [Path(value).resolve() for value in observation["artifacts"].values()]
        visited = set()
        while pending:
            path = pending.pop()
            name = str(path)
            if name in visited:
                continue
            content = path.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            if recorded.get(name) != digest:
                return None
            visited.add(name)
            if path.suffix.lower() == ".json":
                manifest = json.loads(content)
                if isinstance(manifest, dict) and "files" in manifest:
                    if not isinstance(manifest["files"], dict):
                        return None
                    for relative, checksum in manifest["files"].items():
                        source = (path.parent / relative).resolve()
                        if not source.is_relative_to(path.parent) or recorded.get(str(source)) != checksum:
                            return None
                        pending.append(source)
        if visited != set(recorded):
            return None
        for name, checksum in observation.get("artifact_checksums", {}).items():
            if recorded.get(str(Path(observation["artifacts"][name]).resolve())) != checksum:
                return None
        return observation
    except (OSError, ValueError, TypeError, KeyError, RuntimeError):
        return None


def save_observation_cache(cache_dir: Path, key: str, observation: dict) -> None:
    """Atomically retain a completed original observation and its file digests."""
    import os
    import tempfile

    if (not isinstance(key, str) or not isinstance(observation, dict)
            or observation.get("status") != "completed" or observation.get("source") not in {"live", "synthetic"}
            or not isinstance(observation.get("artifacts"), dict)):
        raise ProtocolError("only completed live or synthetic observations can be cached")
    original = json.loads(_json(observation))
    recorded = {}
    try:
        pending = [Path(value).resolve() for value in original["artifacts"].values()]
        while pending:
            path = pending.pop()
            name = str(path)
            if name in recorded:
                continue
            content = path.read_bytes()
            recorded[name] = hashlib.sha256(content).hexdigest()
            if path.suffix.lower() == ".json":
                manifest = json.loads(content)
                if isinstance(manifest, dict) and "files" in manifest:
                    if not isinstance(manifest["files"], dict):
                        raise ProtocolError("malformed artifact file manifest")
                    for relative, checksum in manifest["files"].items():
                        source = (path.parent / relative).resolve()
                        if not source.is_relative_to(path.parent):
                            raise ProtocolError("artifact manifest references an external file")
                        if hashlib.sha256(source.read_bytes()).hexdigest() != checksum:
                            raise ProtocolError("artifact manifest checksum no longer matches its source")
                        pending.append(source)
        for name, checksum in original.get("artifact_checksums", {}).items():
            if recorded.get(str(Path(original["artifacts"][name]).resolve())) != checksum:
                raise ProtocolError("observation artifact changed before cache publication")
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        raise ProtocolError(f"cannot cache incomplete or altered file evidence: {exc}") from exc
    receipt = _json({"schema_version": 1, "key": key, "observation": original, "files": recorded})
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (hashlib.sha256(key.encode()).hexdigest() + ".json")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix=".receipt-", suffix=".tmp",
                                         dir=directory, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(receipt)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
