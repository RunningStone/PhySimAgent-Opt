"""Serial tool execution for one immutable design candidate."""

from __future__ import annotations

import copy
import importlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from typing import Any, Callable

from .environment import (
    BudgetError,
    DependencyError,
    ProtocolError,
    TaskSpec,
    normalize_evaluation,
    parse_design,
    validate_tool_call,
)


def _entrypoint(spec: TaskSpec) -> Callable[[str, dict[str, Any], dict[str, Any]], dict[str, Any]]:
    module_name, attr = spec.environment["entrypoint"].split(":", 1)
    if module_name == "test":
        raise ProtocolError("test entrypoints require an injected invoke_fn")
    function = getattr(importlib.import_module(module_name), attr)
    if not callable(function):
        raise ProtocolError("environment entrypoint is not callable")
    return function


def _cache_key(tool: str, inputs: dict[str, Any]) -> str:
    def clean(value):
        if isinstance(value, dict):
            transient = {"active", "artifact_id", "id", "candidate_id", "call_id"}
            return {key: clean(item) for key, item in value.items() if key not in transient}
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    return json.dumps([tool, clean(inputs)], sort_keys=True, separators=(",", ":"), default=str)


def _descendants(parent_pid: int) -> set[int]:
    """Return descendants without a process-management dependency."""

    try:
        output = subprocess.run(
            ["ps", "-axo", "pid=,ppid="], capture_output=True, text=True, check=False
        ).stdout
    except OSError:
        return set()
    children: dict[int, list[int]] = {}
    for line in output.splitlines():
        try:
            pid, ppid = (int(part) for part in line.split())
        except (ValueError, TypeError):
            continue
        children.setdefault(ppid, []).append(pid)
    found: set[int] = set()
    pending = list(children.get(parent_pid, []))
    while pending:
        pid = pending.pop()
        if pid in found:
            continue
        found.add(pid)
        pending.extend(children.get(pid, []))
    return found


class ToolRuntime:
    """Execute one candidate's declared tools and own its artifact handles."""

    def __init__(
        self,
        task: TaskSpec,
        invoke_fn: Callable | None = None,
        *,
        limits: dict[str, Any] | None = None,
        clock: Callable[[], float] | None = None,
    ):
        self.task = task
        self.invoke_fn = invoke_fn
        self._worker_entrypoint = None if invoke_fn is not None else task.environment["entrypoint"]
        self.limits = copy.deepcopy(task.limits)
        if limits:
            self.limits.update(copy.deepcopy(limits))
            if "validation_reserve" not in limits:
                self.limits["validation_reserve"] = 0
        self.clock = clock or time.monotonic
        self.started_at = self.clock()
        self.run_id = "run-" + uuid.uuid4().hex[:12]
        self._candidate_counter = 0
        self._call_counter = 0
        self._cache: dict[str, dict[str, Any]] = {}
        self._state: dict[str, Any] | None = None
        self.ledger = {
            "tool_queries": 0,
            "candidate_tool_queries": 0,
            "validation_queries": 0,
            "tool_executions": 0,
            "failed_tool_queries": 0,
            "solve_slots": 0,
            "llm_calls": 0,
            "tool_seconds": 0.0,
        }
        self.orphan_processes: list[int] = []

    def _new_state(self, params=None, options=None) -> dict[str, Any]:
        self._candidate_counter += 1
        return {
            "run_id": self.run_id,
            "candidate_id": f"candidate-{self._candidate_counter}",
            "problem_hash": self.task.problem_hash,
            "design_hash": "sha256:" + __import__("hashlib").sha256(
                json.dumps({"params": params or {}, "options": options or {}}, sort_keys=True).encode()
            ).hexdigest(),
            "params": copy.deepcopy(params or {}),
            "options": copy.deepcopy(options or {}),
            "artifacts": {},
            "validated_warm_starts": {},
        }

    def _ensure_state(self, params=None, options=None) -> dict[str, Any]:
        if self._state is None:
            self._state = self._new_state(params, options)
        return self._state

    def _reserve(self, tool: str, validation: bool) -> None:
        total = self.limits.get("total_tools")
        reserve = int(self.limits.get("validation_reserve") or 0)
        used = self.ledger["tool_queries"]
        if total is not None:
            available = int(total) - used
            if validation:
                if available <= 0:
                    raise BudgetError("tool budget exhausted")
            elif available <= reserve:
                raise BudgetError("remaining tool budget is reserved for validation")
        solving = tool in {"simulate", "evaluate"} or bool(
            set(self.task.tools[tool].get("produces", [])) & {"solution", "evaluation", "raw_case"}
        )
        slots = self.limits.get("solve_slots")
        if solving and slots is not None and self.ledger["solve_slots"] >= int(slots):
            raise BudgetError("solve-slot budget exhausted")
        self.ledger["tool_queries"] += 1
        self.ledger["validation_queries" if validation else "candidate_tool_queries"] += 1
        if solving:
            self.ledger["solve_slots"] += 1

    def _effective_options(self, tool_name: str, candidate: dict[str, Any], explicit: dict[str, Any]) -> dict[str, Any]:
        specs = self.task.task.get("options", {})
        tool = self.task.tools[tool_name]
        overrides = set(tool.get("option_overrides", []))
        effective = {}
        for name in overrides:
            if "default" in specs.get(name, {}):
                effective[name] = copy.deepcopy(specs[name]["default"])
            arg_spec = tool.get("args", {}).get(name, {})
            if "default" in arg_spec:
                effective[name] = copy.deepcopy(arg_spec["default"])
        for name, value in (tool.get("defaults") or {}).items():
            if name in overrides:
                effective[name] = copy.deepcopy(value)
        effective.update(copy.deepcopy(candidate))
        for name in set(explicit) & set(specs):
            if name not in overrides:
                raise ProtocolError(f"ambiguous option {name!r} is not listed in option_overrides")
            effective[name] = copy.deepcopy(explicit[name])
        return effective

    def _call_with_timeout(self, tool: str, inputs: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        wall_limit = self.limits.get("wall_clock_s")
        remaining = None if wall_limit is None else float(wall_limit) - (self.clock() - self.started_at)
        if remaining is not None and remaining <= 0:
            raise TimeoutError("wall-clock budget exhausted")
        tool_limit = self.task.environment.get("settings", {}).get("timeout_s")
        deadlines = [float(value) for value in (remaining, tool_limit) if value is not None]
        deadline = min(deadlines) if deadlines else None
        if self._worker_entrypoint is not None:
            return self._call_worker(tool, inputs, context, deadline)
        if deadline is None or not hasattr(signal, "setitimer"):
            return self.invoke_fn(tool, inputs, context)

        before = _descendants(os.getpid())

        def alarm(_signum, _frame):
            raise TimeoutError("environment tool timed out")

        previous = signal.signal(signal.SIGALRM, alarm)
        signal.setitimer(signal.ITIMER_REAL, max(0.001, deadline))
        try:
            return self.invoke_fn(tool, inputs, context)
        except TimeoutError:
            for pid in sorted(_descendants(os.getpid()) - before, reverse=True):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)

    def _call_worker(
        self,
        tool: str,
        inputs: dict[str, Any],
        context: dict[str, Any],
        deadline: float | None,
    ) -> dict[str, Any]:
        request = {
            "entrypoint": self._worker_entrypoint,
            "tool": tool,
            "inputs": inputs,
            "context": context,
        }
        with tempfile.TemporaryDirectory(prefix="agentloop-tool-") as directory:
            request_path = os.path.join(directory, "request.json")
            response_path = os.path.join(directory, "response.json")
            with open(request_path, "w", encoding="utf-8") as handle:
                json.dump(request, handle, sort_keys=True, default=str)
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "pipeline.agentloop.workflow",
                    "--worker",
                    request_path,
                    response_path,
                ],
                cwd=self.task.repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            try:
                stdout, stderr = process.communicate(timeout=deadline)
            except subprocess.TimeoutExpired as exc:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
                raise TimeoutError("environment tool timed out") from exc
            if process.returncode:
                raise RuntimeError(
                    f"environment worker exited {process.returncode}: {(stderr or stdout)[-500:]}"
                )
            try:
                with open(response_path, encoding="utf-8") as handle:
                    response = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError("environment worker returned invalid JSON") from exc
            if not isinstance(response, dict):
                raise RuntimeError("environment worker returned a non-mapping")
            return response

    def _validate_result(
        self,
        raw: dict[str, Any],
        context: dict[str, Any],
        required_ids: list[str],
        pending: dict[str, dict[str, Any]],
    ) -> None:
        terminal = context["tool_name"] == context["task"]["evaluation"]["tool"]
        legacy = context["task"].get("prompt", "").startswith("legacy:")
        injected_fixture = context["environment"].get("entrypoint", "").startswith("test:")
        strict_terminal = terminal and not legacy and not injected_fixture
        if strict_terminal and raw.get("failure") is None:
            if raw.get("status") != "ok":
                raise ProtocolError("terminal result status is not ok")
            provenance = raw.get("provenance")
            if not isinstance(provenance, dict):
                raise ProtocolError("terminal result lacks provenance")
            required_provenance = {
                "run_id", "candidate_id", "call_id", "problem_hash",
                "design_hash", "input_artifact_ids",
            }
            if not required_provenance <= set(provenance):
                raise ProtocolError("terminal result lacks provenance identity")
            if not isinstance(raw.get("versions"), dict) or not raw["versions"]:
                raise ProtocolError("terminal result lacks versions")
            if not isinstance(raw.get("artifacts"), list):
                raise ProtocolError("terminal result lacks artifacts")
        provenance = raw.get("provenance")
        if isinstance(provenance, dict) and any(key in provenance for key in ("run_id", "candidate_id", "call_id")):
            for key in ("run_id", "candidate_id", "call_id", "problem_hash", "design_hash"):
                if provenance.get(key) != context[key]:
                    raise DependencyError(f"returned {key} does not match the current call")
            if list(provenance.get("input_artifact_ids", [])) != required_ids:
                raise DependencyError("returned input artifacts do not match the current call")
        returned = raw.get("artifacts")
        if returned is not None:
            if not isinstance(returned, list):
                raise ProtocolError("returned artifacts must be a list")
            by_name = {item.get("name"): item for item in returned if isinstance(item, dict)}
            if set(by_name) != set(pending):
                raise DependencyError("returned artifacts do not match declared produces")
            for name, expected in pending.items():
                item = by_name[name]
                if item.get("artifact_id", item.get("id")) != expected["artifact_id"]:
                    raise DependencyError(f"returned artifact {name!r} did not use its reserved id")
        conditions = raw.get("conditions")
        if isinstance(conditions, dict):
            invalidated = set(context["tool"].get("invalidates", []))
            surviving_required = {
                artifact_id
                for name, artifact_id in zip(context["tool"].get("requires", []), required_ids)
                if name not in invalidated
            }
            valid_evidence = surviving_required | {
                item["artifact_id"] for item in pending.values()
            }
            for item in conditions.values():
                if not isinstance(item, dict) or not item.get("artifact_ids"):
                    raise DependencyError("condition lacks evidence")
                if not set(item["artifact_ids"]) <= valid_evidence:
                    raise DependencyError("condition cites stale or foreign evidence")

    def invoke(
        self,
        tool: str,
        args: dict[str, Any] | None = None,
        *,
        _validation: bool = False,
        _phase: str = "search",
    ) -> dict[str, Any]:
        supplied = args or {}
        state = self._ensure_state(supplied.get("params"), supplied.get("options"))
        validated = validate_tool_call({"name": tool, "args": supplied}, self.task, state)
        self._reserve(tool, _validation)
        self._call_counter += 1
        call_id = f"call-{self._call_counter}"
        tool_spec = self.task.tools[tool]
        explicit = validated["args"]
        params = copy.deepcopy(explicit.get("params", state.get("params", {})))
        candidate_options = copy.deepcopy(state.get("options", {}))
        if "options" in explicit and explicit["options"] != candidate_options:
            raise ProtocolError("candidate OPTIONS are immutable during a tool workflow")
        required_handles = {
            name: copy.deepcopy(state["artifacts"][name]) for name in tool_spec.get("requires", [])
        }
        inherited_options = next(
            (
                copy.deepcopy(handle["effective_options"])
                for handle in required_handles.values()
                if isinstance(handle.get("effective_options"), dict)
            ),
            candidate_options,
        )
        effective_options = self._effective_options(tool, inherited_options, explicit)
        required_ids = [handle["artifact_id"] for handle in required_handles.values()]
        pending = {
            name: {
                "name": name,
                "artifact_id": f"{state['candidate_id']}:{call_id}:{name}",
                "candidate_id": state["candidate_id"],
                "problem_hash": state["problem_hash"],
                "design_hash": state["design_hash"],
                "call_id": call_id,
                "active": False,
            }
            for name in tool_spec.get("produces", [])
        }
        tool_args = {
            name: copy.deepcopy(value)
            for name, value in explicit.items()
            if name not in {"params", "options"}
        }
        inputs = {
            "params": params,
            "options": candidate_options,
            "args": tool_args,
            "artifacts": required_handles,
        }
        # Compatibility adapters historically read warm_start_from at top level.
        if "warm_start_from" in explicit:
            inputs["warm_start_from"] = explicit["warm_start_from"]
        context = {
            "run_id": state["run_id"],
            "candidate_id": state["candidate_id"],
            "call_id": call_id,
            "problem_hash": state["problem_hash"],
            "design_hash": state["design_hash"],
            "input_artifact_ids": required_ids,
            "output_artifact_ids": {name: item["artifact_id"] for name, item in pending.items()},
            "environment": copy.deepcopy(self.task.environment),
            "task": copy.deepcopy(self.task.task),
            "workflow": copy.deepcopy(self.task.workflow),
            "tool": copy.deepcopy(tool_spec),
            "tool_name": tool,
            "config_hash": self.task.config_hash,
            "repo_root": str(self.task.repo_root),
            "phase": _phase,
        }
        key = _cache_key(tool, {"phase": _phase, **inputs})
        started = self.clock()
        if key in self._cache:
            raw = copy.deepcopy(self._cache[key])
            cache_hit = True
            old_to_new = {}
            provenance = raw.get("provenance")
            old_required_ids = (
                list(provenance.get("input_artifact_ids", []))
                if isinstance(provenance, dict)
                else []
            )
            old_to_new.update(zip(old_required_ids, required_ids))
            for item in raw.get("artifacts", []) or []:
                if isinstance(item, dict) and item.get("name") in pending:
                    old_to_new[item.get("artifact_id", item.get("id"))] = pending[item["name"]]["artifact_id"]
                    item["artifact_id"] = pending[item["name"]]["artifact_id"]
                    item.pop("id", None)
                    for name in ("candidate_id", "call_id", "problem_hash", "design_hash"):
                        if name in item:
                            item[name] = context[name]
            if isinstance(provenance, dict):
                for name in ("run_id", "candidate_id", "call_id", "problem_hash", "design_hash"):
                    provenance[name] = context[name]
                provenance["input_artifact_ids"] = required_ids
            for condition in (raw.get("conditions") or {}).values():
                if isinstance(condition, dict):
                    condition["artifact_ids"] = [
                        old_to_new.get(artifact_id, artifact_id)
                        for artifact_id in condition.get("artifact_ids", [])
                    ]
            for evidence in raw.get("condition_evidence", []) or []:
                if isinstance(evidence, dict):
                    if "artifact_id" in evidence:
                        evidence["artifact_id"] = old_to_new.get(
                            evidence["artifact_id"], evidence["artifact_id"]
                        )
                    if "candidate_id" in evidence:
                        evidence["candidate_id"] = context["candidate_id"]
        else:
            self.ledger["tool_executions"] += 1
            cache_hit = False
            try:
                response = self._call_with_timeout(tool, copy.deepcopy(inputs), copy.deepcopy(context))
                raw = copy.deepcopy(response) if isinstance(response, dict) else {
                    "failure": {"kind": "protocol_error", "evidence": "environment returned a non-mapping"}
                }
            except TimeoutError as exc:
                raw = {"failure": {"kind": "timeout", "evidence": str(exc)}}
            except Exception as exc:  # a tool failure is candidate data, not a host crash
                raw = {"failure": {"kind": "tool_error", "type": type(exc).__name__, "evidence": str(exc)}}
            self._cache[key] = copy.deepcopy(raw)
        elapsed = max(0.0, self.clock() - started)
        self.ledger["tool_seconds"] += elapsed
        try:
            self._validate_result(raw, context, required_ids, pending)
        except (ProtocolError, DependencyError) as exc:
            raw = {"failure": {"kind": exc.kind, "evidence": str(exc)}}
            pending = {}
        reported_options = raw.get("effective_options")
        if reported_options is not None and (
            not isinstance(reported_options, dict)
            or any(reported_options.get(name) != value for name, value in effective_options.items())
        ):
            raw = {
                "failure": {
                    "kind": "protocol_error",
                    "evidence": "environment effective_options disagree with declared option precedence",
                }
            }
            pending = {}
            reported_options = effective_options
        elif reported_options is None:
            reported_options = effective_options
        if raw.get("failure") is not None:
            self.ledger["failed_tool_queries"] += 1
            pending = {}
        else:
            for name in tool_spec.get("invalidates", []):
                state["artifacts"].pop(name, None)
            for name, handle in pending.items():
                handle["active"] = True
                handle["raw_reference"] = {
                    "case_id": raw.get("case_id"), "input_hash": raw.get("input_hash")
                }
                handle["effective_options"] = copy.deepcopy(reported_options)
                state["artifacts"][name] = handle
        return {
            **raw,
            "cache_hit": cache_hit,
            "effective_options": copy.deepcopy(reported_options),
            "call_id": call_id,
            "artifact_handles": copy.deepcopy(pending),
            "tool_seconds": elapsed,
        }

    def run_candidate(
        self,
        params: dict[str, Any],
        options: dict[str, Any] | None = None,
        *,
        choose_next: Callable | None = None,
        phase: str = "search",
    ) -> dict[str, Any]:
        if phase not in {"search", "validation"}:
            raise ProtocolError("phase must be search or validation")
        parse_design(
            f"PARAMS = {params!r}\nOPTIONS = {(options or {})!r}", self.task
        )
        self._state = self._new_state(params, options or {})
        observations: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        choices: list[dict[str, Any]] = []
        evaluation = None
        stop_reason = None
        failure = None
        mode = self.task.workflow["mode"]
        fixed = iter(self.task.workflow.get("fixed", []))
        per_candidate = self.limits.get("max_tools_per_candidate")
        per_candidate = len(self.task.workflow.get("fixed", [])) if per_candidate is None else int(per_candidate)
        while len(events) < per_candidate:
            try:
                if mode == "fixed":
                    call = {"name": next(fixed), "args": {}}
                else:
                    if choose_next is None:
                        raise ProtocolError("agent workflow requires choose_next")
                    call = None
                    invalid_choices = 0
                    while call is None:
                        total_llm = self.limits.get("total_llm")
                        owner = getattr(choose_next, "__self__", None)
                        total_used = (
                            getattr(owner, "n_llm_calls")
                            if isinstance(getattr(owner, "n_llm_calls", None), int)
                            else self.ledger["llm_calls"]
                        )
                        if total_llm is not None and total_used >= int(total_llm):
                            raise BudgetError("LLM budget exhausted")
                        candidate = choose_next(
                            copy.deepcopy(observations), list(self.task.tools), copy.deepcopy(self.limits)
                        )
                        self.ledger["llm_calls"] += 1
                        try:
                            call = validate_tool_call(candidate, self.task, self._state)
                        except DependencyError:
                            # The call is structurally valid.  Let invoke report
                            # the missing current-candidate dependency without
                            # spending format retries on a physical workflow choice.
                            call = candidate
                        except ProtocolError as exc:
                            if str(exc).startswith(("tool PARAMS", "tool OPTIONS")):
                                # A deliberate attempt to mutate the sealed
                                # candidate is a protocol failure, not a JSON
                                # formatting mistake worth asking the model to repeat.
                                call = candidate
                            else:
                                call = None
                                invalid_choices += 1
                                if invalid_choices >= 3:
                                    raise ProtocolError(
                                        "chooser returned three invalid or unknown tool calls"
                                    )
                    recorded_choice = copy.deepcopy(call)
                    recorded_choice["observation_id"] = (
                        observations[-1]["result_id"] if observations else None
                    )
                    choices.append(recorded_choice)
            except StopIteration:
                stop_reason = "workflow_exhausted"
                break
            except (ProtocolError, BudgetError) as exc:
                failure = {"kind": exc.kind, "evidence": str(exc)}
                stop_reason = exc.kind
                break
            except Exception as exc:
                failure = {"kind": "chooser_error", "type": type(exc).__name__, "evidence": str(exc)}
                stop_reason = "chooser_error"
                break
            try:
                result = self.invoke(
                    call.get("name", call.get("tool")), call.get("args", {}), _phase=phase
                )
            except (ProtocolError, DependencyError, BudgetError) as exc:
                failure = {"kind": exc.kind, "evidence": str(exc)}
                stop_reason = exc.kind
                break
            event = {
                "tool": call.get("name", call.get("tool")),
                "args": copy.deepcopy(call.get("args", {})),
                "result": copy.deepcopy(result),
                "result_id": result["call_id"],
                "observation_id": observations[-1]["result_id"] if observations else None,
            }
            events.append(event)
            observations.append(copy.deepcopy(event))
            if result.get("failure") is not None:
                failure = copy.deepcopy(result["failure"])
                stop_reason = failure.get("kind", "tool_failure")
                break
            if event["tool"] == self.task.task["evaluation"]["tool"]:
                evaluation = normalize_evaluation(result, self.task)
                if evaluation["accepted"]:
                    stop_reason = "evaluation_complete"
                else:
                    failure = evaluation.get("failure") or {"kind": "invalid_evaluation"}
                    stop_reason = "invalid_evaluation"
                break
        if evaluation is None and stop_reason is None:
            stop_reason = "max_tools_per_candidate"
        elif evaluation is None and len(events) >= per_candidate:
            stop_reason = "max_tools_per_candidate"
        accepted = bool(evaluation and evaluation["accepted"])
        final_options = copy.deepcopy(options or {})
        if accepted and not self.task.task.get("prompt", "").startswith("legacy:"):
            final_options = copy.deepcopy(events[-1]["result"].get("effective_options", final_options))
        return {
            "status": "complete" if accepted else "failed",
            "metric": evaluation["metric"] if evaluation else None,
            "accepted": accepted,
            "raw": evaluation["raw"] if evaluation else (events[-1]["result"] if events else {}),
            "constraints": evaluation["constraints"] if evaluation else [],
            "summary": evaluation["summary"] if evaluation else "",
            "failure": failure,
            "stop_reason": stop_reason,
            "tool_calls": events,
            "choices": choices,
            "node_count": 1,
            "journal_entries": 1,
            "eval_json_count": 1 if evaluation else 0,
            "is_bad_design": bool(evaluation and not accepted),
            "llm_calls": self.ledger["llm_calls"],
            "params": copy.deepcopy(params),
            "options": final_options,
            "ledger": copy.deepcopy(self.ledger),
        }

    def validate_final(self, params: dict[str, Any], options: dict[str, Any] | None = None) -> dict[str, Any]:
        self._state = self._new_state(params, options or {})
        tool = self.task.task["evaluation"]["tool"]
        # Host validation is allowed to invoke the terminal tool directly even
        # when candidate-time workflow tools normally produce its prerequisites.
        for name in self.task.tools[tool].get("requires", []):
            self._state["artifacts"][name] = {
                "name": name,
                "artifact_id": f"{self._state['candidate_id']}:private-validation:{name}",
                "candidate_id": self._state["candidate_id"],
                "problem_hash": self._state["problem_hash"],
                "design_hash": self._state["design_hash"],
                "call_id": "private-validation",
                "active": True,
            }
        raw = self.invoke(
            tool,
            {"params": params, "options": options or {}},
            _validation=True,
            _phase="validation",
        )
        scored = normalize_evaluation(raw, self.task, validation=True)
        return {
            **scored,
            "trace": {
                "interface": "ToolRuntime",
                "cache_hit": raw.get("cache_hit", False),
                "real_solver_invocations": 0 if raw.get("cache_hit") else 1,
                "ledger": copy.deepcopy(self.ledger),
            },
            "metadata": copy.deepcopy(raw.get("metadata", {})),
            "versions": copy.deepcopy(raw.get("versions", {})),
        }


__all__ = ["ToolRuntime"]


def _worker_main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[0] != "--worker":
        return 2
    request_path, response_path = argv[1:]
    request = json.loads(open(request_path, encoding="utf-8").read())
    module_name, attr = request["entrypoint"].split(":", 1)
    function = getattr(importlib.import_module(module_name), attr)
    response = function(request["tool"], request["inputs"], request["context"])
    with open(response_path, "w", encoding="utf-8") as handle:
        json.dump(response, handle, sort_keys=True, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(_worker_main(sys.argv[1:]))
