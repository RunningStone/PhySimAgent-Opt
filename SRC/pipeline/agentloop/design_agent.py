"""DesignAgent — subclass of aide.agent.Agent. The only file that imports aide.agent.

Overrides: search_policy (SHEPHERD via select_parent), _draft/_improve/_debug (one
prompt builder ``_ask``), plan_and_code_query (json + python fences), parse_exec_result
(VERIFY without an LLM call), update_data_preview (no-op)."""

from __future__ import annotations

import json
import re
import copy
from pathlib import Path

import aide.backend
from aide.agent import Agent
from aide.interpreter import ExecutionResult
from aide.journal import Journal, Node
from aide.utils.metric import MetricValue, WorstMetricValue
from aide.utils.response import wrap_code

from .harness import PLAN_KEYS, parse_plan, parse_term_out, select_parent, select_topk, _first_json_object
from .environment import BudgetError, ProtocolError
from .scenario import Scenario

_PY_FENCE = re.compile(r"```python\s*\n(.*?)```", re.S)


def _without_response_format_section(text: str) -> str:
    """Hide a task's design-output contract from the tool chooser only."""

    heading = re.search(r"(?m)^## Response format[^\n]*(?:\n|$)", text)
    if heading is None:
        return text
    following = re.search(r"(?m)^## ", text[heading.end():])
    end = len(text) if following is None else heading.end() + following.start()
    return text[: heading.start()] + text[end:]

_INTRO = {
    "draft": "You are an aerodynamic design engineer running a design-simulate loop. Propose the first design to evaluate.",
    "improve": ("You are an aerodynamic design engineer running a design-simulate loop. Below is the previous design "
                "and its simulation result. Read it, then propose the next design (one atomic multiplicative change)."),
    "debug": ("You are an aerodynamic design engineer running a design-simulate loop. The previous case failed or is "
              "not trusted. Diagnose it, pick exactly one action from the recovery menu, then propose the next case."),
}


def records_of(journal: Journal) -> list[dict]:
    """Journal -> the pure-data 'records' list used by select_parent / should_stop / promote_tier."""
    out = []
    for n in journal.nodes:
        pj = getattr(n, "plan_json", None) or {}
        valid = bool(pj.get("valid"))
        out.append({
            "step": n.step,
            "parent_step": n.parent.step if n.parent is not None else None,
            "metric": None if n.is_buggy or n.metric is None else n.metric.value,
            "is_bug": bool(n.is_buggy),
            "confidence": pj.get("confidence") if valid else None,
            "intent": pj.get("intent") if valid else None,
        })
    return out


class DesignAgent(Agent):
    def __init__(self, task_desc: str, cfg, journal: Journal, scenario: Scenario, line: str):
        super().__init__(task_desc, cfg, journal)
        self.scenario = scenario
        self.line = line
        self.tau = 0.4
        self.k_backtrack = 4
        self.last_action: str | None = None
        self.n_llm_calls = 0
        self.backend = None
        self.prompt_trace: list[object] = []
        self.run_store = None
        self.initialization = []
        self.budget_view = {}
        self.llm_events = []
        self.llm_cwd = None
        self.llm_timeout = None

    @property
    def experiment(self):
        return self.scenario.task_spec.experiment if self.scenario.task_spec else None

    def _experiment_view(self):
        records = self.run_store.load_committed() if self.run_store is not None else {"observations": [], "assessments": []}
        top = select_topk(records["assessments"], 1)
        confirmed = {a["design_hash"] for a in records["assessments"] if a.get("rank_eligible")}
        unconfirmed = [o for o in records["observations"] if o.get("role") == "exploration" and o["design_hash"] not in confirmed]
        return records, top[0] if top else None, unconfirmed

    def _reserve_llm(self) -> None:
        """Apply the one configured LLM ceiling to proposals and format retries.

        ToolRuntime applies the same ceiling to standalone chooser callbacks.  A
        bound ``choose_tool`` call is already represented by ``n_llm_calls``, so
        the runtime checks this counter before invoking it and keeps its own
        chooser-only audit counter separately.
        """

        task_spec = self.scenario.task_spec
        limit = None if task_spec is None else task_spec.limits.get("total_llm")
        if limit is not None and self.n_llm_calls >= int(limit):
            raise BudgetError("LLM budget exhausted")

    # ---- SHEPHERD -------------------------------------------------------------
    def search_policy(self) -> Node | None:
        if self.experiment:
            records, _, _ = self._experiment_view()
            last = records["observations"][-1] if records["observations"] else None
            self.last_action = "draft" if last is None else ("debug" if last["status"] == "failed" else "improve")
            return self.journal.nodes[-1] if self.journal.nodes else None
        step, self.last_action = select_parent(records_of(self.journal), self.tau, self.k_backtrack)
        return None if step is None else self.journal.nodes[step]

    # ---- INTERPRET + PROPOSE / DIAGNOSE (one prompt builder) --------------------
    def _draft(self) -> Node:
        return self._ask(None, "draft")

    def _improve(self, parent_node: Node) -> Node:
        return self._ask(parent_node, "improve")

    def _debug(self, parent_node: Node) -> Node:
        return self._ask(parent_node, "debug")

    def _ask(self, parent: Node | None, mode: str) -> Node:
        if self.experiment:
            records, best, unconfirmed = self._experiment_view()
            prompt = {"task": copy.deepcopy(self.scenario.task_spec.task), "experiment": copy.deepcopy(self.experiment),
                      "observations": copy.deepcopy(records["observations"]), "best_confirmed": best,
                      "unconfirmed": unconfirmed, "budget": copy.deepcopy(self.budget_view),
                      "initialization": copy.deepcopy(self.initialization), "instructions": self._instructions(mode)}
            prompt["rejections"] = [copy.deepcopy(p) for p in records.get("proposals", []) if p.get("action") == "invalid_proposal"][-3:]
            plan, code = self.plan_and_code_query(prompt)
            node = Node(plan=plan, code=code, parent=parent)
            node.proposal = _first_json_object(plan)
            return node
        prompt: dict = {
            "Introduction": (
                "You are an engineering design agent running a design-simulate loop."
                if self.scenario.task_spec is not None and not self.scenario.task_spec.task.get("prompt", "").startswith("legacy:")
                else _INTRO[mode]
            ),
            "Task description": self.task_desc,
            "Memory": self._memory(),
        }
        if parent is not None:
            prompt["Previous design"] = {
                "PARAMS": wrap_code(parent.code),
                "Outcome": parent.analysis or "",
                "Result": wrap_code(json.dumps(self._result_view(parent), indent=1), lang="json"),
            }
        if mode == "debug":
            prompt["Recovery menu"] = self.scenario.recovery_menu
        prompt["Instructions"] = self._instructions(mode)
        plan, code = self.plan_and_code_query(prompt)
        return Node(plan=plan, code=code, parent=parent)

    def _memory(self) -> str:
        if self.experiment:
            records, best, unconfirmed = self._experiment_view()
            return json.dumps({"best_confirmed": best, "unconfirmed": unconfirmed, "observations": records["observations"]})
        best = self.journal.get_best_node()
        head = "(no trusted design evaluated yet)" if best is None else (
            f"Global best so far: objective {best.metric.value:.4f} at step {best.step} with {best.code.strip()}"
        )
        return head + "\n\n" + (self.journal.generate_summary() or "(no evaluated designs yet)")

    def _result_view(self, node: Node) -> dict:
        d = getattr(node, "eval_json", None)
        if not d or not isinstance(d.get("result"), dict):
            return {"error": node.term_out[-600:]}
        r = d["result"]
        task_spec = self.scenario.task_spec
        if task_spec is not None and not task_spec.task.get("prompt", "").startswith("legacy:"):
            fields = [task_spec.task.get("objective", {}).get("field")]
            fields.extend(
                item.get("field") for item in task_spec.task.get("constraints", [])
            )
            fields.extend(task_spec.task.get("feedback_fields", []))
            public = {
                key: r.get(key)
                for key in (
                    "case_id",
                    "status",
                    "fidelity",
                    "trust",
                    "failure",
                    "summary",
                    *[name for name in fields if isinstance(name, str)],
                )
            }
            return public
        keys = ("case_id", "status", "forces", "trust", "failure") + (("field_summary",) if self.line == "C" else ())
        return {k: r.get(k) for k in keys}

    def _instructions(self, mode: str) -> list[str]:
        if self.experiment:
            exploratory = (self.experiment["level"], self.experiment["arm"]) == (2, "b")
            actions = ["explore", "submit_for_evaluation", "stop"] if exploratory else ["evaluate_design", "stop"]
            return [
                "Return one JSON object only. Allowed keys: action, candidate, question, evidence_refs, requested_numerics, requested_conditions, expected_cost, stop_reason.",
                "Allowed actions: " + ", ".join(actions) + ". Use stop with stop_reason when evidence or budget supports stopping.",
                "candidate is data: representation, parameters, optional shape, optional parent_candidate_ids. Use the initial representation and frozen parameter/shape domain. No code or file paths.",
                "Read the observations and state the next empirical question. Cite observation_id in evidence_refs and candidate_id in parent_candidate_ids. Reuse or change the best confirmed design or the latest promising observation; unsuccessful numerical evaluation is not a score.",
                "Physical model and all formal numerical settings are fixed by the host. No warm starts or custom evaluation rules.",
                ("Exploration yields raw observations only. You decide when to submit_for_evaluation under fixed settings. Authorized exploration controls: "
                 + json.dumps(self.scenario.task_spec.environment["settings"].get("exploration", {}))) if exploratory else "Submit evaluate_design for each chosen design under the fixed formal protocol.",
            ]
        keys = ", ".join(sorted(PLAN_KEYS))
        out = [
            "Respond with exactly two fenced code blocks: first a ```json block, then a ```python block; nothing else in code fences.",
            f"The JSON object must contain exactly the keys: {keys}. intent is one of continue / branch / backtrack; "
            "confidence is a number in [0, 1]; prediction is a number (the predicted objective).",
            f"The python block must contain only PARAMS = {{...}} with the keys {list(self.scenario.param_keys)} "
            "and optionally OPTIONS = {...}; no imports, no comments, no other statements.",
        ]
        task_spec = self.scenario.task_spec
        legacy = task_spec is None or task_spec.task.get("prompt", "").startswith("legacy:")
        if legacy:
            out.append(
                "One primary change per step, expressed multiplicatively against the previous design; write the numeric "
                "kill criterion before running."
            )
        else:
            configured = task_spec.task.get("proposal_instructions")
            if isinstance(configured, str):
                out.append(configured)
            elif isinstance(configured, list):
                out.extend(str(item) for item in configured)
            else:
                out.append(
                    "Propose any legal joint PARAMS change supported by the observations and state a numeric kill criterion."
                )
        if mode == "draft":
            out.append(f"No design has been evaluated yet: start from the sensitivity memory if present, otherwise near {self.scenario.x0}.")
        if mode == "debug":
            out.append("Apply the chosen recovery-menu action through OPTIONS (or new PARAMS for mark_infeasible); "
                       "for accept_with_caveat add the key \"accept_with_caveat\" with your reason to the JSON block.")
        return out

    def plan_and_code_query(self, prompt, retries: int = 3) -> tuple[str, str]:
        """One LLM call per attempt; plan = text before the python fence, code = the fence body."""
        text = ""
        for _ in range(retries):
            self._reserve_llm()
            self.prompt_trace.append(prompt)
            if self.experiment:
                event = {"prompt": copy.deepcopy(prompt), "usage": "unavailable", "model": str(self.acfg.code.model)}
                self.llm_events.append(event)
                self.n_llm_calls += 1
                try:
                    if self.backend is not None:
                        text = str(self.backend(copy.deepcopy(prompt)))
                        event["source"] = "injected"
                    else:
                        def capture(metadata):
                            event["usage"] = metadata
                        text = str(aide.backend.query(
                            system_message=json.dumps(prompt, ensure_ascii=False, sort_keys=True), user_message=None,
                            model=self.acfg.code.model, temperature=self.acfg.code.temp, isolated=True,
                            cwd=str(self.llm_cwd), timeout=self.llm_timeout, metadata_callback=capture))
                        event["source"] = "live"
                    event["response"] = text
                except Exception as exc:
                    event["error"] = f"{type(exc).__name__}: {exc}"
                    raise
                obj = _first_json_object(text)
                if isinstance(obj, dict):
                    return json.dumps(obj), ""
                event["format_error"] = True
                continue
            text = str(
                self.backend(prompt)
                if self.backend is not None
                else aide.backend.query(system_message=prompt, user_message=None,
                                        model=self.acfg.code.model, temperature=self.acfg.code.temp)
            )
            self.n_llm_calls += 1
            m = _PY_FENCE.search(text)
            if m and "PARAMS" in m.group(1):
                return text[: m.start()].strip(), m.group(1).strip()
            if self.backend is not None and text.lstrip().startswith("PARAMS ="):
                return "", text.strip()
        return text, ""

    def choose_tool(self, observations: list[dict], allowed_tools: list[str], limits: dict) -> dict:
        """Ask the same model for exactly the next legal tool call."""

        public_observations = [
            {
                "tool": item.get("tool"),
                "result": {
                    key: value
                    for key, value in (item.get("result") or {}).items()
                    if key not in {"provenance", "metadata", "versions", "artifact_handles"}
                },
            }
            for item in observations
        ]
        tool_catalog = (
            {
                name: {
                    "description": self.scenario.task_spec.tools[name].get("description", ""),
                    "args": self.scenario.task_spec.tools[name].get("args", {}),
                }
                for name in allowed_tools
            }
            if self.scenario.task_spec is not None
            else {name: {"description": "", "args": {}} for name in allowed_tools}
        )
        prompt = {
            "Introduction": "Choose the next simulation tool for the current immutable candidate.",
            "Task description": _without_response_format_section(self.task_desc),
            "Allowed tools": tool_catalog,
            "Observations": public_observations,
            "Limits": {
                key: limits.get(key)
                for key in ("max_tools_per_candidate", "total_tools", "solve_slots")
            },
            "Instructions": (
                "For this tool-selection call, these instructions override every response-format instruction "
                "inside the Task description. Return one raw JSON object with exactly the two top-level keys "
                "\"name\" and \"args\": {\"name\": <allowed tool>, \"args\": {...}}. The candidate PARAMS "
                "and OPTIONS are already fixed. args may contain only keys declared by that tool's args schema; "
                "when the schema is empty, args must be {}. Do not return reading, hypothesis, PARAMS, OPTIONS, "
                "plan fields, markdown, prose, or code fences."
            ),
        }
        self._reserve_llm()
        self.prompt_trace.append(prompt)
        if self.backend is not None:
            response = self.backend(prompt)
        else:
            wire_prompt = dict(prompt)
            for key in ("Allowed tools", "Observations", "Limits"):
                wire_prompt[key] = json.dumps(
                    prompt[key], ensure_ascii=False, sort_keys=True, default=str
                )
            response = aide.backend.query(
                system_message=wire_prompt,
                user_message=None,
                model=self.acfg.code.model,
                temperature=self.acfg.code.temp,
            )
        text = str(response)
        self.n_llm_calls += 1
        match = re.search(r"```json\s*\n(.*?)```", text, re.S)
        payload = match.group(1) if match else text
        try:
            call = json.loads(payload)
        except json.JSONDecodeError:
            return {"invalid": text[-400:]}
        return call if isinstance(call, dict) else {"invalid": payload[-400:]}

    # ---- VERIFY (no LLM) --------------------------------------------------------
    def parse_exec_result(self, node: Node, exec_result: ExecutionResult) -> None:
        node.absorb_exec_result(exec_result)
        d = parse_term_out("".join(exec_result.term_out))
        node.eval_json = d
        if self.experiment:
            assessment = (d or {}).get("assessment")
            node.analysis = json.dumps(d or {})
            node.is_buggy = bool(node.exc_type) or (d or {}).get("status") == "failed"
            node.metric = MetricValue(assessment["objective_value"], maximize=True) if assessment and assessment.get("rank_eligible") else None
            return
        node.plan_json = parse_plan(node.plan or "")
        if d is None:
            node.analysis = "no EVAL_JSON in output" + (f" ({node.exc_type})" if node.exc_type else "")
            node.is_buggy = True
        else:
            node.analysis = str(d.get("summary", ""))
            node.is_buggy = bool(d.get("is_bug")) or d.get("metric") is None or node.exc_type is not None
        node.metric = WorstMetricValue() if node.is_buggy else MetricValue(float(d["metric"]), maximize=True)

    def update_data_preview(self) -> None:
        return None
