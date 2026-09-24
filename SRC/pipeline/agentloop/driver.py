"""Orchestration and I/O: ``Tree`` (experiment-lineage git repo), ``run_line`` (one
experiment line A/B/C, all artefacts under exp_dir), ``summarize_sensitivity`` (walk ->
SENSITIVITY.md text) and the YAML entry point ``python -m pipeline.agentloop.driver <yaml>``.

v2: ``run_line`` takes ``seed`` / ``run_tag`` / ``memory`` (None = no memory) and evaluates the
seeded start design ``perturb_x0(scenario, seed)`` as step 0 on every line without an LLM; the
YAML entry runs ``seeds: [..]`` (run_tag ``s<seed>``), optionally in parallel processes."""

from __future__ import annotations

import argparse
import ast
import copy
import contextlib
import fcntl
import json
import subprocess
import time
import hashlib
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import aide.backend
import yaml
from aide.interpreter import ExecutionResult, Interpreter
from aide.journal import Journal, Node
from aide.utils import serialize
from omegaconf import OmegaConf

from .design_agent import DesignAgent, records_of
from .environment import BudgetError, ProtocolError, ConfigurationError, parse_design, resolve_task, validate_proposal, canonical_design, formal_protocol, _sha
from .records import RunStore, sum_costs
from .harness import (
    LINES, make_exec_code, nelder_mead_propose, objective, perturb_x0, promote_tier, should_stop, select_topk,
)
from .scenario import WING_SCENARIO, Scenario, wing_with_cd_max
from .workflow import ToolRuntime

REPO_ROOT = Path(__file__).resolve().parents[3]


def _generic_objective(_result: dict) -> float:
    return 0.0


# --------------------------------------------------------------------------- #
# Tree: thin wrapper around the experiment-lineage git repo (POC PLAN §7)
# --------------------------------------------------------------------------- #
class Tree:
    def __init__(self, root: Path | str):
        self.root = Path(root)

    def _git(self, *args: str) -> str:
        return subprocess.run(
            ["git", "-c", "commit.gpgsign=false", "-C", str(self.root), *args],  # ponytail: one flag beats a signing failure on a dev box
            check=True, capture_output=True, text=True,
        ).stdout

    @classmethod
    def init(cls, root: Path | str) -> "Tree":
        t = cls(root)
        if not (t.root / ".git").exists():
            t.root.mkdir(parents=True, exist_ok=True)
            t._git("init", "-q", "-b", "main")
            t._git("config", "user.name", "poc")
            t._git("config", "user.email", "poc@local")
        return t

    def commit_case(self, subdir_name: str, files: dict[str, str], message: str) -> str:
        d = self.root / "cases" / subdir_name
        d.mkdir(parents=True, exist_ok=True)
        for name, text in files.items():
            (d / name).write_text(text)
        self._git("add", "-A")
        self._git("commit", "-q", "-m", message)
        return self.head()

    def branch(self, name: str, at_sha: str | None = None) -> None:
        self._git("checkout", "-q", "-b", name, *([at_sha] if at_sha else []))

    def checkout(self, ref: str) -> None:
        self._git("checkout", "-q", ref)

    def head(self) -> str:
        return self._git("rev-parse", "HEAD").strip()

    def log_graph(self) -> str:
        return self._git("log", "--graph", "--oneline", "--decorate", "--all")

    def worktree(self, path: Path | str, branch: str) -> "Tree":
        if self._git("branch", "--list", branch).strip():
            self._git("worktree", "add", "-q", str(path), branch)
        else:
            self._git("worktree", "add", "-q", "-b", branch, str(path))
        return Tree(path)


# --------------------------------------------------------------------------- #
# scenario-1 defaults for the block hooks (overridable through line_cfg)
# --------------------------------------------------------------------------- #
def _simblock_run_case(params: dict, options: dict, store, scalar_only: bool = True) -> dict:
    from pipeline.exp_layer.aero2d import Params, RunOptions, run_case

    return json.loads(run_case(Params(**params), RunOptions(**options), scalar_only=scalar_only, store=store).to_json())


def _simblock_verify(params: dict, options: dict, store) -> tuple[bool, str]:
    from pipeline.exp_layer.aero2d import Params, RunOptions, verify

    return verify(Params(**params), RunOptions(**options), store=store)






BLOCK_HOOKS = {"wing": (_simblock_run_case, _simblock_verify)}


def _agent_cfg(model: str, budget: int, timeout: int):
    return OmegaConf.create({
        "agent": {
            "code": {"model": model, "temp": 0.0},
            "feedback": {"model": model, "temp": 0.0},
            "search": {"max_debug_depth": 3, "debug_prob": 0.0, "num_drafts": 1},
            "steps": budget, "k_fold_validation": 1, "expose_prediction": False, "data_preview": False,
        },
        "exec": {"timeout": timeout},
    })


def _params_from_code(code: str) -> dict:
    """PARAMS literal from design code via ast (never exec'd in the driver process)."""
    try:
        for stmt in ast.parse(code).body:
            if isinstance(stmt, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PARAMS" for t in stmt.targets):
                return dict(ast.literal_eval(stmt.value))
    except (SyntaxError, ValueError):
        pass
    return {}


# --------------------------------------------------------------------------- #
# run_line
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def _tree_lock(exp_dir: Path):
    """Serialise repo init + ``git worktree add`` across parallel seeds (one exp_dir, one lock file);
    commits inside distinct worktrees need no lock (own index / HEAD, distinct branches)."""
    exp_dir.mkdir(parents=True, exist_ok=True)
    with (exp_dir / ".tree.lock").open("a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def run_line(line_cfg: dict) -> dict:
    """Run one experiment line; every artefact lands under ``exp_dir`` (tree/, lines/<line>[-<run_tag>]/)."""
    if line_cfg.get("task_spec") is not None and line_cfg["task_spec"].experiment is not None:
        return _run_experiment_line(line_cfg)
    line = line_cfg["line"]
    if (
        line == "walk"
        and line_cfg.get("task_spec") is not None
        and line_cfg["task_spec"].workflow.get("compat_walk")
    ):
        runtime = ToolRuntime(
            line_cfg["task_spec"], invoke_fn=line_cfg.get("invoke_fn"), limits=line_cfg.get("limits")
        )
        params = dict(line_cfg["scenario"].x0)
        terminal_tool = line_cfg["task_spec"].task["evaluation"]["tool"]
        observed = runtime.invoke(terminal_tool, {"params": params, "options": {}})
        event_calls = [{
            "tool": terminal_tool,
            "args": {},
            "result": {key: value for key, value in observed.items() if key != "tool_seconds"},
            "result_id": observed["call_id"],
            "observation_id": None,
        }]
        accepted = observed.get("failure") is None and observed.get("status") == "ok"
        stop_reason = "evaluation_complete" if accepted else (
            (observed.get("failure") or {}).get("kind", "tool_failure")
        )
        return {
            "trace": {
                "interface": "ToolRuntime", "mode": line_cfg["task_spec"].workflow["mode"],
                "optimizer": "walk", "prompts": [], "params_options": [{"params": params, "options": {}}],
                "tool_calls": event_calls,
                "nodes": [{"step": 0, "status": "complete" if accepted else "failed", "metric": None,
                           "journal_entries": 1, "evaluate_calls": 1}],
                "branch_actions": [], "promotion_nodes": [], "llm_calls": 0,
                "total_llm_calls": 0, "stop_reason": stop_reason, "journal": [], "verification": [],
                "candidate_count": 1, "tool_queries": runtime.ledger["tool_queries"],
                "candidate_tool_queries": runtime.ledger["candidate_tool_queries"],
                "validation_queries": 0, "chooser_calls": runtime.ledger["llm_calls"],
                "allowed_tools": list(line_cfg["task_spec"].tools), "choices": [],
                "orphan_processes": runtime.orphan_processes, "ledger": dict(runtime.ledger),
            }
        }
    if line == "walk" and line_cfg.get("task_spec") is not None:
        runtime = ToolRuntime(
            line_cfg["task_spec"], invoke_fn=line_cfg.get("invoke_fn"), limits=line_cfg.get("limits")
        )
        params = dict(line_cfg["scenario"].x0)
        result = runtime.run_candidate(params, {})
        event_calls = copy.deepcopy(result["tool_calls"])
        for event in event_calls:
            event["result"].pop("tool_seconds", None)
        return {
            "trace": {
                "interface": "ToolRuntime", "mode": line_cfg["task_spec"].workflow["mode"],
                "optimizer": "walk", "prompts": [], "params_options": [{"params": params, "options": {}}],
                "tool_calls": event_calls,
                "nodes": [{"step": 0, "status": result["status"], "metric": result["metric"],
                           "journal_entries": 1, "evaluate_calls": sum(e["tool"] == "evaluate" for e in event_calls)}],
                "branch_actions": [], "promotion_nodes": [], "llm_calls": 0,
                "total_llm_calls": 0,
                "stop_reason": result["stop_reason"], "journal": [], "verification": [],
                "candidate_count": 1, "tool_queries": runtime.ledger["tool_queries"],
                "candidate_tool_queries": runtime.ledger["candidate_tool_queries"],
                "validation_queries": 0, "chooser_calls": runtime.ledger["llm_calls"],
                "allowed_tools": list(line_cfg["task_spec"].tools), "choices": result["choices"],
                "orphan_processes": runtime.orphan_processes, "ledger": dict(runtime.ledger),
            }
        }
    if line not in LINES:
        raise ValueError(f"line must be one of {LINES}, got {line!r}")
    scenario: Scenario = line_cfg["scenario"]
    exp_dir = Path(line_cfg.get("output_dir") or line_cfg["exp_dir"])
    store = Path(line_cfg["store"])
    run_tag = line_cfg.get("run_tag")
    run_name = line if run_tag is None else f"{line}-{run_tag}"  # dir lines/<run_name>, branches run/<run_name>/...
    line_dir = exp_dir / "lines" / run_name
    if line_dir.exists():
        raise FileExistsError(f"line directory already exists (run paths are never reused): {line_dir}")
    seed = int(line_cfg.get("seed") or 0)
    x0_used = perturb_x0(scenario, seed)  # step 0 on every line: the seeded start design, no LLM
    budget = int(line_cfg["budget"])
    verify_every = int(line_cfg.get("verify_every", 10) or 0)
    k_stagnation = int(line_cfg.get("k_stagnation", 8))
    timeout = int(line_cfg.get("timeout", 1800))
    model = line_cfg.get("model") or "claude-cli:none"
    verify_fn = line_cfg.get("verify_fn") or _simblock_verify
    run_case_fn = line_cfg.get("run_case_fn") or _simblock_run_case
    memory_used = bool(line_cfg.get("memory"))  # None / "" -> no memory section in the prompt (ablation)
    task_spec = line_cfg.get("task_spec")
    task_desc = scenario.task_desc
    if task_spec is not None and not task_spec.task.get("prompt", "").startswith("legacy:"):
        task_desc = task_spec.task["prompt"]
    if memory_used:
        task_desc += "\n\n## Sensitivity memory (LLM summary of the walk sweep)\n\n" + line_cfg["memory"]
    t_start = time.time()
    runtime = ToolRuntime(task_spec, invoke_fn=line_cfg.get("invoke_fn"), limits=line_cfg.get("limits")) if task_spec else None
    bootstrap_pending = bool(runtime and task_spec.workflow["mode"] == "agent")
    pending_interface_results: list[dict] = []
    trace_tool_calls: list[dict] = []
    trace_choices: list[dict] = []
    trace_nodes: list[dict] = []
    trace_params_options: list[dict] = []
    branch_actions: list[dict] = []
    promotion_nodes: list[int] = []
    verification_events: list[dict] = []
    interface_budget_stop: str | None = None

    with _tree_lock(exp_dir):
        tree = Tree.init(exp_dir / "tree")
        if not tree._git("branch", "--list").strip():  # ponytail: same-file private use; U7 forbids an init commit inside Tree.init
            (tree.root / "README.md").write_text(f"experiment tree of {exp_dir.name}; one commit per evaluated case\n")
            tree._git("add", "-A")
            tree._git("commit", "-q", "-m", "init tree")
        wt = tree.worktree(line_dir, f"run/{run_name}/main")
    store.mkdir(parents=True, exist_ok=True)

    journal = Journal()  # fresh per run: no persistent agent memory (anti-cheat 4)
    agent = DesignAgent(task_desc, _agent_cfg(model, budget, timeout), journal, scenario, line)
    agent.backend = line_cfg.get("backend")
    agent.tau, agent.k_backtrack = float(line_cfg.get("tau", 0.4)), int(line_cfg.get("k_backtrack", 4))
    interp = None if runtime else Interpreter(wt.root, timeout=timeout)
    cases: list[dict] = []
    step_sha: dict[int, str] = {}

    def exec_cb(code: str, reset_session: bool = True) -> ExecutionResult:
        nonlocal bootstrap_pending
        if runtime is not None:
            try:
                params, options = parse_design(code, task_spec)
            except ProtocolError as e:
                return ExecutionResult([f"ValueError: {e}\n"], 0.0, "ValueError")
            before = time.perf_counter()
            original_task = runtime.task
            if bootstrap_pending:
                runtime.task = replace(
                    original_task,
                    workflow={**original_task.workflow, "mode": "fixed"},
                )
            try:
                candidate = runtime.run_candidate(
                    params,
                    options,
                    choose_next=(
                        agent.choose_tool
                        if task_spec.workflow["mode"] == "agent" and not bootstrap_pending
                        else None
                    ),
                )
            finally:
                runtime.task = original_task
                bootstrap_pending = False
            pending_interface_results.append(candidate)
            payload = {
                "metric": candidate["metric"],
                "is_bug": candidate["is_bad_design"] or candidate["metric"] is None,
                "summary": candidate["summary"] or candidate["stop_reason"],
                "result": candidate["raw"],
                "params": params,
                "options": candidate["options"],
            }
            return ExecutionResult(
                ["EVAL_JSON: " + json.dumps(payload, default=str) + "\n"],
                time.perf_counter() - before,
                None,
            )
        try:
            script = make_exec_code(code, scenario, line, store)
        except ValueError as e:
            return ExecutionResult([f"ValueError: {e}\n"], 0.0, "ValueError")
        return interp.run(script, reset_session=True)

    def after_step(node: Node) -> None:
        """Branch/backtrack per the SHEPHERD action, commit the case if the block ran, verify every m cases."""
        nonlocal interface_budget_stop
        interface_result = pending_interface_results.pop(0) if pending_interface_results else None
        if interface_result is not None:
            if interface_result.get("stop_reason") in {"budget_error", "llm_budget"}:
                interface_budget_stop = interface_result["stop_reason"]
            stable_events = copy.deepcopy(interface_result["tool_calls"])
            for event in stable_events:
                event["result"].pop("tool_seconds", None)
            trace_tool_calls.extend(stable_events)
            trace_choices.extend(interface_result["choices"])
            trace_params_options.append({"params": interface_result["params"], "options": interface_result["options"]})
            trace_nodes.append({
                "step": node.step,
                "status": interface_result["status"],
                "metric": interface_result["metric"],
                "journal_entries": interface_result["journal_entries"],
                "evaluate_calls": sum(
                    event["tool"] == task_spec.task["evaluation"]["tool"]
                    for event in interface_result["tool_calls"]
                ),
            })
            branch_actions.append({"step": node.step, "action": agent.last_action or "direct"})
            params_now = interface_result["params"]
            if "tier" in params_now and int(round(float(params_now["tier"]))) == max(scenario.tiers):
                promotion_nodes.append(node.step)
        d = node.eval_json
        if not d or not isinstance(d.get("result"), dict) or "case_id" not in d["result"]:
            return  # design-format failure: journal only, no case
        if agent.last_action == "branch":
            wt.branch(f"run/{run_name}/b{node.step}")
        elif agent.last_action == "backtrack" and node.parent is not None:
            wt.branch(f"run/{run_name}/bt{node.step}", at_sha=step_sha.get(node.parent.step))
        r = d["result"]
        params = d.get("params") or _params_from_code(node.code)
        options = d.get("options") or {}
        metric = None if node.is_buggy else node.metric.value
        name = r["case_id"] if not (wt.root / "cases" / r["case_id"]).exists() else f"{r['case_id']}-s{node.step}"
        reading = (f"# step {node.step} ({agent.last_action or 'direct'})\n\n## Proposal\n\n"
                   f"{node.plan or '(direct evaluation, no LLM proposal)'}\n\n## Outcome\n\n{node.analysis}\n")
        sha = wt.commit_case(name, {
            "params.json": json.dumps(params, indent=1),
            "result.json": json.dumps(r, indent=1),
            "reading.md": reading,
            "proposal.json": json.dumps(node.plan_json, indent=1),
        }, f"{line} step {node.step}: {node.analysis[:72]}")
        step_sha[node.step] = sha
        cases.append({"step": node.step, "case_id": r["case_id"], "input_hash": r.get("input_hash"), "sha": sha,
                      "params": params, "options": options, "status": r.get("status"), "metric": metric,
                      "plan": node.plan_json, "result": copy.deepcopy(r)})
        if verify_every and len(cases) % verify_every == 0:
            if runtime is not None:
                checked = runtime.validate_final(params, options)
                same = bool(checked["accepted"])
                diff = "" if same else json.dumps(
                    {
                        "stop_reason": checked.get("stop_reason"),
                        "failure": checked.get("failure"),
                        "constraints": checked.get("constraints"),
                    },
                    sort_keys=True,
                )
            else:
                out = verify_fn(params, options, store)
                same, diff = out if isinstance(out, tuple) else (bool(out), "")
            with (line_dir / "verify.jsonl").open("a") as fh:
                verified = {"step": node.step, "case_id": r["case_id"], "same": bool(same), "diff": diff}
                fh.write(json.dumps(verified) + "\n")
                verification_events.append(verified)

    def evaluate(params: dict, parent: Node | None = None, plan: str = "") -> float | None:
        """Direct evaluation without an LLM (seeded step 0, line A, tier promotion)."""
        node = Node(code=f"PARAMS = {json.dumps(params)}", plan=plan, parent=parent)
        agent.last_action = None
        agent.parse_exec_result(node, exec_cb(node.code))
        journal.append(node)
        after_step(node)
        return None if node.is_buggy else node.metric.value

    def params_of(step: int) -> dict:
        return next((c["params"] for c in cases if c["step"] == step), {})

    try:
        if line == "A":
            nelder_mead_propose(scenario, evaluate, budget, x0=x0_used)  # scipy evaluates x0 first
            stop = should_stop(records_of(journal), budget, scenario.target, k_stagnation) or "converged"
        else:
            evaluate(x0_used, plan="seed x0")  # step 0: journal node, counts against budget, no LLM
            recs = records_of(journal)
            pending = promote_tier(recs, params_of, scenario.tiers)
            stop = should_stop(recs, budget, scenario.target, k_stagnation)
            while not stop:
                if pending is not None:
                    evaluate(pending, parent=journal.nodes[-1])
                else:
                    try:
                        agent.step(exec_cb)
                    except BudgetError:
                        stop = "llm_budget"
                        break
                    after_step(journal.nodes[-1])
                if interface_budget_stop:
                    stop = interface_budget_stop
                    break
                recs = records_of(journal)
                pending = promote_tier(recs, params_of, scenario.tiers)
                stop = should_stop(recs, budget, scenario.target, k_stagnation)
    finally:
        if interp is not None:
            interp.cleanup_session()

    # hold-out re-scoring of the best trusted design (anti-cheat 1); never shown to the agent
    best = max((c for c in cases if c["metric"] is not None), key=lambda c: c["metric"], default=None)
    holdout: dict = {"delta": scenario.holdout_delta, "best_params": best["params"] if best else None,
                     "best_metric": best["metric"] if best else None, "cases": []}
    if best is not None:
        for sign in (+1.0, -1.0):
            p = dict(best["params"])
            for k, dv in scenario.holdout_delta.items():
                lo, hi = scenario.bounds[k]
                p[k] = min(max(float(p[k]) + sign * dv, lo), hi)
            if "tier" in p:
                p["tier"] = int(round(float(p["tier"])))  # discrete fidelity level (line A carries x0's tier)
            if runtime is not None:
                validated = runtime.validate_final(p, best["options"])
                r, m = validated["raw"], validated["metric"]
            else:
                r = run_case_fn(p, best["options"], store, scalar_only=True)
                m = objective(r, scenario)
            holdout["cases"].append({"params": p, "case_id": r.get("case_id"), "status": r.get("status"),
                                     "trusted": m is not None, "metric": m})
        ms = [c["metric"] for c in holdout["cases"]]
        holdout["min_metric"] = min([m for m in ms if m is not None], default=None)
        holdout["robust"] = all(m is not None for m in ms)

    summary = {
        "line": line,
        "n_cases": len(cases),
        "n_unique_hash": len({c["input_hash"] or c["case_id"] for c in cases}),
        "best_metric": best["metric"] if best else None,
        "best_params": best["params"] if best else None,
        "best_case_id": best["case_id"] if best else None,
        "stop_reason": stop,
        "n_failed": sum(c["status"] == "failed" for c in cases),
        "n_untrusted_accepted": sum(bool(c["plan"].get("accept_with_caveat")) for c in cases),
        "holdout": holdout,
        "predictions": [{"step": c["step"], "prediction": c["plan"]["prediction"], "outcome": c["metric"]}
                        for c in cases if c["plan"].get("valid")],
        "n_steps": len(journal),
        "n_llm_calls": agent.n_llm_calls,
        "model": None if line == "A" else model,
        "seed": seed,
        "run_tag": run_tag,
        "memory_used": memory_used,
        "x0_used": x0_used,
        "scenario": scenario.name,
        "cd_max": scenario.cd_max,
        "exp_dir": str(exp_dir),
        "runtime_s": round(time.time() - t_start, 1),
    }
    if runtime is not None:
        ledger = dict(runtime.ledger)
        summary["trace"] = {
            "interface": "ToolRuntime",
            "mode": task_spec.workflow["mode"],
            "optimizer": "nelder-mead" if line == "A" else "agent",
            "prompts": agent.prompt_trace,
            "params_options": trace_params_options,
            "tool_calls": trace_tool_calls,
            "nodes": trace_nodes,
            "branch_actions": branch_actions,
            "promotion_nodes": promotion_nodes,
            "llm_calls": agent.n_llm_calls,
            "total_llm_calls": agent.n_llm_calls,
            "stop_reason": stop,
            "journal": records_of(journal),
            "verification": {"periodic": verification_events, "holdout": holdout},
            "candidate_count": len(trace_nodes),
            "tool_queries": ledger["tool_queries"],
            "candidate_tool_queries": ledger["candidate_tool_queries"],
            "validation_queries": ledger["validation_queries"],
            "chooser_calls": ledger["llm_calls"],
            "allowed_tools": list(task_spec.tools),
            "choices": trace_choices,
            "orphan_processes": runtime.orphan_processes,
            "ledger": ledger,
        }
        summary.update({
            "problem_hash": task_spec.problem_hash,
            "config_hash": task_spec.config_hash,
            "environment_name": task_spec.environment["name"],
            "workflow_mode": task_spec.workflow["mode"],
            "best_raw_metrics": copy.deepcopy(best["result"]) if best else None,
        })
    (line_dir / "holdout.json").write_text(json.dumps(holdout, indent=1))
    (line_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    serialize.dump_json(journal, line_dir / "journal.json")
    return summary


# --------------------------------------------------------------------------- #
# walk -> SENSITIVITY.md (one LLM call)
# --------------------------------------------------------------------------- #
_SWEEP_COLS = ("h_c", "alpha_deg", "camber", "Cl", "Cd", "trusted", "failure_type", "cp_peak_val", "sep_x")


def summarize_sensitivity(sweep_rows: list[dict], model: str) -> str:
    """One ``claude -p`` call over the sweep table; returns markdown (>= 5 numeric rules)."""
    table = ("| " + " | ".join(_SWEEP_COLS) + " |\n|" + "---|" * len(_SWEEP_COLS) + "\n"
             + "\n".join("| " + " | ".join(str(r.get(c)) for c in _SWEEP_COLS) + " |" for r in sweep_rows))
    system = (
        "You are an aerodynamicist. The table is a parameter sweep of an inverted single-element wing in ground "
        "effect (h_c = ride height / chord, alpha_deg = angle of attack, camber = NACA max camber; Cl < 0 is "
        "downforce; trusted = all solver trust flags true; cp_peak_val = suction-peak Cp; sep_x = separation "
        "position, 1.0 = attached). Write a markdown bullet list of at least 5 sensitivity rules for a design "
        "agent. Every rule must quote numbers from the table (parameter values, Cl/Cd changes, thresholds where "
        "trust is lost or failures appear). Output only the markdown list."
    )
    try:
        return str(aide.backend.query(system_message=system, user_message=table, model=model))
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        raise RuntimeError(f"claude CLI unavailable: {e}") from e


# --------------------------------------------------------------------------- #
# YAML entry point
# --------------------------------------------------------------------------- #
def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: Path | str) -> dict:
    """YAML with ``defaults: [..]`` deep-merged in order (paths relative to the file)."""
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text()) or {}
    merged: dict = {}
    for d in cfg.pop("defaults", None) or []:
        merged = _deep_merge(merged, load_config(path.parent / d))
    return _deep_merge(merged, cfg)


def _scenario_from_cfg(cfg: dict) -> Scenario:
    name = cfg.get("scenario", "wing")
    if name == "wing":
        return wing_with_cd_max(WING_SCENARIO, float(cfg["cd_max"])) if "cd_max" in cfg else WING_SCENARIO
    raise ValueError(f"unknown scenario {name!r}")


def line_cfg_from_yaml(cfg: dict) -> dict:
    """YAML dict -> ``run_line`` config (single run, ``seed`` / no run_tag; the ``seeds`` entry adds run_tag).
    ``use_memory: false`` or ``sensitivity: null`` (or a missing file) -> ``memory`` None."""
    memory = None
    if cfg.get("use_memory", True) and cfg.get("sensitivity") and (REPO_ROOT / cfg["sensitivity"]).exists():
        memory = (REPO_ROOT / cfg["sensitivity"]).read_text() or None
    resolved_cfg = copy.deepcopy(cfg)
    is_compat_walk = bool(
        isinstance(cfg.get("environment"), dict)
        and "scenario" in cfg
        and cfg.get("line", "walk") == "walk"
    )
    if is_compat_walk:
        resolved_cfg.setdefault("workflow", {})["compat_walk"] = True
        legacy_walk_config = {
            key: copy.deepcopy(value)
            for key, value in cfg.items()
            if key not in {"environment", "task", "tools", "workflow", "limits", "compat_source", "output_dir"}
        }
        resolved_cfg.setdefault("environment", {}).setdefault("settings", {})[
            "compat_walk_config"
        ] = legacy_walk_config
    projected_task = resolve_task(resolved_cfg, REPO_ROOT)
    task_spec = projected_task if isinstance(cfg.get("environment"), dict) else None
    if "scenario" in cfg:
        scenario = _scenario_from_cfg(cfg)
        run_case_fn, verify_fn = BLOCK_HOOKS[scenario.name]
    else:
        params = projected_task.task["params"]
        bounds = {
            name: tuple(
                spec["bounds"]
                if "bounds" in spec
                else [min(spec["allowed"]), max(spec["allowed"])]
            )
            for name, spec in params.items()
        }
        x0 = {
            name: spec.get("default", (bounds[name][0] + bounds[name][1]) / 2)
            for name, spec in params.items()
        }
        scenario = Scenario(
            name=projected_task.task["name"],
            task_desc=projected_task.task["prompt"],
            eval_template="",
            param_keys=tuple(params),
            bounds=bounds,
            x0=x0,
            cd_max=0.0,
            holdout_delta={},
            target=None,
            objective_fn=_generic_objective,
            recovery_menu=projected_task.task.get("recovery_menu", "Choose a declared tool."),
        )
        run_case_fn = verify_fn = None
    scenario = replace(scenario, task_spec=task_spec)
    out = {
        "line": cfg.get("line", "walk"),
        "scenario": scenario,
        "run_case_fn": run_case_fn,
        "verify_fn": verify_fn,
        "exp_dir": Path(cfg.get("output_dir") or (REPO_ROOT / cfg.get("output_root", "OUTPUTs") / cfg["exp_name"])),
        "store": REPO_ROOT / cfg.get("store", "OUTPUTs/batch1/aero2d/store"),
        "budget": cfg.get("budget", 1),
        "model": cfg.get("llm_model"),
        "llm_model": cfg.get("llm_model"),
        "tau": cfg.get("tau", 0.4),
        "k_backtrack": cfg.get("k_backtrack", 4),
        "k_stagnation": cfg.get("k_stagnation", 8),
        "verify_every": cfg.get("verify_every", 10),
        "seed": cfg.get("seed", 0),
        "run_tag": None,
        "timeout": int(cfg.get("timeout_s", 900)) + 300,
        "memory": memory,
        "sensitivity": cfg.get("sensitivity"),
        "use_memory": cfg.get("use_memory", True),
        "seeds": cfg.get("seeds"),
        "parallel": cfg.get("parallel"),
        "verify_cases": cfg.get("verify_cases"),
        "tier2_check": cfg.get("tier2_check"),
        "failure_repro": cfg.get("failure_repro"),
        "cd_max": cfg.get("cd_max"),
        "timeout_s": cfg.get("timeout_s", 900),
        "workflow": projected_task.workflow,
        "limits": projected_task.limits,
        "task_spec": task_spec,
        "run_id": cfg.get("run_id"),
    }
    for key in ("invoke_fn", "backend"):
        if key in cfg:
            out[key] = cfg[key]
    return out


def validate_suite(suite: dict) -> dict:
    allowed = {"schema_version", "suite_id", "profiles", "common", "initialization", "topk", "runs", "source_manifest"}
    if not isinstance(suite, dict) or set(suite) - allowed or suite.get("schema_version") != 1:
        raise ConfigurationError("invalid or unknown suite fields")
    if not isinstance(suite.get("suite_id"), str) or not suite["suite_id"]:
        raise ConfigurationError("suite_id is required")
    if set(suite.get("profiles", {})) != {"M0", "M1"}:
        raise ConfigurationError("the frozen physical path requires M0 and M1")
    if not isinstance(suite.get("topk"), int) or isinstance(suite["topk"], bool) or suite["topk"] < 0:
        raise ConfigurationError("topk must be a nonnegative integer")
    expected = {(0, "base"), (1, "a"), (1, "b"), (2, "a"), (2, "b"), (3, "topk")}
    groups, identities = {}, set()
    for run in suite.get("runs", []):
        if set(run) - {"level", "arm", "stages", "replicate_id", "limits"}:
            raise ConfigurationError("unknown run fields")
        level, arm, replica = run.get("level"), run.get("arm"), run.get("replicate_id")
        if (level, arm) not in expected or replica is None:
            raise ConfigurationError("unknown or incomplete run identity")
        identity = (level, arm, str(replica))
        if identity in identities:
            raise ConfigurationError("duplicate run identity")
        identities.add(identity)
        groups.setdefault(str(replica), set()).add((level, arm))
        stages = ["M0", "M1"] if level in {1, 3} else ["M0"]
        if run.get("stages") != stages:
            raise ConfigurationError("run omits or changes a frozen physical node")
        for stage in stages:
            _suite_task_config(suite, run, stage)
    if not groups or any(arms != expected for arms in groups.values()):
        raise ConfigurationError("every replicate requires all six experimental settings")
    initialization = suite.get("initialization", {})
    if set(initialization) != {"parameterized", "explicit_shape"}:
        raise ConfigurationError("both public design initialization sequences are required")
    for representation, candidates in initialization.items():
        if not isinstance(candidates, list) or not candidates:
            raise ConfigurationError("public initialization cannot be empty")
        for candidate in candidates:
            normalized = canonical_design(candidate, suite["common"]["task"].get("geometry", {}))
            if normalized["representation"] != representation:
                raise ConfigurationError("initialization representation mismatch")
    # Paired arms have the same additional budget; all physical settings come
    # from shared profiles, so no arm can tune the formal protocol.
    for replica in groups:
        for level in (1, 2):
            pair = [r for r in suite["runs"] if r["level"] == level and str(r["replicate_id"]) == replica]
            if _deep_merge(suite["common"]["limits"], pair[0].get("limits", {})) != _deep_merge(suite["common"]["limits"], pair[1].get("limits", {})):
                raise ConfigurationError("paired experimental arms must have equal budgets")
    return copy.deepcopy(suite)


def _suite_task_config(suite: dict, run: dict, stage: str) -> dict:
    config = copy.deepcopy(suite["common"])
    settings = copy.deepcopy(suite["profiles"][stage])
    if settings.get("model") != stage:
        raise ConfigurationError("profile does not implement its physical node")
    config["environment"]["settings"] = settings
    if "numerical_checks" in settings:
        config["task"]["trust"] = {"required": settings["numerical_checks"]}
    config["limits"] = _deep_merge(config["limits"], run.get("limits", {}))
    config["experiment"] = {"schema_version": 1, "level": run["level"], "arm": run["arm"], "stage_id": stage,
                            "parent_stage_id": "M0" if stage == "M1" else None, "replicate_id": run["replicate_id"],
                            "initialization": "cold" if (run["level"], run["arm"], stage) == (1, "b", "M1") else
                            ("topk" if stage == "M1" else "public"), "transfer": {}}
    resolve_task(config, REPO_ROOT)
    return config


def prepare_stage(setup: dict, parent_snapshot: dict | None = None) -> list[dict]:
    """Project only transferable designs and fill from the public sequence."""
    public = copy.deepcopy(setup["public_initialization"])
    count = int(setup.get("topk", 0)) if setup.get("initialization") == "topk" and setup.get("topk", 0) > 0 else len(public)
    seeds, seen = [], set()
    if setup.get("initialization") == "topk" and parent_snapshot is not None:
        for row in select_topk(parent_snapshot.get("formal_results", []), int(setup.get("topk", 0))):
            candidate = row.get("candidate")
            if not candidate:
                raise ConfigurationError("a selected parent has no transferable design")
            projected = {k: copy.deepcopy(candidate[k]) for k in ("representation", "parameters", "shape") if k in candidate}
            projected["parent_candidate_ids"] = [row["candidate_id"]]
            seeds.append({"candidate": projected, "source": "parent_topk", "parent_candidate_id": row["candidate_id"],
                          "parent_stage_id": row["stage_id"]})
            seen.add(row["design_hash"])
    for candidate in public:
        if len(seeds) >= count:
            break
        normalized = canonical_design(candidate, setup.get("geometry", {}))
        if normalized["design_hash"] in seen:
            continue
        seen.add(normalized["design_hash"])
        clean = {k: copy.deepcopy(normalized[k]) for k in ("representation", "parameters", "shape") if k in normalized}
        seeds.append({"candidate": clean, "source": "public_initialization"})
    return seeds


def _run_experiment_line(line_cfg: dict) -> dict:
    task = line_cfg["task_spec"]
    directory = Path(line_cfg["output_dir"]).resolve()
    initialization = copy.deepcopy(line_cfg["initialization"])
    model, seed = line_cfg.get("llm_model"), int(line_cfg.get("seed", 0))
    runtime = ToolRuntime(task, invoke_fn=line_cfg.get("invoke_fn"), output_dir=directory / "cases")
    snapshot = {"config_hash": task.config_hash, "protocol": runtime.protocol, "experiment": task.experiment,
                "initialization": initialization, "model": model, "seed": seed,
                "settings": task.environment["settings"], "task": task.task, "limits": task.limits,
                "suite_snapshot_hash": line_cfg.get("suite_snapshot_hash")}
    if (directory / "run.sqlite3").exists() and not line_cfg.get("resume"):
        raise ConfigurationError("run exists; use resume to validate and continue it")
    store = RunStore(directory, snapshot)
    store.recover_inflight(task.workflow.get("recovery", {}))
    records = store.load_committed()
    if records["checkpoint"].get("completed"):
        return records["checkpoint"]["summary"]
    (directory / "snapshot.json").write_text(json.dumps(snapshot, indent=2))
    total = sum_costs(records["ledger"])
    runtime.ledger.update(tool_queries=total.get("tool_queries", 0), solve_slots=total.get("solver_calls", 0),
                          tool_seconds=total.get("tool_seconds", 0), llm_calls=total.get("llm_calls", 0))
    # A resumed wall budget includes committed tool/LLM time and conservative
    # interruption reservations, instead of starting a fresh clock at zero.
    runtime.started_at -= float(total.get("wall_seconds", 0))
    rng = random.Random(seed)
    def tuple_state(value):
        return tuple(tuple_state(v) for v in value) if isinstance(value, list) else value
    if records["checkpoint"].get("rng_state"):
        rng.setstate(tuple_state(records["checkpoint"]["rng_state"]))
    scenario = line_cfg_from_yaml({**line_cfg["config"], "output_dir": str(directory)})["scenario"]
    agent = DesignAgent(task.task["prompt"], _agent_cfg(model, int(task.limits.get("max_candidates", 4)),
                        int(task.limits.get("wall_clock_s", 600))), Journal(), scenario, "C")
    agent.backend = line_cfg.get("backend")
    agent.run_store, agent.initialization = store, initialization
    agent.n_llm_calls = int(total.get("llm_calls", 0))
    agent.llm_cwd = directory / "agent"
    agent.llm_cwd.mkdir(parents=True, exist_ok=True)
    agent.llm_timeout = float(task.limits.get("llm_timeout_s", task.environment["settings"].get("llm_timeout_s", 120)))
    exploratory = (task.experiment["level"], task.experiment["arm"]) == (2, "b")
    max_candidates = int(task.limits.get("max_candidates", 4))
    if max_candidates <= 0:
        raise ConfigurationError("max_candidates must be positive")
    step = int(records["checkpoint"].get("next_step", 0))
    stop_reason, failed = "candidate_budget", False
    while step < max_candidates:
        records = store.load_committed()
        total = sum_costs(records["ledger"])
        elapsed = runtime.clock() - runtime.started_at
        if task.limits.get("wall_clock_s") is not None and elapsed >= float(task.limits["wall_clock_s"]):
            stop_reason = "wall_clock_budget"
            break
        checkpoint = {"next_step": step, "rng_state": rng.getstate(), "visible_evidence": [o["observation_id"] for o in records["observations"]]}
        if records["checkpoint"].get("pending_proposal") is not None:
            proposal = records["checkpoint"]["pending_proposal"]
        elif not exploratory and step < len(initialization):
            proposal = {"action": "evaluate_design", "candidate": initialization[step]["candidate"],
                        "question": "Evaluate the frozen stage initialization", "evidence_refs": []}
        else:
            agent.budget_view = {"limits": copy.deepcopy(task.limits), "spent": total,
                                 "remaining_candidates": max_candidates - step}
            agent.llm_timeout = min(agent.llm_timeout, max(0.001, float(task.limits.get("wall_clock_s", 1e9)) - elapsed))
            if task.limits.get("total_llm") is not None and agent.n_llm_calls >= task.limits["total_llm"]:
                stop_reason = "llm_budget"
                break
            query_attempt = store.begin_attempt({"action": "request_proposal", "step": step},
                                                {"llm_calls": 1, "wall_seconds": agent.llm_timeout})
            before_calls, before_events, started = agent.n_llm_calls, len(agent.llm_events), time.monotonic()
            try:
                parent = agent.search_policy()
                node = agent._ask(parent, agent.last_action or "draft")
                proposal = node.proposal
                llm_error = None
            except Exception as exc:
                proposal, llm_error = None, str(exc)
            events = agent.llm_events[before_events:]
            for index, event in enumerate(events, before_calls + 1):
                (agent.llm_cwd / f"request-{index:04d}.json").write_text(json.dumps(event, indent=2))
            llm_cost = {"llm_calls": agent.n_llm_calls - before_calls, "wall_seconds": time.monotonic() - started,
                        "llm_tokens": "unavailable"}
            usage = [e.get("usage") for e in events]
            if usage and all(isinstance(u, dict) and isinstance(u.get("input_tokens"), int) and isinstance(u.get("output_tokens"), int) for u in usage):
                llm_cost.update(input_tokens=sum(u["input_tokens"] for u in usage), output_tokens=sum(u["output_tokens"] for u in usage))
                for key in ("cache_creation_input_tokens", "cache_read_input_tokens"):
                    llm_cost[key] = sum((u.get("info", {}).get("usage") or {}).get(key, 0) for u in usage)
                llm_cost["llm_tokens"] = sum(llm_cost[k] for k in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
            store.commit_attempt(query_attempt, [], None, llm_cost, {**checkpoint, "pending_proposal": proposal})
            if llm_error is not None:
                stop_reason, failed = "llm_failure: " + llm_error, True
                break
        try:
            proposal = validate_proposal(proposal, task, [o["observation_id"] for o in records["observations"]])
        except (ProtocolError, ValueError, TypeError) as exc:
            attempt = store.begin_attempt({"action": "invalid_proposal", "proposal": proposal, "reason": str(exc)}, {})
            step += 1
            store.commit_attempt(attempt, [], None, {"invalid_proposals": 1}, {**checkpoint, "next_step": step})
            continue
        if proposal["action"] == "stop":
            attempt = store.begin_attempt(proposal, {})
            store.commit_attempt(attempt, [], None, {}, checkpoint)
            stop_reason = proposal["stop_reason"]
            break
        proposal["candidate"]["candidate_id"] = f"L{task.experiment['level']}-{task.experiment['arm']}-{task.experiment['stage_id']}-r{task.experiment['replicate_id']}-candidate-{step:04d}"
        attempts = 1 + int(task.workflow.get("recovery", {}).get("max_retries", task.limits.get("retries", 0)))
        conditions = len(proposal.get("requested_conditions") or runtime.protocol["required_conditions"])
        reservation = {"tool_queries": 4 * conditions * attempts, "solver_calls": conditions * attempts,
                       "wall_seconds": float(task.limits.get("tool_timeout_s", task.environment["settings"].get("timeout_s", 60))) * 4 * conditions * attempts}
        # Check before beginning an attempt: unaffordable work has no execution
        # cost, while every started attempt remains recoverable and chargeable.
        reserve = int(task.limits.get("validation_reserve", 0)) if proposal["action"] == "explore" else 0
        if ((task.limits.get("total_tools") is not None and runtime.ledger["tool_queries"] + reservation["tool_queries"] + reserve > task.limits["total_tools"])
                or (task.limits.get("solve_slots") is not None and runtime.ledger["solve_slots"] + reservation["solver_calls"] + int(bool(reserve)) > task.limits["solve_slots"])):
            stop_reason = "confirmation_reserve" if reserve else "tool_budget"
            break
        attempt = store.begin_attempt(proposal, reservation)
        if proposal["action"] == "explore":
            result = runtime._experiment_run(proposal["candidate"], role="exploration", numerics=proposal["requested_numerics"], conditions=proposal["requested_conditions"])
        else:
            result = runtime.validate_final(proposal["candidate"])
        step += 1
        checkpoint.update(next_step=step, visible_evidence=checkpoint["visible_evidence"] + [o["observation_id"] for o in result["observations"]])
        store.commit_attempt(attempt, result["observations"], result["assessment"], result["cost"], checkpoint)
        (directory / f"attempt-{step:04d}.json").write_text(json.dumps({"proposal": proposal, "result": result}, indent=2))
    records = store.load_committed()
    ranking = select_topk(records["assessments"], len(records["assessments"]))
    sources = sorted({o["source"] for o in records["observations"]})
    failed = failed or bool(records["observations"] and not any(o["status"] == "completed" for o in records["observations"]))
    total_cost = {"solver_calls": 0, "tool_queries": 0, "tool_seconds": 0.0, "wall_seconds": 0.0, "llm_calls": 0,
                  **sum_costs(records["ledger"])}
    summary = {"schema_version": 1, "stage_id": task.experiment["stage_id"], "status": "failed" if failed else "completed",
               "formal_results": ranking, "best_confirmed": ranking[0] if ranking else None,
               "unconfirmed": [o for o in records["observations"] if o["role"] == "exploration" and o["design_hash"] not in {r["design_hash"] for r in ranking}],
               "cost": total_cost, "output_dir": str(directory),
               "source": sources[0] if len(sources) == 1 else ("mixed" if sources else "none"),
               "stop_reason": stop_reason, "protocol_hash": runtime.protocol["protocol_hash"], "initialization": initialization}
    final_attempt = store.begin_attempt({"action": "finish", "stop_reason": stop_reason}, {})
    store.commit_attempt(final_attempt, [], None, {}, {"completed": not failed, "summary": summary,
        "next_step": step, "rng_state": rng.getstate(), "visible_evidence": [o["observation_id"] for o in records["observations"]]})
    store.export()
    (directory / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_suite(suite: dict, output_dir: Path, *, backend=None, invoke_fn=None, resume=False) -> dict:
    suite = validate_suite(suite)
    output_dir = Path(output_dir).resolve()
    # Freeze dirty source, not only HEAD. Cold arms share this static identity,
    # while their evidence databases and process directories remain separate.
    source_files = [Path(__file__), Path(__file__).with_name("design_agent.py"), Path(__file__).with_name("records.py"),
                    Path(__file__).with_name("harness.py"), Path(__file__).parent / "evaluation/finding_report.py",
                    REPO_ROOT / "RELATED_REPOs/aideml/aide/backend/__init__.py",
                    REPO_ROOT / "RELATED_REPOs/aideml/aide/backend/backend_claude_cli.py",
                    REPO_ROOT / "RELATED_REPOs/aideml-poc.patch"]
    manifest = {str(p.relative_to(REPO_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    for run in suite["runs"]:
        for stage in run["stages"]:
            manifest.update(formal_protocol(resolve_task(_suite_task_config(suite, run, stage)))["implementation"])
    snapshot = {"suite": suite, "source_manifest": manifest}
    suite_store = RunStore(output_dir, snapshot)
    suite_store.recover_inflight({})
    summaries, completed = [], {}
    ordered = sorted(suite["runs"], key=lambda r: (str(r["replicate_id"]), r["level"], r["arm"]))
    for run in ordered:
        stage_results = []
        for stage in run["stages"]:
            replica = str(run["replicate_id"])
            if run["level"] == 1 and stage == "M0":
                stage_results.append(copy.deepcopy(completed[(replica, 0, "base", "M0")]))
                continue
            config = _suite_task_config(suite, run, stage)
            representation = "explicit_shape" if run["level"] >= 2 else "parameterized"
            parent = stage_results[-1] if stage_results else None
            directory = output_dir / f"L{run['level']}-{run['arm']}" / f"replicate-{replica}" / stage
            if parent is not None and parent["status"] != "completed":
                # Parent ranking is not final. Do not freeze a child RunStore
                # against temporary seeds that a successful resume may change.
                result = {"schema_version": 1, "stage_id": stage, "status": "blocked", "formal_results": [],
                          "best_confirmed": None, "unconfirmed": [], "initialization": [],
                          "cost": {"solver_calls": 0, "tool_queries": 0, "tool_seconds": 0.0, "wall_seconds": 0.0, "llm_calls": 0},
                          "output_dir": str(directory), "source": "none", "stop_reason": "parent_stage_incomplete",
                          "protocol_hash": formal_protocol(resolve_task(config))["protocol_hash"],
                          "blocked_by": {"stage_id": parent["stage_id"], "status": parent["status"], "output_dir": parent["output_dir"]}}
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "summary.json").write_text(json.dumps(result, indent=2))
                stage_results.append(result)
                continue
            seeds = prepare_stage({"stage_id": stage, "initialization": config["experiment"]["initialization"],
                                   "topk": suite["topk"], "public_initialization": suite["initialization"][representation],
                                   "geometry": config["task"].get("geometry", {})}, parent)
            result = run_line({"task_spec": resolve_task(config), "config": config, "output_dir": directory,
                               "llm_model": config.get("llm_model"), "seed": config.get("seed", 0), "initialization": seeds,
                               "backend": backend, "invoke_fn": invoke_fn, "resume": resume,
                               "suite_snapshot_hash": _sha(snapshot)})
            stage_results.append(result)
            completed[(replica, run["level"], run["arm"], stage)] = result
        summaries.append({"level": run["level"], "arm": run["arm"], "replicate_id": run["replicate_id"],
                          "stages": stage_results, "analysis_cost": sum_costs([s["cost"] for s in stage_results])})
    unique = {s["output_dir"]: s for run in summaries for s in run["stages"]}
    summary = {"schema_version": 1, "suite_id": suite["suite_id"], "runs": summaries,
               "physical_execution_cost": sum_costs([s["cost"] for s in unique.values()]),
               "status": "completed" if all(s["status"] == "completed" for s in unique.values()) else "failed",
               "source_manifest": manifest}
    (output_dir / "suite_snapshot.json").write_text(json.dumps(snapshot, indent=2))
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> dict | list[dict]:
    """``seeds: [..]`` -> one run per seed (run_tag ``s<seed>``), ``parallel: N`` of them at a time in
    worker processes; returns (and prints) the list of summaries. Without ``seeds``: v1 single run."""
    ap = argparse.ArgumentParser(description="run one experiment line (or its seeds) from a YAML config")
    ap.add_argument("yaml", nargs="?")
    ap.add_argument("--suite")
    ap.add_argument("--output-dir")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)
    if args.suite:
        if not args.output_dir:
            ap.error("--suite requires --output-dir")
        summary = run_suite(load_config(args.suite), Path(args.output_dir), resume=args.resume)
        print(json.dumps(summary, indent=1))
        return summary
    if not args.yaml:
        ap.error("a YAML configuration or --suite is required")
    cfg = load_config(args.yaml)
    if "line" not in cfg and not isinstance(cfg.get("environment"), dict):
        _scenario_from_cfg(cfg)
        from pipeline.exp_layer.aero2d.evaluation.walk import main as walk_main
        return walk_main([args.yaml])
    base = line_cfg_from_yaml(cfg)
    seeds = cfg.get("seeds")
    if not seeds:
        summary = run_line(base)
        print(json.dumps(summary, indent=1))
        return summary
    tag = "s{}" if base.get("memory") else "nomem-s{}"  # ablation runs get their own dirs/branches
    jobs = [{**base, "seed": int(sd), "run_tag": tag.format(int(sd))} for sd in seeds]
    parallel = max(1, int(cfg.get("parallel") or 1))
    if parallel == 1:
        summaries = [run_line(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=min(parallel, len(jobs))) as pool:
            summaries = list(pool.map(run_line, jobs))
    print(json.dumps(summaries, indent=1))
    return summaries


if __name__ == "__main__":
    main()
