"""Fixtures and API binding for the v3 environment-interface contract tests.

All construction and representation assumptions live here. Test modules only
assert observable behaviours from REQUIREMENTS.html.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
COMPAT_ROOTS = (REPO_ROOT / "CONFIGs" / "batch1" / "aero2d" / "compat",)
DETERMINISTIC_ENV_YAML = (
    REPO_ROOT / "SRC" / "test" / "exp" / "deterministic_environment.yaml"
)
OLD_RUN_CONFIGS = tuple(
    REPO_ROOT / "CONFIGs" / "batch1" / experiment / "matrix" / name
    for experiment in ("aero2d",)
    for name in (
        "walk.yaml",
        "lineA.yaml",
        "lineB.yaml",
        "lineC.yaml",
        "seeds_lineA.yaml",
        "seeds_lineB.yaml",
        "seeds_lineB_nomem.yaml",
        "seeds_lineC.yaml",
        "seeds_lineC_nomem.yaml",
    )
)


def value(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        assert key in obj, f"missing required result field: {key}"
        return obj[key]
    assert hasattr(obj, key), f"missing required result attribute: {key}"
    return getattr(obj, key)


def plain(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if isinstance(obj, dict):
        return {key: plain(item) for key, item in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(item) for item in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def error_kind(exc: BaseException) -> str:
    explicit = getattr(exc, "kind", None)
    if explicit:
        return str(explicit).lower()
    return exc.__class__.__name__.replace("Error", "_error").lower()


@dataclass(frozen=True)
class InterfaceContract:
    """The single test-side binding to the anchored R2 API."""

    module: Any
    driver: Any

    @property
    def TaskSpec(self):
        return self.module.TaskSpec

    @property
    def ToolRuntime(self):
        return self.module.ToolRuntime

    def resolve_task(self, merged_config: dict[str, Any]):
        return self.module.resolve_task(copy.deepcopy(merged_config), REPO_ROOT)

    def resolve_yaml(self, path: Path):
        merged = self.driver.load_config(path)
        return self.module.resolve_task(merged, REPO_ROOT)

    def parse_design(self, code: str, task: Any):
        return self.module.parse_design(code, task)

    def validate_tool_call(self, call: dict[str, Any], task: Any, state: dict[str, Any]):
        return self.module.validate_tool_call(copy.deepcopy(call), task, copy.deepcopy(state))

    def normalize_evaluation(self, raw: dict[str, Any], task: Any):
        return self.module.normalize_evaluation(copy.deepcopy(raw), task)

    def runtime(self, task: Any, fake_environment: Any, *, limits=None, clock=None):
        return self.module.ToolRuntime(
            task,
            invoke_fn=fake_environment.invoke,
            limits=copy.deepcopy(limits),
            clock=clock,
        )

    def line_cfg_from_yaml(self, path: Path):
        return self.driver.line_cfg_from_yaml(self.driver.load_config(path))


@pytest.fixture(scope="session")
def interface_contract() -> InterfaceContract:
    module = importlib.import_module("pipeline.agentloop")
    driver = importlib.import_module("pipeline.agentloop.driver")
    required = (
        "TaskSpec",
        "ToolRuntime",
        "resolve_task",
        "parse_design",
        "validate_tool_call",
        "normalize_evaluation",
        "run_line",
    )
    missing = [name for name in required if not hasattr(module, name)]
    driver_missing = [
        name for name in ("load_config", "line_cfg_from_yaml") if not hasattr(driver, name)
    ]
    methods = [
        name
        for name in ("run_candidate", "invoke", "validate_final")
        if hasattr(module, "ToolRuntime") and not hasattr(module.ToolRuntime, name)
    ]
    if missing or driver_missing or methods:
        details = []
        if missing:
            details.append("public names: " + ", ".join(missing))
        if driver_missing:
            details.append("driver names: " + ", ".join(driver_missing))
        if methods:
            details.append("ToolRuntime methods: " + ", ".join(methods))
        pytest.fail("v3 environment interface is not implemented; missing " + "; ".join(details))
    return InterfaceContract(module, driver)


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


@pytest.fixture(scope="session")
def artifact_dir_interface() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = REPO_ROOT / "OUTPUTs"  / f"{stamp}-test-interface-runtime"
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "kind": "test",
        "name": "interface-runtime",
        "date": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_head": _git_head(),
        "python": sys.version,
        "platform": platform.platform(),
        "openfoam2512": shutil.which("openfoam2512"),
        "claude_cli": shutil.which("claude"),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output


@pytest.fixture
def wing_spec_mapping() -> dict[str, Any]:
    return {
        "environment": {
            "name": "wing-contract",
            "entrypoint": "test:wing_environment",
            "version": "wing-v1",
            "data": {"dataset": "wing-public-v1", "version": "sha256:data"},
        },
        "task": {
            "name": "wing-contract",
            "prompt": "Minimise signed lift with the configured drag penalty.",
            "params": {
                "h_c": {"type": "float", "bounds": [0.05, 1.0]},
                "alpha_deg": {"type": "float", "bounds": [1.0, 5.0]},
                "camber": {"type": "float", "bounds": [0.0, 0.1]},
            },
            "options": {
                "tier": {"type": "integer", "allowed": [0, 2], "default": 0}
            },
            "objective": {"field": "Cl", "direction": "minimize", "scale": 1.0},
            "constraints": [
                {
                    "field": "Cd",
                    "kind": "penalty",
                    "relation": "max",
                    "threshold": 0.06,
                    "weight": 10.0,
                }
            ],
            "trust": {"required": ["converged", "mesh_ok"]},
            "evaluation": {
                "tool": "evaluate",
                "required_conditions": ["design"],
                "allowed_fidelity": [0, 2],
            },
            "private_validation": {"holdout_path": "/private/holdout.json"},
        },
        "tools": {
            "simulate": {
                "description": "Run the simulation.",
                "requires": [],
                "produces": ["solution"],
                "invalidates": [],
            },
            "diagnose": {
                "description": "Inspect the current solution.",
                "requires": ["solution"],
                "produces": ["diagnosis"],
                "invalidates": [],
            },
            "remesh": {
                "description": "Replace the current mesh.",
                "requires": [],
                "produces": ["mesh"],
                "invalidates": ["solution", "diagnosis"],
            },
            "evaluate": {
                "description": "Produce the final evaluation.",
                "requires": ["solution"],
                "produces": ["evaluation"],
                "invalidates": [],
            },
        },
        "workflow": {"mode": "fixed", "fixed": ["simulate", "evaluate"]},
        "limits": {
            "max_tools_per_candidate": 4,
            "total_tools": 20,
            "solve_slots": 2,
            "total_llm": 8,
            "wall_clock_s": 60.0,
            "validation_reserve": 1,
        },
    }




@pytest.fixture
def make_spec(interface_contract) -> Callable[[dict[str, Any]], Any]:
    return lambda mapping: interface_contract.resolve_task(mapping)


@pytest.fixture
def valid_wing_result() -> dict[str, Any]:
    return {
        "case_id": "case-wing-1",
        "input_hash": "sha256:wing-input",
        "status": "ok",
        "Cl": -1.4,
        "Cd": 0.05,
        "conditions_completed": ["design"],
        "fidelity": 0,
        "trust": {"converged": True, "mesh_ok": True},
        "failure": None,
        "summary": "trusted wing evaluation",
    }




@dataclass
class RecordingEnvironment:
    responses: dict[str, Any]
    calls: list[dict[str, Any]] = field(default_factory=list)

    def invoke(self, tool: str, inputs: dict[str, Any], context: dict[str, Any]):
        self.calls.append(
            {"tool": tool, "inputs": copy.deepcopy(inputs), "context": copy.deepcopy(context)}
        )
        response = self.responses[tool]
        if isinstance(response, BaseException):
            raise response
        if callable(response):
            return response(inputs, context)
        return copy.deepcopy(response)


@dataclass
class AuditedWorkflowEnvironment:
    """Identity-complete two-tool environment for cache and lineage tests."""

    condition_uses_required_input: bool = False
    calls: list[dict[str, Any]] = field(default_factory=list)

    def invoke(self, tool: str, inputs: dict[str, Any], context: dict[str, Any]):
        self.calls.append(
            {"tool": tool, "inputs": copy.deepcopy(inputs), "context": copy.deepcopy(context)}
        )
        provenance = {
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
        output = next(iter(context["output_artifact_ids"]))
        output_id = context["output_artifact_ids"][output]
        artifact = {
            "name": output,
            "artifact_id": output_id,
            "candidate_id": context["candidate_id"],
            "call_id": context["call_id"],
            "problem_hash": context["problem_hash"],
        }
        if tool == "simulate":
            return {
                "case_id": f"{context['candidate_id']}:{context['call_id']}",
                "trust": {"converged": True, "mesh_ok": True},
                "artifacts": [artifact],
                "provenance": provenance,
                "versions": {"fixture": "1"},
                "failure": None,
            }
        evidence_id = (
            context["input_artifact_ids"][0]
            if self.condition_uses_required_input
            else output_id
        )
        return {
            "case_id": f"{context['candidate_id']}:{context['call_id']}",
            "input_hash": context["design_hash"],
            "status": "ok",
            "Cl": -1.0,
            "Cd": 0.05,
            "fidelity": 0,
            "trust": {"converged": True, "mesh_ok": True},
            "conditions": {
                "design": {
                    "status": "ok",
                    "fidelity": 0,
                    "artifact_ids": [evidence_id],
                }
            },
            "conditions_completed": ["design"],
            "condition_evidence": [
                {
                    "condition": "design",
                    "artifact_id": evidence_id,
                    "candidate_id": context["candidate_id"],
                }
            ],
            "artifacts": [artifact],
            "provenance": provenance,
            "versions": {"fixture": "1"},
            "summary": "audited wing fixture",
            "failure": None,
        }


@pytest.fixture
def fake_environment(valid_wing_result):
    return RecordingEnvironment(
        {
            "simulate": {
                "case_id": "sim-1",
                "trust": {"converged": True, "mesh_ok": True},
                "raw": {"residual": 1e-8},
                "failure": None,
            },
            "diagnose": {"diagnosis": "stable", "failure": None},
            "remesh": {"mesh": "mesh-2", "failure": None},
            "evaluate": valid_wing_result,
        }
    )


@pytest.fixture
def make_runtime(interface_contract, make_spec, wing_spec_mapping, fake_environment):
    def factory(*, spec_mapping=None, limits=None, environment=None, clock=None):
        spec = make_spec(spec_mapping or wing_spec_mapping)
        return interface_contract.runtime(
            spec, environment or fake_environment, limits=limits, clock=clock
        )

    return factory


@pytest.fixture
def make_design_agent(interface_contract):
    def factory(task_desc: str):
        old_cfg = interface_contract.line_cfg_from_yaml(OLD_RUN_CONFIGS[2])
        return interface_contract.driver.DesignAgent(
            task_desc,
            interface_contract.driver._agent_cfg("fixture", 2, 60),
            interface_contract.driver.Journal(),
            old_cfg["scenario"],
            "B",
        )

    return factory


@pytest.fixture
def write_task_yaml(tmp_path):
    def writer(mapping: dict[str, Any], name: str = "environment.yaml") -> Path:
        path = tmp_path / name
        path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
        return path

    return writer


@pytest.fixture(scope="session")
def compat_overlays() -> dict[Path, Path]:
    mapping: dict[Path, Path] = {}
    for compat_root in COMPAT_ROOTS:
        if not compat_root.is_dir():
            continue
        for overlay in sorted(compat_root.glob("*.yaml")):
            loaded = yaml.safe_load(overlay.read_text(encoding="utf-8")) or {}
            source = loaded.get("compat_source")
            assert source, f"compat overlay lacks compat_source: {overlay}"
            source_path = Path(source)
            if not source_path.is_absolute():
                source_path = REPO_ROOT / source_path
            source_path = source_path.resolve()
            assert source_path not in mapping, f"duplicate overlay for {source_path}"
            mapping[source_path] = overlay.resolve()
    return mapping


def semantic_config_projection(line_cfg: Any, task: Any, repo_root: Path = REPO_ROOT):
    """Project exactly the R11 semantics; keep this out of product code."""

    config = plain(line_cfg)
    task_data = plain(task)
    task_body = task_data.get("task", task_data)
    environment = task_data.get("environment", {})
    tools = task_data.get("tools", {})
    workflow = task_data.get("workflow", {})
    limits = task_data.get("limits", {})

    memory_path = config.get("sensitivity")
    memory_hash = None
    if config.get("use_memory", True) and memory_path:
        path = Path(memory_path)
        if not path.is_absolute():
            path = repo_root / path
        if path.is_file():
            memory_hash = hashlib.sha256(path.read_bytes()).hexdigest()

    return {
        "algorithm": config.get("line"),
        "budget": config.get("budget"),
        "seed": config.get("seed"),
        "seeds": config.get("seeds"),
        "parallel": config.get("parallel"),
        "parameter_order": list(task_body.get("params", {})),
        "model": config.get("llm_model"),
        "memory": {"sensitivity_hash": memory_hash, "journal_enabled": True},
        "options_defaults": {
            key: spec.get("default") for key, spec in task_body.get("options", {}).items()
        },
        "scoring": {
            "objective": task_body.get("objective"),
            "constraints": task_body.get("constraints"),
            "trust": task_body.get("trust"),
        },
        "tiers": task_body.get("evaluation", {}).get("allowed_fidelity"),
        "verify": {
            "every": config.get("verify_every"),
            "cases": config.get("verify_cases"),
        },
        "holdout": config.get("holdout"),
        "timeouts": {"per_case_s": config.get("timeout_s")},
        "walk": {
            "tier2_check": config.get("tier2_check"),
            "failure_repro": config.get("failure_repro"),
        },
        "cd_max": config.get("cd_max"),
        "problem_hash": value(task, "problem_hash"),
        "tool_names": list(tools),
        "workflow": workflow,
        "limits": limits,
        "environment_data": environment.get("data"),
    }


def semantic_run_projection(result: Any):
    result_data = plain(result)
    trace = result_data.get("trace") or {}
    if trace.get("journal") is not None:
        journal = trace["journal"]
    else:
        line_dir = Path(result_data["exp_dir"]) / "lines" / result_data["line"]
        journal_path = line_dir / "journal.json"
        stored = json.loads(journal_path.read_text(encoding="utf-8"))
        step_by_id = {node["id"]: node["step"] for node in stored["nodes"]}
        journal = [
            {
                "step": node["step"],
                "parent_step": step_by_id.get(stored["node2parent"].get(node["id"])),
                "metric": node["metric"]["value"],
                "is_bug": node["is_buggy"],
                "confidence": None,
                "intent": None,
            }
            for node in stored["nodes"]
        ]
    return {
        "n_cases": result_data["n_cases"],
        "best_metric": result_data["best_metric"],
        "best_params": result_data["best_params"],
        "stop_reason": result_data["stop_reason"],
        "n_failed": result_data["n_failed"],
        "n_steps": result_data["n_steps"],
        "n_llm_calls": result_data["n_llm_calls"],
        "predictions": result_data["predictions"],
        "holdout": result_data["holdout"],
        "journal": journal,
    }


@dataclass
class ScriptedBackend:
    responses: list[Any]
    prompts: list[str] = field(default_factory=list)

    def __call__(self, prompt: str):
        self.prompts.append(prompt)
        assert self.responses, "scripted backend exhausted"
        return copy.deepcopy(self.responses.pop(0))


@dataclass
class ScriptedEnvironment:
    responses: list[Any]
    calls: list[dict[str, Any]] = field(default_factory=list)

    def invoke(self, tool: str, inputs: dict[str, Any], context: dict[str, Any]):
        self.calls.append(
            {"tool": tool, "inputs": copy.deepcopy(inputs), "context": copy.deepcopy(context)}
        )
        assert self.responses, "scripted environment exhausted"
        return copy.deepcopy(self.responses.pop(0))


@dataclass
class ScriptedLegacyHooks:
    responses: list[Any]
    task: Any
    calls: list[dict[str, Any]] = field(default_factory=list)
    verification_calls: list[dict[str, Any]] = field(default_factory=list)

    def _run_case(self, params, options, store, scalar_only, *, native_result):
        self.calls.append(
            {
                "params": copy.deepcopy(params),
                "options": copy.deepcopy(options),
                "store": str(store),
                "scalar_only": scalar_only,
            }
        )
        assert self.responses, "scripted legacy evaluator exhausted"
        raw = copy.deepcopy(self.responses.pop(0))
        body = plain(self.task).get("task", plain(self.task))
        metric = None
        if raw.get("failure") is None and raw.get("trust") and all(raw["trust"].values()):
            objective = body["objective"]
            raw_value = raw[objective["field"]]
            metric = float(raw_value) * float(objective.get("scale", 1.0))
            if objective["direction"] == "minimize":
                metric = -metric
            for constraint in body.get("constraints", []):
                measured = float(raw[constraint["field"]])
                threshold = float(constraint["threshold"])
                violation = (
                    max(0.0, measured - threshold)
                    if constraint["relation"] == "max"
                    else max(0.0, threshold - measured)
                )
                if constraint["kind"] == "hard" and violation:
                    metric = None
                    break
                if constraint["kind"] == "penalty":
                    metric -= float(constraint["weight"]) * violation
        result = raw
        if native_result:
            result = {
                "case_id": raw.get("case_id"),
                "input_hash": raw.get("input_hash"),
                "status": raw.get("status", "failed" if raw.get("failure") else "ok"),
                "trust": copy.deepcopy(raw.get("trust", {})),
                "field_summary": None,
                "failure": copy.deepcopy(raw.get("failure")),
                "iterations": 1,
                "runtime_s": 0.0,
            }
            if body["objective"]["field"] == "Cl":
                result["forces"] = {
                    "Cl": raw.get("Cl"),
                    "Cd": raw.get("Cd"),
                }
            else:
                result["objectives"] = {
                    "charge_time_s": raw.get("t_s"),
                    "soc_reached": raw.get("SOC"),
                    "T_max": raw.get("T_max"),
                    "dT": raw.get("dT", 0.0),
                }
        return {
            "metric": metric,
            "is_bug": metric is None,
            "summary": raw.get("summary", "scripted legacy evaluation"),
            "result": result,
            "params": copy.deepcopy(params),
            "options": copy.deepcopy(options),
        }

    def run_case(self, params, options, store, scalar_only=True):
        return self._run_case(
            params, options, store, scalar_only, native_result=True
        )["result"]

    def interpreted_case(self, params, options, store, scalar_only=True):
        return self._run_case(
            params, options, store, scalar_only, native_result=False
        )

    def verify(self, params, options, store):
        self.verification_calls.append(
            {"params": copy.deepcopy(params), "options": copy.deepcopy(options), "store": str(store)}
        )
        return True, "scripted legacy verification"


class ScriptedLegacyInterpreter:
    """Interpreter-bound legacy evaluator used by deterministic replay tests."""

    def __init__(
        self,
        hooks: ScriptedLegacyHooks,
        execution_result_type: Any,
        params_from_code: Callable[[str], dict[str, Any]],
        working_dir,
    ):
        self.hooks = hooks
        self.execution_result_type = execution_result_type
        self.params_from_code = params_from_code
        self.working_dir = Path(working_dir)

    def run(self, code: str, reset_session=True):
        body = plain(self.hooks.task).get("task", plain(self.hooks.task))
        params = self.params_from_code(code)
        options = {
            key: spec["default"] for key, spec in body.get("options", {}).items()
        }
        payload = self.hooks.interpreted_case(params, options, self.working_dir)
        return self.execution_result_type(
            term_out=["EVAL_JSON: " + json.dumps(payload) + "\n"],
            exec_time=0.0,
            exc_type=None,
        )

    def cleanup_session(self):
        return None


def compatibility_script(case: str, task: Any) -> dict[str, Any]:
    """Deterministic old/new replay data owned by the test suite."""

    task_body = plain(task).get("task", plain(task))
    params = {
        key: (float(spec["bounds"][0]) + float(spec["bounds"][1])) / 2
        for key, spec in task_body["params"].items()
    }
    options = {
        key: spec["default"] for key, spec in task_body.get("options", {}).items()
    }
    design = f"PARAMS = {params!r}\nOPTIONS = {options!r}"
    raw = {
        "case_id": f"replay-{case}",
        "input_hash": f"sha256:{case}",
        "status": "ok",
        "conditions_completed": task_body["evaluation"]["required_conditions"],
        "fidelity": options.get("tier", 0),
        "trust": {key: True for key in task_body["trust"]["required"]},
        "failure": None,
        "summary": case,
    }
    objective = task_body["objective"]["field"]
    raw[objective] = -1.0 if objective == "Cl" else 600.0
    for constraint in task_body.get("constraints", []):
        raw[constraint["field"]] = constraint["threshold"]
    return {
        "backend_responses": [design] * 64,
        "tool_responses": [raw] * 128,
    }


def inject_interface_line_hooks(
    line_cfg: Any, backend: ScriptedBackend, environment: ScriptedEnvironment
):
    """Inject the environment protocol only into a compat interface line."""

    config = copy.copy(line_cfg)
    assert config.get("task_spec") is not None
    config["backend"] = backend
    config["invoke_fn"] = environment.invoke
    return config


def inject_legacy_line_hooks(
    line_cfg: Any,
    backend: ScriptedBackend,
    hooks: ScriptedLegacyHooks,
    monkeypatch,
    driver,
):
    """Inject legacy hooks at both the config and Interpreter boundaries."""

    config = copy.copy(line_cfg)
    assert config.get("task_spec") is None
    config["backend"] = backend
    config["run_case_fn"] = hooks.run_case
    config["verify_fn"] = hooks.verify
    monkeypatch.setattr(
        driver,
        "Interpreter",
        lambda working_dir, *args, **kwargs: ScriptedLegacyInterpreter(
            hooks, driver.ExecutionResult, driver._params_from_code, working_dir
        ),
    )
    assert "invoke_fn" not in config
    return config


def default_design(task: Any):
    body = plain(task).get("task", plain(task))
    params = {
        key: spec.get("default", (spec["bounds"][0] + spec["bounds"][1]) / 2)
        for key, spec in body["params"].items()
    }
    options = {key: spec.get("default") for key, spec in body.get("options", {}).items()}
    return params, options


def direct_block_run(environment: str, params: dict, options: dict, store: Path):
    block = importlib.import_module(
        "pipeline.exp_layer.aero2d"
    )
    param_fields = {field.name for field in dataclasses.fields(block.Params)}
    option_fields = {field.name for field in dataclasses.fields(block.RunOptions)}
    param_values = {key: item for key, item in params.items() if key in param_fields}
    param_values.update(
        {key: item for key, item in options.items() if key in param_fields}
    )
    option_values = {key: item for key, item in options.items() if key in option_fields}
    result = block.run_case(
        block.Params(**param_values),
        block.RunOptions(**option_values),
        store=store,
        cache=False,
    )
    return json.loads(result.to_json(deterministic=True))
