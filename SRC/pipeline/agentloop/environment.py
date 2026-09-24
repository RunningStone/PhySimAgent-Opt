"""Configuration and protocol boundary for configurable simulation environments.

The module deliberately keeps the public protocol as ordinary dictionaries.  A
``TaskSpec`` is the one frozen configuration object shared by the outer search,
the tool runtime, and an environment adapter.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ENTRYPOINT = re.compile(
    r"^(?P<module>[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*):"
    r"(?P<function>[A-Za-z_][A-Za-z0-9_]*)$"
)
_GENERIC_TOP_LEVEL = {"environment", "task", "tools", "workflow", "limits", "experiment"}
_GENERIC_DRIVER_KEYS = {
    "line", "exp_name", "output_dir", "output_root", "budget", "llm_model", "tau", "k_backtrack",
    "k_stagnation", "verify_every", "seed", "seeds", "parallel", "timeout_s",
    "sensitivity", "use_memory", "run_id",
}


class InterfaceError(ValueError):
    kind = "interface_error"


class ConfigurationError(InterfaceError):
    kind = "configuration_error"


class ProtocolError(InterfaceError):
    kind = "protocol_error"


class DependencyError(InterfaceError):
    kind = "dependency_error"


class BudgetError(InterfaceError):
    kind = "budget_error"


@dataclass(frozen=True)
class TaskSpec:
    environment: dict[str, Any]
    task: dict[str, Any]
    tools: dict[str, Any]
    workflow: dict[str, Any]
    limits: dict[str, Any]
    problem_hash: str
    config_hash: str
    repo_root: Path
    experiment: dict[str, Any] | None = None


def _sha(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return "sha256:" + hashlib.sha256(body.encode("utf-8")).hexdigest()


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _legacy_definition(config: dict[str, Any]) -> dict[str, Any]:
    """Translate the first-batch wing configs to the common task contract.

    This is a deterministic projection of the existing scenario constants; it
    does not select a different solver or alter the old search configuration.
    """

    scenario = config.get("scenario", "wing")
    unlimited = {
        "max_tools_per_candidate": 1,
        "total_tools": None,
        "solve_slots": None,
        "total_llm": None,
        "wall_clock_s": None,
        "validation_reserve": 0,
    }
    if scenario == "wing":
        cd_max = float(config.get("cd_max", 0.06))
        return {
            "environment": {
                "name": "simblock-wing-v1",
                "entrypoint": "pipeline.exp_layer.aero2d.environment:invoke",
                "version": "simblock-v1",
                "data": {
                    "dataset": "zerihan2000",
                    "version": "sha256:8a1b-first-batch-wing-data",
                },
                "settings": {
                    "store": config.get("store", "OUTPUTs/batch1/aero2d/store"),
                    "timeout_s": config.get("timeout_s", 900),
                },
            },
            "task": {
                "name": "wing",
                "prompt": "legacy:WING_TASK_DESC",
                "params": {
                    "h_c": {"type": "float", "bounds": [0.05, 1.0], "default": 0.3},
                    "alpha_deg": {"type": "float", "bounds": [-2.0, 12.0], "default": 2.0},
                    "camber": {"type": "float", "bounds": [0.0, 0.09], "default": 0.04},
                },
                "options": {
                    "gap_refine": {"type": "integer", "allowed": [0, 1, 2, 3], "default": 0},
                    "warm_start_from": {"type": "string", "nullable": True, "default": None},
                    "relax_step": {"type": "integer", "allowed": [0, 1, 2], "default": 0},
                    "iter_mult": {"type": "integer", "allowed": [1, 2, 3, 4], "default": 1},
                },
                "objective": {"field": "Cl", "direction": "minimize", "scale": 1.0},
                "constraints": [{
                    "field": "Cd", "kind": "penalty", "relation": "max",
                    "threshold": cd_max, "weight": 10.0,
                }],
                "trust": {"required": ["converged", "residual_ok", "force_settled", "yplus_ok", "mesh_ok"]},
                "evaluation": {
                    "tool": "evaluate", "required_conditions": ["design"], "allowed_fidelity": [0],
                },
                "private_validation": {"kind": "legacy_holdout"},
            },
            "tools": {
                "evaluate": {
                    "description": "Run one legacy wing case and score it.",
                    "requires": [], "produces": ["raw_case"], "invalidates": [],
                }
            },
            "workflow": {
                "mode": "fixed",
                "fixed": ["evaluate"],
                "scalar_only": config.get("line") in {"A", "B"},
            },
            "limits": unlimited,
        }
    raise ConfigurationError(f"unknown legacy scenario {scenario!r}")


def _validate_name(value: Any, label: str) -> None:
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise ConfigurationError(f"{label} must be a finite identifier")


def _validate_entrypoint(value: Any) -> None:
    match = _ENTRYPOINT.fullmatch(value) if isinstance(value, str) else None
    if not match:
        raise ConfigurationError(f"invalid environment entrypoint {value!r}")
    module_name, function_name = match.group("module"), match.group("function")
    # ``test:...`` is the explicit injected-environment protocol used by the
    # black-box contract suite. Other entrypoints are accepted from trusted
    # resolved YAML only when Python can import the named callable.
    if module_name == "test":
        return
    try:
        function = getattr(importlib.import_module(module_name), function_name)
    except (ImportError, AttributeError) as exc:
        raise ConfigurationError(f"unavailable environment entrypoint {value!r}") from exc
    if not callable(function):
        raise ConfigurationError(f"environment entrypoint is not callable: {value!r}")


def _validate_definition(definition: dict[str, Any]) -> None:
    environment = definition.get("environment")
    task = definition.get("task")
    tools = definition.get("tools")
    workflow = definition.get("workflow")
    if not all(isinstance(v, dict) for v in (environment, task, tools, workflow)):
        raise ConfigurationError("environment, task, tools, and workflow must be mappings")
    _validate_entrypoint(environment.get("entrypoint"))
    params = task.get("params")
    if not isinstance(params, dict) or not params:
        raise ConfigurationError("task.params must be a non-empty mapping")
    for name, spec in params.items():
        _validate_name(name, "parameter name")
        if not isinstance(spec, dict):
            raise ConfigurationError(f"parameter {name!r} must be a mapping")
        if spec.get("type") == "float":
            bounds = spec.get("bounds")
            if not isinstance(bounds, list) or len(bounds) != 2 or not all(_finite_number(v) for v in bounds):
                raise ConfigurationError(f"float parameter {name!r} needs two finite bounds")
            if float(bounds[0]) > float(bounds[1]):
                raise ConfigurationError(f"parameter {name!r} has reversed bounds")
    if not tools:
        raise ConfigurationError("tools must be non-empty")
    for tool_name, tool in tools.items():
        _validate_name(tool_name, "tool name")
        if not isinstance(tool, dict):
            raise ConfigurationError(f"tool {tool_name!r} must be a mapping")
        for field in ("requires", "produces", "invalidates"):
            names = tool.get(field, [])
            if not isinstance(names, list):
                raise ConfigurationError(f"tools.{tool_name}.{field} must be a list")
            for name in names:
                _validate_name(name, f"tools.{tool_name}.{field} entry")
        declared_args = tool.get("args", {})
        if not isinstance(declared_args, dict):
            raise ConfigurationError(f"tools.{tool_name}.args must be a mapping")
        for name in declared_args:
            _validate_name(name, f"tools.{tool_name}.args name")
        overrides = tool.get("option_overrides", [])
        if not isinstance(overrides, list) or any(name not in task.get("options", {}) for name in overrides):
            raise ConfigurationError(f"tools.{tool_name}.option_overrides contains an undeclared option")
    mode = workflow.get("mode")
    if mode not in {"fixed", "agent"}:
        raise ConfigurationError("workflow.mode must be fixed or agent")
    fixed = workflow.get("fixed", [])
    if not isinstance(fixed, list) or any(name not in tools for name in fixed):
        raise ConfigurationError("workflow.fixed contains an unknown tool")
    terminal = task.get("evaluation", {}).get("tool")
    if terminal not in tools:
        raise ConfigurationError("task.evaluation.tool must name a declared tool")


def resolve_task(merged_config: dict[str, Any], repo_root: Path | str | None = None) -> TaskSpec:
    """Validate a resolved YAML mapping and freeze its common-interface view."""

    if not isinstance(merged_config, dict):
        raise ConfigurationError("resolved configuration must be a mapping")
    original = copy.deepcopy(merged_config)
    if "scenario" in original:
        definition = _legacy_definition(original)
        for key in ("environment", "task", "tools"):
            if isinstance(original.get(key), dict):
                definition[key] = copy.deepcopy(original[key])
        # An overlay may set only the decision mode/limits while retaining the
        # exact first-batch physical definition.
        if isinstance(original.get("workflow"), dict):
            definition["workflow"].update(copy.deepcopy(original["workflow"]))
        if isinstance(original.get("limits"), dict):
            definition["limits"] = copy.deepcopy(original["limits"])
    else:
        unknown = set(original) - _GENERIC_TOP_LEVEL - _GENERIC_DRIVER_KEYS
        if unknown:
            raise ConfigurationError(f"unknown top-level fields: {sorted(unknown)}")
        definition = {key: copy.deepcopy(original.get(key, {})) for key in _GENERIC_TOP_LEVEL}
        definition["limits"] = definition.get("limits") or {}
    _validate_definition(definition)
    experiment = definition.get("experiment") or None
    if experiment is not None:
        _validate_experiment(experiment, definition)
    physical = {key: definition[key] for key in ("environment", "task", "tools")}
    root = Path(repo_root or Path(__file__).resolve().parents[3]).resolve()
    return TaskSpec(
        environment=definition["environment"],
        task=definition["task"],
        tools=definition["tools"],
        workflow=definition["workflow"],
        limits=definition["limits"],
        problem_hash=_sha(physical),
        config_hash=_sha(definition),
        repo_root=root,
        experiment=experiment,
    )


def _validate_experiment(experiment: dict, definition: dict) -> None:
    allowed = {"schema_version", "level", "arm", "stage_id", "parent_stage_id", "initialization", "transfer", "replicate_id"}
    if not isinstance(experiment, dict) or set(experiment) - allowed:
        raise ConfigurationError("unknown experiment fields")
    if experiment.get("schema_version") != 1:
        raise ConfigurationError("experiment.schema_version must be 1")
    level, arm = experiment.get("level"), experiment.get("arm")
    if isinstance(level, bool) or (level, arm) not in {(0, "base"), (1, "a"), (1, "b"), (2, "a"), (2, "b"), (3, "topk")}:
        raise ConfigurationError("unknown level/arm")
    settings = definition["environment"].get("settings", {})
    if set(settings) - {"model", "flow", "domain", "mesh", "solver", "geometry", "exploration", "numerical_checks", "versions", "timeout_s", "llm_timeout_s"}:
        raise ConfigurationError("unknown physical profile field")
    for section, names in (("flow", ("U_inf", "chord", "Re")), ("mesh", ("far", "wing", "mesh_scale")),
                           ("solver", ("end_time", "delta_t"))):
        if not isinstance(settings.get(section), dict) or any(not _finite_number(settings[section].get(name)) or settings[section][name] <= 0 for name in names):
            raise ConfigurationError(f"{section} must declare positive finite physical/numerical values")
    if not isinstance(settings.get("domain"), dict) or any(not _finite_number(settings["domain"].get(k)) for k in ("x_min", "x_max", "y_min", "y_max")):
        raise ConfigurationError("far-field domain is not fully specified")
    if not isinstance(settings.get("geometry"), dict):
        raise ConfigurationError("physical geometry contract is required")
    stage = experiment.get("stage_id")
    if stage not in {"M0", "M1"} or settings.get("model") != stage:
        raise ConfigurationError("unknown or inconsistent physical stage")
    if level in {0, 2} and stage != "M0":
        raise ConfigurationError("this level is restricted to M0")
    if definition["workflow"].get("fixed") != ["geometry", "mesh", "solve", "post"]:
        raise ConfigurationError("interview workflow must execute geometry, mesh, solve, post")
    task = definition["task"]
    if task.get("geometry") != settings["geometry"]:
        raise ConfigurationError("task and physical geometry contracts differ; an explicit mapping is required")
    allowed_task = {"name", "prompt", "params", "options", "geometry", "objective", "constraints", "trust", "evaluation", "feedback_fields", "proposal_instructions", "recovery_menu", "required_conditions"}
    if set(task) - allowed_task:
        raise ConfigurationError("unknown task fields")
    allowed_limits = {"max_candidates", "max_tools_per_candidate", "total_tools", "solve_slots", "total_llm", "wall_clock_s", "validation_reserve", "retries", "tool_timeout_s", "llm_timeout_s"}
    if set(definition["limits"]) - allowed_limits:
        raise ConfigurationError("unknown budget fields")
    for key, value in definition["limits"].items():
        if value is not None and (not _finite_number(value) or value < 0):
            raise ConfigurationError(f"invalid budget {key}")
    objective = task.get("objective", {})
    if not objective.get("field") or objective.get("direction") not in {"target", "maximize", "minimize"}:
        raise ConfigurationError("a valid objective must be frozen")
    if objective["direction"] == "target" and not _finite_number(objective.get("target")):
        raise ConfigurationError("target objective needs a finite target")
    if not isinstance(task.get("constraints", []), list):
        raise ConfigurationError("constraints must be a list")
    if level in {2, 3} and not isinstance(task.get("geometry"), dict):
        raise ConfigurationError("shape experiments require a frozen geometry domain")
    if (level, arm) == (2, "b") and not settings.get("exploration"):
        raise ConfigurationError("Level2-b requires a nonempty implemented exploration whitelist")
    if set(settings.get("exploration", {})) - {"mesh_scale"}:
        raise ConfigurationError("unsupported exploration control")
    if "private_validation" in task:
        raise ConfigurationError("new experiments cannot use legacy private validation")


def canonical_design(candidate: dict, geometry_contract: dict) -> dict:
    """Normalize physical design data; solver settings never enter its hash."""
    allowed = {"representation", "parameters", "shape", "candidate_id", "parent_candidate_ids", "design_hash", "geometry_validation"}
    if not isinstance(candidate, dict) or set(candidate) - allowed:
        raise ProtocolError("candidate contains unknown fields or solver settings")
    result = copy.deepcopy(candidate)
    params = result.get("parameters")
    if not isinstance(params, dict) or not params or any(not _finite_number(v) for v in params.values()):
        raise ProtocolError("candidate parameters must be finite literal numbers")
    result["parameters"] = {k: float(v) if float(v) else 0.0 for k, v in params.items()}
    representation = result.get("representation")
    if representation == "explicit_shape":
        from pipeline.exp_layer.aero2d.geometry import normalize_shape, validate_shape
        try:
            result["shape"] = normalize_shape(result.get("shape"), {})
            result["shape"] = validate_shape(result["shape"], geometry_contract)
            result["shape"] = {key: [0.0 if value == 0 else value for value in values]
                               for key, values in result["shape"].items()}
        except (ValueError, TypeError, KeyError) as exc:
            raise ProtocolError(str(exc)) from exc
    elif representation != "parameterized" or "shape" in result:
        raise ProtocolError("unsupported candidate representation")
    parents = result.get("parent_candidate_ids", [])
    if not isinstance(parents, list) or any(not isinstance(p, str) or not p for p in parents):
        raise ProtocolError("parent_candidate_ids must be literal identifiers")
    if "candidate_id" in result and (not isinstance(result["candidate_id"], str) or not result["candidate_id"]):
        raise ProtocolError("candidate_id must be an identifier")
    physical = {key: result[key] for key in ("representation", "parameters", "shape") if key in result}
    result["design_hash"] = _sha(physical)
    result["parent_candidate_ids"] = parents
    return result


def validate_proposal(proposal: dict, task: TaskSpec, allowed_evidence=None) -> dict:
    if task.experiment is None:
        raise ProtocolError("JSON proposals require experiment.schema_version=1")
    allowed = {"action", "candidate", "question", "evidence_refs", "requested_numerics", "requested_conditions", "expected_cost", "stop_reason"}
    if not isinstance(proposal, dict) or set(proposal) - allowed:
        raise ProtocolError("proposal contains unknown or host-owned fields")
    result = copy.deepcopy(proposal)
    exploratory = (task.experiment["level"], task.experiment["arm"]) == (2, "b")
    actions = {"explore", "submit_for_evaluation", "stop"} if exploratory else {"evaluate_design", "stop"}
    action = result.get("action")
    if action not in actions:
        raise ProtocolError("action is not permitted for this experiment arm")
    refs = result.get("evidence_refs", [])
    if not isinstance(refs, list) or any(not isinstance(ref, str) for ref in refs) or not set(refs) <= set(allowed_evidence or []):
        raise ProtocolError("proposal references unknown or inaccessible evidence")
    result["evidence_refs"] = refs
    if action == "stop":
        if not isinstance(result.get("stop_reason"), str) or not result["stop_reason"].strip():
            raise ProtocolError("stop needs a reason")
        if result.get("requested_numerics") or result.get("requested_conditions"):
            raise ProtocolError("stop cannot request execution")
        return result
    candidate = canonical_design(result.get("candidate"), task.task.get("geometry", {}))
    expected_representation = "explicit_shape" if task.experiment["level"] >= 2 else "parameterized"
    if candidate["representation"] != expected_representation:
        raise ProtocolError("candidate representation is not permitted in this level")
    if set(candidate["parameters"]) != set(task.task["params"]):
        raise ProtocolError("candidate parameters do not match the task")
    for key, value in candidate["parameters"].items():
        _validate_value(key, value, task.task["params"][key])
    numerics = result.get("requested_numerics", {})
    conditions = result.get("requested_conditions", [])
    if not isinstance(numerics, dict) or not isinstance(conditions, list):
        raise ProtocolError("requested numerics/conditions have invalid types")
    if action != "explore" and (numerics or conditions):
        raise ProtocolError("formal numerical settings and conditions are host-owned")
    whitelist = task.environment.get("settings", {}).get("exploration", {})
    for key, value in numerics.items():
        if key not in whitelist or key != "mesh_scale":
            raise ProtocolError("unsupported or unauthorized numerical control")
        bound = whitelist[key]
        spec = bound if isinstance(bound, dict) else {"type": "float", "bounds": bound}
        _validate_value(key, value, spec)
        if not _finite_number(value):
            raise ProtocolError("numerical controls must be finite")
    if conditions and not set(conditions) <= set(task.task["evaluation"].get("required_conditions", ["design"])):
        raise ProtocolError("unknown or unauthorized exploration condition")
    if "expected_cost" in result:
        estimate = result["expected_cost"]
        if not (isinstance(estimate, dict) and all(_finite_number(v) and v >= 0 for v in estimate.values())) and not (_finite_number(estimate) and estimate >= 0):
            raise ProtocolError("expected_cost must be a nonnegative estimate")
    result.update(candidate=candidate, requested_numerics=numerics, requested_conditions=conditions)
    return result


def formal_protocol(task: TaskSpec) -> dict:
    """Derive one immutable evaluation identity from physical/task/source truth."""
    evaluation = task.task.get("evaluation", {})
    implementation = {}
    for path in (Path(__file__), Path(__file__).with_name("workflow.py"), Path(__file__).parent / "evaluation/assessment.py",
                 Path(__file__).parents[1] / "exp_layer/aero2d/interview.py", Path(__file__).parents[1] / "exp_layer/aero2d/geometry.py"):
        if path.exists():
            implementation[str(path.relative_to(Path(__file__).parents[1]))] = hashlib.sha256(path.read_bytes()).hexdigest()
    definition = {"environment": task.environment, "task": task.task, "tools": task.tools, "workflow": task.workflow,
                  "implementation": implementation}
    return {"stage_id": task.experiment["stage_id"], "protocol_hash": _sha(definition),
            "objective": copy.deepcopy(task.task["objective"]), "constraints": copy.deepcopy(task.task.get("constraints", [])),
            "required_conditions": copy.deepcopy(evaluation.get("required_conditions", ["design"])),
            "numerical_checks": copy.deepcopy(task.task.get("trust", {}).get("required", [])),
            "allowed_sources": copy.deepcopy(evaluation.get("allowed_sources", ["live"])),
            "replicate_id": task.experiment.get("replicate_id", 0),
            "implementation": implementation, "ranking_tolerance": float(evaluation.get("ranking_tolerance", 1e-12))}


def _validate_value(name: str, value: Any, spec: dict[str, Any]) -> None:
    typ = spec.get("type")
    if value is None and spec.get("nullable"):
        return
    if typ == "float":
        if not _finite_number(value):
            raise ProtocolError(f"{name} must be a finite number")
    elif typ in {"integer", "int"}:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ProtocolError(f"{name} must be an integer")
    elif typ == "string":
        if not isinstance(value, str):
            raise ProtocolError(f"{name} must be a string")
    if "allowed" in spec and value not in spec["allowed"]:
        raise ProtocolError(f"{name} is outside its allowed values")
    if "bounds" in spec and _finite_number(value):
        low, high = spec["bounds"]
        if float(value) < float(low) or float(value) > float(high):
            raise ProtocolError(f"{name} is outside its bounds")


def parse_design(code: str, task: TaskSpec) -> tuple[dict[str, Any], dict[str, Any]]:
    """Parse exactly one literal PARAMS assignment and an optional OPTIONS assignment."""

    try:
        body = ast.parse(code, mode="exec").body
    except SyntaxError as exc:
        raise ProtocolError(str(exc)) from exc
    values: dict[str, Any] = {}
    for statement in body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
            raise ProtocolError("design contains a non-assignment statement")
        name = statement.targets[0].id
        if name not in {"PARAMS", "OPTIONS"} or name in values:
            raise ProtocolError("design must assign PARAMS once and OPTIONS at most once")
        try:
            value = ast.literal_eval(statement.value)
        except (ValueError, TypeError, SyntaxError) as exc:
            raise ProtocolError(f"{name} must be a literal mapping") from exc
        if not isinstance(value, dict):
            raise ProtocolError(f"{name} must be a mapping")
        values[name] = value
    if "PARAMS" not in values:
        raise ProtocolError("PARAMS is required")
    params, options = values["PARAMS"], values.get("OPTIONS") or {}
    expected_params = task.task["params"]
    if set(params) != set(expected_params):
        raise ProtocolError("PARAMS keys do not match the task")
    if not set(options) <= set(task.task.get("options", {})):
        raise ProtocolError("OPTIONS contains an unknown key")
    for name, value in params.items():
        spec = expected_params[name]
        if task.task.get("prompt", "").startswith("legacy:"):
            if spec.get("type") in {"float", "integer", "int"} and not _finite_number(value):
                raise ProtocolError(f"{name} must be a finite number")
            if spec.get("type") == "string" and not isinstance(value, str):
                raise ProtocolError(f"{name} must be a string")
        else:
            _validate_value(name, value, spec)
    for name, value in options.items():
        _validate_value(name, value, task.task["options"][name])
    return copy.deepcopy(params), copy.deepcopy(options)


def validate_tool_call(call: dict[str, Any], task: TaskSpec, state: dict[str, Any]) -> dict[str, Any]:
    """Validate a requested tool and bind its current-candidate dependencies."""

    if not isinstance(call, dict):
        raise ProtocolError("tool call must be a mapping")
    unknown_fields = set(call) - {"name", "tool", "args"}
    if unknown_fields:
        raise ProtocolError(
            f"unknown tool-call fields: {sorted(unknown_fields)}"
        )
    if ("name" in call) == ("tool" in call):
        raise ProtocolError("tool call must contain exactly one of name or tool")
    name = call.get("name", call.get("tool"))
    if name not in task.tools:
        raise ProtocolError(f"unknown tool {name!r}")
    args = call.get("args", {})
    if not isinstance(args, dict):
        raise ProtocolError("tool args must be a mapping")
    tool = task.tools[name]
    allowed = (
        set(tool.get("args", {}))
        | set(tool.get("option_overrides", []))
        | {"params", "options", "warm_start_from"}
        | set(tool.get("requires", []))
    )
    if not set(args) <= allowed:
        raise ProtocolError(f"unknown args for {name}: {sorted(set(args) - allowed)}")
    if "params" in args:
        params = args["params"]
        if not isinstance(params, dict) or (
            "params" in state and params != state["params"]
        ):
            raise ProtocolError("tool PARAMS must equal the immutable candidate PARAMS")
    if "options" in args:
        options = args["options"]
        if not isinstance(options, dict) or (
            "options" in state and options != state["options"]
        ):
            raise ProtocolError("tool OPTIONS must equal the immutable candidate OPTIONS")
    for arg_name, arg_value in args.items():
        if arg_name in tool.get("args", {}):
            _validate_value(arg_name, arg_value, tool["args"][arg_name])
            if (
                arg_name in task.task.get("options", {})
                and arg_name not in tool.get("option_overrides", [])
            ):
                raise ProtocolError(
                    f"ambiguous option {arg_name!r} is not listed in option_overrides"
                )
        elif arg_name in tool.get("option_overrides", []):
            _validate_value(arg_name, arg_value, task.task["options"][arg_name])
    artifacts = state.get("artifacts", {})
    for required in tool.get("requires", []):
        if required not in artifacts:
            raise DependencyError(f"missing required artifact {required!r}")
        handle = args.get(required, artifacts[required])
        if not isinstance(handle, dict):
            raise DependencyError(f"artifact {required!r} must be a handle")
        if handle.get("candidate_id") != state.get("candidate_id") or handle.get("problem_hash") != state.get("problem_hash"):
            raise DependencyError(f"artifact {required!r} belongs to another candidate or problem")
        if handle.get("active", True) is not True:
            raise DependencyError(f"artifact {required!r} is inactive")
    warm = args.get("warm_start_from")
    if warm is not None:
        source = state.get("validated_warm_starts", {}).get(warm)
        if not source or source.get("validated") is not True or source.get("problem_hash") != task.problem_hash:
            raise DependencyError("warm start is not validated for this problem")
    return {"name": name, "args": copy.deepcopy(args)}


def _required_conditions(task: TaskSpec, validation: bool = False) -> tuple[dict[str, list[Any]], list[Any]]:
    evaluation = task.task.get("evaluation", {})
    contract = task.task.get("private_validation", {}) if validation else evaluation
    raw = contract.get(
        "required_conditions",
        task.task.get("required_conditions", evaluation.get("required_conditions", [])),
    )
    default_fidelity = list(contract.get("allowed_fidelity", evaluation.get("allowed_fidelity", [])))
    if isinstance(raw, dict):
        return {
            name: list((spec or {}).get("allowed_fidelities", default_fidelity))
            for name, spec in raw.items()
        }, default_fidelity
    return {str(name): default_fidelity for name in raw}, default_fidelity


def normalize_evaluation(raw: dict[str, Any], task: TaskSpec, *, validation: bool = False) -> dict[str, Any]:
    """Validate public metrics and compute the configured maximise-form score."""

    result = copy.deepcopy(raw) if isinstance(raw, dict) else {}
    constraints_out: list[dict[str, Any]] = []
    accepted = result.get("failure") is None
    required, default_fidelity = _required_conditions(task, validation=validation)
    conditions = result.get("conditions")
    if isinstance(conditions, dict):
        accepted = accepted and set(conditions) == set(required)
        for name, allowed in required.items():
            item = conditions.get(name, {})
            accepted = accepted and item.get("status") == "ok"
            accepted = accepted and (not allowed or item.get("fidelity") in allowed)
            accepted = accepted and bool(item.get("artifact_ids"))
    else:
        completed = result.get("conditions_completed", [])
        accepted = accepted and set(completed) == set(required)
        accepted = accepted and (not default_fidelity or result.get("fidelity") in default_fidelity)
    trust = result.get("trust")
    accepted = accepted and isinstance(trust, dict)
    accepted = accepted and all(trust.get(name) is True for name in task.task.get("trust", {}).get("required", []))
    objective = task.task.get("objective", {})
    fields = [objective.get("field")] + [c.get("field") for c in task.task.get("constraints", [])]
    accepted = accepted and all(field in result and _finite_number(result.get(field)) for field in fields)
    metric: float | None = None
    if accepted:
        value = float(result[objective["field"]]) * float(objective.get("scale", 1.0))
        metric = value if objective.get("direction") == "maximize" else -value
        for constraint in task.task.get("constraints", []):
            actual = float(result[constraint["field"]])
            threshold = float(constraint["threshold"])
            violation = max(0.0, actual - threshold) if constraint.get("relation") == "max" else max(0.0, threshold - actual)
            constraints_out.append({**constraint, "value": actual, "violation": violation})
            if violation and constraint.get("kind") == "hard":
                accepted, metric = False, None
                break
            if violation and constraint.get("kind") == "penalty":
                metric -= float(constraint.get("weight", 0.0)) * violation
        if metric is not None and not math.isfinite(metric):
            accepted, metric = False, None
    return {
        "accepted": bool(accepted),
        "metric": metric if accepted else None,
        "raw": result,
        "constraints": constraints_out,
        "failure": result.get("failure"),
        "summary": result.get("summary", ""),
    }


__all__ = [
    "TaskSpec", "ConfigurationError", "ProtocolError", "DependencyError", "BudgetError",
    "resolve_task", "parse_design", "validate_tool_call", "normalize_evaluation",
]
