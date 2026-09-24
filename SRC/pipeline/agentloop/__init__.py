"""AIDE-based design–simulate loop with a configurable environment boundary."""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "DesignAgent": (".design_agent", "DesignAgent"),
    "Tree": (".driver", "Tree"),
    "run_line": (".driver", "run_line"),
    "summarize_sensitivity": (".driver", "summarize_sensitivity"),
    "TaskSpec": (".environment", "TaskSpec"),
    "normalize_evaluation": (".environment", "normalize_evaluation"),
    "parse_design": (".environment", "parse_design"),
    "resolve_task": (".environment", "resolve_task"),
    "validate_tool_call": (".environment", "validate_tool_call"),
    "LINES": (".harness", "LINES"),
    "PLAN_KEYS": (".harness", "PLAN_KEYS"),
    "aggregate_lines": (".harness", "aggregate_lines"),
    "make_exec_code": (".harness", "make_exec_code"),
    "nelder_mead_propose": (".harness", "nelder_mead_propose"),
    "objective": (".harness", "objective"),
    "parse_plan": (".harness", "parse_plan"),
    "parse_term_out": (".harness", "parse_term_out"),
    "perturb_x0": (".harness", "perturb_x0"),
    "promote_tier": (".harness", "promote_tier"),
    "select_parent": (".harness", "select_parent"),
    "should_stop": (".harness", "should_stop"),
    "WING_SCENARIO": (".scenario", "WING_SCENARIO"),
    "Scenario": (".scenario", "Scenario"),
    "ToolRuntime": (".workflow", "ToolRuntime"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
