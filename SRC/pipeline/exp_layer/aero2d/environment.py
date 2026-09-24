"""Common environment-interface adapter for the existing wing solver block."""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, fields
from pathlib import Path

import yaml

from pipeline.exp_layer import common

from . import Params, RunOptions, run_case, solver_build


def _summary(raw: dict, metric) -> str:
    if raw.get("status") == "ok":
        forces = raw["forces"]
        false = [key for key, value in raw.get("trust", {}).items() if value is not True]
        text = "status=ok Cl=%.4f Cd=%.4f trusted=%s" % (
            forces["Cl"], forces["Cd"], not false
        )
        if false:
            text += " untrusted_flags=%s" % false
        if raw.get("field_summary"):
            text += " " + " ".join(
                "%s=%.4g" % item for item in sorted(raw["field_summary"].items())
            )
    else:
        failure = raw.get("failure") or {}
        text = "status=failed type=%s stage=%s evidence=%s" % (
            failure.get("type"), failure.get("stage"), (failure.get("evidence") or "").strip()[:200]
        )
    return text + " objective=%s" % ("None" if metric is None else "%.4f" % metric)


class Aero2dExperiment(common.Experiment):
    def invoke(self, tool: str, inputs: dict, context: dict) -> dict:
        if context.get("workflow", {}).get("compat_walk"):
            if tool != "evaluate":
                return {"failure": {"kind": "unsupported_tool", "tool": tool}}
            from .evaluation.walk import main as walk_main

            config = context["environment"].get("settings", {}).get("compat_walk_config")
            if not isinstance(config, dict):
                return {"failure": {"kind": "configuration_error", "evidence": "missing frozen walk config"}}
            with tempfile.TemporaryDirectory(prefix="simblock-walk-") as directory:
                snapshot = Path(directory) / "resolved-walk.yaml"
                snapshot.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
                exit_code = walk_main([str(snapshot)])
            return {
                "case_id": f"walk:{context['candidate_id']}",
                "input_hash": context["design_hash"],
                "status": "ok" if exit_code == 0 else "failed",
                "failure": None if exit_code == 0 else {"kind": "walk_failed", "exit_code": exit_code},
                "summary": "existing simblock walk completed through the common tool host",
                "provenance": common.provenance(context),
            }
        if tool not in {"simulate", "evaluate"}:
            return {"failure": {"kind": "unsupported_tool", "tool": tool}}
        params = Params(**{key: float(value) for key, value in inputs["params"].items()})
        option_names = {field.name for field in fields(RunOptions)}
        supplied_options = dict(inputs.get("options", {}))
        for name in context["tool"].get("option_overrides", []):
            if name in inputs.get("args", {}):
                supplied_options[name] = inputs["args"][name]
            elif name not in supplied_options and "default" in context["tool"].get("args", {}).get(name, {}):
                supplied_options[name] = context["tool"]["args"][name]["default"]
        options = RunOptions(**{key: value for key, value in supplied_options.items() if key in option_names})
        settings = context["environment"].get("settings", {})
        store = Path(context["repo_root"]) / settings.get("store", "OUTPUTs/batch1/aero2d/store")
        timeout = float(settings.get("timeout_s", 900))
        result = run_case(
            params,
            options,
            scalar_only=bool(context.get("workflow", {}).get("scalar_only", False)),
            store=store,
            timeout_s=timeout,
        )
        physical = json.loads(result.to_json())
        forces = physical.get("forces") or {}
        cd_max = next(
            (float(item["threshold"]) for item in context["task"].get("constraints", []) if item.get("field") == "Cd"),
            0.06,
        )
        trusted = physical.get("status") == "ok" and all(physical.get("trust", {}).values())
        metric = None if not trusted else -float(forces["Cl"]) - 10.0 * max(0.0, float(forces["Cd"]) - cd_max)
        raw = {
            **physical,
            "Cl": forces.get("Cl"),
            "Cd": forces.get("Cd"),
            "conditions_completed": ["design"] if physical.get("status") == "ok" else [],
            "fidelity": 0,
            "summary": _summary(physical, metric),
            "metadata": {
                "physical_provenance": physical.get("provenance", {}),
                "runtime_s": physical.get("runtime_s"),
            },
            "versions": {
                "config_hash": context["config_hash"],
                "source_version": context["environment"]["version"],
                "data_version": context["environment"]["data"]["version"],
                "solver_build": solver_build(),
            },
            "effective_options": asdict(options),
            "provenance": common.provenance(context),
            "artifacts": [
                {"name": name, "artifact_id": artifact_id, "case_id": physical.get("case_id")}
                for name, artifact_id in context["output_artifact_ids"].items()
            ],
        }
        return raw


invoke = Aero2dExperiment().invoke


__all__ = ["invoke"]
