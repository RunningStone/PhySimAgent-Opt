"""First-batch compatibility gate for all 9 wing matrix configurations."""

from __future__ import annotations

import copy
import importlib

import pytest

from test.exp.agentloop_fixtures import (
    OLD_RUN_CONFIGS,
    ScriptedBackend,
    ScriptedEnvironment,
    ScriptedLegacyHooks,
    compatibility_script,
    inject_interface_line_hooks,
    inject_legacy_line_hooks,
    semantic_config_projection,
    semantic_run_projection,
    value,
)


def _resolved_pair(contract, old_path, overlays):
    overlay = overlays[old_path.resolve()]
    old_cfg = contract.line_cfg_from_yaml(old_path)
    new_cfg = contract.line_cfg_from_yaml(overlay)
    new_task = contract.resolve_yaml(overlay)
    return overlay, old_cfg, new_cfg, None, new_task


def _walk_routes_through_driver_main(
    monkeypatch, interface_contract, old_path, overlay, version
):
    routed = []
    walked = []

    def capture_run_line(line_cfg):
        routed.append(line_cfg)
        return {"line": line_cfg["line"]}

    walk_module = importlib.import_module(
        "pipeline.exp_layer.aero2d.evaluation.walk"
    )
    monkeypatch.setattr(walk_module, "main", lambda argv=None: walked.append(argv) or 0)
    monkeypatch.setattr(interface_contract.driver, "run_line", capture_run_line)
    interface_contract.driver.main([str(old_path)])
    interface_contract.driver.main([str(overlay)])
    assert len(walked) == 1
    assert len(routed) == 1
    return interface_contract.line_cfg_from_yaml(old_path), routed[0]


@pytest.mark.parametrize("old_path", OLD_RUN_CONFIGS, ids=lambda p: f"{p.parts[-3]}-{p.stem}")
def test_all_9_old_configs_have_one_overlay_and_equal_resolved_semantics(
    compat_overlays, interface_contract, old_path
):
    assert old_path.resolve() in compat_overlays, f"missing compat_source overlay for {old_path}"
    _, old_cfg, new_cfg, _, new_task = _resolved_pair(
        interface_contract, old_path, compat_overlays
    )
    assert old_cfg.get("task_spec") is None
    assert callable(old_cfg.get("run_case_fn"))
    assert callable(old_cfg.get("verify_fn"))
    assert new_cfg.get("task_spec") is not None
    assert semantic_config_projection(old_cfg, new_task) == semantic_config_projection(
        new_cfg, new_task
    )

# The inherited wing configuration family is v1.
VERSION_DIRS = {"v1": "aero2d"}
COMPARISON_DIRS = {"wing": "aero2d"}


def test_compatibility_mapping_is_exactly_9_and_bijective(compat_overlays):
    expected = {path.resolve() for path in OLD_RUN_CONFIGS}
    assert set(compat_overlays) == expected
    assert len(set(compat_overlays.values())) == 9


@pytest.mark.parametrize("environment", ["wing"])
def test_repository_comparison_pair_differs_only_by_mode_and_run_identity(
    interface_contract, environment
):
    fixed = interface_contract.driver.load_config(
        f"CONFIGs/batch1/{COMPARISON_DIRS[environment]}/comparison/{environment}_fixed.yaml"
    )
    agent = interface_contract.driver.load_config(
        f"CONFIGs/batch1/{COMPARISON_DIRS[environment]}/comparison/{environment}_agent.yaml"
    )
    fixed_task = interface_contract.resolve_task(fixed)
    agent_task = interface_contract.resolve_task(agent)
    assert fixed_task.problem_hash == agent_task.problem_hash

    def fairness_projection(config):
        projected = copy.deepcopy(config)
        for key in ("exp_name", "run_id", "output_dir", "exp_dir", "run_tag"):
            projected.pop(key, None)
        projected["workflow"] = copy.deepcopy(projected["workflow"])
        projected["workflow"].pop("mode", None)
        return projected

    assert fairness_projection(fixed) == fairness_projection(agent)


@pytest.mark.parametrize("version", ["v1"])
def test_path_changes_preserve_environment_input_content_hashes(
    compat_overlays, interface_contract, version
):
    old_path = next(path for path in OLD_RUN_CONFIGS if VERSION_DIRS[version] in path.parts)
    _, old_cfg, new_cfg, _, new_task = _resolved_pair(
        interface_contract, old_path, compat_overlays
    )
    old = semantic_config_projection(old_cfg, new_task)
    new = semantic_config_projection(new_cfg, new_task)
    assert old["environment_data"] == new["environment_data"]
    assert old["problem_hash"] == new["problem_hash"]


@pytest.mark.parametrize("version", ["v1"])
def test_original_walk_uses_legacy_hooks_and_compat_walk_binds_toolruntime(
    compat_overlays, interface_contract, version, monkeypatch
):
    old_path = next(
        path
        for path in OLD_RUN_CONFIGS
        if VERSION_DIRS[version] in path.parts and path.name == "walk.yaml"
    )
    overlay = compat_overlays[old_path.resolve()]
    old_cfg, new_cfg = _walk_routes_through_driver_main(
        monkeypatch, interface_contract, old_path, overlay, version
    )
    assert old_cfg["line"] == new_cfg["line"] == "walk"
    assert old_cfg.get("task_spec") is None
    assert callable(old_cfg.get("run_case_fn"))
    assert callable(old_cfg.get("verify_fn"))
    assert new_cfg.get("task_spec") is not None
    for key in ("tier2_check", "failure_repro", "verify_every", "verify_cases"):
        assert old_cfg.get(key) == new_cfg.get(key)
    assert tuple(old_cfg["scenario"].param_keys) == tuple(new_cfg["scenario"].param_keys)


@pytest.mark.parametrize("version", ["v1"])
def test_use_memory_false_removes_only_sensitivity_memory(
    compat_overlays, interface_contract, version, tmp_path
):
    with_path = next(
        path
        for path in OLD_RUN_CONFIGS
        if VERSION_DIRS[version] in path.parts and path.name == "seeds_lineB.yaml"
    )
    without_path = next(
        path
        for path in OLD_RUN_CONFIGS
        if VERSION_DIRS[version] in path.parts and path.name == "seeds_lineB_nomem.yaml"
    )
    _, _, with_cfg, _, with_task = _resolved_pair(
        interface_contract, with_path, compat_overlays
    )
    _, _, without_cfg, _, without_task = _resolved_pair(
        interface_contract, without_path, compat_overlays
    )
    # The source project had a local sweep output. A clean checkout must
    # exercise the same memory contract without requiring historical results.
    sensitivity = tmp_path / "SENSITIVITY.md"
    sensitivity.write_text("Synthetic test prior: inspect convergence before scoring.\n")
    configs = []
    for path in (with_path, without_path):
        raw = interface_contract.driver.load_config(compat_overlays[path.resolve()])
        raw["sensitivity"] = str(sensitivity)
        configs.append(interface_contract.driver.line_cfg_from_yaml(raw))
    with_cfg, without_cfg = configs
    assert with_cfg["memory"] == sensitivity.read_text()
    assert without_cfg["memory"] is None
    with_memory = semantic_config_projection(with_cfg, with_task)["memory"]
    without_memory = semantic_config_projection(without_cfg, without_task)["memory"]
    assert with_memory["sensitivity_hash"] is not None
    assert without_memory["sensitivity_hash"] is None
    assert without_memory["journal_enabled"] is True


@pytest.mark.parametrize("version", ["v1"])
def test_missing_memory_file_is_semantically_empty_not_fatal(
    compat_overlays, interface_contract, version, tmp_path
):
    old_path = next(
        path
        for path in OLD_RUN_CONFIGS
        if VERSION_DIRS[version] in path.parts and path.name == "seeds_lineB.yaml"
    )
    _, _, new_cfg, _, new_task = _resolved_pair(
        interface_contract, old_path, compat_overlays
    )
    changed = copy.deepcopy(new_cfg)
    changed["sensitivity"] = str(tmp_path / "missing-sensitivity.md")
    memory = semantic_config_projection(changed, new_task)["memory"]
    assert memory["sensitivity_hash"] is None
    assert memory["journal_enabled"] is True


@pytest.mark.parametrize("version", ["v1"])
def test_seed_matrix_preserves_seeds_parallel_budget_verify_and_timeout(
    compat_overlays, interface_contract, version
):
    old_path = next(
        path
        for path in OLD_RUN_CONFIGS
        if VERSION_DIRS[version] in path.parts and path.name == "seeds_lineA.yaml"
    )
    _, _, new_cfg, _, new_task = _resolved_pair(
        interface_contract, old_path, compat_overlays
    )
    snapshot = semantic_config_projection(new_cfg, new_task)
    assert snapshot["seeds"] == [0, 1, 2]
    assert snapshot["parallel"] == 3
    assert snapshot["budget"] == 30
    assert snapshot["verify"]["every"] == 10
    assert snapshot["timeouts"]["per_case_s"] == 900
    assert snapshot["limits"]["total_tools"] is None
    assert snapshot["limits"]["total_llm"] is None
    assert snapshot["limits"]["wall_clock_s"] is None


def test_v1_cd_max_semantics_are_preserved(compat_overlays, interface_contract):
    path = next(p for p in OLD_RUN_CONFIGS if p.name == "lineA.yaml")
    _, _, cfg, _, task = _resolved_pair(interface_contract, path, compat_overlays)
    assert semantic_config_projection(cfg, task)["cd_max"] == 0.06


def test_wing_penalty_formula_and_tier_zero_acceptance_match_legacy(interface_contract, make_spec, wing_spec_mapping):
    raw = {"Cl": -1.4, "Cd": 0.08, "trust": {"converged": True, "mesh_ok": True},
           "conditions_completed": ["design"], "fidelity": 0, "failure": None}
    scored = interface_contract.normalize_evaluation(raw, make_spec(wing_spec_mapping))
    assert value(scored, "accepted") is True
    assert value(scored, "metric") == pytest.approx(1.2)


@pytest.mark.parametrize(
    "raw",
    [
        {"failure": {"kind": "solver_failed"}, "trust": {"converged": True}},
        {"failure": None, "trust": {}},
        {"failure": None, "trust": {"converged": False}},
    ],
)
def test_first_batch_failed_or_untrusted_result_has_no_score(
    interface_contract, make_spec, wing_spec_mapping, raw
):
    scored = interface_contract.normalize_evaluation(raw, make_spec(wing_spec_mapping))
    assert value(scored, "accepted") is False
    assert value(scored, "metric") is None


@pytest.mark.parametrize(
    "case",
    [
        "seeded_bootstrap",
        "ordinary_improvement",
        "failure_debug",
        "low_confidence_branch",
        "stagnation_backtrack",
        "promotion",
        "format_retry",
        "periodic_verify",
        "holdout_best_options",
        "budget_stop",
        "target_stop",
        "stagnation_stop",
    ],
)
def test_scripted_old_and_interface_runs_have_identical_semantic_trace(
    compat_overlays, interface_contract, case, tmp_path, monkeypatch
):
    old_path = next(path for path in OLD_RUN_CONFIGS if path.name == "lineB.yaml")
    overlay, old_cfg, new_cfg, _, new_task = _resolved_pair(
        interface_contract, old_path, compat_overlays
    )
    script = compatibility_script(case, new_task)

    legacy_hooks = ScriptedLegacyHooks(
        copy.deepcopy(script["tool_responses"]), new_task
    )
    old_backend = ScriptedBackend(copy.deepcopy(script["backend_responses"]))
    new_backend = ScriptedBackend(copy.deepcopy(script["backend_responses"]))
    interface_environment = ScriptedEnvironment(
        copy.deepcopy(script["tool_responses"])
    )
    old_line = inject_legacy_line_hooks(
        old_cfg,
        old_backend,
        legacy_hooks,
        monkeypatch,
        interface_contract.driver,
    )
    new_line = inject_interface_line_hooks(
        new_cfg,
        new_backend,
        interface_environment,
    )
    old_line["budget"] = new_line["budget"] = min(old_line["budget"], 2)
    old_line["output_dir"] = tmp_path / "old" / case
    new_line["output_dir"] = tmp_path / "new" / case
    old = interface_contract.module.run_line(old_line)
    new = interface_contract.module.run_line(new_line)
    assert legacy_hooks.calls
    assert old_backend.prompts == new_backend.prompts
    assert semantic_run_projection(old) == semantic_run_projection(new), overlay


def test_fixed_compatibility_mode_uses_toolruntime_without_chooser(
    compat_overlays, interface_contract, tmp_path
):
    old_path = next(path for path in OLD_RUN_CONFIGS if path.name == "lineB.yaml")
    _, _, new_cfg, _, new_task = _resolved_pair(interface_contract, old_path, compat_overlays)
    script = compatibility_script("ordinary_improvement", new_task)
    line_cfg = inject_interface_line_hooks(
        new_cfg,
        ScriptedBackend(script["backend_responses"]),
        ScriptedEnvironment(script["tool_responses"]),
    )
    line_cfg["output_dir"] = tmp_path
    result = interface_contract.module.run_line(line_cfg)
    trace = value(result, "trace")
    assert trace["interface"] == "ToolRuntime"
    assert trace["mode"] == "fixed"
    assert trace["chooser_calls"] == 0
    assert all(node["evaluate_calls"] == 1 for node in trace["nodes"])


@pytest.mark.parametrize("old_path", OLD_RUN_CONFIGS, ids=lambda p: f"replay-{p.parts[-3]}-{p.stem}")
def test_complete_matrix_has_deterministic_old_vs_interface_replay(
    compat_overlays, interface_contract, old_path, tmp_path, monkeypatch
):
    overlay, old_cfg, new_cfg, _, new_task = _resolved_pair(
        interface_contract, old_path, compat_overlays
    )
    if old_path.stem == "walk":
        version = "v1" if "aero2d" in old_path.parts else "v2"
        routed_old, routed_new = _walk_routes_through_driver_main(
            monkeypatch, interface_contract, old_path, overlay, version
        )
        assert routed_old.get("task_spec") is None
        assert routed_new.get("task_spec") is not None
        return
    script = compatibility_script("matrix_default", new_task)
    legacy_hooks = ScriptedLegacyHooks(
        copy.deepcopy(script["tool_responses"]), new_task
    )
    old_backend = ScriptedBackend(copy.deepcopy(script["backend_responses"]))
    new_backend = ScriptedBackend(copy.deepcopy(script["backend_responses"]))
    interface_environment = ScriptedEnvironment(
        copy.deepcopy(script["tool_responses"])
    )
    old_line = inject_legacy_line_hooks(
        old_cfg,
        old_backend,
        legacy_hooks,
        monkeypatch,
        interface_contract.driver,
    )
    new_line = inject_interface_line_hooks(
        new_cfg,
        new_backend,
        interface_environment,
    )
    old_line["budget"] = new_line["budget"] = min(old_line["budget"], 2)
    old_line["output_dir"] = tmp_path / "old" / old_path.parts[-3] / old_path.stem
    new_line["output_dir"] = tmp_path / "new" / old_path.parts[-3] / old_path.stem
    old = interface_contract.module.run_line(old_line)
    new = interface_contract.module.run_line(new_line)
    assert legacy_hooks.calls
    assert old_backend.prompts == new_backend.prompts
    assert semantic_run_projection(old) == semantic_run_projection(new)
    if old_path.stem in {"lineA", "seeds_lineA"}:
        assert value(new, "trace")["optimizer"] == "nelder-mead"
        assert value(new, "trace")["llm_calls"] == 0
