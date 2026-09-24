"""Unit tests for ``pipeline.agentloop`` v2 -- REQUIREMENTS (agentloop v2) section 3, U12-U14.

Second-scale. No OpenFOAM, no ``claude`` CLI, no LLM calls, nothing written to
disk except under ``tmp_path``. Only names exported by ``pipeline.agentloop``
are imported; ``pipeline.agentloop.driver.Tree.init`` is monkeypatched by
dotted string in U14 to stop ``run_line`` right after its directory check.
"""

from __future__ import annotations

import copy
import dataclasses
import random
from pathlib import Path

import pytest

from pipeline.agentloop import (
    WING_SCENARIO,
    Scenario,
    aggregate_lines,
    perturb_x0,
    run_line,
)

PARAM_KEYS = ("h_c", "alpha_deg", "camber")


def _in_bounds(x: dict, scenario: Scenario) -> bool:
    return all(
        scenario.bounds[k][0] <= float(x[k]) <= scenario.bounds[k][1]
        for k in scenario.param_keys
        if k != "tier"
    )


def _close(a: dict, b: dict, keys, tol: float = 1e-9) -> bool:
    return all(abs(float(a[k]) - float(b[k])) < tol for k in keys)


# ---------------------------------------------------------------------------
# U12  perturb_x0
# ---------------------------------------------------------------------------


@pytest.fixture
def tier_scenario() -> Scenario:
    """Two-tier variant of the wing scenario: ``tier`` is a param key, x0 carries it."""
    return dataclasses.replace(
        WING_SCENARIO,
        param_keys=(*WING_SCENARIO.param_keys, "tier"),
        tiers=(0, 2),
        x0={**WING_SCENARIO.x0, "tier": 0},
    )


def test_u12_seed0_is_x0_and_a_new_dict():
    out = perturb_x0(WING_SCENARIO, 0)
    assert isinstance(out, dict)
    assert out == WING_SCENARIO.x0
    assert out is not WING_SCENARIO.x0
    # mutating the result must not leak into the scenario
    out["h_c"] = 999.0
    assert WING_SCENARIO.x0["h_c"] != 999.0


def test_u12_deterministic_per_seed_and_seeds_differ():
    a1 = perturb_x0(WING_SCENARIO, 1)
    a1b = perturb_x0(WING_SCENARIO, 1)
    a2 = perturb_x0(WING_SCENARIO, 2)
    assert a1 == a1b
    assert a1 != a2
    assert a1 != WING_SCENARIO.x0
    assert set(a1) == set(WING_SCENARIO.param_keys)
    assert list(a1) == list(WING_SCENARIO.param_keys)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 7, 12345])
def test_u12_all_keys_within_bounds(seed):
    out = perturb_x0(WING_SCENARIO, seed)
    assert set(out) == set(WING_SCENARIO.param_keys)
    assert _in_bounds(out, WING_SCENARIO)


@pytest.mark.parametrize("seed", [1, 2, 5])
def test_u12_clipping_when_x0_sits_on_the_boundary(seed):
    # x0 on the lower bound: negative perturbations must be clipped, not exceed lo
    lo_x0 = {k: WING_SCENARIO.bounds[k][0] for k in WING_SCENARIO.param_keys}
    sc = dataclasses.replace(WING_SCENARIO, x0=lo_x0)
    out = perturb_x0(sc, seed)
    assert _in_bounds(out, sc)
    hi_x0 = {k: WING_SCENARIO.bounds[k][1] for k in WING_SCENARIO.param_keys}
    sc_hi = dataclasses.replace(WING_SCENARIO, x0=hi_x0)
    out_hi = perturb_x0(sc_hi, seed)
    assert _in_bounds(out_hi, sc_hi)


@pytest.mark.parametrize("seed", [1, 2, 42])
def test_u12_matches_specified_formula(seed):
    # section 2: rng = random.Random(seed); for each continuous key in param_keys order,
    # u = rng.uniform(-1, 1); x0[k] + u * 0.15 * (hi - lo), clipped to [lo, hi]
    rng = random.Random(seed)
    expected = {}
    for k in WING_SCENARIO.param_keys:
        lo, hi = WING_SCENARIO.bounds[k]
        u = rng.uniform(-1, 1)
        v = WING_SCENARIO.x0[k] + u * 0.15 * (hi - lo)
        expected[k] = min(max(v, lo), hi)
    out = perturb_x0(WING_SCENARIO, seed)
    assert _close(out, expected, WING_SCENARIO.param_keys, tol=1e-12)


def test_u12_tier_key_is_preserved(tier_scenario):
    for seed in (0, 1, 2):
        out = perturb_x0(tier_scenario, seed)
        assert set(out) == set(tier_scenario.param_keys)
        assert list(out) == list(tier_scenario.param_keys)
        assert out["tier"] == 0
        assert _in_bounds(out, tier_scenario)
    # continuous part still perturbed for seed != 0, and identical to the
    # tier-less scenario's perturbation (tier is skipped, not consumed from rng)
    out1 = perturb_x0(tier_scenario, 1)
    assert not _close(out1, tier_scenario.x0, PARAM_KEYS)
    assert _close(out1, perturb_x0(WING_SCENARIO, 1), PARAM_KEYS)


def test_u12_does_not_modify_scenario():
    before_x0 = copy.deepcopy(WING_SCENARIO.x0)
    before_bounds = copy.deepcopy(WING_SCENARIO.bounds)
    for seed in (0, 1, 2):
        perturb_x0(WING_SCENARIO, seed)
    assert WING_SCENARIO.x0 == before_x0
    assert WING_SCENARIO.bounds == before_bounds


# ---------------------------------------------------------------------------
# U13  aggregate_lines
# ---------------------------------------------------------------------------

GROUP_KEYS = {
    "n_runs",
    "best_metric_mean",
    "best_metric_std",
    "best_metric_min",
    "best_metric_max",
    "holdout_robust_frac",
    "n_failed_mean",
    "n_cases_mean",
    "stop_reasons",
}


def _summary(
    line: str,
    best: float | None,
    *,
    memory_used: bool,
    robust: bool,
    n_failed: int = 0,
    n_cases: int = 10,
    stop_reason: str = "budget",
    seed: int = 0,
) -> dict:
    return {
        "line": line,
        "seed": seed,
        "run_tag": f"s{seed}",
        "best_metric": best,
        "best_params": None if best is None else {"h_c": 0.2, "alpha_deg": 3.0, "camber": 0.04},
        "best_case_id": None if best is None else f"{line}-{seed}",
        "holdout": {"robust": robust, "delta": {"h_c": 0.01}},
        "n_failed": n_failed,
        "n_cases": n_cases,
        "n_unique_hash": n_cases,
        "n_untrusted_accepted": 0,
        "stop_reason": stop_reason,
        "memory_used": memory_used,
        "predictions": [],
    }


@pytest.fixture
def six_summaries() -> list[dict]:
    return [
        # A x2, memory
        _summary("A", 1.0, memory_used=True, robust=True, n_failed=0, n_cases=10,
                 stop_reason="budget", seed=0),
        _summary("A", 2.0, memory_used=True, robust=False, n_failed=2, n_cases=12,
                 stop_reason="stagnation", seed=1),
        # B x2, memory (one run has best_metric None)
        _summary("B", 3.0, memory_used=True, robust=True, n_failed=1, n_cases=8,
                 stop_reason="budget", seed=0),
        _summary("B", None, memory_used=True, robust=True, n_failed=3, n_cases=4,
                 stop_reason="budget", seed=1),
        # B x2, no memory (both None)
        _summary("B", None, memory_used=False, robust=False, n_failed=5, n_cases=5,
                 stop_reason="stagnation", seed=0),
        _summary("B", None, memory_used=False, robust=False, n_failed=7, n_cases=9,
                 stop_reason="target", seed=1),
    ]


def test_u13_three_groups_with_expected_keys(six_summaries):
    agg = aggregate_lines(six_summaries)
    assert set(agg) == {"A|mem", "B|mem", "B|nomem"}
    for key, g in agg.items():
        assert isinstance(g, dict), key
        assert GROUP_KEYS <= set(g), (key, sorted(g))
        assert g["n_runs"] == 2


def test_u13_a_mem_exact_statistics(six_summaries):
    g = aggregate_lines(six_summaries)["A|mem"]
    assert g["n_runs"] == 2
    assert abs(g["best_metric_mean"] - 1.5) < 1e-9
    assert abs(g["best_metric_std"] - 0.5) < 1e-9  # population std of {1, 2}
    assert abs(g["best_metric_min"] - 1.0) < 1e-9
    assert abs(g["best_metric_max"] - 2.0) < 1e-9
    assert abs(g["holdout_robust_frac"] - 0.5) < 1e-9
    assert abs(g["n_failed_mean"] - 1.0) < 1e-9
    assert abs(g["n_cases_mean"] - 11.0) < 1e-9
    assert g["stop_reasons"] == {"budget": 1, "stagnation": 1}


def test_u13_none_best_metric_counts_in_n_runs_only(six_summaries):
    g = aggregate_lines(six_summaries)["B|mem"]
    assert g["n_runs"] == 2
    assert abs(g["best_metric_mean"] - 3.0) < 1e-9  # None excluded from the mean
    assert abs(g["best_metric_std"] - 0.0) < 1e-9  # single value -> population std 0
    assert abs(g["best_metric_min"] - 3.0) < 1e-9
    assert abs(g["best_metric_max"] - 3.0) < 1e-9
    assert abs(g["holdout_robust_frac"] - 1.0) < 1e-9
    assert abs(g["n_failed_mean"] - 2.0) < 1e-9
    assert abs(g["n_cases_mean"] - 6.0) < 1e-9
    assert g["stop_reasons"] == {"budget": 2}


def test_u13_all_none_group_has_none_mean(six_summaries):
    g = aggregate_lines(six_summaries)["B|nomem"]
    assert g["n_runs"] == 2
    assert g["best_metric_mean"] is None
    assert abs(g["holdout_robust_frac"] - 0.0) < 1e-9
    assert abs(g["n_failed_mean"] - 6.0) < 1e-9
    assert abs(g["n_cases_mean"] - 7.0) < 1e-9
    assert g["stop_reasons"] == {"stagnation": 1, "target": 1}


def test_u13_population_std_three_runs():
    runs = [
        _summary("C", 1.0, memory_used=True, robust=True, seed=0),
        _summary("C", 2.0, memory_used=True, robust=True, seed=1),
        _summary("C", 6.0, memory_used=True, robust=False, seed=2),
    ]
    g = aggregate_lines(runs)["C|mem"]
    assert g["n_runs"] == 3
    assert abs(g["best_metric_mean"] - 3.0) < 1e-9
    # population std of {1, 2, 6}: sqrt(((4 + 1 + 9) / 3)) = sqrt(14/3)
    assert abs(g["best_metric_std"] - (14.0 / 3.0) ** 0.5) < 1e-9
    assert abs(g["best_metric_min"] - 1.0) < 1e-9
    assert abs(g["best_metric_max"] - 6.0) < 1e-9
    assert abs(g["holdout_robust_frac"] - (2.0 / 3.0)) < 1e-9


def test_u13_single_run_std_is_zero():
    g = aggregate_lines([_summary("A", 1.25, memory_used=False, robust=True)])["A|nomem"]
    assert g["n_runs"] == 1
    assert abs(g["best_metric_mean"] - 1.25) < 1e-9
    assert abs(g["best_metric_std"] - 0.0) < 1e-9
    assert abs(g["best_metric_min"] - 1.25) < 1e-9
    assert abs(g["best_metric_max"] - 1.25) < 1e-9


def test_u13_empty_input_returns_empty_dict():
    assert aggregate_lines([]) == {}


def test_u13_pure_does_not_mutate_input(six_summaries):
    snapshot = copy.deepcopy(six_summaries)
    aggregate_lines(six_summaries)
    assert six_summaries == snapshot


# ---------------------------------------------------------------------------
# U14  run_line directory naming with run_tag (no cases run)
# ---------------------------------------------------------------------------


class _InjectedStop(Exception):
    """Raised from the patched ``Tree.init`` so run_line stops right after its dir check."""


def _stub_verify(*args, **kwargs):
    return {}


def _cfg(exp_dir: Path, store: Path, run_tag: str | None) -> dict:
    return {
        "line": "A",
        "scenario": WING_SCENARIO,
        "budget": 3,
        "verify_every": 1,
        "exp_dir": exp_dir,
        "store": store,
        "model": None,
        "tau": 0.4,
        "k_backtrack": 4,
        "k_stagnation": 8,
        "seed": 1,
        "run_tag": run_tag,
        "memory": None,
        "verify_fn": _stub_verify,
    }


@pytest.fixture
def stop_at_tree_init(monkeypatch):
    def _boom(*args, **kwargs):
        raise _InjectedStop("Tree.init reached")

    monkeypatch.setattr("pipeline.agentloop.driver.Tree.init", staticmethod(_boom))


def test_u14_existing_tagged_dir_raises_file_exists(tmp_path):
    exp_dir = tmp_path / "exp"
    tagged = exp_dir / "lines" / "A-s1"
    tagged.mkdir(parents=True)
    store = tmp_path / "store"
    with pytest.raises(FileExistsError):
        run_line(_cfg(exp_dir, store, run_tag="s1"))
    # nothing ran
    assert list(tagged.iterdir()) == []
    assert list(exp_dir.rglob("summary.json")) == []
    assert not store.exists() or not any(store.iterdir())


def test_u14_existing_untagged_dir_does_not_block_tagged_run(tmp_path, stop_at_tree_init):
    exp_dir = tmp_path / "exp"
    (exp_dir / "lines" / "A").mkdir(parents=True)
    store = tmp_path / "store"
    # lines/A exists but we run with run_tag "s1" -> lines/A-s1 is the target;
    # the directory check must pass and execution proceeds to Tree.init (patched).
    with pytest.raises(_InjectedStop):
        run_line(_cfg(exp_dir, store, run_tag="s1"))
    assert list(exp_dir.rglob("summary.json")) == []


def test_u14_existing_tagged_dir_does_not_block_untagged_run(tmp_path, stop_at_tree_init):
    exp_dir = tmp_path / "exp"
    (exp_dir / "lines" / "A-s1").mkdir(parents=True)
    store = tmp_path / "store"
    # run_tag None -> v1 naming (lines/A); the tagged dir is irrelevant.
    with pytest.raises(_InjectedStop):
        run_line(_cfg(exp_dir, store, run_tag=None))


def test_u14_untagged_existing_dir_still_raises_with_v2_cfg(tmp_path):
    # v1 U10 semantics are unchanged when run_tag is None and memory is given
    exp_dir = tmp_path / "exp"
    (exp_dir / "lines" / "A").mkdir(parents=True)
    store = tmp_path / "store"
    with pytest.raises(FileExistsError):
        run_line(_cfg(exp_dir, store, run_tag=None))
