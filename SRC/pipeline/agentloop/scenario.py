"""Scenario injection: the frozen dataclass plus the scenario-1 instance (inverted wing
in ground effect). ~90 % of this file is prompt / template text, not logic.
Only the 2D wing scenario is distributed in this project."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .environment import TaskSpec


@dataclass(frozen=True)
class Scenario:
    name: str
    task_desc: str            # given to the LLM: physics, variable domain, constraints, holdout statement, response format
    eval_template: str        # python source with {design_code} {scalar_only} {store} {cd_max}; prints one EVAL_JSON: line
    param_keys: tuple[str, ...]
    bounds: dict[str, tuple[float, float]]
    x0: dict[str, float]      # line-A start
    cd_max: float
    holdout_delta: dict[str, float]
    target: float | None
    objective_fn: Callable[[dict], float]   # trusted result dict -> value to maximise
    recovery_menu: str        # DIAGNOSE menu text
    tiers: tuple[int, ...] = (0,)
    task_spec: "TaskSpec | None" = None


# --------------------------------------------------------------------------- #
# scenario 1: inverted single-element wing in ground effect (pipeline.exp_layer.aero2d)
# --------------------------------------------------------------------------- #
WING_CD_MAX = 0.06
WING_PARAM_KEYS = ("h_c", "alpha_deg", "camber")
WING_BOUNDS = {"h_c": (0.05, 1.0), "alpha_deg": (-2.0, 12.0), "camber": (0.0, 0.09)}
WING_X0 = {"h_c": 0.3, "alpha_deg": 2.0, "camber": 0.04}


def wing_objective(result: dict, cd_max: float = WING_CD_MAX) -> float:
    """Downforce minus a drag-budget penalty: -Cl - 10 * max(0, Cd - cd_max)."""
    f = result["forces"]
    return -float(f["Cl"]) - 10.0 * max(0.0, float(f["Cd"]) - cd_max)


def wing_with_cd_max(scenario: Scenario, cd_max: float) -> Scenario:
    """Same scenario with another drag budget (objective_fn and cd_max kept consistent)."""
    return replace(scenario, cd_max=cd_max, objective_fn=partial(wing_objective, cd_max=cd_max))


WING_TASK_DESC = f"""\
## Problem

You are designing a single-element inverted wing (NACA 4-digit section, chord c = 1) running close
to a moving ground, i.e. a race-car front-wing element in ground effect. A deterministic 2-D
steady RANS simulation (OpenFOAM simpleFoam, k-omega SST, Re = 4.5e5) evaluates each design you
propose. The wing is inverted: it produces DOWNFORCE, so the lift coefficient Cl is negative and
"more downforce" means "Cl more negative".

## Physics you should reason with

- Ground effect: as the ride height h/c decreases the flow between the suction (lower) surface and
  the ground accelerates, the suction peak deepens and downforce rises. Below a critical ride height
  the adverse pressure gradient behind the suction peak becomes too steep, the boundary layer on the
  suction surface separates, and downforce COLLAPSES (ground-effect stall). Cl alone keeps saying
  "go lower" right up to the cliff; the field features move first.
- Angle of attack alpha_deg (positive = leading edge pushed down) and camber increase loading and
  drag; both move the collapse to a larger ride height.
- Field features (available on some lines): cp_peak_x / cp_peak_val (location and depth of the
  suction peak on the lower surface), sep_x (chordwise position where wall shear first reverses
  after the peak; 1.0 = attached to the trailing edge), pressure_recovery_slope (Cp slope from the
  peak to the trailing edge). A separation point moving forward and a steepening recovery slope
  signal proximity to the collapse before Cl drops.
- Trust flags: converged, residual_ok, force_settled, yplus_ok, mesh_ok. A result whose trust
  flags are not all true is NOT accepted as an objective value even if it printed forces; a
  "failed" status carries a typed failure (geometry_invalid, mesh_invalid, diverged, not_settled,
  timeout).

## Design variables (PARAMS) and domain

- h_c      ride height / chord, {WING_BOUNDS['h_c'][0]} .. {WING_BOUNDS['h_c'][1]}
- alpha_deg angle of attack in degrees, {WING_BOUNDS['alpha_deg'][0]} .. {WING_BOUNDS['alpha_deg'][1]}
- camber   NACA maximum camber, {WING_BOUNDS['camber'][0]} .. {WING_BOUNDS['camber'][1]}

Optional solver-recovery OPTIONS (integers unless stated; they never change the physics, only how
the solver is run): gap_refine 0..3 (finer mesh in the ground gap), warm_start_from (case_id string
of an earlier ok case), relax_step 0..2 (more under-relaxation), iter_mult 1..4 (more iterations).

## Objective and constraints

Maximise  objective = -Cl - 10 * max(0, Cd - cd_max)  with cd_max = {WING_CD_MAX}, computed only for
trusted results. Keep a margin from the collapse: a design that has separated is worthless.

## Hold-out set

The final design will be re-scored on a hold-out set of ride heights (small +/- shifts of h_c)
that you never see during the search. Robustness near the cliff counts; do not overfit to one
ride height.

## Response format (mandatory)

1. First a ```json block with EXACTLY these keys:
   reading (what the results so far say: position on the downforce-vs-h/c curve and estimated
   distance to the collapse), hypothesis (physical mechanism), change (dict of parameter ->
   multiplicative factor or new value), expected_direction (how Cl, Cd and the field features
   should move), kill_criterion (a NUMERIC criterion written before running that would falsify the
   hypothesis), prediction (a number: the predicted objective value), intent ("continue" |
   "branch" | "backtrack"), confidence (a number in [0, 1]).
2. Then a ```python block containing ONLY
   PARAMS = {{"h_c": <float>, "alpha_deg": <float>, "camber": <float>}}
   and optionally OPTIONS = {{"gap_refine": 0, "warm_start_from": None, "relax_step": 0, "iter_mult": 1}}.
   No imports, no file access, no comments, no other statements.
3. Make ONE primary change per step as a multiplicative update of the previous design (e.g. h_c x 0.8);
   always write the kill criterion before you see the result.
"""

WING_RECOVERY_MENU = """\
Choose exactly one action for the failed / untrusted previous case:
- gap_refine + 1   : OPTIONS["gap_refine"] one level finer (mesh_invalid, yplus_ok False, mesh_ok False)
- warm_start_from  : OPTIONS["warm_start_from"] = case_id of a nearby ok case (diverged, not_settled)
- relax_step + 1   : OPTIONS["relax_step"] one level more under-relaxed (diverged, residual_ok False)
- iter_mult x 2    : OPTIONS["iter_mult"] doubled (not_settled, force_settled False)
- mark_infeasible  : declare the region infeasible and move PARAMS away from it (say so in "hypothesis")
- accept_with_caveat : keep the untrusted reading and continue; you MUST add the key
  "accept_with_caveat": "<reason>" to the JSON block and justify why the value is usable
"""

WING_EVAL_TEMPLATE = '''\
# ---- design (agent output) ----
{design_code}
# ---- evaluation (fixed; nothing below is under the agent's control) ----
import json as _json
import traceback as _tb

_KEYS = ("h_c", "alpha_deg", "camber")
_CD_MAX = {cd_max}


def _emit(metric, is_bug, summary, result, params=None, options=None):
    print("EVAL_JSON: " + _json.dumps({"metric": metric, "is_bug": bool(is_bug), "summary": summary,
                                       "result": result, "params": params, "options": options}, default=str))


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _failed(kind, evidence):
    return {"status": "failed", "forces": None, "trust": {}, "field_summary": None,
            "failure": {"type": kind, "stage": "design", "evidence": str(evidence)[-300:]}}


try:
    _p = dict(PARAMS)
    _o = dict(globals().get("OPTIONS") or {})
    _bad = [k for k in _KEYS if not _num(_p.get(k))] + [k for k in _p if k not in _KEYS]
    if _bad:
        _emit(None, True, "invalid PARAMS (missing / non-numeric / unknown keys): " + ", ".join(_bad),
              _failed("invalid_params", _bad), _p, _o)
    else:
        from pipeline.exp_layer.aero2d import Params, RunOptions, run_case
        _r = _json.loads(run_case(Params(**{k: float(_p[k]) for k in _KEYS}), RunOptions(**_o),
                                  scalar_only={scalar_only}, store="{store}").to_json())
        _metric = None
        if _r["status"] == "ok":
            _false = [k for k, v in _r["trust"].items() if v is not True]
            if _r["trust"] and not _false:
                _metric = -_r["forces"]["Cl"] - 10.0 * max(0.0, _r["forces"]["Cd"] - _CD_MAX)
            _s = "status=ok Cl=%.4f Cd=%.4f trusted=%s" % (_r["forces"]["Cl"], _r["forces"]["Cd"], not _false)
            if _false:
                _s += " untrusted_flags=%s" % _false
            if _r.get("field_summary"):
                _s += " " + " ".join("%s=%.4g" % kv for kv in sorted(_r["field_summary"].items()))
        else:
            _f = _r.get("failure") or {}
            _s = "status=failed type=%s stage=%s evidence=%s" % (
                _f.get("type"), _f.get("stage"), (_f.get("evidence") or "").strip()[:200])
        _s += " objective=%s" % ("None" if _metric is None else "%.4f" % _metric)
        _emit(_metric, _metric is None, _s, _r, _p, _o)
except Exception as _e:
    _emit(None, True, "evaluation error: %s: %s" % (type(_e).__name__, _e), _failed("eval_error", _tb.format_exc()),
          globals().get("PARAMS"), globals().get("OPTIONS"))
'''

WING_SCENARIO = Scenario(
    name="wing",
    task_desc=WING_TASK_DESC,
    eval_template=WING_EVAL_TEMPLATE,
    param_keys=WING_PARAM_KEYS,
    bounds=WING_BOUNDS,
    x0=WING_X0,
    cd_max=WING_CD_MAX,
    holdout_delta={"h_c": 0.01},
    target=None,
    objective_fn=partial(wing_objective, cd_max=WING_CD_MAX),
    recovery_menu=WING_RECOVERY_MENU,
    tiers=(0,),
)
