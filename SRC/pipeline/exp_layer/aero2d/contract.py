"""Pure functions of the simblock contract (REQUIREMENTS §2): data models,
SETTINGS, hashing, validation, NACA geometry, failure classification and
Cp features. No file I/O, no subprocess, no gmsh. Nothing here raises except
``cp_features`` (documented) — the orchestration in ``__init__`` catches it.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# fixed settings (participate in the input hash; keys are contract, values may
# be tuned within an order of magnitude, see REQUIREMENTS §2.3)
# --------------------------------------------------------------------------- #
SETTINGS: dict = {
    "U_inf": 1.0, "chord": 1.0, "Re": 4.5e5,
    "naca_p": 0.4, "naca_t": 0.12, "n_surface": 120,
    "domain": {"x_min": -5.0, "x_max": 12.0, "y_max": 6.0},
    # ponytail: bl_* are part of the key contract but unused in v1 (no gmsh BoundaryLayer field;
    # isotropic wall refinement + wall functions give y+ ~ 30 on the wing).
    "mesh": {"far": 0.5, "wing": 0.006, "gap": 0.02, "wake": 0.06,
             "bl_layers": 12, "bl_first": 0.0015, "bl_ratio": 1.2,
             "algorithm": 6, "gap_refine_factor": 0.6},
    "solver": {"end_time": 1500, "write_interval": 1500,
               "residual_control": {"p": 1e-4, "U": 1e-5, "k": 1e-5, "omega": 1e-5},
               "relax": [{"p": 0.3, "U": 0.7, "turb": 0.7},
                         {"p": 0.2, "U": 0.5, "turb": 0.5},
                         {"p": 0.1, "U": 0.3, "turb": 0.3}]},
    "trust": {"yplus_min": 1.0, "yplus_max": 300.0, "max_nonortho": 70.0, "max_skew": 4.0,
              "force_settle_window": 100, "force_settle_tol": 0.01},
}

FAILURE_TYPES = ("geometry_invalid", "mesh_invalid", "diverged", "not_settled", "timeout")
TRUST_KEYS = ("converged", "residual_ok", "force_settled", "yplus_ok", "mesh_ok")
STAGES = ("geometry", "mesh", "solve", "post")

BOUNDS = {"h_c": (0.03, 2.0), "alpha_deg": (-2.0, 12.0), "camber": (0.0, 0.09)}
OPTION_BOUNDS = {"gap_refine": (0, 3), "relax_step": (0, 2), "iter_mult": (1, 4)}


# --------------------------------------------------------------------------- #
# data models
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Params:
    h_c: float
    alpha_deg: float
    camber: float


@dataclass(frozen=True)
class RunOptions:
    gap_refine: int = 0
    warm_start_from: str | None = None
    relax_step: int = 0
    iter_mult: int = 1


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    input_hash: str
    status: str
    forces: dict | None
    trust: dict
    field_summary: dict | None
    failure: dict | None
    provenance: dict
    iterations: int
    runtime_s: float

    @property
    def trusted(self) -> bool:
        return self.status == "ok" and all(self.trust.get(k) is True for k in TRUST_KEYS)

    def to_json(self, deterministic: bool = False) -> str:
        d = asdict(self)
        if deterministic:
            d.pop("runtime_s")
        return json.dumps(d, sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "CaseResult":
        return cls(**json.loads(s))


def failed_result(h: str, failure: dict, mesh_hash: str | None, build: dict,
                  iterations: int = 0, runtime_s: float = 0.0) -> CaseResult:
    return CaseResult(
        case_id=h[:12], input_hash=h, status="failed", forces=None,
        trust={k: False for k in TRUST_KEYS}, field_summary=None, failure=failure,
        provenance={"mesh_hash": mesh_hash, "solver_build": build},
        iterations=iterations, runtime_s=runtime_s,
    )


# --------------------------------------------------------------------------- #
# hashing and validation
# --------------------------------------------------------------------------- #
def input_hash(params: Params, options: RunOptions = RunOptions(), settings: dict = SETTINGS) -> str:
    payload = {"params": asdict(params), "options": asdict(options), "settings": settings}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def validate(params: Params, options: RunOptions, store) -> str | None:
    """Return an evidence string if the inputs are invalid, else None."""
    for name, (lo, hi) in BOUNDS.items():
        v = getattr(params, name, None)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not lo <= v <= hi:
            return f"{name}={v!r} outside [{lo}, {hi}]"
    for name, (lo, hi) in OPTION_BOUNDS.items():
        v = getattr(options, name, None)
        if not _is_int(v) or not lo <= v <= hi:
            return f"option {name}={v!r} outside [{lo}, {hi}]"
    src = options.warm_start_from
    if src is not None:
        if not isinstance(src, str) or not src or find_case_dir(store, src) is None:
            return f"option warm_start_from={src!r} does not name a case in the store"
    return None


def find_case_dir(store, case_id: str):
    """store/<hash> whose name starts with case_id and holds a result.json, else None."""
    store = Path(store)
    if not store.is_dir():
        return None
    hits = sorted(p for p in store.iterdir() if p.is_dir() and p.name.startswith(case_id) and (p / "result.json").is_file())
    return hits[0] if hits else None


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def naca4(m: float, p: float, t: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Closed NACA 4-digit contour, chord 1, camber up, closed trailing edge.
    Order: TE -> upper surface -> LE -> lower surface -> TE (2n-1 points, last == first)."""
    beta = np.linspace(0.0, math.pi, n)
    x = 0.5 * (1.0 - np.cos(beta))
    yt = 5.0 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)
    yc = np.zeros_like(x)
    dyc = np.zeros_like(x)
    if m > 0.0:
        front = x < p
        yc[front] = m / p**2 * (2 * p * x[front] - x[front] ** 2)
        yc[~front] = m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x[~front] - x[~front] ** 2)
        dyc[front] = 2 * m / p**2 * (p - x[front])
        dyc[~front] = 2 * m / (1 - p) ** 2 * (p - x[~front])
    th = np.arctan(dyc)
    xu, yu = x - yt * np.sin(th), yc + yt * np.cos(th)
    xl, yl = x + yt * np.sin(th), yc - yt * np.cos(th)
    xs = np.concatenate([xu[::-1], xl[1:]])
    ys = np.concatenate([yu[::-1], yl[1:]])
    xs[-1], ys[-1] = xs[0], ys[0]  # exact closure (closed-TE polynomial already gives ~1e-17)
    return xs, ys


def wing_coords(params: Params, n: int = SETTINGS["n_surface"]) -> tuple[np.ndarray, np.ndarray]:
    """Inverted NACA-4 wing, rotated by alpha (positive = leading edge down) about
    the quarter chord, translated so that the leading edge is at x = 0 and the
    lowest point sits at y = h_c above the ground (y = 0)."""
    x0, y0 = naca4(float(params.camber), SETTINGS["naca_p"], SETTINGS["naca_t"], n)
    y0 = -y0  # invert: camber points down, suction surface below
    a = math.radians(float(params.alpha_deg))
    c, s = math.cos(a), math.sin(a)
    xr = 0.25 + (x0 - 0.25) * c - y0 * s
    yr = (x0 - 0.25) * s + y0 * c
    le = n - 1  # index of the nose point (x0 == 0)
    x = xr - xr[le]
    y = yr - yr.min() + float(params.h_c)
    return x, y


# --------------------------------------------------------------------------- #
# failure classification
# --------------------------------------------------------------------------- #
_DIVERGE_RE = re.compile(r"\bnan\b|^Floating point exception|sigFpe::|FOAM FATAL|residual = [0-9.]+e\+[0-9]{2,}", re.M)
_MESH_BAD_RE = re.compile(r"Failed \d+ mesh checks|negative volume|\bFailed\b", re.I)


def _evidence(log: str, pattern: re.Pattern | None, fallback: str) -> str:
    lines = [ln.strip() for ln in log.splitlines() if ln.strip()]
    if pattern is not None:
        hits = [ln for ln in lines if pattern.search(ln)]
        if hits:
            return hits[0][:300]
    tail = " | ".join(lines[-2:]) if lines else ""
    return (tail or fallback)[:300]


_TIME_SPLIT_RE = re.compile(r"^Time = (\d+)", re.M)
_INIT_RES_RE = re.compile(r"Solving for (\w+), Initial residual = ([0-9.eE+-]+|nan)")


def final_initial_residuals(log: str) -> tuple[str, dict[str, float]]:
    """(last iteration number or '?', {field: first initial residual in the final Time block})."""
    blocks = _TIME_SPLIT_RE.split(log)
    if len(blocks) < 2:
        return "?", {}
    res: dict[str, float] = {}
    for name, val in _INIT_RES_RE.findall(blocks[-1]):
        try:
            res.setdefault(name, float(val))
        except ValueError:
            pass
    return blocks[-2], res


def not_settled_evidence(log: str, cl_mean: float | None = None, cd_mean: float | None = None,
                         kind: str | None = None) -> str:
    """Readable cliff indicator for an unconverged solve, e.g.
    ``not converged after 1500 iters; p_res=2.3e-03 U_res=3.2e-04; Cl_last100=-1.2300 Cd_last100=0.0450; kind=residual_stall``.
    Always <= 300 characters. Cl/Cd/kind are appended by run_case when coefficient.dat is available."""
    n, res = final_initial_residuals(log)
    u = [res[k] for k in ("Ux", "Uy", "Uz") if k in res]
    fmt = lambda v: "?" if v is None else f"{v:.1e}"
    ev = f"not converged after {n} iters; p_res={fmt(res.get('p'))} U_res={fmt(max(u) if u else None)}"
    if cl_mean is not None and cd_mean is not None:
        ev += f"; Cl_last100={cl_mean:.4f} Cd_last100={cd_mean:.4f}"
    if kind:
        ev += f"; kind={kind}"
    return ev[:300]


def classify_failure(stage: str, log_text, exit_code, timed_out: bool) -> dict | None:
    """Map (stage, log, exit code, timeout flag) to a typed failure or None. Never raises."""
    try:
        log = log_text if isinstance(log_text, str) else (log_text or b"").decode("utf-8", "replace") if isinstance(log_text, bytes) else str(log_text or "")
    except Exception:  # pragma: no cover - defensive
        log = ""
    bad_exit = exit_code is not None and exit_code != 0
    if timed_out:
        return {"type": "timeout", "stage": stage, "evidence": _evidence(log, None, f"timed out during {stage}")}
    if stage == "geometry":
        return {"type": "geometry_invalid", "stage": stage, "evidence": _evidence(log, None, "geometry invalid")}
    if stage == "mesh":
        if bad_exit or _MESH_BAD_RE.search(log):
            return {"type": "mesh_invalid", "stage": stage,
                    "evidence": _evidence(log, _MESH_BAD_RE, f"mesh stage exit code {exit_code}")}
        return None
    if stage == "solve":
        if bad_exit or _DIVERGE_RE.search(log):
            return {"type": "diverged", "stage": stage,
                    "evidence": _evidence(log, _DIVERGE_RE, f"solver exit code {exit_code}")}
        if "SIMPLE solution converged" not in log:
            return {"type": "not_settled", "stage": stage, "evidence": not_settled_evidence(log)}
        return None
    if stage == "post":
        if bad_exit or log.strip():
            return {"type": "diverged", "stage": stage, "evidence": ("post: " + _evidence(log, None, "parse failed"))[:300]}
        return None
    return None


# --------------------------------------------------------------------------- #
# Cp features
# --------------------------------------------------------------------------- #
def cp_features(x, cp, tau_x) -> dict:
    """Suction-surface features from samples ordered LE -> TE. Raises ValueError on
    fewer than 3 samples or any NaN (caught by run_case as a post failure)."""
    x, cp, tau = (np.asarray(a, dtype=float).ravel() for a in (x, cp, tau_x))
    if not (len(x) == len(cp) == len(tau)) or len(x) < 3:
        raise ValueError(f"cp_features: need >= 3 samples, got {len(x)}")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(cp)) and np.all(np.isfinite(tau))):
        raise ValueError("cp_features: NaN/inf in input")
    k = int(np.argmin(cp))
    sep_x = 1.0
    neg = np.nonzero(tau[k + 1:] < 0.0)[0]
    if len(neg):
        j = k + 1 + int(neg[0])
        t0, t1 = tau[j - 1], tau[j]
        sep_x = float(x[j - 1] + (x[j] - x[j - 1]) * t0 / (t0 - t1)) if t0 > 0.0 else float(x[j])
    # ponytail: slope from a 1-D least-squares fit peak->TE; 0.0 when the peak is the last sample
    slope = float(np.polyfit(x[k:], cp[k:], 1)[0]) if len(x) - k >= 2 else 0.0
    return {"cp_peak_x": float(x[k]), "cp_peak_val": float(cp[k]), "sep_x": sep_x,
            "pressure_recovery_slope": slope}
