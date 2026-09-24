"""One explicit airfoil contract: fixed chordwise stations, two linear surfaces."""

from __future__ import annotations

import math

import numpy as np


def _arrays(shape: dict) -> dict:
    if not isinstance(shape, dict) or set(shape) != {"x", "upper", "lower"}:
        raise ValueError("shape requires exactly x, upper and lower coordinate arrays")
    out = {}
    for key, values in shape.items():
        if not isinstance(values, (list, tuple)) or len(values) < 3:
            raise ValueError(f"{key} requires at least three coordinates")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            raise ValueError(f"{key} must contain finite numeric literals")
        out[key] = [float(v) for v in values]
    if len({len(v) for v in out.values()}) != 1:
        raise ValueError("coordinate array lengths differ")
    if any(b <= a for a, b in zip(out["x"], out["x"][1:])):
        raise ValueError("x must increase strictly")
    if out["x"][0] != 0 or out["x"][-1] != 1:
        raise ValueError("the chord endpoints must be 0 and 1")
    if any(out["upper"][i] != out["lower"][i] for i in (0, -1)):
        raise ValueError("leading and trailing edges must close")
    if any(u <= l for u, l in zip(out["upper"][1:-1], out["lower"][1:-1])):
        raise ValueError("interior thickness must be positive")
    return out


def validate_shape(shape: dict, domain: dict) -> dict:
    """Validate the actual piecewise linear contour; never reconstruct a NACA."""
    out = _arrays(shape)
    if out["x"] != domain.get("x"):
        raise ValueError("x station positions and count are frozen")
    lo, hi = domain["coordinate_bounds"]
    if any(not lo <= v <= hi for key in ("upper", "lower") for v in out[key]):
        raise ValueError("surface coordinate outside the allowed range")
    minimum, maximum = domain["min_thickness"], domain["max_thickness"]
    if any(not minimum <= u - l <= maximum for u, l in zip(out["upper"][1:-1], out["lower"][1:-1])):
        raise ValueError("thickness outside the allowed range")
    return out


def normalize_shape(shape: dict, policy: dict) -> dict:
    """The selected contract already has unit chord; no implicit transformations."""
    if set(policy) - {"chord_normalization"} or policy.get("chord_normalization", False) is not False:
        raise ValueError("this contract accepts only already normalized unit-chord shapes")
    return _arrays(shape)


def shape_coords(shape: dict) -> tuple[np.ndarray, np.ndarray]:
    """Closed TE → upper → LE → lower → TE contour, preserving every station."""
    out = _arrays(shape)
    return (np.asarray(out["x"][::-1] + out["x"][1:]),
            np.asarray(out["upper"][::-1] + out["lower"][1:]))
