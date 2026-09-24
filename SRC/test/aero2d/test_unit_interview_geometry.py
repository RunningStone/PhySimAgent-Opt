"""U02/U18/U19: tests written against REQUIREMENTS.html, not implementation.

These are geometry unit tests, never physical OpenFOAM acceptance evidence.
"""
from __future__ import annotations

import copy
import importlib

import numpy as np
import pytest


@pytest.fixture
def geometry():
    return importlib.import_module("pipeline.exp_layer.aero2d.geometry")


@pytest.fixture
def shape():
    return {"x": [0.0, 0.25, 0.5, 0.75, 1.0],
            "upper": [0.0, 0.07, 0.09, 0.04, 0.0],
            "lower": [0.0, -0.03, -0.04, -0.02, 0.0]}


@pytest.fixture
def domain(shape):
    return {"x": shape["x"], "min_thickness": 0.001,
            "max_thickness": 0.3, "coordinate_bounds": [-0.3, 0.3]}


def test_valid_shape_is_not_changed_or_reconstructed(geometry, shape, domain):
    before = copy.deepcopy(shape)
    actual = geometry.validate_shape(shape, domain)
    for key in ("x", "upper", "lower"):
        assert actual[key] == before[key]
    assert shape == before, "validation must not mutate the proposal"


@pytest.mark.parametrize("field,value", [
    ("x", [0.0, 0.25, 0.25, 0.75, 1.0]),
    ("x", [0.0, 0.5, 0.25, 0.75, 1.0]),
    ("x", [-0.1, 0.25, 0.5, 0.75, 1.0]),
    ("x", [0.0, 0.25, 0.5, 0.75, 1.1]),
    ("x", [0.0, 0.2, 0.5, 0.75, 1.0]),
    ("x", [0.0, 0.5, 1.0]),
    ("x", [0.0, 0.125, 0.25, 0.5, 0.75, 1.0]),
    ("upper", [0.01, 0.07, 0.09, 0.04, 0.0]),
    ("lower", [0.0, -0.03, -0.04, -0.02, -0.01]),
    ("upper", [0.0, 0.07, -0.05, 0.04, 0.0]),
    ("upper", [0.0, -0.03, 0.09, 0.04, 0.0]),
    ("upper", [0.0, 0.29, 0.09, 0.04, 0.0]),
    ("upper", [0.0, 0.31, 0.09, 0.04, 0.0]),
    ("upper", [0.0, float("nan"), 0.09, 0.04, 0.0]),
    ("upper", [0.0, float("inf"), 0.09, 0.04, 0.0]),
    ("upper", [0.0, True, 0.09, 0.04, 0.0]),
    ("upper", [0.0, "0.07", 0.09, 0.04, 0.0]),
    ("upper", []),
    ("lower", None),
])
def test_invalid_shape_is_rejected(geometry, shape, domain, field, value):
    shape[field] = value
    with pytest.raises(ValueError):
        geometry.validate_shape(shape, domain)


@pytest.mark.parametrize("key,value", [
    ("path", "../../outside.dat"),
    ("generator", "__import__('os').system('touch injected')"),
    ("mesh", {"resolution": 99}),
])
def test_shape_cannot_carry_program_path_or_mesh_authority(geometry, shape, domain, key, value):
    shape[key] = value
    with pytest.raises(ValueError):
        geometry.validate_shape(shape, domain)


@pytest.mark.parametrize("key", ["x", "upper", "lower"])
def test_shape_requires_all_coordinate_arrays(geometry, shape, domain, key):
    del shape[key]
    with pytest.raises(ValueError):
        geometry.validate_shape(shape, domain)


def test_normalization_is_idempotent_and_preserves_identity_input(geometry, shape):
    policy = {"chord_normalization": False}
    first = geometry.normalize_shape(shape, policy)
    second = geometry.normalize_shape(first, policy)
    for key in ("x", "upper", "lower"):
        assert first[key] == shape[key]
        assert second[key] == first[key]


def test_shape_coords_uses_each_explicit_surface_point(geometry, shape):
    x, y = geometry.shape_coords(shape)
    x, y = np.asarray(x), np.asarray(y)
    assert x.shape == y.shape and x.ndim == 1
    assert np.isfinite(x).all() and np.isfinite(y).all()
    assert (x[0], y[0]) == (x[-1], y[-1]), "exported boundary must close"
    expected = {(float(a), float(b)) for surface in ("upper", "lower")
                for a, b in zip(shape["x"], shape[surface], strict=True)}
    actual = set(zip(x.tolist(), y.tolist(), strict=True))
    assert actual == expected, "boundary must use the submitted explicit profile"


def test_distinct_legal_profiles_reach_distinct_geometry(geometry, shape, domain):
    other = copy.deepcopy(shape)
    other["upper"][2] += 0.025
    geometry.validate_shape(other, domain)
    xa, ya = geometry.shape_coords(shape)
    xb, yb = geometry.shape_coords(other)
    assert not (np.array_equal(xa, xb) and np.array_equal(ya, yb))
