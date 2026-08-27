"""Unit tests for check_position_delta."""

from __future__ import annotations

import pytest

from util.validate_offline_trajectory import check_position_delta


def _state(positions, initial_position=None):
    s = {
        "object_positions": positions,
        "object_orientations": {k: [0, 0, 0, 1] for k in positions},
    }
    if initial_position is not None:
        s["initial_position"] = initial_position
    return s


# 1. No movement, no bound -> False
def test_no_movement_no_bound_returns_false():
    state = _state({"bowl": [0.0, 0.0, 0.05]})
    assert check_position_delta(state, {"sampleBodyName": "bowl"}) is False


# 2. Movement on X matches minDeltaX -> True (baseline passed explicitly)
def test_x_min_delta_passed():
    state = _state({"bowl": [0.02, 0.0, 0.05]})  # +2 cm
    assert check_position_delta(state, {
        "sampleBodyName": "bowl",
        "axes": ["x"],
        "minDeltaX": 0.018,  # 18 mm
        "initialPosition": [0.0, 0.0, 0.05],  # baseline = origin
    }) is True


# 3. Movement on X below minDeltaX -> False
def test_x_below_min_returns_false():
    state = _state({"bowl": [0.01, 0.0, 0.05]})  # +1 cm, below 18 mm
    assert check_position_delta(state, {
        "sampleBodyName": "bowl",
        "axes": ["x"],
        "minDeltaX": 0.018,
    }) is False


# 4. Negative movement matches maxDeltaX -> True
def test_x_max_delta_passed():
    state = _state({"bowl": [-0.025, 0.0, 0.05]})  # -2.5 cm
    assert check_position_delta(state, {
        "sampleBodyName": "bowl",
        "axes": ["x"],
        "maxDeltaX": -0.018,
        "initialPosition": [0.0, 0.0, 0.05],  # baseline = origin
    }) is True


# 5. Multiple axes (OR semantics) -> True if any matches
def test_multi_axis_or_semantics():
    state = _state({"bowl": [0.05, 0.0, 0.05]})  # X moved, Y not
    assert check_position_delta(state, {
        "sampleBodyName": "bowl",
        "axes": ["x", "y"],
        "minDeltaX": 0.018,
        "minDeltaY": 0.5,  # not satisfied
        "initialPosition": [0.0, 0.0, 0.05],
    }) is True


# 6. Missing body -> False (graceful)
def test_missing_body_returns_false():
    state = _state({})
    assert check_position_delta(state, {"sampleBodyName": "bowl"}) is False


# 7. Explicit initialPosition
def test_explicit_initial_position():
    state = _state({"bowl": [0.020, 0.0, 0.05]})
    assert check_position_delta(state, {
        "sampleBodyName": "bowl",
        "axes": ["x"],
        "minDeltaX": 0.015,
        "initialPosition": [0.000, 0.0, 0.05],  # baseline = origin
    }) is True
    # Same final position, but baseline shifted: now no movement
    state2 = _state({"bowl": [0.020, 0.0, 0.05]})
    assert check_position_delta(state2, {
        "sampleBodyName": "bowl",
        "axes": ["x"],
        "minDeltaX": 0.015,
        "initialPosition": [0.010, 0.0, 0.05],  # baseline closer
    }) is False


# 8. Z axis only
def test_z_axis_only():
    state = _state({"bowl": [0.0, 0.0, 0.10]})
    assert check_position_delta(state, {
        "sampleBodyName": "bowl",
        "axes": ["z"],
        "minDeltaZ": 0.05,
        "initialPosition": [0.0, 0.0, 0.05],
    }) is True


# 9. Empty axes list -> False
def test_empty_axes_returns_false():
    state = _state({"bowl": [0.0, 0.0, 0.05]})
    assert check_position_delta(state, {
        "sampleBodyName": "bowl",
        "axes": [],
    }) is False


# 10. Invalid axis in list -> filtered out; if all invalid -> False
def test_invalid_axis_filtered():
    state = _state({"bowl": [0.02, 0.0, 0.05]})
    # 'foo' is invalid; 'x' is valid
    assert check_position_delta(state, {
        "sampleBodyName": "bowl",
        "axes": ["foo", "x"],
        "minDeltaX": 0.018,
    }) is True
    # All invalid -> False
    assert check_position_delta(state, {
        "sampleBodyName": "bowl",
        "axes": ["foo", "bar"],
    }) is False
