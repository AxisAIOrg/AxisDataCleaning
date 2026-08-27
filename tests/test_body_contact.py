"""
Unit tests for ``check_body_contact`` (Python side of BodyContactChecker).

The JS side will get its own test file using vitest/jest — see
``tests/test_body_contact.js`` (to be added) in AxisWebInfra.

Run with::

    python -m pytest tests/test_body_contact.py -v
"""

from __future__ import annotations

import pytest

from util.validate_offline_trajectory import check_body_contact


def _state(positions):
    return {
        "object_positions": positions,
        "object_orientations": {k: [0, 0, 0, 1] for k in positions},
    }


# --------------------------------------------------------------------------- #
# 1. Bodies in true AABB overlap -> success                                    #
# --------------------------------------------------------------------------- #

def test_bodies_in_aabb_overlap_returns_true():
    state = _state({
        "akita_black_bowl_1_main": [0.0, 0.0, 0.05],
        "plate_1_main":             [0.01, 0.0, 0.05],  # 1 cm apart, both small
    })
    assert check_body_contact(state, {
        "bodyAName": "akita_black_bowl_1_main",
        "bodyBName": "plate_1_main",
    }) is True


# --------------------------------------------------------------------------- #
# 2. Bodies far apart -> failure                                              #
# --------------------------------------------------------------------------- #

def test_bodies_far_apart_returns_false():
    state = _state({
        "akita_black_bowl_1_main": [0.0, 0.0, 0.05],
        "plate_1_main":             [0.50, 0.50, 0.05],  # 70 cm away
    })
    assert check_body_contact(state, {
        "bodyAName": "akita_black_bowl_1_main",
        "bodyBName": "plate_1_main",
    }) is False


# --------------------------------------------------------------------------- #
# 3. max_contact_distance inflates the threshold                              #
# --------------------------------------------------------------------------- #

def test_max_contact_distance_inflates_threshold():
    state = _state({
        "akita_black_bowl_1_main": [0.0, 0.0, 0.05],
        "plate_1_main":             [0.05, 0.0, 0.05],  # 5 cm apart
    })
    # Without inflation: 5 cm > 2.5 cm default half-extent x 2 = 5 cm, exactly at boundary
    # so it should still be in contact (aabb overlap is <=, not <).
    assert check_body_contact(state, {
        "bodyAName": "akita_black_bowl_1_main",
        "bodyBName": "plate_1_main",
    }) is True

    # With explicit half-extent reduction they don't touch without inflation
    state2 = _state({
        "akita_black_bowl_1_main": [0.0, 0.0, 0.05],
        "plate_1_main":             [0.04, 0.0, 0.05],  # 4 cm apart, half-extents 1 cm each
    })
    # Without max_contact_distance, gap = 4 - 1 - 1 = 2 cm -> not touching
    assert check_body_contact(state2, {
        "bodyAName": "akita_black_bowl_1_main",
        "bodyBName": "plate_1_main",
        "default_half_extent": 0.01,
    }) is False
    # With 0.03 m max_contact_distance, gap 2 cm < 3 cm -> touching
    assert check_body_contact(state2, {
        "bodyAName": "akita_black_bowl_1_main",
        "bodyBName": "plate_1_main",
        "default_half_extent": 0.01,
        "maxContactDistance": 0.03,
    }) is True


# --------------------------------------------------------------------------- #
# 4. Missing body -> failure (graceful)                                       #
# --------------------------------------------------------------------------- #

def test_missing_body_returns_false():
    state = _state({"akita_black_bowl_1_main": [0.0, 0.0, 0.05]})
    assert check_body_contact(state, {
        "bodyAName": "akita_black_bowl_1_main",
        "bodyBName": "plate_1_main",  # not in state
    }) is False


# --------------------------------------------------------------------------- #
# 5. Same body on both sides -> rejected at construction time                 #
# --------------------------------------------------------------------------- #

def test_same_body_name_is_rejected():
    state = _state({"akita_black_bowl_1_main": [0.0, 0.0, 0.05]})
    assert check_body_contact(state, {
        "bodyAName": "akita_black_bowl_1_main",
        "bodyBName": "akita_black_bowl_1_main",
    }) is False


# --------------------------------------------------------------------------- #
# 6. snake_case option names also accepted (legacy compat)                    #
# --------------------------------------------------------------------------- #

def test_snake_case_options_accepted():
    state = _state({
        "akita_black_bowl_1_main": [0.0, 0.0, 0.05],
        "plate_1_main":             [0.0, 0.0, 0.05],  # same position
    })
    assert check_body_contact(state, {
        "body_a_name": "akita_black_bowl_1_main",
        "body_b_name": "plate_1_main",
        "max_contact_distance": 0.0,
        "min_contact_count": 1,
    }) is True
