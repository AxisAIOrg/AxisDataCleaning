"""Unit tests for check_ordered_step."""

from __future__ import annotations

import pytest

from util.validate import check_ordered_step


# A tiny fake "dispatcher" that returns cfg["pass"] for demonstration/testing.
def _dispatch(state, cfg):
    # cfg: {"type": ..., "pass": bool}
    return bool(cfg.get("pass", False))


# 1. All steps pass -> True
def test_all_steps_pass():
    states = [{"t": 0}, {"t": 1}, {"t": 2}]
    opts = {
        "steps": [
            {"type": "A", "pass": True},
            {"type": "B", "pass": True},
            {"type": "C", "pass": True},
        ],
        "dispatch": _dispatch,
    }
    assert check_ordered_step(states, opts) is True


# 2. A step never passes -> False (but earlier ones pass)
def test_later_step_fails():
    states = [{"t": 0}, {"t": 1}, {"t": 2}]
    opts = {
        "steps": [
            {"type": "A", "pass": True},
            {"type": "B", "pass": False},  # never satisfied
            {"type": "C", "pass": True},
        ],
        "dispatch": _dispatch,
    }
    assert check_ordered_step(states, opts) is False


# 3. Steps pass on different frames -> still True (in order)
def test_steps_pass_on_different_frames():
    # Frame 0 satisfies step A, frame 1 satisfies B, frame 2 satisfies C.
    # The dispatcher here keys off the frame index via state["t"].
    def dispatch(state, cfg):
        t = state["t"]
        if cfg["type"] == "A":
            return t >= 0
        if cfg["type"] == "B":
            return t >= 1
        if cfg["type"] == "C":
            return t >= 2
        return False

    states = [{"t": 0}, {"t": 1}, {"t": 2}]
    opts = {
        "steps": [{"type": "A"}, {"type": "B"}, {"type": "C"}],
        "dispatch": dispatch,
    }
    assert check_ordered_step(states, opts) is True


# 4. Order matters: A must be satisfied before B advances
def test_order_matters():
    # B is satisfiable from frame 0, but A only becomes true at frame 2.
    # Strict ordering means we wait for A first; since A never passes until
    # frame 2 and there are only 3 frames, B may not have time. Let's use
    # enough frames so B can pass after A.
    def dispatch(state, cfg):
        t = state["t"]
        if cfg["type"] == "A":
            return t >= 2  # A satisfied at t=2
        if cfg["type"] == "B":
            return t >= 3  # B satisfied at t=3
        return False

    # Only 2 frames (t=0,1): A never satisfied -> False
    states = [{"t": 0}, {"t": 1}]
    opts = {
        "steps": [{"type": "A"}, {"type": "B"}],
        "dispatch": dispatch,
    }
    assert check_ordered_step(states, opts) is False

    # 4 frames (t=0..3): A at t=2, B at t=3 -> True
    states2 = [{"t": 0}, {"t": 1}, {"t": 2}, {"t": 3}]
    opts2 = {
        "steps": [{"type": "A"}, {"type": "B"}],
        "dispatch": dispatch,
    }
    assert check_ordered_step(states2, opts2) is True


# 5. Empty steps -> False
def test_empty_steps_returns_false():
    assert check_ordered_step([{"t": 0}], {"steps": [], "dispatch": _dispatch}) is False
    assert check_ordered_step([{"t": 0}], {"dispatch": _dispatch}) is False


# 6. Missing dispatcher -> False (graceful)
def test_missing_dispatcher_returns_false():
    opts = {"steps": [{"type": "A"}]}
    assert check_ordered_step([{"t": 0}], opts) is False


# 7. No frames -> False (can't complete any step)
def test_no_frames_returns_false():
    opts = {"steps": [{"type": "A", "pass": True}], "dispatch": _dispatch}
    assert check_ordered_step([], opts) is False
