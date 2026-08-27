"""
Position delta checker for offline trajectory validation.

Python counterpart to
``AxisWebInfra/src/components/mujoco-framework-next/checkers/PositionDeltaChecker.js``.

Must stay in sync with the JS implementation.

Public API
----------

``check_position_delta(state, opts) -> bool``

Where ``state`` is a single trajectory step dict
(``{"object_positions": {...}, ...}``) and ``opts`` matches the Hub task
JSON schema:

    {
        "sampleBodyName": "akita_black_bowl_1_main",
        "axes": ["x", "y"],
        "minDeltaX": 0.018,        # body must have moved +18 mm on X
        "maxDeltaX": -0.018,       # OR -18 mm on X
        "minDeltaY": 0.018,
        "maxDeltaY": -0.018,
        "captureRuntimeInitial": True,  # default False
        "initialPosition": [x, y, z]    # optional explicit baseline
    }

The Python side is **stateless per call** (unlike the JS side, which keeps
a runtime baseline).  The caller is responsible for passing the right
``initialPosition`` (typically the first step's position in the
trajectory) when ``captureRuntimeInitial=False``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def _get_body_position(state: Dict, body_name: str) -> Optional[Tuple[float, float, float]]:
    positions = state.get("object_positions") or {}
    for key, value in positions.items():
        if key == body_name or key.endswith("/" + body_name) or key.endswith(body_name):
            if isinstance(value, (list, tuple)) and len(value) >= 3:
                try:
                    return (float(value[0]), float(value[1]), float(value[2]))
                except (TypeError, ValueError):
                    return None
    return None


def _num(*candidates) -> Optional[float]:
    for v in candidates:
        if v is None:
            continue
        try:
            n = float(v)
        except (TypeError, ValueError):
            continue
        if n == n:  # not NaN
            return n
    return None


def check_position_delta(state: Dict, opts: Dict) -> bool:
    """Return ``True`` if the named body's position has moved past one of
    the configured per-axis bounds relative to the baseline.
    """
    body = opts.get("sample_body") or opts.get("sampleBodyName") or opts.get("sample_body_name")
    if not body:
        return False

    cur = _get_body_position(state, body)
    if cur is None:
        return False

    axes = opts.get("axes", ["x", "y"])
    if not isinstance(axes, list) or len(axes) == 0:
        return False
    axes = [str(a).lower() for a in axes if str(a).lower() in _AXIS_INDEX]
    if not axes:
        return False

    # Baseline: explicit initialPosition > captureRuntimeInitial (caller-supplied
    # baseline) > state["initial_position"] (per-state) > current position
    # (degenerate: delta is 0; check will only pass if a bound is 0).
    baseline = None
    init = opts.get("initialPosition")
    if isinstance(init, (list, tuple)) and len(init) >= 3:
        try:
            baseline = (float(init[0]), float(init[1]), float(init[2]))
        except (TypeError, ValueError):
            baseline = None
    if baseline is None and "initial_position" in state:
        ip = state["initial_position"]
        if isinstance(ip, dict):
            for k in ("x", "y", "z"):
                ip.setdefault(k, 0.0)
            try:
                baseline = (float(ip["x"]), float(ip["y"]), float(ip["z"]))
            except (TypeError, ValueError, KeyError):
                baseline = None
        elif isinstance(ip, (list, tuple)) and len(ip) >= 3:
            try:
                baseline = (float(ip[0]), float(ip[1]), float(ip[2]))
            except (TypeError, ValueError):
                baseline = None
    if baseline is None:
        # No baseline available. This is a stateless, per-call function, so it
        # cannot "capture the first frame" the way the stateful JS checker does
        # with `captureRuntimeInitial`.  Conservatively report no movement
        # (delta = 0) rather than guessing a baseline that could produce a
        # false positive.  Callers doing offline validation MUST pass the
        # trajectory's initial position via `initialPosition` (or a per-step
        # `state["initial_position"]`) — exactly the baseline the JS side would
        # have captured as its runtime initial.
        return False

    for ax in axes:
        idx = _AXIS_INDEX[ax]
        delta = cur[idx] - baseline[idx]
        hi = _num(opts.get("min_delta_" + ax), opts.get("minDelta" + ax.upper()))
        lo = _num(opts.get("max_delta_" + ax), opts.get("maxDelta" + ax.upper()))
        if hi is not None and delta >= hi:
            return True
        if lo is not None and delta <= lo:
            return True
    return False
