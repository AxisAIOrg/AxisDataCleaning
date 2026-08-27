"""
Ordered step checker for offline trajectory validation.

Python counterpart to
``AxisWebInfra/src/components/mujoco-framework-next/checkers/OrderedStepChecker.js``.

Must stay in sync with the JS implementation.

Public API
----------

``check_ordered_step(steps: List[Dict], opts: Dict) -> bool``

Unlike the other Python checkers which are stateless per call, this one is
inherently stateful across a trajectory: a step is "completed" at some frame,
and later frames only need to satisfy the *next* unfinished step.

The function accepts a **list of state steps** (the whole trajectory) and
walks it once, maintaining the current-step cursor, exactly like the JS
stateful checker does across `check()` calls.

``opts`` matches the Hub task JSON schema:

    {
        "steps": [
            {"type": "RelativePositionBoundsChecker", ...},
            {"type": "GripperOpenChecker", ...}
        ],
        "allow_reentry": false
    }

Returns ``True`` iff, after walking all frames, every step has been satisfied
in order.
"""

from __future__ import annotations

from typing import Any, Dict, List


def check_ordered_step(steps: List[Dict], opts: Dict) -> bool:
    """Return True iff all sub-steps are satisfied in order across the
    given trajectory ``steps`` (a list of state dicts).
    """
    sub_configs = opts.get("steps") or opts.get("checkers") or []
    if not isinstance(sub_configs, list) or len(sub_configs) == 0:
        return False

    # The dispatcher is provided by the caller via opts["dispatch"] so that
    # we don't hard-code every checker here and keep it decoupled from the
    # concrete check_* implementations in validate_offline_trajectory.py.
    dispatch = opts.get("dispatch")
    if not callable(dispatch):
        # No dispatcher provided; we cannot evaluate sub-steps.
        return False

    current = 0
    n = len(sub_configs)
    for state in steps:
        if current >= n:
            break
        cfg = sub_configs[current]
        if dispatch(state, cfg):
            current += 1
    return current >= n
