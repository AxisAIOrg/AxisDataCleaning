"""
Body contact checker for offline trajectory validation.

This is the Python counterpart to
``AxisWebInfra/src/components/mujoco-framework-next/checkers/BodyContactChecker.js``
and MUST stay in sync with it.

The two implementations are contractually linked: the JS side runs in the
browser during teleoperation, the Python side runs offline on saved
trajectories, and they MUST agree on the success condition.  Any future
change to the algorithm must be made in both repos.

Public API
----------

``check_body_contact(state, opts) -> bool``

Where ``state`` is a single trajectory step dict of the form
``{"object_positions": {...}, "object_orientations": {...}, ...}``
and ``opts`` is a dict with the same keys as the Hub task JSON:

    {
        "bodyAName": "akita_black_bowl_1_main",
        "bodyBName": "plate_1_main",
        "maxContactDistance": 0.02,    # meters; default 0
        "minContactCount": 1            # default 1
    }

Because the Python side doesn't have the live MuJoCo contact array, we
approximate contact using a per-body axis-aligned bounding box (AABB)
intersection test on the body's position.  This is a deliberate trade-off:

  - It is cheap (no MuJoCo needed; runs on any saved trajectory).
  - It is conservative: it may say "in contact" when the actual MuJoCo
    contact would say "not in contact" (because AABBs over-approximate the
    body shape), but never the other way around.  This is the safe direction
    for a "is the task done" check.
  - It supports ``maxContactDistance > 0`` for soft-contact semantics by
    inflating the AABBs by the given distance before testing.

If you need true geometric contact in offline analysis, see
``AxisDataCleaning/util/contact_util.py`` (a future work item) which will
load the MuJoCo model and replay contacts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# Default AABB extents (half-sizes) for objects that don't have a registered
# extent.  Tuned for LIBERO-style tabletop manipulation: most objects fit in
# a 5 cm cube.  Override per-call via opts["default_half_extent"].
DEFAULT_HALF_EXTENT = 0.025


def _aabb_for_body(
    state: Dict,
    body_name: str,
    default_half_extent: float,
) -> Optional[Tuple[float, float, float, float, float, float]]:
    """Return the axis-aligned bounding box ``(xmin, ymin, zmin, xmax, ymax, zmax)``
    for ``body_name`` in this step, or ``None`` if the body is missing.
    """
    positions = state.get("object_positions") or {}
    pos = None
    for key, value in positions.items():
        if key == body_name or key.endswith("/" + body_name) or key.endswith(body_name):
            pos = value
            break
    if pos is None or not isinstance(pos, (list, tuple)) or len(pos) < 3:
        return None
    try:
        x = float(pos[0])
        y = float(pos[1])
        z = float(pos[2])
    except (TypeError, ValueError):
        return None
    # If the state has explicit extents for this body, prefer them.
    extents = state.get("object_extents") or {}
    hx, hy, hz = default_half_extent, default_half_extent, default_half_extent
    for key, value in extents.items():
        if key == body_name or key.endswith("/" + body_name) or key.endswith(body_name):
            if isinstance(value, (list, tuple)) and len(value) >= 3:
                hx, hy, hz = (abs(float(value[0])), abs(float(value[1])), abs(float(value[2])))
            break
    return (x - hx, y - hy, z - hz, x + hx, y + hy, z + hz)


def _aabb_intersect_with_distance(
    a: Tuple[float, float, float, float, float, float],
    b: Tuple[float, float, float, float, float, float],
    max_distance: float,
) -> bool:
    """``True`` iff the two AABBs intersect, or the closest-point distance
    between them is within ``max_distance``.  Zero-distance is the standard
    "AABB intersection" test.
    """
    a_min = (a[0] - max_distance, a[1] - max_distance, a[2] - max_distance)
    a_max = (a[3] + max_distance, a[4] + max_distance, a[5] + max_distance)
    # AABB intersection with inflation == closest-point distance <= max_distance.
    if a_max[0] < b[0] or a_min[0] > b[3]:
        return False
    if a_max[1] < b[1] or a_min[1] > b[4]:
        return False
    if a_max[2] < b[2] or a_min[2] > b[5]:
        return False
    return True


def check_body_contact(state: Dict, opts: Dict) -> bool:
    """Return ``True`` if ``body_a`` and ``body_b`` are in (or near) contact.

    Mirrors the JS ``BodyContactChecker.check()`` semantics:
      - Reads the two body names from ``opts["body_a"]`` / ``opts["bodyAName"]``
        and ``opts["body_b"]`` / ``opts["bodyBName"]``.
      - Reads ``opts["max_contact_distance"]`` / ``opts["maxContactDistance"`` (default 0).
      - Reads ``opts["min_contact_count"]`` / ``opts["minContactCount"]`` (default 1).
      - Returns ``True`` iff at least ``min_contact_count`` AABB intersection
        (or near-intersection within ``max_contact_distance``) is observed in
        the current ``state``.  With one AABB per body, this is equivalent
        to "the two bodies overlap".
    """
    body_a = opts.get("body_a") or opts.get("bodyAName") or opts.get("body_a_name")
    body_b = opts.get("body_b") or opts.get("bodyBName") or opts.get("body_b_name")
    if not body_a or not body_b:
        return False
    if body_a == body_b:
        return False

    max_distance = float(opts.get("max_contact_distance", opts.get("maxContactDistance", 0.0)) or 0.0)
    if max_distance < 0:
        return False

    min_count = int(opts.get("min_contact_count", opts.get("minContactCount", 1)) or 1)
    if min_count < 1:
        min_count = 1

    default_half_extent = float(opts.get("default_half_extent", DEFAULT_HALF_EXTENT))

    aabb_a = _aabb_for_body(state, body_a, default_half_extent)
    aabb_b = _aabb_for_body(state, body_b, default_half_extent)
    if aabb_a is None or aabb_b is None:
        return False

    return _aabb_intersect_with_distance(aabb_a, aabb_b, max_distance) and min_count <= 1
