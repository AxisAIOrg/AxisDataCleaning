#!/usr/bin/env python3
"""
离线轨迹 Checker 校验脚本（独立于前端工程）

输入：离线轨迹 JSON 文件（格式见下方）。
从 trajectory_metadata.task_id 自动识别任务，拉取该任务的 checker_config，
用纯 Python 复现前端 checkers 的判定逻辑，输出该条轨迹是否通过 checker。

轨迹 JSON 格式要求：
  - steps: [ { state: { robot_joints, object_positions, object_orientations }, ... } ]
  - trajectory_metadata: { task_id: number, goal_achieved?: boolean, ... }

用法：
  # 从后端 API 拉取任务配置（需指定 API 根地址）
  python validate_offline_trajectory.py trajectory.json --api-base http://localhost:8000

  # 从本地任务配置目录加载（目录下 task_1.json, task_2.json 等，或 task_id 与文件名对应）
  python validate_offline_trajectory.py trajectory.json --tasks-dir ./backend/task_configs

  # 从单个任务 JSON 文件加载（文件内需含 checker_config）
  python validate_offline_trajectory.py trajectory.json --task-config ./task_57.json
"""

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple


# ---------- 从 state 中按名称取位姿/关节 ----------

def _normalize_name(name: str) -> str:
    return (name or "").strip().rstrip("/")


def _find_state_key(name: str, keys: List[str]) -> Optional[str]:
    """在 state 的 key 列表中匹配 body/joint 名称。"""
    if not name or not keys:
        return None
    n = _normalize_name(name)
    for k in keys:
        kn = _normalize_name(k)
        if k == name or kn == n or k.endswith("/" + name) or k.endswith("/" + n):
            return k
    return None


def get_body_position(state: Dict, body_name: str) -> Optional[Tuple[float, float, float]]:
    pos = state.get("object_positions") or {}
    key = _find_state_key(body_name, list(pos.keys()))
    if key is None:
        return None
    v = pos[key]
    if isinstance(v, (list, tuple)) and len(v) >= 3:
        return (float(v[0]), float(v[1]), float(v[2]))
    return None


def get_body_orientation(state: Dict, body_name: str) -> Optional[Tuple[float, float, float, float]]:
    ori = state.get("object_orientations") or {}
    key = _find_state_key(body_name, list(ori.keys()))
    if key is None:
        return None
    q = ori[key]
    if not isinstance(q, (list, tuple)) or len(q) < 4:
        return None
    # 支持 [w,x,y,z] 或 [x,y,z,w]
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    if abs(q[3] - 1.0) < 0.5 and abs(q[0]) < 0.5:
        w, x, y, z = float(q[3]), float(q[0]), float(q[1]), float(q[2])
    norm = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    return (w / norm, x / norm, y / norm, z / norm)


def get_joint_value(state: Dict, joint_name: str) -> Optional[float]:
    # 1) 先查 robot_joints（机械臂关节）
    joints = state.get("robot_joints") or {}
    key = _find_state_key(joint_name, list(joints.keys()))
    if key is not None:
        v = joints[key]
        try:
            return float(v)
        except (TypeError, ValueError):
            pass
    # 2) 再查 object_positions（物体/铰链关节等可能存在于此：单值 [angle] 或 [x,y,z] 取第一分量）
    positions = state.get("object_positions") or {}
    key = _find_state_key(joint_name, list(positions.keys()))
    if key is not None:
        v = positions[key]
        if isinstance(v, (list, tuple)):
            if len(v) == 1:
                try:
                    return float(v[0])
                except (TypeError, ValueError):
                    pass
            if len(v) >= 1:
                try:
                    return float(v[0])
                except (TypeError, ValueError):
                    pass
        try:
            return float(v)
        except (TypeError, ValueError):
            pass
    return None


def quat_apply_up(w: float, x: float, y: float, z: float) -> Tuple[float, float, float]:
    """四元数 (w,x,y,z) 作用在 (0,0,1) 上的结果（单位向量）。"""
    # v' = v + 2*cross(r, cross(r,v) + w*v), r=(x,y,z), v=(0,0,1)
    # cross(r,v) = (-y, x, 0); cross(r,v)+w*v = (-y, x, w)
    # cross(r, (-y,x,w)) = (x*w - z*x, y*w - z*(-y), x*(-y) - y*x) = (w*x-x*z, w*y+y*z, -x*y-y*x)
    # 2*cross = 2*(w*x-x*z, w*y+y*z, -2*x*y). 所以 v' = (2*(w*x-x*z), 2*(w*y+y*z), 1 - 2*(x*x+y*y))
    ux = 2.0 * (w * x - z * x)
    uy = 2.0 * (w * y + z * y)
    uz = 1.0 - 2.0 * (x * x + y * y)
    n = math.sqrt(ux * ux + uy * uy + uz * uz) or 1.0
    return (ux / n, uy / n, uz / n)


def quat_to_matrix(w: float, x: float, y: float, z: float) -> List[float]:
    """四元数 (w,x,y,z) → 3x3 旋转矩阵（行主序，长度 9）。"""
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return [
        1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy),
        2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx),
        2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy),
    ]


def mat_vec_mul(R: List[float], v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """3x3 矩阵乘列向量 v。"""
    return (
        R[0] * v[0] + R[1] * v[1] + R[2] * v[2],
        R[3] * v[0] + R[4] * v[1] + R[5] * v[2],
        R[6] * v[0] + R[7] * v[1] + R[8] * v[2],
    )


def mat_vec_mul_T(R: List[float], v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """3x3 矩阵的转置乘列向量 v。"""
    return (
        R[0] * v[0] + R[3] * v[1] + R[6] * v[2],
        R[1] * v[0] + R[4] * v[1] + R[7] * v[2],
        R[2] * v[0] + R[5] * v[1] + R[8] * v[2],
    )


def rotation_angle_degrees(
    current_quat: Tuple[float, float, float, float],
    initial_quat: Tuple[float, float, float, float],
) -> float:
    """
    两个四元数之间的整体旋转角度（0~180 度）。
    q_rel = current * initial^{-1}, angle = 2 * acos(|w_rel|)。
    """
    cw, cx, cy, cz = current_quat
    iw, ix, iy, iz = initial_quat
    # 先算 initial 的共轭（假设已归一化）
    iw_c, ix_c, iy_c, iz_c = iw, -ix, -iy, -iz
    # q_rel = current * initial^{-1}
    rw = cw * iw_c - cx * ix_c - cy * iy_c - cz * iz_c
    rx = cw * ix_c + cx * iw_c + cy * iz_c - cz * iy_c
    ry = cw * iy_c - cx * iz_c + cy * iw_c + cz * ix_c
    rz = cw * iz_c + cx * iy_c - cy * ix_c + cz * iw_c
    # 角度
    w_abs = max(-1.0, min(1.0, abs(rw)))
    angle_rad = 2.0 * math.acos(w_abs)
    return math.degrees(angle_rad)


def tilt_angle_degrees(
    current_quat: Tuple[float, float, float, float],
    initial_quat: Tuple[float, float, float, float],
) -> float:
    """仅考虑倾斜（up 向量夹角），单位：度。"""
    cu = quat_apply_up(*current_quat)
    iu = quat_apply_up(*initial_quat)
    dot = max(-1.0, min(1.0, cu[0] * iu[0] + cu[1] * iu[1] + cu[2] * iu[2]))
    return math.degrees(math.acos(dot))


# ---------- Checker 实现（仅依赖 state 字典） ----------


def check_relative_position_bounds(state: Dict, opts: Dict) -> bool:
    obj_name = opts.get("obj_name") or opts.get("objName") or ""
    ref_name = opts.get("ref_name") or opts.get("refName") or ""
    obj_pos = get_body_position(state, obj_name)
    ref_pos = get_body_position(state, ref_name)
    if obj_pos is None or ref_pos is None:
        return False
    dx = obj_pos[0] - ref_pos[0]
    dy = obj_pos[1] - ref_pos[1]
    dz = obj_pos[2] - ref_pos[2]
    def in_range(val: float, r: Any) -> bool:
        if not isinstance(r, (list, tuple)) or len(r) < 2:
            return True
        return r[0] < val < r[1]
    checks = []
    if opts.get("x_range") or opts.get("xRange"):
        checks.append(in_range(dx, opts.get("x_range") or opts.get("xRange")))
    if opts.get("y_range") or opts.get("yRange"):
        checks.append(in_range(dy, opts.get("y_range") or opts.get("yRange")))
    if opts.get("z_range") or opts.get("zRange"):
        checks.append(in_range(dz, opts.get("z_range") or opts.get("zRange")))
    return bool(checks) and all(checks)


def check_sample_rotation(state: Dict, opts: Dict) -> bool:
    body_name = opts.get("sampleBodyName") or "sample/"
    threshold = float(opts.get("tipAngleThreshold") or 30.0)
    initial_rot = opts.get("initialRotation") or [0.0, 0.0, 0.0, 1.0]
    if len(initial_rot) < 4:
        initial_rot = [0.0, 0.0, 0.0, 1.0]
    # 支持 wxyz 或 xyzw
    iq = (float(initial_rot[0]), float(initial_rot[1]), float(initial_rot[2]), float(initial_rot[3]))
    if abs(initial_rot[3] - 1.0) < 0.5 and abs(initial_rot[0]) < 0.5:
        iq = (float(initial_rot[3]), float(initial_rot[0]), float(initial_rot[1]), float(initial_rot[2]))
    norm_i = math.sqrt(iq[0]**2 + iq[1]**2 + iq[2]**2 + iq[3]**2) or 1.0
    iq = (iq[0] / norm_i, iq[1] / norm_i, iq[2] / norm_i, iq[3] / norm_i)
    current = get_body_orientation(state, body_name)
    if current is None:
        return False
    angle = tilt_angle_degrees(current, iq)
    return angle >= threshold


def check_sample_position_delta(state: Dict, opts: Dict) -> bool:
    body_name = opts.get("sampleBodyName") or "sample/"
    ip = opts.get("initialPosition") or [0, 0, 0]
    initial = (float(ip[0]), float(ip[1]), float(ip[2]))
    pos = get_body_position(state, body_name)
    if pos is None:
        return False
    dx = pos[0] - initial[0]
    dy = pos[1] - initial[1]
    dz = pos[2] - initial[2]
    axes = opts.get("axes")
    if isinstance(axes, str):
        axes = [a.strip().lower() for a in axes.split(",") if a.strip()]
    if not isinstance(axes, list) or not axes:
        axes = ["x", "z"]
    min_dx = opts.get("minDeltaX")
    min_dy = opts.get("minDeltaY")
    min_dz = opts.get("minDeltaZ")
    max_dx = opts.get("maxDeltaX")
    max_dy = opts.get("maxDeltaY")
    max_dz = opts.get("maxDeltaZ")
    def ok(axis: str, val: float) -> bool:
        min_v = {"x": min_dx, "y": min_dy, "z": min_dz}.get(axis)
        max_v = {"x": max_dx, "y": max_dy, "z": max_dz}.get(axis)
        if min_v is not None and val < float(min_v):
            return False
        if max_v is not None and val > float(max_v):
            return False
        return True
    return all(ok(a, {"x": dx, "y": dy, "z": dz}[a]) for a in axes)


def check_position_delta(state: Dict, opts: Dict) -> bool:
    """
    PositionDeltaChecker: 与 SamplePositionDeltaChecker 几乎相同，只是名字不同，照搬逻辑。
    """
    body_name = opts.get("sampleBodyName") or "sample/"
    ip = opts.get("initialPosition") or [0, 0, 0]
    initial = (float(ip[0]), float(ip[1]), float(ip[2]))
    pos = get_body_position(state, body_name)
    if pos is None:
        return False
    dx = pos[0] - initial[0]
    dy = pos[1] - initial[1]
    dz = pos[2] - initial[2]
    axes = opts.get("axes")
    if isinstance(axes, str):
        axes = [a.strip().lower() for a in axes.split(",") if a.strip()]
    if not isinstance(axes, list) or not axes:
        axes = ["x", "z"]
    min_dx = opts.get("minDeltaX")
    min_dy = opts.get("minDeltaY")
    min_dz = opts.get("minDeltaZ")
    max_dx = opts.get("maxDeltaX")
    max_dy = opts.get("maxDeltaY")
    max_dz = opts.get("maxDeltaZ")
    def ok(axis: str, val: float) -> bool:
        min_v = {"x": min_dx, "y": min_dy, "z": min_dz}.get(axis)
        max_v = {"x": max_dx, "y": max_dy, "z": max_dz}.get(axis)
        if min_v is not None and val < float(min_v):
            return False
        if max_v is not None and val > float(max_v):
            return False
        return True
    return all(ok(a, {"x": dx, "y": dy, "z": dz}[a]) for a in axes)


def check_bowl_position(state: Dict, opts: Dict) -> bool:
    body_name = opts.get("bowlBodyName") or "akita_black_bowl/"
    pos = get_body_position(state, body_name)
    if pos is None:
        return False
    min_b = opts.get("minBounds")
    max_b = opts.get("maxBounds")
    if min_b and max_b and len(min_b) >= 3 and len(max_b) >= 3:
        return (
            min_b[0] <= pos[0] <= max_b[0]
            and min_b[1] <= pos[1] <= max_b[1]
            and min_b[2] <= pos[2] <= max_b[2]
        )
    th = opts.get("positionThreshold")
    if th is not None:
        dist = math.sqrt(pos[0] ** 2 + pos[1] ** 2 + pos[2] ** 2)
        return dist <= float(th)
    return True


def check_bowl_in_drawer(state: Dict, opts: Dict) -> bool:
    """
    BowlInDrawerChecker：近似复现 JS 逻辑，用 state 中的碗/柜体/抽屉位置与抽屉关节位移计算碗是否在抽屉盒子内。
    """
    bowl_name = opts.get("bowlBodyName") or "akita_black_bowl/"
    drawer_body_name = opts.get("drawerBodyName") or "wooden_cabinet/cabinet_top"
    drawer_joint_name = opts.get("drawerJointName") or "wooden_cabinet/top_level"
    cabinet_base_name = opts.get("cabinetBaseBodyName") or "wooden_cabinet/"

    bowl_pos = get_body_position(state, bowl_name)
    if bowl_pos is None:
        return False

    cab_base = get_body_position(state, cabinet_base_name) or (0.5, -0.3, 0.0)
    drawer_body = get_body_position(state, drawer_body_name) or (0.0, 0.0, 0.0)
    joint = get_joint_value(state, drawer_joint_name) or 0.0

    # JS 中的常数（region center 与 half-size）直接搬过来
    region_center_local = (0.00328, 0.01128, 0.18563)
    half_size = (0.02993, 0.07561, 0.10224)

    # regionCenter = cabinetBasePos + drawerBodyPos + jointOffsetY + region_center_local
    region_x = cab_base[0] + drawer_body[0] + region_center_local[0]
    region_y = cab_base[1] + drawer_body[1] + joint + region_center_local[1]
    region_z = cab_base[2] + drawer_body[2] + region_center_local[2]

    diff_x = bowl_pos[0] - region_x
    diff_y = bowl_pos[1] - region_y
    diff_z = bowl_pos[2] - region_z

    return (
        abs(diff_x) <= half_size[0]
        and abs(diff_y) <= half_size[1]
        and abs(diff_z) <= half_size[2]
    )


def check_drawer_position(state: Dict, opts: Dict) -> bool:
    threshold = float(opts.get("threshold") or -0.1)
    names = opts.get("drawerJointNames") or [
        "wooden_cabinet/top_level",
        "wooden_cabinet/middle_level",
        "wooden_cabinet/bottom_level",
    ]
    for jname in names:
        v = get_joint_value(state, jname)
        if v is not None and v < threshold:
            return True
    return False


def check_drawer_bbox(state: Dict, opts: Dict) -> bool:
    """
    DrawerBBoxChecker：判断物体是否在“随抽屉移动的盒子”内。
    近似复现 JS：使用柜体朝向 + base_offset + joint 位移 + relative_quat 生成世界坐标系下的盒子，
    再看 obj 是否落在 half_size 范围内。
    """
    obj_name = opts.get("obj_name") or opts.get("objName") or ""
    cabinet_name = opts.get("cabinet_name") or opts.get("cabinetName") or ""
    joint_name = opts.get("joint_name") or opts.get("jointName") or None
    base_offset = opts.get("base_offset") or opts.get("baseOffset") or [0.0, 0.0, 0.0]
    disp_axis = int(opts.get("displacement_axis") or opts.get("displacementAxis") or 1)
    rel_quat = opts.get("relative_quat") or opts.get("relativeQuat") or [1.0, 0.0, 0.0, 0.0]  # w,x,y,z
    half_size = opts.get("half_size") or opts.get("halfSize") or [0.0, 0.0, 0.0]

    obj_pos = get_body_position(state, obj_name)
    cab_pos = get_body_position(state, cabinet_name)
    if obj_pos is None or cab_pos is None:
        return False

    cab_ori = get_body_orientation(state, cabinet_name)
    if cab_ori is None:
        # 没有朝向时，只能在世界系下用 axis-aligned 盒子近似
        bx = base_offset[0]
        by = base_offset[1]
        bz = base_offset[2]
        joint = get_joint_value(state, joint_name) or 0.0
        if 0 <= disp_axis < 3:
            if disp_axis == 0:
                bx += joint
            elif disp_axis == 1:
                by += joint
            else:
                bz += joint
        center = (cab_pos[0] + bx, cab_pos[1] + by, cab_pos[2] + bz)
        diff = (obj_pos[0] - center[0], obj_pos[1] - center[1], obj_pos[2] - center[2])
        return (
            abs(diff[0]) <= half_size[0] + 1e-6
            and abs(diff[1]) <= half_size[1] + 1e-6
            and abs(diff[2]) <= half_size[2] + 1e-6
        )

    # 构造世界变换：worldR = cabR * relR, worldT = cabR * base + cabPos
    cab_R = quat_to_matrix(*cab_ori)
    bx = base_offset[0]
    by = base_offset[1]
    bz = base_offset[2]
    joint = get_joint_value(state, joint_name) or 0.0
    if 0 <= disp_axis < 3:
        if disp_axis == 0:
            bx += joint
        elif disp_axis == 1:
            by += joint
        else:
            bz += joint
    base_local = (bx, by, bz)
    rel_R = quat_to_matrix(float(rel_quat[0]), float(rel_quat[1]), float(rel_quat[2]), float(rel_quat[3]))
    # worldR = cab_R * rel_R
    worldR = [0.0] * 9
    a = cab_R
    b = rel_R
    worldR[0] = a[0] * b[0] + a[1] * b[3] + a[2] * b[6]
    worldR[1] = a[0] * b[1] + a[1] * b[4] + a[2] * b[7]
    worldR[2] = a[0] * b[2] + a[1] * b[5] + a[2] * b[8]
    worldR[3] = a[3] * b[0] + a[4] * b[3] + a[5] * b[6]
    worldR[4] = a[3] * b[1] + a[4] * b[4] + a[5] * b[7]
    worldR[5] = a[3] * b[2] + a[4] * b[5] + a[5] * b[8]
    worldR[6] = a[6] * b[0] + a[7] * b[3] + a[8] * b[6]
    worldR[7] = a[6] * b[1] + a[7] * b[4] + a[8] * b[7]
    worldR[8] = a[6] * b[2] + a[7] * b[5] + a[8] * b[8]

    worldT = mat_vec_mul(cab_R, base_local)
    worldT = (worldT[0] + cab_pos[0], worldT[1] + cab_pos[1], worldT[2] + cab_pos[2])

    rel = (obj_pos[0] - worldT[0], obj_pos[1] - worldT[1], obj_pos[2] - worldT[2])
    local = mat_vec_mul_T(worldR, rel)
    return (
        abs(local[0]) <= half_size[0] + 1e-6
        and abs(local[1]) <= half_size[1] + 1e-6
        and abs(local[2]) <= half_size[2] + 1e-6
    )


def check_gripper_open(state: Dict, opts: Dict) -> bool:
    threshold = float(opts.get("threshold") or 0.2)
    names = opts.get("fingerJointNames") or opts.get("gripperJointNames") or []
    for jname in names:
        v = get_joint_value(state, jname)
        if v is not None and v > threshold:
            return True
    return False


def check_relative_axis_range(state: Dict, opts: Dict) -> bool:
    """
    RelativeAxisRangeChecker：检查 target 在某个轴上的坐标，是否落在 reference 同轴坐标的 [ref + minOffset, ref + maxOffset] 区间。
    """
    target_name = opts.get("targetBodyName") or "sample/"
    ref_name = opts.get("referenceBodyName") or "target/"
    axis = str(opts.get("axis") or "y").lower()
    axis_idx = {"x": 0, "y": 1, "z": 2}.get(axis)
    if axis_idx is None:
        return False
    min_offset = opts.get("minOffset")
    max_offset = opts.get("maxOffset")
    min_off = float(min_offset) if min_offset is not None else float("-inf")
    max_off = float(max_offset) if max_offset is not None else float("inf")

    tpos = get_body_position(state, target_name)
    rpos = get_body_position(state, ref_name)
    if tpos is None or rpos is None:
        return False
    delta = tpos[axis_idx] - rpos[axis_idx]
    return delta >= min_off and delta <= max_off


def check_joint_threshold(state: Dict, opts: Dict) -> bool:
    joint_name = opts.get("joint_name") or opts.get("jointName")
    obj_name = (opts.get("obj_name") or opts.get("objName") or "").strip().rstrip("/")
    if joint_name and "/" not in joint_name and obj_name:
        joint_name = f"{obj_name}/{joint_name}"
    v = get_joint_value(state, joint_name or "")
    if v is None:
        return False
    th = float(opts.get("threshold") or 0.0)
    mode = (opts.get("mode") or "ge").lower()
    if mode == "gt":
        return v > th
    if mode == "ge":
        return v >= th
    if mode == "lt":
        return v < th
    if mode == "le":
        return v <= th
    return False


def check_box_joint_position(state: Dict, opts: Dict) -> bool:
    joint_name = opts.get("jointName") or "box_base/box_joint"
    threshold = float(opts.get("threshold") or (-14 * math.pi / 180))
    v = get_joint_value(state, joint_name)
    if v is None:
        return False
    return v < threshold


def check_relative_cylinder(state: Dict, opts: Dict) -> bool:
    """
    RelativeCylinderChecker：判断 obj 是否在 ref 周围的圆柱体内（基于 XY 距离和高度区间）。
    """
    obj_name = opts.get("obj_name") or opts.get("objName") or ""
    ref_name = opts.get("ref_name") or opts.get("refName") or ""
    ref_type = (opts.get("ref_type") or opts.get("refType") or "object").lower()
    xy_radius = float(opts.get("xy_radius") or opts.get("xyRadius") or 0.06)
    h_min = float(opts.get("height_min") or opts.get("heightMin") or 0.0)
    h_max = float(opts.get("height_max") or opts.get("heightMax") or 0.03)

    obj_pos = get_body_position(state, obj_name)
    if obj_pos is None:
        return False

    # 离线 JSON 一般没有 site，假设 refType=object 时用 body；refType=site 时尝试用同名 key。
    if ref_type == "site":
        ref_pos = get_body_position(state, ref_name)
    else:
        ref_pos = get_body_position(state, ref_name)
    if ref_pos is None:
        return False

    dx = obj_pos[0] - ref_pos[0]
    dy = obj_pos[1] - ref_pos[1]
    dz = obj_pos[2] - ref_pos[2]
    xy_dist = math.sqrt(dx * dx + dy * dy)
    xy_ok = xy_dist < xy_radius
    h_ok = dz > h_min and dz < h_max
    return xy_ok and h_ok


def check_frame_bbox(state: Dict, opts: Dict) -> bool:
    """
    FrameBBoxChecker：判断 obj 是否落在某个“局部坐标系的包围盒”内。
    使用 frame 的位置 + 朝向（来自 object_orientations），在其局部坐标下检查 [lower, upper]。
    """
    obj_name = opts.get("obj_name") or opts.get("objName") or ""
    frame_name = opts.get("frame_name") or opts.get("frameName") or ""
    frame_type = (opts.get("frame_type") or opts.get("frameType") or "site").lower()
    lower = opts.get("lower") or [-0.0, -0.0, -0.0]
    upper = opts.get("upper") or [0.0, 0.0, 0.0]

    obj_pos = get_body_position(state, obj_name)
    if obj_pos is None:
        return False

    if frame_type == "site":
        frame_pos = get_body_position(state, frame_name)
        frame_ori = get_body_orientation(state, frame_name)
    else:
        frame_pos = get_body_position(state, frame_name)
        frame_ori = get_body_orientation(state, frame_name)
    if frame_pos is None or frame_ori is None:
        return False

    R = quat_to_matrix(*frame_ori)
    rel = (obj_pos[0] - frame_pos[0], obj_pos[1] - frame_pos[1], obj_pos[2] - frame_pos[2])
    local = mat_vec_mul_T(R, rel)
    return (
        local[0] >= lower[0] and local[0] <= upper[0] and
        local[1] >= lower[1] and local[1] <= upper[1] and
        local[2] >= lower[2] and local[2] <= upper[2]
    )


def check_object_rotation(state: Dict, opts: Dict) -> bool:
    """
    ObjectRotationChecker：整体旋转角度差是否超过阈值（任意朝向，而不是只看倾斜）。
    """
    body_name = opts.get("bodyName") or "sample/"
    threshold = float(opts.get("rotationThresholdDeg") or 30.0)

    current = get_body_orientation(state, body_name)
    if current is None:
        return False

    init_rot = opts.get("initialRotation")
    if init_rot and isinstance(init_rot, (list, tuple)) and len(init_rot) >= 4:
        # 支持 xyzw / wxyz，两种都做归一化
        # 约定：若最后一位接近 1，则认为是 xyzw，否则 wxyz
        if abs(init_rot[-1] - 1.0) < 0.5 and abs(init_rot[0]) < 0.5:
            # xyzw: [x,y,z,w]
            iq = (float(init_rot[3]), float(init_rot[0]), float(init_rot[1]), float(init_rot[2]))
        else:
            # wxyz: [w,x,y,z]
            iq = (float(init_rot[0]), float(init_rot[1]), float(init_rot[2]), float(init_rot[3]))
    else:
        # 如果没给初始旋转，就把“当前”当作初始（这样 angle=0，不会触发成功），更安全的做法是视为失败。
        iq = current

    angle = rotation_angle_degrees(current, iq)
    return angle >= threshold


def check_sample_tilt(state: Dict, opts: Dict) -> bool:
    """
    SampleTiltChecker：与 ObjectRotationChecker 类似，但名字不同，阈值字段不同。
    """
    body_name = opts.get("sampleBodyName") or "sample/"
    threshold = float(opts.get("tiltThresholdDegrees") or 30.0)
    current = get_body_orientation(state, body_name)
    if current is None:
        return False
    init_rot = opts.get("initialRotation")
    if init_rot and isinstance(init_rot, (list, tuple)) and len(init_rot) >= 4:
        if abs(init_rot[-1] - 1.0) < 0.5 and abs(init_rot[0]) < 0.5:
            iq = (float(init_rot[3]), float(init_rot[0]), float(init_rot[1]), float(init_rot[2]))
        else:
            iq = (float(init_rot[0]), float(init_rot[1]), float(init_rot[2]), float(init_rot[3]))
    else:
        # JS 里默认是 identity (w=1,x=y=z=0)
        iq = (1.0, 0.0, 0.0, 0.0)
    angle = rotation_angle_degrees(current, iq)
    return angle >= threshold


def check_composite(state: Dict, opts: Dict) -> bool:
    operator = (opts.get("operator") or "AND").upper()
    sub = opts.get("checkers") or []
    results = [run_checker(state, c) for c in sub]
    if not results:
        return False
    if operator == "OR":
        return any(results)
    return all(results)


def run_checker(state: Dict, config: Dict) -> bool:
    """根据单个 checker 配置对 state 做一次判定。"""
    if not config or not isinstance(config, dict):
        return False
    t = (config.get("type") or "").strip()
    opts = {k: v for k, v in config.items() if k != "type" and k != "enabled"}
    if t == "CompositeChecker" or t == "composite":
        return check_composite(state, opts)
    if t == "RelativePositionBoundsChecker":
        return check_relative_position_bounds(state, opts)
    if t == "SampleRotationChecker":
        return check_sample_rotation(state, opts)
    if t == "SamplePositionDeltaChecker":
        return check_sample_position_delta(state, opts)
    if t == "PositionDeltaChecker":
        return check_position_delta(state, opts)
    if t == "BowlPositionChecker":
        return check_bowl_position(state, opts)
    if t == "BowlInDrawerChecker":
        return check_bowl_in_drawer(state, opts)
    if t == "DrawerPositionChecker":
        return check_drawer_position(state, opts)
    if t == "DrawerBBoxChecker":
        return check_drawer_bbox(state, opts)
    if t == "GripperOpenChecker":
        return check_gripper_open(state, opts)
    if t == "JointThresholdChecker":
        return check_joint_threshold(state, opts)
    if t == "BoxJointPositionChecker":
        return check_box_joint_position(state, opts)
    if t == "RelativeCylinderChecker":
        return check_relative_cylinder(state, opts)
    if t == "FrameBBoxChecker":
        return check_frame_bbox(state, opts)
    if t == "ObjectRotationChecker":
        return check_object_rotation(state, opts)
    if t == "SampleTiltChecker":
        return check_sample_tilt(state, opts)
    if t == "RelativeAxisRangeChecker":
        return check_relative_axis_range(state, opts)
    if t == "KeyPressSetChecker":
        return True
    # 未知类型视为不通过或可配置
    return False


def run_checker_config(state: Dict, checker_config: Dict) -> bool:
    """对整份 checker_config 做判定（支持 checker_config.checker 或 legacy 多 key）。"""
    if not checker_config:
        return False
    if "checker" in checker_config:
        return run_checker(state, checker_config["checker"])
    # legacy: 多个 checker 用 AND
    executed = False
    for key, cfg in checker_config.items():
        if not isinstance(cfg, dict) or cfg.get("enabled") is False:
            continue
        type_name = (cfg.get("type") or key[0].upper() + key[1:] if key else "").strip()
        if not type_name:
            continue
        c = dict(cfg)
        c["type"] = type_name
        executed = True
        if not run_checker(state, c):
            return False
    # Fail closed: an empty/disabled/malformed legacy configuration must not
    # be treated as a successful validation with zero checks executed.
    return executed


# ---------- 任务配置加载 ----------


def _normalize_checker_config_raw(raw: Any) -> Optional[Dict]:
    """支持 checker_config 是 dict 或 JSON 字符串的情况，返回规范化的 dict。"""
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception as e:
            print(f"Warning: 无法解析字符串形式的 checker_config: {e}", file=sys.stderr)
            return None
    return None


def _extract_checker_config_from_task(task_data: Optional[Dict]) -> Optional[Dict]:
    """从任务对象中提取并规范化 checker_config。"""
    if not isinstance(task_data, dict):
        return None
    raw = task_data.get("checker_config")
    return _normalize_checker_config_raw(raw)


def load_task_config_from_api(task_id: int, api_base: str) -> Optional[Dict]:
    """从后端 API 拉取任务详情，返回含 checker_config 的字典。"""
    import urllib.request
    base = api_base.rstrip("/")
    url = f"{base}/tasks/{task_id}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            return data
    except Exception as e:
        print(f"Warning: 无法从 API 拉取任务 {task_id}: {e}", file=sys.stderr)
        return None


def load_task_config_from_dir(task_id: int, tasks_dir: str) -> Optional[Dict]:
    """从目录中查找 task_{id}.json 或 id 字段等于 task_id 的 JSON。"""
    if not os.path.isdir(tasks_dir):
        return None
    # 1) task_{id}.json
    path = os.path.join(tasks_dir, f"task_{task_id}.json")
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: 读取 {path} 失败: {e}", file=sys.stderr)
    # 2) 任意 json 中 id 或 task_id 匹配
    for name in os.listdir(tasks_dir):
        if not name.endswith(".json"):
            continue
        p = os.path.join(tasks_dir, name)
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("id") == task_id or data.get("task_id") == task_id:
                return data
        except Exception:
            pass
    return None


def load_task_config_from_file(path: str) -> Optional[Dict]:
    """从单个 JSON 文件加载，文件需含 checker_config。"""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: 读取 {path} 失败: {e}", file=sys.stderr)
        return None


def load_checker_from_tasks_json(task_id: int, path: str) -> Optional[Dict]:
    """
    从“所有任务配置汇总”的 JSON 文件中按 task_id 取 checker_config。
    该文件支持两种结构：
      1) 顶层是数组: [ { \"id\": 22, \"checker_config\": \"{...}\" }, ... ]
      2) 顶层是字典:
         - 直接是某个任务对象 { \"id\": 22, \"checker_config\": \"{...}\" }
         - 或 id 映射: { \"22\": {\"checker_config\": \"{...}\"}, ... }
    """
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Warning: 读取 tasks-json 文件失败: {e}", file=sys.stderr)
        return None

    # 情形 1：顶层是数组
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            if item.get("id") == task_id or item.get("task_id") == task_id:
                return _extract_checker_config_from_task(item)
        return None

    # 情形 2：顶层是字典
    if isinstance(data, dict):
        # 2a: 形如 { \"tasks\": [...] }
        if isinstance(data.get("tasks"), list):
            for item in data["tasks"]:
                if not isinstance(item, dict):
                    continue
                if item.get("id") == task_id or item.get("task_id") == task_id:
                    return _extract_checker_config_from_task(item)
            return None

        # 2a: 自身就是一个任务对象
        if (data.get("id") == task_id or data.get("task_id") == task_id) and "checker_config" in data:
            return _extract_checker_config_from_task(data)
        # 2b: 以 id 为 key 的映射
        key = str(task_id)
        if key in data:
            return _normalize_checker_config_raw(data[key].get("checker_config") if isinstance(data[key], dict) else data[key])
    return None


# ---------- 主流程 ----------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="用任务 checker 校验离线轨迹 JSON 是否成功",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("trajectory_json", help="轨迹 JSON 文件路径")
    parser.add_argument(
        "--api-base",
        default=os.environ.get("API_BASE", ""),
        help="后端 API 根地址，用于按 task_id 拉取 checker_config（如 http://localhost:8000）",
    )
    parser.add_argument(
        "--tasks-dir",
        default="",
        help="任务配置目录，内含 task_1.json 等；用于按 task_id 查找单个任务 JSON",
    )
    parser.add_argument(
        "--tasks-json",
        default="",
        help="包含多条任务的汇总 JSON 文件（如 [{\"id\":22,\"checker_config\":\"{...}\"}, ...]）",
    )
    parser.add_argument(
        "--task-config",
        default="",
        help="单个任务 JSON 文件路径（含 checker_config）；若指定则忽略 task_id 与 api-base/tasks-dir",
    )
    parser.add_argument("--task-id", type=int, default=None, help="显式指定 task_id（覆盖轨迹内 trajectory_metadata.task_id）")
    parser.add_argument("--quiet", "-q", action="store_true", help="仅输出 true/false")
    args = parser.parse_args()

    # 加载轨迹
    traj_path = args.trajectory_json
    if not os.path.isfile(traj_path):
        print(f"Error: 文件不存在: {traj_path}", file=sys.stderr)
        return 1
    try:
        with open(traj_path, "r", encoding="utf-8") as f:
            traj = json.load(f)
    except Exception as e:
        print(f"Error: 无法解析轨迹 JSON: {e}", file=sys.stderr)
        return 1

    # 支持多种轨迹包装结构：
    # 1) 根节点直接有 steps
    # 2) 轨迹数据放在 trajectory_data.steps 里
    steps = traj.get("steps")
    if (not steps or not isinstance(steps, list)) and isinstance(traj.get("trajectory_data"), dict):
        steps = traj["trajectory_data"].get("steps")

    if not steps or not isinstance(steps, list):
        print("Error: 轨迹中缺少 steps 或 steps 非数组", file=sys.stderr)
        return 1
    last = steps[-1]
    state = last.get("state")
    if not state:
        print("Error: 最后一帧缺少 state", file=sys.stderr)
        return 1

    # task_id（仅用于输出与 API/目录查找）
    task_id = args.task_id
    if task_id is None:
        meta = traj.get("trajectory_metadata") or {}
        task_id = meta.get("task_id")
    if task_id is None and not args.task_config:
        print("Error: 未指定 task_id（轨迹内 trajectory_metadata.task_id 缺失，且未传 --task-id 或 --task-config）", file=sys.stderr)
        return 1

    # 加载 checker_config
    checker_config = None
    if args.task_config:
        task_data = load_task_config_from_file(args.task_config)
        if task_data:
            checker_config = _extract_checker_config_from_task(task_data)
            if task_id is None:
                task_id = task_data.get("id") or task_data.get("task_id")
    if checker_config is None and task_id is not None:
        # 1) 优先从 tasks-json 汇总文件中查找
        if args.tasks_json:
            cfg = load_checker_from_tasks_json(task_id, args.tasks_json)
            if cfg is not None:
                checker_config = cfg
        # 2) 其次尝试 API
        if args.api_base:
            task_data = load_task_config_from_api(task_id, args.api_base)
            if task_data:
                checker_config = _extract_checker_config_from_task(task_data)
        if checker_config is None and args.tasks_dir:
            task_data = load_task_config_from_dir(task_id, args.tasks_dir)
            if task_data:
                checker_config = _extract_checker_config_from_task(task_data)

    if checker_config is None:
        print("Error: 无法获取该任务的 checker_config（请提供 --api-base、--tasks-dir 或 --task-config）", file=sys.stderr)
        return 1

    success = run_checker_config(state, checker_config)
    if args.quiet:
        print("true" if success else "false")
    else:
        meta_ok = (traj.get("trajectory_metadata") or {}).get("goal_achieved")
        print(f"task_id: {task_id}")
        print(f"checker_result: {'PASS' if success else 'FAIL'}")
        if meta_ok is not None:
            print(f"metadata_goal_achieved: {meta_ok}")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
