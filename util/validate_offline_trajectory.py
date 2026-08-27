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

# === parity checkers (PR-D) ===
try:
    from check_body_contact import check_body_contact
    from check_position_delta import check_position_delta
    from check_ordered_step import check_ordered_step
except Exception:
    check_body_contact = None
    check_position_delta = None
    check_ordered_step = None



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
    joints = state.get("robot_joints") or {}
    key = _find_state_key(joint_name, list(joints.keys()))
    if key is None:
        return None
    v = joints[key]
    try:
        return float(v)
    except (TypeError, ValueError):
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


def check_gripper_open(state: Dict, opts: Dict) -> bool:
    threshold = float(opts.get("threshold") or 0.2)
    names = opts.get("fingerJointNames") or opts.get("gripperJointNames") or []
    for jname in names:
        v = get_joint_value(state, jname)
        if v is not None and v > threshold:
            return True
    return False


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
    if t == "BowlPositionChecker":
        return check_bowl_position(state, opts)
    if t == "DrawerPositionChecker":
        return check_drawer_position(state, opts)
    if t == "GripperOpenChecker":
        return check_gripper_open(state, opts)
    if t == "JointThresholdChecker":
        return check_joint_threshold(state, opts)
    if t == "BoxJointPositionChecker":
        return check_box_joint_position(state, opts)
    if t == "BodyContactChecker" and check_body_contact is not None:
        return check_body_contact(state, opts)
    if t == "PositionDeltaChecker" and check_position_delta is not None:
        return check_position_delta(state, opts)
    if t == "OrderedStepChecker" and check_ordered_step is not None:
        # OrderedStepChecker 接受一个完整轨迹(steps 列表)而不是单 state;
        # 调用方需把轨迹传入 state['_trajectory']。 若不存在则返回 False。
        return check_ordered_step(state.get("_trajectory", [state]), opts)
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
    for key, cfg in checker_config.items():
        if not isinstance(cfg, dict) or cfg.get("enabled") is False:
            continue
        type_name = (cfg.get("type") or key[0].upper() + key[1:] if key else "").strip()
        if not type_name:
            continue
        c = dict(cfg)
        c["type"] = type_name
        if not run_checker(state, c):
            return False
    return True


# ---------- 任务配置加载 ----------


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
        help="任务配置目录，内含 task_1.json 等；用于按 task_id 查找 checker_config",
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

    steps = traj.get("steps")
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
            checker_config = task_data.get("checker_config")
            if task_id is None:
                task_id = task_data.get("id") or task_data.get("task_id")
    if checker_config is None and task_id is not None:
        if args.api_base:
            task_data = load_task_config_from_api(task_id, args.api_base)
            if task_data:
                checker_config = task_data.get("checker_config")
        if checker_config is None and args.tasks_dir:
            task_data = load_task_config_from_dir(task_id, args.tasks_dir)
            if task_data:
                checker_config = task_data.get("checker_config")

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
