"""
Robot Trajectory Smoothing & Resampling Tool (Enhanced for Franka Gripper)

功能：
- 支持多条轨迹的 JSON 读入（单条或 trajectories 数组）
- 支持 CSV 格式轨迹读入（time + 各关节列；可选 trajectory_id 表示多条）
- 两种模式：仅重采样(resample) 或 平滑+重采样(smooth_resample)
- 可配置参数：目标 dt、静止帧阈值、平滑窗口等

处理流程（resample 模式）：去重 -> 重排时间 -> 三次样条重采样 -> 夹爪限位
处理流程（smooth_resample 模式）：去重 -> 重排时间 -> Savitzky-Golay 平滑 -> 三次样条重采样 -> 夹爪限位
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# 修复 Python 3.13 / NumPy 2.0 兼容：roboticstoolbox 依赖 np.disp
if not hasattr(np, 'disp'):
    np.disp = print

import matplotlib
# 若传入 --show，使用交互式后端以便结束后可旋转/拖动 3D 图
if "--show" in sys.argv:
    try:
        matplotlib.use("TkAgg")
    except Exception:
        matplotlib.use("Qt5Agg")
else:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter

# 可选：用于末端位置 FK，未安装或导入失败时退化为关节空间图
try:
    import roboticstoolbox as rtb
    _RTB_AVAILABLE = True
except Exception:
    _RTB_AVAILABLE = False

# --- 默认配置 ---
DEFAULT_TARGET_DT = 0.015   # 目标插值间隔 (秒)
DEFAULT_MERGE_THRESHOLD = 1e-5  # 判定为静止帧的关节变化阈值
DEFAULT_SMOOTH_WINDOW = 15     # Savitzky-Golay 窗口长度（奇数）
DEFAULT_SMOOTH_POLY = 3        # Savitzky-Golay 多项式阶数
GRIPPER_MIN = 0.0
GRIPPER_MAX = 0.04


# --------------- 数据加载 ---------------

def _detect_format(path: str) -> str:
    """根据扩展名检测格式。"""
    p = path.lower()
    if p.endswith('.json'):
        return 'json'
    if p.endswith('.csv'):
        return 'csv'
    return 'json'  # 默认


def load_trajectories_json(json_path: str) -> list:
    """
    从 JSON 加载一条或多条轨迹。
    支持格式：
    - 单条：{"trajectory_data": {"steps": [...]}} 或 {"attempt_id": ..., "trajectory_data": {...}}
    - 多条：{"trajectories": [{"trajectory_data": {"steps": [...]}}, ...]}
    - 多条：直接为数组 [{"trajectory_data": {...}}, ...]
    返回 list of dict，每个 dict 含 "meta"（可选）和 "steps"（必选）。
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    out = []
    if isinstance(data, list):
        for i, item in enumerate(data):
            steps = _extract_steps(item)
            if steps is not None:
                out.append({"meta": item if isinstance(item, dict) else {}, "steps": steps})
    elif isinstance(data, dict):
        if "trajectories" in data:
            for i, item in enumerate(data["trajectories"]):
                steps = _extract_steps(item)
                if steps is not None:
                    out.append({"meta": item if isinstance(item, dict) else {}, "steps": steps})
        else:
            steps = _extract_steps(data)
            if steps is not None:
                out.append({"meta": data, "steps": steps})
    if not out:
        raise ValueError("JSON 中未找到有效的 trajectory_data.steps。")
    return out


def _extract_steps(obj) -> list | None:
    if not isinstance(obj, dict):
        return None
    td = obj.get("trajectory_data")
    if isinstance(td, dict) and "steps" in td:
        return td["steps"]
    return None


def _row_to_trajectory_task_format(row: dict) -> dict | None:
    """
    将「每行一条轨迹、trajectory_data 为 JSON 字符串」的 CSV 行转为轨迹 dict。
    返回 {"meta": {...}, "steps": [...]} 或 None。
    """
    td = row.get("trajectory_data")
    if not td or not isinstance(td, str):
        return None
    td = td.strip()
    if not td.startswith("{"):
        return None
    try:
        data = json.loads(td)
        steps = data.get("steps") if isinstance(data, dict) else None
        if not steps:
            return None
        meta = {k: v for k, v in row.items() if k != "trajectory_data"}
        return {"meta": meta, "steps": steps}
    except json.JSONDecodeError:
        return None


def _load_trajectories_csv_standard_from_rows(rows: list) -> list:
    """从已读入的 CSV 行列表解析「时间+关节列」格式，返回轨迹列表。"""
    if not rows:
        raise ValueError("CSV 为空或没有数据行。")
    first = rows[0]
    time_key = None
    for k in ("time", "simulation_time", "t", "timestamp","simulation_time_seconds"):
        if k in first:
            time_key = k
            break
    if time_key is None:
        raise ValueError("CSV 中未找到时间列（需要 time / simulation_time / t 之一）。")
    exclude = {time_key, "trajectory_id", "traj_id", "id"}
    joint_cols = [k for k in first.keys() if k not in exclude]
    if not joint_cols:
        raise ValueError("CSV 中未找到关节列。")

    def row_to_step(r):
        t = float(r.get(time_key, 0))
        joints = {}
        for j in joint_cols:
            try:
                joints[j] = float(r.get(j, 0))
            except (ValueError, TypeError):
                joints[j] = 0.0
        return {"simulation_time": t, "state": {"robot_joints": joints}}

    if "trajectory_id" in first or "traj_id" in first:
        tid_key = "trajectory_id" if "trajectory_id" in first else "traj_id"
        groups = {}
        for r in rows:
            tid = r.get(tid_key, "")
            if tid not in groups:
                groups[tid] = []
            groups[tid].append(row_to_step(r))
        out = []
        for tid in sorted(groups.keys(), key=lambda x: (str(x), x)):
            steps = groups[tid]
            steps.sort(key=lambda s: s["simulation_time"])
            out.append({"meta": {"trajectory_id": tid}, "steps": steps})
        return out
    else:
        steps = [row_to_step(r) for r in rows]
        steps.sort(key=lambda s: s["simulation_time"])
        return [{"meta": {}, "steps": steps}]


def _load_trajectories_csv_task_format_streaming(csv_path: str):
    """
    流式读取「每行一条轨迹、trajectory_data 为 JSON」的大 CSV，逐行解析，不一次性读入内存。
    每次 yield 一个 {"meta": {...}, "steps": [...]}。
    """
    import csv
    # 单列可能很大（整条轨迹 JSON），提高字段大小限制
    try:
        max_size = sys.maxsize
    except AttributeError:
        max_size = 2**31 - 1
    prev_limit = csv.field_size_limit()
    csv.field_size_limit(max_size)
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                traj = _row_to_trajectory_task_format(row)
                if traj is not None:
                    yield traj
    finally:
        csv.field_size_limit(prev_limit)


def load_trajectories_csv(csv_path: str) -> list:
    """
    从 CSV 加载一条或多条轨迹。
    支持两种格式（自动检测）：
    1) 大文件格式：表头含 trajectory_data，每行一条轨迹（trajectory_data 为 JSON 字符串）。
       此时使用流式读取，返回一个生成器，避免整文件读入内存。
    2) 标准格式：表头含 time/simulation_time/t 及关节列；可选 trajectory_id 分组。
       返回 list of dict。
    """
    import csv
    # 单列可能很大（整条轨迹 JSON），提高字段大小限制
    try:
        _max_size = sys.maxsize
    except AttributeError:
        _max_size = 2**31 - 1
    _prev_limit = csv.field_size_limit()
    csv.field_size_limit(_max_size)
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            first = next(reader, None)
        if first is None:
            raise ValueError("CSV 为空或没有数据行。")
        # 检测是否为「每行一条轨迹、trajectory_data 为 JSON」的大文件格式
        if "trajectory_data" in first and isinstance(first.get("trajectory_data"), str) and first["trajectory_data"].strip().startswith("{"):
            # 返回生成器，调用方需支持迭代器（main 中 for traj in trajectories 已支持）
            return _load_trajectories_csv_task_format_streaming(csv_path)
        # 标准格式：整表读入（小文件）
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            raise ValueError("CSV 为空或没有数据行。")
        return _load_trajectories_csv_standard_from_rows(rows)
    finally:
        csv.field_size_limit(_prev_limit)


def load_trajectories(path: str, fmt: str = "auto") -> list:
    """统一入口：根据路径或 fmt 加载一条或多条轨迹。"""
    if fmt == "auto":
        fmt = _detect_format(path)
    if fmt == "json":
        return load_trajectories_json(path)
    if fmt == "csv":
        return load_trajectories_csv(path)
    raise ValueError(f"不支持的格式: {fmt}")


# --------------- 清洗与重采样、平滑 ---------------

def get_joint_names_from_steps(steps: list) -> list:
    """从 steps 中取关节名（排序）。"""
    if not steps:
        raise ValueError("steps 为空")
    joints = steps[0]["state"]["robot_joints"]
    return sorted(joints.keys())


def raw_trajectory(steps: list):
    """
    直接从 steps 提取时间与关节数据，不做去重。
    返回 (times, joint_data, joint_names)。
    """
    joint_names = get_joint_names_from_steps(steps)
    times = np.array([s["simulation_time"] for s in steps])
    joint_data = {name: [] for name in joint_names}
    for step in steps:
        for name in joint_names:
            joint_data[name].append(step["state"]["robot_joints"].get(name, 0))
    for name in joint_names:
        joint_data[name] = np.array(joint_data[name])
    return times, joint_data, joint_names


def clean_trajectory(steps: list, merge_threshold: float):
    """
    过滤静止帧并重建紧凑时间轴。
    返回: (compact_times, joint_data, joint_names, filtered_steps)
    filtered_steps 用于后续对 position/velocity 等非 joint 字段做插值。
    """
    joint_names = get_joint_names_from_steps(steps)
    if len(steps) > 1:
        sample_count = min(len(steps), 20)
        t0 = steps[0]["simulation_time"]
        t1 = steps[sample_count - 1]["simulation_time"]
        original_dt = (t1 - t0) / (sample_count - 1)
        if original_dt <= 0:
            original_dt = 0.01
    else:
        original_dt = 0.01

    filtered = [steps[0]]
    for i in range(1, len(steps)):
        curr = steps[i]
        prev = filtered[-1]
        is_static = True
        for name in joint_names:
            vc = curr["state"]["robot_joints"].get(name, 0)
            vp = prev["state"]["robot_joints"].get(name, 0)
            if abs(vc - vp) > merge_threshold:
                is_static = False
                break
        if not is_static:
            filtered.append(curr)

    t_start = steps[0]["simulation_time"]
    compact_times = np.array([t_start + i * original_dt for i in range(len(filtered))])
    joint_data = {name: [] for name in joint_names}
    for step in filtered:
        for name in joint_names:
            joint_data[name].append(step["state"]["robot_joints"].get(name, 0))
    for name in joint_names:
        joint_data[name] = np.array(joint_data[name])
    return compact_times, joint_data, joint_names, filtered


def smooth_joint_series(angles: np.ndarray, is_finger: bool, window: int, poly: int) -> np.ndarray:
    """对一维关节序列做 Savitzky-Golay 平滑；夹爪可不平滑。"""
    if is_finger:
        return angles
    n = len(angles)
    w = min(window, n)
    if w % 2 == 0:
        w -= 1
    if w < 5:
        return angles
    return savgol_filter(angles, window_length=w, polyorder=min(poly, w - 1))


def resample_trajectory(times: np.ndarray, joint_data: dict, target_dt: float, joint_names: list):
    """三次样条重采样，夹爪做限位。返回 (new_times, resampled_joints)。"""
    t_start, t_end = float(times[0]), float(times[-1])
    new_times = np.arange(t_start, t_end, target_dt)
    if new_times.size == 0 or not np.isclose(new_times[-1], t_end):
        new_times = np.append(new_times, t_end)
    else:
        new_times[-1] = t_end
    resampled = {}
    for name in joint_names:
        cs = CubicSpline(times, joint_data[name], bc_type='natural')
        vals = cs(new_times)
        if "finger" in name.lower():
            vals = np.clip(vals, GRIPPER_MIN, GRIPPER_MAX)
        resampled[name] = vals
    return new_times, resampled


# --------------- 非 joint/gripper 的 state 字段：插值（保持格式不变） ---------------

def _value_to_array(val):
    """将单步的 value 转为可插值的向量：(list or dict of numbers) -> (array, is_list, keys). keys 仅 dict 时用。"""
    if val is None:
        return np.array([0.0]), True, None
    if isinstance(val, (int, float)):
        return np.array([float(val)]), True, None
    if isinstance(val, list):
        return np.array([float(x) for x in val]), True, None
    if isinstance(val, dict):
        keys = sorted(val.keys())
        return np.array([float(val[k]) for k in keys]), False, keys
    return np.array([0.0]), True, None


def _array_to_value(arr, is_list, keys=None):
    """将插值后的向量转回 value 格式。keys 为 dict 时的键顺序（与 _value_to_array 一致）。"""
    if is_list:
        return [round(float(x), 6) for x in arr]
    if keys is not None:
        return {keys[i]: round(float(arr[i]), 6) for i in range(min(len(keys), len(arr)))}
    return {str(i): round(float(arr[i]), 6) for i in range(len(arr))}


def extract_extra_state_arrays(steps: list) -> tuple[list, dict, dict]:
    """
    从 steps 提取 state 中除 robot_joints 外的数值字段，便于插值。
    返回 (times, extra_arrays, extra_structure)。
    extra_arrays: (state_key, subkey) -> np.array (n_steps, dim)
    extra_structure: (state_key, subkey) -> (is_list, keys) 其中 keys 为 dict 时的键顺序
    """
    if not steps:
        return [], {}, {}
    times = np.array([s["simulation_time"] for s in steps])
    state0 = steps[0].get("state", {})
    extra_arrays = {}
    extra_structure = {}
    for state_key, top_val in state0.items():
        if state_key == "robot_joints":
            continue
        if not isinstance(top_val, dict):
            continue
        for subkey in top_val.keys():
            vecs = []
            for s in steps:
                val = s.get("state", {}).get(state_key, {}).get(subkey)
                arr, is_list, keys = _value_to_array(val)
                vecs.append(arr)
            if (state_key, subkey) not in extra_structure:
                extra_structure[(state_key, subkey)] = (is_list, keys)
            try:
                mat = np.array(vecs)
            except Exception:
                continue
            extra_arrays[(state_key, subkey)] = mat
    return times, extra_arrays, extra_structure


def interpolate_extra_state(
    times_old: np.ndarray,
    extra_arrays: dict,
    extra_structure: dict,
    times_new: np.ndarray,
) -> list[dict]:
    """
    将 extra_arrays 从 times_old 插值到 times_new（三次样条），返回每步的 extra state 字典。
    返回 list of dict: [{"object_positions": {...}, "robot_velocities": {...}, ...}, ...]
    """
    if not extra_arrays or len(times_old) < 2 or len(times_new) == 0:
        return [{}] * len(times_new)
    n_new = len(times_new)
    out_per_step = [{} for _ in range(n_new)]
    for (state_key, subkey), mat in extra_arrays.items():
        if state_key not in out_per_step[0]:
            out_per_step[0][state_key] = {}
        struct = extra_structure.get((state_key, subkey), (True, None))
        is_list = struct[0] if isinstance(struct, tuple) else struct
        keys = struct[1] if isinstance(struct, tuple) and len(struct) > 1 else None
        try:
            if mat.ndim == 1:
                mat = mat.reshape(-1, 1)
            interp_cols = []
            for d in range(mat.shape[1]):
                cs = CubicSpline(times_old, mat[:, d], bc_type="natural")
                interp_cols.append(cs(times_new))
            interp_mat = np.column_stack(interp_cols)
            for i in range(n_new):
                if state_key not in out_per_step[i]:
                    out_per_step[i][state_key] = {}
                out_per_step[i][state_key][subkey] = _array_to_value(interp_mat[i], is_list, keys)
        except Exception:
            for i in range(n_new):
                if state_key not in out_per_step[i]:
                    out_per_step[i][state_key] = {}
                out_per_step[i][state_key][subkey] = _array_to_value(mat[0], is_list, keys)
    return out_per_step


# --------------- 单条轨迹处理 ---------------

def process_single_trajectory(
    steps: list,
    target_dt: float,
    merge_threshold: float,
    mode: str,
    smooth_window: int,
    smooth_poly: int,
    joint_names: list | None = None,
):
    """
    处理单条轨迹。
    mode: "resample" 仅重采样；"smooth_resample" 先平滑再重采样。
    对 state 中除 robot_joints 外的字段（position、velocity 等）做插值到新时间网格，保持格式不变。
    返回 (new_times, final_joints, joint_names, clean_times, clean_joints, extra_state_per_step, step_template)。
    """
    if joint_names is None:
        joint_names = get_joint_names_from_steps(steps)
    step_template = steps[0] if steps else {}

    if mode == "resample":
        raw_times, raw_joints, joint_names = raw_trajectory(steps)
        final_times, final_joints = resample_trajectory(raw_times, raw_joints, target_dt, joint_names)
        steps_used = steps
        times_used = raw_times
        clean_times, clean_joints = raw_times, raw_joints
    elif mode == "smooth_resample":
        clean_times, clean_joints, joint_names, filtered_steps = clean_trajectory(steps, merge_threshold)
        smoothed = {}
        for name in joint_names:
            is_finger = "finger" in name.lower()
            smoothed[name] = smooth_joint_series(
                clean_joints[name], is_finger, smooth_window, smooth_poly
            )
        final_times, final_joints = resample_trajectory(clean_times, smoothed, target_dt, joint_names)
        steps_used = filtered_steps
        times_used = clean_times
    else:
        raise ValueError(f"未知 mode: {mode}")

    # 对 position、velocity 等非 joint 字段插值到 final_times
    times_old, extra_arrays, extra_structure = extract_extra_state_arrays(steps_used)
    extra_state_per_step = interpolate_extra_state(times_old, extra_arrays, extra_structure, final_times)
    return final_times, final_joints, joint_names, clean_times, clean_joints, extra_state_per_step, step_template


# --------------- 保存 ---------------

def steps_from_arrays(
    times: np.ndarray,
    joint_data: dict,
    joint_names: list,
    extra_state_per_step: list | None = None,
    step_template: dict | None = None,
) -> list:
    """
    由时间数组和关节字典生成 steps 列表，保持格式不变。
    extra_state_per_step: 每步的 state 中非 robot_joints 部分（object_positions、robot_velocities 等）。
    step_template: 首步模板，用于 action、timestamp 等非数值字段；timestamp 可按 simulation_time 缩放。
    """
    steps = []
    n = len(times)
    t0 = float(times[0]) if n else 0.0
    template_state = (step_template or {}).get("state", {})
    for i in range(n):
        step_joints = {name: round(float(joint_data[name][i]), 6) for name in joint_names}
        state = {"robot_joints": step_joints}
        for k in template_state:
            if k == "robot_joints":
                continue
            if extra_state_per_step and i < len(extra_state_per_step) and k in extra_state_per_step[i]:
                state[k] = extra_state_per_step[i][k]
            else:
                state[k] = template_state.get(k, {})
        step = {
            "simulation_time": round(float(times[i]), 4),
            "state": state,
        }
        if step_template:
            if "action" in step_template:
                step["action"] = step_template["action"]
            if "timestamp" in step_template:
                t0_ts = step_template.get("timestamp", 0)
                step["timestamp"] = int(t0_ts + (float(times[i]) - t0) * 1000)
        steps.append(step)
    return steps


def save_trajectories_json(
    results: list,
    output_path: str,
    original_meta: list | None = None,
    single_input_meta: dict | None = None,
):
    """
    results: list of (new_times, final_joints, joint_names, extra_state_per_step, step_template)
    original_meta: 与 results 同长的 list，每项为对应轨迹的 meta（用于保留 attempt_id 等）。
    single_input_meta: 若只有一条轨迹且希望保留原文件顶层字段，可传原 JSON 的根对象。
    """
    if original_meta is None:
        original_meta = [{}] * len(results)
    if len(original_meta) != len(results):
        original_meta = [{}] * len(results)

    if len(results) == 1 and single_input_meta is not None:
        # 单条轨迹：尽量保持与原 JSON 结构一致
        r0 = results[0]
        times, joints, jnames = r0[0], r0[1], r0[2]
        extra_state = r0[3] if len(r0) > 3 else None
        step_tpl = r0[4] if len(r0) > 4 else None
        steps = steps_from_arrays(times, joints, jnames, extra_state, step_tpl)
        out = dict(single_input_meta)
        if "trajectory_data" not in out:
            out["trajectory_data"] = {}
        out["trajectory_data"]["steps"] = steps
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2)
        return

    # 多条轨迹
    traj_list = []
    for res, meta in zip(results, original_meta):
        times, joints, jnames = res[0], res[1], res[2]
        extra_state = res[3] if len(res) > 3 else None
        step_tpl = res[4] if len(res) > 4 else None
        steps = steps_from_arrays(times, joints, jnames, extra_state, step_tpl)
        obj = dict(meta)
        if "trajectory_data" not in obj:
            obj["trajectory_data"] = {}
        obj["trajectory_data"]["steps"] = steps
        traj_list.append(obj)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({"trajectories": traj_list}, f, indent=2)


def save_trajectories_csv(
    results: list,
    output_path: str,
    joint_names: list | None = None,
    original_metas: list | None = None,
):
    """
    将多条轨迹写入 CSV。
    - 若提供 original_metas 且首条 meta 含 id/attempt_id/trajectory_data（task 格式），
      则输出与原始 CSV 同列：id, attempt_id, trajectory_data, metadata, ...，每行一条轨迹。
    - 否则输出标准格式：time + 关节列（或多条时 trajectory_id + time + 关节列）。
    """
    import csv
    if not results:
        return
    # 判断是否为 task CSV 格式（与原始 CSV 列一致）
    task_format = (
        original_metas is not None
        and len(original_metas) == len(results)
        and isinstance(original_metas[0], dict)
        and ("trajectory_data" in original_metas[0] or "attempt_id" in original_metas[0])
    )
    if task_format:
        base = list(original_metas[0].keys())
        if "trajectory_data" not in base:
            base = base[:2] + ["trajectory_data"] + base[2:] if len(base) >= 2 else base + ["trajectory_data"]
        fieldnames = base
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for idx, res in enumerate(results):
                times, joints, jnames = res[0], res[1], res[2]
                extra_state = res[3] if len(res) > 3 else None
                step_tpl = res[4] if len(res) > 4 else None
                steps = steps_from_arrays(times, joints, jnames, extra_state, step_tpl)
                row = {}
                for k, v in original_metas[idx].items():
                    row[k] = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                row["trajectory_data"] = json.dumps({"steps": steps}, ensure_ascii=False)
                if "total_steps" in row:
                    row["total_steps"] = len(steps)
                if "simulation_time" in row and len(times) >= 2:
                    row["simulation_time"] = round(float(times[-1]) - float(times[0]), 6)
                w.writerow(row)
        return
    # 标准格式：time + 关节列
    if joint_names is None and results:
        joint_names = results[0][2]
    if not joint_names:
        return
    fieldnames = ["trajectory_id", "time"] + list(joint_names) if len(results) > 1 else ["time"] + list(joint_names)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for idx, res in enumerate(results):
            times, joints, jnames = res[0], res[1], res[2]
            jn = jnames
            for i in range(len(times)):
                row = {"time": round(float(times[i]), 6)}
                if len(results) > 1:
                    row["trajectory_id"] = idx
                for name in jn:
                    row[name] = round(float(joints[name][i]), 6)
                w.writerow(row)


# --------------- 可视化 ---------------

# Franka 7 轴关节名（用于 FK 末端位置），按 joint1..joint7 顺序
ARM_JOINT_NAMES = [f"franka/panda_joint{i+1}" for i in range(7)]


def _get_arm_q_matrix(joint_data: dict, joint_names: list) -> np.ndarray | None:
    """从 joint_data 中提取 7 个臂关节，得到 (N, 7) 的 q 矩阵；缺臂关节时返回 None。"""
    arm_names = [n for n in ARM_JOINT_NAMES if n in joint_data]
    if len(arm_names) != 7:
        return None
    rows = []
    for i in range(len(joint_data[arm_names[0]])):
        rows.append([float(joint_data[name][i]) for name in arm_names])
    return np.array(rows)


def _compute_fk_ee(q_matrix: np.ndarray) -> np.ndarray | None:
    """由 (N, 7) 关节角计算末端位置 (N, 3)，失败返回 None。"""
    if not _RTB_AVAILABLE or q_matrix is None or q_matrix.shape[1] != 7:
        return None
    try:
        panda = rtb.models.DH.Panda()
        out = panda.fkine(q_matrix)
        t = out.t
        if hasattr(t, 'ndim') and t.ndim == 1:
            t = t.reshape(1, -1)
        return np.asarray(t)
    except Exception:
        return None


def visualize_end_effector(
    clean_times, clean_joints, final_times, final_joints, joint_names,
    title: str, out_path: str,
    show_interactive: bool = False,
):
    """
    绘制轨迹优化前后的机器人末端位置对比图（3D）。
    需要 7 轴臂关节与 roboticstoolbox；若不满足则退化为关节空间图。
    show_interactive=True 时保存后不关闭窗口，并调用 plt.show() 以便旋转/拖动。
    """
    q_before = _get_arm_q_matrix(clean_joints, joint_names)
    q_after = _get_arm_q_matrix(final_joints, joint_names)
    if q_before is None or q_after is None:
        print("[Plot] 缺少 7 轴臂关节，退化为关节空间对比图")
        visualize_trajectory(clean_times, clean_joints, final_times, final_joints, joint_names, title, out_path, show_interactive)
        return
    xyz_before = _compute_fk_ee(q_before)
    xyz_after = _compute_fk_ee(q_after)
    if xyz_before is None or xyz_after is None:
        print("[Plot] FK 计算失败，退化为关节空间对比图")
        visualize_trajectory(clean_times, clean_joints, final_times, final_joints, joint_names, title, out_path, show_interactive)
        return

    # 与 visualize_trajectory 一致：用 optimization 后轨迹的 simulation_time
    time_array = np.asarray(final_times, dtype=float)
    if len(time_array) == 0:
        time_array = np.array([0.0])
    colormap = 'plasma'
    norm = plt.Normalize(time_array.min(), time_array.max())
    total_points = len(xyz_before)
    skip = max(1, total_points // 100)

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 1. 原始轨迹（与 visualize_trajectory 一致）
    ax.plot(xyz_before[:, 0], xyz_before[:, 1], xyz_before[:, 2],
            color='crimson', linestyle='--', linewidth=1.8, alpha=0.9,
            label='Original Line (Raw)', zorder=2)
    ax.scatter(xyz_before[::skip, 0], xyz_before[::skip, 1], xyz_before[::skip, 2],
               color='crimson', marker='o', s=35, alpha=0.9,
               label='Original Points', zorder=2)

    # 2. 优化后轨迹：渐变线 + 渐变散点（与 visualize_trajectory 一致）
    x, y, z = xyz_after[:, 0], xyz_after[:, 1], xyz_after[:, 2]
    if len(xyz_after) > 1:
        points = np.array([x, y, z]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = Line3DCollection(segments, cmap=colormap, norm=norm)
        lc.set_array(time_array[:-1])
        lc.set_linewidth(3.5)
        lc.set_alpha(0.8)
        ax.add_collection(lc)
    sc = ax.scatter(xyz_after[::skip, 0], xyz_after[::skip, 1], xyz_after[::skip, 2],
                    c=time_array[::skip], cmap=colormap, norm=norm,
                    marker='o', s=45, alpha=1.0,
                    label='Smoothed Points (Color=Time)', zorder=3)

    ax.scatter(x[0], y[0], z[0], c='green', s=180, marker='^', label='Start', zorder=10, edgecolors='white')
    ax.scatter(x[-1], y[-1], z[-1], c='black', s=180, marker='x', label='End', zorder=10, linewidth=3)

    cbar = plt.colorbar(sc, ax=ax, pad=0.1, fraction=0.03)
    cbar.set_label('Simulation Time (s)', rotation=270, labelpad=15)

    max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() / 2.0
    mid_x = (x.max() + x.min()) * 0.5
    mid_y = (y.max() + y.min()) * 0.5
    mid_z = (z.max() + z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_title('Franka End-Effector: Highlighting Original vs Smoothed Gradient', fontsize=14)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), markerscale=1.2)
    plt.tight_layout()
    plt.savefig(out_path)
    if show_interactive:
        plt.show()
    else:
        plt.close()
    print("[Plot] 末端位置对比图已保存:", out_path)


def visualize_trajectory(
    clean_times, clean_joints, final_times, final_joints, joint_names,
    title: str, out_path: str,
    show_interactive: bool = False,
):
    """绘制单条轨迹的关节空间对比图（各关节随时间曲线）。"""
    n = len(joint_names)
    fig, axes = plt.subplots(n, 1, figsize=(10, max(1.5 * n, 4)), sharex=True)
    if n == 1:
        axes = [axes]
    for i, name in enumerate(joint_names):
        ax = axes[i]
        ax.plot(clean_times, clean_joints[name], 'o', markersize=2, color='gray', alpha=0.5, label='Keyframes')
        ax.plot(final_times, final_joints[name], '-', linewidth=1.5, color='green' if 'finger' in name else 'orange', label='Processed')
        ax.set_ylabel(name.split('/')[-1], fontsize=8, rotation=0, labelpad=40)
        ax.grid(True, alpha=0.3)
        if "finger" in name:
            ax.axhline(GRIPPER_MAX, color='r', linestyle='--', alpha=0.3, linewidth=1)
            ax.axhline(GRIPPER_MIN, color='r', linestyle='--', alpha=0.3, linewidth=1)
    axes[0].legend(loc='upper right')
    plt.xlabel("Time (s)")
    plt.suptitle(title, y=0.995)
    plt.tight_layout()
    plt.savefig(out_path)
    if show_interactive:
        plt.show()
    else:
        plt.close()


# --------------- CLI ---------------

def parse_args():
    p = argparse.ArgumentParser(description="轨迹重采样与平滑工具：支持 JSON/CSV，单条/多条轨迹，单文件或文件夹批处理。")
    p.add_argument("--input", "-i", required=True, help="输入文件路径或文件夹路径（.json/.csv 或含多个此类文件的目录）")
    p.add_argument("--output", "-o", required=True, help="输出文件路径或输出文件夹路径（输入为文件夹时须为目录）")
    p.add_argument("--format", "-f", choices=["auto", "json", "csv"], default="auto", help="输入格式，默认 auto 按扩展名")
    p.add_argument("--mode", "-m", choices=["resample", "smooth_resample"], default="smooth_resample",
                   help="resample=仅重采样；smooth_resample=先平滑再重采样")
    p.add_argument("--dt", type=float, default=DEFAULT_TARGET_DT, help="目标时间间隔（秒）")
    p.add_argument("--merge-threshold", type=float, default=DEFAULT_MERGE_THRESHOLD, help="静止帧判定阈值")
    p.add_argument("--smooth-window", type=int, default=DEFAULT_SMOOTH_WINDOW, help="Savitzky-Golay 窗口长度（奇数）")
    p.add_argument("--smooth-poly", type=int, default=DEFAULT_SMOOTH_POLY, help="Savitzky-Golay 多项式阶数")
    p.add_argument("--no-visualize", action="store_true", help="不生成可视化图片")
    p.add_argument("--visualize-path", default=None, help="可视化图片保存路径（默认根据 output 生成）")
    p.add_argument("--show", action="store_true", help="保存图后弹出交互窗口，可旋转/拖动 3D 图，关闭窗口后程序才结束")
    return p.parse_args()


def run_single_file(input_path: str | Path, output_path: str | Path, args) -> None:
    """
    对单个输入文件做轨迹平滑/重采样并写入输出文件。
    input_path / output_path 可为 str 或 Path。
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    inp_str = str(input_path)
    out_str = str(output_path)
    fmt = args.format if args.format != "auto" else _detect_format(inp_str)
    out_fmt = _detect_format(out_str)

    trajectories = load_trajectories(inp_str, fmt)
    if hasattr(trajectories, "__len__"):
        print(f"  [Load]   共 {len(trajectories)} 条轨迹")
    else:
        print("  [Load]   流式加载，逐条处理")

    original_metas = [] if not hasattr(trajectories, "__len__") else [t["meta"] for t in trajectories]
    single_input_meta = None
    if fmt == "json" and hasattr(trajectories, "__len__") and len(trajectories) == 1:
        with open(input_path, "r", encoding="utf-8") as f:
            single_input_meta = json.load(f)

    results = []
    for i, traj in enumerate(trajectories):
        steps = traj["steps"]
        if len(steps) < 2:
            print(f"    [Traj {i}] 跳过：仅 {len(steps)} 个时间点")
            continue
        if not hasattr(trajectories, "__len__"):
            original_metas.append(traj["meta"])
        joint_names = get_joint_names_from_steps(steps)
        print(f"    [Traj {i}] steps={len(steps)}")
        final_times, final_joints, jnames, clean_times, clean_joints, extra_state_per_step, step_template = process_single_trajectory(
            steps,
            target_dt=args.dt,
            merge_threshold=args.merge_threshold,
            mode=args.mode,
            smooth_window=args.smooth_window,
            smooth_poly=args.smooth_poly,
            joint_names=joint_names,
        )
        results.append((final_times, final_joints, jnames, extra_state_per_step, step_template))
        if not args.no_visualize and (hasattr(trajectories, "__len__") and len(trajectories) == 1):
            vis_path = output_path.parent / (output_path.stem + "_plot.png")
            visualize_end_effector(
                clean_times, clean_joints, final_times, final_joints, jnames,
                title=f"Franka End-Effector: Before vs After ({args.mode}, dt={args.dt}s)", out_path=str(vis_path),
                show_interactive=args.show,
            )
        elif not args.no_visualize and i == 0:
            vis_path = output_path.parent / (output_path.stem + "_plot.png")
            visualize_end_effector(
                clean_times, clean_joints, final_times, final_joints, jnames,
                title=f"Trajectory 0: Before vs After ({args.mode})", out_path=str(vis_path),
                show_interactive=args.show,
            )

    if out_fmt == "json":
        save_trajectories_json(results, out_str, original_metas, single_input_meta)
    else:
        save_trajectories_csv(results, out_str, original_metas=original_metas)
    print(f"  [Saved]  {output_path}")


def main():
    args = parse_args()
    t0 = time.time()
    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"[Mode]   {args.mode}")

    if input_path.is_dir():
        # 批处理：输入为文件夹，输出为文件夹
        output_dir = output_path if (not output_path.suffix or output_path.suffix.lower() not in (".json", ".csv")) else output_path.parent
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        input_files = sorted(set(input_path.glob("*.json")) | set(input_path.glob("*.csv")), key=lambda p: p.name)
        if not input_files:
            print(f"[Error] 未在目录中找到 .json 或 .csv 文件: {input_path}")
            return
        print(f"[Input]  目录 {input_path}，共 {len(input_files)} 个文件")
        print(f"[Output] 目录 {output_dir}")
        for inp in input_files:
            out_ext = ".json" if args.output.endswith(".json") or not args.output.endswith(".csv") else ".csv"
            out_file = output_dir / (inp.stem + "_smoothed" + out_ext)
            print(f"\n[File]   {inp.name} -> {out_file.name}")
            run_single_file(inp, out_file, args)
    else:
        # 单文件
        if not input_path.exists():
            print(f"[Error] 输入不存在: {input_path}")
            return
        out_fmt = _detect_format(str(output_path))
        if output_path.suffix and output_path.suffix.lower() in (".json", ".csv"):
            pass
        else:
            output_path = Path(str(output_path) + (".json" if out_fmt == "json" else ".csv"))
        print(f"[Input]  {input_path}")
        print(f"[Output] {output_path}")
        run_single_file(input_path, output_path, args)

    elapsed = time.time() - t0
    print("-" * 40)
    print(f"完成，耗时 {elapsed:.3f}s")


if __name__ == "__main__":
    main()