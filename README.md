<div align="center">

# AxisDataCleaning

An offline **trajectory smoothing, resampling, and evaluation toolkit**  
**Raw teleoperation → Clean trajectories → Metrics & offline task checking**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![NumPy](https://img.shields.io/badge/NumPy-%3E%3D1.26-blueviolet)
![SciPy](https://img.shields.io/badge/SciPy-Interpolation%20%26%20Filtering-orange)
![License](https://img.shields.io/badge/License-Apache%202.0-green)

[Overview](#overview) • [Quick Start](#quick-start) • [Core Tools](#core-tools) • [Data Formats](#data-formats) • [Pipelines](#end-to-end-pipelines) • [File Structure](#file-structure) • [Notes](#notes--dependencies)


</div>

---

## Overview

`AxisDataProcessing` provides an end‑to‑end **offline trajectory processing pipeline** for Axis teleoperation data (e.g. from tasks defined in `AxisTaskGen` and executed in `AxisWebInfra`).

It is designed to:

- **Ingest raw trajectories** from JSON or CSV (single or multiple trajectories per file).
- **Smooth and resample** joint trajectories using Savitzky–Golay + cubic spline, with Franka gripper clamping and optional 3D end‑effector visualization.
- **Compute quantitative metrics** comparing before/after trajectories (smoothness, joint deviation, end‑effector deviation, frame‑removal ratio).
- **Run offline task checkers** in pure Python, mirroring the front‑end `checker_config` logic.
- Support **batch processing** via a simple `benchmark.sh` script.

Typical use cases:

- Cleaning large batches of user teleoperation data before training.
- Comparing different smoothing/processing configurations.
- Verifying whether an offline trajectory satisfies a task’s success checker.

---

## Quick Start

### Requirements

- **Python**: 3.10+
- **Core Python deps**:
  - `numpy`
  - `scipy`
  - `matplotlib`
- **Optional** (for end‑effector 3D plots and EE metrics):
  - `roboticstoolbox-python` (imported as `roboticstoolbox`)

Install minimal dependencies:

```bash
pip install numpy scipy matplotlib
# Optional: required for end-effector FK plots & EE metrics
pip install roboticstoolbox-python
```

> If `roboticstoolbox` is not installed, `smooth_resampled_traj.py` and the metrics tools still work — they automatically fall back to **joint‑space visualizations only**.
> `visualize_trajectory.py` plots end‑effector space exclusively and therefore requires it.

---

## Core Tools

### 1) `smooth_resampled_traj.py` — Trajectory Smoothing & Resampling

**Purpose**

- Load one or many trajectories from **JSON** or **CSV**.
- Optionally **merge static frames**, **smooth**, then **cubic‑spline resample** to a uniform `dt`.
- Clamp Franka finger joints to a valid range.
- Optionally generate **comparison plots**:
  - 3D **end‑effector trajectory** (requires `roboticstoolbox`), or
  - Fallback **joint‑space time series**.

**Key Features**

- Supports:
  - Single trajectory: root object with `trajectory_data.steps`.
  - Multiple trajectories: `{"trajectories": [...]}` or root JSON array.
  - CSV with:
    - `time` / `simulation_time` / `t` / `timestamp` + joint columns.
    - Optional `trajectory_id` / `traj_id` for multi‑trajectory tables.
  - Large CSV where each row stores one trajectory in a `trajectory_data` JSON string (streaming mode).
- Modes:
  - `resample`: deduplicate timestamps → cubic spline resample.
  - `smooth_resample`: merge static frames → Savitzky–Golay smoothing → cubic spline resample.
- Automatically clamps finger joints with names containing `"finger"` to \[0.0, 0.04].

**CLI Usage**

```bash
python smooth_resampled_traj.py --input <input-path> --output <output-path> [options]
```

**Required arguments**

| Argument   | Short | Description                                   |
|-----------|-------|-----------------------------------------------|
| `--input` | `-i`  | Input file or directory (`.json` / `.csv`)    |
| `--output`| `-o`  | Output file or directory (`.json` / `.csv`)   |

**Common options**

| Argument              | Short | Default             | Description                                                                 |
|-----------------------|-------|---------------------|-----------------------------------------------------------------------------|
| `--mode`              | `-m`  | `smooth_resample`   | `resample` (only resample) or `smooth_resample` (merge + smooth + resample)|
| `--dt`                |       | `0.015`             | Target time step (seconds) after resampling                                |
| `--merge-threshold`   |       | `1e-5`              | Threshold for detecting “static” frames (joint change < threshold)         |
| `--smooth-window`     |       | `15`                | Savitzky–Golay window length (odd; only for `smooth_resample`)             |
| `--smooth-poly`       |       | `3`                 | Savitzky–Golay polynomial order                                            |
| `--format`            | `-f`  | `auto`              | `auto` (by extension), `json`, or `csv`                                    |
| `--no-visualize`      |       | off                 | Do not generate any plots                                                   |
| `--visualize-path`    |       | derived from output | Custom path for visualization image                                        |
| `--show`              |       | off                 | Show interactive matplotlib window (blocks until closed)                    |

**Examples**

1. **Resample only (dt = 0.015)**

```bash
python smooth_resampled_traj.py \
  -i trajectory.json \
  -o trajectory_resampled.json \
  --mode resample \
  --dt 0.015
```

2. **Smooth + resample (dt = 0.015)**

```bash
python smooth_resampled_traj.py \
  -i trajectory.json \
  -o trajectory_smoothed_resampled.json \
  --mode smooth_resample \
  --dt 0.015
```

3. **CSV → JSON (no visualization)**

```bash
python smooth_resampled_traj.py \
  -i trajectory_sample.csv \
  -o out.json \
  --mode resample \
  --dt 0.02 \
  --no-visualize
```

4. **Tuning thresholds and smoothing window**

```bash
python smooth_resampled_traj.py \
  -i trajectory.json \
  -o out.json \
  --mode smooth_resample \
  --dt 0.02 \
  --merge-threshold 1e-6 \
  --smooth-window 21 \
  --smooth-poly 3
```

5. **JSON → CSV**

```bash
python smooth_resampled_traj.py \
  -i trajectory.json \
  -o trajectory_out.csv \
  --mode resample \
  --dt 0.015
```

6. **Directory batch processing**

If `--input` is a directory:

- All `*.json` / `*.csv` files in that directory are processed.
- `--output` must be a directory (or will be interpreted as such).
- Each input file produces `<stem>_smoothed.json` or `<stem>_smoothed.csv`.

```bash
python smooth_resampled_traj.py \
  -i ./raw_traj \
  -o ./smoothed_traj \
  --mode smooth_resample \
  --dt 0.015
```

---

### 2) `util/metrics_util.py` — Trajectory Metrics

**Purpose**

Compute **quantitative metrics** for optimized trajectories, comparing **before** and **after** versions:

- Smoothness:
  - Discrete acceleration / jerk statistics per joint and aggregated.
- Position deviation:
  - Joint‑space deviation (before interpolated to after’s time grid).
  - Optional end‑effector deviation (Euclidean distance in meters).
- Frame‑removal ratio:
  - Fraction of frames removed by the “merge static frames” logic, using the same `merge_threshold` as `smooth_resampled_traj.py`.

This is intended for **batch evaluation** of smoothing pipelines.

**CLI Usage**

```bash
python -m util.metrics_util \
  --before <before-path> \
  --after <after-path> \
  --output <metrics-output> \
  --output-format both
```

**Arguments**

| Argument          | Short | Description                                                                                 |
|-------------------|-------|---------------------------------------------------------------------------------------------|
| `--before`        | `-b`  | Trajectory file or directory (raw / before optimization)                                   |
| `--after`         | `-a`  | Trajectory file or directory (processed / after optimization)                              |
| `--output`        | `-o`  | Output path **prefix** (extension is determined by `--output-format`)                      |
| `--output-format` |       | `json` \| `csv` \| `both` (default `json`)                                                 |
| `--no-ee`         |       | Skip end‑effector deviation metrics (if `roboticstoolbox` is unavailable, EE is skipped)  |
| `--merge-threshold` |     | Threshold for static‑frame merging (should match `smooth_resampled_traj.py`)               |
| `--test`          |       | Run built‑in tests instead of processing trajectories                                      |

**Input modes**

- **File–file**: `--before` and `--after` are single files:
  - Each may contain single or multiple trajectories.
  - Trajectories are aligned by index (min length of both lists).

- **Directory–directory**:
  - All `*.json` / `*.csv` in `--before` are paired with `--after` by normalized stem:
    - `foo.json` ↔ `foo.json` or `foo_smoothed.json`.
  - Metrics are computed for each pair and for each trajectory in the files.

**Outputs**

- **JSON** (`*.json`):

  ```json
  {
    "num_trajectories": N,
    "metrics_per_trajectory": [
      {
        "trajectory_index": 0,
        "source_before": "57-1.json",
        "source_after": "57-1_smoothed.json",
        "smoothness_before": { "...": "..." },
        "smoothness_after": { "...": "..." },
        "position_deviation_joint": { "...": "..." },
        "position_deviation_ee": { "...": "..." },
        "removed_frames_ratio": 0.42,
        "meta_before": { "...": "..." }
      }
    ]
  }
  ```

- **CSV** (`*.csv`): flattened, one row per trajectory with columns such as:

  - `trajectory_index`
  - `smoothness_before_acc_mean_abs`, `smoothness_after_acc_max_abs`
  - `position_deviation_joint_max_abs`, `position_deviation_ee_mean_m`
  - `removed_frames_ratio`
  - `meta_attempt_id`, `meta_task_id`, etc.

**Example**

```bash
python -m util.metrics_util \
  -b ./raw_traj \
  -a ./smoothed_traj \
  -o ./batch_metrics \
  --output-format both
```

This will produce:

- `batch_metrics.json`
- `batch_metrics.csv`

---

### 3) `validate_offline_trajectory.py` — Offline Checker Validation

**Purpose**

Run **task success checkers** on an offline trajectory JSON, using the same `checker_config` as the web front‑end.

- Loads `steps` and `trajectory_metadata.task_id` from trajectory JSON.
- Fetches the corresponding `checker_config` from one of:
  - A backend API.
  - A local directory of task JSONs.
  - A single task JSON file.
- Re‑implements all supported checker types in pure Python, operating only on the final `state` of the trajectory (position, orientation, joints, etc.).
- Prints whether the trajectory **passes** the checker.

**Trajectory JSON requirements**

- Root object containing:

  ```json
  {
    "trajectory_metadata": {
      "task_id": 57,
      "goal_achieved": true
    },
    "trajectory_data": {
      "steps": [
        {
          "simulation_time": 0.025,
          "state": {
            "robot_joints": { "...": 0.1 },
            "object_positions": { "...": [0.0, 0.0, 0.0] },
            "object_orientations": { "...": [1.0, 0.0, 0.0, 0.0] }
          }
        }
      ]
    }
  }
  ```

  or directly:

  ```json
  {
    "steps": [ "... steps ..." ],
    "trajectory_metadata": { "task_id": 57, "...": "..." }
  }
  ```

**CLI Usage**

```bash
python validate_offline_trajectory.py trajectory.json [options]
```

**Options**

| Option          | Description                                                                                                   |
|----------------|---------------------------------------------------------------------------------------------------------------|
| `--api-base`   | Backend API base URL, e.g. `http://localhost:8000` (used to fetch `/tasks/{task_id}`)                         |
| `--tasks-dir`  | Directory with task configs (e.g. `task_1.json`, `task_2.json`…)                                             |
| `--tasks-json` | Aggregated tasks JSON file; used by the advanced version in `util/validate_offline_trajectory.py`            |
| `--task-config`| Single task JSON file that already contains `checker_config`                                                  |
| `--task-id`    | Explicitly specify task ID (overrides `trajectory_metadata.task_id`)                                          |
| `--quiet`, `-q`| Only print `true` / `false`                                                                                   |

**Examples**

1. Load task from backend API:

```bash
python validate_offline_trajectory.py \
  trajectory.json \
  --api-base http://localhost:8000
```

2. Load from local tasks directory:

```bash
python validate_offline_trajectory.py \
  trajectory.json \
  --tasks-dir ./backend/task_configs
```

3. Use a single task config file:

```bash
python validate_offline_trajectory.py \
  trajectory.json \
  --task-config ./task_57.json
```

> For more advanced use cases (including tasks JSON aggregations and richer checker support), see `util/validate_offline_trajectory.py`.

---

### 4) `extra_json.py` — Extract & Renumber JSONs

**Purpose**

Collect all `.json` files under a source directory (recursively), then copy them into a flat output directory with **renamed, numbered** filenames:

- `PREFIX-1.json`, `PREFIX-2.json`, …

Useful for building **standardized batches** (e.g. `task57` trajectories) before running smoothing + metrics.

**CLI Usage**

```bash
python extra_json.py \
  --source-dir <source-dir> \
  --output-dir <output-dir> \
  --prefix <number-prefix>
```

**Arguments**

| Argument       | Short | Description                                                                |
|----------------|-------|----------------------------------------------------------------------------|
| `--source-dir` | `-s`  | Root folder to search for `.json` files (recursively)                     |
| `--output-dir` | `-o`  | Destination folder (defaults to `source-dir` if omitted)                  |
| `--prefix`     | `-p`  | Numeric prefix for filenames (e.g. `57` → `57-1.json`, `57-2.json`, …)    |

---

### 5) `visualize_trajectory.py` — Enhanced 3D Visualization (Standalone)

A standalone script for **visualizing original vs smoothed trajectories** in 3D end‑effector space:

- Loads two JSON trajectories (e.g. `trajectory.json` and `trajectory_smoothed.json`).
- Uses `roboticstoolbox` Panda DH model for FK. **Required** — this script has no joint‑space fallback.
- Renders:
  - Original trajectory: red dashed line + points.
  - Smoothed trajectory: color‑mapped line and points by time.
  - Start & end markers, time colorbar, equalized 3D bounds.

For most use cases, the built‑in plotting inside `smooth_resampled_traj.py` is sufficient; this script is useful for **ad‑hoc comparison** or debugging.

---

### 6) `benchmark.sh` — One‑Command Batch Pipeline

**Purpose**

Run the entire pipeline:

1. Extract & renumber JSON trajectories.
2. Smooth & resample all trajectories in a folder.
3. Compute metrics comparing **before / after**.

**Script**

```bash
./benchmark.sh
```

**Environment variables / inline config**

Inside `benchmark.sh`:

- `SOURCE_DIR`: source directory containing nested trajectory JSONs.
- `OUTPUT_DIR`: directory to store extracted + processed data (defaults to `SOURCE_DIR`).
- `PREFIX`: numeric filename prefix for extracted JSONs (e.g. `57`).

Steps executed:

1. **Extract JSON**

   ```bash
   python extra_json.py \
     --source-dir "$SOURCE_DIR" \
     --output-dir "$OUTPUT_DIR" \
     --prefix "$PREFIX"
   ```

2. **Smooth trajectories**

   ```bash
   python smooth_resampled_traj.py \
     -i "$TASK_DIR" \
     -o "$TASK_DIR/cleaned_data"
   ```

3. **Compute metrics**

   ```bash
   python -m util.metrics_util \
     -b "$TASK_DIR" \
     -a "$TASK_DIR/cleaned_data" \
     -o "$TASK_DIR/batch_metrics" \
     --output-format both
   ```

Results:

- Cleaned trajectories under `cleaned_data/`.
- Metrics at:
  - `batch_metrics.json`
  - `batch_metrics.csv`

---

## Data Formats

### Trajectory JSON

**Single trajectory**

Root object with `trajectory_data.steps`:

```json
{
  "attempt_id": 26412,
  "trajectory_metadata": {
    "task_id": 57,
    "goal_achieved": true
  },
  "trajectory_data": {
    "steps": [
      {
        "simulation_time": 0.025,
        "state": {
          "robot_joints": {
            "franka/panda_joint1": 0.1,
            "...": 0.2
          },
          "object_positions": { "...": [0.0, 0.0, 0.0] },
          "object_orientations": { "...": [1.0, 0.0, 0.0, 0.0] },
          "robot_velocities": { "...": [0.0, 0.0, 0.0] }
        }
      }
    ]
  }
}
```

**Multiple trajectories**

Any of the following are accepted:

1. Root has `trajectories`:

   ```json
   {
     "trajectories": [
       { "trajectory_data": { "steps": [ "... steps ..." ] }, "...": "..." },
       { "trajectory_data": { "steps": [ "... steps ..." ] }, "...": "..." }
     ]
   }
   ```

2. Root is an array:

   ```json
   [
     { "trajectory_data": { "steps": [ "... steps ..." ] }, "...": "..." },
     { "trajectory_data": { "steps": [ "... steps ..." ] }, "...": "..." }
   ]
   ```

`smooth_resampled_traj.py` and `util/metrics_util.py` preserve as much **metadata** as possible (`attempt_id`, `task_id`, etc.) when writing outputs.

---

### Trajectory CSV

**Standard format (single or multi‑trajectory)**

- **Header**:
  - One time column: `time`, `simulation_time`, `t`, or `timestamp`.
  - One or more joint columns (column names are joint names).
  - Optional `trajectory_id` or `traj_id` column to group multiple trajectories.

Example (single trajectory):

```csv
time,franka/panda_joint1,franka/panda_joint2,...,franka/panda_finger_joint1,franka/panda_finger_joint2
0.025,0.0,-0.785,...,0.04,0.04
0.100,0.0,-0.784,...,0.04,0.04
```

Example (multiple trajectories):

```csv
trajectory_id,time,franka/panda_joint1,franka/panda_joint2
0,0.000,0.0,-0.785
0,0.020,0.0,-0.784
1,0.000,0.1,-0.780
1,0.020,0.1,-0.779
```

**“Task CSV” format (one trajectory per row)**

- Column `trajectory_data` stores a JSON string:

```csv
id,task_id,trajectory_data
1,57,"{\"steps\": [...]}"""
```

This format is handled **streamingly** by `smooth_resampled_traj.py` and `util/metrics_util.py` to support large files.

---

## End‑to‑End Pipelines

### Pipeline A — Clean and Analyze a Task Batch

1. **Extract all trajectories for a task**

   ```bash
   SOURCE_DIR=/path/to/raw/task57
   OUTPUT_DIR=/path/to/processed/task57
   PREFIX=57

   python extra_json.py \
     --source-dir "$SOURCE_DIR" \
     --output-dir "$OUTPUT_DIR" \
     --prefix "$PREFIX"
   ```

2. **Smooth & resample**

   ```bash
   python smooth_resampled_traj.py \
     -i "$OUTPUT_DIR" \
     -o "$OUTPUT_DIR/cleaned_data" \
     --mode smooth_resample \
     --dt 0.015
   ```

3. **Compute metrics**

   ```bash
   python -m util.metrics_util \
     -b "$OUTPUT_DIR" \
     -a "$OUTPUT_DIR/cleaned_data" \
     -o "$OUTPUT_DIR/batch_metrics" \
     --output-format both
   ```

### Pipeline B — Offline Task Validation

Given a single offline trajectory JSON:

```bash
python validate_offline_trajectory.py \
  trajectory.json \
  --api-base http://localhost:8000
# or:
python validate_offline_trajectory.py \
  trajectory.json \
  --tasks-dir ./backend/task_configs
```

Use the exit code / printed result to label trajectories as **PASS** / **FAIL**.

---

## File Structure

High‑level layout:

```text
AxisDataProcessing/
├── raw_traj/                      # (example) raw trajectories
├── smoothed_traj/                 # (example) processed trajectories
├── util/
│   ├── __init__.py
│   ├── metrics_util.py            # Metrics: smoothness, deviation, removed frames
│   └── validate_offline_trajectory.py  # Extended offline checker utilities
├── extra_json.py                  # Extract & renumber JSON trajectories
├── smooth_resampled_traj.py       # Core smoothing + resampling CLI
├── validate_offline_trajectory.py # Standalone offline checker runner
├── visualize_trajectory.py        # Standalone 3D visualization
├── benchmark.sh                   # One-command batch pipeline
├── tasks_config.json              # (optional) example / aggregated task configs
├── batch_metrics.json             # (example) metrics output
├── batch_metrics.csv              # (example) metrics output
└── README.md
```

---

## Notes / Dependencies

- All scripts are pure Python and run on the **offline trajectories only** — no MuJoCo or web stack is required.
- `roboticstoolbox` is used for **Franka Panda** forward kinematics:
  - End‑effector plots in `smooth_resampled_traj.py` and `visualize_trajectory.py`.
  - End‑effector deviation metrics in `util/metrics_util.py`.
- When `roboticstoolbox` is missing:
  - EE‑related features are automatically disabled.
  - Joint‑space smoothing, resampling, metrics, and checkers still function normally.
- Many thresholds (e.g. `merge_threshold`, checker tolerances) are intentionally exposed as CLI flags for **experimentation** and **ablation**.

---

## Acknowledgements

This project builds upon the excellent work of the robotics and Python communities:

- [NumPy](https://numpy.org/) — Core numerical computing library.
- [SciPy](https://scipy.org/) — Signal processing and spline interpolation.
- [matplotlib](https://matplotlib.org/) — 2D/3D visualization and plotting.
- [roboticstoolbox-python](https://github.com/petercorke/roboticstoolbox-python) — Kinematics and dynamics for the Franka Panda arm.

---

## License

This project is licensed under the **Apache License 2.0** — see the `LICENSE` file for details.

[Back to Top](#axisdataprocessing)

Built for robotic teleoperation data cleaning and analysis in the Axis ecosystem.

