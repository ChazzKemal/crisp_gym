#!/usr/bin/env python3
"""Offline calibration report for `18_calibration_record_geo.py` datasets.

Loads a LeRobot v3 dataset produced by `18_calibration_record_geo.py`,
regenerates the ground-truth commanded trajectory (circle or square) from
the same CLI arguments, and prints a markdown report comparing the
recorded columns against the ground truth. Optionally writes plots to an
analysis directory alongside the dataset.

Sibling of `17_calibration_report.py`, which analyses axis-aligned sweeps
from `16_calibration_record.py`. Both reports share the same metric
layout (§1-§6) so they can be read side-by-side.

This is a **read-only** analyzer. It never touches the robot, never
writes to the dataset, and never imports from `crisp_gym` or `crisp_py`
(no ROS stack required). Run it in any env that has `numpy`, `pandas`,
and `pyarrow`. For plots, also `matplotlib`.

Usage:
    # Default — report for a square dataset with recorder defaults.
    python examples/19_calibration_report_geo.py \\
        --repo-id calib_geo_square_001 --shape square --size 0.05

    # Circle at different velocity (must match recorder CLI).
    python examples/19_calibration_report_geo.py \\
        --repo-id calib_geo_circle_slow --shape circle --size 0.05 \\
        --velocity 0.02

    # Write report + plots to disk.
    python examples/19_calibration_report_geo.py \\
        --repo-id calib_geo_square_001 --shape square --size 0.05 \\
        --out analysis --plot

The trajectory-shape arguments (`--shape`, `--size`, `--plane`,
`--velocity`, `--loops`, `--dwell-seconds`, `--fps`,
`--include-gripper-toggle` / `--no-include-gripper-toggle`) **must match**
the arguments passed to `18_calibration_record_geo.py`. If they don't,
the ground-truth phase boundaries and waypoint counts will be offset and
the "action vs ground truth" diff will blow up.

What the report surfaces:

1. Action vs ground truth — are the saved commands exactly what the
   recorder intended?
2. Orientation invariance — the recorder holds orientation constant, so
   `std(action[3:6])` and `std(target[3:6])` should be ~0. Non-zero std
   on action or target is a representation bug. Measured cartesian uses
   circular std so Euler-XYZ ±π wraparound does not look like a wobble.
3. Action vs recorded env target — should match to sub-mm / sub-mrad.
4. Position fidelity per phase — measured cartesian vs commanded action,
   labeled by trajectory phase (pre_dwell / approach / shape_open /
   shape_closed / return / post_dwell).
5. **Shape geometry** — specific to this variant:
   - For a **circle**: mean / max / std of the commanded radial distance
     from the anchor (must equal `--size`), and the plane-off-normal
     component (must be ~0 in the chosen plane).
   - For a **square**: max-off-axis test per edge (each edge has one
     constant coordinate in the shape plane), plus edge-length stats.
   - **Closure**: the last shape frame's commanded position vs the first,
     and the measured cartesian closure gap (how far the arm ended up
     from where it started the shape after tracing).
6. Gripper consistency — action[6] vs gripper_target (both flipped to
   crisp_py convention).
7. Timing — ⚠ synthetic LeRobot timestamps, not wall-clock.
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from lerobot.utils.constants import HF_LEROBOT_HOME
except ImportError:
    try:
        from lerobot.constants import HF_LEROBOT_HOME
    except ImportError:
        HF_LEROBOT_HOME = Path.home() / ".cache" / "huggingface" / "lerobot"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plane axes — must match 18_calibration_record_geo.py exactly.
# ---------------------------------------------------------------------------

PLANE_AXES = {
    "xy": (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
    "xz": (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
    "yz": (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
}

# Normal to each plane (cross product of u and v), used to measure
# out-of-plane drift of the commanded trajectory.
PLANE_NORMAL = {
    "xy": np.array([0.0, 0.0, 1.0]),
    "xz": np.array([0.0, 1.0, 0.0]),
    "yz": np.array([1.0, 0.0, 0.0]),
}


# ---------------------------------------------------------------------------
# Phase labels — dynamically-generated, stable indices.
# ---------------------------------------------------------------------------

PHASE_PRE_DWELL = 0
PHASE_APPROACH = 1
PHASE_SHAPE_OPEN = 2
PHASE_SHAPE_CLOSED = 3
PHASE_RETURN = 4
PHASE_POST_DWELL = 5

PHASE_NAMES: List[str] = [
    "pre_dwell",
    "approach",
    "shape_open",
    "shape_closed",
    "return",
    "post_dwell",
]


# ---------------------------------------------------------------------------
# Ground-truth trajectory (pure numpy, mirrors 18_calibration_record_geo.py)
# ---------------------------------------------------------------------------

@dataclass
class ShapeTrajectorySpec:
    shape: str          # "circle" | "square"
    size: float         # circle radius OR square side length (m)
    plane: str          # "xy" | "xz" | "yz"
    velocity: float     # m/s along the shape
    fps: int
    loops: int
    dwell_seconds: float
    include_gripper_toggle: bool

    def __post_init__(self) -> None:
        if self.shape not in {"circle", "square"}:
            raise ValueError(f"unknown shape: {self.shape}")
        if self.plane not in PLANE_AXES:
            raise ValueError(f"unknown plane: {self.plane}")
        if self.size <= 0:
            raise ValueError("--size must be positive")
        if self.velocity <= 0:
            raise ValueError("--velocity must be positive")
        if self.loops < 1:
            raise ValueError("--loops must be >= 1")
        if self.dwell_seconds < 0:
            raise ValueError("--dwell-seconds must be >= 0")

    @property
    def n_dwell(self) -> int:
        return max(1, round(self.dwell_seconds * self.fps))

    @property
    def n_approach(self) -> int:
        # Mirrors recorder: for circle, approach = size. For square, size/2.
        approach_dist = self.size if self.shape == "circle" else self.size / 2.0
        return max(1, round(approach_dist / self.velocity * self.fps))

    @property
    def n_per_loop(self) -> int:
        """Shape frames per single loop."""
        if self.shape == "circle":
            angle_per_frame = self.velocity / (self.size * self.fps)
            return max(8, round((2.0 * np.pi) / angle_per_frame))
        # square: 8 edges, each of length size/2
        n_per_edge = max(1, round((self.size / 2.0) / self.velocity * self.fps))
        return 8 * n_per_edge

    @property
    def n_shape(self) -> int:
        return self.n_per_loop * self.loops

    @property
    def n_total(self) -> int:
        return 2 * self.n_dwell + 2 * self.n_approach + self.n_shape


def _segment(start: np.ndarray, end: np.ndarray, n: int) -> np.ndarray:
    """Linear interpolation, exclusive of start, inclusive of end. (N, 3)."""
    ts = np.linspace(1.0 / n, 1.0, n).reshape(-1, 1)
    return start + ts * (end - start)


def _hold(p: np.ndarray, n: int) -> np.ndarray:
    return np.tile(p.reshape(1, -1), (n, 1))


def _circle_positions_gt(
    center: np.ndarray,
    radius: float,
    u: np.ndarray,
    v: np.ndarray,
    velocity: float,
    fps: int,
    loops: int,
) -> np.ndarray:
    """Exact copy of 18_calibration_record_geo._circle_positions's math."""
    angle_per_frame = velocity / (radius * fps)
    n_per_loop = max(8, round((2.0 * np.pi) / angle_per_frame))
    rows: List[np.ndarray] = []
    for _ in range(loops):
        for i in range(1, n_per_loop + 1):
            theta = 2.0 * np.pi * (i / n_per_loop)
            rows.append(center + radius * (np.cos(theta) * u + np.sin(theta) * v))
    return np.stack(rows) if rows else np.zeros((0, 3), dtype=np.float64)


def _square_positions_gt(
    center: np.ndarray,
    side: float,
    u: np.ndarray,
    v: np.ndarray,
    velocity: float,
    fps: int,
    loops: int,
) -> np.ndarray:
    """Exact copy of 18_calibration_record_geo._square_positions's math."""
    h = side / 2.0
    offsets = [
        ( h,  0.0),   # 0 right midpoint
        ( h,    h),   # 1 top-right corner
        (0.0,   h),   # 2 top midpoint
        (-h,    h),   # 3 top-left corner
        (-h,  0.0),   # 4 left midpoint
        (-h,   -h),   # 5 bottom-left corner
        (0.0,  -h),   # 6 bottom midpoint
        ( h,   -h),   # 7 bottom-right corner
    ]
    vertices = [center + du * u + dv * v for (du, dv) in offsets]
    n_per_edge = max(1, round(h / velocity * fps))
    rows: List[np.ndarray] = []
    for _ in range(loops):
        for i in range(8):
            start = vertices[i]
            end = vertices[(i + 1) % 8]
            for k in range(1, n_per_edge + 1):
                t = k / n_per_edge
                rows.append(start + t * (end - start))
    return np.stack(rows) if rows else np.zeros((0, 3), dtype=np.float64)


def build_shape_ground_truth(
    anchor_pose6: np.ndarray,
    spec: ShapeTrajectorySpec,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct the commanded geometric trajectory from a ``ShapeTrajectorySpec``.

    Returns:
        positions  shape (N, 3) — per-frame commanded position
        orientation shape (N, 3) — per-frame commanded orientation (constant)
        grippers   shape (N,)   — per-frame commanded grip (crisp_py conv)
        phase_idx  shape (N,)   — integer phase label per frame (into PHASE_NAMES)
    """
    pos0 = np.asarray(anchor_pose6[:3], dtype=np.float64)
    rot0 = np.asarray(anchor_pose6[3:6], dtype=np.float64)

    u, v = PLANE_AXES[spec.plane]

    # First shape position (the approach endpoint).
    if spec.shape == "circle":
        first_pt = pos0 + spec.size * u
        shape_positions = _circle_positions_gt(
            pos0, spec.size, u, v, spec.velocity, spec.fps, spec.loops
        )
    else:
        first_pt = pos0 + (spec.size / 2.0) * u
        shape_positions = _square_positions_gt(
            pos0, spec.size, u, v, spec.velocity, spec.fps, spec.loops
        )

    # Phase assembly, mirroring 18_calibration_record_geo.build_shape_trajectory.
    pre_dwell_pos = _hold(pos0, spec.n_dwell)
    approach_pos = _segment(pos0, first_pt, spec.n_approach)
    return_pos = _segment(shape_positions[-1], pos0, spec.n_approach)
    post_dwell_pos = _hold(pos0, spec.n_dwell)

    positions = np.concatenate(
        [pre_dwell_pos, approach_pos, shape_positions, return_pos, post_dwell_pos],
        axis=0,
    )

    # Phase index per frame.
    n_shape = shape_positions.shape[0]
    toggle_idx = n_shape // 2 if spec.include_gripper_toggle else n_shape

    phase_idx = np.concatenate(
        [
            np.full(spec.n_dwell, PHASE_PRE_DWELL, dtype=np.int32),
            np.full(spec.n_approach, PHASE_APPROACH, dtype=np.int32),
            np.concatenate(
                [
                    np.full(toggle_idx, PHASE_SHAPE_OPEN, dtype=np.int32),
                    np.full(n_shape - toggle_idx, PHASE_SHAPE_CLOSED, dtype=np.int32),
                ]
            ),
            np.full(spec.n_approach, PHASE_RETURN, dtype=np.int32),
            np.full(spec.n_dwell, PHASE_POST_DWELL, dtype=np.int32),
        ]
    )

    orientation = np.tile(rot0.reshape(1, -1), (positions.shape[0], 1))

    # Gripper schedule — mirrors build_shape_trajectory in 18.
    OPEN = 1.0
    CLOSED = 0.0
    grips = np.full(positions.shape[0], OPEN, dtype=np.float64)
    if spec.include_gripper_toggle:
        grips[phase_idx == PHASE_SHAPE_CLOSED] = CLOSED
        grips[phase_idx == PHASE_RETURN] = CLOSED
        # post_dwell reopens (already OPEN by default)

    return positions, orientation, grips, phase_idx


# ---------------------------------------------------------------------------
# Dataset loading (shared with 17)
# ---------------------------------------------------------------------------

def load_dataset(repo_id: str) -> Tuple[pd.DataFrame, dict]:
    root = Path(HF_LEROBOT_HOME) / repo_id
    info_path = root / "meta" / "info.json"
    data_file = root / "data" / "chunk-000" / "file-000.parquet"

    if not data_file.exists():
        raise FileNotFoundError(
            f"No parquet file at {data_file}. Does the dataset exist?"
        )

    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    df = pd.read_parquet(data_file)

    if "episode_index" in df.columns:
        n_episodes = df["episode_index"].nunique()
        if n_episodes > 1:
            logger.warning(
                f"Dataset contains {n_episodes} episodes — report uses the first."
            )
        df = df[df["episode_index"] == df["episode_index"].iloc[0]].reset_index(drop=True)

    return df, info


# ---------------------------------------------------------------------------
# Analysis helpers (shared with 17)
# ---------------------------------------------------------------------------

def _stack_col(df: pd.DataFrame, col: str) -> np.ndarray:
    raw = df[col].to_numpy()
    if raw.size == 0:
        return np.empty((0, 0), dtype=np.float64)
    first = raw[0]
    if np.ndim(first) == 0:
        return np.asarray(raw, dtype=np.float64).reshape(-1, 1)
    return np.stack([np.asarray(v, dtype=np.float64) for v in raw])


def _fmt_vec(v: np.ndarray, n: int = 4, width: int = 8) -> str:
    return "[" + ", ".join(f"{x:+.{n}f}".rjust(width + n - 3) for x in v) + "]"


def _circular_std(angles: np.ndarray) -> float:
    if angles.size == 0:
        return 0.0
    c = np.cos(angles).mean()
    s = np.sin(angles).mean()
    r = float(np.hypot(c, s))
    r = min(max(r, 1e-12), 1.0)
    return float(np.sqrt(-2.0 * np.log(r)))


def _count_branch_flips(angles: np.ndarray, threshold: float = np.pi) -> int:
    if angles.size < 2:
        return 0
    d = np.diff(angles)
    return int(np.sum(np.abs(d) > threshold))


def _phase_ranges(phase_idx: np.ndarray) -> List[Tuple[int, int, int]]:
    ranges: List[Tuple[int, int, int]] = []
    if phase_idx.size == 0:
        return ranges
    start = 0
    cur = int(phase_idx[0])
    for i in range(1, phase_idx.size):
        if int(phase_idx[i]) != cur:
            ranges.append((cur, start, i))
            start = i
            cur = int(phase_idx[i])
    ranges.append((cur, start, phase_idx.size))
    return ranges


# ---------------------------------------------------------------------------
# Shape-specific metrics
# ---------------------------------------------------------------------------

def _radial_offset(
    pos: np.ndarray,
    center: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    normal: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (radial_distance, out_of_plane) for each position in the (u,v) plane.

    radial_distance is sqrt(<pos-center, u>^2 + <pos-center, v>^2).
    out_of_plane is the signed projection on the plane normal.
    """
    rel = pos - center
    ru = rel @ u
    rv = rel @ v
    rn = rel @ normal
    return np.sqrt(ru * ru + rv * rv), rn


def _circle_geometry(
    shape_pos: np.ndarray,
    center: np.ndarray,
    spec: ShapeTrajectorySpec,
) -> dict:
    """Commanded circle: radial distance should equal size everywhere, normal ≈ 0."""
    u, v = PLANE_AXES[spec.plane]
    n = PLANE_NORMAL[spec.plane]
    r, out = _radial_offset(shape_pos, center, u, v, n)
    return {
        "radial_mean": float(r.mean()),
        "radial_std": float(r.std()),
        "radial_max_err": float(np.max(np.abs(r - spec.size))),
        "normal_max_abs": float(np.max(np.abs(out))),
        "nominal_radius": float(spec.size),
    }


def _square_geometry(
    shape_pos: np.ndarray,
    center: np.ndarray,
    spec: ShapeTrajectorySpec,
) -> dict:
    """Commanded square: on each edge, one in-plane coordinate is constant at ±s/2.

    Returns stats on ("constant" coord deviation, edge lengths). Also reports
    max distance from center (should be <= s·√2/2 at corners).
    """
    u, v = PLANE_AXES[spec.plane]
    n_normal = PLANE_NORMAL[spec.plane]
    rel = shape_pos - center
    ru = rel @ u
    rv = rel @ v
    rn = rel @ n_normal
    in_plane_dist = np.sqrt(ru * ru + rv * rv)
    corner_dist = float(spec.size * np.sqrt(2.0) / 2.0)
    return {
        "u_min": float(ru.min()),
        "u_max": float(ru.max()),
        "v_min": float(rv.min()),
        "v_max": float(rv.max()),
        "in_plane_max": float(in_plane_dist.max()),
        "in_plane_expected_corner": corner_dist,
        "normal_max_abs": float(np.max(np.abs(rn))),
        "half_side": float(spec.size / 2.0),
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    repo_id: str,
    spec: ShapeTrajectorySpec,
    plot_dir: Optional[Path] = None,
) -> str:
    df, info = load_dataset(repo_id)
    n = len(df)

    if info.get("fps") and int(info["fps"]) != spec.fps:
        logger.warning(
            f"Dataset fps {info['fps']} != --fps {spec.fps}. "
            "Ground-truth phase boundaries may be offset."
        )
    if info.get("total_frames") and int(info["total_frames"]) != spec.n_total:
        logger.warning(
            f"Dataset total_frames {info['total_frames']} != expected "
            f"{spec.n_total} from spec. Did you change --shape / --size / "
            "--velocity / --loops / --dwell-seconds between record and report?"
        )

    action = _stack_col(df, "action")                             # (N, 7)
    cart = _stack_col(df, "observation.state.cartesian")          # (N, 6)
    target = _stack_col(df, "observation.state.target")           # (N, 6)
    grip_obs = _stack_col(df, "observation.state.gripper")        # (N, 1)
    grip_tgt = _stack_col(df, "observation.state.gripper_target") # (N, 1)

    # Ground truth anchored at action[0, :6] (frame 0 = first pre-dwell frame).
    anchor6 = action[0, :6].copy()
    gt_pos, gt_rot, gt_grip, phase_idx = build_shape_ground_truth(anchor6, spec)

    if gt_pos.shape[0] != n:
        logger.warning(
            f"Ground-truth length {gt_pos.shape[0]} != dataset length {n}. "
            "Spec arguments probably don't match the recording."
        )
        L = min(gt_pos.shape[0], n)
        action = action[:L]
        cart = cart[:L]
        target = target[:L]
        grip_obs = grip_obs[:L]
        grip_tgt = grip_tgt[:L]
        gt_pos = gt_pos[:L]
        gt_rot = gt_rot[:L]
        gt_grip = gt_grip[:L]
        phase_idx = phase_idx[:L]
        n = L

    gt_action6 = np.concatenate([gt_pos, gt_rot], axis=1)  # (N, 6)

    # --- 1. action vs ground truth ---
    act_gt_diff = action[:, :6] - gt_action6
    act_gt_pos_err = np.linalg.norm(act_gt_diff[:, :3], axis=1)
    act_gt_rot_err = np.linalg.norm(act_gt_diff[:, 3:], axis=1)

    # --- 2. orientation invariance ---
    action_rot_std = action[:, 3:6].std(axis=0)
    target_rot_std = target[:, 3:].std(axis=0)
    cart_rot_std_linear = cart[:, 3:].std(axis=0)
    cart_rot_std_circular = np.array(
        [_circular_std(cart[:, 3 + i]) for i in range(3)]
    )
    cart_branch_flips = np.array(
        [_count_branch_flips(cart[:, 3 + i]) for i in range(3)]
    )

    # --- 3. action vs recorded target ---
    act_tgt_diff = action[:, :6] - target
    act_tgt_max_pos = float(np.max(np.abs(act_tgt_diff[:, :3])))
    act_tgt_max_rot = float(np.max(np.abs(act_tgt_diff[:, 3:])))

    # --- 4. position fidelity ---
    pos_err = np.linalg.norm(action[:, :3] - cart[:, :3], axis=1)

    # --- 5. shape-specific geometry ---
    shape_mask = (phase_idx == PHASE_SHAPE_OPEN) | (phase_idx == PHASE_SHAPE_CLOSED)
    anchor_pos = anchor6[:3]
    shape_gt = gt_pos[shape_mask]
    shape_cart = cart[shape_mask][:, :3]
    shape_act = action[shape_mask][:, :3]

    if spec.shape == "circle":
        geom_cmd = _circle_geometry(shape_gt, anchor_pos, spec)
        geom_act = _circle_geometry(shape_act, anchor_pos, spec)
        # Measured cartesian: run the same analysis to see the realised shape.
        geom_meas = _circle_geometry(shape_cart, anchor_pos, spec)
    else:
        geom_cmd = _square_geometry(shape_gt, anchor_pos, spec)
        geom_act = _square_geometry(shape_act, anchor_pos, spec)
        geom_meas = _square_geometry(shape_cart, anchor_pos, spec)

    # Path lengths — commanded vs realised.
    def _path_length(p: np.ndarray) -> float:
        if p.shape[0] < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())

    cmd_path_total = _path_length(action[:, :3])
    meas_path_total = _path_length(cart[:, :3])
    cmd_path_shape = _path_length(shape_act)
    meas_path_shape = _path_length(shape_cart)

    # Closure gap — how far the measured cartesian "end of shape" is from
    # where the shape started (should be ~ 0 for a closed loop).
    closure_gap_cmd = 0.0
    closure_gap_meas = 0.0
    if shape_gt.shape[0] >= 2:
        closure_gap_cmd = float(np.linalg.norm(shape_gt[-1] - shape_gt[0]))
    if shape_cart.shape[0] >= 2:
        closure_gap_meas = float(np.linalg.norm(shape_cart[-1] - shape_cart[0]))

    # --- 6. gripper consistency ---
    grip_obs_crisp = 1.0 - grip_obs[:, 0]
    grip_tgt_crisp = 1.0 - grip_tgt[:, 0]
    grip_action = action[:, 6]
    grip_tgt_diff = np.abs(grip_action - grip_tgt_crisp)
    grip_cmd_vs_target_max = float(grip_tgt_diff.max())
    grip_cmd_vs_target_bad_frames = int(np.sum(grip_tgt_diff > 0.05))
    grip_cmd_vs_obs_max = float(np.max(np.abs(grip_action - grip_obs_crisp)))
    grip_transitions = np.where(np.abs(np.diff(grip_action)) > 0.5)[0]
    grip_transition_consistent = True
    for t in grip_transitions:
        if grip_tgt_diff[t] > 0.5 or grip_tgt_diff[t + 1] > 0.5:
            grip_transition_consistent = False
            break

    # --- 7. timing (synthetic) ---
    ts = df["timestamp"].to_numpy().astype(np.float64) if "timestamp" in df.columns else None
    ts_stats = None
    if ts is not None and len(ts) > 1:
        dt = np.diff(ts)
        ts_stats = {
            "span_s": float(ts[-1] - ts[0]),
            "mean_dt_s": float(dt.mean()),
            "median_dt_s": float(np.median(dt)),
            "max_dt_s": float(dt.max()),
            "min_dt_s": float(dt.min()),
            "effective_fps": float(1.0 / dt.mean()) if dt.mean() > 0 else float("nan"),
            "stddev_dt_s": float(dt.std()),
        }

    # Phase summary
    phase_summary_rows: List[dict] = []
    for pid, a, b in _phase_ranges(phase_idx):
        name = PHASE_NAMES[pid]
        if b == a:
            continue
        seg_pos_err = pos_err[a:b]
        seg_gt = gt_pos[a:b]
        gt_delta_start = (seg_gt[0] - anchor_pos) * 1000.0
        gt_delta_end = (seg_gt[-1] - anchor_pos) * 1000.0
        phase_summary_rows.append(
            {
                "id": pid,
                "name": name,
                "start": a,
                "end": b,
                "n": b - a,
                "rms_pos_err": float(np.sqrt(np.mean(seg_pos_err ** 2))),
                "max_pos_err": float(np.max(seg_pos_err)) if seg_pos_err.size else 0.0,
                "gt_delta_start_mm": gt_delta_start,
                "gt_delta_end_mm": gt_delta_end,
            }
        )

    # ------------------------------------------------------------------
    # Build the markdown.
    # ------------------------------------------------------------------

    lines: List[str] = []
    h = lines.append
    h(f"# Geometric calibration report — `{repo_id}`")
    h("")
    h(f"- Dataset path: `{Path(HF_LEROBOT_HOME) / repo_id}`")
    h(f"- Robot type: `{info.get('robot_type', 'unknown')}`")
    h(f"- Shape: **{spec.shape}**  size: **{spec.size} m**  plane: **{spec.plane}**")
    h(f"- Frames: **{n}**   (spec expected {spec.n_total})")
    h(f"- Nominal fps: **{info.get('fps', 'unknown')}**")
    h("")
    h("## Trajectory spec used for ground truth")
    h("")
    h("| param | value |")
    h("| --- | --- |")
    h(f"| shape | {spec.shape} |")
    h(f"| size (radius / side) | {spec.size} m |")
    h(f"| plane | {spec.plane} |")
    h(f"| velocity | {spec.velocity} m/s |")
    h(f"| loops | {spec.loops} |")
    h(f"| dwell_seconds | {spec.dwell_seconds} s |")
    h(f"| fps | {spec.fps} |")
    h(f"| include_gripper_toggle | {spec.include_gripper_toggle} |")
    h(f"| n_dwell (frames / dwell) | {spec.n_dwell} |")
    h(f"| n_approach (frames / approach) | {spec.n_approach} |")
    h(f"| n_per_loop (shape frames / loop) | {spec.n_per_loop} |")
    h(f"| n_shape (total shape frames) | {spec.n_shape} |")
    h(f"| n_total | {spec.n_total} |")
    h("")
    h("**If these parameters don't match the recording, every metric below "
      "is misaligned. Pass the same values to this script and the recorder.**")
    h("")
    h("## Anchor pose (frame 0 action[:6])")
    h("")
    h("```")
    h(f"position (m)  : {_fmt_vec(anchor6[:3])}")
    h(f"orientation   : {_fmt_vec(anchor6[3:])}  (Euler XYZ, rad)")
    h("```")
    h("")

    # --- Section 1 ---
    h("## 1. Action vs ground truth")
    h("")
    h("`action[:6]` should exactly equal the commanded setpoint at every "
      "frame. Non-zero diff means either the spec passed here doesn't "
      "match the recording, or `Pose.to_array` mangled the command.")
    h("")
    h("| metric | value |")
    h("| --- | --- |")
    h(f"| max ‖Δpos‖ (m) | {act_gt_pos_err.max():.6e} |")
    h(f"| mean ‖Δpos‖ (m) | {act_gt_pos_err.mean():.6e} |")
    h(f"| max ‖Δrot‖ (rad) | {act_gt_rot_err.max():.6e} |")
    h(f"| mean ‖Δrot‖ (rad) | {act_gt_rot_err.mean():.6e} |")
    h("")
    _assess_gt(h, act_gt_pos_err, act_gt_rot_err)

    # --- Section 2 ---
    h("## 2. Orientation invariance")
    h("")
    h("The recorder holds orientation constant. Linear std on `action[3:6]` "
      "and `observation.state.target[3:6]` should be ~0. Measured "
      "`observation.state.cartesian[3:6]` uses **circular std** — linear "
      "std blows up near Euler ±π wraparound even on a physically "
      "stationary rotation.")
    h("")
    h("| column | std kind | roll | pitch | yaw |")
    h("| --- | --- | --- | --- | --- |")
    h(f"| action[3:6]                      | linear   | {action_rot_std[0]:.3e} | {action_rot_std[1]:.3e} | {action_rot_std[2]:.3e} |")
    h(f"| observation.state.target[3:6]    | linear   | {target_rot_std[0]:.3e} | {target_rot_std[1]:.3e} | {target_rot_std[2]:.3e} |")
    h(f"| observation.state.cartesian[3:6] | linear   | {cart_rot_std_linear[0]:.3e} | {cart_rot_std_linear[1]:.3e} | {cart_rot_std_linear[2]:.3e} |")
    h(f"| observation.state.cartesian[3:6] | circular | {cart_rot_std_circular[0]:.3e} | {cart_rot_std_circular[1]:.3e} | {cart_rot_std_circular[2]:.3e} |")
    h("")
    h(f"**Euler-XYZ branch flips per axis** (|Δ| > π between consecutive frames): "
      f"roll={cart_branch_flips[0]}, pitch={cart_branch_flips[1]}, yaw={cart_branch_flips[2]}.")
    h("")
    _assess_orientation(
        h, action_rot_std, target_rot_std,
        cart_rot_std_linear, cart_rot_std_circular, cart_branch_flips,
    )

    # --- Section 3 ---
    h("## 3. Action vs recorded env target")
    h("")
    h("`observation.state.target[:6]` is what `env.robot.target_pose` "
      "returned at each frame. Since the recorder calls "
      "`env.robot.set_target(pose=...)` directly, this should equal "
      "`action[:6]` to sub-mm / sub-mrad.")
    h("")
    h("| metric | value |")
    h("| --- | --- |")
    h(f"| max |Δpos| per-axis (m) | {act_tgt_max_pos:.6e} |")
    h(f"| max |Δrot| per-axis (rad) | {act_tgt_max_rot:.6e} |")
    h("")
    _assess_act_vs_target(h, act_tgt_max_pos, act_tgt_max_rot)

    # --- Section 4 ---
    h("## 4. Position fidelity per phase")
    h("")
    h("Euclidean distance between commanded `action[:3]` and measured "
      "`observation.state.cartesian[:3]`. **Not a bug detector** — it "
      "measures cartesian impedance controller lag, which is proportional "
      "to speed / stiffness. Large values during the shape trace point at "
      "a slow `data_fn` (frame-rate warnings in the recorder log) or "
      "overly soft cartesian gains.")
    h("")
    h("| phase | frames | RMS err (mm) | max err (mm) | GT Δ start (mm) | GT Δ end (mm) |")
    h("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in phase_summary_rows:
        h(
            f"| `{row['name']}` | {row['start']}..{row['end']} ({row['n']}) | "
            f"{row['rms_pos_err']*1000:7.2f} | "
            f"{row['max_pos_err']*1000:7.2f} | "
            f"[{row['gt_delta_start_mm'][0]:+6.1f}, {row['gt_delta_start_mm'][1]:+6.1f}, {row['gt_delta_start_mm'][2]:+6.1f}] | "
            f"[{row['gt_delta_end_mm'][0]:+6.1f}, {row['gt_delta_end_mm'][1]:+6.1f}, {row['gt_delta_end_mm'][2]:+6.1f}] |"
        )
    h("")
    h(f"Overall: RMS cartesian error = **{np.sqrt(np.mean(pos_err ** 2)) * 1000:.2f} mm**, "
      f"max = **{pos_err.max() * 1000:.2f} mm**.")
    h("")
    h(f"Path length (commanded action[:3])                   : **{cmd_path_total:.4f} m**")
    h(f"Path length (measured observation.state.cartesian[:3]): **{meas_path_total:.4f} m**")
    if cmd_path_total > 1e-9:
        h(f"**Actual / commanded path length ratio: "
          f"{meas_path_total / cmd_path_total * 100:.1f} %**")
    h("")
    if cmd_path_shape > 1e-9:
        h(f"Within the shape trace only:")
        h(f"- commanded: {cmd_path_shape:.4f} m")
        h(f"- measured : {meas_path_shape:.4f} m  "
          f"({meas_path_shape / cmd_path_shape * 100:.1f} %)")
        h("")
    _assess_path_ratio(h, meas_path_total, cmd_path_total)

    # --- Section 5 ---
    h("## 5. Shape geometry")
    h("")
    h("Shape-specific checks on the **commanded** shape trace (ground "
      "truth regenerated here, and `action[:3]` as saved) plus the "
      "**measured** shape trace for comparison. The commanded stats are a "
      "sanity check on the recorder's math; the measured stats show how "
      "much the impedance controller deformed the shape.")
    h("")

    if spec.shape == "circle":
        h("### Circle")
        h("")
        h("Radial distance from the anchor in the shape plane. For a "
          f"{spec.size} m circle this must equal `size` everywhere; "
          "out-of-plane drift should be 0 (the recorder writes planar "
          "motion only).")
        h("")
        h("| source | mean radius (m) | std (m) | max |r − size| (m) | max |normal| (m) |")
        h("| --- | ---: | ---: | ---: | ---: |")
        h(f"| ground truth | {geom_cmd['radial_mean']:.6f} | {geom_cmd['radial_std']:.2e} | {geom_cmd['radial_max_err']:.2e} | {geom_cmd['normal_max_abs']:.2e} |")
        h(f"| action[:3]   | {geom_act['radial_mean']:.6f} | {geom_act['radial_std']:.2e} | {geom_act['radial_max_err']:.2e} | {geom_act['normal_max_abs']:.2e} |")
        h(f"| cartesian    | {geom_meas['radial_mean']:.6f} | {geom_meas['radial_std']:.6f} | {geom_meas['radial_max_err']:.6f} | {geom_meas['normal_max_abs']:.6f} |")
        h("")
        _assess_circle_geometry(h, geom_cmd, geom_act, geom_meas, spec.size)
    else:
        h("### Square")
        h("")
        h(f"In-plane offsets from anchor on `({spec.plane[0]}, {spec.plane[1]})` axes. "
          f"Expected: commanded u/v ∈ [±{spec.size/2:.3f}] (side/2), max in-plane "
          f"distance ≤ {spec.size * np.sqrt(2)/2:.4f} m at corners. "
          "Out-of-plane normal offset should be 0.")
        h("")
        h("| source | u range (m) | v range (m) | max in-plane (m) | max |normal| (m) |")
        h("| --- | --- | --- | ---: | ---: |")
        h(f"| ground truth | [{geom_cmd['u_min']:+.4f}, {geom_cmd['u_max']:+.4f}] | "
          f"[{geom_cmd['v_min']:+.4f}, {geom_cmd['v_max']:+.4f}] | "
          f"{geom_cmd['in_plane_max']:.6f} | {geom_cmd['normal_max_abs']:.2e} |")
        h(f"| action[:3]   | [{geom_act['u_min']:+.4f}, {geom_act['u_max']:+.4f}] | "
          f"[{geom_act['v_min']:+.4f}, {geom_act['v_max']:+.4f}] | "
          f"{geom_act['in_plane_max']:.6f} | {geom_act['normal_max_abs']:.2e} |")
        h(f"| cartesian    | [{geom_meas['u_min']:+.4f}, {geom_meas['u_max']:+.4f}] | "
          f"[{geom_meas['v_min']:+.4f}, {geom_meas['v_max']:+.4f}] | "
          f"{geom_meas['in_plane_max']:.6f} | {geom_meas['normal_max_abs']:.2e} |")
        h("")
        _assess_square_geometry(h, geom_cmd, geom_act, geom_meas, spec.size)

    h("")
    h("### Closure (last shape frame vs first shape frame)")
    h("")
    h("| metric | value (m) |")
    h("| --- | ---: |")
    h(f"| commanded closure gap | {closure_gap_cmd:.6e} |")
    h(f"| measured  closure gap | {closure_gap_meas:.6e} |")
    h("")
    if closure_gap_meas < 0.005:
        h(f"OK — measured closure gap is {closure_gap_meas*1000:.2f} mm.")
    elif closure_gap_meas < 0.02:
        h(f"ℹ  measured closure gap is {closure_gap_meas*1000:.2f} mm — within "
          "typical cartesian impedance settling error.")
    else:
        h(f"⚠  measured closure gap is {closure_gap_meas*1000:.2f} mm — too "
          "large to be controller lag alone. Check the recorder log for "
          "`Frame processing took too long` warnings and confirm the "
          "Orbbec camera is at 20 Hz (socket buffers fixed).")
    h("")

    # --- Section 6 ---
    h("## 6. Gripper consistency")
    h("")
    h("`action[6]` uses crisp_py convention (1=open, 0=closed) while "
      "`observation.state.gripper` and `.gripper_target` use LeRobot "
      "convention (0=open, 1=closed). This section flips the obs to "
      "crisp_py convention before subtracting.")
    h("")
    h("| comparison | max \u007cΔ\u007c | frames with \u007cΔ\u007c > 0.05 |")
    h("| --- | ---: | ---: |")
    h(f"| action[6] vs (1 − gripper_target) | {grip_cmd_vs_target_max:.4f} | {grip_cmd_vs_target_bad_frames} |")
    h(f"| action[6] vs (1 − gripper)        | {grip_cmd_vs_obs_max:.4f} | (measurement lag, not a bug) |")
    h("")
    n_transitions = int(grip_transitions.size)
    if n_transitions == 0:
        h("No gripper transitions in this episode.")
    elif grip_transition_consistent:
        h(f"✅ Transitions: `action[6]` == `gripper_target` at all "
          f"{n_transitions} transition boundaries and their follow-up frames.")
    else:
        h(f"❌ Transitions: `action[6]` ≠ `gripper_target` at one or more "
          "of the transition frames. The recorder is missing a "
          "`env.gripper.set_target(...)` call.")
    h("")
    if grip_cmd_vs_target_bad_frames > 0 and grip_transition_consistent:
        h(f"⚠  `action[6]` ≠ `gripper_target` on {grip_cmd_vs_target_bad_frames} "
          "**non-transition** frames. This is the **crisp_py gripper echo "
          "race** — see §5 of `17_calibration_report.py`'s output for the "
          "full explanation. The authoritative commanded value is "
          "`action[6]`; the obs column flickers due to a subscription "
          "echo. Does not affect replay.")
    elif grip_cmd_vs_target_bad_frames == 0:
        h("OK — `action[6]` and `gripper_target` agree on every frame.")
    h("")

    # --- Section 7 ---
    h("## 7. Timing — ⚠ synthetic timestamps")
    h("")
    h("**The `timestamp` column in LeRobot v3 parquet is `frame_index / "
      "fps`, not wall-clock.** It does NOT reflect how long the recorder "
      "actually spent in each frame. If the recorder logged "
      "`frame-too-long` warnings during the run, those are invisible "
      "here — the saved timestamps will still look perfect.")
    h("")
    if ts_stats is not None:
        h("| metric | value |")
        h("| --- | --- |")
        h(f"| span | {ts_stats['span_s']:.3f} s |")
        h(f"| mean dt | {ts_stats['mean_dt_s']*1000:.2f} ms |")
        h(f"| median dt | {ts_stats['median_dt_s']*1000:.2f} ms |")
        h(f"| max dt | {ts_stats['max_dt_s']*1000:.2f} ms |")
        h(f"| min dt | {ts_stats['min_dt_s']*1000:.2f} ms |")
        h(f"| stddev dt | {ts_stats['stddev_dt_s']*1000:.3f} ms |")
        h(f"| effective fps (1/mean dt) | {ts_stats['effective_fps']:.2f} |")
        h("")
        if ts_stats["stddev_dt_s"] < 1e-4:
            h("Timestamps are uniformly spaced (synthetic). True wall-clock "
              "timing was not captured. To measure real per-frame jitter "
              "you need to log `time.monotonic()` inside `data_fn` during "
              "recording and diff post-hoc.")
    else:
        h("No `timestamp` column found in the parquet.")
    h("")

    # --- Summary ---
    h("## Summary")
    h("")
    verdicts: List[str] = []
    if act_gt_pos_err.max() < 1e-5 and act_gt_rot_err.max() < 1e-5:
        verdicts.append("✅ action = ground truth")
    else:
        verdicts.append(
            f"❌ action ≠ ground truth "
            f"(max pos err {act_gt_pos_err.max():.2e} m, rot err {act_gt_rot_err.max():.2e} rad)"
        )

    if action_rot_std.max() < 1e-5:
        verdicts.append("✅ action orientation is constant")
    else:
        verdicts.append(f"❌ action orientation wobbles (max std {action_rot_std.max():.2e})")

    if target_rot_std.max() < 1e-5:
        verdicts.append("✅ recorded target orientation is constant")
    else:
        verdicts.append(
            f"❌ recorded target orientation wobbles "
            f"(max std {target_rot_std.max():.2e}) — representation bug"
        )

    if act_tgt_max_pos < 1e-4 and act_tgt_max_rot < 1e-4:
        verdicts.append("✅ action == env.robot.target_pose")
    else:
        verdicts.append(
            f"❌ action ≠ env.robot.target_pose "
            f"(max pos {act_tgt_max_pos:.2e}, rot {act_tgt_max_rot:.2e})"
        )

    if cart_rot_std_linear.max() > 0.5 and cart_rot_std_circular.max() < 0.05:
        verdicts.append(
            f"⚠  measured cartesian orientation is stable angularly "
            f"(circular std {cart_rot_std_circular.max():.2e}) but "
            f"linear std is {cart_rot_std_linear.max():.2f} rad "
            f"and there are {int(cart_branch_flips.sum())} Euler-XYZ "
            "branch flips — representation artifact, "
            "see `docs/ridgeback_replay_orientation_bug.md`"
        )
    else:
        verdicts.append(
            f"ℹ  measured cartesian orientation circular std "
            f"{cart_rot_std_circular.max():.2e} rad, "
            f"{int(cart_branch_flips.sum())} branch flips"
        )

    if cmd_path_total > 1e-9:
        ratio = meas_path_total / cmd_path_total * 100
        if ratio < 70:
            verdicts.append(
                f"❌ measured path length is only {ratio:.1f}% of commanded — "
                "the arm did not follow. Check recorder frame-rate warnings "
                "and the Orbbec socket buffer setting."
            )
        elif ratio < 90:
            verdicts.append(
                f"⚠  measured path length is {ratio:.1f}% of commanded — "
                "controller lag or slow data_fn"
            )
        else:
            verdicts.append(f"✅ measured path length is {ratio:.1f}% of commanded")

    if closure_gap_meas < 0.005:
        verdicts.append(f"✅ shape closes within {closure_gap_meas*1000:.2f} mm")
    elif closure_gap_meas < 0.02:
        verdicts.append(f"ℹ  shape closes within {closure_gap_meas*1000:.2f} mm (controller settling)")
    else:
        verdicts.append(f"⚠  shape closure gap {closure_gap_meas*1000:.2f} mm (large)")

    if grip_transitions.size > 0 and grip_transition_consistent and grip_cmd_vs_target_bad_frames > 0:
        verdicts.append(
            f"⚠  gripper: transitions clean but "
            f"{grip_cmd_vs_target_bad_frames} non-transition frames flicker"
        )
    elif grip_transitions.size > 0 and grip_transition_consistent:
        verdicts.append("✅ gripper transitions and frames clean")
    elif grip_transitions.size > 0:
        verdicts.append("❌ gripper transitions inconsistent")

    verdicts.append(
        f"ℹ  RMS cartesian tracking error {np.sqrt(np.mean(pos_err ** 2)) * 1000:.2f} mm "
        "(controller lag)"
    )

    for v in verdicts:
        h(f"- {v}")
    h("")

    report = "\n".join(lines)

    if plot_dir is not None:
        _write_plots(plot_dir, action, cart, target, gt_pos, phase_idx, spec, anchor_pos)

    return report


# ---------------------------------------------------------------------------
# Section assessors
# ---------------------------------------------------------------------------

def _assess_gt(h, pos_err: np.ndarray, rot_err: np.ndarray) -> None:
    if pos_err.max() < 1e-5 and rot_err.max() < 1e-5:
        h("OK — action exactly matches ground truth (float64 noise only).")
    else:
        h("⚠  action diverges from ground truth. Most likely the "
          "`--shape` / `--size` / `--velocity` / `--loops` / "
          "`--dwell-seconds` / `--fps` / `--include-gripper-toggle` "
          "passed here don't match the recording. Check the recorder "
          "log and re-run this script with matching values.")
    h("")


def _assess_orientation(
    h, action_std: np.ndarray, target_std: np.ndarray,
    cart_linear: np.ndarray, cart_circular: np.ndarray, branch_flips: np.ndarray,
) -> None:
    if action_std.max() < 1e-6:
        h("OK — `action[3:6]` is exactly constant (good).")
    else:
        h("❌  `action[3:6]` is NOT constant. Representation bug in the "
          "recorder's `Pose.to_array(euler)` path.")
    if target_std.max() < 1e-6:
        h("OK — `observation.state.target[3:6]` is exactly constant (good).")
    else:
        h("❌  `observation.state.target[3:6]` is NOT constant. "
          "`env.robot.set_target` or `.target_pose` is mutating the orientation.")

    linear_max = float(cart_linear.max())
    circular_max = float(cart_circular.max())
    flips_total = int(branch_flips.sum())
    if linear_max > 0.5 and circular_max < 0.05:
        h(f"⚠  Measured cartesian orientation has huge linear std "
          f"({linear_max:.2f} rad) but tiny circular std "
          f"({circular_max:.2e} rad), and {flips_total} Euler-XYZ branch "
          "flips. The physical rotation is essentially constant; the "
          "linear std is a **representation artifact**. See "
          "`docs/ridgeback_replay_orientation_bug.md`.")
    elif circular_max > 0.02:
        h(f"⚠  Measured cartesian orientation circular std = "
          f"{circular_max:.3e} rad — physical drift across frames.")
    else:
        h(f"OK — measured cartesian orientation is stable "
          f"(circular std ≤ {circular_max:.2e} rad, {flips_total} branch flips).")
    h("")


def _assess_act_vs_target(h, max_pos: float, max_rot: float) -> None:
    if max_pos < 1e-4 and max_rot < 1e-4:
        h("OK — action matches recorded env target to sub-mm / sub-mrad.")
    else:
        h("⚠  action diverges from `observation.state.target`. Check "
          "`env.robot.set_target` and `env.robot.target_pose` for mutation.")
    h("")


def _assess_path_ratio(h, meas: float, cmd: float) -> None:
    if cmd < 1e-9:
        return
    ratio = meas / cmd * 100
    if ratio >= 90:
        h(f"OK — arm tracked {ratio:.1f}% of the commanded path length.")
    elif ratio >= 70:
        h(f"⚠  arm tracked only {ratio:.1f}% of the commanded path length. "
          "Possible causes: cartesian impedance too soft, slow `data_fn` "
          "(check recorder frame-rate warnings), or Orbbec socket buffer "
          "not fixed.")
    else:
        h(f"❌  arm tracked only {ratio:.1f}% of the commanded path length. "
          "The recording is not usable for calibration. Root cause is "
          "almost certainly a slow `data_fn` (Orbbec blocking) — re-run "
          "after `sudo sysctl -w net.core.rmem_max=2147483647; "
          "sudo sysctl -w net.core.wmem_max=2147483647` and verify the "
          "recorder logs zero `Frame processing took too long` warnings.")
    h("")


def _assess_circle_geometry(h, cmd: dict, act: dict, meas: dict, size: float) -> None:
    if cmd["radial_max_err"] > 1e-9:
        h(f"❌  commanded radial distance error {cmd['radial_max_err']:.2e} m — "
          "ground-truth generator is broken (please report).")
    if act["radial_max_err"] > 1e-6:
        h(f"⚠  `action[:3]` radial error on the commanded circle is "
          f"{act['radial_max_err']:.2e} m — recorder is mutating positions.")
    meas_err = meas["radial_max_err"]
    if meas_err < 0.01:
        h(f"OK — measured circle radial error ≤ {meas_err*1000:.2f} mm "
          "(within typical cartesian impedance tracking).")
    elif meas_err < 0.03:
        h(f"ℹ  measured circle radial error ≤ {meas_err*1000:.2f} mm — "
          "controller lag, cleanly recognisable shape.")
    else:
        h(f"❌  measured circle radial error ≤ {meas_err*1000:.2f} mm — "
          f"expected {size*1000:.1f} mm radius, actual shape is "
          "heavily deformed. Check recorder frame-rate warnings.")
    h("")


def _assess_square_geometry(h, cmd: dict, act: dict, meas: dict, size: float) -> None:
    # Commanded should be exactly ±s/2 on u/v, and max in-plane distance ~ s·√2/2.
    expected_half = size / 2.0
    expected_corner = size * np.sqrt(2.0) / 2.0
    cmd_corner_err = abs(cmd["in_plane_max"] - expected_corner)
    if cmd_corner_err > 1e-9:
        h(f"❌  commanded corner distance off by {cmd_corner_err:.2e} m — "
          "ground-truth generator is broken (please report).")
    act_u_span = (act["u_max"] - act["u_min"]) / 2.0
    act_v_span = (act["v_max"] - act["v_min"]) / 2.0
    u_err = abs(act_u_span - expected_half)
    v_err = abs(act_v_span - expected_half)
    if max(u_err, v_err) > 1e-6:
        h(f"⚠  `action[:3]` square half-span error: u={u_err:.2e} m, "
          f"v={v_err:.2e} m — recorder is mutating positions.")
    meas_u = (meas["u_max"] - meas["u_min"]) / 2.0
    meas_v = (meas["v_max"] - meas["v_min"]) / 2.0
    meas_u_ratio = meas_u / expected_half * 100
    meas_v_ratio = meas_v / expected_half * 100
    if meas_u_ratio >= 90 and meas_v_ratio >= 90:
        h(f"OK — measured square tracks commanded size "
          f"(u {meas_u_ratio:.1f}%, v {meas_v_ratio:.1f}% of half-side).")
    elif meas_u_ratio >= 70 and meas_v_ratio >= 70:
        h(f"⚠  measured square is shrunk: u {meas_u_ratio:.1f}%, "
          f"v {meas_v_ratio:.1f}% of commanded half-side. Controller lag "
          "or slow data_fn.")
    else:
        h(f"❌  measured square is heavily shrunk: u {meas_u_ratio:.1f}%, "
          f"v {meas_v_ratio:.1f}% of commanded half-side. Recording is "
          "not usable for calibration — check Orbbec socket buffers and "
          "recorder frame-rate warnings.")
    h("")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _write_plots(
    plot_dir: Path,
    action: np.ndarray,
    cart: np.ndarray,
    target: np.ndarray,
    gt_pos: np.ndarray,
    phase_idx: np.ndarray,
    spec: ShapeTrajectorySpec,
    anchor_pos: np.ndarray,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping --plot output.")
        return

    plot_dir.mkdir(parents=True, exist_ok=True)
    frames = np.arange(action.shape[0])

    # --- Position traces ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for i, (ax, label) in enumerate(zip(axes, ("x", "y", "z"))):
        ax.plot(frames, gt_pos[:, i], label="ground truth", linewidth=2.0, alpha=0.6)
        ax.plot(frames, action[:, i], label="action", linewidth=1.0, linestyle="--")
        ax.plot(frames, target[:, i], label="target (obs)", linewidth=1.0, linestyle=":")
        ax.plot(frames, cart[:, i], label="cartesian (measured)", linewidth=1.0)
        ax.set_ylabel(f"{label} (m)")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("frame index")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"Geometric calibration — position traces ({spec.shape}, size={spec.size} m)")
    fig.tight_layout()
    fig.savefig(plot_dir / "position_traces.png", dpi=120)
    plt.close(fig)

    # --- Orientation traces ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for i, (ax, label) in enumerate(zip(axes, ("roll", "pitch", "yaw"))):
        ax.plot(frames, action[:, 3 + i], label="action", linewidth=1.0, linestyle="--")
        ax.plot(frames, target[:, 3 + i], label="target (obs)", linewidth=1.0, linestyle=":")
        ax.plot(frames, cart[:, 3 + i], label="cartesian (measured)", linewidth=1.0)
        ax.set_ylabel(f"{label} (rad)")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("frame index")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Geometric calibration — orientation traces (action/target should be flat)")
    fig.tight_layout()
    fig.savefig(plot_dir / "orientation_traces.png", dpi=120)
    plt.close(fig)

    # --- Cartesian tracking error ---
    err = np.linalg.norm(action[:, :3] - cart[:, :3], axis=1) * 1000
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(frames, err, linewidth=1.0)
    ax.set_xlabel("frame index")
    ax.set_ylabel("‖action[:3] − cartesian[:3]‖ (mm)")
    ax.set_title("Cartesian tracking error (controller lag)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "tracking_error.png", dpi=120)
    plt.close(fig)

    # --- Top-down (u, v) shape overlay ---
    u, v = PLANE_AXES[spec.plane]
    def _proj(p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rel = p - anchor_pos
        return rel @ u, rel @ v

    shape_mask = (phase_idx == PHASE_SHAPE_OPEN) | (phase_idx == PHASE_SHAPE_CLOSED)
    gu_shape, gv_shape = _proj(gt_pos[shape_mask])
    au_shape, av_shape = _proj(action[shape_mask][:, :3])
    cu_shape, cv_shape = _proj(cart[shape_mask][:, :3])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(gu_shape, gv_shape, label="ground truth", linewidth=2.5, alpha=0.5)
    ax.plot(au_shape, av_shape, label="action", linewidth=1.0, linestyle="--")
    ax.plot(cu_shape, cv_shape, label="cartesian (measured)", linewidth=1.0)
    ax.scatter([0], [0], color="black", s=40, zorder=5, label="anchor")
    ax.set_xlabel(f"u = {spec.plane[0]} offset (m)")
    ax.set_ylabel(f"v = {spec.plane[1]} offset (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(
        f"Shape overlay — {spec.shape} size={spec.size} m, plane={spec.plane}"
    )
    fig.tight_layout()
    fig.savefig(plot_dir / "shape_overlay.png", dpi=120)
    plt.close(fig)

    logger.info(f"Plots written to {plot_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--repo-id", type=str, default="calib_geo_square_001")
    parser.add_argument(
        "--shape", type=str, choices=["circle", "square"], required=True,
    )
    parser.add_argument("--size", type=float, required=True)
    parser.add_argument(
        "--plane", type=str, choices=["xy", "xz", "yz"], default="xy",
    )
    parser.add_argument("--velocity", type=float, default=0.05)
    parser.add_argument("--loops", type=int, default=1)
    parser.add_argument("--dwell-seconds", type=float, default=0.5)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--include-gripper-toggle",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Directory to write the report + plots into (default: stdout only).",
    )
    parser.add_argument(
        "--plot", action="store_true", default=False,
        help="Also write position/orientation/tracking/shape plots (requires matplotlib).",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(levelname)s %(message)s",
    )

    spec = ShapeTrajectorySpec(
        shape=args.shape,
        size=args.size,
        plane=args.plane,
        velocity=args.velocity,
        fps=args.fps,
        loops=args.loops,
        dwell_seconds=args.dwell_seconds,
        include_gripper_toggle=args.include_gripper_toggle,
    )

    out_dir: Optional[Path] = None
    plot_dir: Optional[Path] = None
    if args.out is not None:
        out_dir = Path(args.out).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.plot:
            plot_dir = out_dir
    elif args.plot:
        plot_dir = Path(HF_LEROBOT_HOME) / args.repo_id / "analysis"

    try:
        report = generate_report(args.repo_id, spec, plot_dir=plot_dir)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(2)

    print(report)

    if out_dir is not None:
        report_path = out_dir / f"{args.repo_id}_calibration_report.md"
        report_path.write_text(report)
        logger.info(f"Report written to {report_path}")
        if plot_dir is not None:
            logger.info(f"Plots written to {plot_dir}")
    elif plot_dir is not None:
        logger.info(f"Plots written to {plot_dir}")


if __name__ == "__main__":
    main()
