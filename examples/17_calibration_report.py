#!/usr/bin/env python3
"""Offline calibration report for `16_calibration_record.py` datasets.

Loads a LeRobot v3 dataset produced by `16_calibration_record.py`,
regenerates the ground-truth commanded trajectory from the same CLI
arguments, and prints a markdown report comparing the recorded columns
against the ground truth. Optionally writes plots to an analysis
directory alongside the dataset.

This is a **read-only** analyzer. It never touches the robot, never
writes to the dataset, and never imports from `crisp_gym` or `crisp_py`
(no ROS stack required). You can run it in any env that has `numpy`,
`pandas`, and `pyarrow`. For plots, also `matplotlib`.

Usage:
    # Default — dataset at ~/.cache/huggingface/lerobot/calib_axis_001,
    # same trajectory settings as the recorder's defaults.
    python examples/17_calibration_report.py --repo-id calib_axis_001

    # Non-default trajectory settings (must match what was recorded).
    python examples/17_calibration_report.py \\
        --repo-id calib_axis_slow \\
        --velocity 0.02 --displacement 0.10 --dwell-seconds 0.5 --fps 20

    # Also write report + plots to disk.
    python examples/17_calibration_report.py --repo-id calib_axis_001 \\
        --out analysis --plot

The trajectory-shape arguments (`--velocity`, `--displacement`,
`--dwell-seconds`, `--fps`, `--include-gripper-toggle`,
`--no-include-gripper-toggle`) must match the arguments passed to the
recorder. If they don't, the ground-truth segment boundaries will be
offset and the report's "action vs ground truth" diff will blow up.

What the report surfaces (see §4 of
`docs/ridgeback_calibration_recording_plan.md`):

1. Action vs ground truth — are the saved commands exactly what the
   script intended?
2. Orientation invariance — the recorder holds orientation constant,
   so `std(action[3:6])`, `std(target[3:6])`, and ideally
   `std(cartesian[3:6])` should all be tiny. Nonzero std on action or
   target is a representation bug.
3. Action vs recorded env target — `action[:6]` should equal
   `observation.state.target[:6]` to sub-mm / sub-mrad.
4. Position fidelity — per phase, measured `cartesian[:3]` vs commanded
   `action[:3]`. Lag/offset is controller stiffness, not a bug, but is
   useful to quantify.
5. Gripper consistency — `action[6]` (crisp_py: 1=open) vs
   `observation.state.gripper_target` and `observation.state.gripper`
   (LeRobot: 0=open). Done with an explicit convention flip.
6. Timing — the parquet `timestamp` column is **synthetic** (nominal
   `frame_index / fps`), so per-frame wall-clock jitter is invisible
   here. The report says so explicitly; real timing data needs a
   side-channel log during recording.
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

try:
    from lerobot.utils.constants import HF_LEROBOT_HOME
except ImportError:  # older lerobot
    try:
        from lerobot.constants import HF_LEROBOT_HOME
    except ImportError:  # lerobot not installed — fall back to the default path
        HF_LEROBOT_HOME = Path.home() / ".cache" / "huggingface" / "lerobot"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ground-truth trajectory (pure numpy, mirrors 16_calibration_record.py)
# ---------------------------------------------------------------------------

@dataclass
class TrajectorySpec:
    velocity: float
    displacement: float
    dwell_seconds: float
    fps: int
    include_gripper_toggle: bool

    @property
    def n_motion(self) -> int:
        return max(1, round((self.displacement / self.velocity) * self.fps))

    @property
    def n_dwell(self) -> int:
        return max(1, round(self.dwell_seconds * self.fps))

    @property
    def n_total(self) -> int:
        return 8 * self.n_motion + 9 * self.n_dwell


# Logical phase names in order. Index matches the segment index in
# ``build_ground_truth``, so (name, kind, start, end) tuples can be derived
# by walking the segment list.
PHASE_NAMES: List[str] = [
    "dwell_0_anchor",
    "move_1_plus_z",
    "dwell_1_plus_z",
    "move_2_return",
    "dwell_2_anchor",
    "move_3_minus_z",
    "dwell_3_minus_z",
    "move_4_return",
    "dwell_4_anchor",
    "move_5_plus_y",
    "dwell_5_plus_y",
    "move_6_return",
    "dwell_6_anchor",
    "move_7_minus_y",
    "dwell_7_minus_y",
    "move_8_return",
    "dwell_8_anchor",
]


def _segment(start: np.ndarray, end: np.ndarray, n: int) -> np.ndarray:
    ts = np.linspace(1.0 / n, 1.0, n).reshape(-1, 1)
    return start + ts * (end - start)


def _hold(p: np.ndarray, n: int) -> np.ndarray:
    return np.tile(p.reshape(1, -1), (n, 1))


def build_ground_truth(
    anchor_pose6: np.ndarray,
    spec: TrajectorySpec,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct the commanded trajectory from the anchor and a ``TrajectorySpec``.

    Returns:
        positions  shape (N, 3) — per-frame commanded position.
        orientation shape (N, 3) — per-frame commanded orientation (constant).
        grippers   shape (N,)   — per-frame commanded grip (crisp_py conv).
        phase_idx  shape (N,)   — integer phase label per frame, into
                                    ``PHASE_NAMES``.
    """
    pos0 = np.asarray(anchor_pose6[:3], dtype=np.float64)
    rot0 = np.asarray(anchor_pose6[3:6], dtype=np.float64)

    p_up = pos0 + np.array([0.0, 0.0, +spec.displacement])
    p_down = pos0 + np.array([0.0, 0.0, -spec.displacement])
    p_left = pos0 + np.array([0.0, +spec.displacement, 0.0])
    p_right = pos0 + np.array([0.0, -spec.displacement, 0.0])

    segments: List[np.ndarray] = [
        _hold(pos0, spec.n_dwell),                 # 0  dwell_0_anchor
        _segment(pos0, p_up, spec.n_motion),       # 1  move_1_plus_z
        _hold(p_up, spec.n_dwell),                 # 2  dwell_1_plus_z
        _segment(p_up, pos0, spec.n_motion),       # 3  move_2_return
        _hold(pos0, spec.n_dwell),                 # 4  dwell_2_anchor
        _segment(pos0, p_down, spec.n_motion),     # 5  move_3_minus_z
        _hold(p_down, spec.n_dwell),               # 6  dwell_3_minus_z
        _segment(p_down, pos0, spec.n_motion),     # 7  move_4_return
        _hold(pos0, spec.n_dwell),                 # 8  dwell_4_anchor
        _segment(pos0, p_left, spec.n_motion),     # 9  move_5_plus_y
        _hold(p_left, spec.n_dwell),               # 10 dwell_5_plus_y
        _segment(p_left, pos0, spec.n_motion),     # 11 move_6_return
        _hold(pos0, spec.n_dwell),                 # 12 dwell_6_anchor
        _segment(pos0, p_right, spec.n_motion),    # 13 move_7_minus_y
        _hold(p_right, spec.n_dwell),              # 14 dwell_7_minus_y
        _segment(p_right, pos0, spec.n_motion),    # 15 move_8_return
        _hold(pos0, spec.n_dwell),                 # 16 dwell_8_anchor
    ]
    positions = np.concatenate(segments, axis=0)
    orientation = np.tile(rot0.reshape(1, -1), (positions.shape[0], 1))

    phase_idx = np.concatenate(
        [np.full(seg.shape[0], i, dtype=np.int32) for i, seg in enumerate(segments)]
    )

    # Gripper schedule — mirrors `build_trajectory` in 16_calibration_record.py.
    OPEN = 1.0
    CLOSED = 0.0
    grips = np.full(positions.shape[0], OPEN, dtype=np.float64)
    if spec.include_gripper_toggle:
        # Closed for phases 9..15 (move_5_plus_y through move_8_return),
        # open elsewhere including the final dwell_8_anchor.
        closed_mask = (phase_idx >= 9) & (phase_idx <= 15)
        grips[closed_mask] = CLOSED

    return positions, orientation, grips, phase_idx


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(repo_id: str) -> Tuple[pd.DataFrame, dict]:
    """Load a single-episode calibration dataset as a DataFrame + info dict.

    We read the parquet file directly rather than going through
    ``LeRobotDataset`` — that avoids pulling in torch, lerobot-specific
    transforms, and video decoding, none of which we need for this analysis.
    """
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
# Analysis metrics
# ---------------------------------------------------------------------------

def _stack_col(df: pd.DataFrame, col: str) -> np.ndarray:
    """Stack a pandas column into a 2D (N, K) float64 array.

    Handles the case where LeRobot parquet stores a 1-element feature
    (e.g. gripper) as a scalar instead of a 1-element array.
    """
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
    """Circular standard deviation of an angle column, in radians.

    Uses the unit-circle definition: std = sqrt(-2 ln R) where R is the
    mean resultant length. Robust to wraparound across ±π, which plain
    `np.std` is not — a roll angle that oscillates between -3.14 and
    +3.14 is angularly near-constant but has huge linear std.
    """
    if angles.size == 0:
        return 0.0
    c = np.cos(angles).mean()
    s = np.sin(angles).mean()
    r = float(np.hypot(c, s))
    r = min(max(r, 1e-12), 1.0)
    return float(np.sqrt(-2.0 * np.log(r)))


def _count_branch_flips(angles: np.ndarray, threshold: float = np.pi) -> int:
    """Count frame-to-frame deltas larger than ``threshold`` — proxy for
    Euler-XYZ branch flips at ±π.
    """
    if angles.size < 2:
        return 0
    d = np.diff(angles)
    return int(np.sum(np.abs(d) > threshold))


def _phase_ranges(phase_idx: np.ndarray) -> List[Tuple[int, int, int]]:
    """Return (phase_id, start_frame, end_frame_exclusive) tuples."""
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
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    repo_id: str,
    spec: TrajectorySpec,
    plot_dir: Path | None = None,
) -> str:
    """Generate the markdown report for ``repo_id`` given the trajectory spec."""
    df, info = load_dataset(repo_id)
    n = len(df)

    if info.get("fps") and int(info["fps"]) != spec.fps:
        logger.warning(
            f"Dataset fps {info['fps']} != --fps {spec.fps}. "
            "Ground-truth segment boundaries may be offset."
        )
    if info.get("total_frames") and int(info["total_frames"]) != spec.n_total:
        logger.warning(
            f"Dataset total_frames {info['total_frames']} != expected "
            f"{spec.n_total} from spec. Did you change --velocity / "
            "--displacement / --dwell-seconds between record and report?"
        )

    action = _stack_col(df, "action")                          # (N, 7)
    cart = _stack_col(df, "observation.state.cartesian")       # (N, 6)
    target = _stack_col(df, "observation.state.target")        # (N, 6)
    joints = _stack_col(df, "observation.state.joints")        # (N, 6)
    grip_obs = _stack_col(df, "observation.state.gripper")     # (N, 1)
    grip_tgt = _stack_col(df, "observation.state.gripper_target")  # (N, 1)

    # Ground truth. We use action[0, :6] as the authoritative anchor pose,
    # because that's what the recorder commanded on frame 0 (first dwell).
    anchor6 = action[0, :6].copy()
    gt_pos, gt_rot, gt_grip, phase_idx = build_ground_truth(anchor6, spec)

    if gt_pos.shape[0] != n:
        logger.warning(
            f"Ground-truth length {gt_pos.shape[0]} != dataset length {n}. "
            "Spec arguments probably don't match the recording."
        )
        # Clip both to the shorter for downstream metrics to still work.
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

    # --- 2. orientation invariance (should be exactly constant) ---
    # Linear std on the commanded columns is correct because the recorder
    # writes a literally constant value, so there is no wraparound — any
    # nonzero std is a real representation bug. For the *measured*
    # cartesian column we must use circular std; the Euler-XYZ conversion
    # hits ±π branch flips near the singularity even when the physical
    # rotation is nearly stationary.
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
    act_tgt_max_pos = np.max(np.abs(act_tgt_diff[:, :3]))
    act_tgt_max_rot = np.max(np.abs(act_tgt_diff[:, 3:]))

    # --- 4. position fidelity (per phase) ---
    pos_err = np.linalg.norm(action[:, :3] - cart[:, :3], axis=1)

    # --- 5. gripper consistency ---
    # Flip LeRobot convention -> crisp_py convention for the comparison.
    grip_obs_crisp = 1.0 - grip_obs[:, 0]
    grip_tgt_crisp = 1.0 - grip_tgt[:, 0]
    grip_action = action[:, 6]
    grip_tgt_diff = np.abs(grip_action - grip_tgt_crisp)
    grip_cmd_vs_target_max = float(grip_tgt_diff.max())
    grip_cmd_vs_target_bad_frames = int(np.sum(grip_tgt_diff > 0.05))
    # grip_action vs grip_obs is expected to lag — the obs is the *measured*
    # gripper position after the command. We report the max but don't flag.
    grip_cmd_vs_obs_max = float(np.max(np.abs(grip_action - grip_obs_crisp)))
    # Transition consistency: at every frame where action[6] changes value,
    # the immediately-following frame should also have a matching
    # gripper_target (mod the crisp_py echo bug). We check this explicitly
    # so the reader can distinguish "no transitions captured" from
    # "sporadic echo races".
    grip_transitions = np.where(np.abs(np.diff(grip_action)) > 0.5)[0]
    grip_transition_consistent = True
    for t in grip_transitions:
        # compare frame t and t+1 flipped values
        if grip_tgt_diff[t] > 0.5 or grip_tgt_diff[t + 1] > 0.5:
            grip_transition_consistent = False
            break

    # --- 6. timing (synthetic) ---
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

    # Phase segmentation summary
    phase_summary_rows: List[dict] = []
    for pid, a, b in _phase_ranges(phase_idx):
        name = PHASE_NAMES[pid]
        seg_pos_err = pos_err[a:b]
        gt_start = gt_pos[a] - gt_pos[0]  # offset from anchor at segment start
        gt_end = gt_pos[b - 1] - gt_pos[0]  # offset from anchor at segment end
        phase_summary_rows.append(
            {
                "id": pid,
                "name": name,
                "start": a,
                "end": b,
                "n": b - a,
                "rms_pos_err": float(np.sqrt(np.mean(seg_pos_err ** 2))),
                "max_pos_err": float(np.max(seg_pos_err)) if seg_pos_err.size else 0.0,
                "gt_delta_start_mm": gt_start * 1000,
                "gt_delta_end_mm": gt_end * 1000,
            }
        )

    # Build the markdown.
    lines: List[str] = []
    h = lines.append
    h(f"# Calibration report — `{repo_id}`")
    h("")
    h(f"- Dataset path: `{Path(HF_LEROBOT_HOME) / repo_id}`")
    h(f"- Robot type: `{info.get('robot_type', 'unknown')}`")
    h(f"- Frames: **{n}**   (spec expected {spec.n_total})")
    h(f"- Nominal fps: **{info.get('fps', 'unknown')}**")
    h("")
    h("## Trajectory spec used for ground truth")
    h("")
    h("| param | value |")
    h("| --- | --- |")
    h(f"| velocity | {spec.velocity} m/s |")
    h(f"| displacement | {spec.displacement} m |")
    h(f"| dwell_seconds | {spec.dwell_seconds} s |")
    h(f"| fps | {spec.fps} |")
    h(f"| include_gripper_toggle | {spec.include_gripper_toggle} |")
    h(f"| n_motion (frames / phase) | {spec.n_motion} |")
    h(f"| n_dwell (frames / dwell) | {spec.n_dwell} |")
    h(f"| n_total (8·n_motion + 9·n_dwell) | {spec.n_total} |")
    h("")
    h("**If these parameters don't match the recording, every metric below is "
      "misaligned. Pass the same values to this script and the recorder.**")
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
    h("The recorder's `action[:6]` should *exactly* equal the commanded "
      "setpoint at every frame, since both come from the same `Pose` "
      "object in the script. Non-zero diff means either the spec passed "
      "to this analyzer doesn't match the recording, or `Pose.to_array` "
      "is mangling the command on its way to disk.")
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
    h("The recorder holds orientation constant at `R0` for every frame. "
      "`action[3:6]` and `observation.state.target[3:6]` should have std "
      "**exactly zero** (linear) — the recorder writes a literally "
      "constant value. For the *measured* column "
      "`observation.state.cartesian[3:6]` we need two statistics: linear "
      "std catches physical wobble, but it also blows up near Euler "
      "singularities (±π) because `Rotation.as_euler('xyz')` can flip "
      "sign on a stationary rotation. **Circular std** is the right "
      "metric for measured angles — if it is small while linear std is "
      "huge, the physical rotation is constant and you are staring at a "
      "representation artifact, not a hardware wobble.")
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
        h,
        action_rot_std,
        target_rot_std,
        cart_rot_std_linear,
        cart_rot_std_circular,
        cart_branch_flips,
    )

    # --- Section 3 ---
    h("## 3. Action vs recorded env target")
    h("")
    h("`observation.state.target[:6]` is what `env.robot.target_pose` "
      "returned at each frame — i.e. what the env-side `crisp_py.Robot` "
      "thinks its internal target is. Since `16_calibration_record.py` "
      "calls `env.robot.set_target(pose=...)` directly, this should "
      "equal `action[:6]` to sub-mm / sub-mrad. Any divergence means "
      "`set_target` is mutating the target, or `target_pose` is stale.")
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
      "`observation.state.cartesian[:3]`. This is **not a bug detector** "
      "— it measures cartesian impedance controller lag, which is "
      "proportional to speed / stiffness. Use it to see whether the "
      "trajectory was tracked, and whether the symmetry check in §4.5 of "
      "the plan doc passes.")
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

    # --- Section 5 ---
    h("## 5. Gripper consistency")
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
          f"{n_transitions} transition boundaries and their follow-up "
          "frames.")
    else:
        h(f"❌ Transitions: `action[6]` ≠ `gripper_target` at one or "
          "more of the transition frames. The recorder is missing a "
          "`env.gripper.set_target(...)` call.")
    h("")
    if grip_cmd_vs_target_bad_frames > 0 and grip_transition_consistent:
        h(f"⚠  `action[6]` ≠ `gripper_target` on {grip_cmd_vs_target_bad_frames} "
          "**non-transition** frames, scattered (not clustered). "
          "Transitions themselves are clean. This is the **crisp_py "
          "gripper echo race**: `Gripper.set_target(v)` writes `_target "
          "= _unnormalize(v)` (raw hardware value), publishes to "
          "`target_gripper_state`, and the same instance's "
          "`_callback_target_state` subscription receives the echo and "
          "writes back `_target = float(msg.data)` (normalized "
          "crisp_py value). The two conventions collide, and depending "
          "on whether the echo callback fires before the next "
          "`env.get_obs()`, the `gripper.target` property reads a raw "
          "or a normalized value and the obs column flickers. It does "
          "**not** mean the actual gripper command was wrong — "
          "`action[6]` in this dataset is the authoritative commanded "
          "value. Fix belongs in `crisp_py/gripper/gripper.py`: make "
          "`_callback_target_state` also apply `_unnormalize`, or drop "
          "the self-subscription entirely.")
    elif grip_cmd_vs_target_bad_frames == 0:
        h("OK — `action[6]` and `gripper_target` agree on every frame.")
    h("")

    # --- Section 6 ---
    h("## 6. Timing — ⚠ synthetic timestamps")
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
              "you would need to log `time.monotonic()` inside `data_fn` "
              "during recording and diff it post-hoc.")
    else:
        h("No `timestamp` column found in the parquet.")
    h("")

    # --- Summary ---
    h("## Summary")
    h("")
    verdicts = []
    if act_gt_pos_err.max() < 1e-5 and act_gt_rot_err.max() < 1e-5:
        verdicts.append("✅ action = ground truth")
    else:
        verdicts.append(f"❌ action ≠ ground truth (max pos err {act_gt_pos_err.max():.2e} m, rot err {act_gt_rot_err.max():.2e} rad)")

    if action_rot_std.max() < 1e-5:
        verdicts.append("✅ action orientation is constant")
    else:
        verdicts.append(f"❌ action orientation wobbles (max std {action_rot_std.max():.2e})")

    if target_rot_std.max() < 1e-5:
        verdicts.append("✅ recorded target orientation is constant")
    else:
        verdicts.append(f"❌ recorded target orientation wobbles (max std {target_rot_std.max():.2e}) — representation bug")

    if act_tgt_max_pos < 1e-4 and act_tgt_max_rot < 1e-4:
        verdicts.append("✅ action == env.robot.target_pose")
    else:
        verdicts.append(f"❌ action ≠ env.robot.target_pose (max pos {act_tgt_max_pos:.2e}, rot {act_tgt_max_rot:.2e})")

    # Measured-orientation verdict uses circular std so we don't falsely
    # flag Euler-XYZ branch flips as physical wobble.
    if cart_rot_std_linear.max() > 0.5 and cart_rot_std_circular.max() < 0.05:
        verdicts.append(
            f"⚠  measured cartesian orientation is "
            f"stable angularly (circular std {cart_rot_std_circular.max():.2e}) "
            f"but linear std is {cart_rot_std_linear.max():.2f} rad and "
            f"there are {int(cart_branch_flips.sum())} Euler-XYZ branch "
            "flips — representation artifact, "
            "see `docs/ridgeback_replay_orientation_bug.md`"
        )
    else:
        verdicts.append(
            f"ℹ  measured cartesian orientation circular std "
            f"{cart_rot_std_circular.max():.2e} rad, "
            f"{int(cart_branch_flips.sum())} branch flips"
        )

    if grip_transitions.size > 0 and grip_transition_consistent and grip_cmd_vs_target_bad_frames > 0:
        verdicts.append(
            f"⚠  gripper: transitions clean but "
            f"{grip_cmd_vs_target_bad_frames} non-transition frames "
            "flicker — crisp_py gripper echo race (see §5)"
        )
    elif grip_transitions.size > 0 and grip_transition_consistent and grip_cmd_vs_target_bad_frames == 0:
        verdicts.append("✅ gripper transitions and frames clean")
    elif grip_transitions.size > 0:
        verdicts.append("❌ gripper transitions inconsistent")

    verdicts.append(f"ℹ  RMS cartesian tracking error {np.sqrt(np.mean(pos_err ** 2)) * 1000:.2f} mm (not a bug, controller lag)")
    verdicts.append(f"ℹ  final dwell cartesian offset from anchor: {np.linalg.norm(cart[-1, :3] - anchor6[:3]) * 1000:.2f} mm")

    for v in verdicts:
        h(f"- {v}")
    h("")

    report = "\n".join(lines)

    if plot_dir is not None:
        _write_plots(plot_dir, action, cart, target, gt_pos, gt_rot, phase_idx)

    return report


def _assess_gt(h, pos_err: np.ndarray, rot_err: np.ndarray) -> None:
    if pos_err.max() < 1e-5 and rot_err.max() < 1e-5:
        h("OK — action exactly matches ground truth (float64 noise only).")
    else:
        h("⚠  action diverges from ground truth. Most likely the "
          "`--velocity` / `--displacement` / `--dwell-seconds` / `--fps` / "
          "`--include-gripper-toggle` passed here don't match the "
          "recording. Check the recorder's log and re-run this script with "
          "the matching values.")
    h("")


def _assess_orientation(
    h,
    action_std: np.ndarray,
    target_std: np.ndarray,
    cart_linear: np.ndarray,
    cart_circular: np.ndarray,
    branch_flips: np.ndarray,
) -> None:
    if action_std.max() < 1e-6:
        h("OK — `action[3:6]` is exactly constant (good).")
    else:
        h("❌  `action[3:6]` is NOT constant. The recorder is writing a "
          "varying orientation into the action column even though the "
          "script commands a fixed `Pose.orientation`. This is a "
          "representation bug in `Pose.to_array(euler)` or upstream.")
    if target_std.max() < 1e-6:
        h("OK — `observation.state.target[3:6]` is exactly constant (good).")
    else:
        h("❌  `observation.state.target[3:6]` is NOT constant. Either "
          "`env.robot.set_target` is mutating the `Pose` or "
          "`env.robot.target_pose` is reading from somewhere else.")

    linear_max = float(cart_linear.max())
    circular_max = float(cart_circular.max())
    flips_total = int(branch_flips.sum())
    if linear_max > 0.5 and circular_max < 0.05:
        h(f"⚠  Measured cartesian orientation has huge linear std "
          f"({linear_max:.2f} rad) but tiny circular std "
          f"({circular_max:.2e} rad), and {flips_total} Euler-XYZ branch "
          "flips. The physical rotation is essentially constant; the "
          "linear std is a **representation artifact** of "
          "`Rotation.as_euler('xyz')` flipping sign at ±π. This is the "
          "bug documented in "
          "`docs/ridgeback_replay_orientation_bug.md` — the *measured* "
          "column is unreliable, but the *commanded* column is clean, "
          "so replay via `--target-source action` or `--target-source "
          "target` is the right workaround until the representation is "
          "switched away from Euler-XYZ (quaternion or angle-axis).")
    elif circular_max > 0.02:
        h(f"⚠  Measured cartesian orientation circular std = "
          f"{circular_max:.3e} rad — the physical rotation is drifting "
          "noticeably across frames even after wraparound correction. "
          "Expected to be small but nonzero (noise + controller lag); "
          "if it's large, check the cartesian controller's orientation "
          "stiffness.")
    else:
        h(f"OK — measured cartesian orientation is stable "
          f"(circular std ≤ {circular_max:.2e} rad, "
          f"{flips_total} branch flips).")
    h("")


def _assess_act_vs_target(h, max_pos: float, max_rot: float) -> None:
    if max_pos < 1e-4 and max_rot < 1e-4:
        h("OK — action matches recorded env target to sub-mm / sub-mrad.")
    else:
        h("⚠  action diverges from `observation.state.target`. Check "
          "`env.robot.set_target` and `env.robot.target_pose` for mutation.")
    h("")


def _write_plots(
    plot_dir: Path,
    action: np.ndarray,
    cart: np.ndarray,
    target: np.ndarray,
    gt_pos: np.ndarray,
    gt_rot: np.ndarray,
    phase_idx: np.ndarray,
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

    # --- Position traces (action / target / cartesian / ground truth) ---
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
    fig.suptitle("Calibration — position traces")
    fig.tight_layout()
    fig.savefig(plot_dir / "position_traces.png", dpi=120)
    plt.close(fig)

    # --- Orientation traces ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for i, (ax, label) in enumerate(zip(axes, ("roll", "pitch", "yaw"))):
        ax.plot(frames, gt_rot[:, i], label="ground truth", linewidth=2.0, alpha=0.6)
        ax.plot(frames, action[:, 3 + i], label="action", linewidth=1.0, linestyle="--")
        ax.plot(frames, target[:, 3 + i], label="target (obs)", linewidth=1.0, linestyle=":")
        ax.plot(frames, cart[:, 3 + i], label="cartesian (measured)", linewidth=1.0)
        ax.set_ylabel(f"{label} (rad)")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("frame index")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Calibration — orientation traces (should be flat)")
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

    logger.info(f"Plots written to {plot_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--repo-id", type=str, default="calib_axis_001")
    parser.add_argument("--velocity", type=float, default=0.05)
    parser.add_argument("--displacement", type=float, default=0.10)
    parser.add_argument("--dwell-seconds", type=float, default=0.5)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--include-gripper-toggle",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Directory to write the report + plots into (default: stdout only).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="Also write position/orientation/tracking plots (requires matplotlib).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(levelname)s %(message)s",
    )

    spec = TrajectorySpec(
        velocity=args.velocity,
        displacement=args.displacement,
        dwell_seconds=args.dwell_seconds,
        fps=args.fps,
        include_gripper_toggle=args.include_gripper_toggle,
    )

    out_dir: Path | None = None
    plot_dir: Path | None = None
    if args.out is not None:
        out_dir = Path(args.out).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.plot:
            plot_dir = out_dir
    elif args.plot:
        # --plot without --out: write under ~/.cache/huggingface/lerobot/<repo>/analysis
        plot_dir = Path(HF_LEROBOT_HOME) / args.repo_id / "analysis"

    report = generate_report(args.repo_id, spec, plot_dir=plot_dir)

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
