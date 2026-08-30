#!/usr/bin/env python3
"""Replay a recorded LeRobot v3 dataset on the real robot via env.step().

Loads a dataset recorded by script 16 (or any LeRobot v3 dataset with a
7-dim action [x, y, z, roll, pitch, yaw, gripper]) and replays it on the
real UR10e by calling ``env.step(action)`` once per frame at the recorded
FPS.

Flow:
  1. Load the dataset from ~/.cache/huggingface/lerobot/<repo_id>.
  2. Move the arm to the first joint configuration via JTC (env.home).
  3. Switch to cartesian_controller.
  4. Replay: call env.step(action, block=False) per frame, rate-limited
     to recording_fps * speed.
  5. Return to home (unless --no-home).

This uses the standard crisp_gym ``env.step()`` path — the same code path
that policy deployment uses. ``env.step()`` calls
``robot.set_target(pose)`` + ``gripper.set_target(grip)`` internally.

Action convention:
  action[0:3] — absolute target position (x, y, z) in base frame
  action[3:6] — absolute target orientation (roll, pitch, yaw) euler xyz
  action[6]   — gripper target, crisp_py convention (1=open, 0=closed)

Usage:
    cd Yunfei/crisp_gym
    pixi run -e jazzy-lerobot python examples/17_replay_dataset.py \\
        --repo-id camera_test --episode-idx 0

    # Half speed:
    pixi run -e jazzy-lerobot python examples/17_replay_dataset.py \\
        --repo-id camera_test --speed 0.5

Prerequisites:
    - Robot up, controller_manager running (startup_robot.py).
    - cartesian_controller and joint_trajectory_controller both loaded.
    - load_crisp.py run (for pose_broadcaster + cartesian_controller).
"""

import argparse
import json
import logging
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rclpy
from control_msgs.action import GripperCommand
from geometry_msgs.msg import PoseStamped
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters, SetParameters
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node as RclpyNode
from rclpy.qos import qos_profile_system_default
from std_msgs.msg import Float32, Float64MultiArray

from scipy.spatial.transform import Rotation

from crisp_gym.deploy.timing import (
    CONTROL_DT,
    build_speed_queue_arrays,
    compute_speed_schedule,
    compute_speed_schedule_cumangle,
    compute_speed_schedule_drop_holds,
)
from crisp_gym.deploy.dataset import (
    LEROBOT_CACHE,
    load_dataset_info,
    load_episode_frames,
    load_episodes_meta,
)
from crisp_gym.deploy.gains import (
    DEFAULT_GRIPPER_SPEED,
    GRIPPER_MAX_SPEED_MPS,
    KD_TASK_KEYS,
    KNUCKLE_JOINT,
    KP_TASK_KEYS,
    SPEED_CMDS_TOPIC,
    SPEED_CTL_NAME,
    SPEED_CTL_TYPE,
    SPEED_INTERFACE,
    ReplayScaler,
    _spawn_gripper_speed_controller,
)
from crisp_gym.deploy.patches import (
    enable_target_pose_publishing,
    fix_gripper_self_subscription,
)
from crisp_gym.deploy.sender import TargetItem, TargetSenderThread
from crisp_gym.envs.manipulator_env import make_env
from crisp_gym.util.setup_logger import setup_logging
from crisp_py.utils.geometry import Pose
from crisp_gym.envs.manipulator_env_config import OrientationRepresentation


logger = logging.getLogger(__name__)


# Kp + gripper-speed scaling constants. Mirrors clearpath_remote_ws's

# Module-level flag set by Phase 2e (--no-camera) right before we destroy
# camera subs/timers. While True, our threading.excepthook below silently
# absorbs the rclpy InvalidHandle exception that fires once on each camera's
# daemon spin thread when its executor's next wait_for_ready_callbacks hits
# the destroyed handles we just yanked out from under it.
#
# This is purely cosmetic — the camera thread was going to die anyway (that
# is exactly what --no-camera wants) — but the default uncaught-exception
# handler dumps a multi-line traceback that drowns out the rest of the
# replay log.
_CAMERA_TEARDOWN_IN_PROGRESS = False


def _install_camera_teardown_excepthook() -> None:
    """Suppress InvalidHandle from camera daemon threads during teardown.

    See ``_CAMERA_TEARDOWN_IN_PROGRESS`` above. Safe to call multiple times;
    only installs the hook on the first invocation. We chain through to the
    original ``threading.excepthook`` for every other exception so we don't
    accidentally silence unrelated bugs.
    """
    if getattr(threading.excepthook, "__crisp_camera_filter__", False):
        return

    # Look up InvalidHandle once. The exact import path varies between rclpy
    # builds (`rclpy._rclpy_pybind11` vs `rclpy.exceptions`); guard both.
    invalid_handle_cls: type | None = None
    try:
        from rclpy._rclpy_pybind11 import InvalidHandle as _IH
        invalid_handle_cls = _IH
    except ImportError:
        try:
            from rclpy.exceptions import InvalidHandle as _IH  # type: ignore
            invalid_handle_cls = _IH
        except ImportError:
            pass

    prev_hook = threading.excepthook

    def _hook(args):  # noqa: ANN001
        if (
            _CAMERA_TEARDOWN_IN_PROGRESS
            and invalid_handle_cls is not None
            and isinstance(args.exc_value, invalid_handle_cls)
        ):
            # Expected: we just destroyed the camera's subs/timers; the
            # camera's executor noticed on its next wait_for_ready_callbacks
            # and the daemon thread is exiting cleanly. Swallow.
            return
        prev_hook(args)

    _hook.__crisp_camera_filter__ = True  # type: ignore[attr-defined]
    threading.excepthook = _hook



class DatasetProducer:
    """Pre-computes the target stream once; deadlines anchored lazily.

    Construction does all the expensive work — Rotation.from_euler in a
    Python loop over T frames (~500 ms-1 s for typical episodes) — but
    deliberately does NOT compute absolute deadlines. The caller must call
    `set_anchor(start_mono)` after construction and right before `fill()`,
    so the deadlines reflect the actual moment the sender is ready to
    consume rather than the moment _build_arrays began. Without this,
    every replay run starts with 100+ "underruns" while the sender races
    to catch up the construction delay.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        s_raw: np.ndarray | None,
        replay_fps: float,
        *,
        gripper_unnormalize_fn,
        gripper_invert: bool,
        gripper_enabled: bool,
        retime: bool,
        start_mono: float | None = None,
    ):
        self._df = df
        self._s_raw = s_raw
        self._replay_fps = float(replay_fps)
        self._gripper_unnormalize_fn = gripper_unnormalize_fn
        self._gripper_invert = gripper_invert
        self._gripper_enabled = gripper_enabled
        self._retime = retime
        self.n_frames = len(df)
        self.dt_base = 1.0 / self._replay_fps
        # Will be set by _build_arrays (cycles/dt_eff/s_eff) and remain
        # None until set_anchor (deadlines).
        self.deadlines: np.ndarray | None = None
        self._build_arrays()
        if start_mono is not None:
            self.set_anchor(float(start_mono))

    def _build_arrays(self) -> None:
        n = self.n_frames
        self.target_xyz = np.zeros((n, 3), dtype=np.float64)
        self.target_quat = np.zeros((n, 4), dtype=np.float64)
        self.grip_raw = np.zeros(n, dtype=np.float64)
        self.has_action = np.zeros(n, dtype=bool)
        self.actions: list[np.ndarray] = [None] * n
        for k in range(n):
            a = np.asarray(self._df.iloc[k]["action"], dtype=np.float32)
            self.actions[k] = a
            a64 = a.astype(np.float64, copy=False)
            if a64.shape[0] >= 7:
                self.target_xyz[k] = a64[:3]
                self.target_quat[k] = Rotation.from_euler("xyz", a64[3:6]).as_quat()
                self.has_action[k] = True
                if self._gripper_enabled and self._gripper_unnormalize_fn is not None:
                    g = float(np.clip(a64[6], 0.0, 1.0))
                    if self._gripper_invert:
                        g = 1.0 - g
                    self.grip_raw[k] = float(self._gripper_unnormalize_fn(g))

        self.cycles, self.dt_eff, self.s_eff = build_speed_queue_arrays(
            self._s_raw, self.dt_base, n, retime=self._retime,
        )
        # Relative cumulative offsets — wall-clock-agnostic until anchored.
        self._cum_dt = np.cumsum(self.dt_eff)

    def set_anchor(self, start_mono: float) -> None:
        """Anchor deadlines to a wall-clock moment.

        Call this right before `fill()`, after the sender thread has
        started — so the moment the sender's first `q.get()` lands,
        `time.monotonic()` is close to `start_mono` and `deadline[0] =
        start_mono + dt_eff[0]` is comfortably in the future (sleep_t
        positive, no underrun).
        """
        self.deadlines = float(start_mono) + self._cum_dt

    def fill(self, q: queue.Queue) -> None:
        """Push every frame onto the queue, then a None sentinel.

        Bounded: queue.put() blocks while the queue is full, naturally
        back-pressuring the producer against the sender's drain rate. Fine
        for dataset replay; future PolicyProducer would do the same.
        """
        if self.deadlines is None:
            raise RuntimeError(
                "DatasetProducer: call set_anchor(start_mono) before fill()"
            )
        for i in range(self.n_frames):
            if not self.has_action[i]:
                continue
            grip = float(self.grip_raw[i]) if self._gripper_enabled else None
            item = TargetItem(
                pose_xyz=self.target_xyz[i],
                pose_quat=self.target_quat[i],
                grip_raw=grip,
                action=self.actions[i],
                deadline_mono=float(self.deadlines[i]),
                frame_idx=i,
                s_eff=float(self.s_eff[i]),
                cycles=int(self.cycles[i]),
            )
            q.put(item)
        q.put(None)


# ---------------------------------------------------------------------------
# Preview GUI (--preview)
# ---------------------------------------------------------------------------


def _decode_episode_videos(
    repo_id: str, episode_idx: int, frame_cap: int,
) -> tuple[list[str], dict[str, np.ndarray]]:
    """Decode all video features for one episode via LeRobotDataset (pyav).

    Returns (camera_names, {name: (n_cached, H, W, 3) uint8}). Frames are
    uniformly subsampled to fit ``frame_cap`` per camera, downsampled to
    a thumbnail width <= 320 to bound memory.
    """
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except Exception:
        logger.warning(
            "preview: LeRobotDataset not importable; skipping video panel"
        )
        return [], {}

    try:
        ds = LeRobotDataset(
            repo_id, root=str(LEROBOT_CACHE / repo_id),
            episodes=[episode_idx], video_backend="pyav",
        )
    except Exception:
        logger.exception(
            "preview: failed to open dataset %s for video decode", repo_id,
        )
        return [], {}

    cam_keys: list[str] = []
    feats = getattr(ds, "features", {}) or {}
    for k, v in feats.items():
        dtype = (v.get("dtype") if isinstance(v, dict) else None)
        if dtype == "video":
            cam_keys.append(k)
    cam_keys.sort()
    if not cam_keys:
        return [], {}

    n_avail = len(ds)
    n_keep = min(n_avail, max(frame_cap, 1))
    indices = np.linspace(0, n_avail - 1, n_keep).astype(int)
    logger.info(
        "preview: decoding %d / %d frames for %d camera(s): %s",
        n_keep, n_avail, len(cam_keys), ", ".join(cam_keys),
    )

    buffers: dict[str, list[np.ndarray]] = {k: [] for k in cam_keys}
    target_w = 320
    for i in indices:
        try:
            item = ds[int(i)]
        except Exception:
            logger.exception("preview: dataset[%d] failed", int(i))
            return cam_keys, {}
        for k in cam_keys:
            img = item.get(k)
            if img is None:
                continue
            if hasattr(img, "detach"):
                img = img.detach().cpu()
            if hasattr(img, "numpy"):
                img = img.numpy()
            if img.ndim == 3 and img.shape[0] in (1, 3, 4):
                img = np.transpose(img, (1, 2, 0))
            if img.dtype != np.uint8:
                img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
            h, w = img.shape[:2]
            if w > target_w:
                stride = max(1, w // target_w)
                img = img[::stride, ::stride]
            buffers[k].append(img)

    cam_frames: dict[str, np.ndarray] = {}
    for k, frames in buffers.items():
        if frames:
            try:
                cam_frames[k] = np.stack(frames, axis=0)
            except ValueError:
                logger.warning(
                    "preview: %s frames have inconsistent shapes; skipping",
                    k,
                )
    return cam_keys, cam_frames


def show_preview_gui(
    actions: np.ndarray,
    schedule: np.ndarray | None,
    fps: float,
    repo_id: str,
    episode_idx: int,
    frame_cap: int = 600,
) -> None:
    """Open a matplotlib preview of (speed schedule, 3D trajectory, videos).

    Blocks until the window is closed. Used by ``--preview`` to inform the
    [Y/n] confirmation prompt that follows. Degrades gracefully if
    matplotlib isn't usable, video decoding fails, or no display is
    attached.
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider, Button
    except ImportError:
        logger.warning("preview: matplotlib not available; skipping --preview")
        return

    backend = matplotlib.get_backend().lower()
    if backend == "agg":
        logger.warning(
            "preview: matplotlib backend is 'agg' (no display); skipping. "
            "Set MPLBACKEND=TkAgg/QtAgg or run from a desktop session.",
        )
        return

    n_frames = int(actions.shape[0])
    if n_frames == 0:
        logger.warning("preview: zero frames; skipping")
        return
    has_pose = actions.shape[1] >= 6
    if schedule is None:
        schedule = np.ones(n_frames, dtype=np.float64)

    cam_names, cam_frames = _decode_episode_videos(
        repo_id, episode_idx, frame_cap,
    )
    cam_names = [k for k in cam_names if k in cam_frames]
    n_cam = len(cam_names)

    # ── Layout ──
    fig = plt.figure(figsize=(15, 9 if n_cam else 6))
    fig.canvas.manager.set_window_title(
        f"replay preview — {repo_id} ep {episode_idx}"
    )
    if n_cam > 0:
        outer = fig.add_gridspec(2, 1, height_ratios=[3, 2])
        top = outer[0].subgridspec(1, 2)
        bot = outer[1].subgridspec(1, n_cam)
        ax_speed = fig.add_subplot(top[0, 0])
        ax_3d = fig.add_subplot(top[0, 1], projection="3d")
        video_axes = [fig.add_subplot(bot[0, i]) for i in range(n_cam)]
    else:
        outer = fig.add_gridspec(1, 2)
        ax_speed = fig.add_subplot(outer[0, 0])
        ax_3d = fig.add_subplot(outer[0, 1], projection="3d")
        video_axes = []

    t_axis = np.arange(n_frames) / fps

    # ── Speed panel ──
    ax_speed.plot(t_axis, schedule, color="tab:blue", linewidth=1.0)
    s_lo = float(min(schedule.min(), 0.95))
    s_hi = float(max(schedule.max(), 1.05))
    pad = 0.05 * max(s_hi - s_lo, 0.1)
    ax_speed.set_ylim(s_lo - pad, s_hi + pad)
    ax_speed.set_xlim(t_axis[0], t_axis[-1])
    ax_speed.set_ylabel("kp scale factor")
    ax_speed.set_xlabel("time (s)")
    ax_speed.set_title("Speed schedule")
    ax_speed.grid(True, alpha=0.3)
    speed_marker = ax_speed.axvline(0, color="red", linewidth=1.5)

    # ── 3D trajectory ──
    cur_dot = None
    if has_pose:
        xs = actions[:, 0].astype(np.float64)
        ys = actions[:, 1].astype(np.float64)
        zs = actions[:, 2].astype(np.float64)
        ax_3d.plot(xs, ys, zs, color="gray", alpha=0.3, linewidth=0.6)
        sc = ax_3d.scatter(
            xs, ys, zs, c=schedule, cmap="viridis", s=6,
            vmin=float(schedule.min()), vmax=float(schedule.max()),
        )
        cur_dot, = ax_3d.plot(
            [xs[0]], [ys[0]], [zs[0]], "o", color="red", markersize=10,
        )
        ax_3d.set_xlabel("x (m)")
        ax_3d.set_ylabel("y (m)")
        ax_3d.set_zlabel("z (m)")
        ax_3d.set_title("Trajectory (color = kp factor)")
        try:
            fig.colorbar(sc, ax=ax_3d, fraction=0.04, pad=0.02, shrink=0.6)
        except Exception:
            pass
    else:
        ax_3d.text(0.5, 0.5, 0.5, "no pose dims", ha="center", va="center")
        ax_3d.set_axis_off()

    # ── Video panels ──
    cam_artists: list[tuple[object, str, int]] = []
    for ax, name in zip(video_axes, cam_names):
        n_cached = int(cam_frames[name].shape[0])
        artist = ax.imshow(cam_frames[name][0])
        ax.set_title(name.replace("observation.images.", ""), fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        cam_artists.append((artist, name, n_cached))

    # ── Slider + Play button ──
    fig.subplots_adjust(bottom=0.13, top=0.94, left=0.06, right=0.97,
                        hspace=0.45, wspace=0.25)
    ax_slider = fig.add_axes((0.10, 0.045, 0.62, 0.025))
    ax_play = fig.add_axes((0.76, 0.04, 0.07, 0.045))

    slider = Slider(
        ax_slider, "frame", 0, n_frames - 1, valinit=0, valstep=1,
        valfmt="%d",
    )
    btn = Button(ax_play, "Play")

    state = {"playing": False, "timer": None}

    def _render(frame_idx: int) -> None:
        frame_idx = int(np.clip(frame_idx, 0, n_frames - 1))
        speed_marker.set_xdata([t_axis[frame_idx], t_axis[frame_idx]])
        if cur_dot is not None and has_pose:
            cur_dot.set_data([actions[frame_idx, 0]], [actions[frame_idx, 1]])
            cur_dot.set_3d_properties([actions[frame_idx, 2]])
        for artist, name, n_cached in cam_artists:
            j = (
                int(round(frame_idx * (n_cached - 1) / max(n_frames - 1, 1)))
                if n_cached > 1 else 0
            )
            artist.set_data(cam_frames[name][j])
        fig.canvas.draw_idle()

    def _on_slider(val) -> None:
        _render(int(val))

    def _tick() -> None:
        if not state["playing"]:
            return
        cur = int(slider.val) + 1
        if cur >= n_frames - 1:
            state["playing"] = False
            btn.label.set_text("Play")
            if state["timer"] is not None:
                state["timer"].stop()
            slider.set_val(n_frames - 1)
            return
        slider.set_val(cur)

    def _toggle_play(_event) -> None:
        state["playing"] = not state["playing"]
        btn.label.set_text("Pause" if state["playing"] else "Play")
        if state["playing"]:
            if state["timer"] is None:
                interval_ms = max(int(round(1000.0 / max(fps, 1.0))), 16)
                state["timer"] = fig.canvas.new_timer(interval=interval_ms)
                state["timer"].add_callback(_tick)
            state["timer"].start()
        else:
            if state["timer"] is not None:
                state["timer"].stop()

    slider.on_changed(_on_slider)
    btn.on_clicked(_toggle_play)

    _render(0)
    logger.info(
        "preview: showing GUI (close window to continue to the [Y/n] prompt)",
    )
    try:
        plt.show()
    except Exception:
        logger.exception("preview: matplotlib show() failed")
    finally:
        if state["timer"] is not None:
            try:
                state["timer"].stop()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repo-id", required=True,
        help="LeRobot dataset name under ~/.cache/huggingface/lerobot/.",
    )
    parser.add_argument(
        "--episode-idx", type=int, default=0,
        help="Episode index to replay (default: 0).",
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="Replay speed multiplier (default: 1.0 = real time).",
    )
    parser.add_argument(
        "--no-home", action="store_true",
        help="Skip the final return-to-home after replay.",
    )
    parser.add_argument(
        "--no-initial-home", action="store_true",
        help="Skip the JTC homing to the recording's first joint config. "
             "Replay starts from wherever the arm currently is — the arm "
             "will snap to the first cartesian target on controller "
             "activation. Use when the recording's initial joint config "
             "looks wrong but the cartesian trajectory is correct. "
             "WARNING: if the current EE is >pose_safety.max_distance "
             "(default 0.4 m) from the first action target, CRISP will "
             "reject the target and freeze.",
    )
    parser.add_argument(
        "--no-state-capture", action="store_true",
        help="Skip reading env.robot.end_effector_pose and "
             "env.robot.joint_values in the replay loop. Drops the "
             "replay.cartesian / replay.joints columns from the saved "
             "parquet, but removes the largest GIL-contention tail "
             "(ee_read max 382 ms / joints_read max 538 ms in profiling).",
    )
    parser.add_argument(
        "--isolate-spin", action="store_true",
        help="Destroy env.robot._joint_subscriber, env.robot._pose_subscriber, "
             "and env.gripper._joint_subscriber before Phase 3. The spin "
             "thread then has no Python callbacks to run during replay, "
             "eliminating GIL contention on the publish hot path. Implies "
             "--no-state-capture (the destroyed buffers stop updating).",
    )
    parser.add_argument(
        "--debug-publish", action="store_true",
        help="Time just the target_pose_pub.publish() call per iteration and "
             "print mean/median/p90/p99/max at end of Phase 3. Used to "
             "distinguish DDS-internal publish stalls from GIL contention "
             "happening elsewhere in the loop.",
    )
    parser.add_argument(
        "--no-gripper", action="store_true",
        help="Fully neutralize the gripper during replay. Destroys ALL "
             "subscriptions and timers on env.gripper.node, and skips the "
             "per-frame /target_gripper_state publish. The gripper position "
             "stays wherever it was when Phase 3 started. Use to test "
             "whether the gripper's spin thread (20 Hz async action goals) "
             "is what's causing the tail stalls.",
    )
    parser.add_argument(
        "--no-gripper-state", action="store_true",
        help="Destroy only env.gripper._joint_subscriber so the gripper's "
             "spin thread stops handling the 500 Hz /gripper/joint_states "
             "callback. The gripper still moves: /target_gripper_state "
             "subscription, 20 Hz action-goal timer, and action client all "
             "stay alive. Implies env.gripper.value will be stale during "
             "replay (state-capture reads gripper state as None).",
    )
    parser.add_argument(
        "--gripper-direct-action", action="store_true",
        help="Take over the gripper action client from the main thread. "
             "Destroys ALL timers on env.gripper.node (including the 20 Hz "
             "_callback_publish_target timer that sends goals on the gripper "
             "spin thread), then sends GripperCommand action goals directly "
             "from the replay loop. Eliminates the dominant remaining "
             "gripper-side GIL contention while keeping the gripper "
             "functional. State subscribers (joint_states, target_state) "
             "stay alive unless --no-gripper-state is also passed.",
    )
    parser.add_argument(
        "--no-camera", action="store_true",
        help="Neutralize each camera in env.cameras during replay. Destroys "
             "ALL subscriptions and timers on each camera's node. Cameras "
             "subscribe to /camera/color/image_raw/compressed at ~30 Hz with "
             "~400 KB messages — their callbacks are a likely source of "
             "GIL contention that survives --isolate-spin (which only "
             "touches env.robot.node).",
    )
    parser.add_argument(
        "--debug-threads", action="store_true",
        help="At start of Phase 3, log every active Python thread "
             "(threading.enumerate()) so we can see all spin threads / "
             "daemon threads in the process.",
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="Skip the confirmation prompt before replay.",
    )
    parser.add_argument(
        "--invert-gripper", action="store_true",
        help="Flip the gripper action (1-grip). Use this for datasets recorded "
             "before the gripper default was fixed (0=open instead of 0=closed).",
    )
    parser.add_argument(
        "--no-safety-clip", action="store_true",
        help="Disable the env's safety box position clipping during replay. "
             "Use this when the recorded trajectory extends beyond the YAML "
             "safety limits (e.g. max_x=0.9) — without it, env.step() silently "
             "clips the target and the arm diverges from the recording.",
    )
    parser.add_argument(
        "--env-config", default="ur10e_ridgeback_env",
        help="crisp_gym env config name (default: ur10e_ridgeback_env).",
    )
    parser.add_argument(
        "--scale-kp", action="store_true",
        help="Apply xVLA-aligned speed scaling: k = k_base * s_eff**2 "
             "(quadratic) on task.k_pos_{x,y,z} + task.k_rot_{x,y,z}; "
             "d = d_base * s_eff (linear) on non-auto-damping task.d_* axes; "
             "gripper speed = --gripper-base-speed * s_eff on "
             "/gripper/gripper_speed_controller/commands; and inter-frame "
             "period scaled accordingly (cycle-snapped to multiples of "
             "CONTROL_DT). See docs/variable_impedance_design.md. All values "
             "restored on shutdown. No-op when --speed == 1.0 and no "
             "--max-speed.",
    )
    parser.add_argument(
        "--gripper-base-speed", type=float, default=DEFAULT_GRIPPER_SPEED,
        help=f"Baseline Robotiq gripper speed in m/s at s_eff=1.0 "
             f"(default: {DEFAULT_GRIPPER_SPEED} m/s = 25%% of driver max "
             f"{GRIPPER_MAX_SPEED_MPS}). Scaled replay speed = base * s_eff, "
             f"clamped to {GRIPPER_MAX_SPEED_MPS} m/s.",
    )
    parser.add_argument(
        "--controller-node", default="/cartesian_controller",
        help="CRISP cartesian controller node name (for --scale-kp).",
    )
    parser.add_argument(
        "--gripper-cm", default="/gripper/controller_manager",
        help="Controller manager hosting the Robotiq gripper controllers.",
    )
    parser.add_argument(
        "--kp-scale-warn", type=float, default=3.0,
        help="Print a warning if the kp peak factor (= s_eff_peak ** 2) "
             "exceeds this while --scale-kp is on (default: 3.0). "
             "Does not abort. See variable_impedance_design.md:24.",
    )
    parser.add_argument(
        "--max-speed", type=float, default=None,
        help="Adaptive speed scaling: peak s_eff at straight + low-rotation "
             "frames. When set, kp/kd/time/gripper all use the per-frame "
             "trajectory-aware schedule. When unset, --scale-kp falls back "
             "to constant kp ∝ --speed**2.",
    )
    parser.add_argument(
        "--min-speed", type=float, default=1.0,
        help="Adaptive speed scaling: floor s_eff at sharp curves / large "
             "orientation deltas (default: 1.0).",
    )
    parser.add_argument(
        "--clamp-deg", type=float, default=5.0,
        help="Adaptive speed scaling: per-step orientation-delta magnitude "
             "(deg) below which the orientation channel keeps max_speed "
             "(default: 5.0). Larger rotations linearly decay to min_speed.",
    )
    parser.add_argument(
        "--lookahead", type=int, default=0,
        help="Adaptive speed scaling: forward-window size for cumulative "
             "bending. 0 = vanilla per-step (xVLA reactive); >0 = sum the "
             "next N+1 direction-change angles and slow down BEFORE curves "
             "(default: 0).",
    )
    parser.add_argument(
        "--lookbehind", type=int, default=0,
        help="Adaptive speed scaling: backward-window size, symmetric "
             "counterpart to --lookahead. Sums the previous M "
             "direction-change angles into the same factor so the arm stays "
             "slow on the EXIT of a curve, not just the entry. 0 = "
             "forward-only (legacy behaviour). Window length becomes "
             "M + N + 1 with the averaging denominator scaled in lockstep.",
    )
    parser.add_argument(
        "--drop-holds", action="store_true",
        help="Compute the speed schedule on moving frames only; held frames "
             "(zero-motion stalls — common in teleop recordings where the "
             "operator command rate is below the recording rate) inherit "
             "the speed of the next moving frame. Without this, each "
             "transition in/out of a hold injects a spurious 90 deg angle "
             "into _per_step_angle and pins the position channel to "
             "--min-speed at the boundary, dominating the schedule. "
             "No-op outside --scale-kp + --max-speed (adaptive mode).",
    )
    parser.add_argument(
        "--hold-eps", type=float, default=1e-6,
        help="Minimum per-step position delta (m) below which a frame is "
             "considered 'held' for --drop-holds (default: 1e-6).",
    )
    parser.add_argument(
        "--no-retime", action="store_true",
        help="Ablation: keep --scale-kp's kp/kd/gripper scaling active, but "
             "leave inter-frame timing uniform (no cycle-snap, no per-frame "
             "dt warp). Useful for isolating the gain-only effect from the "
             "time-warp effect.",
    )
    parser.add_argument(
        "--kp-exp", type=float, default=2.0,
        help="Exponent for kp scaling: k = k_base * s_eff**kp_exp. Default "
             "2.0 matches xVLA / variable_impedance_design.md. Set to 1.0 "
             "to recover the pre-refactor linear-kp behaviour (useful for "
             "A/B testing if quadratic gains cause vibration on this arm).",
    )
    parser.add_argument(
        "--kd-exp", type=float, default=1.0,
        help="Exponent for kd scaling: d = d_base * s_eff**kd_exp on "
             "non-auto-damping axes. Default 1.0 matches xVLA. Set to 0.0 "
             "to disable explicit kd updates (auto-damping axes are always "
             "skipped regardless — controller auto-tracks 2*sqrt(k)).",
    )
    parser.add_argument(
        "--action-stride", type=int, default=1,
        help="Subsample the recorded action stream: keep every Nth frame, "
             "drop the rest. Mirrors xVLA's action_stride "
             "(modeling_xvla.py:743). Gripper speed is multiplied by N to "
             "cover the (N-fold longer) per-action window — matches "
             "lerobot_eval.py:440. CAVEAT: stride*max_speed compounds in "
             "the cycle-snap math (larger dt_base ⇒ s_eff lands higher for "
             "the same s_raw); stride 2 + max_speed 4 has pushed kp past "
             "the safe envelope on this arm in the past. Recommend using "
             "--min-speed == --max-speed (flat schedule, no trajectory "
             "lag estimation) when stride > 1.",
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Open a matplotlib preview window BEFORE the [Y/n] prompt: "
             "speed schedule + 3D trajectory + per-camera video, with a "
             "play button and frame slider. Close the window to fall "
             "through to the existing confirmation. Requires a display.",
    )
    parser.add_argument(
        "--preview-frame-cap", type=int, default=600,
        help="Cap the number of video frames decoded per camera for the "
             "preview (uniformly subsampled). Bounds memory on long "
             "episodes (default: 600 ≈ 20 s @ 30 fps).",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    args = parser.parse_args()

    setup_logging(level=args.log_level)

    # ---- Load dataset ----
    dataset_dir = LEROBOT_CACHE / args.repo_id
    if not dataset_dir.exists():
        logger.error("Dataset not found: %s", dataset_dir)
        sys.exit(1)

    info = load_dataset_info(dataset_dir)
    episodes_df = load_episodes_meta(dataset_dir)
    df = load_episode_frames(dataset_dir, info, episodes_df, args.episode_idx)

    if len(df) == 0:
        logger.error("Episode %d has zero frames.", args.episode_idx)
        sys.exit(1)

    fps = float(info.get("fps", 30))
    original_n_frames = len(df)
    duration = original_n_frames / fps

    # ---- Apply --action-stride (xVLA modeling_xvla.py:743 — keep every Nth
    # frame, drop the rest). All downstream math (compute_speed_schedule,
    # cycle-snap, producer dt_base, summary, replay log) operates on the
    # post-stride frame set. fps_eff = fps / stride is the effective sampling
    # rate of kept frames. Frame 0 is always kept so first_joints / first
    # action are unchanged regardless of stride.
    stride = max(1, int(args.action_stride))
    if stride > 1:
        df = df.iloc[::stride].reset_index(drop=True)
    n_frames = len(df)
    fps_eff = fps / stride
    replay_fps = fps_eff * args.speed

    # Check action dimensionality
    first_action = np.asarray(df.iloc[0]["action"], dtype=np.float32)
    action_dim = first_action.shape[0]

    # Stack the (strided) episode's actions once. compute_speed_schedule
    # below operates on this — matches xVLA: stride first, derive speeds
    # from the strided chunk.
    actions_arr = np.stack(
        [np.asarray(a, dtype=np.float64) for a in df["action"].to_numpy()],
        axis=0,
    )

    # Extract first joint config for homing (frame 0 of the strided df is
    # the same as frame 0 of the original df).
    first_joints = None
    if "observation.state.joints" in df.columns:
        first_joints = np.asarray(
            df.iloc[0]["observation.state.joints"], dtype=np.float64
        ).tolist()

    # ---- Build per-frame speed factor (--scale-kp) ----
    # Adaptive mode (--max-speed set): per-frame s_raw in [min_speed, max_speed]
    # via compute_speed_schedule (xVLA cumulative_bending). Constant mode
    # (no --max-speed): full(T, --speed) for back-compat. The cycle-snap pass
    # in DatasetProducer turns this into s_eff (always <= s_raw); s_eff is
    # what actually drives kp/kd/time/gripper.
    schedule: np.ndarray | None = None
    schedule_mode: str = "off"
    if args.scale_kp:
        adaptive = args.max_speed is not None
        if adaptive and action_dim < 6:
            logger.warning(
                "scale-kp: dataset action_dim=%d < 6; cannot compute "
                "trajectory-aware schedule. Falling back to constant "
                "kp ∝ --max-speed=%.2f.", action_dim, args.max_speed,
            )
            schedule = np.full(n_frames, float(args.max_speed))
            schedule_mode = f"constant {args.max_speed:.2f}x (action_dim<6)"
        elif adaptive:
            if args.min_speed > args.max_speed:
                logger.error(
                    "--min-speed (%.2f) > --max-speed (%.2f); refusing.",
                    args.min_speed, args.max_speed,
                )
                sys.exit(2)
            schedule_fn = (
                compute_speed_schedule_drop_holds if args.drop_holds
                else compute_speed_schedule
            )
            extra = (
                {"motion_eps": float(args.hold_eps)} if args.drop_holds else {}
            )
            schedule = schedule_fn(
                actions_arr[:, :6],
                max_speed=args.max_speed,
                min_speed=args.min_speed,
                clamp_deg=args.clamp_deg,
                n_lookahead=args.lookahead,
                n_lookbehind=args.lookbehind,
                **extra,
            )
            schedule_mode = (
                f"adaptive [{args.min_speed:.2f}-{args.max_speed:.2f}]x "
                f"lookahead={args.lookahead} lookbehind={args.lookbehind} "
                f"clamp_deg={args.clamp_deg:.1f}"
                + (
                    f" drop_holds(eps={args.hold_eps:.0e})"
                    if args.drop_holds else ""
                )
            )
        elif args.speed != 1.0:
            schedule = np.full(n_frames, float(args.speed))
            schedule_mode = f"constant {args.speed:.2f}x"
        else:
            schedule_mode = "off (--scale-kp + --speed=1.0 + no --max-speed)"

    # When --scale-kp drives a schedule, all speed scaling lives in the
    # schedule itself; the producer's dt_base is the recorded period after
    # striding (1/fps_eff = stride/fps). Otherwise --speed alone sets the
    # period via replay_fps. This separation avoids double-applying
    # args.speed in constant-mode scale-kp.
    if args.scale_kp and schedule is not None:
        producer_fps = fps_eff
    else:
        producer_fps = replay_fps
    dt_base_preview = 1.0 / max(producer_fps, 1e-9)

    # Cycle-snap the schedule into the s_eff that actually feeds the gains
    # and the inter-frame timing. Built here so the summary reflects what
    # will run on the robot, not the pre-snap idealised numbers.
    retime_enabled = (schedule is not None) and (not args.no_retime)
    cycles_preview, dt_eff_preview, s_eff_preview = build_speed_queue_arrays(
        schedule, dt_base_preview, n_frames, retime=retime_enabled,
    )
    replay_duration_eff = float(np.sum(dt_eff_preview))

    # ---- Summary ----
    first_act = np.asarray(df.iloc[0]["action"], dtype=np.float64)
    last_act = np.asarray(df.iloc[-1]["action"], dtype=np.float64)

    print()
    print("=== Replay summary ===")
    print(f"  dataset:       {args.repo_id}")
    print(f"  episode:       {args.episode_idx}")
    if stride > 1:
        print(f"  frames:        {n_frames}  (kept every {stride}th of "
              f"{original_n_frames})")
    else:
        print(f"  frames:        {n_frames}")
    duration_line = f"  duration:      {duration:.2f} s  @ {fps:.0f} Hz"
    if stride > 1:
        duration_line += f"  (strided fps_eff={fps_eff:.2f})"
    print(duration_line)
    print(f"  speed:         {args.speed}x  ({duration / args.speed:.2f} s actual)")
    if args.scale_kp:
        if schedule is not None:
            s_eff_peak = float(s_eff_preview.max())
            s_eff_floor = float(s_eff_preview.min())
            s_eff_mean = float(s_eff_preview.mean())
            grip_peak_raw = args.gripper_base_speed * s_eff_peak * stride
            peak_grip = min(grip_peak_raw, GRIPPER_MAX_SPEED_MPS)
            cycles_min = int(cycles_preview.min())
            cycles_max = int(cycles_preview.max())
            print(f"  scale-kp:      ON  {schedule_mode}")
            print(f"                     s_raw peak={schedule.max():.3f} "
                  f"floor={schedule.min():.3f} mean={schedule.mean():.3f}")
            kp_peak_factor = s_eff_peak ** args.kp_exp
            kd_peak_factor = s_eff_peak ** args.kd_exp
            print(f"                     s_eff peak={s_eff_peak:.3f} "
                  f"floor={s_eff_floor:.3f} mean={s_eff_mean:.3f}  "
                  f"(kp peak ×{kp_peak_factor:.2f} "
                  f"[exp={args.kp_exp:.1f}], "
                  f"kd peak ×{kd_peak_factor:.2f} [exp={args.kd_exp:.1f}])")
            print(f"                     cycles min={cycles_min} max={cycles_max} "
                  f"({CONTROL_DT * 1000:.0f} ms/cycle); retime="
                  f"{'ON' if retime_enabled else 'OFF'}; "
                  f"action_stride={stride}; "
                  f"dt_eff total={replay_duration_eff:.2f} s")
            grip_clamp_note = ""
            if grip_peak_raw > GRIPPER_MAX_SPEED_MPS:
                grip_clamp_note = f"  [CLAMPED from {grip_peak_raw:.4f}]"
            print(f"                     gripper base {args.gripper_base_speed:.4f} → "
                  f"peak {peak_grip:.4f} m/s "
                  f"(× s_eff {s_eff_peak:.2f} × stride {stride}){grip_clamp_note}")
            if stride > 1 and kp_peak_factor > args.kp_scale_warn:
                print(f"  ⚠  stride×s_eff compound: kp peak factor "
                      f"{kp_peak_factor:.2f} > --kp-scale-warn "
                      f"{args.kp_scale_warn:.2f}. Consider --max-speed "
                      f"<= {(args.kp_scale_warn ** (1/args.kp_exp)):.2f} "
                      f"when stride={stride}, or set --min-speed == "
                      f"--max-speed for flat (non-adaptive) speedup.")
        else:
            print(f"  scale-kp:      {schedule_mode}")
    print(f"  action dim:    {action_dim}")
    if action_dim >= 7:
        print(f"  first action:  pos=({first_act[0]:+.3f}, {first_act[1]:+.3f}, {first_act[2]:+.3f})"
              f"  rpy=({first_act[3]:+.3f}, {first_act[4]:+.3f}, {first_act[5]:+.3f})"
              f"  grip={first_act[6]:.2f}")
        print(f"  last  action:  pos=({last_act[0]:+.3f}, {last_act[1]:+.3f}, {last_act[2]:+.3f})"
              f"  rpy=({last_act[3]:+.3f}, {last_act[4]:+.3f}, {last_act[5]:+.3f})"
              f"  grip={last_act[6]:.2f}")
    if first_joints is not None:
        joints_str = ", ".join(f"{j:+.3f}" for j in first_joints)
        print(f"  first joints:  [{joints_str}]")
    print()

    if action_dim == 1:
        logger.warning(
            "Action is 1-dim (dummy). This dataset was recorded without "
            "--with-action. The arm will NOT move — only the env.step() "
            "pipeline is exercised. Record with --with-action for real replay."
        )

    if args.preview:
        show_preview_gui(
            actions=actions_arr,
            schedule=schedule,
            fps=fps,
            repo_id=args.repo_id,
            episode_idx=args.episode_idx,
            frame_cap=args.preview_frame_cap,
        )

    if not args.yes:
        try:
            ans = input("  Replay this episode? [Y/n] ").strip().lower()
        except EOFError:
            sys.exit(0)
        if ans not in ("", "y", "yes"):
            logger.info("Aborted.")
            sys.exit(0)

    # ---- Create environment ----
    logger.info("Creating environment: %s", args.env_config)
    env = make_env(env_type=args.env_config, control_type="cartesian", namespace="")

    logger.info("Enabling /target_pose publisher (replay owns the topic)")
    enable_target_pose_publishing(env)

    logger.info("Fixing gripper self-subscription race condition")
    fix_gripper_self_subscription(env)

    if args.no_safety_clip:
        logger.info("Disabling safety box clipping (--no-safety-clip)")
        env.config.safety_box = None

    logger.info("Waiting for robot to be ready...")
    env.wait_until_ready()
    logger.info("Robot ready.")

    # Kp + gripper speed scaling (applied after cartesian_controller is active).
    # Pass the cycle-snapped s_eff_preview rather than the raw schedule so the
    # scaler's peak-warning logic reflects what will actually be applied.
    scaler: ReplayScaler | None = None
    if args.scale_kp and schedule is not None:
        scaler = ReplayScaler(
            env,
            s_eff=s_eff_preview,
            base_gripper_speed=args.gripper_base_speed,
            controller_node=args.controller_node,
            gripper_cm=args.gripper_cm,
            kp_warn_threshold=args.kp_scale_warn,
            kp_exp=args.kp_exp,
            kd_exp=args.kd_exp,
            gripper_stride=stride,
        )
    elif args.scale_kp:
        logger.info("scale-kp: %s", schedule_mode)

    interrupted = False
    try:
        # ---- Phase 1: home to first joint config ----
        if args.no_initial_home:
            logger.warning(
                "Phase 1: SKIPPED (--no-initial-home). Arm stays where it is; "
                "expect a snap to the first cartesian target on controller "
                "activation. If current EE is >pose_safety.max_distance from "
                "the first action target, CRISP will reject the pose."
            )
        elif first_joints is not None:
            logger.info("Phase 1: moving to first joint config via JTC")
            env.home(home_config=first_joints, blocking=True)
            logger.info("Phase 1: at first joint config")
        else:
            logger.info("Phase 1: no joint data in dataset, homing to default")
            env.home(blocking=True)

        # ---- Phase 2: switch to cartesian controller ----
        logger.info("Phase 2: switching to cartesian controller")
        env.switch_controller("cartesian")

        # ---- Phase 2b: scale kp + gripper speed (if requested) ----
        if scaler is not None:
            logger.info("Phase 2b: applying kp schedule (%s)", schedule_mode)
            scaler.apply()

        # ---- Phase 2c: isolate spin thread (--isolate-spin) ----
        # Destroy EVERY subscription and timer on env.robot.node so the daemon
        # spin thread has nothing Python-side to do during replay. The earlier
        # "only destroy named attributes" version missed /current_twist
        # (anonymous subscription) and the target_joint / target_wrench publish
        # timers, so the spin thread was still firing callbacks and ambushing
        # our publishes on the GIL. Implies --no-state-capture (state buffers
        # stop updating) AND --no-home (Phase 4's wait_until_ready relies on
        # the destroyed subscribers; user must home manually afterward).
        if args.isolate_spin:
            args.no_state_capture = True
            args.no_home = True
            rnode = env.robot.node
            n_subs = 0
            for sub in list(rnode.subscriptions):
                rnode.destroy_subscription(sub)
                n_subs += 1
            n_timers = 0
            for t in list(rnode.timers):
                rnode.destroy_timer(t)
                n_timers += 1
            # Null the named attributes too so nothing stale tries to use them.
            env.robot._joint_subscriber = None
            env.robot._pose_subscriber = None
            # The gripper has its own spin thread; we touch its _joint_subscriber
            # here for parity. Full gripper neutralization happens in Phase 2d
            # below if --no-gripper is also set.
            if env.gripper is not None:
                gs = getattr(env.gripper, "_joint_subscriber", None)
                if gs is not None:
                    env.gripper.node.destroy_subscription(gs)
                    env.gripper._joint_subscriber = None
            logger.info(
                "Phase 2c: --isolate-spin destroyed %d subscription(s) and "
                "%d timer(s) on env.robot.node. State buffers stop updating; "
                "Phase 4 (home) is auto-skipped — use './tools/master_launch.sh "
                "home' (or equivalent) to home the arm manually after replay.",
                n_subs, n_timers,
            )

        # ---- Phase 2d-action: take over gripper action client ----
        # Destroy all timers on env.gripper.node (just the 20 Hz
        # _callback_publish_target one) so the gripper spin thread stops
        # doing the periodic send_goal_async work. Keep the action client
        # alive (its internal status/feedback subscribers stay on
        # gripper.node, but they only fire when goals complete — light).
        # The replay loop will then call env.gripper._command_action_client
        # .send_goal_async(goal) directly from the main thread.
        if args.gripper_direct_action and env.gripper is not None:
            gnode = env.gripper.node
            n_timers = 0
            for t in list(gnode.timers):
                gnode.destroy_timer(t)
                n_timers += 1
            if env.gripper._command_action_client is None:
                logger.warning(
                    "Phase 2d-action: --gripper-direct-action requested but "
                    "env.gripper._command_action_client is None (gripper "
                    "config has use_gripper_command_action=False). Will fall "
                    "back to /target_gripper_state publishing."
                )
            else:
                logger.info(
                    "Phase 2d-action: --gripper-direct-action destroyed "
                    "%d timer(s) on gripper.node. Replay loop will send "
                    "GripperCommand goals directly via "
                    "env.gripper._command_action_client.",
                    n_timers,
                )

        # ---- Phase 2d-pre: drop gripper state reader (--no-gripper-state) ----
        # Cheaper than --no-gripper: keeps the gripper functional (its
        # /target_gripper_state sub + 20 Hz publish-target timer + action
        # client all stay alive, so gripper.set_target / our per-frame
        # publishes still command the hardware), but destroys the 500 Hz
        # /gripper/joint_states callback that is the heaviest single source
        # of GIL contention on the gripper spin thread.
        if args.no_gripper_state and env.gripper is not None:
            gs = getattr(env.gripper, "_joint_subscriber", None)
            if gs is not None:
                env.gripper.node.destroy_subscription(gs)
                env.gripper._joint_subscriber = None
                logger.info(
                    "Phase 2d-pre: --no-gripper-state destroyed gripper "
                    "joint_states subscriber. env.gripper.value will be stale."
                )

        # ---- Phase 2d: neutralize gripper (--no-gripper) ----
        # The gripper class runs its OWN daemon spin thread separate from
        # env.robot's. It has subscriptions (incl. /target_gripper_state we
        # publish to each frame) and a 20 Hz timer that fires send_goal_async
        # to the gripper action server. All of these can ambush the main
        # thread's GIL between our /target_pose publishes. To eliminate this
        # source of contention, destroy every subscription and timer on the
        # gripper's node — its spin thread then sleeps in spin_once with no
        # work to do.
        if args.no_gripper and env.gripper is not None:
            gnode = env.gripper.node
            n_subs = 0
            for sub in list(gnode.subscriptions):
                gnode.destroy_subscription(sub)
                n_subs += 1
            n_timers = 0
            for t in list(gnode.timers):
                gnode.destroy_timer(t)
                n_timers += 1
            logger.info(
                "Phase 2d: --no-gripper — destroyed %d subscription(s) and "
                "%d timer(s) on the gripper node. Gripper will not move "
                "during replay.",
                n_subs, n_timers,
            )

        # ---- Phase 2e: neutralize cameras (--no-camera) ----
        # Each Camera in env.cameras runs its own daemon spin thread and
        # subscribes to high-rate image topics (~30 Hz, ~400 KB messages).
        # Each image callback decompresses / stores the frame, holding the
        # GIL for non-trivial time. To prevent those callbacks from ambushing
        # our publish hot path, destroy every subscription and timer on each
        # camera's node — the spin threads then sleep in spin_once with no
        # work to do. Images stop arriving during replay (irrelevant since
        # the replay loop never reads them — that's env.step()'s job).
        #
        # crisp_py.Camera owns its executor + spin thread internally and has
        # no clean-shutdown API, so we can't tell the thread to stop before
        # yanking its waitables. The thread WILL hit InvalidHandle on its
        # next wait_for_ready_callbacks — that's the desired outcome (thread
        # dies) but the default uncaught-exception handler dumps a traceback.
        # We install an excepthook that swallows that one expected exception
        # during the gated window, then release the gate.
        if args.no_camera and env.cameras:
            global _CAMERA_TEARDOWN_IN_PROGRESS
            _install_camera_teardown_excepthook()
            _CAMERA_TEARDOWN_IN_PROGRESS = True
            total_subs = 0
            total_timers = 0
            for cam in env.cameras:
                cnode = cam.node
                for sub in list(cnode.subscriptions):
                    cnode.destroy_subscription(sub)
                    total_subs += 1
                for t in list(cnode.timers):
                    cnode.destroy_timer(t)
                    total_timers += 1
            # Give each camera's executor one cycle past its 100 ms spin_once
            # timeout to hit the wait_for_ready_callbacks → InvalidHandle path
            # and exit. 150 ms is conservative; the actual crash usually
            # happens within ~20 ms of destruction.
            time.sleep(0.15)
            _CAMERA_TEARDOWN_IN_PROGRESS = False
            logger.info(
                "Phase 2e: --no-camera — destroyed %d subscription(s) and "
                "%d timer(s) across %d camera node(s).",
                total_subs, total_timers, len(env.cameras),
            )

        # ---- Phase 3: replay via Producer + SenderThread ----
        # DatasetProducer pre-computes (poses, gripper raws, cycle-snapped
        # deadlines, s_eff) once. TargetSenderThread owns the publishers and
        # the deadline-based sleep; rclpy publishes release the GIL inside
        # the C extension, so the main thread is free of publish work. Same
        # substrate plugs in a future PolicyProducer.
        logger.info(
            "Phase 3: replaying %d frames (base %.1f Hz × --speed %.2fx; "
            "action_stride=%d; retime=%s)",
            n_frames, fps, args.speed, stride,
            "ON" if retime_enabled else "OFF",
        )
        if args.debug_threads:
            active = threading.enumerate()
            logger.info("  active threads at Phase 3 start (%d total):", len(active))
            for t in active:
                logger.info(
                    "    [%s] daemon=%s alive=%s ident=%s",
                    t.name, t.daemon, t.is_alive(), t.ident,
                )

        # Publish gripper targets as RAW values on /target_gripper_state, the
        # same convention track_mocap.py uses during live teleop. This avoids
        # a bug in crisp_py.Gripper.set_target() which publishes normalized
        # values and then self-overwrites _target via its own subscriber,
        # inverting the hardware command direction. See plan file.
        gripper_raw_pub = None
        gripper_action_client = None
        gripper_max_effort = 0.0
        gripper_unnormalize_fn = None
        gripper_enabled = env.gripper is not None and not args.no_gripper
        if gripper_enabled:
            gripper_unnormalize_fn = env.gripper._unnormalize
            gripper_max_effort = float(env.gripper.config.max_effort)
            if (
                args.gripper_direct_action
                and env.gripper._command_action_client is not None
            ):
                # Action-client path: producer pre-bakes raw values; sender
                # constructs a fresh Goal() per item and fires send_goal_async.
                gripper_action_client = env.gripper._command_action_client
            else:
                # Default: route through /target_gripper_state with Float32.
                gripper_raw_pub = env.robot.node.create_publisher(
                    Float32, "/target_gripper_state", 1
                )

        # Direct /target_pose publishing. We REUSE the publisher created by
        # enable_target_pose_publishing() — creating a second publisher would
        # trip CRISP's check_topic_publisher_count() in cartesian_controller.cpp
        # and the controller would silently drop every message.
        #
        # CRITICAL: enable_target_pose_publishing also installed a 20 Hz timer
        # that calls _callback_publish_target_pose, which reads
        # env.robot._target_pose (initialised by _callback_current_pose to the
        # CURRENT EE pose at startup). If we don't disable that timer, it
        # republishes the initial pose 20×/s and fights with our loop —
        # producing the "weird" jerky behaviour. We neutralize it by keeping a
        # local reference to the publisher object, then setting
        # env.robot._target_pose_publisher to None so the timer's
        # `if self._target_pose_publisher is None: return` early-return fires.
        target_pose_pub = env.robot._target_pose_publisher
        env.robot._target_pose_publisher = None
        base_frame_id = env.robot.config.base_frame
        # Reusable PoseStamped — only stamp + position + orientation are
        # rewritten each iteration; the sender thread owns it.
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = base_frame_id

        replay_log: list[dict] = []

        # State-capture closure runs on the sender thread. Reads from buffers
        # populated by crisp_py's own executor; no shared state with the
        # publisher path beyond those read-only attributes.
        state_capture_fn = None
        if not args.no_state_capture:
            def _capture_state(row: dict) -> None:
                try:
                    ee = env.robot.end_effector_pose
                    ee_arr = ee.to_array(representation=OrientationRepresentation.EULER)
                    row["replay.cartesian"] = ee_arr.astype(np.float32)
                except RuntimeError:
                    pass
                try:
                    row["replay.joints"] = env.robot.joint_values.copy()
                except RuntimeError:
                    pass
            state_capture_fn = _capture_state

        # Bounded queue: the dataset fits comfortably, but bounding it makes
        # the future policy producer back-pressure naturally instead of
        # blowing memory.
        q: queue.Queue = queue.Queue(maxsize=128)
        # Build the producer first — this does the ~500 ms-1 s of
        # Rotation.from_euler in a Python loop. Don't capture start_mono
        # yet; deadlines computed off an early timestamp would be already
        # in the past by the time the sender starts consuming, and the
        # sender would log 100+ false underruns while catching up. We
        # anchor below, right before fill().
        producer = DatasetProducer(
            df=df,
            s_raw=schedule,
            replay_fps=producer_fps,
            gripper_unnormalize_fn=gripper_unnormalize_fn,
            gripper_invert=args.invert_gripper,
            gripper_enabled=gripper_enabled,
            retime=retime_enabled,
        )
        sender = TargetSenderThread(
            q,
            target_pose_pub=target_pose_pub,
            gripper_raw_pub=gripper_raw_pub,
            gripper_action_client=gripper_action_client,
            gripper_max_effort=gripper_max_effort,
            pose_msg=pose_msg,
            clock=env.robot.node.get_clock(),
            scaler=scaler,
            replay_log=replay_log,
            state_capture_fn=state_capture_fn,
            debug_publish=args.debug_publish,
        )
        sender.start()
        # Anchor deadlines AT THIS MOMENT — sender thread is alive and
        # ready to consume; the build cost (~500 ms) and sender.start()
        # latency are both behind us, so deadline[0] = now + dt_eff[0]
        # is genuinely in the future.
        producer.set_anchor(time.monotonic())
        # producer.fill blocks on a full queue, so the sender drains it as
        # we go. For the dataset case this is effectively instant after the
        # first few frames. Producer always pushes a None sentinel last.
        producer.fill(q)
        sender.join()

        logger.info(
            "Phase 3: replay done — published %d frames, %d underruns, "
            "queue depth seen [min=%d, max=%d]",
            sender.n_published, sender.underrun_count,
            sender.queue_depth_min if sender.queue_depth_min != 2 ** 31 else 0,
            sender.queue_depth_max,
        )
        if args.debug_publish and sender.pub_dt_samples:
            arr = np.asarray(sender.pub_dt_samples)
            ref_period = float(producer.dt_eff.min()) if producer.n_frames else producer.dt_base
            logger.info("  publish() timing (ms):")
            logger.info(
                "    mean=%.2f  median=%.2f  p90=%.2f  p99=%.2f  max=%.2f",
                arr.mean() * 1000.0,
                np.median(arr) * 1000.0,
                np.percentile(arr, 90) * 1000.0,
                np.percentile(arr, 99) * 1000.0,
                arr.max() * 1000.0,
            )
            half_p = ref_period / 2.0
            slow = int((arr > half_p).sum())
            very_slow = int((arr > ref_period).sum())
            logger.info(
                "    publishes > %.1f ms (period/2 @ min dt_eff): %d / %d",
                half_p * 1000.0, slow, len(arr),
            )
            logger.info(
                "    publishes > %.1f ms (full period @ min dt_eff): %d / %d",
                ref_period * 1000.0, very_slow, len(arr),
            )

        # ---- Save replay log ----
        replay_dir = dataset_dir / "replay"
        replay_dir.mkdir(exist_ok=True)
        replay_path = replay_dir / f"episode_{args.episode_idx}_speed{args.speed}.parquet"

        replay_df = pd.DataFrame(replay_log)
        # Expand array columns into separate columns for easy comparison
        if "replay.cartesian" in replay_df.columns:
            cart = np.stack(replay_df["replay.cartesian"].values)
            for j, name in enumerate(["x", "y", "z", "roll", "pitch", "yaw"]):
                replay_df[f"replay.cart.{name}"] = cart[:, j]
            replay_df = replay_df.drop(columns=["replay.cartesian"])
        if "replay.joints" in replay_df.columns:
            joints = np.stack(replay_df["replay.joints"].values)
            for j in range(joints.shape[1]):
                replay_df[f"replay.joint.{j}"] = joints[:, j]
            replay_df = replay_df.drop(columns=["replay.joints"])
        if "replay.action" in replay_df.columns:
            act = np.stack(replay_df["replay.action"].values)
            for j in range(act.shape[1]):
                replay_df[f"replay.action.{j}"] = act[:, j]
            replay_df = replay_df.drop(columns=["replay.action"])

        replay_df.to_parquet(replay_path, index=False)
        logger.info("Replay log saved to %s (%d frames)", replay_path, len(replay_df))

    except KeyboardInterrupt:
        interrupted = True
        logger.warning("Interrupted. Arm frozen at last commanded pose.")
    except Exception:
        logger.exception("Replay failed")
        raise
    else:
        # ---- Phase 4: return home ----
        if not args.no_home:
            try:
                logger.info("Phase 4: returning to home")
                env.home(blocking=True)
                logger.info("Replay complete. Robot homed.")
            except KeyboardInterrupt:
                logger.warning("Interrupted during return-to-home.")
        else:
            logger.info("Replay complete. Skipping home (--no-home).")
    finally:
        if scaler is not None:
            try:
                scaler.restore()
            except Exception:
                logger.exception("scaler.restore() raised")
        try:
            env.close()
        except Exception:
            logger.exception("env.close() raised")
        if rclpy.ok():
            rclpy.shutdown()

    sys.exit(1 if interrupted else 0)


if __name__ == "__main__":
    main()
