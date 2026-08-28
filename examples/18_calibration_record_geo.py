#!/usr/bin/env python3
"""Geometric calibration recorder for UR10e crisp_gym — circle / square sweep.

Sibling of ``16_calibration_record.py``. Same contract (deterministic cartesian
trajectory, constant orientation, auto-start/auto-save LeRobot v3 recording,
no mocap / no human input), but the trajectory is a **geometric shape**
(circle or square) centred at the anchor pose ``P0`` captured after
``env.home()``. Intended as a second axis of the record/replay smoke test:

- ``16_calibration_record.py`` — axis-aligned sweeps (±z, ±y). Isolates
  representation bugs on orthogonal translations.
- ``18_calibration_record_geo.py`` — closed-loop shape trace. Isolates
  record/replay bugs on continuous curved or polygonal paths, and gives a
  clean closed-loop trajectory for record-vs-replay overlay plots.

The full motivation, acceptance checks, and failure-mode catalogue live in
``docs/ridgeback_calibration_recording_plan.md``. The operator-facing guide
for *this* script is ``docs/ridgeback_calibration_recording_geo.md``.

This script is intentionally independent of ``13_/15_ridgeback_mocap_record.py``
and ``14_ridgeback_replay.py`` — it reimplements the topic-ownership patch
inline rather than importing from script 14, so that edits to any of the
existing scripts cannot break this one.

Shapes (plane = xy by default, frame = ``arm_0_base_link``):

    circle   N waypoints on a circle of radius = --size, centred at P0.
             Angular resolution is derived from --velocity / fps so the
             cartesian speed along the circumference is constant. First
             point is (+size, 0) from P0, traversed CCW.

    square   8 shape corners (4 midpoints + 4 corners) of a square of
             side = --size, centred at P0, traversed CCW. Each edge is
             linearly interpolated at --velocity m/s. First corner is
             (+size/2, 0) from P0.

Trajectory layout (all at constant orientation R0 = anchor.orientation):

    dwell(P0) → approach (P0 → first_shape_pt) → shape trace × --loops
              → return (last_shape_pt → P0) → dwell(P0)

Usage:
    # 1. Bring up the robot with CRISP cartesian controller:
    #        ./tools/master_launch.sh up --controller crisp
    #    (NO --track — mocap / track_mocap.py must NOT be running, because
    #    this script owns /target_pose itself.)
    #
    # 2. Run the calibration recorder:
    cd Yunfei/crisp_gym
    pixi run -e jazzy-lerobot python examples/18_calibration_record_geo.py \\
        --repo-id calib_geo_001 --shape circle --size 0.05

Prerequisites:
    - cartesian_controller active on the robot
    - joint_state_broadcaster + pose_broadcaster active
    - Camera running (e.g. ``pixi run orbbec`` in clearpath_remote_ws)
    - ``track_mocap.py`` NOT running (this script publishes /target_pose)

Auto-start / auto-save:
    The recording starts automatically when the robot is ready (no 'r'
    keypress) and saves automatically when the trajectory ends (no 's'
    keypress). The script exits on its own. Avoid pressing 'r' / 's' / 'd'
    during the run or you will perturb the recorder state machine.
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import qos_profile_system_default
from scipy.spatial.transform import Rotation

try:
    from lerobot.utils.constants import HF_LEROBOT_HOME
except ImportError:
    from lerobot.constants import HF_LEROBOT_HOME

from crisp_gym.envs.manipulator_env import make_env
from crisp_gym.record.recording_manager import make_recording_manager
from crisp_gym.record.recording_manager_config import RecordingManagerConfig
from crisp_gym.util.lerobot_features import get_features
from crisp_gym.util.setup_logger import setup_logging
from crisp_py.utils.geometry import Pose

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plane axes. u = first in-plane axis, v = second. Shape is traced in (u, v).
# ---------------------------------------------------------------------------

PLANE_AXES = {
    "xy": (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
    "xz": (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
    "yz": (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
}


# ---------------------------------------------------------------------------
# Topic-ownership patch — inverse of ``silence_env_target_publishers`` from
# scripts 13/15. Copied inline from 14_ridgeback_replay.py so this script
# has no runtime dependency on any sibling example.
# ---------------------------------------------------------------------------

def enable_env_target_pose_publishing(env) -> None:
    """Make ``env.robot`` own ``/target_pose`` (publish, not subscribe).

    ``ur10e_ridgeback_env.yaml`` has ``publish_target_pose: false`` for the
    mocap flow. In that branch ``crisp_py.Robot.__init__`` creates a
    subscription instead of a publisher and the env's ``set_target`` calls
    never reach the controller. This helper destroys the subscription,
    creates the missing publisher + 20 Hz publish timer, and flips the flag.

    Must be called BEFORE ``wait_until_ready()`` and BEFORE the first
    ``set_target`` call. See ``docs/ridgeback_target_pose_ownership.md``.
    """
    robot = env.robot

    sub = getattr(robot, "_target_pose_subscriber", None)
    if sub is not None:
        robot.node.destroy_subscription(sub)
        robot._target_pose_subscriber = None

    if getattr(robot, "_target_pose_publisher", None) is None:
        robot._target_pose_publisher = robot.node.create_publisher(
            PoseStamped,
            robot.config.target_pose_topic,
            qos_profile_system_default,
        )

    robot.node.create_timer(
        1.0 / robot.config.publish_frequency,
        robot._callback_publish_target_pose,
        ReentrantCallbackGroup(),
    )

    robot.config.publish_target_pose = True


# ---------------------------------------------------------------------------
# Trajectory construction
# ---------------------------------------------------------------------------

def _linear_segment(
    start_pos: np.ndarray,
    end_pos: np.ndarray,
    orientation: Rotation,
    n_frames: int,
    grip: float,
) -> List[Tuple[Pose, float]]:
    """Linear cartesian interpolation ``start`` (exclusive) → ``end`` (inclusive).

    ``n_frames`` ticks at fractions ``1/n, 2/n, …, 1``, so the last frame is
    exactly at ``end_pos`` and the first is one step off ``start_pos``. Matches
    the convention in ``16_calibration_record.py``.
    """
    frames: List[Tuple[Pose, float]] = []
    for i in range(1, n_frames + 1):
        t = i / n_frames
        pos = start_pos + t * (end_pos - start_pos)
        frames.append((Pose(position=pos.copy(), orientation=orientation), grip))
    return frames


def _dwell(
    pos: np.ndarray,
    orientation: Rotation,
    n_frames: int,
    grip: float,
) -> List[Tuple[Pose, float]]:
    """Hold ``pos`` for ``n_frames`` ticks at constant orientation and grip."""
    return [
        (Pose(position=pos.copy(), orientation=orientation), grip)
        for _ in range(n_frames)
    ]


def _circle_positions(
    center: np.ndarray,
    radius: float,
    u: np.ndarray,
    v: np.ndarray,
    velocity: float,
    fps: int,
    loops: int,
) -> List[np.ndarray]:
    """Generate one position per frame tracing a circle at constant speed.

    The angle advances by ``velocity / radius / fps`` radians per frame, so
    the tangential speed is ``velocity`` m/s. First frame of the trace is at
    angle = ``2π/n_per_loop`` (one step off the starting point (+radius, 0)),
    matching ``_linear_segment``'s "start exclusive" convention so the approach
    segment can hand off cleanly.
    """
    if radius <= 0:
        raise ValueError("circle radius must be positive")
    if velocity <= 0:
        raise ValueError("velocity must be positive")
    angle_per_frame = velocity / (radius * fps)
    n_per_loop = max(8, round((2 * np.pi) / angle_per_frame))
    positions: List[np.ndarray] = []
    for _ in range(loops):
        # i = 1 .. n_per_loop — first shape frame is one tick off the start.
        for i in range(1, n_per_loop + 1):
            theta = 2.0 * np.pi * (i / n_per_loop)
            pos = center + radius * (np.cos(theta) * u + np.sin(theta) * v)
            positions.append(pos)
    return positions


def _square_positions(
    center: np.ndarray,
    side: float,
    u: np.ndarray,
    v: np.ndarray,
    velocity: float,
    fps: int,
    loops: int,
) -> List[np.ndarray]:
    """Generate positions tracing an 8-vertex square at constant speed.

    8 vertices = 4 edge midpoints + 4 corners, CCW starting at (+side/2, 0).
    Each of the 8 edges is linearly interpolated with one frame per
    ``velocity`` m/s step. First frame of each edge is one step off the
    edge's start vertex (so ``_linear_segment``'s exclusive-start convention
    is preserved across edge boundaries).
    """
    if side <= 0:
        raise ValueError("square side must be positive")
    if velocity <= 0:
        raise ValueError("velocity must be positive")

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

    # Every edge has length h = side/2. Frames per edge from velocity.
    n_per_edge = max(1, round(h / velocity * fps))

    positions: List[np.ndarray] = []
    for _ in range(loops):
        for i in range(8):
            start = vertices[i]
            end = vertices[(i + 1) % 8]
            for k in range(1, n_per_edge + 1):
                t = k / n_per_edge
                positions.append(start + t * (end - start))
    return positions


def build_shape_trajectory(
    anchor: Pose,
    shape: str,
    size: float,
    plane: str,
    velocity: float,
    fps: int,
    loops: int,
    dwell_seconds: float,
    include_gripper_toggle: bool,
) -> List[Tuple[Pose, float]]:
    """Build the full geometric calibration trajectory as a list of (Pose, grip) frames.

    Layout:
        dwell(P0)
        approach segment (P0 → first shape position)
        shape trace (circle or square × ``loops``, one frame per 1/fps)
        return segment (last shape position → P0)
        dwell(P0)

    Orientation is held constant at ``anchor.orientation`` throughout — any
    nonzero variation in the recorded orientation columns is therefore a
    representation bug.

    Gripper convention is crisp_py: ``1.0 = open``, ``0.0 = closed``. When
    ``include_gripper_toggle`` is true the gripper is open during the first
    half of the shape trace (and during the approach + pre-dwell), closed
    during the second half of the shape trace and the return segment, then
    reopens for the final post-dwell. Otherwise the grip stays open for the
    entire trajectory.
    """
    if shape not in {"circle", "square"}:
        raise ValueError(f"unknown shape: {shape!r} (expected 'circle' or 'square')")
    if plane not in PLANE_AXES:
        raise ValueError(f"unknown plane: {plane!r} (expected xy/xz/yz)")

    anchor_pos = anchor.position.copy()
    orientation = Rotation.from_quat(anchor.orientation.as_quat())  # deep copy
    u, v = PLANE_AXES[plane]
    n_dwell = max(1, round(dwell_seconds * fps))

    OPEN = 1.0
    CLOSED = 0.0

    # 1. Shape positions (ignoring dwells / approach / return).
    if shape == "circle":
        shape_positions = _circle_positions(
            anchor_pos, size, u, v, velocity, fps, loops
        )
        # First/last point of the closed circle — both at (+size, 0).
        first_pt = anchor_pos + size * u
    else:
        shape_positions = _square_positions(
            anchor_pos, size, u, v, velocity, fps, loops
        )
        # First/last point of the closed square — both at (+side/2, 0).
        first_pt = anchor_pos + (size / 2.0) * u

    if not shape_positions:
        raise ValueError(
            f"build_shape_trajectory: no shape positions generated for shape={shape}, "
            f"size={size}, velocity={velocity}, fps={fps}, loops={loops}"
        )

    # 2. Approach segment: center → first shape vertex.
    approach_dist = float(np.linalg.norm(first_pt - anchor_pos))
    n_approach = max(1, round(approach_dist / velocity * fps))

    # 3. Gripper toggle schedule (expressed as a predicate over shape-frame index).
    n_shape = len(shape_positions)
    toggle_idx = n_shape // 2 if include_gripper_toggle else n_shape

    # 4. Assemble the trajectory.
    traj: List[Tuple[Pose, float]] = []
    traj += _dwell(anchor_pos, orientation, n_dwell, OPEN)                    # pre-dwell
    traj += _linear_segment(anchor_pos, first_pt, orientation, n_approach, OPEN)

    for idx, pos in enumerate(shape_positions):
        grip = OPEN if idx < toggle_idx else CLOSED
        traj.append((Pose(position=pos.copy(), orientation=orientation), grip))

    return_grip = CLOSED if include_gripper_toggle else OPEN
    traj += _linear_segment(
        shape_positions[-1], anchor_pos, orientation, n_approach, return_grip
    )
    traj += _dwell(anchor_pos, orientation, n_dwell, OPEN)                    # post-dwell

    return traj


def prompt_remove_existing_dataset(
    repo_id: str,
    assume_yes: bool = False,
) -> None:
    """Interactively offer to ``rm -rf`` a stale LeRobot dataset directory.

    ``RecordingManager._create_dataset`` refuses to overwrite an existing
    ``repo_id`` — even if the previous run crashed and left a 0-episode
    stub behind. We check the same path it checks, and if it exists and
    the user agrees (or ``assume_yes`` is set), delete it and continue.
    On "no", exit with status 0 so the operator can pick a new
    ``--repo-id`` without looking like a crash.

    Skip-invariant: callers must NOT invoke this when ``--resume`` is set.
    Resume expects the directory to already exist.
    """
    dataset_path = Path(HF_LEROBOT_HOME) / repo_id
    if not dataset_path.exists():
        return

    try:
        n_episodes = sum(
            1 for _ in (dataset_path / "meta" / "episodes").rglob("file-*.parquet")
        )
    except Exception:
        n_episodes = -1

    if n_episodes > 0:
        summary = f"{n_episodes} recorded episode(s)"
    elif n_episodes == 0:
        summary = "no recorded episodes (empty stub from a previous run)"
    else:
        summary = "unknown contents"

    print(
        f"\n[calibration-geo] LeRobot dataset directory already exists:\n"
        f"  {dataset_path}\n"
        f"  Contains: {summary}\n"
        f"  Deleting will permanently remove everything under this path."
    )

    if assume_yes:
        print("  --yes passed; deleting without prompting.")
        shutil.rmtree(dataset_path)
        print(f"  Deleted {dataset_path}.\n")
        return

    try:
        ans = input("  Delete it and continue? [y/N] ").strip().lower()
    except EOFError:
        ans = ""

    if ans in ("y", "yes"):
        shutil.rmtree(dataset_path)
        print(f"  Deleted {dataset_path}.\n")
        return

    print(
        "  Keeping the existing dataset. Pick a different --repo-id "
        "(or pass --resume if you meant to append) and re-run.\n"
    )
    sys.exit(0)


def validate_against_workspace(
    traj: List[Tuple[Pose, float]],
    safety_box: dict,
) -> None:
    """Abort with a clear error if any waypoint is outside the env workspace box.

    ``safety_box`` is the dict built by ``ManipulatorEnvConfig.__post_init__``:
    ``{"lower": [min_x, min_y, min_z], "upper": [max_x, max_y, max_z]}`` with
    unset limits coerced to ±inf.
    """
    lower = np.asarray(safety_box["lower"], dtype=np.float64)
    upper = np.asarray(safety_box["upper"], dtype=np.float64)
    for idx, (pose, _) in enumerate(traj):
        pos = pose.position
        if np.any(pos < lower) or np.any(pos > upper):
            violated = []
            for axis, name in enumerate("xyz"):
                if pos[axis] < lower[axis]:
                    violated.append(
                        f"{name}={pos[axis]:+.3f} < min_{name}={lower[axis]:+.3f}"
                    )
                if pos[axis] > upper[axis]:
                    violated.append(
                        f"{name}={pos[axis]:+.3f} > max_{name}={upper[axis]:+.3f}"
                    )
            raise ValueError(
                f"Calibration waypoint {idx} at position {pos} violates workspace "
                f"limits: {', '.join(violated)}. Check the anchor pose, --size, "
                f"and --plane, or widen the limits in ur10e_ridgeback_env.yaml."
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: C901 — linear top-level script
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--repo-id", type=str, default="calib_geo_001")
    parser.add_argument("--task", type=str, default="calibration geometric sweep")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--shape",
        type=str,
        choices=["circle", "square"],
        required=True,
        help="Geometric shape to trace around the anchor pose.",
    )
    parser.add_argument(
        "--size",
        type=float,
        required=True,
        help="Circle radius OR square side length (metres).",
    )
    parser.add_argument(
        "--plane",
        type=str,
        choices=["xy", "xz", "yz"],
        default="xy",
        help="Plane of the shape in arm_0_base_link frame.",
    )
    parser.add_argument(
        "--velocity",
        type=float,
        default=0.05,
        help="Cartesian speed along the shape (m/s). Default 5 cm/s.",
    )
    parser.add_argument(
        "--loops",
        type=int,
        default=1,
        help="Number of full shape repetitions.",
    )
    parser.add_argument(
        "--dwell-seconds",
        type=float,
        default=0.5,
        help="Dwell duration at the anchor before and after the shape trace (seconds).",
    )
    parser.add_argument(
        "--include-gripper-toggle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exercise the gripper action column with one close/open cycle.",
    )
    parser.add_argument(
        "--env-config",
        type=str,
        default="ur10e_ridgeback_env",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--push-to-hub",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        default=False,
        help=(
            "Auto-confirm interactive prompts (currently: deletion of an "
            "existing dataset directory with the same --repo-id). Use with "
            "care — this bypasses the last confirmation before rm -rf."
        ),
    )
    parser.add_argument(
        "--skip-workspace-check",
        action="store_true",
        default=False,
        help=(
            "Skip the pre-flight workspace-box validation. Use when the env "
            "YAML's min/max_x/y/z are narrower than the actual home pose and "
            "you have already confirmed the trajectory is physically safe."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    args = parser.parse_args()

    setup_logging(level=args.log_level)

    if args.velocity <= 0:
        raise ValueError("--velocity must be positive")
    if args.size <= 0:
        raise ValueError("--size must be positive")
    if args.dwell_seconds < 0:
        raise ValueError("--dwell-seconds must be >= 0")
    if args.loops < 1:
        raise ValueError("--loops must be >= 1")

    # Rough trajectory budget logged up front so the operator can sanity-check
    # duration before anything moves. Exact per-frame count is determined by
    # ``build_shape_trajectory`` once the anchor is known.
    if args.shape == "circle":
        circumference = 2.0 * np.pi * args.size
        est_shape_frames = max(
            8, round(circumference / args.velocity * args.fps)
        ) * args.loops
    else:
        perimeter = 4.0 * args.size
        est_shape_frames = max(
            8, round(perimeter / args.velocity * args.fps)
        ) * args.loops
    n_dwell_est = max(1, round(args.dwell_seconds * args.fps))
    est_total = 2 * n_dwell_est + est_shape_frames  # approx (excl. approach/return)
    logger.info(
        f"Approximate trajectory budget: ~{est_shape_frames} shape frames "
        f"({args.shape}, size={args.size} m, v={args.velocity} m/s, loops={args.loops}) "
        f"+ 2×{n_dwell_est} dwell frames ≈ {est_total} frames "
        f"({est_total / args.fps:.2f} s at {args.fps} Hz, approach/return extra)."
    )

    # Offer to clean up a stale dataset directory from a crashed previous
    # run BEFORE spinning up ROS / the env.
    if not args.resume:
        prompt_remove_existing_dataset(args.repo_id, assume_yes=args.yes)

    logger.info(f"Creating environment: {args.env_config}")
    env = make_env(
        env_type=args.env_config,
        control_type="cartesian",
        namespace="",
    )

    # Topic ownership must be fixed BEFORE wait_until_ready / set_target.
    logger.info(
        "Taking ownership of /target_pose (this script publishes setpoints; "
        "make sure track_mocap.py is NOT running)."
    )
    enable_env_target_pose_publishing(env)

    logger.info("Waiting for robot to be ready...")
    env.wait_until_ready()
    logger.info("Robot ready.")

    features = get_features(env=env)
    logger.debug(f"Features: {list(features.keys())}")

    rec_config = RecordingManagerConfig(
        features=features,
        repo_id=args.repo_id,
        robot_type="ur10e",
        fps=args.fps,
        num_episodes=1,
        resume=args.resume,
        push_to_hub=args.push_to_hub,
    )
    recording_manager = make_recording_manager(
        recording_manager_type="keyboard",
        config=rec_config,
    )
    recording_manager.wait_until_ready()
    logger.info("Recording manager ready.")

    # Home the arm before sampling the anchor. env.home() opens the gripper
    # and moves to the joint-space home via JTC, then we capture the
    # resulting cartesian pose as the trajectory anchor P0.
    logger.info("Homing robot...")
    env.home()
    env.reset()

    anchor_pose = env.robot.end_effector_pose  # Pose, deep copy
    logger.info(
        f"Anchor P0: position={anchor_pose.position.tolist()}, "
        f"euler_xyz={anchor_pose.orientation.as_euler('xyz', degrees=False).tolist()}"
    )
    logger.info(
        f"Shape: {args.shape}  size={args.size} m  plane={args.plane}  "
        f"loops={args.loops}  velocity={args.velocity} m/s"
    )

    trajectory = build_shape_trajectory(
        anchor=anchor_pose,
        shape=args.shape,
        size=args.size,
        plane=args.plane,
        velocity=args.velocity,
        fps=args.fps,
        loops=args.loops,
        dwell_seconds=args.dwell_seconds,
        include_gripper_toggle=args.include_gripper_toggle and env.gripper is not None,
    )
    logger.info(
        f"Built trajectory with {len(trajectory)} frames "
        f"({len(trajectory) / args.fps:.2f} s at {args.fps} Hz)."
    )

    if args.skip_workspace_check:
        logger.warning(
            "--skip-workspace-check: bypassing workspace-box validation. "
            "Make sure the trajectory is physically safe for your robot."
        )
    else:
        validate_against_workspace(trajectory, env.config.safety_box)
        logger.info("Trajectory waypoints are inside the workspace safety box.")

    if args.include_gripper_toggle and env.gripper is None:
        logger.warning(
            "--include-gripper-toggle set but env.gripper is None; gripper toggle skipped."
        )

    frame_idx = {"value": 0}

    def data_fn():
        i = frame_idx["value"]
        if i >= len(trajectory):
            return None, None

        target_pose, target_grip = trajectory[i]
        env.robot.set_target(pose=target_pose)
        if env.gripper is not None:
            env.gripper.set_target(float(target_grip))

        obs = env.get_obs()

        pose_vec = target_pose.to_array(
            representation=env.config.orientation_representation
        ).astype(np.float32)
        action = np.concatenate(
            [pose_vec, np.array([target_grip], dtype=np.float32)]
        )

        frame_idx["value"] = i + 1
        if frame_idx["value"] >= len(trajectory):
            # Auto-save: flip the state machine so record_episode's while
            # loop exits on the next iteration and _handle_post_episode
            # routes directly to SAVE_EPISODE without waiting on keyboard.
            recording_manager.state = "to_be_saved"

        return obs, action

    def on_start():
        env.reset()

    def on_end():
        env.robot.reset_targets()
        env.home(blocking=False)
        if env.gripper is not None:
            env.gripper.open()

    try:
        with recording_manager:
            # Auto-start: skip the 'r' keypress entirely. Must happen after
            # entering the context manager so the keyboard listener is alive
            # in case the user wants to hit 'q' to abort mid-run.
            logger.info("Auto-starting recording (no 'r' keypress required).")
            recording_manager.state = "recording"

            logger.info("Recording geometric calibration episode (1 / 1)")
            recording_manager.record_episode(
                data_fn=data_fn,
                task=args.task,
                on_start=on_start,
                on_end=on_end,
            )

            if frame_idx["value"] < len(trajectory):
                logger.warning(
                    f"Episode ended early: {frame_idx['value']} / {len(trajectory)} "
                    "frames written. Did you press 'r' or 'q' mid-run?"
                )
            else:
                logger.info(
                    f"Calibration trajectory complete: {frame_idx['value']} frames written."
                )

        logger.info("Recording complete. Homing robot.")
        env.home()

    except Exception:
        logger.exception("Error during geometric calibration recording.")
        raise
    finally:
        env.close()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
