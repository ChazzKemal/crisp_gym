#!/usr/bin/env python3
"""Scripted calibration recorder for UR10e crisp_gym — record/replay smoke test.

This script drives the arm itself along a *deterministic* cartesian trajectory
with constant orientation and records each commanded frame as a LeRobot v3
dataset. There is no mocap, no SpaceMouse, no human input — the setpoints are
computed in code. Whatever ends up on disk can be diffed against the
ground-truth setpoint sequence to isolate record-path bugs from replay-path
bugs from mocap quality. See
``docs/ridgeback_calibration_recording_plan.md`` for the full motivation.

This script is intentionally independent of ``13_/15_ridgeback_mocap_record.py``
and ``14_ridgeback_replay.py`` — it reimplements the topic-ownership patch
inline rather than importing from script 14, so that edits to any of the
existing scripts cannot break this one.

Trajectory (at 20 Hz, 5 cm/s, ±10 cm, 0.5 s dwells, gripper toggle on):

    dwell(open) → +z → dwell(open) → −z(via anchor) → dwell(open)
                → −z → dwell(open) → +z(via anchor) → dwell(open)
                → +y(close) → dwell(closed) → −y(via anchor) → dwell(closed)
                → −y → dwell(closed) → +y(via anchor) → dwell(closed→open)

8 motion phases × 40 frames + 9 dwells × 10 frames = 410 frames ≈ 20.5 s.

Usage:
    # 1. Bring up the robot with CRISP cartesian controller:
    #        ./tools/master_launch.sh up --controller crisp
    #    (NO --track — mocap / track_mocap.py must NOT be running, because
    #    this script owns /target_pose itself.)
    #
    # 2. Run the calibration recorder:
    cd Yunfei/crisp_gym
    pixi run -e jazzy-lerobot python examples/16_calibration_record.py \\
        --repo-id calib_axis_001

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
# Topic-ownership patch — inverse of ``silence_env_target_publishers`` from
# scripts 13/15. Copied inline from 14_ridgeback_replay.py:269 so this script
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
    """Linear cartesian interpolation from ``start`` (exclusive) to ``end`` (inclusive).

    ``n_frames`` ticks are emitted at fractions ``1/n, 2/n, …, 1``, so the
    last frame is exactly at ``end_pos`` and the first frame is one step off
    ``start_pos``. This is what we want when segments are chained through a
    shared dwell — the dwell already covered ``start_pos``.
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


def build_trajectory(
    anchor: Pose,
    displacement: float,
    n_motion: int,
    n_dwell: int,
    include_gripper_toggle: bool,
) -> List[Tuple[Pose, float]]:
    """Build the full calibration trajectory as a list of (Pose, grip) frames.

    Structure: 9 dwells × n_dwell + 8 motion phases × n_motion frames,
    alternating dwell/motion/dwell/…/motion/dwell. Orientation is held
    constant at ``anchor.orientation`` throughout — any nonzero variation in
    the recorded orientation columns is therefore a representation bug.

    Gripper convention is crisp_py: ``1.0 = open``, ``0.0 = closed``. When
    ``include_gripper_toggle`` is true the gripper is open for the z-axis
    phases and the first y-axis close segment, closed through the rest of
    the y-axis phases, and reopens at the final dwell. Otherwise the grip
    stays open for the whole trajectory.
    """
    anchor_pos = anchor.position.copy()
    orientation = Rotation.from_quat(anchor.orientation.as_quat())  # deep copy

    p_up = anchor_pos + np.array([0.0, 0.0, +displacement])
    p_down = anchor_pos + np.array([0.0, 0.0, -displacement])
    p_left = anchor_pos + np.array([0.0, +displacement, 0.0])
    p_right = anchor_pos + np.array([0.0, -displacement, 0.0])

    OPEN = 1.0
    CLOSED = 0.0

    if include_gripper_toggle:
        g_dwell0 = OPEN  # pre-phase-1
        g_phase1 = OPEN  # up
        g_dwell1 = OPEN  # at +z
        g_phase2 = OPEN  # back to anchor
        g_dwell2 = OPEN  # at anchor
        g_phase3 = OPEN  # down
        g_dwell3 = OPEN  # at -z
        g_phase4 = OPEN  # back to anchor
        g_dwell4 = OPEN  # at anchor (pre-y)
        g_phase5 = CLOSED  # left — close at phase 5 start
        g_dwell5 = CLOSED  # at +y
        g_phase6 = CLOSED  # back to anchor
        g_dwell6 = CLOSED  # at anchor
        g_phase7 = CLOSED  # right
        g_dwell7 = CLOSED  # at -y
        g_phase8 = CLOSED  # back to anchor
        g_dwell8 = OPEN  # post-phase-8 — open at phase 8 end
    else:
        (
            g_dwell0,
            g_phase1,
            g_dwell1,
            g_phase2,
            g_dwell2,
            g_phase3,
            g_dwell3,
            g_phase4,
            g_dwell4,
            g_phase5,
            g_dwell5,
            g_phase6,
            g_dwell6,
            g_phase7,
            g_dwell7,
            g_phase8,
            g_dwell8,
        ) = (OPEN,) * 17

    traj: List[Tuple[Pose, float]] = []
    traj += _dwell(anchor_pos, orientation, n_dwell, g_dwell0)
    traj += _linear_segment(anchor_pos, p_up, orientation, n_motion, g_phase1)
    traj += _dwell(p_up, orientation, n_dwell, g_dwell1)
    traj += _linear_segment(p_up, anchor_pos, orientation, n_motion, g_phase2)
    traj += _dwell(anchor_pos, orientation, n_dwell, g_dwell2)
    traj += _linear_segment(anchor_pos, p_down, orientation, n_motion, g_phase3)
    traj += _dwell(p_down, orientation, n_dwell, g_dwell3)
    traj += _linear_segment(p_down, anchor_pos, orientation, n_motion, g_phase4)
    traj += _dwell(anchor_pos, orientation, n_dwell, g_dwell4)
    traj += _linear_segment(anchor_pos, p_left, orientation, n_motion, g_phase5)
    traj += _dwell(p_left, orientation, n_dwell, g_dwell5)
    traj += _linear_segment(p_left, anchor_pos, orientation, n_motion, g_phase6)
    traj += _dwell(anchor_pos, orientation, n_dwell, g_dwell6)
    traj += _linear_segment(anchor_pos, p_right, orientation, n_motion, g_phase7)
    traj += _dwell(p_right, orientation, n_dwell, g_dwell7)
    traj += _linear_segment(p_right, anchor_pos, orientation, n_motion, g_phase8)
    traj += _dwell(anchor_pos, orientation, n_dwell, g_dwell8)

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
        n_episodes = -1  # unknown / unreadable

    if n_episodes > 0:
        summary = f"{n_episodes} recorded episode(s)"
    elif n_episodes == 0:
        summary = "no recorded episodes (empty stub from a previous run)"
    else:
        summary = "unknown contents"

    print(
        f"\n[calibration] LeRobot dataset directory already exists:\n"
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
                f"limits: {', '.join(violated)}. Check the anchor pose and "
                f"--displacement, or widen the limits in ur10e_ridgeback_env.yaml."
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: C901 — linear top-level script
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--repo-id", type=str, default="calib_axis_001")
    parser.add_argument("--task", type=str, default="calibration sweep")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--velocity",
        type=float,
        default=0.05,
        help="Cartesian speed during motion phases (m/s). Default 5 cm/s.",
    )
    parser.add_argument(
        "--displacement",
        type=float,
        default=0.10,
        help="Half-range of each axis sweep (metres). Default 10 cm.",
    )
    parser.add_argument(
        "--dwell-seconds",
        type=float,
        default=0.5,
        help="Dwell duration at the start/end of each phase (seconds).",
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
    if args.displacement <= 0:
        raise ValueError("--displacement must be positive")
    if args.dwell_seconds < 0:
        raise ValueError("--dwell-seconds must be >= 0")

    n_motion = max(1, round((args.displacement / args.velocity) * args.fps))
    n_dwell = max(1, round(args.dwell_seconds * args.fps))
    total_frames = 8 * n_motion + 9 * n_dwell
    logger.info(
        f"Trajectory budget: {n_motion} motion frames × 8 + {n_dwell} dwell frames × 9 "
        f"= {total_frames} frames ({total_frames / args.fps:.2f} s at {args.fps} Hz)."
    )

    # Offer to clean up a stale dataset directory from a crashed previous
    # run BEFORE spinning up ROS / the env — if the user declines we exit
    # immediately without touching any hardware. Skipped when --resume is
    # set, since resume expects the directory to exist.
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

    trajectory = build_trajectory(
        anchor=anchor_pose,
        displacement=args.displacement,
        n_motion=n_motion,
        n_dwell=n_dwell,
        include_gripper_toggle=args.include_gripper_toggle and env.gripper is not None,
    )
    logger.info(f"Built trajectory with {len(trajectory)} frames.")

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
            # Defensive: state should already be "to_be_saved" at this point.
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

            logger.info("Recording calibration episode (1 / 1)")
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
        logger.exception("Error during calibration recording.")
        raise
    finally:
        env.close()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
