#!/usr/bin/env python3
"""Dual-camera teleop recording through the C++ crisp_camera_bridge (SHM path).

This script is the spiritual successor to ``16_camera_only_record.py`` but:

  * Records BOTH cameras (Orbbec ``/camera`` + RealSense D405 ``/d405``) into
    a single LeRobot v3 dataset under ``observation.images.camera`` and
    ``observation.images.d405``.
  * Pulls frames from the C++ crisp_camera_bridge via POSIX shared memory
    (``/dev/shm/crisp_camera_{camera,d405}``) instead of subscribing in
    Python. No ``cv_bridge`` JPEG decode and no rclpy wait-set bookkeeping
    on the per-frame path → no GIL contention with the recording loop.
  * Starts the LeRobot image writer inside the writer subprocess (same fix
    as ``16_ridgeback_mocap_record_fast.py``) so two camera streams at 30 Hz
    don't fill the mp queue and starve the recording loop.

State / action recorded:

  observation.images.{camera,d405}              video (640x480x3 RGB)
  observation.timestamps.wall                   monotonic seconds
  observation.timestamps.camera_header.{name}   ROS header.stamp (s) per cam
  observation.state.joints                      6 UR10e joints (rad)
                                                  from /joint_states
  observation.state.gripper                     1 normalized gripper state
                                                  (1=open, 0=closed), from
                                                  /gripper/joint_states
  observation.state.cartesian                   EE pose [x,y,z,rpy] (m, rad)
                                                  from /current_pose, "just in
                                                  case" — not strictly needed
                                                  if the policy reads joints +
                                                  cameras only
  action                                        [x,y,z,rpy,gripper] (7d)
                                                  target_pose + normalized
                                                  gripper target

Drop ``--no-pose`` / ``--no-joints`` / ``--no-action`` to omit any of the
non-essential streams.

Prereqs that MUST be running before this script:

  1. Orbbec on ``/camera/color/image_raw``
       cd clearpath_remote_ws && pixi run orbbec
  2. RealSense D405 on ``/d405/camera/color/image_rect_raw``
       cd clearpath_remote_ws && pixi run realsense
  3. C++ camera bridge populating /dev/shm/crisp_camera_{camera,d405}
       ./tools/master_launch.sh camera-bridge-cpp
     (rebuild after touching it:
       cd clearpath_remote_ws && pixi run colcon build --packages-select tum09_custom)
  4. Robot + controllers + a teleop source publishing /target_pose
     (mocap track or spacemouse). E.g.
       ./tools/master_launch.sh up --track

Usage::

    cd Yunfei/crisp_gym
    pixi run -e jazzy-lerobot python examples/26_dual_cam_shm_record.py \\
        --repo-id pick_red_cube_dual --task "pick up the red cube" \\
        --fps 30 --num-episodes 20

Keyboard controls (same as scripts 12 / 15 / 16):
    r → start / stop recording
    s → save episode
    d → discard episode
    q → quit
"""

# IMPORT ORDER MATTERS — touching ``cv2`` (transitively via ``crisp_py.camera``)
# before any torch / torchvision / lerobot import pins libjpeg.so.8 to the
# conda env's copy and avoids the libtiff relocation crash documented in
# 16_ridgeback_mocap_record_fast.py. Keep crisp_py / crisp_gym imports above
# the recording_manager import.
import argparse
import logging
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data, qos_profile_system_default
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32

from crisp_py.camera.camera import Camera
from crisp_py.camera.camera_config import CameraConfig

from crisp_gym.record.recording_manager import (
    HF_LEROBOT_HOME,
    KeyboardRecordingManager,
    LeRobotDataset,
)
from crisp_gym.record.recording_manager_config import RecordingManagerConfig
from crisp_gym.util.setup_logger import setup_logging

logger = logging.getLogger(__name__)


# Must match ``config/envs/ur10e_ridgeback_dual_cam_env.yaml`` AND
# ``clearpath_remote_ws/.../config/crisp_camera_bridge.yaml``. The bridge
# writes to ``/dev/shm/crisp_camera_<name>`` so ``name`` is what links the
# two sides; topic / resolution are stored here only for dataset metadata
# (the bridge does the actual subscribing).
DEFAULT_CAMERAS: list[dict] = [
    {
        "name": "camera",
        "topic": "/camera/color/image_raw",
        "frame": "camera_frame",
        "resolution": [480, 640],
        "max_image_delay": 1.0,
    },
    {
        "name": "d405",
        "topic": "/d405/camera/color/image_rect_raw",
        "frame": "d405_color_optical_frame",
        "resolution": [480, 640],
        "max_image_delay": 0.5,
    },
]


class FastKeyboardRecordingManager(KeyboardRecordingManager):
    """``KeyboardRecordingManager`` with the LeRobot image writer enabled.

    Copy of the same subclass from ``16_ridgeback_mocap_record_fast.py`` —
    starts ``dataset.start_image_writer`` inside the writer subprocess so
    multi-camera frame encoding doesn't block the recording loop. See
    ``docs/ridgeback_mocap_record_fps_image_writer.md`` for the writeup.
    """

    def _create_dataset(self) -> LeRobotDataset:
        if self.config.resume:
            logger.info("Resuming dataset: %s", self.config.repo_id)
            dataset = LeRobotDataset(repo_id=self.config.repo_id)
            if self.config.num_episodes <= dataset.num_episodes:
                logger.error(
                    "Dataset already has %d episodes — pass a larger --num-episodes.",
                    dataset.num_episodes,
                )
                raise SystemExit(1)
            self.episode_count_queue.put(dataset.num_episodes - 1)
        else:
            logger.info("Creating new dataset: %s", self.config.repo_id)
            existing = Path(HF_LEROBOT_HOME / self.config.repo_id)
            if existing.exists():
                raise FileExistsError(
                    f"Dataset already exists: {existing}\n"
                    "Use --resume to extend it, or `rm -r` it first."
                )
            dataset = LeRobotDataset.create(
                repo_id=self.config.repo_id,
                fps=self.config.fps,
                robot_type=self.config.robot_type,
                features=self.config.features,
                use_videos=True,
                image_writer_threads=1,
                image_writer_processes=16,
            )
            n_proc = getattr(self.config, "image_writer_processes", 8)
            n_thr = getattr(self.config, "image_writer_threads", 1)
            if n_proc > 0:
                logger.info("Starting image writer: processes=%d threads=%d", n_proc, n_thr)
                dataset.start_image_writer(num_processes=n_proc, num_threads=n_thr)
        return dataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--repo-id", type=str, default="dual_cam_shm_record",
                        help="LeRobot dataset name (default: dual_cam_shm_record)")
    parser.add_argument("--task", type=str, default="perform task",
                        help="Task description label applied to every episode")
    parser.add_argument("--fps", type=int, default=30,
                        help="Recording frame rate (default: 30 — matches the bridge "
                             "publish cadence at the dev rig)")
    parser.add_argument("--num-episodes", type=int, default=20,
                        help="Number of episodes to record (0 = unlimited; default: 20)")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Append to an existing dataset under --repo-id")

    parser.add_argument("--joint-topic", type=str, default="/joint_states")
    parser.add_argument(
        "--joint-names", type=str, nargs="+",
        default=[
            "arm_0_shoulder_pan_joint",
            "arm_0_shoulder_lift_joint",
            "arm_0_elbow_joint",
            "arm_0_wrist_1_joint",
            "arm_0_wrist_2_joint",
            "arm_0_wrist_3_joint",
        ],
        help="Joint names to extract from /joint_states (default: UR10e 6 joints)",
    )
    parser.add_argument("--no-joints", action="store_true", default=False)
    parser.add_argument("--gripper-joint-state-topic", type=str,
                        default="/gripper/joint_states",
                        help="Gripper JointState topic — first joint pos taken as the "
                             "gripper observation, normalized to [0..1] using "
                             "(raw - 0.8) / (0.0 - 0.8) (1=open).")
    parser.add_argument("--no-gripper-state", action="store_true", default=False)
    parser.add_argument("--pose-topic", type=str, default="/current_pose")
    parser.add_argument("--no-pose", action="store_true", default=False)
    parser.add_argument("--target-pose-topic", type=str, default="/target_pose")
    parser.add_argument("--gripper-target-topic", type=str,
                        default="/target_gripper_state")
    parser.add_argument("--no-action", action="store_true", default=False,
                        help="Record a dummy zero action (passive observation mode)")
    parser.add_argument("--require-teleop", action="store_true", default=False,
                        help="Abort startup if no /target_pose seen within 5 s")

    parser.add_argument("--image-writer-processes", type=int, default=8,
                        help="LeRobot image writer worker procs (8 covers 2x640x480@30 Hz)")
    parser.add_argument("--image-writer-threads", type=int, default=1)
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    return parser.parse_args()


# Match ``ur10e_ridgeback_env.yaml`` gripper_config: min=0.8 (closed raw joint
# pos), max=0.0 (open raw joint pos). track_mocap.py publishes RAW values on
# /target_gripper_state; dataset action expects the normalized convention
# (0=closed, 1=open) used by crisp_py's Gripper class.
_GRIP_RAW_CLOSED = 0.8
_GRIP_RAW_OPEN = 0.0


def _normalize_grip(raw: float) -> float:
    return float(np.clip(
        (raw - _GRIP_RAW_CLOSED) / (_GRIP_RAW_OPEN - _GRIP_RAW_CLOSED), 0.0, 1.0,
    ))


def main() -> None:
    args = _parse_args()
    setup_logging(level=args.log_level)
    rclpy.init()

    # ------------------------------------------------------------------ #
    # Cameras — opened in SHM mode, no rclpy subscriptions.
    # ------------------------------------------------------------------ #
    cameras: dict[str, Camera] = {}
    shared_node = None
    for entry in DEFAULT_CAMERAS:
        cfg = CameraConfig(
            camera_color_image_topic=entry["topic"],
            camera_color_info_topic=None,
            camera_name=entry["name"],
            camera_frame=entry["frame"],
            resolution=entry["resolution"],
            max_image_delay=entry["max_image_delay"],
            use_shared_memory=True,
        )
        logger.info(
            "Attaching to /dev/shm/crisp_camera_%s (topic %s)",
            entry["name"], entry["topic"],
        )
        cam = Camera(config=cfg, node=shared_node, spin_node=False)
        if shared_node is None:
            shared_node = cam.node
        cam.wait_until_ready(timeout=15.0)
        img = cam.current_image
        cameras[entry["name"]] = cam
        logger.info(
            "  ready — shape=%s dtype=%s header_stamp=%.3fs",
            img.shape, img.dtype, cam.current_image_stamp,
        )

    # ------------------------------------------------------------------ #
    # Auxiliary subscriptions (joints / current pose / action target / gripper).
    # All share ``shared_node`` and one MultiThreadedExecutor spin thread.
    # ------------------------------------------------------------------ #
    joint_lock = threading.Lock()
    joint_values: np.ndarray | None = None
    gripper_state_lock = threading.Lock()
    gripper_state_value: float | None = None   # normalized [0..1], 1=open
    pose_lock = threading.Lock()
    pose_values: np.ndarray | None = None
    target_pose_lock = threading.Lock()
    target_pose_values: np.ndarray | None = None
    gripper_target_lock = threading.Lock()
    gripper_target_value: float = 1.0   # default normalized = open
    gripper_target_received = False

    use_joints = not args.no_joints
    use_gripper_state = not args.no_gripper_state
    use_pose = not args.no_pose
    use_action = not args.no_action
    nq = len(args.joint_names)
    name_to_idx = {n: i for i, n in enumerate(args.joint_names)}

    if use_joints:
        def _on_joints(msg: JointState) -> None:
            nonlocal joint_values
            vals = np.zeros(nq, dtype=np.float32)
            for name, pos in zip(msg.name, msg.position):
                i = name_to_idx.get(name)
                if i is not None:
                    vals[i] = pos
            with joint_lock:
                joint_values = vals

        shared_node.create_subscription(
            JointState, args.joint_topic, _on_joints,
            qos_profile_sensor_data,
            callback_group=ReentrantCallbackGroup(),
        )
        logger.info("Subscribed to %s for %d joints", args.joint_topic, nq)

    if use_gripper_state:
        def _on_gripper_state(msg: JointState) -> None:
            nonlocal gripper_state_value
            if not msg.position:
                return
            raw = float(msg.position[0])
            with gripper_state_lock:
                gripper_state_value = _normalize_grip(raw)

        shared_node.create_subscription(
            JointState, args.gripper_joint_state_topic, _on_gripper_state,
            qos_profile_sensor_data,
            callback_group=ReentrantCallbackGroup(),
        )
        logger.info("Subscribed to %s for gripper state",
                    args.gripper_joint_state_topic)

    if use_pose:
        def _on_pose(msg: PoseStamped) -> None:
            nonlocal pose_values
            p, q = msg.pose.position, msg.pose.orientation
            rpy = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")
            with pose_lock:
                pose_values = np.array(
                    [p.x, p.y, p.z, rpy[0], rpy[1], rpy[2]], dtype=np.float32,
                )

        shared_node.create_subscription(
            PoseStamped, args.pose_topic, _on_pose,
            qos_profile_sensor_data,
            callback_group=ReentrantCallbackGroup(),
        )
        logger.info("Subscribed to %s for EE pose", args.pose_topic)

    if use_action:
        def _on_target_pose(msg: PoseStamped) -> None:
            nonlocal target_pose_values
            p, q = msg.pose.position, msg.pose.orientation
            rpy = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")
            with target_pose_lock:
                target_pose_values = np.array(
                    [p.x, p.y, p.z, rpy[0], rpy[1], rpy[2]], dtype=np.float32,
                )

        shared_node.create_subscription(
            PoseStamped, args.target_pose_topic, _on_target_pose,
            qos_profile_sensor_data,
            callback_group=ReentrantCallbackGroup(),
        )
        logger.info("Subscribed to %s for action target pose", args.target_pose_topic)

        def _on_grip_target(msg: Float32) -> None:
            nonlocal gripper_target_value, gripper_target_received
            with gripper_target_lock:
                gripper_target_value = _normalize_grip(float(msg.data))
                gripper_target_received = True

        shared_node.create_subscription(
            Float32, args.gripper_target_topic, _on_grip_target,
            qos_profile_system_default,
            callback_group=ReentrantCallbackGroup(),
        )
        logger.info(
            "Subscribed to %s for action gripper target", args.gripper_target_topic,
        )

    executor = MultiThreadedExecutor()
    executor.add_node(shared_node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    def _wait_for(predicate, label: str, timeout: float, fatal: bool) -> None:
        deadline = time.time() + timeout
        while not predicate() and time.time() < deadline:
            time.sleep(0.05)
        if not predicate():
            msg = f"No data on {label} within {timeout:.0f}s."
            if fatal:
                raise TimeoutError(msg)
            logger.warning(msg)

    if use_joints:
        _wait_for(lambda: joint_values is not None, args.joint_topic, 10.0, fatal=True)
        logger.info("Joints initial: %s", joint_values)
    if use_gripper_state:
        _wait_for(lambda: gripper_state_value is not None,
                  args.gripper_joint_state_topic, 10.0, fatal=True)
        logger.info("Gripper state initial (normalized): %.3f", gripper_state_value)
    if use_pose:
        _wait_for(lambda: pose_values is not None, args.pose_topic, 10.0, fatal=True)
        logger.info("Pose initial: %s", pose_values)
    if use_action:
        _wait_for(
            lambda: target_pose_values is not None,
            args.target_pose_topic, 5.0, fatal=args.require_teleop,
        )
        if target_pose_values is not None:
            logger.info("Target pose initial: %s", target_pose_values)
        if not gripper_target_received:
            logger.warning(
                "No %s yet — holding default (open) until first grip command.",
                args.gripper_target_topic,
            )

    # ------------------------------------------------------------------ #
    # LeRobot dataset features.
    # ------------------------------------------------------------------ #
    features: dict[str, dict] = {}
    for cam_name, cam in cameras.items():
        h, w, _ = cam.current_image.shape
        features[f"observation.images.{cam_name}"] = {
            "dtype": "video",
            "shape": (h, w, 3),
            "names": ["height", "width", "channels"],
            "video_info": {
                "video.fps": args.fps,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
        features[f"observation.timestamps.camera_header.{cam_name}"] = {
            "dtype": "float64", "shape": (1,), "names": ["seconds"],
        }
    features["observation.timestamps.wall"] = {
        "dtype": "float64", "shape": (1,), "names": ["seconds"],
    }

    state_components: list[tuple[str, int, list[str]]] = []
    if use_joints:
        state_components.append(("joints", nq, list(args.joint_names)))
    if use_gripper_state:
        state_components.append(("gripper", 1, ["gripper"]))
    if use_pose:
        state_components.append(("cartesian", 6, ["x", "y", "z", "roll", "pitch", "yaw"]))
    if not state_components:
        state_components.append(("dummy", 1, ["dummy_state"]))
    state_dim = sum(d for _, d, _ in state_components)
    all_state_names = [n for _, _, names in state_components for n in names]
    for comp_name, comp_dim, comp_names in state_components:
        features[f"observation.state.{comp_name}"] = {
            "dtype": "float32", "shape": (comp_dim,), "names": comp_names,
        }
    features["observation.state"] = {
        "dtype": "float32", "shape": (state_dim,), "names": all_state_names,
    }
    if use_action:
        features["action"] = {
            "dtype": "float32", "shape": (7,),
            "names": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
        }
    else:
        features["action"] = {"dtype": "float32", "shape": (1,), "names": ["dummy"]}

    # ------------------------------------------------------------------ #
    # Recording manager.
    # ------------------------------------------------------------------ #
    rec_config = RecordingManagerConfig(
        features=features,
        repo_id=args.repo_id,
        robot_type="ur10e",
        fps=args.fps,
        num_episodes=args.num_episodes,
        resume=args.resume,
    )
    rec_config.image_writer_processes = args.image_writer_processes
    rec_config.image_writer_threads = args.image_writer_threads
    recording_manager = FastKeyboardRecordingManager(config=rec_config)
    recording_manager.wait_until_ready()
    logger.info("Recording manager ready.")

    dummy_state = np.zeros(1, dtype=np.float32)
    dummy_action = np.zeros(1, dtype=np.float32)
    zero_pose = np.zeros(6, dtype=np.float32)

    def data_fn():
        obs: dict[str, np.ndarray] = {}
        obs["observation.timestamps.wall"] = np.array(
            [time.monotonic()], dtype=np.float64,
        )
        for cam_name, cam in cameras.items():
            obs[f"observation.images.{cam_name}"] = cam.current_image
            obs[f"observation.timestamps.camera_header.{cam_name}"] = np.array(
                [cam.current_image_stamp], dtype=np.float64,
            )

        if use_joints:
            with joint_lock:
                obs["observation.state.joints"] = (
                    joint_values.copy() if joint_values is not None
                    else np.zeros(nq, dtype=np.float32)
                )
        if use_gripper_state:
            with gripper_state_lock:
                gv = gripper_state_value if gripper_state_value is not None else 1.0
            obs["observation.state.gripper"] = np.array([gv], dtype=np.float32)
        if use_pose:
            with pose_lock:
                obs["observation.state.cartesian"] = (
                    pose_values.copy() if pose_values is not None else zero_pose.copy()
                )
        if not (use_joints or use_gripper_state or use_pose):
            obs["observation.state.dummy"] = dummy_state.copy()

        if use_action:
            with target_pose_lock:
                tp = (target_pose_values.copy() if target_pose_values is not None
                      else zero_pose.copy())
            with gripper_target_lock:
                gt = gripper_target_value
            action = np.concatenate(
                [tp, np.array([gt], dtype=np.float32)],
            ).astype(np.float32)
        else:
            action = dummy_action

        return obs, action

    # ------------------------------------------------------------------ #
    # Record loop.
    # ------------------------------------------------------------------ #
    try:
        with recording_manager:
            while not recording_manager.done():
                ep_num = recording_manager.episode_count + 1
                num_str = str(args.num_episodes) if args.num_episodes > 0 else "inf"
                logger.info("Episode %d / %s", ep_num, num_str)
                recording_manager.record_episode(data_fn=data_fn, task=args.task)
        logger.info("Recording complete.")
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
    except Exception:
        logger.exception("Recording failed.")
        raise
    finally:
        executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
