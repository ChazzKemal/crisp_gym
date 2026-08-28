#!/usr/bin/env python3
"""Modular sensor recording — camera, joint states, EE pose, and optional
teleop actions.

Records frames from a compressed image topic, ``/joint_states``, and
``/current_pose`` into a LeRobot v3 dataset. Optionally records the
teleop action from ``/target_pose`` + ``/target_gripper_state`` when
``--with-action`` is passed (requires an active teleop source such as
``track_mocap.py`` or spacemouse).

Usage:
    # Passive observation (dummy action):
    pixi run -e jazzy-lerobot python examples/16_camera_only_record.py \\
        --repo-id sensor_test --fps 30 --num-episodes 1 --task "observe scene"

    # With teleop action (mocap or spacemouse must be publishing):
    pixi run -e jazzy-lerobot python examples/16_camera_only_record.py \\
        --repo-id teleop_test --fps 30 --with-action --task "pick cube"

    # Camera only (no joint states, no pose, no action):
    pixi run -e jazzy-lerobot python examples/16_camera_only_record.py \\
        --repo-id camera_test --fps 30 --no-joints --no-pose

Keyboard controls (same as scripts 12–15):
    r  →  start / stop recording
    s  →  save episode
    d  →  discard episode
    q  →  quit

Prerequisites:
    - Orbbec (or any camera) publishing CompressedImage on the topic.
      Default topic: /camera/color/image_raw  (the script appends /compressed).
    - For joint states: robot running with joint_state_broadcaster active.
    - For current pose: pose_broadcaster active (loaded by load_crisp.py).
    - For action: teleop source publishing /target_pose (and optionally
      /target_gripper_state for grip commands).
    - ROS2 environment configured (pixi shell -e jazzy-lerobot handles this).
"""

import argparse
import logging
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import qos_profile_sensor_data, qos_profile_system_default
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32

from crisp_py.camera.camera import Camera
from crisp_py.camera.camera_config import CameraConfig

from crisp_gym.record.recording_manager import make_recording_manager
from crisp_gym.record.recording_manager_config import RecordingManagerConfig
from crisp_gym.util.setup_logger import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repo-id", type=str, default="camera_only",
        help="LeRobot dataset name (default: camera_only)",
    )
    parser.add_argument(
        "--task", type=str, default="observe scene",
        help="Task description label for all episodes",
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="Recording frame rate (default: 30)",
    )
    parser.add_argument(
        "--num-episodes", type=int, default=5,
        help="Number of episodes to record, 0 = unlimited (default: 5)",
    )
    parser.add_argument(
        "--topic", type=str, default="/camera/color/image_raw",
        help="Base image topic (script appends /compressed). "
             "Default: /camera/color/image_raw",
    )
    parser.add_argument(
        "--resolution", type=int, nargs=2, default=[640, 480],
        metavar=("H", "W"),
        help="Target resolution height width (default: 640 480)",
    )
    parser.add_argument(
        "--joint-topic", type=str, default="/joint_states",
        help="Joint state topic (default: /joint_states)",
    )
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
    parser.add_argument(
        "--no-joints", action="store_true", default=False,
        help="Skip joint state recording (camera only)",
    )
    parser.add_argument(
        "--pose-topic", type=str, default="/current_pose",
        help="EE pose topic (PoseStamped). Default: /current_pose",
    )
    parser.add_argument(
        "--no-pose", action="store_true", default=False,
        help="Skip current pose recording",
    )
    parser.add_argument(
        "--with-action", action="store_true", default=False,
        help="Record action from /target_pose + /target_gripper_state "
             "(requires active teleop source). Without this flag, action "
             "is a dummy zero.",
    )
    parser.add_argument(
        "--target-pose-topic", type=str, default="/target_pose",
        help="Target pose topic for action (PoseStamped). Default: /target_pose",
    )
    parser.add_argument(
        "--gripper-target-topic", type=str, default="/target_gripper_state",
        help="Gripper target topic for action (Float32, 0=open 1=closed in "
             "crisp_py convention). Default: /target_gripper_state",
    )
    parser.add_argument(
        "--resume", action="store_true", default=False,
        help="Resume recording from an existing dataset",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    args = parser.parse_args()

    setup_logging(level=args.log_level)

    # ---- Camera ----
    logger.info("Initialising camera on %s/compressed", args.topic)
    rclpy.init()

    camera_config = CameraConfig(
        camera_color_image_topic=args.topic,
        camera_color_info_topic=None,
        camera_name="camera",
        resolution=args.resolution,
    )
    camera = Camera(config=camera_config, spin_node=True)

    logger.info("Waiting for first image...")
    camera.wait_until_ready(timeout=15.0)
    logger.info(
        "Camera ready — image shape %s, dtype %s",
        camera.current_image.shape, camera.current_image.dtype,
    )

    # ---- Joint state subscriber (optional) ----
    use_joints = not args.no_joints
    joint_lock = threading.Lock()
    joint_values: np.ndarray | None = None
    nq = len(args.joint_names)

    if use_joints:
        def _joint_callback(msg: JointState) -> None:
            nonlocal joint_values
            vals = np.zeros(nq, dtype=np.float32)
            name_to_idx = {n: i for i, n in enumerate(args.joint_names)}
            for name, pos in zip(msg.name, msg.position):
                if name in name_to_idx:
                    vals[name_to_idx[name]] = pos
            with joint_lock:
                joint_values = vals

        camera.node.create_subscription(
            JointState,
            args.joint_topic,
            _joint_callback,
            qos_profile_sensor_data,
            callback_group=ReentrantCallbackGroup(),
        )
        logger.info("Subscribed to %s for %d joints", args.joint_topic, nq)

        import time
        deadline = time.time() + 10.0
        while joint_values is None and time.time() < deadline:
            time.sleep(0.1)
        if joint_values is None:
            raise TimeoutError(
                f"No message on {args.joint_topic} within 10 s. "
                "Is joint_state_broadcaster running? Use --no-joints to skip."
            )
        logger.info("Joint states ready: %s", joint_values)

    # ---- Current pose subscriber (optional) ----
    use_pose = not args.no_pose
    pose_lock = threading.Lock()
    pose_values: np.ndarray | None = None

    if use_pose:
        def _pose_callback(msg: PoseStamped) -> None:
            nonlocal pose_values
            p = msg.pose.position
            q = msg.pose.orientation
            rpy = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")
            with pose_lock:
                pose_values = np.array(
                    [p.x, p.y, p.z, rpy[0], rpy[1], rpy[2]], dtype=np.float32
                )

        camera.node.create_subscription(
            PoseStamped,
            args.pose_topic,
            _pose_callback,
            qos_profile_sensor_data,
            callback_group=ReentrantCallbackGroup(),
        )
        logger.info("Subscribed to %s for EE pose", args.pose_topic)

        import time as _time
        deadline = _time.time() + 10.0
        while pose_values is None and _time.time() < deadline:
            _time.sleep(0.1)
        if pose_values is None:
            raise TimeoutError(
                f"No message on {args.pose_topic} within 10 s. "
                "Is pose_broadcaster active? Use --no-pose to skip."
            )
        logger.info("Current pose ready: %s", pose_values)

    # ---- Action subscribers: /target_pose + /target_gripper_state (optional) ----
    use_action = args.with_action
    target_pose_lock = threading.Lock()
    target_pose_values: np.ndarray | None = None
    gripper_target_lock = threading.Lock()
    gripper_target_value: float = 1.0  # crisp_py convention: 1=open (default until first grip command)
    gripper_target_received = False

    if use_action:
        def _target_pose_callback(msg: PoseStamped) -> None:
            nonlocal target_pose_values
            p = msg.pose.position
            q = msg.pose.orientation
            rpy = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")
            with target_pose_lock:
                target_pose_values = np.array(
                    [p.x, p.y, p.z, rpy[0], rpy[1], rpy[2]], dtype=np.float32
                )

        camera.node.create_subscription(
            PoseStamped,
            args.target_pose_topic,
            _target_pose_callback,
            qos_profile_sensor_data,
            callback_group=ReentrantCallbackGroup(),
        )
        logger.info("Subscribed to %s for action (target pose)", args.target_pose_topic)

        # track_mocap.py publishes RAW joint angles on /target_gripper_state
        # (0.0 = open, 0.8 = closed). env.step → gripper.set_target expects
        # NORMALIZED values (0.0 = closed, 1.0 = open). We normalize here
        # using the gripper config: normalized = (raw - min) / (max - min).
        # With min=0.8 (closed), max=0.0 (open): normalized = (0.8 - raw) / 0.8.
        _grip_min = 0.8   # raw closed position (from ur10e_ridgeback_env.yaml)
        _grip_max = 0.0   # raw open position
        def _normalize_grip(raw: float) -> float:
            return np.clip((raw - _grip_min) / (_grip_max - _grip_min), 0.0, 1.0)

        def _gripper_target_callback(msg: Float32) -> None:
            nonlocal gripper_target_value, gripper_target_received
            with gripper_target_lock:
                gripper_target_value = _normalize_grip(float(msg.data))
                gripper_target_received = True

        camera.node.create_subscription(
            Float32,
            args.gripper_target_topic,
            _gripper_target_callback,
            qos_profile_system_default,
            callback_group=ReentrantCallbackGroup(),
        )
        logger.info(
            "Subscribed to %s for action (gripper). Note: this topic only "
            "fires on grip changes — the last value is held between commands.",
            args.gripper_target_topic,
        )

        import time as _time2
        deadline = _time2.time() + 10.0
        while target_pose_values is None and _time2.time() < deadline:
            _time2.sleep(0.1)
        if target_pose_values is None:
            raise TimeoutError(
                f"No message on {args.target_pose_topic} within 10 s. "
                "Is the teleop source (track_mocap.py / spacemouse) running? "
                "Drop --with-action for passive recording."
            )
        logger.info("Target pose ready: %s", target_pose_values)
        if not gripper_target_received:
            logger.warning(
                "No gripper target received yet on %s — using 0.0 (closed) "
                "until the first grip command arrives.",
                args.gripper_target_topic,
            )

    # ---- Features (hand-built, no ManipulatorEnv needed) ----
    h, w = args.resolution

    state_components = []
    state_names = []
    if use_joints:
        state_components.append(("joints", nq, list(args.joint_names)))
    if use_pose:
        state_components.append(("cartesian", 6, ["x", "y", "z", "roll", "pitch", "yaw"]))
    if not state_components:
        state_components.append(("dummy", 1, ["dummy_state"]))

    state_dim = sum(dim for _, dim, _ in state_components)
    all_state_names = [n for _, _, names in state_components for n in names]

    features = {
        "observation.images.camera": {
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
        },
        "observation.timestamps.wall": {
            "dtype": "float64",
            "shape": (1,),
            "names": ["seconds"],
        },
        "observation.timestamps.camera_header": {
            "dtype": "float64",
            "shape": (1,),
            "names": ["seconds"],
        },
    }
    for comp_name, comp_dim, comp_names in state_components:
        features[f"observation.state.{comp_name}"] = {
            "dtype": "float32",
            "shape": (comp_dim,),
            "names": comp_names,
        }
    features["observation.state"] = {
        "dtype": "float32",
        "shape": (state_dim,),
        "names": all_state_names,
    }
    if use_action:
        features["action"] = {
            "dtype": "float32",
            "shape": (7,),
            "names": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
        }
    else:
        features["action"] = {
            "dtype": "float32",
            "shape": (1,),
            "names": ["dummy"],
        }

    # ---- Recording manager ----
    rec_config = RecordingManagerConfig(
        features=features,
        repo_id=args.repo_id,
        robot_type="camera",
        fps=args.fps,
        num_episodes=args.num_episodes,
        resume=args.resume,
    )
    recording_manager = make_recording_manager(
        recording_manager_type="keyboard",
        config=rec_config,
    )
    recording_manager.wait_until_ready()
    logger.info("Recording manager ready.")

    # ---- Data function ----
    dummy_action = np.zeros(1, dtype=np.float32)
    dummy_state = np.zeros(1, dtype=np.float32)

    def data_fn():
        obs = {"observation.images.camera": camera.current_image}
        obs["observation.timestamps.wall"] = np.array(
            [time.monotonic()], dtype=np.float64
        )
        obs["observation.timestamps.camera_header"] = np.array(
            [camera.current_image_stamp], dtype=np.float64
        )

        if use_joints:
            with joint_lock:
                obs["observation.state.joints"] = joint_values.copy()
        if use_pose:
            with pose_lock:
                obs["observation.state.cartesian"] = pose_values.copy()
        if not use_joints and not use_pose:
            obs["observation.state.dummy"] = dummy_state

        if use_action:
            with target_pose_lock:
                tp = target_pose_values.copy()
            with gripper_target_lock:
                grip = gripper_target_value
            action = np.concatenate([tp, np.array([grip], dtype=np.float32)])
        else:
            action = dummy_action

        return obs, action

    # ---- Record loop ----
    try:
        with recording_manager:
            while not recording_manager.done():
                ep_num = recording_manager.episode_count + 1
                num_str = str(args.num_episodes) if args.num_episodes > 0 else "inf"
                logger.info("Episode %d / %s", ep_num, num_str)
                recording_manager.record_episode(
                    data_fn=data_fn,
                    task=args.task,
                )
        logger.info("Recording complete.")
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
    except Exception:
        logger.exception("Recording failed.")
        raise
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
