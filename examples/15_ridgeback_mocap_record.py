#!/usr/bin/env python3
"""UR10e crisp_gym environment — mocap-driven data recording (script 15).

Iterated copy of ``13_ridgeback_mocap_record.py``. The dataset format and
the recording flow are unchanged. The only difference is one bug fix:

- ``silence_env_target_publishers`` now tolerates
  ``env.robot._target_pose_publisher is None``. Script 13 unconditionally
  reassigned ``publish`` on it, which crashes when
  ``ur10e_ridgeback_env.yaml`` has ``publish_target_pose: false`` (the
  current default for mocap recording), because in that branch
  ``crisp_py.Robot.__init__`` never creates the publisher in the first
  place. See ``docs/ridgeback_target_pose_ownership.md`` (fix #4 in the
  "Suggested proper fixes" section).

Original docstring follows.

This is the dedicated recorder for mocap teleop flows where the target pose
is published to `/target_pose` by an *external* node (typically
`track_mocap.py` in clearpath_remote_ws). It is a sibling of
`12_ridgeback_record.py`, but differs in two critical ways so that mocap
can actually drive the arm while the recorder is running:

1. The recorder's own ``crisp_py.Robot`` is prevented from republishing to
   ``/target_pose`` and ``target_joint``. ``crisp_py.Robot`` normally owns
   those topics and ticks a 20 Hz timer that re-sends whatever its internal
   ``_target_pose`` buffer currently holds. During mocap teleop that buffer
   is frozen at whatever the arm pose happened to be when the episode
   started, so the controller ends up averaging mocap's commands with a
   stale "hold here" stream and the arm barely moves. We silence those
   publishers at the instance level so mocap is the sole authority on
   ``/target_pose``.

2. The action column for every recorded frame is sourced from a dedicated
   subscription to ``/target_pose`` — i.e. exactly what the mocap tracker
   just told the controller to do. Without this, the Cartesian half of the
   action would be a constant (again, the frozen env buffer), which is
   useless for imitation learning.

The gripper action uses the same mechanism as ``12_ridgeback_record.py``:
we read ``env.gripper.target`` when the mocap tracker has issued a
``/target_gripper_state`` message, and fall back to the observed gripper
value before the tracker's first grip command lands.

Usage:
    # 1. (Terminal 1) Start robot, controllers and mocap in clearpath_remote_ws.
    #    Cleanest: use the master launcher, which toggles cartesian_controller
    #    and starts track_mocap.py for you:
    #        ./tools/master_launch.sh up --track --controller crisp
    #
    #    Or manually inside clearpath_remote_ws:
    #        pixi run -- python src/tum09_ridgeback/tum09_custom/scripts/toggle_controller.py
    #        pixi run -- python src/tum09_ridgeback/tum09_custom/scripts/track_mocap.py
    #
    # 2. (Terminal 2) Run this recorder:
    python examples/13_ridgeback_mocap_record.py \\
        --repo-id pick_red_cube_001 \\
        --task "pick up the red cube and place it in the bowl" \\
        --num-episodes 50

Keyboard controls (episode management):
    r  →  start recording
    r  →  stop / pause recording
    s  →  save episode
    d  →  discard episode
    q  →  quit

Prerequisites:
    - cartesian_controller active on the robot
    - track_mocap.py (or another node) publishing /target_pose
      and, for grip commands, /target_gripper_state
    - joint_state_broadcaster + pose_broadcaster active
    - Camera running (e.g. pixi run orbbec in clearpath_remote_ws)
"""

import argparse
import logging
import threading

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import qos_profile_sensor_data

from crisp_gym.envs.manipulator_env import make_env
from crisp_gym.record.recording_manager import make_recording_manager
from crisp_gym.record.recording_manager_config import RecordingManagerConfig
from crisp_gym.util.lerobot_features import get_features
from crisp_gym.util.setup_logger import setup_logging
from crisp_py.utils.geometry import Pose


class MocapTargetCapture:
    """Thread-safe holder for the latest ``/target_pose`` message from mocap.

    We subscribe to ``/target_pose`` on the env's ROS node and store the most
    recent pose. ``get()`` returns a snapshot (or ``None`` if nothing has
    arrived yet) and is safe to call from the main recording loop while the
    ROS executor thread is writing.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pose: Pose | None = None
        self._count = 0

    def on_msg(self, msg: PoseStamped) -> None:
        pose = Pose.from_ros_msg(msg)
        with self._lock:
            self._pose = pose
            self._count += 1

    def get(self) -> Pose | None:
        with self._lock:
            return self._pose.copy() if self._pose is not None else None

    @property
    def count(self) -> int:
        with self._lock:
            return self._count


def silence_env_target_publishers(env) -> None:
    """Neuter the env's own ``/target_pose`` and ``target_joint`` publishers.

    Two publishers can fight the mocap tracker for control of the arm:

    1. ``_target_pose_publisher`` — only created when the YAML has
       ``publish_target_pose: true``. ``ur10e_ridgeback_env.yaml`` sets it
       to ``false`` so this publisher is normally ``None`` and there is
       nothing to silence. We still tolerate the ``true`` case defensively
       in case the YAML is ever flipped back.

    2. ``_target_joint_publisher`` — ``crisp_py.Robot`` always creates this
       one regardless of ``publish_target_pose``, plus a 20 Hz timer that
       re-sends ``self._target_joint``. After the home move that buffer
       holds the arm's joint positions at episode start, which is a
       competing command path during mocap teleop. This is the publisher
       that actually matters to silence.

    We replace ``publish()`` on the instance's publishers with a no-op.
    The publishers stay advertised in the DDS graph (they were created
    during ``Robot.__init__`` and can't be cleanly destroyed from outside
    crisp_py), but they emit nothing.

    See ``docs/ridgeback_target_pose_ownership.md`` for the full story.
    """

    def _noop(_msg):  # noqa: ANN001
        return None

    if env.robot._target_pose_publisher is not None:
        env.robot._target_pose_publisher.publish = _noop
    env.robot._target_joint_publisher.publish = _noop


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="ridgeback_mocap_recordings",
        help="Repository ID for the LeRobot dataset",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="perform task",
        help="Task description label for all episodes",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Recording frame rate (default: 20)",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=50,
        help="Number of episodes to record, 0 = unlimited (default: 50)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume recording from an existing dataset",
    )
    parser.add_argument(
        "--push-to-hub",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Push dataset to Hugging Face Hub when done",
    )
    parser.add_argument(
        "--env-config",
        type=str,
        default="ur10e_ridgeback_env",
        help="Environment config name (default: ur10e_ridgeback_env)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument(
        "--require-mocap",
        action="store_true",
        default=False,
        help="Abort on startup if no /target_pose message arrives within 5s.",
    )
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    setup_logging(level=args.log_level)

    logger.info(f"Creating environment: {args.env_config}")
    env = make_env(env_type=args.env_config, control_type="cartesian", namespace="")

    # ------------------------------------------------------------------ #
    # Mocap plumbing (applied BEFORE wait_until_ready so the subscription
    # is live as early as possible, and BEFORE the env starts publishing
    # its stale-target stream so we don't race with it).
    # ------------------------------------------------------------------ #
    logger.info("Silencing env's /target_pose and target_joint publishers — mocap owns those topics.")
    silence_env_target_publishers(env)

    mocap_capture = MocapTargetCapture()
    env.robot.node.create_subscription(
        PoseStamped,
        env.config.robot_config.target_pose_topic,
        mocap_capture.on_msg,
        qos_profile_sensor_data,
        callback_group=ReentrantCallbackGroup(),
    )
    logger.info(
        f"Subscribed to mocap target: {env.config.robot_config.target_pose_topic}"
    )

    logger.info("Waiting for robot to be ready...")
    env.wait_until_ready()
    logger.info("Robot ready.")

    if args.require_mocap:
        import time
        deadline = time.time() + 5.0
        while mocap_capture.count == 0 and time.time() < deadline:
            time.sleep(0.1)
        if mocap_capture.count == 0:
            raise TimeoutError(
                f"--require-mocap: no message on {env.config.robot_config.target_pose_topic} "
                "within 5 s. Is track_mocap.py running?"
            )
        logger.info(f"Mocap target stream live ({mocap_capture.count} msgs in 5 s).")
    else:
        if mocap_capture.count == 0:
            logger.warning(
                "No mocap /target_pose message received yet. The arm will not move "
                "until the tracker starts publishing. Pass --require-mocap to abort instead."
            )

    features = get_features(env=env)
    logger.debug(f"Features: {list(features.keys())}")

    rec_config = RecordingManagerConfig(
        features=features,
        repo_id=args.repo_id,
        robot_type="ur10e",
        fps=args.fps,
        num_episodes=args.num_episodes,
        resume=args.resume,
        push_to_hub=args.push_to_hub,
    )
    recording_manager = make_recording_manager(
        recording_manager_type="keyboard",
        config=rec_config,
    )
    recording_manager.wait_until_ready()
    logger.info("Recording manager ready.")

    logger.info("Homing robot...")
    env.home()
    env.reset()

    def data_fn():
        obs = env.get_obs()

        # Cartesian action = latest mocap /target_pose. If the tracker hasn't
        # sent anything yet (shouldn't happen after --require-mocap) we fall
        # back to the current observed pose — a static action is better than
        # a crash.
        mocap_pose = mocap_capture.get()
        if mocap_pose is not None:
            target_pose = mocap_pose.to_array(
                representation=env.config.orientation_representation
            ).astype(np.float32)
        else:
            target_pose = env.robot.current_pose.to_array(
                representation=env.config.orientation_representation
            ).astype(np.float32)

        # Gripper action = commanded target. Mocap publishes normalized
        # grip commands to /target_gripper_state, which crisp_py's Gripper
        # class mirrors into gripper._target. Before the first grip event,
        # _target is None and we fall back to the current observed value.
        if env.gripper is not None and env.gripper._target is not None:
            grip_action = float(env.gripper.target)
        elif env.gripper is not None:
            grip_action = float(env.gripper.value)
        else:
            grip_action = 0.0

        action = np.concatenate(
            [target_pose, np.array([grip_action], dtype=np.float32)]
        )
        return obs, action

    def on_start():
        env.reset()

    def on_end():
        env.robot.reset_targets()
        env.home(blocking=False)
        env.gripper.open()

    try:
        with recording_manager:
            while not recording_manager.done():
                ep_num = recording_manager.episode_count + 1
                num_ep_str = str(args.num_episodes) if args.num_episodes > 0 else "∞"
                logger.info(f"Episode {ep_num} / {num_ep_str}")
                recording_manager.record_episode(
                    data_fn=data_fn,
                    task=args.task,
                    on_start=on_start,
                    on_end=on_end,
                )

        logger.info("Recording complete. Homing robot.")
        env.home()

    except Exception:
        logger.exception("Error during recording.")
        raise
    finally:
        env.close()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
