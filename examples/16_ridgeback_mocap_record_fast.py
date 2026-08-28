#!/usr/bin/env python3
"""UR10e crisp_gym environment — mocap-driven recording with image writer (script 16).

Iterated copy of ``15_ridgeback_mocap_record.py``. The mocap flow and the
recorded fields are unchanged. The only difference is one FPS fix:

- The writer subprocess starts a LeRobot image writer immediately after
  ``LeRobotDataset.create(...)``. Without it, ``dataset.add_frame`` encodes
  camera frames to the per-episode PNG staging pool **synchronously** on the
  writer subprocess. For the Ridgeback rig (multiple Orbbec RGB streams at
  20 Hz) that synchronous encode cannot keep up, the 16-slot
  ``mp.JoinableQueue`` between the recording loop and the writer subprocess
  fills, the parent-side ``queue.put(...)`` blocks, and the recording loop
  collapses below target FPS (visible as the
  ``"Frame processing took too long"`` warning from
  ``recording_manager.py:319``).

  This script defines ``FastKeyboardRecordingManager``, a local subclass of
  ``KeyboardRecordingManager`` that copies the shared ``_create_dataset``
  logic verbatim and adds one call:

  .. code-block:: python

      dataset.start_image_writer(
          num_processes=self.config.image_writer_processes,
          num_threads=self.config.image_writer_threads,
      )

  Called inside the writer subprocess (the correct process tree for the
  LeRobot writer workers), 8 workers are enough to cover 3 Orbbec 1080p
  streams at 20 Hz with headroom. See
  ``docs/ridgeback_mocap_record_fps_image_writer.md`` for the full writeup
  (symptom, root cause, alternative FPS causes ruled out, verification plan).

Everything else — mocap plumbing, ``silence_env_target_publishers``, the
``/target_pose`` subscription, ``data_fn``, ``on_start``/``on_end``, the
``--require-mocap`` guard — is identical to script 15. Nothing in the shared
``crisp_gym/record/`` package is modified; 12/13/15/spacemouse recorders are
unaffected by this file.

Drift caveat: the ``_create_dataset`` override is a verbatim copy of the
resume branch + collision check + create call from
``crisp_gym/record/recording_manager.py:_create_dataset``. If that shared
function is upgraded later (lerobot API changes, new feature, etc.), this
override will drift and must be re-synced by hand.

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
    python examples/16_ridgeback_mocap_record_fast.py \\
        --repo-id pick_red_cube_001 \\
        --task "pick up the red cube and place it in the bowl" \\
        --num-episodes 50

    # Tune the image writer worker count if 8 is not enough:
    python examples/16_ridgeback_mocap_record_fast.py \\
        --image-writer-processes 16 --image-writer-threads 1 \\
        --repo-id pick_red_cube_001 ...

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
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import qos_profile_sensor_data

# IMPORT ORDER MATTERS. `crisp_gym.envs.manipulator_env` must be imported
# BEFORE anything that pulls in torch / torchvision / lerobot. Reason: the
# env chain imports `crisp_py.camera` which does `import cv2`, which loads
# conda's libtiff.so.6, which NEEDEDs libjpeg.so.8. If torch/torchvision has
# already been loaded into the process, a bundled libjpeg is dlopen'd first
# and shadows the env's conda libjpeg-turbo; libtiff's relocation for
# `jpeg12_write_raw_data@@LIBJPEG_8.0` then binds against the bundled copy
# (which doesn't export that symbol) and ImportError fires. Loading cv2
# first pins libjpeg.so.8 to the env's copy in the loader's tables before
# any other libjpeg can shadow it. Script 15 works by accident because of
# this ordering; keep it explicit here so reorderings don't resurrect the
# crash.
from crisp_gym.envs.manipulator_env import make_env

# Safe to pull in lerobot-touching modules now that cv2 is bound. We
# re-export HF_LEROBOT_HOME and LeRobotDataset from recording_manager's
# namespace rather than importing them from lerobot directly, so we avoid
# duplicating the lerobot-version shim and there is only one module in the
# tree that knows where those symbols live.
from crisp_gym.record.recording_manager import (
    HF_LEROBOT_HOME,
    KeyboardRecordingManager,
    LeRobotDataset,
)
from crisp_gym.record.recording_manager_config import RecordingManagerConfig
from crisp_gym.util.lerobot_features import get_features
from crisp_gym.util.setup_logger import setup_logging
from crisp_py.utils.geometry import Pose

logger = logging.getLogger(__name__)


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

    See ``15_ridgeback_mocap_record.py`` for the full explanation — this is a
    verbatim copy so that 16 behaves identically to 15 w.r.t. topic ownership.
    """

    def _noop(_msg):  # noqa: ANN001
        return None

    if env.robot._target_pose_publisher is not None:
        env.robot._target_pose_publisher.publish = _noop
    env.robot._target_joint_publisher.publish = _noop


class FastKeyboardRecordingManager(KeyboardRecordingManager):
    """Keyboard recording manager with the LeRobot image writer enabled.

    Overrides ``_create_dataset`` with a verbatim copy of the shared
    implementation (``crisp_gym/record/recording_manager.py:130``) plus a
    ``dataset.start_image_writer(...)`` call right after
    ``LeRobotDataset.create(...)``. The call is inside ``_create_dataset``,
    which is invoked from ``_writer_proc``, so the writer workers spawn as
    children of the correct mp.Process subprocess.

    The image-writer config is read via ``getattr`` from
    ``self.config`` so this override works without modifying the shared
    ``RecordingManagerConfig`` dataclass. The main() function attaches
    ``image_writer_processes`` and ``image_writer_threads`` to the config
    instance at runtime before handing it to the manager.
    """

    def _create_dataset(self) -> LeRobotDataset:
        logger.debug("Creating dataset object.")
        if self.config.resume:
            logger.info(f"Resuming recording from existing dataset: {self.config.repo_id}")
            dataset = LeRobotDataset(repo_id=self.config.repo_id)
            if self.config.num_episodes <= dataset.num_episodes:
                logger.error(
                    f"The dataset already has {dataset.num_episodes} recorded. "
                    "Please select a larger number."
                )
                exit()
            logger.info(
                f"Resuming from episode {dataset.num_episodes} with "
                f"{self.config.num_episodes} episodes to record."
            )
            self.episode_count_queue.put(dataset.num_episodes - 1)
        else:
            logger.info(
                f"[green]Creating new dataset: {self.config.repo_id}",
                extra={"markup": True},
            )
            if Path(HF_LEROBOT_HOME / self.config.repo_id).exists():
                msg = (
                    f"The repo_id already exists. If you intended to resume the "
                    f"collection of data, then execute this script with the "
                    f"--resume flag. Otherwise remove it:\n"
                    f"'rm -r {str(Path(HF_LEROBOT_HOME / self.config.repo_id))}'."
                )
                logger.error(msg)
                raise FileExistsError(msg)
            dataset = LeRobotDataset.create(
                repo_id=self.config.repo_id,
                fps=self.config.fps,
                robot_type=self.config.robot_type,
                features=self.config.features,
                use_videos=True,
                image_writer_threads=1,
                image_writer_processes=16
            )
            logger.info("Dataset created, starting image writer...")
            dataset.start_image_writer(num_processes=8, num_threads=1)
            logger.info("Image writer started inside subprocess!")
            logger.debug(f"Dataset created with meta: {dataset.meta}")

            # FPS fix: start the LeRobot image writer inside the writer
            # subprocess. Without this, add_frame encodes camera frames
            # synchronously here and the 16-slot mp queue fills, starving the
            # parent recording loop. See
            # docs/ridgeback_mocap_record_fps_image_writer.md.
            num_processes = getattr(self.config, "image_writer_processes", 0)
            num_threads = getattr(self.config, "image_writer_threads", 1)
            if num_processes > 0:
                logger.info(
                    f"Starting image writer: processes={num_processes}, "
                    f"threads={num_threads}"
                )
                dataset.start_image_writer(
                    num_processes=num_processes,
                    num_threads=num_threads,
                )
                logger.info("Image writer started inside writer subprocess.")

            logger.debug(f"Dataset created with meta: {dataset.meta}")
        return dataset


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
    parser.add_argument(
        "--image-writer-processes",
        type=int,
        default=8,
        help="LeRobot image writer worker process count. 0 disables the writer "
             "(synchronous encode — causes FPS drops on multi-camera rigs). "
             "Default: 8 — enough for 3 Orbbec 1080p streams at 20 Hz.",
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=1,
        help="LeRobot image writer threads per worker process. "
             "Default: 1 (PIL PNG encode is single-threaded; threads > 1 "
             "yields no benefit).",
    )
    args = parser.parse_args()

    setup_logging(level=args.log_level)

    logger.info(f"Creating environment: {args.env_config}")
    env = make_env(env_type=args.env_config, control_type="cartesian", namespace="")

    # ------------------------------------------------------------------ #
    # Mocap plumbing (applied BEFORE wait_until_ready so the subscription
    # is live as early as possible, and BEFORE the env starts publishing
    # its stale-target stream so we don't race with it).
    # ------------------------------------------------------------------ #
    logger.info(
        "Silencing env's /target_pose and target_joint publishers — mocap owns those topics."
    )
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

    # Attach image-writer config at runtime. RecordingManagerConfig is a
    # non-frozen dataclass, so new attributes stick. FastKeyboardRecordingManager
    # reads them via getattr with defaults, so this stays backwards compatible
    # with the shared RecordingManagerConfig definition.
    rec_config.image_writer_processes = args.image_writer_processes
    rec_config.image_writer_threads = args.image_writer_threads

    recording_manager = FastKeyboardRecordingManager(config=rec_config)
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
