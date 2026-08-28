#!/usr/bin/env python3
"""UR10e crisp_gym environment — data recording example.

Records robot demonstrations in LeRobot format. The robot is controlled
externally (spacemouse, mocap, manual jogging, etc.) — this script captures
observations and the current target pose as the action for each frame.

Usage:
    python examples/12_ridgeback_record.py --repo-id my_dataset
    python examples/12_ridgeback_record.py --repo-id my_dataset --resume
    python examples/12_ridgeback_record.py --repo-id my_dataset --num-episodes 50

Keyboard controls (episode management):
    r  →  start recording
    r  →  stop / pause recording
    s  →  save episode
    d  →  discard episode
    q  →  quit

Prerequisites:
    - Controllers loaded and running on the real robot
    - cartesian_controller activated
    - joint_state_broadcaster and pose_broadcaster active
    - Camera running (pixi run orbbec in clearpath_remote_ws)
"""

import argparse
import logging

import numpy as np
import rclpy

from crisp_gym.envs.manipulator_env import make_env
from crisp_gym.record.recording_manager import make_recording_manager
from crisp_gym.record.recording_manager_config import RecordingManagerConfig
from crisp_gym.util.lerobot_features import get_features
from crisp_gym.util.setup_logger import setup_logging


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="ridgeback_recordings",
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
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    setup_logging(level=args.log_level)

    logger.info(f"Creating environment: {args.env_config}")
    env = make_env(env_type=args.env_config, control_type="cartesian", namespace="")

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

        # Action = what a policy must reproduce:
        #   - Cartesian: commanded target_pose (from mocap tracker via /target_pose)
        #   - Gripper:   commanded target      (from mocap tracker via /target_gripper_state)
        # Fall back to current gripper reading only before the tracker has
        # issued its first grip command.
        target_pose = env.robot.target_pose.to_array(
            representation=env.config.orientation_representation
        ).astype(np.float32)

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
