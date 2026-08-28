#!/usr/bin/env python3
"""UR10e crisp_gym environment — minimal test & mocap recording template.

Three modes of operation:
  1. Read-only:  just print robot state (default)
  2. Home:       move to home config, then print state
  3. Teleop:     mocap-based teleoperation with data collection

Usage:
    # 1. Read robot state
    python examples/11_ur10e_env_test.py

    # 2. Home then read state
    python examples/11_ur10e_env_test.py --home

    # 3. Verify env creation only (no robot needed)
    python examples/11_ur10e_env_test.py --dry-run

Prerequisites:
    - Controllers loaded and running on the real robot
    - cartesian_controller activated (for Cartesian env)
    - pose_broadcaster and joint_state_broadcaster active
"""

import argparse
import sys

import numpy as np


def format_gripper_target(env) -> str:
    """Return the normalized commanded gripper target, or explain why it is unavailable."""
    if getattr(env, "gripper", None) is None:
        return "none"
    if getattr(env.gripper, "_target", None) is None:
        return "unset"
    try:
        return f"{float(env.gripper.target):.4f}"
    except Exception as exc:
        return f"unavailable ({exc})"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--home",
        action="store_true",
        help="Move robot to home config before reading state",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only create the env config, don't connect to robot",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of steps to run in the observation loop (default 100)",
    )
    args = parser.parse_args()

    # --- 1. Create environment config ---
    from crisp_gym.envs.manipulator_env_config import make_env_config

    print("Loading ur10e_ridgeback_env config...")
    config = make_env_config("ur10e_ridgeback_env")

    print(f"  Robot type: {config.robot_config.__class__.__name__}")
    print(f"  Joints: {config.robot_config.joint_names}")
    print(f"  Base frame: {config.robot_config.base_frame}")
    print(f"  EE frame: {config.robot_config.target_frame}")
    print(f"  Controller: {config.robot_config.cartesian_impedance_controller_name}")
    print(f"  Control freq: {config.control_frequency} Hz")
    print(f"  Relative actions: {config.use_relative_actions}")
    print(f"  Gripper mode: {config.gripper_mode}")
    print(f"  Cameras: {[c.camera_name for c in config.camera_configs]}")
    print(f"  Observations: {config.observations_to_include_to_state}")

    if args.dry_run:
        print("\n[DRY RUN] Config created successfully. Exiting.")
        return

    # --- 2. Create Gymnasium environment ---
    from crisp_gym.envs.manipulator_env import ManipulatorCartesianEnv

    print("\nCreating ManipulatorCartesianEnv...")
    env = ManipulatorCartesianEnv(config=config, namespace="")

    print("Waiting for robot to be ready...")
    env.wait_until_ready()
    print("Robot ready!")

    # --- 3. Optional homing ---
    if args.home:
        print("\nHoming robot...")
        env.home()
        print("Homing complete.")

    # --- 4. Reset and read observations ---
    print("\nResetting environment...")
    obs, info = env.reset()

    print("\n=== Initial Observation ===")
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: shape={value.shape}, values={np.array2string(value, precision=4)}")
        else:
            print(f"  {key}: {value}")
    print(f"  gripper target: {format_gripper_target(env)}")

    # --- 5. Read loop: print state for N steps ---
    print(f"\nRunning {args.steps} read-only steps at {config.control_frequency} Hz...\n")

    for step in range(args.steps):
        # No-op action: hold current pose (absolute mode, so we re-send current target)
        obs = env.get_obs()

        if step % 20 == 0:
            cart = obs.get("observation.state.cartesian", None)
            joints = obs.get("observation.state.joints", None)
            target = obs.get("observation.state.target", None)
            gripper = obs.get("observation.state.gripper", None)
            gripper_target = format_gripper_target(env)

            print(f"[Step {step:4d}]")
            if cart is not None:
                print(f"  EE pose:    {np.array2string(cart, precision=4)}")
            if joints is not None:
                print(f"  Joints:     {np.array2string(joints, precision=4)}")
            if target is not None:
                print(f"  Target:     {np.array2string(target, precision=4)}")
            if gripper is not None:
                print(f"  Gripper:    {gripper}")
            print(f"  Grip target:{gripper_target}")

       
        env.control_rate.sleep()

    print("\nDone. Closing environment...")
    env.close()


if __name__ == "__main__":
    main()
