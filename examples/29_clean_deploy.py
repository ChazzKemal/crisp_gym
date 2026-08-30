#!/usr/bin/env python3
r"""Minimal LeRobot policy deployer on a CRISP robot.

What this is:
  - One loop. Read observation → policy.select_action → env.step.
  - Uses only the standard crisp_gym ManipulatorEnv API (env.get_obs,
    env.step) so cameras, gripper, and pose targets all flow through the
    same channels crisp_py already provides.
  - Uses LeRobot's stock single-step inference API (`policy.select_action`),
    which handles ACT and diffusion uniformly — for diffusion the policy
    manages its `_queues` internally, so we don't touch them.

What this is NOT:
  - No xVLA speedup pipeline (no per-frame speed schedule, no cycle-snap,
    no bounded queue, no sender thread). See examples/19_deploy_policy.py
    for that path.
  - No shared-memory camera bridge, no direct gripper-action publish, no
    multiprocessing inference subprocess, no recording manager.
  - No data recording. If you want to record while deploying, use
    `crisp-deploy-policy` (crisp_gym/scripts/deploy_policy.py) instead.

Usage:
    cd Yunfei/crisp_gym
    pixi run -e jazzy-lerobot python examples/29_clean_deploy.py \\
        --env-config ur10e_ridgeback_dual_cam_deploy_env \\
        --pretrained-path outputs/train/diffusion_pickplace_cart7_v2/035000/pretrained_model \\
        --fps 20 \\
        --num-inference-steps 10 --noise-scheduler-type DDIM

Prerequisites:
    - Robot up, controller_manager running, cartesian controller loaded.
    - A LeRobot checkpoint at --pretrained-path whose `output_features.action`
      shape matches the env's action_space (typically 7 for cartesian:
      x,y,z, 3-DoF rotation, gripper).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

# crisp_gym must be imported BEFORE torch / lerobot. The crisp_gym import
# chain pulls in rclpy → cv2, and cv2's bundled libtiff requires the system
# libjpeg's `jpeg12_write_raw_data` symbol. When torch is imported first it
# loads a different libjpeg into the process and cv2 then fails with
# "undefined symbol: jpeg12_write_raw_data". Importing rclpy/cv2 first via
# crisp_gym pins the correct libjpeg early.
import numpy as np

from crisp_gym.envs.manipulator_env import make_env
from crisp_gym.util.lerobot_features import concatenate_state_features, numpy_obs_to_torch
from crisp_gym.util.setup_logger import setup_logging

import torch  # noqa: E402  (after crisp_gym on purpose — see comment above)
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.policies.factory import get_policy_class, make_pre_post_processors  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Policy load
# ---------------------------------------------------------------------------

def load_policy(
    pretrained_path: str,
    *,
    num_inference_steps: int | None = None,
    noise_scheduler_type: str | None = None,
):
    """Load a LeRobot policy + its pre/post processors from a local checkpoint.

    Applies the optional diffusion overrides BEFORE policy construction —
    DiffusionPolicy builds its noise scheduler in __init__, so a later
    mutation of `config.num_inference_steps` would be silently ignored.
    The overrides are `hasattr`-guarded so non-diffusion configs (e.g. ACT)
    never gain spurious attributes.

    Returns:
        (policy, preprocessor, postprocessor, config)
    """
    cfg = PreTrainedConfig.from_pretrained(pretrained_path)
    if cfg is None:
        raise ValueError(f"Missing policy config at {pretrained_path}")

    if num_inference_steps is not None and hasattr(cfg, "num_inference_steps"):
        logger.info("override num_inference_steps: %s -> %d",
                    cfg.num_inference_steps, num_inference_steps)
        cfg.num_inference_steps = int(num_inference_steps)
    if noise_scheduler_type is not None and hasattr(cfg, "noise_scheduler_type"):
        logger.info("override noise_scheduler_type: %s -> %s",
                    cfg.noise_scheduler_type, noise_scheduler_type)
        cfg.noise_scheduler_type = str(noise_scheduler_type)

    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(pretrained_path, config=cfg)
    policy.reset()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device).eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=pretrained_path,
    )

    logger.info(
        "loaded %s policy (type=%s) on %s. n_obs_steps=%s n_action_steps=%s",
        policy.name, cfg.type, device,
        getattr(cfg, "n_obs_steps", "?"), getattr(cfg, "n_action_steps", "?"),
    )
    return policy, preprocessor, postprocessor, cfg


# ---------------------------------------------------------------------------
# Observation assembly
# ---------------------------------------------------------------------------

def build_state(obs: dict, state_subkeys: list[str]) -> np.ndarray:
    """Build the flat `observation.state` the policy was trained on.

    Concatenates only the sub-keys the policy declares (in its declared
    order). If the policy declares only a flat `observation.state`, falls
    back to concatenating every `observation.state.*` the env emits. The
    caller is expected to assert the resulting dim matches the policy's
    declared input dim before inference.
    """
    if state_subkeys:
        return np.concatenate(
            [np.asarray(obs[k], dtype=np.float32).reshape(-1) for k in state_subkeys]
        )
    return concatenate_state_features(obs)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def deploy(
    env,
    policy,
    preprocessor,
    postprocessor,
    *,
    fps: float,
    max_steps: int,
    task: str,
    dry_run: bool,
) -> int:
    """Run the policy in a tight obs→action loop until max_steps or Ctrl-C."""
    period = 1.0 / max(fps, 1e-9)

    # State sub-keys the policy declares (authoritative order).
    state_subkeys = [
        k for k in policy.config.input_features
        if k.startswith("observation.state.")
    ]
    state_feat = policy.config.input_features.get("observation.state")
    expected_state_dim = int(state_feat.shape[0]) if state_feat is not None else None

    logger.info(
        "loop config: fps=%.1f  period=%.1fms  max_steps=%s  dry_run=%s",
        fps, period * 1000.0,
        "unbounded" if max_steps <= 0 else max_steps, dry_run,
    )
    logger.info("state_subkeys=%s  expected_state_dim=%s",
                state_subkeys or "<all observation.state.*>", expected_state_dim)
    if dry_run:
        logger.warning("--dry-run: actions WILL be computed but env.step is SKIPPED")

    step = 0
    inf_ms: list[float] = []
    late_ticks = 0

    t_next = time.monotonic()
    try:
        while max_steps <= 0 or step < max_steps:
            t_loop_start = time.monotonic()

            # 1. Observation
            obs = env.get_obs()
            obs["observation.state"] = build_state(obs, state_subkeys)
            if expected_state_dim is not None and obs["observation.state"].shape[0] != expected_state_dim:
                raise ValueError(
                    f"state dim mismatch: env produced "
                    f"{obs['observation.state'].shape[0]}, policy expects "
                    f"{expected_state_dim} (from subkeys {state_subkeys}). "
                    f"Check env config's `observations_to_include_to_state`."
                )
            if task:
                obs["task"] = task

            # 2. Inference (LeRobot's single-step API; handles ACT and
            #    diffusion uniformly via the policy's internal queues).
            t_inf = time.monotonic()
            with torch.inference_mode():
                batch = numpy_obs_to_torch(obs)
                batch = preprocessor(batch)
                action_t = policy.select_action(batch)
                action_t = postprocessor(action_t)
            action = action_t.squeeze(0).to("cpu").numpy()
            inf_ms.append((time.monotonic() - t_inf) * 1000.0)

            # 3. Apply (ManipulatorCartesianEnv.step publishes /target_pose
            #    and the gripper command through crisp_py's standard path).
            if not dry_run:
                env.step(action, block=False)

            # Diagnostic action log — every step at DEBUG, every 10th at INFO.
            # Prints predicted action AND the absolute delta from the current
            # EE state. If delta is consistently near-zero, the policy is in
            # a "stay here" attractor (home/state mismatch with training).
            current_state = obs["observation.state"]
            if expected_state_dim is not None and current_state.shape[0] == 7 and action.shape[0] == 7:
                delta = action[:6] - current_state[:6]
                if step % 10 == 0:
                    logger.info(
                        "step %d  action[xyz]=(% .3f,% .3f,% .3f) action[rpy]=(% .2f,% .2f,% .2f) gripper=% .3f "
                        "| state[xyz]=(% .3f,% .3f,% .3f) | |Δxyz|=%.3f mm |Δrpy|=%.2f rad",
                        step, *action[:6], float(action[6]),
                        *current_state[:3],
                        float(np.linalg.norm(delta[:3])) * 1000.0,
                        float(np.linalg.norm(delta[3:6])),
                    )
                else:
                    logger.debug(
                        "step %d  action=%s  state[xyz]=%s",
                        step, np.round(action, 3).tolist(),
                        np.round(current_state[:3], 3).tolist(),
                    )

            step += 1

            # 4. Rate control (sleep until next deadline; resync if behind).
            t_next += period
            sleep_for = t_next - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                late_ticks += 1
                t_next = time.monotonic()  # drop missed ticks rather than catch up

            # Periodic stats
            if step % max(1, int(fps) * 5) == 0:  # ~every 5 s
                arr = np.asarray(inf_ms[-int(fps) * 5:])
                logger.info(
                    "step %d  inf median=%.1f ms  p95=%.1f ms  max=%.1f ms  "
                    "loop_jitter=%.1f ms  late_ticks=%d",
                    step,
                    float(np.median(arr)), float(np.percentile(arr, 95)), float(arr.max()),
                    (time.monotonic() - t_loop_start) * 1000.0 - period * 1000.0,
                    late_ticks,
                )

    except KeyboardInterrupt:
        logger.info("stopped by Ctrl-C at step %d", step)

    if inf_ms:
        arr = np.asarray(inf_ms)
        logger.info(
            "summary: %d steps  inf median=%.1f ms  p95=%.1f ms  max=%.1f ms  "
            "late_ticks=%d",
            step, float(np.median(arr)), float(np.percentile(arr, 95)),
            float(arr.max()), late_ticks,
        )
    return step


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--env-config", required=True,
        help="ManipulatorEnv config name (e.g. ur10e_ridgeback_dual_cam_deploy_env).",
    )
    parser.add_argument(
        "--pretrained-path", required=True,
        help="Path to the LeRobot pretrained_model directory.",
    )
    parser.add_argument(
        "--namespace", default="",
        help="ROS namespace for the robot (default: empty).",
    )
    parser.add_argument(
        "--joint-control", action="store_true",
        help="Use joint-space control. Default is cartesian.",
    )
    parser.add_argument(
        "--fps", type=float, default=20.0,
        help="Control loop rate (default: 20 Hz).",
    )
    parser.add_argument(
        "--max-steps", type=int, default=0,
        help="Stop after this many steps. 0 = unbounded (Ctrl-C to stop).",
    )
    parser.add_argument(
        "--task", default="",
        help="Task string passed to the policy (only used by VLA-style policies).",
    )
    parser.add_argument(
        "--num-inference-steps", type=int, default=None,
        help="Diffusion-only: override denoising-loop length. None = checkpoint default.",
    )
    parser.add_argument(
        "--noise-scheduler-type", default=None, choices=["DDPM", "DDIM"],
        help="Diffusion-only: override noise scheduler. None = checkpoint default. "
             "DDIM allows aggressive --num-inference-steps reductions (10 or 5).",
    )
    parser.add_argument(
        "--no-home", action="store_true",
        help="Skip env.home() at startup and shutdown. Default is to home both times.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Compute actions but skip env.step. Robot does not move; useful for "
             "validating the inference path + measuring per-step latency.",
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="Skip the [y/N] confirmation prompt.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    args = parser.parse_args()

    setup_logging(level=args.log_level)

    if not args.dry_run and not args.yes:
        try:
            ans = input("Deploy policy — the arm WILL move. Continue? [y/N] ").strip().lower()
        except EOFError:
            ans = "n"
        if ans not in ("y", "yes"):
            logger.info("Aborted by user.")
            return 1

    ctrl_type = "joint" if args.joint_control else "cartesian"
    logger.info("Creating env: %s (control=%s, namespace=%r)",
                args.env_config, ctrl_type, args.namespace)
    env = make_env(args.env_config, control_type=ctrl_type, namespace=args.namespace)

    logger.info("Waiting for env to be ready...")
    env.wait_until_ready()
    logger.info("Env ready. Action space: %s", env.action_space)

    policy = preprocessor = postprocessor = None
    try:
        policy, preprocessor, postprocessor, _ = load_policy(
            args.pretrained_path,
            num_inference_steps=args.num_inference_steps,
            noise_scheduler_type=args.noise_scheduler_type,
        )

        if not args.no_home:
            logger.info("Homing robot before deploy...")
            env.home()

        # Switch to the deploy-time controller. CRITICAL: env.home(blocking=True)
        # leaves the joint_trajectory_controller active and DOES NOT auto-switch
        # back (manipulator_env.py:home() only auto-switches when blocking=False).
        # Without this, /target_pose has no active subscriber and env.step()
        # publishes into the void — robot stays put even though the policy is
        # producing valid actions. Always run the switch, even with --no-home,
        # in case a prior invocation left JTC active. Mirrors 19_deploy_policy.py's
        # "Phase 2: switching to cartesian controller".
        ctrl_for_deploy = "joint" if args.joint_control else "cartesian"
        logger.info("Switching to %s controller for deploy...", ctrl_for_deploy)
        env.switch_controller(ctrl_for_deploy)

        steps = deploy(
            env, policy, preprocessor, postprocessor,
            fps=args.fps,
            max_steps=args.max_steps,
            task=args.task,
            dry_run=args.dry_run,
        )
        logger.info("Deploy finished after %d step(s).", steps)

    finally:
        if not args.no_home:
            try:
                logger.info("Homing robot before shutdown...")
                env.home(blocking=False)
            except Exception as e:  # noqa: BLE001
                logger.warning("home() failed: %s", e)
        try:
            env.close()
        except Exception as e:  # noqa: BLE001
            logger.warning("env.close() failed: %s", e)

    return 0


if __name__ == "__main__":
    sys.exit(main())
