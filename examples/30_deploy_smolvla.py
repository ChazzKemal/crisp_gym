#!/usr/bin/env python3
"""Deploy SmolVLA + Real-Time Chunking on the UR10e.

Sister script to ``19_deploy_policy.py``. Where 19 uses the pinned upstream
lerobot 0.4.2 (no RTC) under env ``jazzy-lerobot``, this script targets the
local ``speedup_lerobot`` fork (lerobot 0.4.3, ``lerobot.policies.smolvla`` +
``lerobot.policies.rtc``) under env ``jazzy-lerobot-rtc``.

What's different from 19:

* The policy is always SmolVLA. Language instruction (``task``) is required
  (resolved from CLI / checkpoint metadata / training-dataset features).
* Replay machinery is reused: ``TargetSenderThread`` consumes ``TargetItem``s
  from a bounded queue at cycle-snapped deadlines. The scaler is optional.
* Inference runs in a background ``InferenceDispatcherThread`` that watches
  the RTC ``ActionQueue``: when the queue drops to <= ``queue_refill_threshold``
  remaining actions, the dispatcher fires the next inference asynchronously,
  passing the unconsumed prefix to SmolVLA so it inpaints a smooth handover.
* The main producer loop pops one action at a time from the RTC queue, cycle-
  snaps it for one control tick, and pushes a single TargetItem.

Usage::

    cd Yunfei/crisp_gym
    pixi run -e jazzy-lerobot-rtc python examples/30_deploy_smolvla.py \\
        --pretrained-path /path/to/smolvla/checkpoint \\
        --policy-config 2_smolvla_rtc_config \\
        --task "Pick up the lego block." \\
        --fps 20

Prerequisites:
    - Robot up, controller_manager running, cartesian controller available.
    - A SmolVLA checkpoint trained with this repo's 7-dim action layout
      (xyz + rotvec + gripper).
    - speedup_lerobot installed editable via pixi feature ``lerobot-rtc``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import queue
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32

from crisp_gym.config.path import find_config
from crisp_gym.envs.manipulator_env import make_env
from crisp_gym.policy.async_smolvla_policy import AsyncSmolVLAPolicy
from crisp_gym.util.setup_logger import setup_logging


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reuse helpers from 17_replay_dataset.py and 19_deploy_policy.py via importlib
# (leading-digit filenames are not valid Python module names).
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent


def _load_neighbour(name: str, module_name: str):
    path = _HERE / name
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to locate {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_replay17 = _load_neighbour("17_replay_dataset.py", "replay17")
_deploy19 = _load_neighbour("19_deploy_policy.py", "deploy19")

CONTROL_DT = _replay17.CONTROL_DT
DEFAULT_GRIPPER_SPEED = _replay17.DEFAULT_GRIPPER_SPEED
TargetItem = _replay17.TargetItem
TargetSenderThread = _replay17.TargetSenderThread
ReplayScaler = _replay17.ReplayScaler
build_speed_queue_arrays = _replay17.build_speed_queue_arrays
enable_target_pose_publishing = _replay17.enable_target_pose_publishing
fix_gripper_self_subscription = _replay17.fix_gripper_self_subscription

_build_obs_schema = _deploy19._build_obs_schema
_get_obs_zerofill = _deploy19._get_obs_zerofill
_pre_compute_chunk_arrays = _deploy19._pre_compute_chunk_arrays


# ---------------------------------------------------------------------------
# Task resolution
# ---------------------------------------------------------------------------


def _resolve_task(pretrained_path: Path, cli_task: str | None) -> str:
    """Pick a language instruction.

    Order of precedence:
      1. ``--task`` CLI flag if non-empty.
      2. SmolVLA checkpoint metadata (``train_config.json`` /
         ``policy_config.json`` ``task`` field).
      3. Training-dataset features (``LeRobotDataset(repo_id).meta.tasks``):
         unique task → that one; multiple → abort with the list.
      4. Final fallback: abort with a clear error.
    """
    if cli_task and cli_task.strip():
        logger.info("Task: %r (from --task)", cli_task)
        return cli_task

    # 2. Checkpoint metadata. We do NOT parse train_config.json with draccus
    #    (forked variants can break strict parsing); just json.load and look
    #    for likely fields.
    for fname in ("train_config.json", "policy_config.json"):
        path = pretrained_path / fname
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            logger.debug("could not parse %s; skipping", path)
            continue
        for key in ("task", "task_description", "language_instruction"):
            v = data.get(key)
            if isinstance(v, str) and v.strip():
                logger.info("Task: %r (from %s.%s)", v, fname, key)
                return v
        repo_id = data.get("dataset", {}).get("repo_id") if isinstance(data.get("dataset"), dict) else None
        if not repo_id:
            repo_id = data.get("repo_id")
        if isinstance(repo_id, str):
            try:
                from lerobot.datasets.lerobot_dataset import LeRobotDataset

                ds = LeRobotDataset(repo_id)
                tasks = list(getattr(ds.meta, "tasks", []) or [])
            except Exception:
                logger.debug("LeRobotDataset(%r) lookup failed", repo_id)
                tasks = []
            if len(tasks) == 1:
                logger.info("Task: %r (from dataset %s)", tasks[0], repo_id)
                return tasks[0]
            if len(tasks) > 1:
                raise SystemExit(
                    f"--task not given and the training dataset {repo_id} has "
                    f"multiple tasks: {tasks}. Pass --task <one of these>."
                )

    raise SystemExit(
        "Could not resolve a language instruction from the checkpoint or its "
        "dataset metadata. Pass --task \"<your instruction>\" on the CLI."
    )


# ---------------------------------------------------------------------------
# RTC chunk source
# ---------------------------------------------------------------------------


class _LatencyTracker:
    """Bounded rolling max of inference latencies (seconds)."""

    def __init__(self, maxlen: int = 16):
        self._samples: deque[float] = deque(maxlen=maxlen)

    def add(self, value: float) -> None:
        self._samples.append(float(value))

    def max(self) -> float:
        if not self._samples:
            return 0.0
        return max(self._samples)


class _SmolVLARTCChunkSource:
    """Owns the SmolVLA inference subprocess + RTC ActionQueue + dispatcher.

    Exposes :meth:`pop_action` to the producer loop (returns one action
    tensor or ``None`` when no action is ready) and :meth:`shutdown`.
    """

    def __init__(
        self,
        *,
        pretrained_path: str,
        env,
        task: str,
        rtc_cfg_dict: dict,
        queue_refill_threshold: int,
        num_inference_steps: int | None,
        allow_empty_cameras: int | None,
        dt_base: float,
        obs_buf: deque,
        obs_schema: dict,
        last_obs_holder: list,
        action_dim: int,
    ):
        from lerobot.policies.rtc.action_queue import ActionQueue
        from lerobot.policies.rtc.configuration_rtc import RTCConfig
        from lerobot.configs.types import RTCAttentionSchedule

        rtc_kwargs = dict(rtc_cfg_dict)
        sched = rtc_kwargs.get("prefix_attention_schedule")
        if isinstance(sched, str):
            rtc_kwargs["prefix_attention_schedule"] = RTCAttentionSchedule[sched.upper()]
        self._rtc_cfg = RTCConfig(**rtc_kwargs)
        self._action_queue = ActionQueue(self._rtc_cfg)

        self._policy = AsyncSmolVLAPolicy(
            pretrained_path=pretrained_path,
            env=env,
            rtc=rtc_cfg_dict,
            queue_refill_threshold=queue_refill_threshold,
            num_inference_steps=num_inference_steps,
            allow_empty_cameras=allow_empty_cameras,
        )
        self.n_obs = self._policy.n_obs
        self.n_act = self._policy.n_act
        self.action_dim = action_dim or self._policy.action_dim
        self._env = env
        self._task = task
        self._dt_base = float(dt_base)
        self._queue_refill_threshold = int(queue_refill_threshold)
        self._obs_buf = obs_buf
        self._obs_schema = obs_schema
        self._last_obs_holder = last_obs_holder
        self._latency_tracker = _LatencyTracker(maxlen=16)
        self._stop = threading.Event()
        self._dispatcher: threading.Thread | None = None

        # Per-chunk telemetry (read by main thread for logging).
        self.chunk_count = 0
        self.last_real_delay = 0
        self.last_latency_s = 0.0

    # --- queue surface --------------------------------------------------

    def qsize(self) -> int:
        return self._action_queue.qsize()

    def pop_action(self) -> np.ndarray | None:
        action = self._action_queue.get()
        if action is None:
            return None
        return action.cpu().numpy() if hasattr(action, "cpu") else np.asarray(action)

    # --- lifecycle ------------------------------------------------------

    def bootstrap(self) -> None:
        """Fire the first inference synchronously so the queue is non-empty
        before the dispatcher and the main loop start."""
        logger.info("Bootstrapping first SmolVLA chunk (synchronous)...")
        self._run_one_inference()
        logger.info(
            "First chunk ready (qsize=%d, latency=%.0fms)",
            self.qsize(), self.last_latency_s * 1000.0,
        )

    def start_dispatcher(self) -> None:
        if self._dispatcher is not None:
            return
        self._dispatcher = threading.Thread(
            target=self._dispatcher_loop, name="SmolVLA-RTC-Dispatcher", daemon=True,
        )
        self._dispatcher.start()
        logger.info("Started RTC inference dispatcher thread.")

    def shutdown(self) -> None:
        self._stop.set()
        if self._dispatcher is not None:
            self._dispatcher.join(timeout=5.0)
        self._policy.shutdown()

    # --- internals ------------------------------------------------------

    def _dispatcher_loop(self) -> None:
        while not self._stop.is_set():
            if self._action_queue.qsize() > self._queue_refill_threshold:
                # Sleep a fraction of one action's duration so we re-check
                # promptly without burning CPU.
                self._stop.wait(timeout=max(self._dt_base / 4.0, 0.005))
                continue
            try:
                self._run_one_inference()
            except Exception:
                logger.exception("RTC dispatcher: inference failed")
                # Back off so a tight error loop doesn't spin.
                self._stop.wait(timeout=0.5)

    def _run_one_inference(self) -> None:
        action_index_before = self._action_queue.get_action_index()
        prev = self._action_queue.get_left_over()
        prev_np: np.ndarray | None = None
        if prev is not None:
            prev_np = prev.cpu().numpy() if hasattr(prev, "cpu") else np.asarray(prev)

        # Refresh the obs buffer right before the request: the env's most
        # recent frame is the most informative for this inference.
        self._obs_buf.append(
            _get_obs_zerofill(self._env, self._obs_schema, self._last_obs_holder)
        )

        # Estimate delay (in action steps) from past latencies. First chunk:
        # 0 → ignored by the RTC processor anyway (no prev_chunk_left_over).
        est_latency_s = self._latency_tracker.max()
        inference_delay_hint = max(0, math.ceil(est_latency_s / self._dt_base))

        resp = self._policy.request(
            obs_seq=list(self._obs_buf),
            task=self._task,
            prev_chunk_left_over=prev_np,
            action_index_before_inference=action_index_before,
            inference_delay_hint=inference_delay_hint,
        )

        latency_s = float(resp["latency_s"])
        real_delay = max(0, math.ceil(latency_s / self._dt_base))
        self._latency_tracker.add(latency_s)
        self.last_latency_s = latency_s
        self.last_real_delay = real_delay
        self.chunk_count += 1

        import torch

        original = torch.from_numpy(np.asarray(resp["original"]))
        processed = torch.from_numpy(np.asarray(resp["processed"]))
        # Guard: ActionQueue.merge requires real_delay strictly less than the
        # chunk length on the RTC (replace) path; clip so a slow inference
        # doesn't trip the assertion.
        capped_delay = min(real_delay, max(0, processed.shape[0] - 1))
        if capped_delay != real_delay:
            logger.warning(
                "[Dispatcher] real_delay %d >= chunk_len %d; clipping to %d. "
                "The RTC overlap window is too short for the measured "
                "latency — consider lowering execution_horizon or "
                "increasing chunk_size at training time.",
                real_delay, processed.shape[0], capped_delay,
            )
        self._action_queue.merge(
            original, processed, capped_delay, action_index_before,
        )
        logger.info(
            "[Dispatcher] chunk %d: latency=%.0fms real_delay=%d qsize=%d "
            "(threshold %d)",
            self.chunk_count, latency_s * 1000.0, capped_delay,
            self._action_queue.qsize(), self._queue_refill_threshold,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pretrained-path", required=True,
        help="Path to the SmolVLA pretrained checkpoint directory.",
    )
    parser.add_argument(
        "--policy-config", default="2_smolvla_rtc_config",
        help="Name of the deploy-side policy YAML under crisp_gym/config/policy/. "
             "Provides the RTC knobs (execution_horizon, inference_delay, …) "
             "and queue_refill_threshold.",
    )
    parser.add_argument(
        "--task", default=None,
        help="Language instruction for SmolVLA. If omitted, this script tries "
             "the checkpoint metadata then the training-dataset features.",
    )
    parser.add_argument(
        "--env-config", default="ur10e_ridgeback_dual_cam_env",
        help="crisp_gym env config name (default: dual-cam Orbbec + D405).",
    )
    parser.add_argument(
        "--fps", type=float, default=20.0,
        help="Control-loop tick rate. Each tick pops one action from the RTC "
             "queue and pushes one TargetItem (cycle-snapped to the 500 Hz "
             "CRISP controller).",
    )
    parser.add_argument("--max-speed", type=float, default=1.0)
    parser.add_argument("--min-speed", type=float, default=1.0)
    parser.add_argument("--clamp-deg", type=float, default=5.0)
    parser.add_argument(
        "--scale-kp", action="store_true",
        help="Enable xVLA kp²/kd/gripper/time scaling at the constant s_eff "
             "= --max-speed.",
    )
    parser.add_argument("--kp-exp", type=float, default=2.0)
    parser.add_argument("--kd-exp", type=float, default=1.0)
    parser.add_argument("--kp-scale-warn", type=float, default=3.0)
    parser.add_argument(
        "--gripper-base-speed", type=float, default=DEFAULT_GRIPPER_SPEED,
    )
    parser.add_argument("--controller-node", default="/cartesian_controller")
    parser.add_argument("--gripper-cm", default="/gripper/controller_manager")
    parser.add_argument(
        "--gripper-direct-action", action="store_true",
        help="Send gripper commands via env.gripper._command_action_client "
             "(action goal) instead of the /target_gripper_state Float32 publisher.",
    )
    parser.add_argument(
        "--gripper-no-edge-detect", action="store_true",
        help="DISABLE the sender's gripper edge-detection.",
    )
    parser.add_argument(
        "--gripper-latch-frames", type=int, default=0,
        help="Hysteresis latch on gripper command (Python sender only).",
    )
    parser.add_argument(
        "--invert-gripper", action="store_true",
        help="Flip the gripper action (1-grip).",
    )
    parser.add_argument(
        "--no-camera", action="store_true",
        help="Destroy camera subscriptions (smoke test only).",
    )
    parser.add_argument(
        "--no-gripper-state", action="store_true",
        help="Destroy the gripper joint_states subscription.",
    )
    parser.add_argument(
        "--no-safety-clip", action="store_true",
        help="Disable env's safety box position clipping. Use with caution.",
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="Skip the bits that require a controller-manager.",
    )
    parser.add_argument(
        "--max-chunks", type=int, default=-1,
        help="Stop after this many inference chunks (-1 = run until Ctrl-C).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Rehearse the pipeline (queue, sender pacing, RTC dispatcher) "
             "without touching /target_pose / gripper / scaler.",
    )
    parser.add_argument(
        "--rtc-execution-horizon", type=int, default=None,
        help="Override RTC execution_horizon from the YAML.",
    )
    parser.add_argument(
        "--rtc-inference-delay", type=int, default=None,
        help="Override RTC initial inference_delay hint from the YAML.",
    )
    parser.add_argument(
        "--rtc-max-guidance-weight", type=float, default=None,
        help="Override RTC max_guidance_weight from the YAML.",
    )
    parser.add_argument(
        "--rtc-disable", action="store_true",
        help="Set RTCConfig.enabled=false (falls back to ActionQueue append "
             "semantics). Useful for ablations.",
    )
    parser.add_argument(
        "--queue-refill-threshold", type=int, default=None,
        help="Override queue_refill_threshold from the YAML.",
    )
    parser.add_argument(
        "--num-inference-steps", type=int, default=None,
        help="Override SmolVLA's flow-matching denoise steps at load time.",
    )
    parser.add_argument(
        "--allow-empty-cameras", type=int, default=None,
        help="Override the checkpoint's `empty_cameras` slot count. Use when "
             "the model was trained with more cameras than the env provides "
             "(e.g. trained with camera1/2/3, deployed with camera+d405 — "
             "set to 1 to zero-pad the missing camera3). 0 = strict; None = "
             "leave the checkpoint config untouched.",
    )
    parser.add_argument(
        "--startup-delay", type=float, default=0.0,
        help="Sleep this many seconds after the sender starts but before the "
             "producer pushes the first TargetItem.",
    )
    parser.add_argument(
        "--debug-publish", action="store_true",
        help="Record per-publish timing on the sender thread.",
    )
    parser.add_argument("--yes", action="store_true", help="Skip the [Y/n] prompt.")
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    args = parser.parse_args()
    setup_logging(level=args.log_level)

    if args.min_speed > args.max_speed:
        logger.error("--min-speed (%.2f) > --max-speed (%.2f).",
                     args.min_speed, args.max_speed)
        return 2
    pretrained_path = Path(args.pretrained_path)
    if not pretrained_path.exists():
        logger.error("Pretrained path does not exist: %s", pretrained_path)
        return 1

    # ---- Load policy YAML (RTC config + dispatcher knob) ----
    policy_yaml_path = find_config(f"policy/{args.policy_config}.yaml")
    if policy_yaml_path is None:
        logger.error("Policy config not found: %s", args.policy_config)
        return 1
    with open(policy_yaml_path, "r") as f:
        policy_yaml = yaml.safe_load(f)
    if policy_yaml.get("name") != "smolvla_rtc_policy":
        logger.error(
            "Policy config %s has name=%r; expected 'smolvla_rtc_policy'.",
            args.policy_config, policy_yaml.get("name"),
        )
        return 1
    rtc_cfg_dict: dict[str, Any] = dict(policy_yaml.get("rtc") or {})
    if args.rtc_execution_horizon is not None:
        rtc_cfg_dict["execution_horizon"] = int(args.rtc_execution_horizon)
    if args.rtc_inference_delay is not None:
        rtc_cfg_dict["inference_delay"] = int(args.rtc_inference_delay)
    if args.rtc_max_guidance_weight is not None:
        rtc_cfg_dict["max_guidance_weight"] = float(args.rtc_max_guidance_weight)
    if args.rtc_disable:
        rtc_cfg_dict["enabled"] = False

    queue_refill_threshold = int(
        args.queue_refill_threshold
        if args.queue_refill_threshold is not None
        else policy_yaml.get("queue_refill_threshold", 12)
    )
    num_inference_steps = (
        args.num_inference_steps
        if args.num_inference_steps is not None
        else policy_yaml.get("num_inference_steps")
    )

    # ---- Task resolution ----
    task = _resolve_task(pretrained_path, args.task)

    # ---- Pre-flight summary ----
    print()
    print("=== SmolVLA + RTC deploy ===")
    print(f"  pretrained:    {pretrained_path}")
    print(f"  task:          {task!r}")
    print(f"  env:           {args.env_config}")
    print(f"  fps:           {args.fps}  (dt_base={1.0/args.fps*1000:.2f} ms)")
    print(f"  rtc:           {rtc_cfg_dict}")
    print(f"  refill_th:     {queue_refill_threshold}")
    if args.scale_kp:
        print(f"  scale-kp:      ON  s_eff={args.max_speed} kp_exp={args.kp_exp} kd_exp={args.kd_exp}")
    else:
        print(f"  scale-kp:      OFF")
    print(
        f"  dry-run:       {'ON (no /target_pose / gripper publishes)' if args.dry_run else 'OFF — robot WILL move'}"
    )
    print()
    if not args.yes:
        prompt = (
            "  Dry-run rehearsal. Continue? [y/N] " if args.dry_run
            else "  Deploy — the arm WILL move along the SmolVLA trajectory. Continue? [y/N] "
        )
        try:
            ans = input(prompt).strip().lower()
        except EOFError:
            return 0
        if ans not in ("y", "yes"):
            logger.info("Aborted.")
            return 0

    # ---- Env bringup ----
    logger.info("Creating environment: %s", args.env_config)
    env = make_env(env_type=args.env_config, control_type="cartesian", namespace="")
    enable_target_pose_publishing(env)
    fix_gripper_self_subscription(env)
    if args.no_safety_clip:
        env.config.safety_box = None
    if args.offline:
        from crisp_gym.envs.manipulator_env_config import GripperMode

        env.robot.wait_until_ready(timeout=3)
        if env.config.gripper_mode != GripperMode.NONE:
            env.gripper.wait_until_ready(timeout=3)
        for camera in env.cameras:
            camera.wait_until_ready(timeout=3)
        for sensor in env.sensors:
            sensor.wait_until_ready(timeout=3)
        logger.info("Robot ready (offline — controller_manager skipped).")
    else:
        env.wait_until_ready()
        logger.info("Robot ready.")

    # ---- Obs schema + rolling buffer ----
    obs_schema = _build_obs_schema(env)
    last_obs: list[dict | None] = [None]
    logger.info("obs schema: %d keys", len(obs_schema))

    # ---- Phase 1: home ----
    if args.offline:
        logger.info("Phase 1: SKIPPED (offline)")
    else:
        logger.info("Phase 1: homing to env default")
        env.home(blocking=True)

    if (
        args.gripper_direct_action
        and env.gripper is not None
        and env.gripper._command_action_client is not None
    ):
        env.gripper._target = None  # silence crisp_py's 30 Hz relay; sender owns gripper

    # ---- Phase 2: cartesian controller ----
    if args.offline:
        logger.info("Phase 2: SKIPPED (offline)")
    else:
        logger.info("Phase 2: switching to cartesian controller")
        env.switch_controller("cartesian")

    # ---- Phase 2b: scaler (optional) ----
    scaler = None
    if args.scale_kp:
        scaler = ReplayScaler(
            env,
            s_eff=np.array([float(args.max_speed)]),
            base_gripper_speed=args.gripper_base_speed,
            controller_node=args.controller_node,
            gripper_cm=args.gripper_cm,
            kp_warn_threshold=args.kp_scale_warn,
            kp_exp=args.kp_exp,
            kd_exp=args.kd_exp,
            gripper_stride=1,
        )
        logger.info("Phase 2b: applying scaler at constant s_eff=%.2f", args.max_speed)
        scaler.apply()

    # ---- Phase 2c: GIL-hygiene ----
    if args.no_gripper_state and env.gripper is not None:
        gs = getattr(env.gripper, "_joint_subscriber", None)
        if gs is not None:
            env.gripper.node.destroy_subscription(gs)
            env.gripper._joint_subscriber = None
    if args.no_camera and env.cameras:
        for cam in env.cameras:
            for sub in list(cam.node.subscriptions):
                cam.node.destroy_subscription(sub)
            for t in list(cam.node.timers):
                cam.node.destroy_timer(t)
        logger.warning("--no-camera: destroyed camera subs/timers")

    # ---- Phase 3: publish channels ----
    base_frame_id = env.robot.config.base_frame
    target_pose_pub = env.robot._target_pose_publisher
    env.robot._target_pose_publisher = None  # silence crisp_py's 20Hz timer
    pose_msg = PoseStamped()
    pose_msg.header.frame_id = base_frame_id

    gripper_raw_pub = None
    gripper_action_client = None
    gripper_max_effort = 0.0
    gripper_unnormalize_fn = None
    gripper_enabled = env.gripper is not None
    if gripper_enabled:
        gripper_unnormalize_fn = env.gripper._unnormalize
        gripper_max_effort = float(env.gripper.config.max_effort)
        if args.gripper_direct_action and env.gripper._command_action_client is not None:
            gripper_action_client = env.gripper._command_action_client
        else:
            gripper_raw_pub = env.robot.node.create_publisher(
                Float32, "/target_gripper_state", 1
            )

    # ---- Phase 4: sender ----
    replay_log: list[dict] = []
    q: queue.Queue[TargetItem] = queue.Queue(maxsize=128)
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
        state_capture_fn=None,
        debug_publish=args.debug_publish,
        dry_run=args.dry_run,
        gripper_edge_detect=not args.gripper_no_edge_detect,
        gripper_latch_frames=args.gripper_latch_frames,
    )
    sender.start()
    if args.startup_delay > 0:
        logger.info("Phase 4b: sleeping %.2fs for /target_pose discovery.", args.startup_delay)
        time.sleep(args.startup_delay)

    # ---- Phase 5: chunk source bootstrap ----
    dt_base = 1.0 / max(args.fps, 1e-9)
    obs_buf: deque = deque(maxlen=1)  # SmolVLA uses n_obs_steps=1
    for _ in range(1):
        obs_buf.append(_get_obs_zerofill(env, obs_schema, last_obs))

    chunk_source = _SmolVLARTCChunkSource(
        pretrained_path=str(pretrained_path),
        env=env,
        task=task,
        rtc_cfg_dict=rtc_cfg_dict,
        queue_refill_threshold=queue_refill_threshold,
        num_inference_steps=num_inference_steps,
        allow_empty_cameras=args.allow_empty_cameras,
        dt_base=dt_base,
        obs_buf=obs_buf,
        obs_schema=obs_schema,
        last_obs_holder=last_obs,
        action_dim=7,
    )
    chunk_source.bootstrap()
    chunk_source.start_dispatcher()

    # ---- Phase 6: producer loop ----
    run_started_at = datetime.now().isoformat(timespec="seconds")
    run_started_mono = time.monotonic()
    stopped_by = "unknown"
    last_pushed_deadline: float | None = None
    next_tick = time.monotonic()
    pushed_items = 0
    starved_ticks = 0
    interrupted = False

    s_eff = float(args.max_speed)
    cycles_const = max(1, int(math.ceil((dt_base / max(s_eff, 1e-9)) / CONTROL_DT)))
    dt_eff_const = cycles_const * CONTROL_DT
    logger.info(
        "Phase 6: producer running — s_eff=%.2f → cycles=%d dt_eff=%.1fms",
        s_eff, cycles_const, dt_eff_const * 1000.0,
    )

    try:
        while True:
            if args.max_chunks > 0 and chunk_source.chunk_count >= args.max_chunks:
                logger.info("Reached --max-chunks=%d.", args.max_chunks)
                stopped_by = "normal"
                break

            # Pace to dt_base.
            now = time.monotonic()
            if next_tick < now:
                next_tick = now
            else:
                time.sleep(next_tick - now)
            next_tick += dt_base

            action = chunk_source.pop_action()
            if action is None:
                starved_ticks += 1
                if starved_ticks % 50 == 1:
                    logger.warning(
                        "Action queue empty for %d ticks (inference latency "
                        "outruns the refill window?).", starved_ticks,
                    )
                continue
            starved_ticks = 0

            # SmolVLA checkpoints come in two action layouts:
            #   * 7D: [x, y, z, r0, r1, r2, grip]  — full deploy convention.
            #   * 6D: [x, y, z, r0, r1, r2]        — "nogrip" policy that does
            #     NOT command the gripper. We disable gripper for the
            #     pre-compute call but keep the env's gripper publishers
            #     alive so a previously-set grip (e.g. from env.home()) is
            #     preserved without us re-issuing goals every frame.
            action_has_gripper = action.shape[-1] >= 7
            chunk_gripper = gripper_enabled and action_has_gripper
            if action.shape[-1] < 6:
                logger.warning(
                    "Skipping action with shape %s (need >= 6 dims).",
                    action.shape,
                )
                continue

            # Cycle-snap deadline = previous deadline + dt_eff (or now if empty).
            now = time.monotonic()
            if last_pushed_deadline is None or last_pushed_deadline < now:
                deadline = now + dt_eff_const
            else:
                deadline = last_pushed_deadline + dt_eff_const
            last_pushed_deadline = deadline

            single_chunk = np.atleast_2d(action.astype(np.float64))
            target_xyz, target_quat, grip_raw, actions_f32 = _pre_compute_chunk_arrays(
                single_chunk,
                args=args,
                gripper_enabled=chunk_gripper,
                gripper_unnormalize_fn=gripper_unnormalize_fn,
                rotation_from_action=env.action_to_rotation,
            )
            grip = float(grip_raw[0]) if chunk_gripper else None
            item = TargetItem(
                pose_xyz=target_xyz[0],
                pose_quat=target_quat[0],
                grip_raw=grip,
                action=actions_f32[0],
                deadline_mono=float(deadline),
                frame_idx=pushed_items,
                s_eff=float(s_eff),
                cycles=int(cycles_const),
            )
            q.put(item)
            pushed_items += 1

    except KeyboardInterrupt:
        logger.info("Ctrl-C received; shutting down.")
        interrupted = True
        stopped_by = "ctrl_c"
    except Exception:
        logger.exception("Producer loop crashed.")
        stopped_by = "error"
    finally:
        # ---- Teardown ----
        logger.info("Shutting down chunk source...")
        try:
            chunk_source.shutdown()
        except Exception:
            logger.exception("chunk_source.shutdown() raised")
        logger.info("Waiting for sender to drain...")
        try:
            q.put(None)  # sentinel: TargetSenderThread exits on None
            sender.join(timeout=5.0)
        except Exception:
            logger.exception("sender shutdown raised")
        run_ended_mono = time.monotonic()
        wall_s = run_ended_mono - run_started_mono
        logger.info(
            "Run summary: stopped_by=%s chunks=%d items_pushed=%d "
            "starved_ticks=%d wall=%.1fs",
            stopped_by, chunk_source.chunk_count, pushed_items,
            starved_ticks, wall_s,
        )

    return 0 if not interrupted else 130


if __name__ == "__main__":
    sys.exit(main())
