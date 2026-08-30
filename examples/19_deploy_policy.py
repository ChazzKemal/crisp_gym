#!/usr/bin/env python3
"""Deploy a LeRobot-trained policy on the UR10e with the xVLA speedup pipeline.

Supports both ACT (``n_obs_steps=1``, single-step observation) and
diffusion-family policies (``n_obs_steps>=2``, stacked observation window).
Buffer sizes are auto-detected from the loaded checkpoint config — no flags.

The policy runs in a separate process (AsyncLerobotPolicy → multiprocessing
inference_worker → torch on GPU). This script plays the producer role:

  loop:
    1. read latest env._get_obs() into a rolling n_obs buffer (sized to
       policy.config.n_obs_steps)
    2. send obs_seq to the inference subprocess via parent_conn
    3. recv action chunk (K = policy.n_action_steps)
    4. compute_speed_schedule(chunk[:, :6]) → per-frame s_raw
       (xVLA n_lookahead within the chunk: the chunk's tail informs the
       earlier actions' speed factors → slow-before-curve still works
       per-chunk even when replanning at chunk boundaries.)
    5. cycle-snap → s_eff, dt_eff, cycles, absolute deadlines
    6. push K TargetItem(s) onto a bounded queue
    7. wait for the queue to fully drain before requesting the next chunk
       (sequential replan; ~50 ms ACT / ~50–300 ms diffusion inference gap
        between chunks depending on num_inference_steps — sender idles
        across that gap, but it's simplest to debug)

Consumer (publish path) is unchanged from 17_replay_dataset.py:

  TargetSenderThread pops from the queue, sleeps until item.deadline_mono,
  calls scaler.step_to(item.s_eff) at integer-cycle segment boundaries
  (one batched SetParameters per boundary), publishes /target_pose +
  /target_gripper_state. rclpy.publish releases the GIL inside C, so the
  producer's inference latency never stalls the publish cadence.

Usage:
    cd Yunfei/crisp_gym
    pixi run -e jazzy-lerobot python examples/19_deploy_policy.py \\
        --pretrained-path /path/to/lerobot/checkpoint \\
        --fps 20 --scale-kp --max-speed 1.0 --min-speed 1.0 \\
        --gripper-direct-action --no-camera --no-gripper-state

Prerequisites:
    - Robot up, controller_manager running, cartesian + JTC controllers loaded.
    - A LeRobot pretrained model directory at --pretrained-path. The policy's
      action_dim must match the env's 7-dim convention (x,y,z,r,p,y,grip).
    - The same crisp_controllers.yaml baselines used during recording. If a
      prior --scale-kp run was Ctrl-C'd and left inflated kp values, run
      `ros2 run tum09_custom reset_crisp_kp.py` before this script — see
      docs/troubleshooting_replay_inflated_kp_after_crash.md.
"""

import argparse
import csv
import json
import logging
import queue
import re
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32, Float64MultiArray
from scipy.spatial.transform import Rotation

from crisp_gym.deploy.cli import build_parser
from crisp_gym.deploy.dataset import (
    LEROBOT_CACHE,
    _load_dataset_actions,
    _strip_held_frames,
    load_dataset_info,
    load_episode_frames,
    load_episodes_meta,
)
from crisp_gym.deploy.gains import (
    DEFAULT_GRIPPER_SPEED,
    GRIPPER_MAX_SPEED_MPS,
    SPEED_CMDS_TOPIC,
    ReplayScaler,
    _spawn_gripper_speed_controller,
)
from crisp_gym.deploy.obs import _build_obs_schema, _get_obs_zerofill
from crisp_gym.deploy.sources import (
    ChunkSource,
    DatasetExhausted,
    _FakeChunkSource,
    _LeRobotChunkSource,
    _SyncLeRobotChunkSource,
)
from crisp_gym.deploy.patches import (
    enable_target_pose_publishing,
    fix_gripper_self_subscription,
)
from crisp_gym.deploy.pipeline import (
    _build_chunk_speed_schedule,
    _inpaint_blend_into_history,
)
from crisp_gym.deploy.shadow import _ShadowACTPolicy
from crisp_gym.deploy.video import (
    _VideoRecorder,
    _find_crisp_video_recorder_binary,
)
from crisp_gym.deploy.sender import TargetItem, TargetSenderThread
from crisp_gym.deploy.timing import (
    CONTROL_DT,
    _pre_compute_chunk_arrays,
    build_speed_queue_arrays,
    compute_speed_schedule,
    compute_speed_schedule_cumangle,
)
from crisp_gym.envs.manipulator_env import make_env
from crisp_gym.envs.manipulator_env_config import OrientationRepresentation
from crisp_gym.policy.async_lerobot_policy import AsyncLerobotPolicy
from crisp_gym.util.setup_logger import setup_logging


logger = logging.getLogger(__name__)


def main() -> int:
    parser = build_parser(__doc__)
    args = parser.parse_args()

    setup_logging(level=args.log_level)

    if args.min_speed > args.max_speed:
        logger.error(
            "--min-speed (%.2f) > --max-speed (%.2f); refusing.",
            args.min_speed, args.max_speed,
        )
        return 2

    if args.cpp_sender and args.gripper_direct_action:
        logger.error(
            "--cpp-sender does not support --gripper-direct-action "
            "(action-client gripper publishes are Python-only). Drop one "
            "of the two flags.",
        )
        return 2
    if args.gripper_latch_frames < 0:
        logger.error(
            "--gripper-latch-frames must be >= 0; got %d", args.gripper_latch_frames
        )
        return 2
    if args.gripper_latch_frames > 0 and args.cpp_sender:
        logger.warning(
            "--gripper-latch-frames=%d ignored: the hysteresis latch lives in "
            "the Python TargetSenderThread and is not implemented in the C++ "
            "sender. Drop --cpp-sender to use it.",
            args.gripper_latch_frames,
        )
    if args.rt_priority < 0 or args.rt_priority > 99:
        logger.error("--rt-priority must be in [0, 99]; got %d", args.rt_priority)
        return 2
    if args.rt_priority > 0 and not args.cpp_sender:
        logger.warning(
            "--rt-priority %d ignored: only applies with --cpp-sender",
            args.rt_priority,
        )
    if args.stride < 1:
        logger.error("--stride must be >= 1; got %d", args.stride)
        return 2
    if args.stride > 1:
        logger.info(
            "Chunk stride = %d: producer will slice chunk[::%d] before "
            "speed schedule. Each published target represents %d original "
            "action frames; trajectory advances %dx faster than the policy "
            "intended at the same dt_eff cadence.",
            args.stride, args.stride, args.stride, args.stride,
        )

    if bool(args.pretrained_path) == bool(args.fake_mode):
        logger.error(
            "Specify exactly one of --pretrained-path or --fake-mode "
            "(got pretrained=%r, fake=%r).",
            args.pretrained_path, args.fake_mode,
        )
        return 2

    pretrained_path = None
    if args.pretrained_path is not None:
        pretrained_path = Path(args.pretrained_path)
        if not pretrained_path.exists():
            logger.error("Pretrained path does not exist: %s", pretrained_path)
            return 1

    if args.fake_mode == "dataset" and not args.fake_repo_id:
        logger.error(
            "--fake-mode dataset requires --fake-repo-id <name>.",
        )
        return 2

    # ---- Pre-flight summary ----
    print()
    print("=== Deploy summary ===")
    if pretrained_path is not None:
        print(f"  source:        LeRobot policy")
        print(f"  pretrained:    {pretrained_path}")
    else:
        print(f"  source:        FAKE ({args.fake_mode})")
        if args.fake_mode == "dataset":
            print(f"  fake repo:     {args.fake_repo_id} ep {args.fake_episode_idx}")
            print(
                f"  fake loop:     {'ON — wraps forever' if args.fake_loop else 'OFF — exits after one pass'}"
            )
            if args.fake_drop_holds:
                print(f"  drop-holds:    ON (strip frames, eps={args.hold_eps:.0e})")
        print(f"  fake n_act:    {args.fake_n_act}")
        print(f"  fake n_obs:    {args.fake_n_obs}")
    print(f"  env:           {args.env_config}")
    print(f"  fps:           {args.fps}  (dt_base={1.0/args.fps*1000:.2f} ms)")
    if args.scale_kp:
        cum_str = (
            f" cum_lookahead={args.cum_lookahead}" if args.cum_lookahead > 0
            else ""
        )
        print(f"  scale-kp:      ON  max={args.max_speed} min={args.min_speed} "
              f"clamp_deg={args.clamp_deg} lookahead={args.lookahead}"
              f"{cum_str}")
        print(f"                     kp_exp={args.kp_exp} kd_exp={args.kd_exp}")
    else:
        print(f"  scale-kp:      OFF")
    print(f"  max chunks:    {'unbounded' if args.max_chunks <= 0 else args.max_chunks}")
    if args.overlap_threshold > 0:
        budget_ms = args.overlap_threshold * 1000.0 / max(args.fps, 1e-9)
        print(
            f"  overlap:       trigger at q<={args.overlap_threshold} "
            f"(inference budget {budget_ms:.0f}ms before sender starves)"
        )
    else:
        print(f"  overlap:       OFF — wait for full drain between chunks")
    if args.blend_overlap > 0:
        if args.blend_mode == "hermite":
            print(
                f"  blend:         overlap {args.blend_overlap} frames, "
                f"HERMITE cubic bridge (matches pos+vel at both seam "
                f"ends; gripper from new)"
            )
        else:
            skip_txt = (
                f", first {args.blend_skip} held verbatim from prev chunk"
                if args.blend_skip > 0 else ""
            )
            print(
                f"  blend:         overlap {args.blend_overlap} frames, "
                f"LINEAR ramp old→new{skip_txt} (pose blended; gripper "
                f"from new)"
            )
    else:
        print("  blend:         OFF — chunks stitched head-to-tail")
    if args.shadow_act:
        te = args.shadow_temporal_ensemble
        rtc = "RTC-configured" if te is not None else "no RTC"
        inpaint = (
            f"inpaint-tail={args.shadow_inpaint_tail}"
            if args.shadow_inpaint_tail > 0 else "no inpaint"
        )
        print(
            f"  shadow ACT:    ON ({rtc}, {inpaint}, device={args.shadow_device or 'auto'}) "
            f"— forward pass per chunk, output goes to shadow history only"
        )
    if args.dry_run:
        print(
            f"  dry-run:       ON — queue + sender run at REAL cadence; "
            f"ROS publishes gated; robot does NOT move"
        )
    else:
        print(f"  dry-run:       OFF — robot WILL move along the chunk source's trajectory")
    print(
        f"  zero-fill:     ON — missing sensors substituted with zeros of the "
        f"right shape, counted in summary.json (zerofill.n_substitutions)"
    )
    if args.offline:
        print(
            f"  offline:       ON — skipping controller_manager wait, env.home(), "
            f"switch_controller. Scaler + /target_pose publish still run "
            f"(unless --dry-run)."
        )
    print()
    if not args.yes:
        prompt = (
            "  Dry-run: full pipeline + sender pacing, no /target_pose. Continue? [y/N] "
            if args.dry_run
            else "  Deploy policy — the arm WILL move along the chunk source. Continue? [y/N] "
        )
        try:
            ans = input(prompt).strip().lower()
        except EOFError:
            return 0
        if ans not in ("y", "yes"):
            logger.info("Aborted.")
            return 0

    # ---- Create env ----
    logger.info("Creating environment: %s", args.env_config)
    env = make_env(env_type=args.env_config, control_type="cartesian", namespace="")
    enable_target_pose_publishing(env)
    fix_gripper_self_subscription(env)
    if args.no_safety_clip:
        env.config.safety_box = None
    logger.info("Waiting for robot to be ready...")
    if args.offline:
        # Same checks as env.wait_until_ready() minus _wait_for_controllers
        # (which requires a live controller_manager). Each component just
        # needs a topic message to land.
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

    # ---- Obs schema + zero-fill state ----
    # Schema is derived from env config (camera resolutions, configured
    # state sub-keys) so we have the right shapes even if a sensor never
    # publishes. `last_obs` is a 1-element box so _get_obs_zerofill can
    # rebind it from the closure when a fresh obs lands.
    obs_schema = _build_obs_schema(env)
    last_obs: list[dict | None] = [None]
    logger.info(
        "obs schema: %d keys — %s",
        len(obs_schema),
        {k: f"{v[0]} {v[1].name}" for k, v in obs_schema.items()},
    )

    # ---- Build chunk source (real policy or fake) ----
    chunk_source: _LeRobotChunkSource | _SyncLeRobotChunkSource | _FakeChunkSource
    if pretrained_path is not None:
        logger.info("Loading policy from %s ... (mode=%s)",
                    pretrained_path, "sync/in-process" if args.sync else "async/subprocess")
        if args.num_inference_steps is not None or args.noise_scheduler_type is not None:
            logger.info(
                "Diffusion overrides: num_inference_steps=%s, noise_scheduler_type=%s",
                args.num_inference_steps, args.noise_scheduler_type,
            )
        if args.sync:
            chunk_source = _SyncLeRobotChunkSource(
                pretrained_path=str(pretrained_path), env=env,
                num_inference_steps=args.num_inference_steps,
                noise_scheduler_type=args.noise_scheduler_type,
                n_action_steps=args.n_act,
            )
        else:
            chunk_source = _LeRobotChunkSource(
                pretrained_path=str(pretrained_path), env=env,
                num_inference_steps=args.num_inference_steps,
                noise_scheduler_type=args.noise_scheduler_type,
                n_action_steps=args.n_act,
            )
        logger.info(
            "LeRobot chunk source ready (n_obs=%d, n_act=%d, sync=%s)",
            chunk_source.n_obs, chunk_source.n_act, args.sync,
        )
    else:
        fake_actions = None
        if args.fake_mode == "dataset":
            logger.info("Loading fake dataset %s ep %d ...",
                        args.fake_repo_id, args.fake_episode_idx)
            fake_actions = _load_dataset_actions(
                args.fake_repo_id, args.fake_episode_idx,
            )
            if args.fake_drop_holds:
                n_before = fake_actions.shape[0]
                fake_actions = _strip_held_frames(
                    fake_actions, motion_eps=float(args.hold_eps),
                )
                n_after = fake_actions.shape[0]
                n_stripped = n_before - n_after
                pct = 100.0 * n_stripped / max(n_before, 1)
                fps = max(args.fps, 1e-9)
                dur_before = n_before / fps
                dur_after = n_after / fps
                dur_saved = dur_before - dur_after
                logger.info(
                    "fake dataset (eps=%.0e): %d -> %d frames "
                    "(dropped %d, %.1f%%)",
                    args.hold_eps, n_before, n_after, n_stripped, pct,
                )
                logger.info(
                    "  playback duration @ %.1f fps: %.2f s -> %.2f s "
                    "(saved %.2f s)",
                    args.fps, dur_before, dur_after, dur_saved,
                )
                if n_stripped == 0:
                    logger.warning(
                        "no held frames detected with --hold-eps=%.0e; "
                        "recording noise floor may be above this threshold "
                        "— try a larger value (e.g. 1e-4).",
                        args.hold_eps,
                    )
        chunk_source = _FakeChunkSource(
            env,
            mode=args.fake_mode,
            n_act=args.fake_n_act,
            n_obs=args.fake_n_obs,
            dataset_actions=fake_actions,
            loop=args.fake_loop,
        )
        logger.info(
            "Fake chunk source ready (mode=%s, n_obs=%d, n_act=%d)",
            args.fake_mode, chunk_source.n_obs, chunk_source.n_act,
        )

    n_obs = chunk_source.n_obs
    n_act = chunk_source.n_act
    if args.lookahead >= n_act:
        logger.warning(
            "--lookahead=%d >= n_act=%d; lookahead window extends past "
            "chunk boundary and the tail will be edge-padded by "
            "_forward_window_sum.", args.lookahead, n_act,
        )

    # ---- Phase 1: home ----
    if args.offline:
        logger.info("Phase 1: SKIPPED (offline — no joint_trajectory_controller)")
    else:
        logger.info("Phase 1: homing to env default")
        env.home(blocking=True)
        logger.info("Phase 1: homed.")

    # In direct-action mode the deploy sender drives the gripper's
    # GripperCommand action server itself. env.home() above called
    # gripper.open(), which left crisp_py's _target set; crisp_py's own 30 Hz
    # _callback_publish_target relay would then keep streaming goals toward that
    # stale target, preempting the sender's goals every ~33 ms — the gripper
    # "sends then stops". Null _target so that relay early-returns (it returns
    # when _target is None, gripper.py:_callback_publish_target), leaving the
    # sender as the SOLE writer. Guarded to the case where the direct path is
    # actually taken (mirrors the sender wiring below); the Float32 path RELIES
    # on crisp_py's relay, so it must keep _target.
    if (
        args.gripper_direct_action
        and env.gripper is not None
        and env.gripper._command_action_client is not None
    ):
        env.gripper._target = None
        logger.info(
            "Direct-action gripper: silenced crisp_py's 30 Hz relay "
            "(gripper._target=None) — deploy sender is sole writer."
        )

    # ---- Phase 2: switch to cartesian ----
    if args.offline:
        logger.info(
            "Phase 2: SKIPPED (offline — no controller_manager to switch)"
        )
    else:
        logger.info("Phase 2: switching to cartesian controller")
        env.switch_controller("cartesian")

    # ---- Phase 2b: scaler ----
    # We don't have the full s_eff schedule upfront (policy generates chunks
    # online); seed with the worst-case peak (--max-speed) for the kp_warn
    # check, then the sender thread drives step_to() per chunk's per-frame
    # s_eff values at segment boundaries. Under --offline this still runs:
    # if no controller is alive, scaler.apply()'s wait_for_service hits a
    # 5s timeout and the scaler logs an error and continues. To measure
    # realistic scaler RPC cost in --offline mode, run fake_sensors.py
    # with its fake /cartesian_controller node (default) so the
    # GetParameters/SetParameters round-trips actually complete.
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
            gripper_stride=args.stride,
        )
        logger.info("Phase 2b: applying scaler (peak s_eff ≤ %.2f)", args.max_speed)
        scaler.apply()

    # ---- Phase 2b': pin gripper speed_limit to driver max ----
    # Independent of --scale-kp. The scaler (if any) reads the SAME
    # gripper_speed_controller and may overwrite this value on its first
    # step_to() call, so this only sticks for the no-scaler path. If the
    # user has BOTH --gripper-max-speed and --scale-kp set, warn that the
    # scaler will subsequently drive the speed back down to base_gripper_speed
    # * s_eff per the cycle-snap schedule.
    # TEMP_DISABLE_GRIPPER_SPEED: gripper_speed_controller adjustment is
    # currently disabled — drop the `and False` to re-enable.
    if args.gripper_max_speed and False:
        gripper_present = env.gripper is not None
        if not args.offline and gripper_present:
            try:
                ok, msg = _spawn_gripper_speed_controller(args.gripper_cm)
                if ok:
                    pub = env.robot.node.create_publisher(
                        Float64MultiArray, SPEED_CMDS_TOPIC, 1,
                    )
                    # One-shot publish; the controller latches the last value.
                    pub.publish(Float64MultiArray(data=[float(GRIPPER_MAX_SPEED_MPS)]))
                    time.sleep(0.3)  # let DDS deliver before sender starts
                    logger.info(
                        "Phase 2b': pinned gripper speed_limit to %.3f m/s "
                        "(driver max). controller: %s",
                        GRIPPER_MAX_SPEED_MPS, msg,
                    )
                else:
                    logger.warning(
                        "Phase 2b': could not spawn gripper_speed_controller "
                        "(%s) — gripper will use whatever speed_limit was set "
                        "before this deploy started. Run "
                        "`ros2 control list_controllers -c %s` to inspect.",
                        msg, args.gripper_cm,
                    )
            except Exception:
                logger.exception("Phase 2b': failed to pin gripper max speed")
        if args.scale_kp:
            logger.warning(
                "Both --gripper-max-speed and --scale-kp are set. The scaler "
                "will overwrite the driver's speed_limit per chunk based on "
                "--gripper-base-speed * s_eff. --gripper-max-speed only "
                "affects the BASELINE (before scaler.step_to fires)."
            )

    # ---- Phase 2c: GIL-hygiene flags ----
    if args.no_gripper_state and env.gripper is not None:
        gs = getattr(env.gripper, "_joint_subscriber", None)
        if gs is not None:
            env.gripper.node.destroy_subscription(gs)
            env.gripper._joint_subscriber = None
            logger.info("Phase 2c: --no-gripper-state destroyed gripper joint_states sub")

    if args.no_camera and env.cameras:
        total_subs = 0
        total_timers = 0
        for cam in env.cameras:
            cnode = cam.node
            for sub in list(cnode.subscriptions):
                cnode.destroy_subscription(sub)
                total_subs += 1
            for t in list(cnode.timers):
                cnode.destroy_timer(t)
                total_timers += 1
        logger.warning(
            "Phase 2c: --no-camera destroyed %d sub(s) and %d timer(s) — "
            "env._get_obs() will return stale image frames.",
            total_subs, total_timers,
        )

    # ---- Phase 3 setup: publish channels ----
    # Mirrors 17_replay_dataset.py's Phase 3 setup. Steal the target_pose
    # publisher created by enable_target_pose_publishing(), null the
    # attribute so the 20 Hz timer in crisp_py becomes a no-op (otherwise
    # it'd republish the initial pose at 20 Hz and fight with our sender
    # thread). With --cpp-sender we ALSO destroy the Python publisher so
    # the C++ subprocess can be the only publisher on /target_pose.
    base_frame_id = env.robot.config.base_frame
    if args.cpp_sender:
        # Destroy and forget the rclpy publisher. The C++ binary will create
        # its own on the same topic.
        py_pose_pub = env.robot._target_pose_publisher
        env.robot._target_pose_publisher = None
        if py_pose_pub is not None:
            try:
                env.robot.node.destroy_publisher(py_pose_pub)
            except Exception:
                logger.exception("failed to destroy py-side target_pose publisher")
        target_pose_pub = None
        pose_msg = None
    else:
        target_pose_pub = env.robot._target_pose_publisher
        env.robot._target_pose_publisher = None
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
        if (
            args.gripper_direct_action
            and env.gripper._command_action_client is not None
        ):
            gripper_action_client = env.gripper._command_action_client
        elif not args.cpp_sender:
            # Python sender path: create the Float32 publisher here. With
            # --cpp-sender, the C++ subprocess creates its own.
            gripper_raw_pub = env.robot.node.create_publisher(
                Float32, "/target_gripper_state", 1
            )

    # ---- Phase 3: start sender (Python thread or C++ subprocess) ----
    replay_log: list[dict] = []
    if args.cpp_sender:
        from crisp_gym.deploy.cpp_sender import CppSenderHandle
        gripper_topic = None
        if gripper_enabled and not args.gripper_direct_action:
            gripper_topic = "/target_gripper_state"
        sender = CppSenderHandle(
            target_pose_topic=env.robot.config.target_pose_topic,
            gripper_topic=gripper_topic,
            frame_id=base_frame_id,
            scaler=scaler,
            replay_log=replay_log,
            state_capture_fn=None,
            debug_publish=args.debug_publish,
            dry_run=args.dry_run,
            rt_priority=args.rt_priority,
        )
        # The producer reads `q.qsize()` to track queue depth. Make the
        # sender double as the queue.
        q = sender
    else:
        q = queue.Queue(maxsize=128)
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
    logger.info("Phase 3: sender %s started",
                "(C++ subprocess)" if args.cpp_sender else "(Python thread)")

    # ---- Phase 3a: spawn video recorders BEFORE the startup delay ----
    # Each crisp_video_recorder subprocess subscribes to a camera topic over
    # DDS; endpoint discovery + first-frame arrival takes ~2-3 s (grep a
    # video_recorder_*.log for "subscribing" -> "video writer opened"). They
    # MUST be spawned before the startup_delay sleep so that delay doubles as
    # their settle window. Spawned after it, the arm starts moving before the
    # recorder is subscribed, the opening best-effort frames are dropped, and
    # the start of the episode never reaches the mp4 (the missing-start bug).
    # NOTE: this relies on --startup-delay being set (your runs use 4.0); with
    # --startup-delay 0 the recorders still get no settle time.
    #
    # out_dir / run_started_at are computed here so each subprocess streams
    # straight into the run folder; run_started_mono (the duration anchor)
    # stays down by the loop so duration_s still excludes this delay.
    run_started_at = datetime.now().isoformat(timespec="seconds")
    ts_dir = run_started_at.replace(":", "").replace("-", "")
    if getattr(args, "run_tag", None):
        safe = re.sub(r"[^A-Za-z0-9._-]+", "-", args.run_tag).strip("-")
        if safe:
            ts_dir = f"{ts_dir}_{safe}"
    out_dir = LEROBOT_CACHE / "deploy_runs" / ts_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("deploy run folder: %s", out_dir)

    # --save-video: spawn one crisp_video_recorder subprocess per requested
    # camera. Each writes video_<cam_name>.mp4 + video_recorder_<cam_name>.log
    # into out_dir. Names not in env.cameras are warned + skipped; the run
    # continues without that recorder. Empty list (e.g. env has no cameras)
    # makes --save-video a silent no-op.
    video_recorders: list[_VideoRecorder] = []
    if args.save_video:
        if args.video_camera.strip().lower() == "all":
            requested = [
                getattr(c.config, "camera_name", "?") for c in env.cameras
            ]
        else:
            requested = [
                s.strip() for s in args.video_camera.split(",") if s.strip()
            ]
        by_name = {
            getattr(c.config, "camera_name", "?"): c for c in env.cameras
        }
        if not env.cameras:
            logger.warning("--save-video ignored: env has no cameras.")
        for name in requested:
            cam_match = by_name.get(name)
            if cam_match is None:
                logger.warning(
                    "--save-video: camera '%s' not found in env "
                    "(have: %s); skipping.",
                    name, list(by_name.keys()),
                )
                continue
            video_path = out_dir / f"video_{name}.mp4"
            video_log = out_dir / f"video_recorder_{name}.log"
            recorder = _VideoRecorder(
                camera=cam_match,
                out_path=video_path,
                fps=args.video_fps,
                log_path=video_log,
            )
            recorder.start()
            video_recorders.append(recorder)
            logger.info(
                "Phase 2c: --save-video on (camera=%s → %s @ %.1f Hz, "
                "subprocess log: %s)",
                name, video_path, args.video_fps, video_log,
            )

    if args.startup_delay > 0:
        logger.info(
            "Phase 3b: %.2fs startup delay — lets the cartesian_controller "
            "subscriber match the sender's /target_pose publisher AND the "
            "video recorders finish subscribing to their camera topics "
            "before the first chunk lands and the arm starts moving.",
            args.startup_delay,
        )
        time.sleep(args.startup_delay)

    # ---- Phase 4: producer loop ----
    dt_base = 1.0 / max(args.fps, 1e-9)
    obs_buf: deque = deque(maxlen=n_obs)

    logger.info("Phase 4: filling initial obs buffer (n_obs=%d)", n_obs)
    for _ in range(n_obs):
        obs_buf.append(_get_obs_zerofill(env, obs_schema, last_obs))

    # ---- Phase 4b: optional shadow ACT instantiation ----
    # Built AFTER the initial obs fill so we can auto-derive input_features
    # (image/state shapes) from a real observation. Only constructed when
    # --shadow-act is set; otherwise stays None and the loop skips it.
    shadow_policy: _ShadowACTPolicy | None = None
    pred_dt_samples_shadow: list[float] = []
    if args.shadow_act:
        if not args.fake_mode:
            logger.warning(
                "--shadow-act ignored when using a real policy via "
                "--pretrained-path. Shadow mode is for the fake-source "
                "smoke-test only.",
            )
        else:
            logger.info(
                "Phase 4b: instantiating shadow ACT (temporal_ensemble=%s, "
                "device=%s)...",
                args.shadow_temporal_ensemble,
                args.shadow_device or "auto",
            )
            try:
                shadow_policy = _ShadowACTPolicy(
                    obs_sample=obs_buf[-1],
                    n_act=n_act,
                    action_dim=7,
                    device=args.shadow_device,
                    temporal_ensemble_coeff=args.shadow_temporal_ensemble,
                )
            except Exception:
                logger.exception(
                    "Phase 4b: shadow ACT construction raised — running "
                    "without shadow.",
                )
                shadow_policy = None

    # Shadow inpaint state — bounded history of the shadow's predicted
    # actions, plus aggregate stats for the shutdown log. Only populated
    # when shadow_policy is alive; never consumed for execution.
    shadow_action_history: deque = deque(maxlen=max(n_act * 4, 64))
    shadow_inpaint_blend_total: int = 0    # number of action-frames blended
    shadow_inpaint_delta_sum: float = 0.0  # sum of mean L2 deltas (× n_blended)

    chunk_count = 0
    interrupted = False
    failed = False
    # 'stopped_by' reason for the summary.json written in the finally block:
    # starts 'unknown', then set to one of 'normal' (max_chunks reached),
    # 'ctrl_c', 'error', 'chunk_source_pipe_closed' along the way.
    # run_started_at, out_dir and the video recorders were set up earlier
    # (Phase 3a, before the startup delay) so the recorders catch the start
    # of the episode; run_started_mono stays here so duration_s still
    # excludes the startup delay.
    run_started_mono = time.monotonic()
    stopped_by = "unknown"
    # Producer-local: deadline of the last item we pushed to the queue.
    # New chunks anchor their first deadline AT this value + dt_eff[0] so
    # overlap-mode append doesn't double-schedule against existing items.
    # None means "queue is empty / first chunk" → anchor at time.monotonic().
    last_pushed_deadline: float | None = None
    # --gripper-slowdown-frames state. prev_grip_closed = last frame's commanded
    # gripper state (None until the first chunk; carried across chunks so an
    # open→close edge at a chunk's frame 0 is still caught). close_slow_remaining
    # = real-time frames of an in-progress grab window that still spill into the
    # next chunk.
    prev_grip_closed: bool | None = None
    close_slow_remaining: int = 0
    # Producer-side carry buffer for --blend-overlap: the last N raw action
    # frames of the previous chunk, held back (not pushed) so they can be
    # averaged with the next chunk's first N frames at the seam. None until
    # the first chunk has been processed (and whenever blending is disabled).
    blend_carry: np.ndarray | None = None
    # Producer-side: the last 2 ACTUALLY EMITTED frames of the previous
    # chunk, kept around for --blend-mode hermite so it can extract the
    # incoming velocity (last_emitted[-1] - last_emitted[-2]) to anchor
    # the cubic. None until the first chunk has emitted >= 2 frames; in
    # linear mode it stays None (cost: zero).
    prev_emitted_tail: np.ndarray | None = None
    # Producer-side: rolling buffer of the last --lookbehind action rows
    # actually pushed to the sender (post-blend, post-emit slice). Fed into
    # _build_chunk_speed_schedule so the centered window can see real past
    # motion at the chunk's left boundary instead of edge-padding. Empty
    # (and a no-op) when --lookbehind == 0. Stored as a deque of (>=6,)
    # arrays so the per-chunk concatenate is one np.asarray call.
    lookbehind_buf: deque = deque(maxlen=max(0, int(args.lookbehind)))
    # Inference-latency samples (mirror of sender.pub_dt_samples). Logged
    # as percentiles at shutdown.
    pred_dt_samples: list[float] = []
    # Starvation-risk tracking: number of chunks where inference latency
    # exceeded the previous chunk's queue-tail drain budget. Counts how
    # often the sender's queue likely went empty mid-inference (the
    # sender's underrun_count is the symptom; this is the producer-side
    # cause).
    starvation_event_count: int = 0
    # Per-stage timing of the producer loop. Always-on: lets us localize
    # where time is going when chunk-to-chunk cadence drifts away from
    # n_act * dt_eff. Summarized at shutdown (console + summary.json) and
    # optionally dumped per-chunk to chunks.csv alongside summary.json.
    stage_samples_producer: dict[str, list[float]] = {
        "get_obs_ms": [],   # env._get_obs() through crisp_py
        "synth_ms": [],     # chunk source request (mirror of inference_ms)
        "build_ms": [],     # speed schedule + cycle-snap + pre-compute arrays
        "push_ms": [],      # K * q.put()
        "drain_wait_ms": [],  # wait for queue to drop to overlap threshold
    }
    chunk_rows: list[dict] = []  # one per chunk, dumped to chunks.csv
    # --record-trace storage. Each entry is a dict for one captured chunk;
    # converted to stacked arrays + JPEG files at shutdown. We bind the
    # output directory at shutdown (it's derived from run_started_at), so
    # images during the loop go into a TEMPORARY list of (filename, bytes)
    # and we only touch disk for the JPEGs at shutdown — keeps the producer
    # loop's I/O profile unchanged when --record-trace is off, and on
    # bounded when it's on (no concurrent disk syscalls per chunk).
    trace_records: list[dict] = []
    trace_images_buf: list[tuple[str, np.ndarray]] = []  # (filename, BGR uint8)
    # Mean dt_eff of the most recently pushed chunk, used to compute the
    # drain budget for the NEXT chunk's pre-inference queue check.
    # Seeded with dt_base so the first iteration has a sensible budget.
    dt_eff_mean_prev: float = 1.0 / max(args.fps, 1e-9)

    try:
        logger.info(
            "Phase 4: deploying — Ctrl-C to stop. Overlap threshold = %d "
            "(next inference fires when queue <= %d).",
            args.overlap_threshold, args.overlap_threshold,
        )
        while True:
            if args.max_chunks > 0 and chunk_count >= args.max_chunks:
                logger.info("Reached --max-chunks=%d, stopping", args.max_chunks)
                stopped_by = "normal"
                break

            # 1. Refresh obs buffer (cameras / joints / ee / gripper). This
            #    is the ONLY hot-path touch of env._get_obs(); the sender
            #    thread never reads obs. Cameras live behind their own
            #    daemon spinners — the get_obs call below is cheap (dict
            #    copy of the latest cached frame) but does momentarily
            #    grab the GIL. That's fine here, off the publish path.
            #    Wrapped in _get_obs_zerofill so a missing/silent sensor
            #    substitutes a zero array of the right shape instead of
            #    raising RuntimeError mid-deploy.
            _t_stage = time.perf_counter()
            obs_buf.append(_get_obs_zerofill(env, obs_schema, last_obs))
            get_obs_ms = (time.perf_counter() - _t_stage) * 1000.0
            stage_samples_producer["get_obs_ms"].append(get_obs_ms)

            # 2. Request a chunk. For a real policy this blocks for inference
            #    latency (~10-50 ms typical, ACT can be <10 ms, diffusion
            #    50-200 ms). For fake sources it returns immediately.
            #    Pre-inference snapshot: how many items are queued AND what
            #    drain budget that buys us at the previous chunk's cadence.
            #    If inference latency exceeds the budget, the sender will
            #    block on q.get() and we'll observe an underrun cluster.
            q_before_inf = q.qsize()
            tail_budget_ms = q_before_inf * dt_eff_mean_prev * 1000.0
            t_send = time.monotonic()
            try:
                chunk = chunk_source.request(obs_buf)
            except DatasetExhausted as e:
                logger.info("Fake dataset exhausted (%s) — exiting cleanly.", e)
                stopped_by = "dataset_exhausted"
                break
            except (BrokenPipeError, EOFError):
                logger.error("Chunk source pipe closed; exiting loop.")
                failed = True
                stopped_by = "chunk_source_pipe_closed"
                break
            inf_dt = time.monotonic() - t_send
            inf_dt_ms = inf_dt * 1000.0
            pred_dt_samples.append(inf_dt)
            stage_samples_producer["synth_ms"].append(inf_dt_ms)
            chunk_count += 1

            # --record-trace capture. Done right after the chunk arrives,
            # BEFORE the speed-schedule/cycle-snap step modifies the
            # publish cadence. The action vectors themselves aren't
            # modified by the scaler (only timing is), so the chunk we
            # capture here is exactly what the sender will publish.
            if args.record_trace and (chunk_count - 1) % max(1, args.record_trace_every) == 0:
                obs_now = obs_buf[-1]
                record = {
                    "chunk_idx": int(chunk_count - 1),  # match chunk_rows
                    "wall_ns": int(time.time_ns()),
                    "mono_ns": int(time.monotonic_ns()),
                    "chunk": np.asarray(chunk, dtype=np.float32),
                }
                for k, v in obs_now.items():
                    if k.startswith("observation.state."):
                        record[k] = np.asarray(v, dtype=np.float32).reshape(-1)
                task = obs_now.get("task", "")
                if task:
                    record["task"] = str(task)
                trace_records.append(record)

                # Buffer JPEG-encodable image arrays for shutdown-time disk
                # write. crisp_py returns HxWxC uint8 RGB; cv2.imwrite needs
                # BGR. We do the cheap channel-flip in-line here and defer
                # the actual write to keep the hot loop fast.
                if not args.record_trace_no_images:
                    for k, v in obs_now.items():
                        if not k.startswith("observation.images."):
                            continue
                        img = np.asarray(v)
                        if img.ndim != 3 or img.shape[-1] != 3:
                            continue
                        cam = k.rsplit(".", 1)[-1]
                        fname = f"chunk_{chunk_count - 1:06d}_{cam}.jpg"
                        # Channel-flip view; cv2 will encode at write time.
                        trace_images_buf.append((fname, img[..., ::-1].copy()))

            if inf_dt_ms > tail_budget_ms:
                starvation_event_count += 1
                logger.warning(
                    "chunk %d: inference (%.1fms) > queue tail budget "
                    "(%.1fms, q_before_inf=%d * dt_eff_mean_prev=%.1fms). "
                    "Sender likely starved; bump --overlap-threshold or "
                    "accept underruns.",
                    chunk_count, inf_dt_ms, tail_budget_ms,
                    q_before_inf, dt_eff_mean_prev * 1000.0,
                )

            # 2b. Run the shadow ACT forward pass on the same obs the fake
            #     source just saw. We discard the output for execution — only
            #     the wall time matters — but we DO route it into the shadow
            #     history deque and optionally inpaint-blend it there
            #     (--shadow-inpaint-tail). The shadow history is never
            #     consumed by the robot; it's purely a smoke test of the
            #     blending math, exercising the code path a real RTC-enabled
            #     producer would run.
            if shadow_policy is not None:
                t_shadow = time.monotonic()
                try:
                    shadow_chunk = shadow_policy.predict(obs_buf[-1])
                except Exception:
                    logger.exception(
                        "shadow predict() raised at chunk %d; disabling shadow.",
                        chunk_count,
                    )
                    shadow_policy = None
                else:
                    pred_dt_samples_shadow.append(time.monotonic() - t_shadow)
                    if args.shadow_inpaint_tail > 0:
                        n_blended, mean_delta = _inpaint_blend_into_history(
                            shadow_action_history,
                            shadow_chunk,
                            args.shadow_inpaint_tail,
                        )
                        if n_blended > 0:
                            shadow_inpaint_blend_total += n_blended
                            shadow_inpaint_delta_sum += mean_delta * n_blended
                    else:
                        # No blending — still track the chunk in history so
                        # the size of the history reflects real usage.
                        for action in shadow_chunk:
                            shadow_action_history.append(np.asarray(action).copy())

            if not isinstance(chunk, np.ndarray) or chunk.ndim != 2:
                logger.warning("Chunk %d: unexpected payload %r — skipping",
                               chunk_count, type(chunk).__name__)
                continue
            K_raw = chunk.shape[0]
            if K_raw == 0 or chunk.shape[1] < 7:
                logger.warning("Chunk %d: bad shape %s — skipping",
                               chunk_count, chunk.shape)
                continue

            # 2c. Stride: decimate the chunk before speed schedule. Each
            #     remaining frame still gets one dt_eff worth of sender time,
            #     so the trajectory advances `stride` times faster per
            #     published target. The speed schedule + cycle-snap run on
            #     the strided chunk; deltas between consecutive entries are
            #     `stride` times larger, which compute_speed_schedule sees
            #     directly. Combine with --max-speed for total speedup =
            #     stride × s_eff at dt_eff = dt_base / s_eff cadence.
            if args.stride > 1:
                chunk = chunk[::args.stride].copy()
            K = chunk.shape[0]
            if K == 0:
                logger.warning(
                    "Chunk %d: stride=%d produced empty chunk from K_raw=%d; "
                    "skipping", chunk_count, args.stride, K_raw,
                )
                continue

            # 2d. Chunk-seam blending (temporal ensembling). Hold back the
            #     last N raw frames of this chunk; average them with the next
            #     chunk's first N frames, ramping the weight old->new so the
            #     seam stays continuous with what's executing but converges to
            #     the fresher prediction. Operates on the RAW action array
            #     (xyz + rotvec) BEFORE pose/quat conversion; the gripper
            #     channel [6] is NEVER averaged (binary) — it takes the new
            #     chunk's value. Producer-side → applies to both senders. N is
            #     clamped to K//2 so the blended head [0:N] and held-back tail
            #     [K-N:] never overlap. --blend-overlap 0 keeps head-to-tail.
            if args.blend_overlap > 0 and K >= 2:
                N = min(int(args.blend_overlap), K // 2)
                if blend_carry is not None:
                    if args.blend_mode == "hermite" and prev_emitted_tail is not None and K > N + 1:
                        # Cubic Hermite bridge from (p_start, v_start) at
                        # the last actually-emitted frame to (p_end, v_end)
                        # at the first verbatim new-chunk frame after the
                        # blend zone. The blend slots chunk[0:N] are filled
                        # with N interior samples of the cubic. Bridges
                        # both position AND velocity -> no boundary kink.
                        #
                        # Parameterization: cubic on s in [0, 1], with N+1
                        # equal subdivisions (slot 0 at s=1/(N+1), slot N-1
                        # at s=N/(N+1)). Frame-step deltas (no dt scaling)
                        # because dt cancels between v and T in the
                        # standard Hermite form.
                        p_start = prev_emitted_tail[-1, :6].astype(np.float64)
                        v_start = (
                            prev_emitted_tail[-1, :6].astype(np.float64)
                            - prev_emitted_tail[-2, :6].astype(np.float64)
                        )
                        p_end = chunk[N, :6].astype(np.float64)
                        v_end = (
                            chunk[N + 1, :6].astype(np.float64)
                            - chunk[N, :6].astype(np.float64)
                        )
                        T_frames = float(N + 1)
                        s_vec = (np.arange(N) + 1) / T_frames   # (N,)
                        h00 = 2 * s_vec ** 3 - 3 * s_vec ** 2 + 1
                        h10 = s_vec ** 3 - 2 * s_vec ** 2 + s_vec
                        h01 = -2 * s_vec ** 3 + 3 * s_vec ** 2
                        h11 = s_vec ** 3 - s_vec ** 2
                        bridge = (
                            h00[:, None] * p_start
                            + (h10[:, None] * T_frames) * v_start
                            + h01[:, None] * p_end
                            + (h11[:, None] * T_frames) * v_end
                        )
                        chunk[:N, :6] = bridge.astype(chunk.dtype)
                        # Gripper [6] left as the new chunk's value
                        # (NEVER interpolated, matches linear mode).
                    else:
                        # Linear path (existing behaviour). Per-frame
                        # weighted average; skips the first `skip` frames
                        # as committed.
                        n = min(len(blend_carry), N)
                        skip = min(max(0, int(args.blend_skip)), n)
                        n_blend = n - skip  # frames actually averaged
                        for i in range(n):
                            if i < skip:
                                # Commit horizon: execute the previous chunk's
                                # prediction VERBATIM (pose AND gripper) for these
                                # already-in-flight timesteps; blending starts
                                # after them.
                                chunk[i, :] = blend_carry[i, :]
                            else:
                                # Ramp restarted across the (n_blend) blended
                                # frames: frame `skip` stays close to the committed
                                # old plan (w small) and converges to new by the
                                # end. Gripper [6] is left as the new chunk's value
                                # (never averaged).
                                j = i - skip
                                w = (j + 1) / (n_blend + 1)  # old-heavy -> new-heavy
                                chunk[i, :6] = (
                                    (1.0 - w) * blend_carry[i, :6] + w * chunk[i, :6]
                                )
                blend_carry = chunk[K - N:].copy()   # hold back for next seam
                chunk = chunk[: K - N].copy()         # emit the rest now
                K = chunk.shape[0]
                # Save the last 2 actually-emitted frames for the next
                # iteration's Hermite v_start. Only needed in hermite mode,
                # but the cost is one ndarray copy of shape (2, 7) per
                # chunk so we do it unconditionally to keep the code paths
                # symmetric. K >= 2 by the outer `if K >= 2` guard above
                # (post-emit K is K_orig - N, which is >= K_orig // 2 >= 1;
                # for K_orig >= 4 it's >= 2).
                if K >= 2:
                    prev_emitted_tail = chunk[K - 2:K].copy()

            # 3. Speed schedule on the (possibly strided) chunk.
            _t_stage = time.perf_counter()
            past = (
                np.asarray(lookbehind_buf, dtype=np.float64)
                if len(lookbehind_buf) > 0 else None
            )
            s_raw = _build_chunk_speed_schedule(
                chunk.astype(np.float64), args, past_buffer=past,
            )

            # 3b. Gripper-grab slowdown (--gripper-slowdown-frames). On each
            #     open→close transition, force s_raw = 1.0 (real-time) for that
            #     frame + the next N-1, so the arm runs at normal speed *while it
            #     grabs* and resumes speedup for the carry. Edge-triggered on the
            #     CLOSE transition, NOT the level — staying closed during the
            #     carry fires nothing, so transport keeps the speedup. The window
            #     can straddle chunk boundaries (close_slow_remaining carries the
            #     leftover). No-op when N=0, and a no-op anyway with no speedup
            #     (s_raw already 1.0). Baked into s_raw → flows through cycle-snap
            #     into dt_eff/deadlines, so it also works with --cpp-sender.
            N_grip_slow = int(getattr(args, "gripper_slowdown_frames", 0))
            if N_grip_slow > 0 and gripper_enabled:
                g_norm = np.clip(chunk[:, 6], 0.0, 1.0)
                if args.invert_gripper:
                    g_norm = 1.0 - g_norm
                closed = g_norm < 0.5  # commanded "closed" (post-invert)
                slow_mask = np.zeros(K, dtype=bool)
                # Carry-in: window opened by a close near a prior chunk's end.
                if close_slow_remaining > 0:
                    c = min(close_slow_remaining, K)
                    slow_mask[:c] = True
                    close_slow_remaining -= c
                # New open→close edges this chunk (prev_grip_closed seeds frame 0).
                was_closed = (
                    bool(prev_grip_closed) if prev_grip_closed is not None else False
                )
                for i in range(K):
                    if closed[i] and not was_closed:  # open→close edge = a grab
                        end = i + N_grip_slow
                        slow_mask[i:min(end, K)] = True
                        if end > K:
                            close_slow_remaining = max(close_slow_remaining, end - K)
                    was_closed = bool(closed[i])
                prev_grip_closed = bool(closed[-1])
                if slow_mask.any():
                    s_raw[slow_mask] = 1.0

            # 4. Cycle-snap.
            cycles, dt_eff, s_eff = build_speed_queue_arrays(
                s_raw, dt_base, K, retime=True,
            )

            # 5. Pre-compute pose / gripper for each frame.
            target_xyz, target_quat, grip_raw, actions_f32 = _pre_compute_chunk_arrays(
                chunk,
                args=args,
                gripper_enabled=gripper_enabled,
                gripper_unnormalize_fn=gripper_unnormalize_fn,
                rotation_from_action=env.action_to_rotation,
            )
            build_ms = (time.perf_counter() - _t_stage) * 1000.0
            stage_samples_producer["build_ms"].append(build_ms)

            # 6. Anchor deadlines. Two regimes:
            #    (a) Queue is empty / first chunk → anchor at now. Sender
            #        publishes frame 0 at now + dt_eff[0].
            #    (b) Items still in queue (overlap append) → anchor at the
            #        last-pushed item's deadline. New frame 0 publishes
            #        dt_eff[0] AFTER the last in-flight item finishes,
            #        giving the controller a clean dt_eff[0] window. No
            #        deadline collisions, no payload overwrites.
            # Capture queue depth pre-push for the log AFTER the q.put loop
            # (we log *after* pushing so the logger.info doesn't sit
            # between now_mono and the first q.put — that gap was costing
            # ~30 ms of Rich-rendering and pushing item 0's deadline into
            # the past, causing cascading underruns).
            q_before_push = q.qsize()

            now_mono = time.monotonic()
            if last_pushed_deadline is None or last_pushed_deadline < now_mono:
                # Empty queue, or last deadline already passed — anchor at now.
                anchor = now_mono
                anchor_mode = "fresh"
            else:
                anchor = last_pushed_deadline
                anchor_mode = "overlap"
            deadlines = anchor + np.cumsum(dt_eff)

            # 7. Push K TargetItems onto the queue IMMEDIATELY after the
            #    anchor decision — no logging in between. Always pushes
            #    (even in --dry-run); sender's dry_run flag handles the
            #    ROS-side gating.
            _t_stage = time.perf_counter()
            for i in range(K):
                grip = float(grip_raw[i]) if gripper_enabled else None
                item = TargetItem(
                    pose_xyz=target_xyz[i],
                    pose_quat=target_quat[i],
                    grip_raw=grip,
                    action=actions_f32[i],
                    deadline_mono=float(deadlines[i]),
                    frame_idx=(chunk_count - 1) * K + i,
                    s_eff=float(s_eff[i]),
                    cycles=int(cycles[i]),
                )
                q.put(item)
            push_ms = (time.perf_counter() - _t_stage) * 1000.0
            stage_samples_producer["push_ms"].append(push_ms)
            last_pushed_deadline = float(deadlines[-1])
            # Carry the chunk's mean dt_eff into the next iteration so the
            # pre-inference budget check is accurate.
            dt_eff_mean_prev = float(np.mean(dt_eff))

            # Feed the emitted action rows into the lookbehind buffer so the
            # next chunk's speed schedule can see real past motion at its
            # left boundary. Uses the post-blend, post-emit chunk (what was
            # actually queued for publish) — the truncated `K-N` slice when
            # --blend-overlap is active, the full K rows otherwise. deque
            # maxlen enforces the window size; appends are no-ops when
            # --lookbehind == 0.
            if lookbehind_buf.maxlen and lookbehind_buf.maxlen > 0:
                for i in range(K):
                    lookbehind_buf.append(
                        np.asarray(chunk[i, :6], dtype=np.float64)
                    )

            # NOW it's safe to log — sender has had a chance to pick up
            # item 0 from a deadline that's still genuinely in the future.
            logger.info(
                "Chunk %d: shape=%s inf=%.1fms anchor=%s  s_raw[%.2f-%.2f] "
                "s_eff[%.2f-%.2f] dt_eff[%.0f-%.0f]ms  "
                "q_before_inf=%d budget=%.0fms q_before_push=%d",
                chunk_count, tuple(chunk.shape), inf_dt_ms, anchor_mode,
                float(s_raw.min()), float(s_raw.max()),
                float(s_eff.min()), float(s_eff.max()),
                float(dt_eff.min()) * 1000, float(dt_eff.max()) * 1000,
                q_before_inf, tail_budget_ms, q_before_push,
            )

            # 8. Wait for the queue to drain to the overlap threshold, then
            #    loop back to request the next chunk. With threshold=2,
            #    inference fires when 2 items are still in flight — if
            #    inference latency < 2 * dt_eff (~66 ms at 30 Hz), the
            #    sender thread never sees an empty queue at chunk
            #    boundaries. Threshold=0 reverts to "wait for full drain"
            #    (sequential replan, ~inf_dt_ms gap per chunk).
            _t_stage = time.perf_counter()
            thresh = max(0, int(args.overlap_threshold))
            while q.qsize() > thresh:
                time.sleep(0.005)
            drain_wait_ms = (time.perf_counter() - _t_stage) * 1000.0
            stage_samples_producer["drain_wait_ms"].append(drain_wait_ms)

            # Per-chunk row for chunks.csv. Captures everything observable
            # at the producer level for post-hoc analysis. Negative
            # drain_wait_ms is impossible; near-zero means the sender was
            # already at/below threshold by the time we got back here.
            chunk_rows.append({
                "chunk_idx": chunk_count,
                "q_before_inf": q_before_inf,
                "q_before_push": q_before_push,
                "anchor_mode": anchor_mode,
                "K": K,
                "dt_eff_mean_ms": float(np.mean(dt_eff)) * 1000.0,
                "get_obs_ms": get_obs_ms,
                "synth_ms": inf_dt_ms,
                "build_ms": build_ms,
                "push_ms": push_ms,
                "drain_wait_ms": drain_wait_ms,
            })

    except KeyboardInterrupt:
        interrupted = True
        stopped_by = "ctrl_c"
        logger.warning("Interrupted by Ctrl-C. Draining queue + shutting down.")
    except Exception:
        logger.exception("Deploy failed")
        failed = True
        stopped_by = "error"
    finally:
        # Drain the sender thread.
        try:
            q.put(None)
        except Exception:
            logger.exception("failed to put sentinel on queue")
        try:
            sender.join(timeout=5.0)
            if sender.is_alive():
                logger.warning("sender did not exit within 5s")
        except Exception:
            logger.exception("sender.join() raised")

        # ─── Trajectory wall time ─────────────────────────────────────────
        # All robot motion has stopped at this point; everything below is
        # analytics + teardown. Logged here (not after the percentile blocks)
        # so it stays visible at a glance even when --debug-publish is on.
        # Same value is also written to summary.json as "duration_s".
        duration_s = time.monotonic() - run_started_mono

        def _fmt_dur(s: float) -> str:
            if s < 60.0:
                return f"{s:.2f}s"
            m, sec = divmod(s, 60.0)
            if m < 60.0:
                return f"{int(m)}m{sec:.1f}s"
            h, m = divmod(int(m), 60)
            return f"{h}h{m:02d}m{sec:.0f}s"

        n_published = int(sender.n_published)
        realized_fps = n_published / duration_s if duration_s > 0 else 0.0
        gripper_dedupe = int(getattr(sender, "gripper_dedupe_count", 0))
        logger.info(
            "Trajectory complete: %s wall-clock (%.2fs), %d chunks inferred, "
            "%d action frames published (%.1f fps realized vs %.1f baseline). "
            "Stopped by: %s.",
            _fmt_dur(duration_s), duration_s, chunk_count, n_published,
            realized_fps, args.fps, stopped_by,
        )
        if gripper_dedupe > 0:
            logger.info(
                "gripper edge-detect: %d publishes elided (out of %d frames; "
                "%.1f%%). Driver got to complete the trajectory between "
                "command changes instead of being preempted every tick.",
                gripper_dedupe, n_published,
                100.0 * gripper_dedupe / max(1, n_published),
            )
        gripper_latched = int(getattr(sender, "gripper_latch_blocked_count", 0))
        if gripper_latched > 0:
            logger.info(
                "gripper latch (--gripper-latch-frames=%d): %d gripper changes "
                "suppressed (out of %d frames; %.1f%%). Each blocked change was "
                "held at the last sent value for the latch window.",
                args.gripper_latch_frames, gripper_latched, n_published,
                100.0 * gripper_latched / max(1, n_published),
            )

        if args.debug_publish and sender.pub_dt_samples:
            arr = np.asarray(sender.pub_dt_samples)
            logger.info(
                "publish() ms: mean=%.2f median=%.2f p90=%.2f p99=%.2f max=%.2f  "
                "(N=%d, underruns=%d, queue depth [%d..%d])",
                arr.mean() * 1000, np.median(arr) * 1000,
                np.percentile(arr, 90) * 1000, np.percentile(arr, 99) * 1000,
                arr.max() * 1000,
                len(arr), sender.underrun_count,
                sender.queue_depth_min if sender.queue_depth_min != 2 ** 31 else 0,
                sender.queue_depth_max,
            )

        # Per-frame deadline slack: how much time the sender slept before
        # publishing each item. Negative slack = popped past the deadline =
        # action was effectively "skipped" in cadence terms (published
        # late). Always-on (cheap to capture); summarised here and in
        # summary.json. Look at p1 / min: if those go far negative, the
        # producer can't feed the queue fast enough.
        if sender.slack_samples_ms:
            slack_arr = np.asarray(sender.slack_samples_ms)
            late_pct = 100.0 * sender.n_late_frames / max(len(slack_arr), 1)
            logger.info(
                "deadline slack ms: median=%.1f p10=%.1f p1=%.1f min=%.1f  "
                "(N=%d, late frames=%d / %.1f%%, starvation events=%d)",
                float(np.median(slack_arr)),
                float(np.percentile(slack_arr, 10)),
                float(np.percentile(slack_arr, 1)),
                float(slack_arr.min()),
                len(slack_arr), sender.n_late_frames, late_pct,
                starvation_event_count,
            )

        # Shadow ACT latency. Compared against the chunk-source latency,
        # this tells you what a real ACT inference would have cost without
        # actually swapping in a trained model.
        if pred_dt_samples_shadow:
            arr = np.asarray(pred_dt_samples_shadow)
            flavor = shadow_policy.flavor if shadow_policy is not None else "?"
            logger.info(
                "shadow (%s) ms: mean=%.1f median=%.1f p90=%.1f p99=%.1f max=%.1f  (N=%d)",
                flavor,
                arr.mean() * 1000, np.median(arr) * 1000,
                np.percentile(arr, 90) * 1000, np.percentile(arr, 99) * 1000,
                arr.max() * 1000, len(arr),
            )

        # Shadow inpaint summary. Mean blend-delta tells you whether the
        # weighted average actually moved the actions meaningfully (large
        # delta = chunks were predicting very different things at the
        # overlap, smoothing did real work). Zero blends = the shadow was
        # disabled or --shadow-inpaint-tail was 0; history len reflects
        # how much of the shadow's prediction stream got cached.
        if shadow_inpaint_blend_total > 0:
            avg_delta = shadow_inpaint_delta_sum / max(shadow_inpaint_blend_total, 1)
            logger.info(
                "shadow inpaint: %d action-frames blended across %d chunks  "
                "(tail=%d each), mean |blended - new_raw| L2 = %.4f  "
                "(history len=%d)",
                shadow_inpaint_blend_total, chunk_count,
                args.shadow_inpaint_tail, avg_delta,
                len(shadow_action_history),
            )
        elif args.shadow_inpaint_tail > 0 and args.shadow_act:
            logger.info(
                "shadow inpaint: no blends recorded (shadow failed early or "
                "shadow_history was empty for every chunk)."
            )

        # Inference latency percentiles. Tells you whether the chosen
        # --overlap-threshold actually hides inference behind the in-flight
        # tail of each chunk. If p99 inference > threshold * dt_eff, the
        # sender will see queue starvation at chunk boundaries and you'll
        # need to bump --overlap-threshold (or speed up the model).
        if pred_dt_samples:
            arr = np.asarray(pred_dt_samples)
            dt_eff_ms = 1000.0 / max(args.fps, 1e-9)
            threshold_budget_ms = args.overlap_threshold * dt_eff_ms
            logger.info(
                "inference ms: mean=%.1f median=%.1f p90=%.1f p99=%.1f max=%.1f  "
                "(N=%d chunks, overlap budget=%d*%.1fms=%.1fms)",
                arr.mean() * 1000, np.median(arr) * 1000,
                np.percentile(arr, 90) * 1000, np.percentile(arr, 99) * 1000,
                arr.max() * 1000,
                len(arr), args.overlap_threshold, dt_eff_ms, threshold_budget_ms,
            )
            if args.overlap_threshold > 0 and arr.max() * 1000 > threshold_budget_ms:
                logger.warning(
                    "inference max (%.1fms) exceeded overlap budget (%.1fms). "
                    "Sender thread saw the queue empty at one or more chunk "
                    "boundaries. Consider --overlap-threshold=%d.",
                    arr.max() * 1000, threshold_budget_ms,
                    int(np.ceil(arr.max() * 1000 / dt_eff_ms)) + 1,
                )

        # Producer per-stage timing. Localises where chunk-to-chunk time
        # is going beyond the natural drain_wait_ms pause. In fake mode,
        # synth_ms should be ~0 and the largest non-drain stage points
        # straight at the bottleneck (e.g. get_obs_ms spiking = camera
        # callback GIL contention).
        if chunk_rows:
            for stage in ("get_obs_ms", "synth_ms", "build_ms", "push_ms", "drain_wait_ms"):
                samples = stage_samples_producer.get(stage, [])
                if not samples:
                    continue
                a = np.asarray(samples, dtype=np.float64)
                logger.info(
                    "producer %s: mean=%.2f median=%.2f p90=%.2f p99=%.2f max=%.2f  (N=%d)",
                    stage, a.mean(), float(np.median(a)),
                    float(np.percentile(a, 90)), float(np.percentile(a, 99)),
                    a.max(), a.size,
                )

        # Sender per-stage timing. sleep_overshoot_ms is the canonical
        # GIL-starvation indicator: high values mean time.sleep returned
        # late because the sender thread wasn't scheduled in time. pop_ms
        # high = producer not feeding the queue fast enough. pub_pose_ms
        # high = ROS publish is slow (DDS / network / serialization).
        sender_stage_samples = getattr(sender, "stage_samples", {}) or {}
        for stage in (
            "pop_ms", "scaler_rpc_ms", "sleep_overshoot_ms",
            "pub_pose_ms", "pub_grip_ms", "loop_total_ms",
        ):
            samples = sender_stage_samples.get(stage, [])
            if not samples:
                continue
            a = np.asarray(samples, dtype=np.float64)
            logger.info(
                "sender %s: mean=%.2f median=%.2f p90=%.2f p99=%.2f max=%.2f  (N=%d)",
                stage, a.mean(), float(np.median(a)),
                float(np.percentile(a, 90)), float(np.percentile(a, 99)),
                a.max(), a.size,
            )

        # ─── summary.json: durable record of this run ─────────────────────
        # Written BEFORE chunk_source.shutdown() / env.close() so any
        # teardown failure can't lose the analysis. All percentile blocks
        # above have already run, so we just collect what's already in
        # memory + the args + the timing bookkeeping. Output path mirrors
        # 17_replay_dataset.py's replay log layout.
        try:
            run_ended_at = datetime.now().isoformat(timespec="seconds")
            # duration_s already computed above (right after sender.join)

            def _percentiles(samples: list[float]) -> dict | None:
                if not samples:
                    return None
                a = np.asarray(samples, dtype=np.float64) * 1000.0
                return {
                    "n": int(a.size),
                    "mean_ms": float(a.mean()),
                    "median_ms": float(np.median(a)),
                    "p90_ms": float(np.percentile(a, 90)),
                    "p99_ms": float(np.percentile(a, 99)),
                    "max_ms": float(a.max()),
                }

            def _ms_percentiles(samples_ms: list[float]) -> dict | None:
                """Like _percentiles but takes samples already in ms — used
                for stage timers captured via time.perf_counter * 1000.
                """
                if not samples_ms:
                    return None
                a = np.asarray(samples_ms, dtype=np.float64)
                return {
                    "n": int(a.size),
                    "mean_ms": float(a.mean()),
                    "median_ms": float(np.median(a)),
                    "p90_ms": float(np.percentile(a, 90)),
                    "p99_ms": float(np.percentile(a, 99)),
                    "max_ms": float(a.max()),
                }

            def _slack_stats(samples_ms: list[float]) -> dict | None:
                """Slack samples are already in ms and span both signs —
                reuse the percentile shape but expose low percentiles (p1, p10)
                since the failure mode is 'we got popped after the deadline'.
                """
                if not samples_ms:
                    return None
                a = np.asarray(samples_ms, dtype=np.float64)
                return {
                    "n": int(a.size),
                    "mean_ms": float(a.mean()),
                    "median_ms": float(np.median(a)),
                    "p10_ms": float(np.percentile(a, 10)),
                    "p1_ms": float(np.percentile(a, 1)),
                    "min_ms": float(a.min()),
                    "max_ms": float(a.max()),
                }

            def _arg_value(v):
                if isinstance(v, Path):
                    return str(v)
                return v

            summary: dict = {
                "started_at": run_started_at,
                "ended_at": run_ended_at,
                "duration_s": duration_s,
                "stopped_by": stopped_by,
                "chunks_run": int(chunk_count),
                "n_act": int(n_act),
                "n_obs": int(n_obs),
                "args": {k: _arg_value(v) for k, v in vars(args).items()},
                "sender": {
                    "n_processed": int(sender.n_published),
                    "underrun_count": int(sender.underrun_count),
                    "gripper_dedupe_count": int(
                        getattr(sender, "gripper_dedupe_count", 0)
                    ),
                    "gripper_latch_blocked_count": int(
                        getattr(sender, "gripper_latch_blocked_count", 0)
                    ),
                    "n_late_frames": int(sender.n_late_frames),
                    "late_frame_pct": (
                        100.0 * sender.n_late_frames
                        / max(len(sender.slack_samples_ms), 1)
                    ),
                    "queue_depth_min": (
                        int(sender.queue_depth_min)
                        if sender.queue_depth_min != 2 ** 31 else None
                    ),
                    "queue_depth_max": int(sender.queue_depth_max),
                },
                "fps_baseline": float(args.fps),
                "control_dt_ms": float(CONTROL_DT * 1000.0),
                "starvation_events": int(starvation_event_count),
                "slack_ms": _slack_stats(sender.slack_samples_ms),
                "zerofill": {
                    "n_substitutions": int(sum(_ZEROFILL_COUNTS.values())),
                    "by_error": dict(_ZEROFILL_COUNTS),
                },
                "inference_ms": _percentiles(pred_dt_samples),
                "shadow_ms": _percentiles(pred_dt_samples_shadow),
                "publish_ms": _percentiles(sender.pub_dt_samples),
                "producer_stages_ms": {
                    stage: _ms_percentiles(stage_samples_producer.get(stage, []))
                    for stage in ("get_obs_ms", "synth_ms", "build_ms", "push_ms", "drain_wait_ms")
                },
                "sender_stages_ms": {
                    stage: _ms_percentiles(sender_stage_samples.get(stage, []))
                    for stage in (
                        "pop_ms", "scaler_rpc_ms", "sleep_overshoot_ms",
                        "pub_pose_ms", "pub_grip_ms", "loop_total_ms",
                    )
                },
                "shadow_inpaint": (
                    {
                        "tail_per_chunk": int(args.shadow_inpaint_tail),
                        "n_action_frames_blended": int(shadow_inpaint_blend_total),
                        "mean_l2_delta": (
                            shadow_inpaint_delta_sum
                            / max(shadow_inpaint_blend_total, 1)
                        ),
                        "history_len_final": len(shadow_action_history),
                    }
                    if args.shadow_inpaint_tail > 0 and shadow_inpaint_blend_total > 0
                    else None
                ),
                "overlap_budget_ms": (
                    args.overlap_threshold * 1000.0 / max(args.fps, 1e-9)
                ),
                "shadow_flavor": (
                    shadow_policy.flavor if shadow_policy is not None else None
                ),
            }

            # out_dir + ts_dir were computed up-front (right after
            # run_started_at) so the video writer subprocess can stream
            # into the same folder during the run. Reuse them here.
            summary_path = out_dir / "summary.json"
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)
            logger.info("summary written to %s", summary_path)

            # Per-chunk CSV: one row per producer iteration, all the stage
            # timings + queue state. Useful for plotting drift over time
            # or for spotting outlier chunks the percentile summary hides.
            if chunk_rows:
                chunks_csv = out_dir / "chunks.csv"
                try:
                    with open(chunks_csv, "w", newline="") as f:
                        writer = csv.DictWriter(
                            f, fieldnames=list(chunk_rows[0].keys()),
                        )
                        writer.writeheader()
                        writer.writerows(chunk_rows)
                    logger.info("chunks.csv written to %s (N=%d)",
                                chunks_csv, len(chunk_rows))
                except Exception:
                    logger.exception("failed to write chunks.csv")

            # --record-trace dump. trace.npz holds the obs→chunk pairing
            # (numerical only); per-chunk camera frames are written as
            # JPEGs under trace_images/ for visual review.
            if args.record_trace and trace_records:
                trace_npz = out_dir / "trace.npz"
                try:
                    # Discover the union of keys across records. Most should
                    # be present in every record (uniform env schema), but
                    # we guard with np.stack-with-fallback per key.
                    all_keys: list[str] = []
                    seen: set = set()
                    for r in trace_records:
                        for k in r:
                            if k not in seen:
                                seen.add(k)
                                all_keys.append(k)

                    bundles: dict = {}
                    for k in all_keys:
                        if k == "task":
                            bundles[k] = np.array(
                                [r.get(k, "") for r in trace_records], dtype=object,
                            )
                            continue
                        try:
                            arrs = [np.asarray(r[k]) for r in trace_records if k in r]
                            if len(arrs) == len(trace_records):
                                bundles[k] = np.stack(arrs)
                            else:
                                # Sparse key (not in every record); save as
                                # an object array of variable entries.
                                bundles[k] = np.array(
                                    [r.get(k) for r in trace_records], dtype=object,
                                )
                        except Exception:
                            logger.exception("trace: failed to stack key %r", k)

                    np.savez(trace_npz, **bundles)
                    logger.info("trace.npz written to %s (N=%d)",
                                trace_npz, len(trace_records))
                except Exception:
                    logger.exception("failed to write trace.npz")

                # Flush JPEGs. cv2 imports through crisp_py.camera already,
                # so this is just a re-use of the loaded module.
                if trace_images_buf:
                    try:
                        import cv2  # noqa: PLC0415
                        img_dir = out_dir / "trace_images"
                        img_dir.mkdir(exist_ok=True)
                        n_ok = 0
                        for fname, bgr in trace_images_buf:
                            path = img_dir / fname
                            if cv2.imwrite(
                                str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 85]
                            ):
                                n_ok += 1
                        logger.info(
                            "trace_images: %d JPEGs written to %s (of %d buffered)",
                            n_ok, img_dir, len(trace_images_buf),
                        )
                    except Exception:
                        logger.exception("failed to flush trace_images JPEGs")

            # Per-frame CSV from the sender thread. Pair with chunks.csv
            # to correlate producer-side stalls with sender-side underruns.
            sender_frame_rows = getattr(sender, "frame_rows", []) or []
            if sender_frame_rows:
                frames_csv = out_dir / "frames.csv"
                try:
                    with open(frames_csv, "w", newline="") as f:
                        writer = csv.DictWriter(
                            f, fieldnames=list(sender_frame_rows[0].keys()),
                        )
                        writer.writeheader()
                        writer.writerows(sender_frame_rows)
                    logger.info("frames.csv written to %s (N=%d)",
                                frames_csv, len(sender_frame_rows))
                except Exception:
                    logger.exception("failed to write frames.csv")
        except Exception:
            logger.exception("failed to write summary.json")

        # Shutdown shadow policy (frees GPU memory).
        if shadow_policy is not None:
            try:
                shadow_policy.shutdown()
                logger.info("shadow policy shut down")
            except Exception:
                logger.exception("shadow_policy.shutdown() raised")

        # Stop --save-video subprocesses (one per camera). SIGINT triggers
        # a clean rclcpp shutdown → cv::VideoWriter.release() with the
        # writer lock held, which flushes the mp4 trailer. Per-camera
        # stderr/stdout is in video_recorder_<name>.log inside out_dir;
        # tail those for frame counts / any open() failures. Independent
        # of the deploy process — the C++ binaries own the rclcpp
        # subscriptions, so the deploy loop was never on their critical
        # path.
        for rec in video_recorders:
            try:
                rec.stop(timeout=5.0)
            except Exception:
                logger.exception("video_recorder.stop() raised")
        if video_recorders:
            logger.info(
                "video recorders stopped: %d (see video_recorder_*.log)",
                len(video_recorders),
            )

        # Shutdown chunk source (real policy ⇒ joins inference subprocess;
        # fake source ⇒ no-op).
        try:
            chunk_source.shutdown()
            logger.info("chunk source shut down")
        except Exception:
            logger.exception("chunk_source.shutdown() raised")

        # Restore scaler (guarded against rclpy SIGINT shutdown — see
        # ReplayScaler.restore() docstring + troubleshooting doc).
        if scaler is not None:
            try:
                scaler.restore()
            except Exception:
                logger.exception("scaler.restore() raised")

        try:
            env.close()
        except Exception:
            logger.exception("env.close() raised")

        if rclpy.ok():
            rclpy.shutdown()

    if interrupted:
        return 130
    if failed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
