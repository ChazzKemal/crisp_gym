"""Bringing the robot up, in the order the hardware requires.

Moved out of ``main()`` in ``examples/19_deploy_policy.py``. Almost none of this is
logic -- the work each phase does already lives in :mod:`crisp_gym.deploy` and has
been verified there. What this module contributes is the *ordering*, which is the
part that is genuinely load-bearing:

* the controller must be switched before any target is published, or the commands
  go to a controller that is not listening;
* gains must be scaled before the first chunk, not during it;
* the sender must be started, and its subscriber matched, before the producer
  begins -- the reason ``--startup-delay`` exists;
* camera and gripper subscriptions are destroyed *after* the env is ready but
  *before* the loop, because the point is to stop them competing for the GIL with
  inference, not to avoid constructing them.

Each phase is a plain function over ``(env, args)`` so a different runner can
compose the same sequence without inheriting the CLI it came from.
"""

from __future__ import annotations

import logging
import queue
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32, Float64MultiArray

from crisp_gym.deploy.cpp_sender import CppSenderHandle
from crisp_gym.deploy.dataset import LEROBOT_CACHE
from crisp_gym.deploy.gains import (
    GRIPPER_MAX_SPEED_MPS,
    SPEED_CMDS_TOPIC,
    ReplayScaler,
    _spawn_gripper_speed_controller,
)
from crisp_gym.deploy.patches import (
    enable_target_pose_publishing,
    fix_gripper_self_subscription,
)
from crisp_gym.deploy.sender import TargetSenderThread
from crisp_gym.deploy.video import _VideoRecorder, _find_crisp_video_recorder_binary
from crisp_gym.envs.manipulator_env import make_env

logger = logging.getLogger(__name__)


@dataclass
class PublishChannels:
    """Everything the sender needs in order to talk to the robot.

    Gathered into one object because these eight values are produced together by a
    single phase and consumed together by the next one; passing them individually
    was only ever an artefact of living in one function's scope.
    """

    base_frame_id: str = ""
    target_pose_pub: Any = None
    pose_msg: Any = None
    gripper_raw_pub: Any = None
    gripper_action_client: Any = None
    gripper_max_effort: float = 0.0
    gripper_unnormalize_fn: Any = None
    gripper_enabled: bool = False
    py_pose_pub: Any = None


def build_env(args):
    """Construct the env, apply the two mandatory patches, wait for readiness."""
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
    return env


def phase_home(env, args):
    """Phase 1 -- home the arm, unless --offline. THIS MOVES THE ROBOT."""
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


def phase_switch_controller(env, args):
    """Phase 2 -- switch to the cartesian controller. Nothing may be published before this."""
    # ---- Phase 2: switch to cartesian ----
    if args.offline:
        logger.info(
            "Phase 2: SKIPPED (offline — no controller_manager to switch)"
        )
    else:
        logger.info("Phase 2: switching to cartesian controller")
        env.switch_controller("cartesian")


def phase_scaler(env, args):
    """Phase 2b -- controller gain scaling. Returns None when --scale-kp is off."""
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
    return scaler


def phase_pin_gripper_speed(env, args):
    """Phase 2b' -- pin the gripper's speed_limit to the driver maximum."""
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


def phase_gil_hygiene(env, args):
    """Phase 2c -- drop subscriptions that would compete with inference for the GIL."""
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


def phase_publish_channels(env, args):
    """Phase 3 setup -- take ownership of the publishers the sender will drive."""
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

    return PublishChannels(
        base_frame_id=base_frame_id,
        target_pose_pub=target_pose_pub,
        pose_msg=pose_msg,
        gripper_raw_pub=gripper_raw_pub,
        gripper_action_client=gripper_action_client,
        gripper_max_effort=gripper_max_effort,
        gripper_unnormalize_fn=gripper_unnormalize_fn,
        gripper_enabled=gripper_enabled,
        py_pose_pub=py_pose_pub,
    )


def phase_start_sender(env, args, scaler, ch):
    """Phase 3 -- start the sender: a Python thread, or the C++ subprocess."""
    # Unpacked so the moved body reads exactly as it did inside main().
    base_frame_id = ch.base_frame_id
    target_pose_pub = ch.target_pose_pub
    pose_msg = ch.pose_msg
    gripper_raw_pub = ch.gripper_raw_pub
    gripper_action_client = ch.gripper_action_client
    gripper_max_effort = ch.gripper_max_effort
    gripper_enabled = ch.gripper_enabled
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
    return sender, q


def phase_video_and_delay(env, args, n_obs, n_act):
    """Phase 3a/3b -- video recorders, then the DDS subscriber-match delay."""
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
    return run_started_at, out_dir, video_recorders
