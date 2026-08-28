#!/usr/bin/env python3
"""Profiling version of ``16_ridgeback_mocap_record_fast.py`` (script 17).

Runs **exactly** like script 16 — same CLI, same mocap pipeline, same image
writer subclass, same recorded fields — but layers timing instrumentation
over the parent-side recording loop so we can localize the cause of
``"Frame processing took too long"`` warnings without guessing.

What it measures, per frame, in the **parent** process:

1. ``data_fn_ms`` — wall-clock duration of the whole ``data_fn`` call.
2. ``get_obs_ms`` — just the ``env.get_obs()`` subcall inside ``data_fn``.
3. ``queue_put_ms`` — wall-clock duration of ``self.queue.put(FRAME)``,
   i.e. the parent→writer-subprocess handoff. High values here mean the
   writer subprocess is not draining fast enough and the parent is
   backpressured.
4. ``queue_qsize_pre_put`` — the ``mp.JoinableQueue.qsize()`` reading taken
   immediately before each ``put()``. Near 0 means the writer is comfortably
   keeping up; near 16 (the ``queue_size`` default) means the parent is
   consistently at the backpressure limit.
5. ``frame_overshoot_ms`` — how much the overall per-frame wall clock
   exceeded the ``1 / fps`` budget. Only recorded for late frames, so
   ``count`` = number of late frames and mean/p95/max tell you how bad the
   spikes are.

What it does NOT profile:

- The writer subprocess itself. The writer runs in a separate ``mp.Process``
  spawned by the shared ``RecordingManager.__init__``. Instrumenting its
  message loop would require touching ``crisp_gym/record/recording_manager.py``,
  which this script explicitly avoids. You can still infer writer-side
  slowness from parent-side ``queue_put_ms`` + ``queue_qsize_pre_put``: both
  high → writer backpressure; both low → bottleneck is parent-side.
- Internals of ``env.get_obs()``. We measure the call as a single box. If
  it turns out to be dominant, a follow-up profiling pass inside the env
  (or a py-spy / ``cProfile`` over a single-frame call) is the next step.

How to read the output

At the end of every episode (regardless of save/discard) and at shutdown,
the script prints a summary table of all probes with count / min / mean /
p50 / p95 / p99 / max in milliseconds, plus a one-line **verdict** based on
simple thresholds. The verdict is a hint, not gospel — if the numbers are
ambiguous, trust the table.

Interpretation guide:

- ``data_fn_ms`` p95 high, ``queue_put_ms`` p95 low, ``qsize_pre_put`` mean
  near 0 → parent-side stall. The loop is slow *before* it even tries to
  hand a frame off. Likely inside ``env.get_obs()`` — follow up by profiling
  get_obs individually.
- ``queue_put_ms`` p95 high, ``qsize_pre_put`` p95 at or near 16 →
  writer-subprocess backpressure. The writer subprocess cannot drain the
  16-slot queue fast enough, so ``put()`` blocks. Likely inside
  ``dataset.add_frame`` (even with the image writer running — on a single
  small camera the add_frame bookkeeping can still dominate). Reducing
  image resolution, dropping cameras, or moving image copies to shared
  memory would be the usual responses.
- Both ``data_fn_ms`` and ``queue_put_ms`` low, but ``frame_overshoot_ms``
  still shows large overshoots → neither probe caught the stall. The
  wall clock is slipping somewhere we haven't instrumented — typically
  system-level (GC, SHM transport failure, UDP packet drops). Check the
  startup banner for ``[RTPS_TRANSPORT_SHM Error]`` and verify
  ``net.core.rmem_max`` is large (the activation script warns about this).

Usage (same as 16):

    python examples/17_ridgeback_mocap_record_profile.py \\
        --repo-id fps_profile_001 \\
        --task "fps profile" \\
        --num-episodes 1 \\
        --require-mocap

The script will record one episode, save or discard it via the usual
``s`` / ``d`` keys, print the per-episode profile table, then shut down and
print a final summary covering all episodes combined.

Nothing in ``crisp_gym/record/`` is modified; 12/13/15/16/spacemouse
recorders are unaffected by this file.
"""

import argparse
import logging
import statistics
import threading
import time
from collections import defaultdict
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
# any other libjpeg can shadow it. Same ordering as script 16.
from crisp_gym.envs.manipulator_env import make_env

# Safe to pull in lerobot-touching modules now that cv2 is bound. We
# re-export HF_LEROBOT_HOME and LeRobotDataset from recording_manager's
# namespace rather than importing them from lerobot directly.
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


# --------------------------------------------------------------------- #
# Profile stats
# --------------------------------------------------------------------- #

class ProfileStats:
    """Thread-safe parent-process sample collector for timing probes.

    Stores raw samples per named probe and produces a summary table on
    demand. All samples are plain floats (typically milliseconds, but this
    class is agnostic — it just runs min/mean/percentile/max over whatever
    you give it).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._samples: dict[str, list[float]] = defaultdict(list)
        self._frame_count: int = 0
        self._late_frame_count: int = 0

    def record(self, name: str, value: float) -> None:
        with self._lock:
            self._samples[name].append(value)

    def mark_frame(self, late: bool) -> None:
        with self._lock:
            self._frame_count += 1
            if late:
                self._late_frame_count += 1

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()
            self._frame_count = 0
            self._late_frame_count = 0

    def summary(self, title: str = "Profile summary") -> str:
        with self._lock:
            samples = {k: list(v) for k, v in self._samples.items()}
            frame_count = self._frame_count
            late_frame_count = self._late_frame_count

        if not samples:
            return f"\n=== {title} ===\n(no samples collected)\n"

        def pct(vals: list[float], q: float) -> float:
            if not vals:
                return float("nan")
            if len(vals) == 1:
                return vals[0]
            # statistics.quantiles with n=100 gives 99 cut points; index q-1.
            qs = statistics.quantiles(vals, n=100, method="inclusive")
            idx = max(0, min(len(qs) - 1, int(round(q)) - 1))
            return qs[idx]

        # Probe ordering: data_fn first, then its inner get_obs, then the
        # queue side, then the overshoot bucket. Keeps the table readable.
        preferred_order = [
            "data_fn_ms",
            "get_obs_ms",
            "queue_put_ms",
            "queue_qsize_pre_put",
            "frame_overshoot_ms",
        ]
        ordered_names = [n for n in preferred_order if n in samples] + [
            n for n in samples if n not in preferred_order
        ]

        header = (
            f"{'probe':<22}{'count':>8}{'min':>10}{'mean':>10}"
            f"{'p50':>10}{'p95':>10}{'p99':>10}{'max':>10}"
        )
        rows = [header, "-" * len(header)]
        for name in ordered_names:
            vals = samples[name]
            if not vals:
                continue
            if len(vals) == 1:
                row = (
                    f"{name:<22}{len(vals):>8}"
                    f"{vals[0]:>10.2f}{vals[0]:>10.2f}"
                    f"{vals[0]:>10.2f}{vals[0]:>10.2f}{vals[0]:>10.2f}{vals[0]:>10.2f}"
                )
            else:
                row = (
                    f"{name:<22}{len(vals):>8}"
                    f"{min(vals):>10.2f}{statistics.fmean(vals):>10.2f}"
                    f"{pct(vals, 50):>10.2f}{pct(vals, 95):>10.2f}"
                    f"{pct(vals, 99):>10.2f}{max(vals):>10.2f}"
                )
            rows.append(row)

        # Top-line stats
        effective_fps_note = ""
        if "data_fn_ms" in samples and frame_count > 0:
            # Rough effective FPS assuming loop wall-clock ≈ max(data_fn, 1/fps)
            mean_loop_ms = statistics.fmean(samples["data_fn_ms"]) + statistics.fmean(
                samples.get("queue_put_ms", [0.0])
            )
            if mean_loop_ms > 0:
                effective_fps_note = (
                    f"\nApprox. effective FPS (1 / (mean data_fn + mean queue_put)): "
                    f"{1000.0 / mean_loop_ms:.2f}"
                )

        late_pct = (
            100.0 * late_frame_count / frame_count if frame_count > 0 else 0.0
        )
        summary_lines = [
            f"\n=== {title} ===",
            f"Frames: {frame_count}",
            f"Late frames: {late_frame_count} ({late_pct:.1f}%)",
            *rows,
            effective_fps_note,
            self._verdict(samples),
        ]
        return "\n".join(line for line in summary_lines if line != "")

    def _verdict(self, samples: dict[str, list[float]]) -> str:
        """Interpret the probe distributions and print a one-line hint."""
        def p95(name: str) -> float:
            vals = samples.get(name)
            if not vals:
                return 0.0
            if len(vals) == 1:
                return vals[0]
            qs = statistics.quantiles(vals, n=100, method="inclusive")
            return qs[94]

        def mean(name: str) -> float:
            vals = samples.get(name)
            if not vals:
                return 0.0
            return statistics.fmean(vals)

        data_fn_p95 = p95("data_fn_ms")
        get_obs_p95 = p95("get_obs_ms")
        put_p95 = p95("queue_put_ms")
        qsize_mean = mean("queue_qsize_pre_put")
        overshoot_p95 = p95("frame_overshoot_ms")

        verdict_lines = ["\n=== Verdict ==="]

        if data_fn_p95 < 5.0 and put_p95 < 5.0 and overshoot_p95 > 50.0:
            verdict_lines.append(
                "Both probes clean but frames still overshoot — the stall is "
                "NOT in data_fn or queue.put. Suspect system-level (SHM / UDP "
                "rmem_max / GC / thermal). Check startup banner for "
                "[RTPS_TRANSPORT_SHM Error] and verify net.core.rmem_max is "
                "large."
            )
        elif put_p95 > max(data_fn_p95, 20.0) and qsize_mean > 8.0:
            verdict_lines.append(
                "WRITER-BOUND: queue_put p95 dominates AND mp queue is "
                "persistently near full. The writer subprocess is the "
                "bottleneck — even with the image writer, dataset.add_frame "
                "can't drain fast enough. Reduce payload size (smaller "
                "resolution, fewer features) or profile _writer_proc."
            )
        elif data_fn_p95 > max(put_p95, 20.0):
            if get_obs_p95 > 0.6 * data_fn_p95:
                verdict_lines.append(
                    "PARENT-BOUND: data_fn p95 dominates AND env.get_obs() is "
                    "most of that. The stall is on the parent side, before "
                    "the frame is even handed off. Likely ROS callback "
                    "contention or a blocking camera read inside get_obs. "
                    "Next step: instrument get_obs internals (cameras, "
                    "robot state, gripper state separately)."
                )
            else:
                verdict_lines.append(
                    "PARENT-BOUND: data_fn p95 dominates, but env.get_obs() "
                    "is only part of it. Something else in data_fn (pose "
                    "conversion, numpy concat, mocap_capture.get()) is slow "
                    "— unusual but worth instrumenting further."
                )
        elif overshoot_p95 < 10.0:
            verdict_lines.append(
                "Looks healthy. Loop is consistently within budget."
            )
        else:
            verdict_lines.append(
                "Mixed signal — no single probe clearly dominates. Inspect "
                "the table above, look at which p95 / p99 is largest, and "
                "run a longer capture if samples are sparse."
            )

        return "\n".join(verdict_lines)


# --------------------------------------------------------------------- #
# Mocap helpers (verbatim from 16)
# --------------------------------------------------------------------- #

class MocapTargetCapture:
    """Thread-safe holder for the latest ``/target_pose`` message from mocap."""

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
    """Neuter the env's own ``/target_pose`` and ``target_joint`` publishers."""

    def _noop(_msg):  # noqa: ANN001
        return None

    if env.robot._target_pose_publisher is not None:
        env.robot._target_pose_publisher.publish = _noop
    env.robot._target_joint_publisher.publish = _noop


# --------------------------------------------------------------------- #
# Recording manager: image writer + parent-side instrumentation
# --------------------------------------------------------------------- #

class FastKeyboardRecordingManager(KeyboardRecordingManager):
    """Keyboard recording manager with the LeRobot image writer enabled.

    Verbatim copy of the 16_… class. See
    ``docs/ridgeback_mocap_record_fps_image_writer.md`` for the rationale.
    Kept local so 17 is self-contained and does not depend on 16's file.
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
                    f"The repo_id already exists. If you intended to resume "
                    f"the collection of data, then execute this script with "
                    f"the --resume flag. Otherwise remove it:\n"
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
            )

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


class InstrumentedFastKeyboardRecordingManager(FastKeyboardRecordingManager):
    """Wraps ``record_episode`` to time ``queue.put`` and observe qsize.

    The instrumentation is a thin shim:

    - ``data_fn`` is swapped for a closure that times its wall clock and
      records the sample, then returns whatever the real ``data_fn`` did.
    - ``self.queue.put`` is monkey-patched on the instance for the duration
      of the episode. The patched version reads ``qsize()`` before the
      call, times the ``put()``, and records both samples.

    Restored on exit so the manager can be reused.
    """

    def __init__(
        self,
        *,
        config: RecordingManagerConfig | None = None,
        profile_stats: ProfileStats,
        **kwargs,  # noqa: ANN003
    ) -> None:
        # Must set before super().__init__ so the attribute exists on self
        # at the moment the writer subprocess forks (it inherits a copy of
        # self — the child never touches _profile_stats, but defensive).
        self._profile_stats = profile_stats
        super().__init__(config=config, **kwargs)

    def record_episode(
        self,
        data_fn,  # noqa: ANN001
        task: str,
        on_start=None,  # noqa: ANN001
        on_end=None,  # noqa: ANN001
    ) -> None:
        stats = self._profile_stats

        def timed_data_fn():
            t0 = time.perf_counter()
            result = data_fn()
            stats.record("data_fn_ms", (time.perf_counter() - t0) * 1000.0)
            return result

        original_put = self.queue.put

        def timed_put(msg, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
            # qsize() on mp queues is approximate on some platforms but
            # reliable enough for orders-of-magnitude trend analysis.
            try:
                qsize = self.queue.qsize()
            except (NotImplementedError, OSError):
                qsize = -1.0
            stats.record("queue_qsize_pre_put", float(qsize))
            t0 = time.perf_counter()
            result = original_put(msg, *args, **kwargs)
            stats.record("queue_put_ms", (time.perf_counter() - t0) * 1000.0)

            # Any overshoot is computed by the shared record_episode via
            # frame_start, but we also want it in the stats for the verdict
            # logic. We can't observe it here directly — it's in the
            # parent's loop. Instead we reconstruct it below via a monkey
            # patch on the warning call isn't worth the complexity. The
            # shared code already logs overshoot as a warning; we capture
            # those via the logger instead of intercepting here.
            return result

        self.queue.put = timed_put  # type: ignore[method-assign]
        try:
            super().record_episode(
                data_fn=timed_data_fn, task=task, on_start=on_start, on_end=on_end
            )
        finally:
            self.queue.put = original_put  # type: ignore[method-assign]


# --------------------------------------------------------------------- #
# Frame-overshoot log scraper
# --------------------------------------------------------------------- #

class OvershootLogHandler(logging.Handler):
    """Scrape ``"Frame processing took too long"`` warnings from the shared
    recording loop and feed them into ProfileStats.

    The shared ``recording_manager.record_episode`` emits a warning at
    ``recording_manager.py:319`` with text of the form::

        Frame processing took too long: 0.123 seconds too long i.e. 5.78 FPS.

    Parsing that message gives us the per-frame overshoot in seconds (and
    implicitly marks the frame as late). Zero shared-code changes needed.
    """

    def __init__(self, profile_stats: ProfileStats) -> None:
        super().__init__(level=logging.WARNING)
        self._stats = profile_stats

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001
            return
        if "Frame processing took too long" not in msg:
            return
        # "... too long: 0.123 seconds too long i.e. 5.78 FPS."
        try:
            after_colon = msg.split("took too long:", 1)[1]
            seconds_str = after_colon.strip().split()[0]
            overshoot_ms = float(seconds_str) * 1000.0
        except (IndexError, ValueError):
            return
        self._stats.record("frame_overshoot_ms", overshoot_ms)
        self._stats.mark_frame(late=True)

    # Defensive: never let the handler itself raise.
    def handleError(self, record: logging.LogRecord) -> None:  # noqa: D401
        return


# --------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="ridgeback_mocap_profile",
        help="Repository ID for the LeRobot dataset",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="fps profile run",
        help="Task description label for all episodes",
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument(
        "--push-to-hub", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--env-config", type=str, default="ur10e_ridgeback_env")
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
        help="LeRobot image writer worker process count. Matches 16_... default.",
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=1,
        help="LeRobot image writer threads per worker process.",
    )
    args = parser.parse_args()

    setup_logging(level=args.log_level)

    profile_stats = ProfileStats()

    # Attach the overshoot log scraper to the root logger so it sees the
    # warning emitted by the shared recording_manager module (regardless of
    # which logger name it uses).
    overshoot_handler = OvershootLogHandler(profile_stats)
    logging.getLogger().addHandler(overshoot_handler)

    logger.info(f"Creating environment: {args.env_config}")
    env = make_env(env_type=args.env_config, control_type="cartesian", namespace="")

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
    rec_config.image_writer_processes = args.image_writer_processes
    rec_config.image_writer_threads = args.image_writer_threads

    recording_manager = InstrumentedFastKeyboardRecordingManager(
        config=rec_config,
        profile_stats=profile_stats,
    )
    recording_manager.wait_until_ready()
    logger.info("Recording manager ready.")

    logger.info("Homing robot...")
    env.home()
    env.reset()

    def data_fn():
        # Wrap env.get_obs() separately so we can see whether it dominates
        # data_fn_ms. Everything else in this function is cheap attribute
        # reads and numpy concatenation, so we don't bother timing those.
        t0 = time.perf_counter()
        obs = env.get_obs()
        profile_stats.record("get_obs_ms", (time.perf_counter() - t0) * 1000.0)

        mocap_pose = mocap_capture.get()
        if mocap_pose is not None:
            target_pose = mocap_pose.to_array(
                representation=env.config.orientation_representation
            ).astype(np.float32)
        else:
            target_pose = env.robot.current_pose.to_array(
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
        # Print and reset per-episode stats. Do this AFTER home() so the
        # summary lands after the episode is actually wrapped up.
        print(profile_stats.summary(title=f"Episode profile"))
        profile_stats.reset()

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
        # If stats have been reset in on_end() for every episode this will
        # just be an empty final summary — that's fine. If the run was
        # aborted mid-episode this will show what we did capture.
        print(profile_stats.summary(title="Final profile (post-shutdown)"))
        logging.getLogger().removeHandler(overshoot_handler)
        env.close()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
