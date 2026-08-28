#!/usr/bin/env python3
"""Async-camera version of ``17_ridgeback_mocap_record_profile.py`` (script 18).

Same mocap flow, same image writer, same profiler as 17 — but the camera
feed is decoupled from the env's ROS executor using the pattern LeRobot
upstream uses for cameras in its recording loop (``async_read``). See
``huggingface/lerobot`` issue #652 for the upstream symptom (high-rate
control loop dragged down to camera rate when a camera is added), and
``docs/ridgeback_mocap_record_fps_image_writer.md`` + the earlier profile
runs on this tree for the local investigation.

Why 17 still shows slow frames in a healthy env
-----------------------------------------------

``crisp_py.Camera`` subscribes to ``/camera/color/image_raw/compressed`` and
does ``cv_bridge.compressed_imgmsg_to_cv2`` + ``cv2.resize`` **inside the
ROS callback thread** (``crisp_py/camera/camera.py:162-166`` and ``:260``).
That thread runs in the same Python process as the recording loop and
shares the GIL. cv_bridge in particular has Python-level marshalling that
holds the GIL for a noticeable fraction of each callback. At ~30 Hz of
1280x720 JPEG work the callback burns ~300 ms/s of Python-level GIL time,
and when a burst lands on top of a ``data_fn`` iteration the main thread
stalls for 100-250 ms — exactly the bimodal pattern we observed in 17's
profile data.

The fix this script applies
---------------------------

1. Let the env come up normally (including ``crisp_py.Camera`` getting its
   first image through its own subscription) so ``env.wait_until_ready()``
   passes.
2. Immediately **destroy** ``env.cameras[0]._camera_subscriber``. This
   stops the env's camera callback from running entirely, eliminating the
   cv_bridge + resize work from the main ROS executor.
3. Spin up ``AsyncCameraSubscriber`` — a brand-new ``rclpy`` node with its
   own ``SingleThreadedExecutor``, running on its own ``threading.Thread``.
   The subscriber callback uses ``cv2.imdecode`` directly (no cv_bridge),
   which drops to C code and releases the GIL for the decode. It stores
   the latest frame under a lock and exposes ``get()``.
4. In ``data_fn``, call ``env.get_obs()`` as before for robot / joints /
   gripper / task, but **overwrite** the ``observation.images.<camera>``
   entry with the fresh frame from the async subscriber.

The "dedicated thread" part isn't magic — it still contends for the GIL.
The point is that ``cv2.imdecode`` and ``cv2.cvtColor`` + ``cv2.resize``
release the GIL around their C paths, so the total GIL hold per callback
drops from ~10-15 ms (cv_bridge + resize) to <1 ms (thin C wrappers).
With a total GIL-held cost of ~0.5 ms/frame × 30 fps = 15 ms/s of
contention instead of 300 ms/s, the main recording loop runs without
being pushed off its budget on most frames.

Trade-offs and limits
---------------------

- We completely bypass ``env.cameras[0]``'s image pipeline. The env's
  ``_current_image`` is frozen at whatever was stored just before the
  ``destroy_subscription`` call. Nothing else in the recorder reads it.
- Our ``AsyncCameraSubscriber`` replicates the aspect-ratio-preserving
  resize-with-crop that ``crisp_py.Camera._resize_with_aspect_ratio`` does
  (``crisp_py/camera/camera.py:241-267``), so the saved images have the
  same shape and framing as previous recordings. The recorder's LeRobot
  schema (built from ``env.cameras[0].config.resolution``) is unchanged.
- This does NOT fix the upstream ``crisp_py`` issue for every consumer
  of the library. It is a local workaround scoped to this recording
  script. A permanent fix belongs in ``crisp_py.Camera`` — that is a
  shared-code change and out of scope here.

Everything else (CLI args, mocap plumbing, profiler, image writer, etc.)
is identical to 17. This script prints the same profile tables at episode
end and shutdown, so comparing 17 vs 18 on the same machine is the
verification procedure.

Usage (same as 17, with the same keys):

    python examples/18_ridgeback_mocap_record_async_camera.py \\
        --repo-id ridgeback_profile_async_camera \\
        --task "fps profile async camera" \\
        --num-episodes 1 \\
        --require-mocap

Keyboard controls (episode management):
    r  →  start recording
    r  →  stop / pause recording
    s  →  save episode
    d  →  discard episode
    q  →  quit
"""

import argparse
import io  # noqa: F401 — reserved for optional PIL fallback
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
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage

# IMPORT ORDER MATTERS. `crisp_gym.envs.manipulator_env` must be imported
# BEFORE anything that pulls in torch / torchvision / lerobot. See 17's
# header comment for the full libjpeg / libtiff / cv2 explanation. Keeping
# the same ordering here.
from crisp_gym.envs.manipulator_env import make_env

# Safe to pull in cv2 directly now — the env chain already loaded it via
# crisp_py.camera, so libjpeg has been bound against the conda env's copy.
# We use it in the hot path of AsyncCameraSubscriber._on_msg.
import cv2  # noqa: E402

# lerobot-touching modules after make_env.
from crisp_gym.record.recording_manager import (  # noqa: E402
    HF_LEROBOT_HOME,
    KeyboardRecordingManager,
    LeRobotDataset,
)
from crisp_gym.record.recording_manager_config import RecordingManagerConfig  # noqa: E402
from crisp_gym.util.lerobot_features import get_features  # noqa: E402
from crisp_gym.util.setup_logger import setup_logging  # noqa: E402
from crisp_py.utils.geometry import Pose  # noqa: E402

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Profile stats (verbatim from 17)
# --------------------------------------------------------------------- #

class ProfileStats:
    """Thread-safe parent-process sample collector for timing probes."""

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
            late_frame_count = self._late_frame_count

        if not samples:
            return f"\n=== {title} ===\n(no samples collected)\n"

        def pct(vals: list[float], q: float) -> float:
            if not vals:
                return float("nan")
            if len(vals) == 1:
                return vals[0]
            qs = statistics.quantiles(vals, n=100, method="inclusive")
            idx = max(0, min(len(qs) - 1, int(round(q)) - 1))
            return qs[idx]

        preferred_order = [
            "data_fn_ms",
            "get_obs_ms",
            "camera_get_ms",
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
        total_frames_row = 0
        for name in ordered_names:
            vals = samples[name]
            if not vals:
                continue
            if name == "data_fn_ms":
                total_frames_row = len(vals)
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

        effective_fps_note = ""
        if "data_fn_ms" in samples and total_frames_row > 0:
            mean_loop_ms = statistics.fmean(samples["data_fn_ms"]) + statistics.fmean(
                samples.get("queue_put_ms", [0.0])
            )
            if mean_loop_ms > 0:
                effective_fps_note = (
                    f"\nApprox. effective FPS (1 / (mean data_fn + mean queue_put)): "
                    f"{1000.0 / mean_loop_ms:.2f}"
                )

        late_pct = (
            100.0 * late_frame_count / total_frames_row
            if total_frames_row > 0
            else 0.0
        )
        summary_lines = [
            f"\n=== {title} ===",
            f"Frames recorded: {total_frames_row}",
            f"Late frames: {late_frame_count} ({late_pct:.1f}%)",
            *rows,
            effective_fps_note,
            self._verdict(samples),
        ]
        return "\n".join(line for line in summary_lines if line != "")

    def _verdict(self, samples: dict[str, list[float]]) -> str:
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
        camera_get_p95 = p95("camera_get_ms")
        put_p95 = p95("queue_put_ms")
        qsize_mean = mean("queue_qsize_pre_put")
        overshoot_p95 = p95("frame_overshoot_ms")

        lines = ["\n=== Verdict ==="]

        if data_fn_p95 < 5.0 and put_p95 < 5.0 and overshoot_p95 > 50.0:
            lines.append(
                "Both data_fn and queue.put are fast but frames still overshoot. "
                "The stall is somewhere between them — possibly GC or a system "
                "event the probes don't cover."
            )
        elif put_p95 > max(data_fn_p95, 20.0) and qsize_mean > 8.0:
            lines.append(
                "WRITER-BOUND: queue_put p95 dominates AND mp queue is near full. "
                "Writer subprocess cannot drain fast enough."
            )
        elif data_fn_p95 > max(put_p95, 20.0):
            if camera_get_p95 > 0.6 * data_fn_p95:
                lines.append(
                    "CAMERA-BOUND: camera_get p95 dominates data_fn. The async "
                    "subscriber's lock or the frame handoff is the bottleneck "
                    "— unusual, worth inspecting AsyncCameraSubscriber."
                )
            elif get_obs_p95 > 0.6 * data_fn_p95:
                lines.append(
                    "PARENT-BOUND inside env.get_obs(): robot state / joint "
                    "reads / gripper / something NOT the camera. Async camera "
                    "fix cleared the camera path but something else in the "
                    "env observation pipeline is slow. Next step: per-sub- "
                    "component timing inside get_obs()."
                )
            else:
                lines.append(
                    "PARENT-BOUND but neither camera nor get_obs dominates. "
                    "Something in data_fn itself — pose conversion, numpy "
                    "concat, or mocap_capture.get() — is the bottleneck."
                )
        elif overshoot_p95 < 10.0:
            lines.append("Looks healthy. Loop is consistently within budget.")
        else:
            lines.append(
                "Mixed signal — no single probe clearly dominates. Inspect "
                "the table above directly."
            )

        return "\n".join(lines)


# --------------------------------------------------------------------- #
# Overshoot log scraper (verbatim from 17)
# --------------------------------------------------------------------- #

class OvershootLogHandler(logging.Handler):
    """Scrape ``"Frame processing took too long"`` warnings into ProfileStats."""

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
        try:
            after_colon = msg.split("took too long:", 1)[1]
            seconds_str = after_colon.strip().split()[0]
            overshoot_ms = float(seconds_str) * 1000.0
        except (IndexError, ValueError):
            return
        self._stats.record("frame_overshoot_ms", overshoot_ms)
        self._stats.mark_frame(late=True)

    def handleError(self, record: logging.LogRecord) -> None:  # noqa: D401
        return


# --------------------------------------------------------------------- #
# Mocap helpers (verbatim from 17)
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
# Async camera subscriber — the actual fix
# --------------------------------------------------------------------- #

def _resize_with_aspect_crop(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Replica of ``crisp_py.Camera._resize_with_aspect_ratio`` without the
    pre-crop arguments (we don't use them in this env).

    Short-circuits when the incoming image already matches the target shape.
    Otherwise scales by the larger of the two axis ratios (so the scaled
    image covers the target box) then center-crops to the exact target.
    Keeps the saved image visually consistent with earlier recordings that
    went through the crisp_py path.
    """
    h, w = img.shape[:2]
    if h == target_h and w == target_w:
        return img
    scale = max(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    start_x = (new_w - target_w) // 2
    start_y = (new_h - target_h) // 2
    return resized[start_y : start_y + target_h, start_x : start_x + target_w]


class AsyncCameraSubscriber:
    """Latest-frame camera subscriber on a dedicated rclpy executor + thread.

    Mirrors the ``async_read`` pattern LeRobot upstream uses for cameras in
    its recording loop: a background reader produces frames, the main loop
    consumes the latest reference without blocking on I/O or decode.

    Decode uses ``cv2.imdecode`` directly (not ``cv_bridge``) because OpenCV
    releases the Python GIL for most of the JPEG decode path, which cuts
    the per-callback GIL-held time by an order of magnitude vs. cv_bridge.
    """

    def __init__(
        self,
        topic: str,
        target_hw: tuple[int, int] | None = None,
        node_name: str = "async_camera_subscriber",
    ) -> None:
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._count = 0
        self._target_hw = target_hw  # (H, W) — matches crisp_py.Camera convention
        self._stop = threading.Event()

        if not rclpy.ok():
            rclpy.init()
        self._node = rclpy.create_node(node_name)
        self._sub = self._node.create_subscription(
            CompressedImage,
            topic,
            self._on_msg,
            qos_profile_sensor_data,
        )
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)

        self._thread = threading.Thread(
            target=self._spin, name="async_camera_spin", daemon=True
        )
        self._thread.start()

        logger.info(
            f"AsyncCameraSubscriber: subscribed to {topic} on node "
            f"{node_name} (target_hw={target_hw})"
        )

    def _spin(self) -> None:
        while not self._stop.is_set() and rclpy.ok():
            try:
                self._executor.spin_once(timeout_sec=0.1)
            except Exception:  # noqa: BLE001
                logger.exception("AsyncCameraSubscriber executor error")
                time.sleep(0.1)

    def _on_msg(self, msg: CompressedImage) -> None:
        # cv2.imdecode handles JPEG / PNG / etc. based on magic bytes.
        # Releases the GIL for the hot decode path.
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img_bgr is None:
            logger.warning("AsyncCameraSubscriber: cv2.imdecode returned None")
            return
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if self._target_hw is not None:
            target_h, target_w = self._target_hw
            img_rgb = _resize_with_aspect_crop(img_rgb, target_h, target_w)

        with self._lock:
            self._frame = img_rgb
            self._count += 1

    def get(self) -> np.ndarray | None:
        """Return a reference (no copy) to the latest decoded frame."""
        with self._lock:
            return self._frame

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    def wait_until_ready(self, timeout: float = 5.0) -> None:
        deadline = time.time() + timeout
        while self.count == 0 and time.time() < deadline:
            time.sleep(0.05)
        if self.count == 0:
            raise TimeoutError(
                f"AsyncCameraSubscriber: no frame arrived within {timeout:.1f}s. "
                "Is the camera topic being published?"
            )

    def stop(self) -> None:
        self._stop.set()
        try:
            self._thread.join(timeout=2.0)
        except Exception:  # noqa: BLE001
            pass
        try:
            self._executor.shutdown()
        except Exception:  # noqa: BLE001
            pass
        try:
            self._node.destroy_node()
        except Exception:  # noqa: BLE001
            pass


def bypass_env_camera_subscription(env) -> tuple[str, str, tuple[int, int]] | None:
    """Destroy ``env.cameras[0]._camera_subscriber`` and return its topic.

    After this call the env's camera subscriber no longer receives messages
    and the callback thread no longer does cv_bridge / resize work on the
    env's ROS executor. ``env.cameras[0].current_image`` returns whatever
    frame was stored just before this call, which is fine because the
    caller is expected to overwrite the ``observation.images.<name>`` entry
    in the obs dict with a fresh frame from its own subscriber.

    Returns (compressed_topic, image_key, target_hw) so the caller knows
    what to subscribe to, which obs dict key to overwrite, and what
    resolution to deliver — all three read from the existing env objects
    so there's nothing to keep in sync by hand.
    """
    if not env.cameras:
        logger.warning("bypass_env_camera_subscription: env has no cameras")
        return None
    camera = env.cameras[0]
    sub = camera._camera_subscriber
    topic = sub.topic_name
    image_key = f"observation.images.{camera.config.camera_name}"
    target_hw = tuple(camera.config.resolution)  # (H, W) per crisp_py convention
    logger.info(
        f"Destroying env camera subscription on {topic} — async subscriber "
        f"will take over. Image key: {image_key}, target_hw: {target_hw}"
    )
    camera.node.destroy_subscription(sub)
    return topic, image_key, target_hw


# --------------------------------------------------------------------- #
# Recording manager — same as 17 (image writer + instrumentation)
# --------------------------------------------------------------------- #

class FastKeyboardRecordingManager(KeyboardRecordingManager):
    """Keyboard recording manager with the LeRobot image writer enabled.

    Verbatim copy of 16/17's override. Kept local so this file is
    self-contained.
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
    """Wraps ``record_episode`` to time ``queue.put`` and observe qsize."""

    def __init__(
        self,
        *,
        config: RecordingManagerConfig | None = None,
        profile_stats: ProfileStats,
        **kwargs,  # noqa: ANN003
    ) -> None:
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
            try:
                qsize = self.queue.qsize()
            except (NotImplementedError, OSError):
                qsize = -1.0
            stats.record("queue_qsize_pre_put", float(qsize))
            t0 = time.perf_counter()
            result = original_put(msg, *args, **kwargs)
            stats.record("queue_put_ms", (time.perf_counter() - t0) * 1000.0)
            return result

        self.queue.put = timed_put  # type: ignore[method-assign]
        try:
            super().record_episode(
                data_fn=timed_data_fn, task=task, on_start=on_start, on_end=on_end
            )
        finally:
            self.queue.put = original_put  # type: ignore[method-assign]


# --------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--repo-id", type=str, default="ridgeback_mocap_async_camera")
    parser.add_argument("--task", type=str, default="fps async camera run")
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
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--async-camera-timeout",
        type=float,
        default=5.0,
        help="Seconds to wait for the async camera subscriber to receive "
             "its first frame before aborting.",
    )
    args = parser.parse_args()

    setup_logging(level=args.log_level)

    profile_stats = ProfileStats()
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

    # --------- async camera setup ---------
    # IMPORTANT: env.wait_until_ready() above already subscribed the env's
    # camera and waited for its first frame, so crisp_py.Camera's internal
    # _current_image is populated and env.observation_space has the image
    # feature baked in. We destroy that subscription next, and then drop the
    # camera from env.cameras entirely so _get_obs() no longer reads
    # camera.current_image — which, after the subscriber is gone, would
    # emit a "image data is stale" warn on every tick (rclpy logger + stderr
    # flush from the main thread was costing 70 ms mean / 290 ms p95 inside
    # get_obs()). data_fn overwrites observation.images.<name> from the
    # async subscriber, and get_features() reads env.observation_space (not
    # env.cameras), so the dataset schema is unaffected.
    bypass = bypass_env_camera_subscription(env)
    if bypass is None:
        raise RuntimeError(
            "18_…_async_camera.py requires an env with at least one camera. "
            "The ur10e_ridgeback_env yaml has one, so this should not happen."
        )
    compressed_topic, image_key, target_hw = bypass
    env.cameras = []

    async_camera = AsyncCameraSubscriber(
        topic=compressed_topic,
        target_hw=target_hw,
    )
    try:
        async_camera.wait_until_ready(timeout=args.async_camera_timeout)
    except TimeoutError:
        logger.exception(
            "Async camera subscriber never received a frame. Aborting."
        )
        async_camera.stop()
        env.close()
        if rclpy.ok():
            rclpy.shutdown()
        raise
    logger.info(
        f"Async camera stream live ({async_camera.count} frames in "
        f"{args.async_camera_timeout:.1f}s)."
    )

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
                "No mocap /target_pose message received yet. The arm will not "
                "move until the tracker starts publishing."
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
        # env.get_obs() now does robot state / joints / gripper / task only —
        # env.cameras was emptied after the async subscriber took over, so
        # there is no stale-camera read (and no rclpy "image data is stale"
        # warn flood). We time just the get_obs() part so we can see how
        # much of data_fn is non-camera env work.
        t0 = time.perf_counter()
        obs = env.get_obs()
        profile_stats.record("get_obs_ms", (time.perf_counter() - t0) * 1000.0)

        # Overwrite the camera entry with the async subscriber's latest
        # frame. Time this separately so we can see whether the lock or the
        # handoff is the bottleneck (expected: << 1 ms).
        t0 = time.perf_counter()
        async_frame = async_camera.get()
        profile_stats.record("camera_get_ms", (time.perf_counter() - t0) * 1000.0)
        if async_frame is not None:
            obs[image_key] = async_frame
        # else: fall through with the stale env.cameras[0].current_image —
        # shouldn't happen after wait_until_ready passed on the async sub.

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
        print(profile_stats.summary(title="Episode profile"))
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
        print(profile_stats.summary(title="Final profile (post-shutdown)"))
        logging.getLogger().removeHandler(overshoot_handler)
        try:
            async_camera.stop()
        except Exception:  # noqa: BLE001
            logger.exception("Error stopping async camera subscriber")
        env.close()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
