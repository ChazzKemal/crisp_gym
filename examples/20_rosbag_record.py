#!/usr/bin/env python3
"""Keyboard-driven rosbag recorder with per-episode mp4 extraction.

Diagnostic companion to ``16_camera_only_record.py``. Where script 16
runs a Python tick loop at ``--fps`` and sub-samples topic callbacks
(single-slot last-value semantics), this script asks ``ros2 bag record``
to write every received message to disk, so the resulting bag reflects
what the DDS layer actually delivered — no recorder-loop downsampling,
no shared-buffer race, no video re-encoding.

Controls (same r/s/d/q scheme as `16_camera_only_record.py`):
    r  →  start recording (first press) / stop recording (second press)
    s  →  keep the just-recorded bag (queued for processing)
    d  →  discard the just-recorded bag (deletes the bag directory)
    q  →  quit (discards any paused bag, then processes queued bags)

By default, processing is *deferred*: pressing ``s`` only keeps the bag,
so episodes record back-to-back with no wait. When the session ends
(``q`` or ``--num-episodes``) every kept bag is processed in recording
order. Pass ``--process-immediately`` to instead process each bag inline
on the ``s`` press (the old blocking behaviour).

For each kept episode the script:
    1. runs ``ros2 bag info`` on the bag,
    2. extracts the compressed-camera topic to ``camera_native.mp4``
       in the bag dir (skip with ``--no-extract``),
    3. if ``--downsample-fps N`` is set, also writes ``camera_<N>fps.mp4``
       by nearest-source-timestamp sampling.
A bag that fails to process is logged and kept on disk; the rest of the
queue still runs.

Optional ``--go-home`` (off by default) imports the crisp_gym env,
sends the arm to its home pose via the joint-trajectory controller,
then switches back to the cartesian controller so teleop can start
streaming target poses before you press ``r``.

Optional ``--rehome-each-episode`` repeats that home+switch after every
episode (saved or discarded), before waiting for the next ``r``, so each
episode starts from the home pose. It implies ``--go-home``.

Pass ``--from-bag DIR`` to skip recording entirely and run steps 1–3
on an existing bag.

Usage:
    # Interactive session, unlimited episodes, stop each with 'r':
    pixi run -e jazzy-lerobot python examples/20_rosbag_record.py

    # Same but home the arm first and switch to cartesian for teleop:
    pixi run -e jazzy-lerobot python examples/20_rosbag_record.py \\
        --go-home --env-config ur10e_ridgeback_env

    # Cap each episode at 30 s and save a 30 fps downsample on 's':
    pixi run -e jazzy-lerobot python examples/20_rosbag_record.py \\
        --duration 30 --downsample-fps 30

    # Auto-convert each saved episode to a LeRobot dataset named
    # 'teleop_demo' (re-using the same name appends episodes):
    pixi run -e jazzy-lerobot python examples/20_rosbag_record.py \\
        --go-home --repo-id teleop_demo --lerobot-task "pick cube"

    # Re-process an existing bag (no recording):
    pixi run -e jazzy-lerobot python examples/20_rosbag_record.py \\
        --from-bag ./rosbag_recordings/bag_20260417_160000 \\
        --downsample-fps 30
"""

import argparse
import datetime
import logging
import os
import queue
import select
import shutil
import signal
import subprocess
import sys
import termios
import threading
import time
import tty
from fractions import Fraction
from pathlib import Path


logger = logging.getLogger(__name__)


DEFAULT_TOPICS = [
    # Cameras — both included by default. ros2 bag record waits silently for
    # absent topics, so if you've only got the Orbbec up (no `pixi run
    # realsense`), the /d405/... topic is just skipped — no error, no empty
    # entry in the resulting bag.
    "/camera/color/image_raw/compressed",                # Orbbec Femto Bolt
    "/d405/camera/color/image_rect_raw/compressed",      # Intel RealSense D405
    "/joint_states",                                     # arm joints
    "/gripper/joint_states",                             # Robotiq 2F-85 gripper joints
    "/current_pose",
    # Action topics — recorded by default so 21_bag_to_lerobot.py --with-action
    # has everything it needs. If the teleop source isn't publishing, ros2 bag
    # record simply captures zero messages on these (no error).
    "/target_pose",
    "/target_gripper_state",
]


# ---------- Keyboard listener ----------

class KeyboardListener:
    """Non-blocking single-char keyboard input from stdin.

    Puts each pressed character onto a queue. Uses termios cbreak mode so
    keys are delivered without Enter and without echo, while Ctrl+C still
    raises SIGINT (unlike raw mode). Call ``start()`` before use and
    ``stop()`` before program exit to restore terminal settings.
    """

    def __init__(self) -> None:
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._fd: int | None = None
        self._old_settings = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="kb-listener"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._fd is not None and self._old_settings is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
            except Exception:
                pass

    def _loop(self) -> None:
        while not self._stop.is_set():
            r, _, _ = select.select([sys.stdin], [], [], 0.05)
            if r:
                try:
                    ch = sys.stdin.read(1)
                except Exception:
                    continue
                if ch:
                    self._queue.put(ch)

    def get_nowait(self) -> str | None:
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def flush(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return


# ---------- Recording subprocess ----------

def build_record_cmd(bag_dir: Path, topics: list[str], storage: str) -> list[str]:
    return [
        "ros2", "bag", "record",
        "-o", str(bag_dir),
        "-s", storage,
        *topics,
    ]


def start_bag_subprocess(
    bag_dir: Path, topics: list[str], storage: str
) -> subprocess.Popen:
    cmd = build_record_cmd(bag_dir, topics, storage)
    logger.info("Starting: %s", " ".join(cmd))
    return subprocess.Popen(cmd, start_new_session=True)


def stop_bag_subprocess(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=10.0)
        return
    except subprocess.TimeoutExpired:
        logger.warning("ros2 bag record did not stop in 10 s; sending SIGTERM")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=5.0)
        return
    except subprocess.TimeoutExpired:
        logger.error("ros2 bag record did not stop in 5 s after SIGTERM; sending SIGKILL")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        return
    proc.wait(timeout=5.0)


def run_bag_info(bag_dir: Path) -> None:
    info_cmd = ["ros2", "bag", "info", str(bag_dir)]
    logger.info("Running: %s", " ".join(info_cmd))
    sys.stdout.flush()
    try:
        subprocess.run(info_cmd, check=False)
    except FileNotFoundError:
        logger.error("ros2 not found on PATH — skipping `ros2 bag info`.")


# ---------- Env setup (optional --go-home) ----------

def home_and_switch_to_cartesian(env_config: str):
    """Build a crisp_gym env, home via JTC, then switch back to cartesian.

    Returns the env so the caller can keep it alive for the session.
    """
    from crisp_gym.envs.manipulator_env import make_env

    logger.info("Creating env %r for homing...", env_config)
    env = make_env(env_type=env_config, control_type="cartesian", namespace="")
    logger.info("Waiting for robot + gripper to be ready...")
    env.wait_until_ready()
    logger.info("Homing via joint_trajectory_controller (blocking)...")
    env.home(blocking=True)
    logger.info("Switching back to cartesian_controller...")
    env.switch_controller("cartesian")
    logger.info("Env ready. Start your teleop source now (if using --with-action).")
    return env


# ---------- Extraction ----------

def _open_reader(bag_dir: Path, storage_id: str = ""):
    """Open a rosbag2 SequentialReader. Empty storage_id lets rosbag2 auto-detect."""
    import rosbag2_py
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(
        uri=str(bag_dir), storage_id=storage_id
    )
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader.open(storage_options, converter_options)
    return reader


def _topic_types(bag_dir: Path, storage_id: str = "") -> dict[str, str]:
    reader = _open_reader(bag_dir, storage_id=storage_id)
    return {t.name: t.type for t in reader.get_all_topics_and_types()}


def _iter_topic(bag_dir: Path, topic: str, storage_id: str = ""):
    """Yield (bag_timestamp_ns, serialized_bytes) for every message on `topic`."""
    import rosbag2_py
    reader = _open_reader(bag_dir, storage_id=storage_id)
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    while reader.has_next():
        tn, data, ts = reader.read_next()
        if tn == topic:
            yield ts, data


def _autodetect_camera_topic(bag_dir: Path, storage_id: str = "") -> str | None:
    """Pick the first CompressedImage topic in the bag."""
    types = _topic_types(bag_dir, storage_id=storage_id)
    for name, ty in types.items():
        if ty == "sensor_msgs/msg/CompressedImage":
            return name
    return None


def _nearest_timestamp_indices(
    ts_list: list[int], target_fps: float
) -> list[int]:
    """Sample indices of `ts_list` at `target_fps` using nearest-neighbor
    to a uniform tick grid starting at ts_list[0].
    """
    tick_period_ns = max(1, int(1e9 / target_fps))
    indices: list[int] = []
    cursor = 0
    next_tick = ts_list[0]
    last_ts = ts_list[-1]
    n = len(ts_list)
    while next_tick <= last_ts:
        while cursor + 1 < n and ts_list[cursor + 1] <= next_tick:
            cursor += 1
        if cursor + 1 < n:
            if (ts_list[cursor + 1] - next_tick) < (next_tick - ts_list[cursor]):
                indices.append(cursor + 1)
            else:
                indices.append(cursor)
        else:
            indices.append(cursor)
        next_tick += tick_period_ns
    return indices


def extract_video(
    bag_dir: Path,
    topic: str,
    output: Path,
    target_fps: float | None,
    storage_id: str = "",
) -> dict:
    """Extract a CompressedImage topic from a bag to an mp4.

    target_fps: None -> encode every message (CFR at round(effective_source_fps)).
                else  -> downsample to that rate via nearest-timestamp sampling.
    Returns a dict of stats (useful for logging).
    """
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import CompressedImage
    from cv_bridge import CvBridge
    import av

    types = _topic_types(bag_dir, storage_id=storage_id)
    if topic not in types:
        raise ValueError(
            f"Topic {topic!r} not in bag. Available topics: {sorted(types)}"
        )
    if types[topic] != "sensor_msgs/msg/CompressedImage":
        raise ValueError(
            f"Topic {topic!r} is {types[topic]}, expected sensor_msgs/msg/CompressedImage"
        )

    records = list(_iter_topic(bag_dir, topic, storage_id=storage_id))
    if len(records) < 2:
        raise ValueError(
            f"Only {len(records)} message(s) on {topic!r}; cannot encode video."
        )

    ts_list = [r[0] for r in records]
    span_s = (ts_list[-1] - ts_list[0]) * 1e-9
    if span_s <= 0:
        raise ValueError(
            f"All {len(records)} messages on {topic!r} share the same timestamp "
            f"({ts_list[0]} ns); cannot derive a frame rate."
        )
    effective_fps = (len(records) - 1) / span_s

    if target_fps is None or target_fps <= 0:
        indices = list(range(len(records)))
        write_rate = max(1, int(round(effective_fps)))
    else:
        indices = _nearest_timestamp_indices(ts_list, target_fps)
        write_rate = max(1, int(round(target_fps)))

    bridge = CvBridge()

    first_msg = deserialize_message(records[indices[0]][1], CompressedImage)
    first_rgb = bridge.compressed_imgmsg_to_cv2(first_msg, desired_encoding="rgb8")
    h, w = first_rgb.shape[:2]
    # h264 + yuv420p requires even dimensions.
    h_enc = h - (h % 2)
    w_enc = w - (w % 2)

    output.parent.mkdir(parents=True, exist_ok=True)

    duplicate_samples = 0
    last_sel = None
    with av.open(str(output), mode="w") as container:
        stream = container.add_stream(
            "libx264", rate=write_rate,
            options={"crf": "23", "preset": "veryfast"},
        )
        stream.pix_fmt = "yuv420p"
        stream.width = w_enc
        stream.height = h_enc

        for i, sel in enumerate(indices):
            if i == 0:
                rgb = first_rgb
            else:
                msg = deserialize_message(records[sel][1], CompressedImage)
                rgb = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="rgb8")
            if sel == last_sel:
                duplicate_samples += 1
            last_sel = sel
            if rgb.shape[:2] != (h_enc, w_enc):
                rgb = rgb[:h_enc, :w_enc]
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame = frame.reformat(format="yuv420p")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    return {
        "topic": topic,
        "output": str(output),
        "n_source_messages": len(records),
        "source_span_s": span_s,
        "effective_source_fps": effective_fps,
        "n_frames_written": len(indices),
        "write_rate_fps": write_rate,
        "duplicate_samples": duplicate_samples,
        "target_fps": target_fps,
    }


def log_extract_stats(stats: dict) -> None:
    logger.info("Wrote %s", stats["output"])
    logger.info(
        "  source: %d msgs over %.3f s -> effective %.2f Hz",
        stats["n_source_messages"],
        stats["source_span_s"],
        stats["effective_source_fps"],
    )
    if stats["target_fps"] is None:
        logger.info(
            "  encoded: %d frames at CFR %d fps (native, no resample)",
            stats["n_frames_written"],
            stats["write_rate_fps"],
        )
    else:
        dup = stats["duplicate_samples"]
        n = stats["n_frames_written"]
        dup_pct = 100.0 * dup / n if n else 0.0
        logger.info(
            "  encoded: %d frames at CFR %d fps (target %.1f); "
            "%d repeated source frames (%.1f%%)",
            n,
            stats["write_rate_fps"],
            stats["target_fps"],
            dup,
            dup_pct,
        )


def postprocess_bag(
    bag_dir: Path,
    *,
    no_extract: bool,
    extract_topic_arg: str | None,
    downsample_fps: float | None,
    storage_hint: str,
) -> None:
    """bag info + optional mp4 extraction + optional downsampled mp4."""
    run_bag_info(bag_dir)

    if no_extract:
        logger.info("Skipping mp4 extraction (--no-extract).")
        return

    extract_topic = extract_topic_arg or _autodetect_camera_topic(
        bag_dir, storage_id=storage_hint
    )
    if extract_topic is None:
        logger.info(
            "No CompressedImage topic found in bag; skipping mp4 extraction. "
            "Pass --extract-topic to override."
        )
        return

    logger.info("Extracting topic %s to mp4...", extract_topic)

    native_out = bag_dir / "camera_native.mp4"
    try:
        stats = extract_video(
            bag_dir, extract_topic, native_out,
            target_fps=None, storage_id=storage_hint,
        )
        log_extract_stats(stats)
    except Exception:
        logger.exception("Native extraction failed.")

    if downsample_fps:
        ds_out = bag_dir / f"camera_{int(round(downsample_fps))}fps.mp4"
        try:
            stats = extract_video(
                bag_dir, extract_topic, ds_out,
                target_fps=downsample_fps, storage_id=storage_hint,
            )
            log_extract_stats(stats)
        except Exception:
            logger.exception("Downsample extraction failed.")


# ---------- Auto-convert to LeRobot dataset ----------

CONVERTER_SCRIPT = Path(__file__).parent / "21_bag_to_lerobot.py"
LEROBOT_CACHE = Path.home() / ".cache" / "huggingface" / "lerobot"


def convert_bag_to_lerobot(
    bag_dir: Path,
    repo_id: str,
    fps: int,
    task: str,
    with_action: bool,
) -> bool:
    """Invoke 21_bag_to_lerobot.py as a subprocess. Returns True on success.

    Automatically passes --resume if ~/.cache/huggingface/lerobot/<repo_id>
    already exists, so calling this repeatedly with the same repo_id keeps
    appending episodes to a single dataset.
    """
    if not CONVERTER_SCRIPT.exists():
        logger.error("Converter script missing: %s", CONVERTER_SCRIPT)
        return False

    resume = (LEROBOT_CACHE / repo_id).exists()
    cmd = [
        sys.executable, str(CONVERTER_SCRIPT),
        "--bag", str(bag_dir),
        "--repo-id", repo_id,
        "--fps", str(fps),
        "--task", task,
    ]
    if with_action:
        cmd.append("--with-action")
    if resume:
        cmd.append("--resume")

    logger.info(
        "Converting bag to LeRobot dataset %r (%s)...",
        repo_id, "appending" if resume else "new",
    )
    logger.debug("Converter cmd: %s", " ".join(cmd))
    try:
        result = subprocess.run(cmd, check=False)
    except FileNotFoundError:
        logger.exception("Failed to launch converter.")
        return False
    if result.returncode != 0:
        logger.error(
            "Converter exited with code %d; see messages above.",
            result.returncode,
        )
        return False
    logger.info("LeRobot dataset %r ready at %s", repo_id, LEROBOT_CACHE / repo_id)
    return True


def process_one_bag(bag_dir: Path, args: argparse.Namespace) -> None:
    """Run bag-info + mp4 extraction + optional LeRobot conversion on one bag.

    Shared by the immediate (--process-immediately) and deferred (default,
    end-of-session) paths. Never raises: a failure is logged and the bag is
    kept on disk so the rest of the queue can still be processed.
    """
    try:
        postprocess_bag(
            bag_dir,
            no_extract=args.no_extract,
            extract_topic_arg=args.extract_topic,
            downsample_fps=args.downsample_fps,
            storage_hint=args.storage,
        )
    except Exception:
        logger.exception("Post-processing failed; bag kept at %s.", bag_dir)

    if args.to_lerobot:
        try:
            convert_bag_to_lerobot(
                bag_dir,
                repo_id=args.to_lerobot,
                fps=args.lerobot_fps,
                task=args.lerobot_task,
                with_action=not args.lerobot_no_action,
            )
        except Exception:
            logger.exception(
                "Auto-convert to LeRobot failed; bag kept at %s.", bag_dir,
            )


# ---------- Interactive session ----------

INSTRUCTIONS = (
    "\n"
    "  Keys:\n"
    "    r  start / stop recording\n"
    "    s  save the just-recorded bag (processed at end of session;\n"
    "       use --process-immediately to extract/convert inline instead)\n"
    "    d  discard the just-recorded bag (delete directory)\n"
    "    q  quit (then processes any queued bags)\n"
)


def _announce(msg: str) -> None:
    # Using print + explicit \r because stdin is in cbreak; without \r the
    # terminal does not move the cursor to column 0 after a newline.
    sys.stdout.write("\r" + msg + "\n")
    sys.stdout.flush()


def run_session(args: argparse.Namespace) -> None:
    topics = list(args.topic) if args.topic else list(DEFAULT_TOPICS)

    env = None
    if args.go_home:
        try:
            env = home_and_switch_to_cartesian(args.env_config)
        except Exception:
            logger.exception("--go-home failed; aborting session.")
            return

    def _on_sigterm(signum, _frame):
        raise SystemExit(128 + signum)
    prev_sigterm = signal.signal(signal.SIGTERM, _on_sigterm)

    kb = KeyboardListener()
    kb.start()
    proc: subprocess.Popen | None = None
    # Bags kept with 's'. In the default (deferred) mode these are processed
    # in a batch after the session ends; with --process-immediately they are
    # processed inline and this list is only used for the final summary.
    pending_bags: list[Path] = []
    try:
        _announce("Ready.")
        _announce(INSTRUCTIONS)
        episode = 0
        while True:
            # ---- re-home between episodes ----
            # episode > 0 means a previous episode ran; the initial --go-home
            # already homed before the first one, so we only re-home here for
            # the 2nd episode onward. On quit the loop has already exited, so
            # this never homes just to immediately stop.
            if args.rehome_each_episode and episode > 0 and env is not None:
                _announce("[homing] returning to home pose via JTC...")
                try:
                    env.home(blocking=True)
                    env.switch_controller("cartesian")
                    _announce("[homing] done; back on cartesian_controller.")
                except Exception:
                    logger.exception("Re-home failed; continuing anyway.")

            # ---- waiting state ----
            # Intentionally do NOT flush the queue here: a key pressed during
            # the slow postprocess/lerobot block above (e.g. anticipatory 'r'
            # to start the next episode, or 'q' to quit) should be honored.
            # Stale non-r/non-q chars are silently consumed by the inner loop
            # below via `if ch != "r": continue`.
            _announce(f"[waiting] press 'r' to start episode {episode + 1}, 'q' to quit.")
            ch = None
            while ch is None:
                ch = kb.get_nowait()
                if ch is None:
                    time.sleep(0.05)
            if ch == "q":
                _announce("Quit requested.")
                break
            if ch != "r":
                continue

            # ---- recording state ----
            episode += 1
            # Millisecond precision so two episodes within the same wall second
            # don't collide on bag_dir (ros2 bag record refuses to start when
            # the output dir already exists).
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            bag_dir = args.output_dir / f"bag_{stamp}"
            args.output_dir.mkdir(parents=True, exist_ok=True)
            _announce(f"[recording] episode {episode} -> {bag_dir}")
            if args.duration:
                _announce(f"  (auto-stop in {args.duration:.1f} s, or press 'r')")
            else:
                _announce("  press 'r' to stop")

            proc = start_bag_subprocess(bag_dir, topics, args.storage)
            start_t = time.monotonic()
            deadline = start_t + args.duration if args.duration else None
            stopped_by_user = False
            while True:
                if proc.poll() is not None:
                    _announce("  ros2 bag record exited early")
                    break
                if deadline and time.monotonic() >= deadline:
                    _announce("  duration reached; stopping")
                    break
                ch = kb.get_nowait()
                if ch == "r":
                    stopped_by_user = True
                    # Announce *before* the slow bag shutdown so the user
                    # sees their key was heard — otherwise they keep
                    # mashing 'r' for the 5–10 s it takes ros2 bag record
                    # to finalise the mcap footer.
                    _announce("  stopping... (finalising bag, may take a few seconds)")
                    break
                time.sleep(0.05)
            stop_bag_subprocess(proc)
            proc = None
            _announce(
                f"  stopped after {time.monotonic() - start_t:.1f} s "
                f"({'user' if stopped_by_user else 'timeout/exit'})"
            )

            # ---- paused state ----
            # Do NOT flush: anticipatory 's'/'d'/'q' pressed while
            # stop_bag_subprocess was finalising the bag is still valid here.
            # Spurious extra 'r' presses (the most common case — user mashed
            # 'r' wondering if it registered) are silently consumed by the
            # while-loop below until a real action key arrives.
            _announce("[paused]  's' save, 'd' discard, 'q' quit (discards).")
            ch = None
            while ch not in ("s", "d", "q"):
                ch = kb.get_nowait()
                if ch is None:
                    time.sleep(0.05)

            if ch in ("d", "q"):
                _announce(f"  discarding {bag_dir}")
                shutil.rmtree(bag_dir, ignore_errors=True)
                if ch == "q":
                    break
                continue

            # ch == "s": keep the bag.
            pending_bags.append(bag_dir)
            if args.process_immediately:
                # Heavy work runs now, blocking the next episode. Leave cbreak
                # so anticipatory 'r'/'q' presses during the slow work are
                # preserved (no flush) and honoured by the next loop iteration.
                process_one_bag(bag_dir, args)
            else:
                _announce(
                    f"  saved (queued for end-of-session processing): {bag_dir}"
                )

            if args.num_episodes and episode >= args.num_episodes:
                _announce(f"Reached --num-episodes {args.num_episodes}; done.")
                break

        # ---- deferred batch processing ----
        # In the default mode nothing was processed inline; do it all now,
        # in recording order so --resume appends episodes to the LeRobot
        # dataset in sequence. process_one_bag never raises, so one bad bag
        # does not abort the rest of the queue.
        if not args.process_immediately and pending_bags:
            _announce(f"\nProcessing {len(pending_bags)} queued bag(s)...")
            for i, queued_bag in enumerate(pending_bags, start=1):
                _announce(f"[process {i}/{len(pending_bags)}] {queued_bag}")
                process_one_bag(queued_bag, args)
            _announce("All queued bags processed.")
    finally:
        if proc is not None and proc.poll() is None:
            try:
                stop_bag_subprocess(proc)
            except Exception:
                logger.exception("Failed to stop ros2 bag record during cleanup")
        kb.stop()
        signal.signal(signal.SIGTERM, prev_sigterm)
        if env is not None:
            try:
                env.close()
            except Exception:
                logger.debug("env.close() raised", exc_info=True)


# ---------- CLI ----------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # output + topics
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path.cwd() / "rosbag_recordings",
        help="Parent directory for new recordings. Each episode goes in "
             "<output-dir>/bag_<timestamp>/ (default: ./rosbag_recordings/).",
    )
    parser.add_argument(
        "--topic", action="append", default=None,
        help="Topic to record (repeatable). If omitted, uses: "
             + ", ".join(DEFAULT_TOPICS),
    )
    parser.add_argument(
        "--storage", type=str, default="mcap",
        choices=["mcap", "sqlite3"],
        help="Storage backend (default: mcap, Jazzy default).",
    )

    # session / timing
    parser.add_argument(
        "--duration", type=float, default=None,
        help="Auto-stop each episode after this many seconds "
             "(omit = only stops on 'r').",
    )
    parser.add_argument(
        "--num-episodes", type=int, default=0,
        help="Quit after this many saved episodes (0 = unlimited).",
    )

    # pre-session setup
    parser.add_argument(
        "--go-home", action="store_true", default=False,
        help="Before waiting for 'r', build a crisp_gym env, home via JTC, "
             "then switch to cartesian_controller. Default off.",
    )
    parser.add_argument(
        "--env-config", type=str, default="ur10e_ridgeback_env",
        help="crisp_gym env config (used with --go-home).",
    )
    parser.add_argument(
        "--rehome-each-episode", action="store_true", default=False,
        help="After every episode (saved or discarded), re-home the arm via "
             "joint_trajectory_controller and switch back to cartesian_controller "
             "before waiting for the next 'r'. Implies --go-home.",
    )

    # --from-bag: skip recording, just post-process
    parser.add_argument(
        "--from-bag", type=Path, default=None,
        help="Skip recording; run bag-info + extraction on this existing bag dir.",
    )

    # post-processing
    parser.add_argument(
        "--no-extract", action="store_true", default=False,
        help="Skip mp4 extraction after each saved episode.",
    )
    parser.add_argument(
        "--extract-topic", type=str, default=None,
        help="CompressedImage topic to extract (default: first one found in the bag).",
    )
    parser.add_argument(
        "--downsample-fps", type=float, default=None,
        help="Also emit camera_<N>fps.mp4 at this rate via nearest-timestamp sampling.",
    )
    parser.add_argument(
        "--process-immediately", action="store_true", default=False,
        help="Run bag-info + mp4 extraction (+ LeRobot conversion) inline on "
             "each 's' press, blocking the next episode. Default: defer all "
             "processing to the end of the session so episodes record "
             "back-to-back with no wait.",
    )

    # LeRobot auto-conversion
    parser.add_argument(
        "--repo-id", "--to-lerobot", dest="to_lerobot",
        type=str, default=None, metavar="REPO_ID",
        help="After each saved episode, auto-convert the bag to a LeRobot "
             "dataset with this repo-id (runs 21_bag_to_lerobot.py). "
             "Subsequent episodes append to the same dataset. "
             "(--to-lerobot accepted as alias for backwards compat.)",
    )
    parser.add_argument(
        "--lerobot-fps", type=int, default=30,
        help="Target fps for LeRobot dataset (used with --to-lerobot, default 30).",
    )
    parser.add_argument(
        "--lerobot-task", type=str, default="observe scene",
        help="Task label for LeRobot episodes (used with --to-lerobot).",
    )
    parser.add_argument(
        "--lerobot-no-action", action="store_true", default=False,
        help="Skip action features in the LeRobot dataset (used with --to-lerobot). "
             "Default: include action (requires teleop to be publishing).",
    )

    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if args.rehome_each_episode and not args.go_home:
        logger.info("--rehome-each-episode implies --go-home; enabling it.")
        args.go_home = True

    if args.from_bag is not None:
        if not args.from_bag.exists():
            logger.error("--from-bag %s does not exist", args.from_bag)
            sys.exit(1)
        logger.info("Post-processing existing bag: %s", args.from_bag)
        postprocess_bag(
            args.from_bag,
            no_extract=args.no_extract,
            extract_topic_arg=args.extract_topic,
            downsample_fps=args.downsample_fps,
            storage_hint="",  # auto-detect for existing bags
        )
        if args.to_lerobot:
            convert_bag_to_lerobot(
                args.from_bag,
                repo_id=args.to_lerobot,
                fps=args.lerobot_fps,
                task=args.lerobot_task,
                with_action=not args.lerobot_no_action,
            )
        return

    run_session(args)


if __name__ == "__main__":
    main()
