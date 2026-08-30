"""Drop-in replacement for TargetSenderThread that proxies to a C++ subprocess.

The C++ binary (`crisp_sender`, in
`clearpath_remote_ws/src/tum09_ridgeback/tum09_custom/src/crisp_sender.cpp`)
owns the timing-critical sleep + publish loop. This Python class is what
`19_deploy_policy.py` instantiates instead of `TargetSenderThread` when
`--cpp-sender` is set.

Architecture (see plan @ ~/.claude/plans/okay-plan-option-a-5-fizzy-giraffe.md):

  Python deploy ─push TargetItem─► /dev/shm/crisp_sender_actions (ring) ─► C++
                                  /dev/shm/crisp_sender_setup    (one-shot) ─► C++
                                  /dev/shm/crisp_sender_scaler   (cycle bumps) ─► C++
                                  /dev/shm/crisp_sender_stats    ◄─ C++ writes

Why this exists: Python's `time.sleep` accuracy + GIL + CFS preemption produce
sleep_overshoot p99 ≈ 5 ms with rare 70+ ms outliers, which makes 3-4x speedup
unreliable. The C++ binary uses `clock_nanosleep(TIMER_ABSTIME)` (drift-free)
+ optional `SCHED_FIFO` (preemption-free), and shaves both the floor and the
tail.

Public surface mirrors `TargetSenderThread`:
    .put(item)        # item is a TargetItem or None (sentinel)
    .qsize()          # producer reads this for q_before_inf/q_before_push
    .start()          # no-op (subprocess is started in __init__)
    .join(timeout)    # send sentinel, wait for C++ exit, ingest stats
    .is_alive()       # subprocess.poll() is None
    .stage_samples    # dict of per-stage timing lists (ms)
    .slack_samples_ms # per-frame deadline slack (ms; negative = late)
    .n_late_frames    # count of slack<0 entries
    .n_published      # count of frames C++ saw
    .frame_rows       # list of dicts for frames.csv
    .pub_dt_samples   # debug timestamp diffs (only filled with --debug-publish)
    .queue_depth_min/.queue_depth_max  # tracked locally on .put()
    .underrun_count   # alias for n_late_frames
"""
from __future__ import annotations

import logging
import mmap
import os
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---- Layout constants (must match src/crisp_sender.cpp) -------------------

# Setup segment.
_SETUP_SIZE              = 8192
_SETUP_COMPLETE_OFF      = 0
# Layout guard. Every offset in this file is duplicated by hand in
# clearpath_remote_ws/.../src/crisp_sender.cpp -- two lists in two languages that
# agree only because someone typed the same numbers twice. Nothing verified that,
# which was harmless while both shipped from one tree and were rebuilt together.
# Once crisp_gym is pinned by SHA and clearpath_remote_ws moves independently they
# can drift, and the failure is silent: insert one field mid-struct and Python
# writes the deadline where C++ reads the pose, so the arm goes to a nonsense
# absolute position with no error.
#
# So the "setup complete" flag doubles as a layout stamp: instead of 1, Python
# writes MAGIC<<32 | VERSION and C++ requires that exact value. No new offsets --
# deliberately, since a new offset is itself another number to keep in sync.
# BUMP THE VERSION whenever any offset or slot size in this file changes.
_SHM_LAYOUT_MAGIC        = 0x43535044   # "CSPD", crisp sender protocol
_SHM_LAYOUT_VERSION      = 1
_SETUP_COMPLETE_VALUE    = (_SHM_LAYOUT_MAGIC << 32) | _SHM_LAYOUT_VERSION
_SETUP_READY_OFF         = 56   # C++ sets after rclcpp init + publisher discovery
_SETUP_KP_EXP_OFF        = 8
_SETUP_KD_EXP_OFF        = 16
_SETUP_BASE_GRIP_SPD_OFF = 24
_SETUP_GRIP_STRIDE_OFF   = 32
_SETUP_NUM_KP_OFF        = 36
_SETUP_NUM_KD_OFF        = 40
_SETUP_HAS_GRIP_SPD_OFF  = 44
_SETUP_GRIP_MAX_OFF      = 48
_SETUP_CTRL_NAME_OFF     = 64    # 128 bytes
_SETUP_GRIP_SPD_TOPIC_OFF = 192  # 128 bytes
_SETUP_PARAMS_OFF        = 320
_SETUP_PARAM_SLOT_SIZE   = 72    # name[64] + double
_SETUP_MAX_PARAMS        = 32    # up to 16 kp + 16 kd

# Action ring.
_ACTION_HEADER_SIZE     = 64
_ACTION_SLOT_SIZE       = 64
_ACTION_RING_CAPACITY   = 512
_ACTION_RING_SIZE       = _ACTION_HEADER_SIZE + _ACTION_RING_CAPACITY * _ACTION_SLOT_SIZE
_AH_HEAD_OFF            = 0
_AH_TAIL_OFF            = 8
_AH_CAPACITY_OFF        = 16
_AH_STOP_OFF            = 24
_AH_DT_HINT_OFF         = 32
_AS_SEQNUM_OFF          = 0
_AS_DEADLINE_OFF        = 8
_AS_POSE_XYZ_OFF        = 16
_AS_POSE_QUAT_OFF       = 28
_AS_GRIP_RAW_OFF        = 44
_AS_FLAGS_OFF           = 48
_AS_FRAME_IDX_OFF       = 52
_AS_S_EFF_OFF           = 56
_AS_CYCLES_OFF          = 60
_FLAG_HAS_GRIPPER       = 1 << 0
_FLAG_SENTINEL          = 1 << 1

# Scaler segment.
_SCALER_SIZE     = 32
_SC_REQ_OFF      = 0
_SC_ACK_OFF      = 8
_SC_S_EFF_OFF    = 16

# Stats ring.
_STATS_HEADER_SIZE   = 32
_STATS_SLOT_SIZE     = 48
_STATS_RING_CAPACITY = 8192
_STATS_RING_SIZE     = _STATS_HEADER_SIZE + _STATS_RING_CAPACITY * _STATS_SLOT_SIZE
_SH_HEAD_OFF         = 0
_SH_TAIL_OFF         = 8
_SH_CAPACITY_OFF     = 16
_SH_DROPPED_OFF      = 24
_SS_SEQNUM_OFF       = 0
_SS_DEADLINE_OFF     = 8
_SS_PRE_SLEEP_OFF    = 16
_SS_POST_SLEEP_OFF   = 24
_SS_POST_POSE_OFF    = 32
_SS_POST_GRIP_OFF    = 40

# Sentinel object reused across calls.
_SENTINEL = object()


def _open_shm(path: str, size: int) -> mmap.mmap:
    """Create + truncate + mmap a /dev/shm file. Zeros the contents."""
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        os.ftruncate(fd, size)
        mm = mmap.mmap(fd, size, prot=mmap.PROT_READ | mmap.PROT_WRITE)
    finally:
        os.close(fd)
    mm[:size] = b"\x00" * size
    return mm


def _pack_cstr(mm: mmap.mmap, off: int, s: str, max_len: int) -> None:
    data = s.encode("utf-8")[: max_len - 1]
    mm[off : off + max_len] = data + b"\x00" * (max_len - len(data))


class CppSenderHandle:
    """C++ subprocess sender; quacks like TargetSenderThread."""

    def __init__(
        self,
        *,
        target_pose_topic: str,
        gripper_topic: Optional[str],
        frame_id: str,
        scaler: Any,                 # ReplayScaler (or None)
        replay_log: list,
        state_capture_fn: Any = None,
        debug_publish: bool = False,
        log_interval_s: float = 5.0,
        dry_run: bool = False,
        rt_priority: int = 0,
        binary_path: Optional[str] = None,
    ) -> None:
        if gripper_topic is not None and gripper_topic == "":
            gripper_topic = None

        self._target_pose_topic = target_pose_topic
        self._gripper_topic = gripper_topic
        self._frame_id = frame_id
        self._scaler = scaler
        self._replay_log = replay_log
        self._state_capture_fn = state_capture_fn
        self._debug_publish = debug_publish
        self._log_interval_s = log_interval_s
        self._dry_run = bool(dry_run)
        self._rt_priority = int(rt_priority)

        # Diagnostics (matching TargetSenderThread surface).
        self.pub_dt_samples: list[float] = []
        self.queue_depth_min: int = 2 ** 31
        self.queue_depth_max: int = 0
        self.n_published: int = 0
        self.underrun_count: int = 0
        self.slack_samples_ms: list[float] = []
        self.n_late_frames: int = 0
        self.stage_samples: dict[str, list[float]] = {
            "pop_ms": [],
            "scaler_rpc_ms": [],
            "sleep_overshoot_ms": [],
            "pub_pose_ms": [],
            "pub_grip_ms": [],
            "loop_total_ms": [],
        }
        self.frame_rows: list[dict] = []

        # Per-put tracking (used to feed the producer's q.qsize()).
        self._n_pushed: int = 0
        self._prev_cycles: Optional[int] = None
        self._scaler_seqnum: int = 0
        self._action_seqnum: int = 0

        # Create SHM segments. Distinct filenames per process to avoid
        # collision with a stale C++ binary still draining its mmap.
        suffix = f"_{os.getpid()}"
        self._setup_path  = f"/dev/shm/crisp_sender_setup{suffix}"
        self._action_path = f"/dev/shm/crisp_sender_actions{suffix}"
        self._scaler_path = f"/dev/shm/crisp_sender_scaler{suffix}"
        self._stats_path  = f"/dev/shm/crisp_sender_stats{suffix}"

        self._setup_mm  = _open_shm(self._setup_path,  _SETUP_SIZE)
        self._action_mm = _open_shm(self._action_path, _ACTION_RING_SIZE)
        self._scaler_mm = _open_shm(self._scaler_path, _SCALER_SIZE)
        self._stats_mm  = _open_shm(self._stats_path,  _STATS_RING_SIZE)

        # Initialise action-ring header.
        struct.pack_into("<Q", self._action_mm, _AH_CAPACITY_OFF,
                         _ACTION_RING_CAPACITY)
        # Initialise stats-ring header.
        struct.pack_into("<Q", self._stats_mm, _SH_CAPACITY_OFF,
                         _STATS_RING_CAPACITY)

        # Populate setup SHM from scaler (if any).
        self._write_setup()

        # Stderr log file for the subprocess so we can surface crashes.
        self._stderr_log_path = (Path(tempfile.gettempdir()) /
                                  f"crisp_sender_{os.getpid()}.stderr.log")
        self._stderr_fh = open(self._stderr_log_path, "wb")

        # Resolve binary path. Default: rely on `ros2 run` resolving
        # `tum09_custom crisp_sender`; supports an explicit override for
        # tests.
        if binary_path is None:
            binary_path = os.environ.get("CRISP_SENDER_BIN")
        argv = self._build_subprocess_argv(binary_path)
        logger.info("spawning crisp_sender subprocess: %s", " ".join(argv))
        self._proc = subprocess.Popen(
            argv,
            stdout=self._stderr_fh,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )

    # ----- Subprocess argv -------------------------------------------------

    def _build_subprocess_argv(self, binary_path: Optional[str]) -> list[str]:
        if binary_path is None:
            # The C++ sender lives in clearpath_remote_ws/install but the
            # deploy runs in jazzy-lerobot's pixi env (different ROS search
            # path). Try a known install location first, then fall back to
            # `ros2 run` which works if the user has setup.bash sourced.
            candidate = Path(
                "/home/ali/Coding/Robot_Control/clearpath_remote_ws/install/"
                "tum09_custom/lib/tum09_custom/crisp_sender"
            )
            if candidate.exists():
                binary_path = str(candidate)
                logger.info("using crisp_sender at %s", binary_path)
        if binary_path is None:
            base = ["ros2", "run", "tum09_custom", "crisp_sender"]
        else:
            base = [binary_path]
        args = [
            *base,
            "--setup-shm",   self._setup_path,
            "--action-shm",  self._action_path,
            "--scaler-shm",  self._scaler_path,
            "--stats-shm",   self._stats_path,
            "--pose-topic",  self._target_pose_topic,
            "--gripper-topic", self._gripper_topic or "",
            "--frame-id",    self._frame_id,
            "--rt-priority", str(self._rt_priority),
        ]
        if self._dry_run:
            args.append("--dry-run")
        return args

    # ----- Setup SHM -------------------------------------------------------

    def _write_setup(self) -> None:
        scaler = self._scaler
        kp_params: list[tuple[str, float]] = []
        kd_params: list[tuple[str, float]] = []
        kp_exp = 2.0
        kd_exp = 1.0
        base_gripper_speed = 0.0
        gripper_stride = 1
        gripper_speed_topic = ""
        controller_node = ""

        if scaler is not None:
            kp_exp = float(getattr(scaler, "kp_exp", 2.0))
            kd_exp = float(getattr(scaler, "kd_exp", 1.0))
            base_gripper_speed = float(getattr(scaler, "base_gripper_speed", 0.0))
            gripper_stride = int(getattr(scaler, "gripper_stride", 1))
            controller_node = str(getattr(scaler, "controller_node", ""))
            for name, orig in getattr(scaler, "_original_kp", {}).items():
                kp_params.append((str(name), float(orig)))
            # Skip kd params that the controller is auto-damping.
            kd_is_auto = getattr(scaler, "_kd_is_auto", {})
            for name, orig in getattr(scaler, "_original_kd", {}).items():
                if not kd_is_auto.get(name, True):
                    kd_params.append((str(name), float(orig)))
            # Gripper speed topic — only populated if the scaler has
            # spawned the gripper_speed_controller successfully.
            speed_pub = getattr(scaler, "_speed_pub", None)
            if speed_pub is not None:
                gripper_speed_topic = speed_pub.topic_name
                # ROS adds a leading '/' if missing; keep consistent.

        if len(kp_params) + len(kd_params) > _SETUP_MAX_PARAMS:
            raise RuntimeError(
                f"too many scaler params for SHM setup: {len(kp_params)} kp + "
                f"{len(kd_params)} kd > {_SETUP_MAX_PARAMS} max"
            )

        struct.pack_into("<d", self._setup_mm, _SETUP_KP_EXP_OFF, kp_exp)
        struct.pack_into("<d", self._setup_mm, _SETUP_KD_EXP_OFF, kd_exp)
        struct.pack_into("<d", self._setup_mm, _SETUP_BASE_GRIP_SPD_OFF,
                         base_gripper_speed)
        struct.pack_into("<I", self._setup_mm, _SETUP_GRIP_STRIDE_OFF,
                         gripper_stride)
        struct.pack_into("<I", self._setup_mm, _SETUP_NUM_KP_OFF, len(kp_params))
        struct.pack_into("<I", self._setup_mm, _SETUP_NUM_KD_OFF, len(kd_params))
        struct.pack_into("<I", self._setup_mm, _SETUP_HAS_GRIP_SPD_OFF,
                         1 if gripper_speed_topic else 0)
        _pack_cstr(self._setup_mm, _SETUP_CTRL_NAME_OFF, controller_node, 128)
        _pack_cstr(self._setup_mm, _SETUP_GRIP_SPD_TOPIC_OFF,
                   gripper_speed_topic, 128)
        for i, (name, orig) in enumerate(kp_params + kd_params):
            slot_off = _SETUP_PARAMS_OFF + i * _SETUP_PARAM_SLOT_SIZE
            _pack_cstr(self._setup_mm, slot_off, name, 64)
            struct.pack_into("<d", self._setup_mm, slot_off + 64, orig)
        # Last: stamp the layout version, which also releases the C++ side.
        # A mismatch here is refused by crisp_sender rather than producing motion
        # from misread bytes.
        struct.pack_into("<Q", self._setup_mm, _SETUP_COMPLETE_OFF,
                         _SETUP_COMPLETE_VALUE)

        # Neuter scaler.step_to so the producer's existing cycle-change
        # detection still fires step_to(...) but the Python method is now
        # a no-op (the C++ side does the SetParameters call instead).
        if scaler is not None:
            def _noop(*_args, **_kwargs):
                return None
            scaler.step_to = _noop  # type: ignore[assignment]

    # ----- TargetSenderThread-compatible API -------------------------------

    def start(self) -> None:
        # Block until the C++ subprocess has finished rclcpp init +
        # publisher discovery and set the ready flag. Otherwise the producer
        # pushes the first chunk with deadlines based on time.monotonic()
        # at push-time, the C++ binary catches up ~400 ms later, and the
        # whole first chunk lands as "late frames". Cap at 30 s for sanity.
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"crisp_sender subprocess exited during startup "
                    f"with code {self._proc.returncode}; see "
                    f"{self._stderr_log_path}"
                )
            ready = struct.unpack_from("<Q", self._setup_mm, _SETUP_READY_OFF)[0]
            if ready:
                logger.info("crisp_sender reported ready")
                return
            time.sleep(0.02)
        raise RuntimeError(
            f"crisp_sender did not signal ready within 30 s; see "
            f"{self._stderr_log_path}"
        )

    def put(self, item: Any) -> None:
        """Mirror queue.Queue.put. None = sentinel (end of stream)."""
        # Crash detection on every push.
        rc = self._proc.poll()
        if rc is not None:
            raise RuntimeError(
                f"crisp_sender subprocess died (code {rc}); see "
                f"{self._stderr_log_path}"
            )

        if item is None:
            # Push a sentinel slot.
            self._write_action_slot(
                seqnum=self._action_seqnum,
                deadline_ns=0,
                pose_xyz=(0.0, 0.0, 0.0),
                pose_quat=(0.0, 0.0, 0.0, 1.0),
                grip_raw=0.0,
                flags=_FLAG_SENTINEL,
                frame_idx=0,
                s_eff=0.0,
                cycles=0,
            )
            return

        # Scaler segment-boundary detection: bump scaler seqnum if
        # item.cycles changed.
        if self._prev_cycles is None or item.cycles != self._prev_cycles:
            self._scaler_seqnum += 1
            struct.pack_into("<d", self._scaler_mm, _SC_S_EFF_OFF,
                             float(item.s_eff))
            struct.pack_into("<Q", self._scaler_mm, _SC_REQ_OFF,
                             self._scaler_seqnum)
            self._prev_cycles = item.cycles

        # Write action slot.
        flags = 0
        if item.grip_raw is not None:
            flags |= _FLAG_HAS_GRIPPER
            grip_val = float(item.grip_raw)
        else:
            grip_val = 0.0

        # deadline_mono is in seconds (time.monotonic float). Convert to ns.
        deadline_ns = int(item.deadline_mono * 1e9)

        self._write_action_slot(
            seqnum=self._action_seqnum,
            deadline_ns=deadline_ns,
            pose_xyz=tuple(float(x) for x in item.pose_xyz),
            pose_quat=tuple(float(x) for x in item.pose_quat),
            grip_raw=grip_val,
            flags=flags,
            frame_idx=int(item.frame_idx),
            s_eff=float(item.s_eff),
            cycles=int(item.cycles),
        )

        # Replay log row (matches TargetSenderThread).
        row: dict = {
            "frame_index": item.frame_idx,
            "timestamp": time.time(),
            "replay.s_eff": float(item.s_eff),
            "replay.cycles": int(item.cycles),
            "replay.action": item.action.copy(),
        }
        if self._state_capture_fn is not None:
            self._state_capture_fn(row)
        self._replay_log.append(row)

        # Track queue depth for the producer's logging.
        depth = self.qsize()
        if depth < self.queue_depth_min:
            self.queue_depth_min = depth
        if depth > self.queue_depth_max:
            self.queue_depth_max = depth

    def _write_action_slot(self, *, seqnum: int, deadline_ns: int,
                            pose_xyz: tuple, pose_quat: tuple,
                            grip_raw: float, flags: int, frame_idx: int,
                            s_eff: float, cycles: int) -> None:
        head = struct.unpack_from("<Q", self._action_mm, _AH_HEAD_OFF)[0]
        tail = struct.unpack_from("<Q", self._action_mm, _AH_TAIL_OFF)[0]
        # Block until there's space (queue.Queue semantics for put()).
        while head - tail >= _ACTION_RING_CAPACITY:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"crisp_sender died while waiting for ring slot "
                    f"(code {self._proc.returncode}); see "
                    f"{self._stderr_log_path}"
                )
            time.sleep(0.0005)
            head = struct.unpack_from("<Q", self._action_mm, _AH_HEAD_OFF)[0]
            tail = struct.unpack_from("<Q", self._action_mm, _AH_TAIL_OFF)[0]

        slot_off = _ACTION_HEADER_SIZE + (head % _ACTION_RING_CAPACITY) * _ACTION_SLOT_SIZE
        struct.pack_into("<Q", self._action_mm, slot_off + _AS_SEQNUM_OFF, seqnum)
        struct.pack_into("<Q", self._action_mm, slot_off + _AS_DEADLINE_OFF, deadline_ns)
        struct.pack_into("<fff", self._action_mm, slot_off + _AS_POSE_XYZ_OFF,
                         pose_xyz[0], pose_xyz[1], pose_xyz[2])
        struct.pack_into("<ffff", self._action_mm, slot_off + _AS_POSE_QUAT_OFF,
                         pose_quat[0], pose_quat[1], pose_quat[2], pose_quat[3])
        struct.pack_into("<f", self._action_mm, slot_off + _AS_GRIP_RAW_OFF, grip_raw)
        struct.pack_into("<I", self._action_mm, slot_off + _AS_FLAGS_OFF, flags)
        struct.pack_into("<I", self._action_mm, slot_off + _AS_FRAME_IDX_OFF, frame_idx)
        struct.pack_into("<f", self._action_mm, slot_off + _AS_S_EFF_OFF, s_eff)
        struct.pack_into("<I", self._action_mm, slot_off + _AS_CYCLES_OFF, cycles)
        struct.pack_into("<Q", self._action_mm, _AH_HEAD_OFF, head + 1)
        self._action_seqnum += 1
        self._n_pushed += 1

    def qsize(self) -> int:
        head = struct.unpack_from("<Q", self._action_mm, _AH_HEAD_OFF)[0]
        tail = struct.unpack_from("<Q", self._action_mm, _AH_TAIL_OFF)[0]
        return int(head - tail)

    def is_alive(self) -> bool:
        return self._proc.poll() is None

    def join(self, timeout: Optional[float] = None) -> None:
        # Push stop flag so C++ exits its main loop cleanly.
        struct.pack_into("<Q", self._action_mm, _AH_STOP_OFF, 1)
        try:
            self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.warning("crisp_sender did not exit within %s s; terminating",
                           timeout)
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()

        # Ingest stats SHM into the same shape as TargetSenderThread.
        self._ingest_stats()

        # Cleanup SHM segments.
        try:
            self._stderr_fh.close()
        except Exception:
            pass
        for mm, path in [
            (self._setup_mm, self._setup_path),
            (self._action_mm, self._action_path),
            (self._scaler_mm, self._scaler_path),
            (self._stats_mm, self._stats_path),
        ]:
            try:
                mm.close()
            except Exception:
                pass
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass

    def _ingest_stats(self) -> None:
        head = struct.unpack_from("<Q", self._stats_mm, _SH_HEAD_OFF)[0]
        tail = struct.unpack_from("<Q", self._stats_mm, _SH_TAIL_OFF)[0]
        dropped = struct.unpack_from("<Q", self._stats_mm, _SH_DROPPED_OFF)[0]
        n = head - tail
        if dropped > 0:
            logger.warning("crisp_sender stats ring overflowed %d slots; "
                           "consider increasing _STATS_RING_CAPACITY",
                           dropped)
        # Walk slots oldest to newest. Track previous post_grip for
        # loop_total_ms.
        prev_post_grip_ns: Optional[int] = None
        for i in range(n):
            idx = (tail + i) % _STATS_RING_CAPACITY
            off = _STATS_HEADER_SIZE + idx * _STATS_SLOT_SIZE
            seqnum, deadline_ns, pre_sleep_ns, post_sleep_ns, post_pose_ns, post_grip_ns = (
                struct.unpack_from("<QQQQQQ", self._stats_mm, off)
            )

            slack_ms = (deadline_ns - pre_sleep_ns) * 1e-6
            sleep_target = max(deadline_ns, pre_sleep_ns)
            sleep_overshoot_ms = max(0.0, (post_sleep_ns - sleep_target) * 1e-6)
            pub_pose_ms = max(0.0, (post_pose_ns - post_sleep_ns) * 1e-6)
            pub_grip_ms = max(0.0, (post_grip_ns - post_pose_ns) * 1e-6)
            # pop_ms approx — gap between previous post_grip and this pre_sleep.
            if prev_post_grip_ns is None:
                pop_ms = 0.0
                loop_total_ms = max(0.0, (post_grip_ns - pre_sleep_ns) * 1e-6)
            else:
                pop_ms = max(0.0, (pre_sleep_ns - prev_post_grip_ns) * 1e-6)
                loop_total_ms = max(0.0, (post_grip_ns - prev_post_grip_ns) * 1e-6)
            prev_post_grip_ns = post_grip_ns

            self.slack_samples_ms.append(slack_ms)
            if slack_ms < 0.0:
                self.n_late_frames += 1
                self.underrun_count += 1
            self.stage_samples["pop_ms"].append(pop_ms)
            self.stage_samples["scaler_rpc_ms"].append(0.0)
            self.stage_samples["sleep_overshoot_ms"].append(sleep_overshoot_ms)
            self.stage_samples["pub_pose_ms"].append(pub_pose_ms)
            self.stage_samples["pub_grip_ms"].append(pub_grip_ms)
            self.stage_samples["loop_total_ms"].append(loop_total_ms)
            self.frame_rows.append({
                "frame_idx": seqnum,
                "queue_depth": -1,   # not tracked by C++ stats
                "slack_ms": slack_ms,
                "pop_ms": pop_ms,
                "scaler_rpc_ms": 0.0,
                "sleep_overshoot_ms": sleep_overshoot_ms,
                "pub_pose_ms": pub_pose_ms,
                "pub_grip_ms": pub_grip_ms,
                "loop_total_ms": loop_total_ms,
            })
            self.n_published += 1
            if self._debug_publish:
                self.pub_dt_samples.append(pub_pose_ms / 1000.0)
