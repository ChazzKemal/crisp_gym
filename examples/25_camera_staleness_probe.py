#!/usr/bin/env python3
"""Camera SHM staleness probe.

Watches the /dev/shm/crisp_camera_<name> segments populated by
`crisp_camera_bridge` and prints percentiles of:

  age_ms        wall_now - msg.header.stamp_ns at sample time.
                Captures the FULL lag: publisher backdating + DDS
                transport + bridge decode + bridge write → reader.

  gap_ms        Wall-clock time between successive bridge writes (we
                detect a write by seeing seqnum advance). Inverse =
                effective bridge update rate.

  seqs_per_tick How many bridge writes happened per probe sample. >1 means
                the bridge is faster than --rate (we missed intermediate
                frames); 0 means a stall.

  read_us       µs to seqlock-read the segment. Sanity check that the
                reader is not the bottleneck.

This script does NOT use rclpy / crisp_py.Camera. It opens /dev/shm
directly so its own work cannot pollute the measurement.

Reading the output:

  * Large age_ms with small gap_ms → publisher is back-dating stamps.
    Real Orbbec/RealSense drivers do this (stamp = sensor capture time,
    not publish time). For fake_sensors.py, stamps are publish time so
    age_ms ≈ bridge processing + DDS transport.
  * Large gap_ms → bridge or its publisher is dropping/stalling.
  * Large age_ms AND large gap_ms → bridge is decoding too slowly to
    keep up with publisher rate (look for CPU contention).
  * Large read_us → mmap / seqlock contention, but should be <100µs.

Usage:
    pixi run -e jazzy-lerobot python examples/25_camera_staleness_probe.py
    pixi run -e jazzy-lerobot python examples/25_camera_staleness_probe.py --duration 30 --rate 200
    pixi run -e jazzy-lerobot python examples/25_camera_staleness_probe.py --segments crisp_camera_camera

The bridge must already be running:
    ./tools/master_launch.sh camera-bridge
"""

import argparse
import glob
import mmap
import os
import statistics
import struct
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

# Must match crisp_py/camera/camera.py and crisp_camera_bridge.py.
HEADER_SIZE = 32
SEQNUM_OFF = 0
STAMP_OFF = 8
HEIGHT_OFF = 16
WIDTH_OFF = 20


@dataclass
class SegmentTracker:
    """Per-segment rolling stats."""

    path: str
    mmap: mmap.mmap
    h: int
    w: int

    # None until the first non-odd sample arrives. Without this, the first
    # sample would count every bridge write since the bridge process
    # started as "happened during this probe window".
    last_seqnum: Optional[int] = None
    last_update_wall_ns: Optional[int] = None

    ages_ms: list[float] = field(default_factory=list)
    gaps_ms: list[float] = field(default_factory=list)
    read_us: list[float] = field(default_factory=list)
    seqs_per_tick: list[int] = field(default_factory=list)
    ticks: int = 0
    # When seqnum was odd at probe time (writer mid-update).
    odd_seqnum_hits: int = 0


def open_segment(path: str) -> SegmentTracker:
    fd = os.open(path, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size
        mm = mmap.mmap(fd, size, prot=mmap.PROT_READ)
    finally:
        os.close(fd)
    h = struct.unpack_from("<i", mm, HEIGHT_OFF)[0]
    w = struct.unpack_from("<i", mm, WIDTH_OFF)[0]
    return SegmentTracker(path=path, mmap=mm, h=h, w=w)


def probe_once(tracker: SegmentTracker, now_ns: int) -> None:
    """Sample the segment and update the tracker's rolling stats."""
    t0 = time.perf_counter_ns()
    seq = struct.unpack_from("<Q", tracker.mmap, SEQNUM_OFF)[0]
    if seq & 1:
        # Writer is mid-update; record this and skip — don't pollute
        # age stats with a torn header (stamp may not be written yet).
        tracker.odd_seqnum_hits += 1
        return
    stamp_ns = struct.unpack_from("<Q", tracker.mmap, STAMP_OFF)[0]
    t1 = time.perf_counter_ns()
    tracker.read_us.append((t1 - t0) / 1000.0)
    tracker.ticks += 1

    if seq == 0:
        # Bridge hasn't written yet.
        tracker.seqs_per_tick.append(0)
        return

    # First sample: don't count pre-existing seqnum as "writes during the
    # probe window". Just baseline and wait for the next tick.
    if tracker.last_seqnum is None:
        tracker.last_seqnum = seq
        tracker.last_update_wall_ns = now_ns
        # Record the very first frame's age so the user sees a value
        # even on short probes.
        tracker.ages_ms.append((now_ns - stamp_ns) * 1e-6)
        tracker.seqs_per_tick.append(0)
        return

    # Each bridge write bumps seqnum by 2 (odd→even). So a delta of 2
    # means one new frame, 4 means two, etc.
    delta = seq - tracker.last_seqnum
    new_frames = max(0, delta // 2)
    tracker.seqs_per_tick.append(new_frames)

    if new_frames > 0:
        age_ms = (now_ns - stamp_ns) * 1e-6
        tracker.ages_ms.append(age_ms)
        if tracker.last_update_wall_ns is not None:
            gap_ms = (now_ns - tracker.last_update_wall_ns) * 1e-6
            tracker.gaps_ms.append(gap_ms)
        tracker.last_update_wall_ns = now_ns
        tracker.last_seqnum = seq


def _percentiles(xs: list[float]) -> str:
    if not xs:
        return "(no samples)"
    xs_sorted = sorted(xs)
    n = len(xs_sorted)

    def p(q: float) -> float:
        idx = max(0, min(n - 1, int(q * n)))
        return xs_sorted[idx]

    return (
        f"n={n} mean={statistics.mean(xs):.2f} "
        f"median={p(0.5):.2f} p90={p(0.9):.2f} "
        f"p99={p(0.99):.2f} max={xs_sorted[-1]:.2f} min={xs_sorted[0]:.2f}"
    )


def print_report(trackers: list[SegmentTracker], duration_s: float) -> None:
    print()
    print("=" * 78)
    print(f"Probe duration: {duration_s:.2f} s")
    print("=" * 78)
    for t in trackers:
        new_frame_count = sum(t.seqs_per_tick)
        update_hz = new_frame_count / duration_s if duration_s > 0 else 0.0
        sample_hz = t.ticks / duration_s if duration_s > 0 else 0.0
        print(f"\n[{os.path.basename(t.path)}]  segment={t.h}x{t.w}")
        print(f"  probe samples: {t.ticks} ({sample_hz:.1f} Hz)")
        print(f"  bridge writes seen: {new_frame_count} ({update_hz:.1f} Hz effective)")
        print(f"  odd-seqnum hits (writer mid-update): {t.odd_seqnum_hits}")
        print(f"  age_ms        {_percentiles(t.ages_ms)}")
        print(f"  gap_ms        {_percentiles(t.gaps_ms)}")
        print(f"  seqs_per_tick {_percentiles([float(x) for x in t.seqs_per_tick])}")
        print(f"  read_us       {_percentiles(t.read_us)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--segments",
        nargs="*",
        default=None,
        help="SHM segment basenames to probe (e.g. crisp_camera_camera "
        "crisp_camera_d405). If omitted, auto-discovers /dev/shm/crisp_camera_*.",
    )
    parser.add_argument(
        "--duration", type=float, default=10.0,
        help="Total probe duration in seconds (default: 10).",
    )
    parser.add_argument(
        "--rate", type=float, default=100.0,
        help="Probe sample rate in Hz (default: 100). Should be > expected "
        "bridge rate (30 Hz) to avoid missing intermediate writes.",
    )
    parser.add_argument(
        "--shm-dir", default="/dev/shm",
        help="Directory containing the SHM segments (default: /dev/shm).",
    )
    args = parser.parse_args()

    if args.segments is None:
        paths = sorted(glob.glob(os.path.join(args.shm_dir, "crisp_camera_*")))
        if not paths:
            print(f"No /dev/shm/crisp_camera_* segments found in {args.shm_dir}.",
                  file=sys.stderr)
            print("Start the bridge first:", file=sys.stderr)
            print("  ./tools/master_launch.sh camera-bridge", file=sys.stderr)
            print("or directly:", file=sys.stderr)
            print("  cd clearpath_remote_ws && pixi run python "
                  "src/tum09_ridgeback/tum09_custom/scripts/crisp_camera_bridge.py "
                  "--config src/tum09_ridgeback/tum09_custom/config/crisp_camera_bridge.yaml",
                  file=sys.stderr)
            return 1
    else:
        paths = [
            seg if os.path.isabs(seg) else os.path.join(args.shm_dir, seg)
            for seg in args.segments
        ]

    trackers: list[SegmentTracker] = []
    for path in paths:
        if not os.path.exists(path):
            print(f"Segment not found: {path}", file=sys.stderr)
            return 1
        trackers.append(open_segment(path))
        print(f"watching {path} ({trackers[-1].h}x{trackers[-1].w})")

    period_s = 1.0 / args.rate
    print(f"probing at {args.rate:.1f} Hz for {args.duration:.1f} s "
          f"(period {period_s * 1000:.2f} ms)")

    t_start = time.monotonic()
    t_end = t_start + args.duration
    next_tick = time.monotonic()
    try:
        while True:
            now_mono = time.monotonic()
            if now_mono >= t_end:
                break
            now_ns = time.time_ns()
            for tracker in trackers:
                probe_once(tracker, now_ns)
            next_tick += period_s
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                # Probe is falling behind; reset the schedule.
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        pass

    print_report(trackers, time.monotonic() - t_start)
    return 0


if __name__ == "__main__":
    sys.exit(main())
