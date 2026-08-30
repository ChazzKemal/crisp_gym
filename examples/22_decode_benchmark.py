"""Decode benchmark for crisp_py's image pipeline at 30 Hz.

Reproduces the JPEG-decode + resize work that crisp_py.Camera performs on
every incoming CompressedImage frame, *without* running ROS. Two pipelines
are exercised:

    codec     cv2.imencode -> cv2.imdecode -> cv2.resize
    cvbridge  cv2.imencode -> CvBridge.compressed_imgmsg_to_cv2 -> cv2.resize

The ``cvbridge`` path is what crisp_py actually calls — see
``crisp_py/camera/camera.py`` ``_uncompress`` (line ~163). The ``codec`` path
is the same work without sensor_msgs / cv_bridge overhead, so the delta
between them isolates the cv_bridge cost.

By default we spawn two decoder threads concurrently to mirror the dual-camera
env (``ur10e_ridgeback_dual_cam_env``: ``camera`` + ``d405``). The two threads
share the Python GIL exactly like the two crisp_py.Camera callback threads
in a real deploy, so the per-frame numbers include any GIL contention.

Each thread decodes a pre-encoded JPEG buffer at ``--fps`` for ``--duration``,
busy-paced via ``time.sleep`` to mimic ROS-driven 30 Hz frame arrival. The
script records per-frame decode latency and decode+resize latency, then
reports mean/p50/p95/p99/max + overrun count + effective FPS.

Usage (from ``Yunfei/crisp_gym``):

    pixi run -e jazzy-lerobot python examples/22_decode_benchmark.py
    pixi run -e jazzy-lerobot python examples/22_decode_benchmark.py \\
        --threads 1 2 --duration 10
    pixi run -e jazzy-lerobot python examples/22_decode_benchmark.py \\
        --mode codec --resolutions 1280x720

The ``cvbridge`` mode requires the ROS python bindings to be importable
(cv_bridge + sensor_msgs), which they are inside the ``jazzy-lerobot`` pixi
env. Outside that env, run with ``--mode codec``.
"""

from __future__ import annotations

import argparse
import statistics
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Stats:
    decode_ms: list[float] = field(default_factory=list)
    total_ms: list[float] = field(default_factory=list)
    overruns: int = 0
    elapsed: float = 0.0


def _make_jpeg(h: int, w: int, quality: int) -> bytes:
    """Build a realistic-entropy JPEG buffer at (h, w) and the given quality.

    Pure noise compresses badly and pure gradients compress too well; we
    combine the two so the encoded size — and thus decode cost — is closer
    to a real camera frame than either extreme.
    """
    rng = np.random.default_rng(seed=0)
    img = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    xs = np.linspace(0, 4 * np.pi, w, dtype=np.float32)
    ys = np.linspace(0, 4 * np.pi, h, dtype=np.float32)
    pattern = (np.sin(xs)[None, :] * np.cos(ys)[:, None] * 64.0 + 128.0).astype(np.int32)
    blended = (img.astype(np.int32) // 2 + pattern[..., None] // 2).clip(0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", blended, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return bytes(buf)


def _worker_codec(
    jpeg: bytes,
    target_hw: tuple[int, int],
    n_frames: int,
    period: float,
    stats: Stats,
    barrier: threading.Barrier,
) -> None:
    target_h, target_w = target_hw
    arr = np.frombuffer(jpeg, dtype=np.uint8)
    barrier.wait()
    next_deadline = time.perf_counter()
    for _ in range(n_frames):
        t0 = time.perf_counter()
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        t_dec = time.perf_counter()
        if img is not None and (img.shape[0], img.shape[1]) != (target_h, target_w):
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        t_end = time.perf_counter()
        stats.decode_ms.append((t_dec - t0) * 1000.0)
        stats.total_ms.append((t_end - t0) * 1000.0)
        next_deadline += period
        rem = next_deadline - time.perf_counter()
        if rem > 0:
            time.sleep(rem)
        else:
            stats.overruns += 1


def _worker_cvbridge(
    jpeg: bytes,
    target_hw: tuple[int, int],
    n_frames: int,
    period: float,
    stats: Stats,
    barrier: threading.Barrier,
    cv_bridge,
    CompressedImage,
) -> None:
    target_h, target_w = target_hw
    msg = CompressedImage()
    msg.format = "jpeg"
    msg.data = jpeg
    barrier.wait()
    next_deadline = time.perf_counter()
    for _ in range(n_frames):
        t0 = time.perf_counter()
        img = cv_bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="rgb8")
        t_dec = time.perf_counter()
        if (img.shape[0], img.shape[1]) != (target_h, target_w):
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        t_end = time.perf_counter()
        stats.decode_ms.append((t_dec - t0) * 1000.0)
        stats.total_ms.append((t_end - t0) * 1000.0)
        next_deadline += period
        rem = next_deadline - time.perf_counter()
        if rem > 0:
            time.sleep(rem)
        else:
            stats.overruns += 1


def _run_scenario(
    mode: str,
    src_hw: tuple[int, int],
    tgt_hw: tuple[int, int],
    n_threads: int,
    duration: float,
    fps: float,
    quality: int,
) -> list[Stats]:
    h, w = src_hw
    jpeg = _make_jpeg(h, w, quality)
    n_frames = max(1, int(duration * fps))
    period = 1.0 / fps

    cv_bridge_obj = None
    CompressedImage = None
    if mode == "cvbridge":
        from cv_bridge import CvBridge
        from sensor_msgs.msg import CompressedImage as _CI
        cv_bridge_obj = CvBridge()
        CompressedImage = _CI

    stats_list = [Stats() for _ in range(n_threads)]
    barrier = threading.Barrier(n_threads + 1)
    threads: list[threading.Thread] = []
    for s in stats_list:
        if mode == "codec":
            t = threading.Thread(
                target=_worker_codec,
                args=(jpeg, tgt_hw, n_frames, period, s, barrier),
                daemon=True,
            )
        else:
            t = threading.Thread(
                target=_worker_cvbridge,
                args=(jpeg, tgt_hw, n_frames, period, s, barrier, cv_bridge_obj, CompressedImage),
                daemon=True,
            )
        t.start()
        threads.append(t)

    barrier.wait()
    t0 = time.perf_counter()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t0
    for s in stats_list:
        s.elapsed = elapsed
    return stats_list


def _summarize(stats_list: list[Stats], header: str, n_threads: int, fps: float) -> None:
    decode = [v for s in stats_list for v in s.decode_ms]
    total = [v for s in stats_list for v in s.total_ms]
    overruns = sum(s.overruns for s in stats_list)
    elapsed = stats_list[0].elapsed if stats_list else 0.0
    n_frames = len(decode)
    if n_frames == 0:
        print(header)
        print("  (no frames captured)")
        return
    target_total = int(fps * n_threads * (elapsed if elapsed > 0 else 1.0))

    def pct(arr: list[float], q: float) -> float:
        return float(np.percentile(arr, q))

    print(header)
    print(
        f"  frames: {n_frames}  elapsed: {elapsed:.2f}s  combined FPS: "
        f"{n_frames / elapsed:.1f}  target: {target_total}"
    )
    print(f"  overruns (deadline missed): {overruns}")
    print(
        "  decode-only      ms  "
        f"mean={statistics.fmean(decode):.2f}  "
        f"p50={statistics.median(decode):.2f}  "
        f"p95={pct(decode, 95):.2f}  "
        f"p99={pct(decode, 99):.2f}  "
        f"max={max(decode):.2f}"
    )
    print(
        "  decode + resize  ms  "
        f"mean={statistics.fmean(total):.2f}  "
        f"p50={statistics.median(total):.2f}  "
        f"p95={pct(total, 95):.2f}  "
        f"p99={pct(total, 99):.2f}  "
        f"max={max(total):.2f}"
    )


def _parse_resolution(s: str) -> tuple[int, int]:
    try:
        w_str, h_str = s.lower().split("x")
        return int(h_str), int(w_str)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"bad resolution {s!r} (want WxH)") from e


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--mode",
        choices=["codec", "cvbridge", "both"],
        default="both",
        help="Decode path to benchmark. Default: both.",
    )
    p.add_argument(
        "--resolutions",
        nargs="+",
        type=_parse_resolution,
        default=[_parse_resolution("640x480"), _parse_resolution("1280x720")],
        help="Source resolutions (WxH). Default: 640x480 1280x720.",
    )
    p.add_argument(
        "--target",
        type=_parse_resolution,
        default=_parse_resolution("640x480"),
        help="Target resolution after resize (WxH). Default: 640x480 (matches dual_cam env yaml).",
    )
    p.add_argument(
        "--threads",
        type=int,
        nargs="+",
        default=[2],
        help="Concurrent decoder threads. Default: 2 (dual-cam). Pass `1 2` to compare baselines.",
    )
    p.add_argument("--duration", type=float, default=5.0, help="Seconds per scenario. Default: 5.")
    p.add_argument("--fps", type=float, default=30.0, help="Per-thread pacing rate. Default: 30.")
    p.add_argument("--quality", type=int, default=80, help="JPEG encode quality. Default: 80.")
    args = p.parse_args()

    modes = ["codec", "cvbridge"] if args.mode == "both" else [args.mode]
    tgt_h, tgt_w = args.target

    print(f"target resize -> {tgt_w}x{tgt_h}, jpeg quality {args.quality}, fps {args.fps:.0f}, duration {args.duration:.1f}s")

    for src_h, src_w in args.resolutions:
        for mode in modes:
            for n_threads in args.threads:
                header = (
                    f"\n=== mode={mode}  src={src_w}x{src_h}  tgt={tgt_w}x{tgt_h}  "
                    f"threads={n_threads} ==="
                )
                try:
                    stats = _run_scenario(
                        mode, (src_h, src_w), (tgt_h, tgt_w),
                        n_threads, args.duration, args.fps, args.quality,
                    )
                except ImportError as e:
                    print(header)
                    print(f"  skipped: {e} (run inside `pixi -e jazzy-lerobot` or use --mode codec)")
                    continue
                _summarize(stats, header, n_threads, args.fps)


if __name__ == "__main__":
    main()
