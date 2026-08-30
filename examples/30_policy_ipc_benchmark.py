#!/usr/bin/env python3
"""Benchmark producer<->policy IPC paths for the deploy pipeline.

No ROS, no live cameras, no real model. Synthesises a deploy-shaped obs
(2 RGB frames + a 6-DoF state) and a fake policy that ``time.sleep``s to
mimic the GPU forward pass. Compares three IPC layouts:

  A) pipe_baseline       — parent pickles the whole obs dict through
                            multiprocessing.Pipe (matches the current
                            ``AsyncLerobotPolicy`` path in
                            ``crisp_gym/policy/async_lerobot_policy.py``).
  B) pipe_shm_cameras    — SHM segments are written in the exact wire
                            format of crisp_camera_bridge (header offsets
                            from ``crisp_py.camera.camera``). The producer
                            forwards only a tiny dict
                            ``{state, cams: {name: (seqnum, stamp_ns)}}``
                            through the Pipe; the child mmaps the segments
                            once on startup and seqlock-reads frames on
                            every request. This is the proposed fix:
                            extend the existing SHM bridge across the
                            producer->policy hop.
  C) in_process          — same model work, but the worker runs in the
                            calling process. Lower bound for the IPC layer.

For each variant we time the full ``request()`` round-trip from the parent
perspective (the same span ``19_deploy_policy.py`` records as
``synth_ms``). Every variant exercises an identical child-side workload:
read state, *touch every pixel* of both frames (to force the memory copy
that an image preprocessor would do), sleep ``--model-sleep-ms`` to mimic
the forward pass, and return a (16, 7) float32 chunk.

Run::

    pixi run -e dev python examples/30_policy_ipc_benchmark.py
"""

from __future__ import annotations

import argparse
import mmap
import multiprocessing as mp
import os
import struct
import sys
import time
from dataclasses import dataclass

import numpy as np

# Wire format — must match crisp_py/camera/camera.py exactly.
_SHM_HEADER_SIZE = 32
_SHM_SEQNUM_OFF = 0
_SHM_STAMP_OFF = 8
_SHM_HEIGHT_OFF = 16
_SHM_WIDTH_OFF = 20
_SHM_CHANNELS_OFF = 24
_SHM_DTYPE_OFF = 28
_SHM_MAX_READ_RETRIES = 8

_BENCH_PREFIX = "crisp_camera_bench_"  # never collides with the real bridge


# ---------------------------------------------------------------------------
# SHM helpers (mirrors crisp_camera_bridge write side + camera.py read side).
# ---------------------------------------------------------------------------

def _shm_path(camera_name: str) -> str:
    return f"/dev/shm/{_BENCH_PREFIX}{camera_name}"


def shm_create_and_write(camera_name: str, frame: np.ndarray, stamp_ns: int) -> str:
    """Allocate /dev/shm/<prefix><name> and publish ``frame`` with seqnum=2.

    Uses the seqlock convention: writer bumps to odd before writing, even
    when done. We do one write here and leave seqnum=2 for the rest of the
    benchmark (reader still does the seqlock check; it just always passes).
    """
    assert frame.dtype == np.uint8 and frame.ndim == 3 and frame.shape[-1] == 3
    h, w, c = frame.shape
    total = _SHM_HEADER_SIZE + h * w * c
    path = _shm_path(camera_name)
    if os.path.exists(path):
        os.unlink(path)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        os.ftruncate(fd, total)
        mm = mmap.mmap(fd, total, prot=mmap.PROT_READ | mmap.PROT_WRITE)
    finally:
        os.close(fd)
    try:
        # Bump to odd (mid-write) per the seqlock convention.
        struct.pack_into("<Q", mm, _SHM_SEQNUM_OFF, 1)
        struct.pack_into("<Q", mm, _SHM_STAMP_OFF, stamp_ns)
        struct.pack_into("<i", mm, _SHM_HEIGHT_OFF, h)
        struct.pack_into("<i", mm, _SHM_WIDTH_OFF, w)
        struct.pack_into("<i", mm, _SHM_CHANNELS_OFF, c)
        struct.pack_into("<i", mm, _SHM_DTYPE_OFF, 0)  # rgb8 sentinel
        mm[_SHM_HEADER_SIZE:total] = frame.tobytes()
        # Even seqnum = clean read.
        struct.pack_into("<Q", mm, _SHM_SEQNUM_OFF, 2)
    finally:
        mm.close()
    return path


def shm_open_read(camera_name: str) -> mmap.mmap:
    path = _shm_path(camera_name)
    fd = os.open(path, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size
        return mmap.mmap(fd, size, prot=mmap.PROT_READ)
    finally:
        os.close(fd)


def shm_read(mm: mmap.mmap) -> tuple[np.ndarray, int]:
    """Seqlock read; same retry pattern as crisp_py.camera.Camera._read_shm."""
    last_arr: np.ndarray | None = None
    last_stamp = 0
    for _ in range(_SHM_MAX_READ_RETRIES):
        seq1 = struct.unpack_from("<Q", mm, _SHM_SEQNUM_OFF)[0]
        if seq1 & 1:
            time.sleep(0.00005)
            continue
        if seq1 == 0:
            raise RuntimeError("shm reader: seqnum is 0; writer has not published yet")
        stamp_ns = struct.unpack_from("<Q", mm, _SHM_STAMP_OFF)[0]
        h = struct.unpack_from("<i", mm, _SHM_HEIGHT_OFF)[0]
        w = struct.unpack_from("<i", mm, _SHM_WIDTH_OFF)[0]
        arr = (
            np.frombuffer(mm, dtype=np.uint8, count=h * w * 3, offset=_SHM_HEADER_SIZE)
            .reshape(h, w, 3)
            .copy()
        )
        seq2 = struct.unpack_from("<Q", mm, _SHM_SEQNUM_OFF)[0]
        if seq1 == seq2:
            return arr, stamp_ns
        last_arr, last_stamp = arr, stamp_ns
    if last_arr is not None:
        return last_arr, last_stamp
    raise RuntimeError("shm reader: exhausted retries")


# ---------------------------------------------------------------------------
# Synthetic obs + fake "model" workload (identical across variants).
# ---------------------------------------------------------------------------

def make_obs_frames(h: int, w: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Build two HxWx3 uint8 frames + a 6-DoF state vector."""
    return {
        "observation.images.camera": rng.integers(0, 256, (h, w, 3), dtype=np.uint8),
        "observation.images.d405":   rng.integers(0, 256, (h, w, 3), dtype=np.uint8),
        "observation.state.cartesian": rng.random(6, dtype=np.float32),
    }


def fake_model_work(frames: list[np.ndarray], state: np.ndarray, sleep_ms: float) -> np.ndarray:
    """Touch every pixel (forces memory read), sleep to mimic forward, return chunk.

    The pixel touch is what any real image preprocessor (resize / normalize /
    to-tensor) would do anyway. Without it, the SHM variants would look
    unrealistically cheap because the kernel never has to fault the pages
    into the child's address space.
    """
    s = float(state.sum())
    for fr in frames:
        # uint8 .sum() promotes to int64 internally; this is enough to force
        # the kernel to fault every page of the frame buffer.
        s += float(fr.sum())
    if sleep_ms > 0:
        time.sleep(sleep_ms * 1e-3)
    chunk = np.full((16, 7), s % 1.0, dtype=np.float32)
    return chunk


# ---------------------------------------------------------------------------
# Variant A — Pipe baseline (the current AsyncLerobotPolicy layout).
# ---------------------------------------------------------------------------

def _worker_pipe_baseline(conn, sleep_ms: float) -> None:
    while True:
        msg = conn.recv()
        if msg is None:
            return
        frames = [
            msg["observation.images.camera"],
            msg["observation.images.d405"],
        ]
        state = msg["observation.state.cartesian"]
        chunk = fake_model_work(frames, state, sleep_ms)
        conn.send(chunk)


# ---------------------------------------------------------------------------
# Variant B — Pipe + SHM cameras (proposed fix; producer never copies frames).
# ---------------------------------------------------------------------------

def _worker_pipe_shm(conn, camera_names: list[str], sleep_ms: float) -> None:
    mmaps = {name: shm_open_read(name) for name in camera_names}
    try:
        while True:
            msg = conn.recv()
            if msg is None:
                return
            frames = []
            for name in camera_names:
                arr, _stamp = shm_read(mmaps[name])
                frames.append(arr)
            state = msg["state"]
            chunk = fake_model_work(frames, state, sleep_ms)
            conn.send(chunk)
    finally:
        for mm in mmaps.values():
            mm.close()


# ---------------------------------------------------------------------------
# Variant C — in-process (lower bound).
# ---------------------------------------------------------------------------

def _in_process_request(obs: dict, sleep_ms: float) -> np.ndarray:
    frames = [obs["observation.images.camera"], obs["observation.images.d405"]]
    state = obs["observation.state.cartesian"]
    return fake_model_work(frames, state, sleep_ms)


# ---------------------------------------------------------------------------
# Driver + reporting.
# ---------------------------------------------------------------------------

@dataclass
class VariantStats:
    name: str
    samples_ms: list[float]
    notes: str = ""

    def percentiles(self) -> dict[str, float]:
        arr = np.asarray(self.samples_ms)
        return {
            "n": len(arr),
            "mean": float(arr.mean()),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p99": float(np.percentile(arr, 99)),
            "max": float(arr.max()),
        }


def run_pipe_baseline(args, obs: dict) -> VariantStats:
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    proc = ctx.Process(target=_worker_pipe_baseline, args=(child, args.model_sleep_ms))
    proc.start()
    try:
        for _ in range(args.warmup):
            parent.send(obs)
            parent.recv()
        samples: list[float] = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            parent.send(obs)
            _ = parent.recv()
            samples.append((time.perf_counter() - t0) * 1000.0)
        parent.send(None)
    finally:
        proc.join(timeout=2.0)
        if proc.is_alive():
            proc.terminate()
    return VariantStats(
        "A pipe_baseline",
        samples,
        "obs dict (incl. 2 frames) pickled through mp.Pipe; matches current AsyncLerobotPolicy",
    )


def run_pipe_shm(args, obs: dict) -> VariantStats:
    camera_names = ["camera", "d405"]
    # Pre-publish each camera's SHM segment once. In real life the bridge
    # would keep republishing at camera FPS; we leave seqnum=2 for the whole
    # bench because the seqlock cost is in the read, not the write.
    paths: list[str] = []
    for name in camera_names:
        frame = obs[f"observation.images.{name}"]
        paths.append(shm_create_and_write(name, frame, stamp_ns=time.time_ns()))

    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    proc = ctx.Process(
        target=_worker_pipe_shm,
        args=(child, camera_names, args.model_sleep_ms),
    )
    proc.start()
    try:
        small_msg = {
            "state": obs["observation.state.cartesian"],
            "cams": {n: (2, 0) for n in camera_names},
        }
        for _ in range(args.warmup):
            parent.send(small_msg)
            parent.recv()
        samples: list[float] = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            parent.send(small_msg)
            _ = parent.recv()
            samples.append((time.perf_counter() - t0) * 1000.0)
        parent.send(None)
    finally:
        proc.join(timeout=2.0)
        if proc.is_alive():
            proc.terminate()
        for p in paths:
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass
    return VariantStats(
        "B pipe_shm_cameras",
        samples,
        "Pipe carries only state + (seqnum, stamp); child mmaps /dev/shm bridge segments",
    )


def run_in_process(args, obs: dict) -> VariantStats:
    for _ in range(args.warmup):
        _in_process_request(obs, args.model_sleep_ms)
    samples: list[float] = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        _in_process_request(obs, args.model_sleep_ms)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return VariantStats(
        "C in_process",
        samples,
        "no IPC; pixel-touch + sleep only",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--frame-h", type=int, default=480)
    ap.add_argument("--frame-w", type=int, default=640)
    ap.add_argument(
        "--model-sleep-ms", type=float, default=20.0,
        help="Synthetic model latency; 20ms ~= ACT pure forward pass.",
    )
    ap.add_argument(
        "--also-no-sleep", action="store_true",
        help="Also run a model-sleep-ms=0 pass to isolate pure IPC overhead.",
    )
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    obs = make_obs_frames(args.frame_h, args.frame_w, rng)
    frame_bytes = args.frame_h * args.frame_w * 3
    print(
        f"Bench config: 2 x {args.frame_h}x{args.frame_w}x3 uint8 frames "
        f"({2 * frame_bytes / 1e6:.2f} MB obs payload), "
        f"model_sleep_ms={args.model_sleep_ms}, iters={args.iters}, warmup={args.warmup}",
    )
    print()

    passes = [(args.model_sleep_ms, "with model sleep")]
    if args.also_no_sleep:
        passes.append((0.0, "no model sleep (IPC-only)"))

    for sleep_ms, label in passes:
        args.model_sleep_ms = sleep_ms
        print(f"=== {label} (sleep={sleep_ms} ms) ===")
        stats = [
            run_pipe_baseline(args, obs),
            run_pipe_shm(args, obs),
            run_in_process(args, obs),
        ]

        header = f"{'variant':<22} {'n':>4} {'mean':>8} {'p50':>8} {'p90':>8} {'p99':>8} {'max':>8}"
        print(header)
        print("-" * len(header))
        for s in stats:
            p = s.percentiles()
            print(
                f"{s.name:<22} {int(p['n']):>4d} "
                f"{p['mean']:>8.2f} {p['p50']:>8.2f} {p['p90']:>8.2f} "
                f"{p['p99']:>8.2f} {p['max']:>8.2f}  ms"
            )

        # Headline: speedup vs baseline at p50.
        baseline_p50 = stats[0].percentiles()["p50"]
        print()
        print(f"Speedup vs baseline at p50:")
        for s in stats[1:]:
            p50 = s.percentiles()["p50"]
            print(f"  {s.name:<22}: {baseline_p50 / p50:.2f}x  ({baseline_p50 - p50:.2f} ms saved)")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
