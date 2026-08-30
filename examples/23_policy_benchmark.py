"""Policy-only benchmark: obs -> network -> action chunk, no ROS, no env.

Three modes (``--mode``):

  inprocess (default)
      Run ``_ShadowACTPolicy.predict()`` in the same Python process. Measures
      pure model compute (obs->batch, ACT forward, .cpu().numpy()). No IPC.

  ipc-echo
      Spawn a worker subprocess via ``multiprocessing.Pipe`` that mirrors the
      ``AsyncLerobotPolicy`` wire protocol (parent sends
      ``{"type": "OBS_SEQ", "obs_seq": [...]}``; worker replies with an
      ndarray chunk) but runs NO model. Isolates the pure IPC cost: pickle
      serialization + Pipe transport of ``n_obs * one full obs dict``.

  real
      Run the actual ``crisp_gym.policy.async_lerobot_policy.AsyncLerobotPolicy``
      against a real LeRobot checkpoint and time the parent->child->parent
      round-trip (``--pretrained-path`` required). Measures the production
      cost: IPC + LeRobot processors + ACT forward + chunk back.

Comparing modes for the same n_obs/n_act:

  real_total  ≈  ipc_echo_total  +  model_forward_steady_state
                                       (plus LeRobot Normalize/preprocess overhead)

So a gap between (ipc-echo + inprocess) and real is the LeRobot pre/post
processing cost; a gap between real and your live deploy is everything outside
this script (camera callbacks, env._get_obs, queue drain, etc.).

Usage (from ``Yunfei/crisp_gym``):

    pixi run -e jazzy-lerobot python examples/23_policy_benchmark.py
    pixi run -e jazzy-lerobot python examples/23_policy_benchmark.py --mode ipc-echo
    pixi run -e jazzy-lerobot python examples/23_policy_benchmark.py \\
        --mode real --pretrained-path /path/to/lerobot/checkpoint
"""

from __future__ import annotations

import cv2  # noqa: F401  (before torch/lerobot: pins libjpeg — see 29_clean_deploy.py)

import argparse
import importlib.util
import logging
import statistics
import sys
import time
from pathlib import Path

import numpy as np


logger = logging.getLogger(__name__)


_DEPLOY19_PATH = Path(__file__).parent / "19_deploy_policy.py"


def _load_deploy19():
    spec = importlib.util.spec_from_file_location("deploy19", _DEPLOY19_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to locate 19_deploy_policy.py at {_DEPLOY19_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["deploy19"] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_fake_obs(
    rng: np.random.Generator,
    cam_keys: tuple[str, ...],
    cam_hw: tuple[int, int],
    state_dims: dict[str, int],
) -> dict:
    """Build a single obs dict with random uint8 frames + random float state.

    Same shape/dtype contract as ``ManipulatorEnv._get_obs()`` for the
    dual-cam env (see ``_build_obs_schema`` in 19_deploy_policy.py).
    """
    obs: dict = {}
    h, w = cam_hw
    for key in cam_keys:
        obs[key] = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    for key, dim in state_dims.items():
        obs[key] = rng.standard_normal(dim).astype(np.float32)
    obs["task"] = ""
    return obs


def _obs_seq_size_bytes(
    n_obs: int, cam_keys: tuple[str, ...], cam_hw: tuple[int, int], state_dims: dict[str, int],
) -> int:
    """Approximate wire size of a single send (n_obs obs dicts)."""
    h, w = cam_hw
    per_obs = len(cam_keys) * h * w * 3 + 4 * sum(state_dims.values())
    return n_obs * per_obs


def _summarize(label: str, samples_ms: list[float]) -> None:
    if not samples_ms:
        print(f"{label}: (no samples)")
        return
    arr = np.asarray(samples_ms)
    print(label)
    print(
        "  total per-request (ms)   "
        f"mean={statistics.fmean(samples_ms):.2f}  "
        f"p50={statistics.median(samples_ms):.2f}  "
        f"p95={float(np.percentile(arr, 95)):.2f}  "
        f"p99={float(np.percentile(arr, 99)):.2f}  "
        f"max={max(samples_ms):.2f}  "
        f"min={min(samples_ms):.2f}"
    )


def _echo_worker(conn, n_act: int, action_dim: int, seed: int = 0) -> None:
    """Worker for ``--mode ipc-echo``.

    Mirrors the ``AsyncLerobotPolicy`` wire protocol minus the model: receive
    one message, send back one ``(n_act, action_dim)`` float32 ndarray. Loop
    until ``None`` is received or the pipe closes.
    """
    import numpy as _np
    rng = _np.random.default_rng(seed)
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            break
        if msg is None:
            break
        chunk = rng.standard_normal((n_act, action_dim)).astype(_np.float32)
        try:
            conn.send(chunk)
        except (BrokenPipeError, OSError):
            break
    try:
        conn.close()
    except Exception:
        pass


def run_inprocess(
    args, cam_keys: tuple[str, ...], cam_hw: tuple[int, int], state_dims: dict[str, int],
) -> int:
    rng = np.random.default_rng(args.seed)
    obs_sample = _make_fake_obs(rng, cam_keys, cam_hw, state_dims)

    deploy19 = _load_deploy19()
    ShadowACTPolicy = deploy19._ShadowACTPolicy

    import torch

    policy = ShadowACTPolicy(
        obs_sample=obs_sample,
        n_act=args.n_act,
        action_dim=args.action_dim,
        device=args.device,
        temporal_ensemble_coeff=None,
    )
    on_cuda = policy.device.startswith("cuda") and torch.cuda.is_available()
    policy._predict_debug_count = 0  # re-enable per-stage log for our calls

    print(
        f"policy: flavor={policy.flavor} device={policy.device} "
        f"img_keys={policy._img_keys} state_dim={policy._state_total_dim}"
    )
    print(f"running {args.n_requests} timed predict() call(s) on fresh random obs...\n")

    totals_ms: list[float] = []
    for i in range(args.n_requests):
        if args.request_interval > 0 and i > 0:
            time.sleep(args.request_interval)
        obs = _make_fake_obs(rng, cam_keys, cam_hw, state_dims)
        if on_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        chunk = policy.predict(obs)
        if on_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        totals_ms.append((t1 - t0) * 1000.0)
        print(
            f"  request[{i}]: chunk.shape={chunk.shape} dtype={chunk.dtype} "
            f"wall={(t1 - t0) * 1000.0:.2f} ms"
        )
    _summarize(f"\n=== inprocess summary over {len(totals_ms)} requests ===", totals_ms)
    policy.shutdown()
    return 0


def run_ipc_echo(
    args, cam_keys: tuple[str, ...], cam_hw: tuple[int, int], state_dims: dict[str, int],
) -> int:
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    proc = ctx.Process(
        target=_echo_worker,
        args=(child_conn, args.n_act, args.action_dim, args.seed + 1000),
        daemon=True,
    )
    proc.start()

    rng = np.random.default_rng(args.seed)
    payload_bytes = _obs_seq_size_bytes(args.n_obs, cam_keys, cam_hw, state_dims)
    print(
        f"ipc-echo: n_obs={args.n_obs}  obs_seq payload ~= {payload_bytes / 1e6:.2f} MB "
        f"(raw uint8+float32, before pickle framing)"
    )

    try:
        # Warmup: amortize first pickle / cold pipe.
        obs_seq = [_make_fake_obs(rng, cam_keys, cam_hw, state_dims) for _ in range(args.n_obs)]
        t0 = time.perf_counter()
        parent_conn.send({"type": "OBS_SEQ", "obs_seq": obs_seq})
        _ = parent_conn.recv()
        print(f"ipc-echo warmup round-trip: {(time.perf_counter() - t0) * 1000.0:.2f} ms\n")

        totals_ms: list[float] = []
        for i in range(args.n_requests):
            if args.request_interval > 0 and i > 0:
                time.sleep(args.request_interval)
            obs_seq = [_make_fake_obs(rng, cam_keys, cam_hw, state_dims) for _ in range(args.n_obs)]
            t0 = time.perf_counter()
            parent_conn.send({"type": "OBS_SEQ", "obs_seq": obs_seq})
            chunk = parent_conn.recv()
            t1 = time.perf_counter()
            totals_ms.append((t1 - t0) * 1000.0)
            print(
                f"  request[{i}]: chunk.shape={chunk.shape} dtype={chunk.dtype} "
                f"wall={(t1 - t0) * 1000.0:.2f} ms"
            )
        _summarize(f"\n=== ipc-echo summary over {len(totals_ms)} requests ===", totals_ms)
    finally:
        try:
            parent_conn.send(None)
        except (BrokenPipeError, OSError):
            pass
        proc.join(timeout=2.0)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=1.0)
    return 0


def run_real_async(
    args, cam_keys: tuple[str, ...], cam_hw: tuple[int, int], state_dims: dict[str, int],
) -> int:
    if not args.pretrained_path:
        raise SystemExit("--mode real requires --pretrained-path")
    if not Path(args.pretrained_path).exists():
        raise SystemExit(f"--pretrained-path does not exist: {args.pretrained_path}")

    from crisp_gym.policy.async_lerobot_policy import AsyncLerobotPolicy

    # env=None is safe: the worker accepts the kwarg but never reads it, and
    # we talk to parent_conn directly (same pattern _LeRobotChunkSource uses
    # in 19_deploy_policy.py), so make_data_fn() — the only consumer of
    # self.env — is never invoked.
    # Forward --n-act like 19_deploy_policy.py does. Without this the policy
    # falls back to the checkpoint's n_action_steps, which for a chunk_size=50
    # ACT equals the horizon and trips the worker's `steps < horizon` check.
    policy = AsyncLerobotPolicy(
        pretrained_path=args.pretrained_path,
        env=None,
        n_action_steps=args.n_act,
    )
    print(
        f"AsyncLerobotPolicy: n_obs={policy.n_obs}  n_act={policy.n_act}  "
        f"replan_time={policy.replan_time}  inpainting={policy.inpainting}"
    )
    n_obs = policy.n_obs

    rng = np.random.default_rng(args.seed)
    payload_bytes = _obs_seq_size_bytes(n_obs, cam_keys, cam_hw, state_dims)
    print(
        f"real: obs_seq payload ~= {payload_bytes / 1e6:.2f} MB per request "
        f"(raw uint8+float32, before pickle framing)\n"
    )

    try:
        # Warmup: pays the worker's policy_cls.from_pretrained startup the
        # first time it tries to satisfy a request, plus the first ACT forward
        # (cudnn benchmark + kernel JIT inside the child process).
        obs_seq = [_make_fake_obs(rng, cam_keys, cam_hw, state_dims) for _ in range(n_obs)]
        t0 = time.perf_counter()
        policy.parent_conn.send({"type": "OBS_SEQ", "obs_seq": obs_seq})
        _ = policy.recv_chunk()
        print(f"real warmup round-trip: {(time.perf_counter() - t0) * 1000.0:.2f} ms\n")

        totals_ms: list[float] = []
        for i in range(args.n_requests):
            if args.request_interval > 0 and i > 0:
                time.sleep(args.request_interval)
            obs_seq = [_make_fake_obs(rng, cam_keys, cam_hw, state_dims) for _ in range(n_obs)]
            t0 = time.perf_counter()
            policy.parent_conn.send({"type": "OBS_SEQ", "obs_seq": obs_seq})
            chunk = policy.recv_chunk()
            t1 = time.perf_counter()
            totals_ms.append((t1 - t0) * 1000.0)
            print(
                f"  request[{i}]: chunk.shape={tuple(chunk.shape)} dtype={chunk.dtype} "
                f"wall={(t1 - t0) * 1000.0:.2f} ms"
            )
        _summarize(f"\n=== real-async summary over {len(totals_ms)} requests ===", totals_ms)
    finally:
        try:
            policy.shutdown()
        except Exception:
            logger.exception("policy.shutdown() raised")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--mode",
        choices=["inprocess", "ipc-echo", "real"],
        default="inprocess",
        help="Which path to benchmark. See module docstring.",
    )
    p.add_argument(
        "--state-dims", nargs="+", default=None, metavar="KEY=DIM",
        help="Override the fake observation.state.* keys and dims, e.g. "
             "--state-dims observation.state.cartesian=6. Must sum to the state "
             "width the checkpoint was trained on (config.json input_features), "
             "or the worker raises a dim mismatch. Default: cartesian=6 joints=6 "
             "gripper=1 (13 total).",
    )
    p.add_argument("--n-requests", type=int, default=3, help="Timed requests after warmup. Default: 3.")
    p.add_argument(
        "--request-interval", type=float, default=0.0, metavar="SEC",
        help="Sleep this long between timed requests. Default 0 = back-to-back, "
             "which keeps the GPU clocked up and UNDERSTATES real deploy latency. "
             "In deploy, chunks are ~n_act/fps seconds apart (e.g. 50/20 = 2.5s) and "
             "the GPU downclocks to idle in between. Set this to that cadence to "
             "reproduce the real duty cycle.",
    )
    p.add_argument("--n-act", type=int, default=32, help="Chunk size (used by inprocess and ipc-echo).")
    p.add_argument("--action-dim", type=int, default=7, help="Action dimensionality.")
    p.add_argument(
        "--n-obs",
        type=int,
        default=2,
        help="Obs window length for ipc-echo (real mode reads it from AsyncLerobotPolicy). Default: 2.",
    )
    p.add_argument(
        "--cameras",
        nargs="+",
        default=["camera", "d405"],
        help="Camera names. Default: camera d405 (dual_cam env).",
    )
    p.add_argument(
        "--cam-resolution",
        default="640x480",
        help="Camera frame resolution WxH (HWC uint8). Default: 640x480.",
    )
    p.add_argument("--device", default=None, help="Override torch device (inprocess only).")
    p.add_argument("--pretrained-path", default=None, help="LeRobot checkpoint dir (real mode).")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        w_str, h_str = args.cam_resolution.lower().split("x")
        cam_hw = (int(h_str), int(w_str))
    except ValueError:
        raise SystemExit(f"bad --cam-resolution {args.cam_resolution!r} (want WxH)")

    cam_keys = tuple(f"observation.images.{name}" for name in args.cameras)
    if args.state_dims:
        state_dims = {}
        for spec in args.state_dims:
            if "=" not in spec:
                raise SystemExit(f"--state-dims expects KEY=DIM, got {spec!r}")
            k, _, v = spec.partition("=")
            state_dims[k] = int(v)
    else:
        state_dims = {
            "observation.state.cartesian": 6,
            "observation.state.joints": 6,
            "observation.state.gripper": 1,
        }

    print(
        f"mode: {args.mode}  cameras: {list(cam_keys)}  frame: {cam_hw[1]}x{cam_hw[0]} HWC uint8  "
        f"request_interval: {args.request_interval}s  "
        f"state: {list(state_dims)}  chunk: n_act={args.n_act} action_dim={args.action_dim}"
    )

    if args.mode == "inprocess":
        return run_inprocess(args, cam_keys, cam_hw, state_dims)
    if args.mode == "ipc-echo":
        return run_ipc_echo(args, cam_keys, cam_hw, state_dims)
    if args.mode == "real":
        return run_real_async(args, cam_keys, cam_hw, state_dims)
    raise SystemExit(f"unknown --mode {args.mode}")


if __name__ == "__main__":
    sys.exit(main())
