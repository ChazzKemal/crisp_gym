#!/usr/bin/env python3
"""Per-stage profile of the deploy-side ``predict_action_chunk`` round-trip.

Loads a real LeRobot ACT checkpoint and times every stage of
``_SyncLerobotPolicy.request()`` (see ``examples/19_deploy_policy.py``):

    1. numpy_obs_to_torch     (CPU->GPU H2D + uint8->float + /255 per frame)
    2. preprocessor           (LeRobot Normalize chain)
    3. predict_action_chunk   (ACT forward pass)
    4. postprocessor          (Unnormalize chain)
    5. cpu_numpy              (.squeeze(0).to('cpu').numpy() - D2H sync)

No ROS, no env, no camera bridge. Synthetic obs uses the exact shapes the
checkpoint declares (image_features) so the preprocessor/normalizer chain
runs without any input-mismatch surprises.

Run::

    pixi run -e jazzy-lerobot python examples/31_predict_chunk_profile.py \\
        --pretrained-path outputs/train/act_infinite/checkpoints/080000/pretrained_model
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import statistics
import sys
import time
from pathlib import Path

import numpy as np


def _pct(samples_ms, q):
    return float(np.percentile(samples_ms, q))


def _summary(samples_ms):
    a = np.asarray(samples_ms)
    return {
        "mean": float(a.mean()),
        "p50": _pct(a, 50),
        "p90": _pct(a, 90),
        "p99": _pct(a, 99),
        "max": float(a.max()),
    }


def _async_worker(conn, pretrained_path: str, device_str: str) -> None:
    """Subprocess mirror of crisp_gym.policy.async_lerobot_policy.inference_worker.

    Loads the policy, then services requests in a loop: recv obs_np ->
    numpy_obs_to_torch -> preprocessor -> predict_action_chunk ->
    postprocessor -> .cpu().numpy() -> send back.

    Exits on receiving None.
    """
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    device = torch.device(device_str)
    cfg = PreTrainedConfig.from_pretrained(pretrained_path)
    if hasattr(cfg, "device"):
        cfg.device = str(device)
    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(pretrained_path, config=cfg)
    policy.reset()
    policy.to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=pretrained_path,
    )
    conn.send("READY")

    while True:
        msg = conn.recv()
        if msg is None:
            return
        obs_np = msg
        with torch.inference_mode():
            batch = {}
            for key, value in obs_np.items():
                if key.startswith("observation.state"):
                    batch[key] = (
                        torch.from_numpy(value).unsqueeze(0).to(device).float()
                    )
                elif key.startswith("observation.images"):
                    batch[key] = (
                        torch.from_numpy(value)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                        .to(device)
                        .float()
                        / 255.0
                    )
            batch = preprocessor(batch)
            chunk = policy.predict_action_chunk(batch)
            chunk = postprocessor(chunk)
            chunk_np = chunk.squeeze(0).to(device="cpu").numpy()
        conn.send(chunk_np)


def run_async_mode(args, obs_np, device) -> None:
    """Time the full AsyncLerobotPolicy-style Pipe round-trip."""
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    proc = ctx.Process(
        target=_async_worker, args=(child, args.pretrained_path, str(device)),
    )
    proc.start()
    try:
        ready = parent.recv()
        assert ready == "READY"
        for _ in range(args.warmup):
            parent.send(obs_np)
            parent.recv()
        samples = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            parent.send(obs_np)
            _ = parent.recv()
            samples.append((time.perf_counter() - t0) * 1000.0)
        parent.send(None)
    finally:
        proc.join(timeout=5.0)
        if proc.is_alive():
            proc.terminate()

    s = _summary(samples)
    print()
    print("Async (subprocess + Pipe, mirrors AsyncLerobotPolicy):")
    print(
        f"  total round-trip: mean={s['mean']:7.2f}  p50={s['p50']:7.2f}  "
        f"p90={s['p90']:7.2f}  p99={s['p99']:7.2f}  max={s['max']:7.2f} ms  (n={len(samples)})"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pretrained-path", type=str,
        default="outputs/train/act_infinite/checkpoints/080000/pretrained_model",
        help="Path to a LeRobot pretrained-model directory.",
    )
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument(
        "--device", type=str, default="auto",
        choices=("auto", "cuda", "cpu"),
        help="Force CPU to verify the CPU-inference hypothesis.",
    )
    ap.add_argument(
        "--also-async", action="store_true",
        help="After the in-process per-stage profile, also run an "
             "AsyncLerobotPolicy-style subprocess+Pipe round-trip to compare "
             "against the deploy run's synth_ms.",
    )
    args = ap.parse_args()

    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.utils.constants import ACTION, OBS_IMAGES
    from lerobot.policies.utils import populate_queues

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Loading policy from {args.pretrained_path} on device {device}")
    cfg = PreTrainedConfig.from_pretrained(args.pretrained_path)
    # The checkpoint config has `device` baked in. Some ACT internals build
    # tensors from cfg.device at instantiation time (latent sample, etc.) —
    # `policy.to(device)` afterwards doesn't catch them. Override before load.
    if hasattr(cfg, "device"):
        cfg.device = str(device)
    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(args.pretrained_path, config=cfg)
    policy.reset()
    policy.to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=args.pretrained_path,
    )
    print(
        f"  type={policy.config.type}  n_obs={policy.config.n_obs_steps}  "
        f"n_act={policy.config.n_action_steps}  chunk_size="
        f"{getattr(policy.config, 'chunk_size', '?')}"
    )

    # Build a synthetic obs that matches the checkpoint's declared feature
    # shapes. We hand it to numpy_obs_to_torch in the same form the producer
    # in 19_deploy_policy.py builds for a real env: HxWx3 uint8 images,
    # 1-D float state.
    img_shape_by_key: dict[str, tuple[int, int, int]] = {}
    for key, ft in policy.config.input_features.items():
        if key.startswith("observation.images."):
            # input_features stores (C, H, W). We want (H, W, 3) for the
            # numpy obs.
            c, h, w = ft.shape
            assert c == 3
            img_shape_by_key[key] = (h, w, c)
    state_dim = int(policy.config.input_features["observation.state"].shape[0])

    rng = np.random.default_rng(0)
    obs_np = {
        key: rng.integers(0, 256, hwc, dtype=np.uint8)
        for key, hwc in img_shape_by_key.items()
    }
    obs_np["observation.state"] = rng.random(state_dim).astype(np.float32)

    has_internal_queue = (
        hasattr(policy, "_queues") and policy._queues is not None
    )

    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)

    def run_once(record=None):
        timings = {}
        sync(); t0 = time.perf_counter()

        # Stage 1: numpy -> torch + H2D + float + /255 (per frame), same as
        # crisp_gym/util/lerobot_features.py:numpy_obs_to_torch.
        batch = {}
        for key, value in obs_np.items():
            if key.startswith("observation.state"):
                batch[key] = (
                    torch.from_numpy(value).unsqueeze(0).to(device).float()
                )
            elif key.startswith("observation.images"):
                batch[key] = (
                    torch.from_numpy(value)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .to(device)
                    .float()
                    / 255.0
                )
        sync(); t1 = time.perf_counter()
        timings["numpy_obs_to_torch_ms"] = (t1 - t0) * 1000.0

        # Stage 2: preprocessor (Normalize chain).
        batch = preprocessor(batch)
        sync(); t2 = time.perf_counter()
        timings["preprocessor_ms"] = (t2 - t1) * 1000.0

        # Stage 3: predict_action_chunk. Includes diffusion queue dance if
        # the policy has _queues; ACT path is the simple branch.
        with torch.inference_mode():
            if has_internal_queue:
                queue_batch = dict(batch)
                queue_batch.pop(ACTION, None)
                if policy.config.image_features:
                    queue_batch[OBS_IMAGES] = torch.stack(
                        [queue_batch[k] for k in policy.config.image_features],
                        dim=-4,
                    )
                populate_queues(policy._queues, queue_batch)
                chunk = policy.predict_action_chunk(queue_batch)
            else:
                chunk = policy.predict_action_chunk(batch)
        sync(); t3 = time.perf_counter()
        timings["predict_action_chunk_ms"] = (t3 - t2) * 1000.0

        # Stage 4: postprocessor (Unnormalize chain).
        chunk = postprocessor(chunk)
        sync(); t4 = time.perf_counter()
        timings["postprocessor_ms"] = (t4 - t3) * 1000.0

        # Stage 5: GPU->CPU sync + numpy materialization.
        out = chunk.squeeze(0).to(device="cpu").numpy()
        sync(); t5 = time.perf_counter()
        timings["cpu_numpy_ms"] = (t5 - t4) * 1000.0

        timings["total_ms"] = (t5 - t0) * 1000.0
        if record is not None:
            for k, v in timings.items():
                record.setdefault(k, []).append(v)
        return out

    print(f"Warmup x{args.warmup} ...")
    for _ in range(args.warmup):
        run_once()
    if device.type == "cuda":
        torch.cuda.synchronize()

    print(f"Profile x{args.iters} ...\n")
    record: dict[str, list[float]] = {}
    for _ in range(args.iters):
        run_once(record)

    stage_order = [
        "numpy_obs_to_torch_ms",
        "preprocessor_ms",
        "predict_action_chunk_ms",
        "postprocessor_ms",
        "cpu_numpy_ms",
        "total_ms",
    ]
    header = f"{'stage':<28}  {'mean':>8} {'p50':>8} {'p90':>8} {'p99':>8} {'max':>8}   {'% of total p50':>15}"
    print(header)
    print("-" * len(header))
    total_p50 = _summary(record["total_ms"])["p50"]
    for stage in stage_order:
        s = _summary(record[stage])
        pct = (s["p50"] / total_p50) * 100.0 if stage != "total_ms" else 100.0
        print(
            f"{stage:<28}  {s['mean']:>8.2f} {s['p50']:>8.2f} {s['p90']:>8.2f} "
            f"{s['p99']:>8.2f} {s['max']:>8.2f}   {pct:>14.1f}%"
        )

    # Free the GPU before the async fork (so the subprocess starts with
    # the GPU empty and we don't OOM running both back-to-back).
    del policy, preprocessor, postprocessor
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if args.also_async:
        run_async_mode(args, obs_np, device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
