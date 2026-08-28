#!/usr/bin/env python3
"""Find out which half of a B-spline prediction is wrong: knots or control points.

Decoding mixes the two, so a bad waypoint tells you nothing about the cause.
This swaps one half of the prediction for ground truth at a time:

    both predicted     what the policy actually delivers
    GT knots           how good are the predicted control points alone?
    GT control points  how good is the predicted knot column alone?

It also reports per-channel error in *normalised* units, which is what the
training loss is actually minimising, so the two can be compared.

    conda run -n lerobot-041 python diagnose_prediction.py --ckpt <PRETRAINED_MODEL>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bspline_core.bspline_action import decode_bspline_action  # noqa: E402
from bspline_core.knots import safer_knots  # noqa: E402
from lerobot_bridge import load_lerobot_actions  # noqa: E402


def decode_pos(params, degree, n=16):
    p = np.array(params, dtype=np.float64, copy=True)
    p[:, 0] = safer_knots(p[:, 0])
    t_min, t_max = p[degree, 0], p[-(degree + 1), 0]
    if not (t_max > t_min):
        return None
    return decode_bspline_action(p, degree=degree, num_actions=n)[:, :3]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--dataset-root", type=Path,
                    default=Path("/home/batur/Coding/data/merged_bspline_20260528"))
    ap.add_argument("--n-samples", type=int, default=150)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from policy_io import load_policy, predict_action

    meta = json.loads((args.dataset_root / "meta" / "bspline.json").read_text())
    NS, NC, K = meta["n_action_steps"], meta["n_action_channels"], meta["degree"]
    conv = load_lerobot_actions(args.dataset_root)

    ds = LeRobotDataset(args.dataset_root.name, root=args.dataset_root, video_backend="pyav")
    policy, pre, post = load_policy(args.ckpt, args.device)

    stats = json.loads((args.dataset_root / "meta" / "stats.json").read_text())["action"]
    mean = np.asarray(stats["mean"]).reshape(NS, NC)
    std = np.asarray(stats["std"]).reshape(NS, NC)

    rng = np.random.default_rng(0)
    idx = rng.choice(len(ds), size=args.n_samples, replace=False)

    err = {"both predicted": [], "GT knots": [], "GT control points": []}
    spans = {"predicted": [], "truth": []}
    nz = []                      # per-channel normalised abs error
    knot_diff_pred, knot_diff_true = [], []

    for i in idx:
        sample = ds[int(i)]
        pred = predict_action(policy, pre, post, sample, args.device).reshape(NS, NC)
        true = conv.actions[int(sample["index"])].reshape(NS, NC).astype(np.float64)

        nz.append(np.abs((pred - true) / std).mean(axis=0))
        knot_diff_pred.append(np.diff(pred[:, 0]))
        knot_diff_true.append(np.diff(true[:, 0]))
        spans["predicted"].append(pred[NS - K - 1, 0] - pred[K, 0])
        spans["truth"].append(true[NS - K - 1, 0] - true[K, 0])

        ref = decode_pos(true, K)
        if ref is None:
            continue
        variants = {
            "both predicted": pred,
            "GT knots": np.column_stack([true[:, 0], pred[:, 1:]]),
            "GT control points": np.column_stack([pred[:, 0], true[:, 1:]]),
        }
        for name, p in variants.items():
            got = decode_pos(p, K)
            err[name].append(np.nan if got is None else np.abs(got - ref).max())

    print(f"\ncheckpoint: {args.ckpt}\n{args.n_samples} samples\n")
    print("Position error against the SAME chunk decoded from ground truth:")
    print(f"  {'variant':<20}{'p50 [mm]':>12}{'p90 [mm]':>12}")
    print("  " + "-" * 44)
    for name, v in err.items():
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v)]
        print(f"  {name:<20}{np.percentile(v, 50) * 1000:>12.1f}{np.percentile(v, 90) * 1000:>12.1f}")

    print("\nChunk span [frames]:")
    for name, v in spans.items():
        v = np.asarray(v)
        print(f"  {name:<20}p50 {np.percentile(v, 50):>7.1f}   p90 {np.percentile(v, 90):>7.1f}")

    nz = np.asarray(nz).mean(axis=0)
    names = ["knot", "x", "y", "z", "r0", "r1", "r2", "r3", "r4", "r5", "grip"]
    print("\nMean |error| per channel, in normalised units (what the loss sees):")
    for n, v in zip(names, nz):
        bar = "#" * int(round(v * 40))
        print(f"  {n:<6}{v:>7.3f}  {bar}")

    dp, dt = np.concatenate(knot_diff_pred), np.concatenate(knot_diff_true)
    print("\nAdjacent knot spacing [frames]:")
    print(f"  recorded   mean {dt.mean():>6.2f}   negative: {100 * (dt < 0).mean():.1f}%")
    print(f"  predicted  mean {dp.mean():>6.2f}   negative: {100 * (dp < 0).mean():.1f}%")
    print(f"\n  knot-column std used for normalisation: {std[:, 0].mean():.2f} frames")
    print(f"  typical spacing the decode depends on:  {dt.mean():.2f} frames")
    print(f"  -> the network is normalised over a range {std[:, 0].mean() / max(dt.mean(), 1e-9):.1f}x "
          f"coarser than the quantity that matters")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
