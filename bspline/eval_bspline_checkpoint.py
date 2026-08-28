#!/usr/bin/env python3
"""Decode a trained checkpoint's predictions and score them against the data.

Training loss on B-spline parameters is not interpretable -- an L1 error on a
knot value means nothing physical. This script closes the loop: run the policy
on real observations, decode the predicted parameter matrix into end-effector
waypoints, and measure the error in millimetres and degrees against the
recorded trajectory.

It also reports how often the predicted knot column comes out non-monotone,
which is the failure mode the representation is most exposed to.

    conda run -n lerobot-041 python eval_bspline_checkpoint.py \\
        --ckpt outputs/train/bspline_act_merged_20260528/checkpoints/last/pretrained_model
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))

from decode_rollout import decode  # noqa: E402
from lerobot_bridge import load_lerobot_actions  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--dataset-root", type=Path,
                    default=Path("/home/batur/Coding/data/merged_bspline_20260528"))
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--num-actions", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from policy_io import load_policy, predict_action

    meta = json.loads((args.dataset_root / "meta" / "bspline.json").read_text())
    fps = meta["fps"]
    source = load_lerobot_actions(meta["source_dataset"])
    starts, ends = source.episode_starts, source.episode_ends

    ds = LeRobotDataset(args.dataset_root.name, root=args.dataset_root, video_backend="pyav")
    policy, pre, post = load_policy(args.ckpt, args.device)

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(ds), size=min(args.n_samples, len(ds)), replace=False)

    pos_err, rot_err, grip_err, spans = [], [], [], []
    n_nonmonotone = 0
    n_scored = 0

    for i in idx:
        sample = ds[int(i)]
        pred = predict_action(policy, pre, post, sample, args.device)

        grid = pred.reshape(meta["n_action_steps"], meta["n_action_channels"])
        if np.any(np.diff(grid[:, 0]) < 0):
            n_nonmonotone += 1

        out = decode(pred, chunk_size=meta["chunk_size"], degree=meta["degree"],
                     num_actions=args.num_actions, fps=fps,
                     relative_knots=meta["relative_knots"],
                     n_action_channels=meta["n_action_channels"])
        spans.append(out.span_frames)

        ep = int(sample["episode_index"])
        a, b = int(starts[ep]), int(ends[ep])
        local = int(sample["frame_index"])
        ep_raw = source.actions[a:b]
        frames = np.arange(len(ep_raw))
        abs_frames = local + out.times * fps
        if abs_frames[-1] >= len(ep_raw) - 1:
            continue
        truth = np.stack([np.interp(abs_frames, frames, ep_raw[:, d]) for d in range(7)], axis=1)
        pos_err.append(np.abs(out.actions[:, :3] - truth[:, :3]).max())
        rot_err.append(
            (Rotation.from_rotvec(out.actions[:, 3:6])
             * Rotation.from_rotvec(truth[:, 3:6]).inv()).magnitude().max()
        )
        grip_err.append(np.abs(out.actions[:, 6] - truth[:, 6]).max())
        n_scored += 1

    if not n_scored:
        print("no scorable samples (all chunks ran past their episode end)")
        return 1

    pos, rot, grip, spans = map(np.asarray, (pos_err, rot_err, grip_err, spans))
    print(f"\ncheckpoint: {args.ckpt}")
    print(f"scored {n_scored}/{len(idx)} samples, {args.num_actions} waypoints each\n")
    print(f"{'metric':<22}{'p50':>10}{'p90':>10}{'max':>10}")
    print("-" * 52)
    print(f"{'position error [mm]':<22}"
          + "".join(f"{v * 1000:>10.1f}" for v in np.percentile(pos, [50, 90, 100])))
    print(f"{'rotation error [deg]':<22}"
          + "".join(f"{np.degrees(v):>10.2f}" for v in np.percentile(rot, [50, 90, 100])))
    print(f"{'gripper error':<22}"
          + "".join(f"{v:>10.3f}" for v in np.percentile(grip, [50, 90, 100])))
    print(f"{'chunk span [frames]':<22}"
          + "".join(f"{v:>10.1f}" for v in np.percentile(spans, [50, 90, 100])))
    print(f"\nchunk span p50 = {np.percentile(spans, 50) / fps:.2f} s of trajectory "
          f"from {meta['flat_action_dim']} predicted numbers")
    print(f"non-monotone knot predictions: {n_nonmonotone}/{len(idx)} "
          f"({100 * n_nonmonotone / len(idx):.1f}%) -- repaired by safer_knots()")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
