#!/usr/bin/env python3
"""Run a trained checkpoint on recorded observations and dump everything a
report needs -- predictions, decoded waypoints, ground truth, images.

Stage 1 of 2. It runs inside ``lerobot-041`` (which has the policy stack but no
matplotlib); ``build_inference_report.py`` renders the dump in an environment
that does. Nothing here plots.

    conda run -n lerobot-041 python bspline/run_inference_dump.py \
        --ckpt outputs/train/bspline_act_merged_20260528/checkpoints/last/pretrained_model
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from decode_rollout import decode  # noqa: E402
from lerobot_bridge import load_lerobot_actions  # noqa: E402


def save_jpeg(tensor, path: Path, quality: int = 82) -> None:
    """(3, H, W) float in [0, 1] -> jpeg on disk."""
    arr = tensor.detach().cpu().numpy()
    arr = np.transpose(arr, (1, 2, 0))
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path, quality=quality)


def gt_at(ep_raw: np.ndarray, abs_frames: np.ndarray) -> np.ndarray:
    """Ground-truth action interpolated at (possibly fractional) frame indices."""
    frames = np.arange(len(ep_raw))
    clipped = np.clip(abs_frames, 0, len(ep_raw) - 1)
    return np.stack([np.interp(clipped, frames, ep_raw[:, d]) for d in range(7)], axis=1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path,
                    default=Path("outputs/train/bspline_act_merged_20260528/checkpoints/last/pretrained_model"))
    ap.add_argument("--dataset-root", type=Path,
                    default=Path("/home/batur/Coding/data/merged_bspline_20260528"))
    ap.add_argument("--baseline-ckpt", type=Path,
                    default=Path("outputs/train/act_cart7_v2_angleaxis_nogrip_chunk100_ft_20260528/checkpoints/last/pretrained_model"))
    ap.add_argument("--baseline-root", type=Path,
                    default=Path("/home/batur/Coding/data/merged_act_finetune_20260528"))
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--episode", type=int, default=58, help="episode_index to walk through")
    ap.add_argument("--snapshots", type=int, default=6, help="observation/prediction pairs to render")
    ap.add_argument("--replan-frames", type=int, default=10,
                    help="receding-horizon replanning period, in source frames")
    ap.add_argument("--num-actions", type=int, default=16, help="waypoints decoded per chunk")
    ap.add_argument("--score-samples", type=int, default=150,
                    help="random frames across the whole dataset for the error table")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/inference_report"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from policy_io import load_policy, predict_action

    out = args.out_dir
    (out / "frames").mkdir(parents=True, exist_ok=True)

    meta = json.loads((args.dataset_root / "meta" / "bspline.json").read_text())
    fps = float(meta["fps"])
    src = load_lerobot_actions(meta["source_dataset"])
    ep_pos = int(np.where(src.episode_indices == args.episode)[0][0])
    ep_raw = src.episode(ep_pos)          # (T, 7) recorded cartesian action
    ep_start = int(src.episode_starts[ep_pos])
    T = len(ep_raw)
    print(f"episode {args.episode}: {T} frames, global start {ep_start}")

    ds = LeRobotDataset(args.dataset_root.name, root=args.dataset_root, video_backend="pyav")
    probe = ds[ep_start]
    assert int(probe["episode_index"]) == args.episode and int(probe["frame_index"]) == 0, (
        f"index mapping wrong: ds[{ep_start}] is ep {int(probe['episode_index'])} "
        f"frame {int(probe['frame_index'])}"
    )

    policy, pre, post = load_policy(args.ckpt, args.device)
    dump: dict[str, np.ndarray] = {}

    # ---------------------------------------------------------------- snapshots
    span_guess = 45  # frames a chunk typically covers; keep snapshots inside the episode
    last = max(1, T - span_guess)
    snap_frames = np.unique(np.linspace(5, last, args.snapshots).astype(int))
    snaps = []
    for k, f in enumerate(snap_frames):
        sample = ds[ep_start + int(f)]
        pred = predict_action(policy, pre, post, sample, args.device)
        grid = pred.reshape(meta["n_action_steps"], meta["n_action_channels"])
        dec = decode(pred, chunk_size=meta["chunk_size"], degree=meta["degree"],
                     num_actions=args.num_actions, fps=fps,
                     relative_knots=meta["relative_knots"],
                     n_action_channels=meta["n_action_channels"])
        abs_frames = f + dec.times * fps
        truth = gt_at(ep_raw, abs_frames)
        for cam in ("camera", "d405"):
            save_jpeg(sample[f"observation.images.{cam}"], out / "frames" / f"snap{k}_{cam}.jpg")
        snaps.append(dict(
            frame=int(f), t=float(f) / fps,
            state=sample["observation.state"].numpy(),
            params=grid, waypoints=dec.actions, times=dec.times,
            abs_frames=abs_frames, truth=truth,
            span_frames=float(dec.span_frames), padded=bool(dec.padded),
            monotone=bool(np.all(np.diff(grid[:, 0]) >= 0)),
            beyond_end=bool(abs_frames[-1] > T - 1),
        ))
        print(f"  snapshot {k}: frame {f:4d}  span {dec.span_frames:5.1f} fr  "
              f"pos err {1000 * np.abs(dec.actions[:, :3] - truth[:, :3]).max():5.1f} mm")

    for key in ("params", "waypoints", "times", "abs_frames", "truth", "state"):
        dump[f"snap_{key}"] = np.stack([s[key] for s in snaps])
    for key in ("frame", "t", "span_frames", "padded", "monotone", "beyond_end"):
        dump[f"snap_{key}"] = np.asarray([s[key] for s in snaps])

    # ------------------------------------------------------- receding-horizon rollout
    roll_paths, roll_frames, roll_starts, roll_spans = [], [], [], []
    f = 0
    while f < T - 2:
        sample = ds[ep_start + int(f)]
        pred = predict_action(policy, pre, post, sample, args.device)
        dec = decode(pred, chunk_size=meta["chunk_size"], degree=meta["degree"],
                     num_actions=args.num_actions, fps=fps,
                     relative_knots=meta["relative_knots"],
                     n_action_channels=meta["n_action_channels"])
        abs_frames = f + dec.times * fps
        roll_paths.append(dec.actions)
        roll_frames.append(abs_frames)
        roll_starts.append(f)
        roll_spans.append(float(dec.span_frames))
        f += args.replan_frames
    dump["roll_paths"] = np.stack(roll_paths)
    dump["roll_frames"] = np.stack(roll_frames)
    dump["roll_starts"] = np.asarray(roll_starts)
    dump["roll_spans"] = np.asarray(roll_spans)
    print(f"rollout: {len(roll_starts)} replans every {args.replan_frames} frames "
          f"({args.replan_frames / fps:.2f} s), median span {np.median(roll_spans):.1f} frames")

    # ------------------------------------------------------------- scoring sweep
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(ds), size=min(args.score_samples, len(ds)), replace=False)
    pos_err, rot_err, grip_err, spans, nonmono = [], [], [], [], 0
    from scipy.spatial.transform import Rotation
    for i in idx:
        sample = ds[int(i)]
        pred = predict_action(policy, pre, post, sample, args.device)
        grid = pred.reshape(meta["n_action_steps"], meta["n_action_channels"])
        nonmono += int(np.any(np.diff(grid[:, 0]) < 0))
        dec = decode(pred, chunk_size=meta["chunk_size"], degree=meta["degree"],
                     num_actions=args.num_actions, fps=fps,
                     relative_knots=meta["relative_knots"],
                     n_action_channels=meta["n_action_channels"])
        spans.append(dec.span_frames)
        e = int(np.where(src.episode_indices == int(sample["episode_index"]))[0][0])
        raw = src.episode(e)
        af = int(sample["frame_index"]) + dec.times * fps
        if af[-1] >= len(raw) - 1:
            continue
        truth = gt_at(raw, af)
        pos_err.append(np.abs(dec.actions[:, :3] - truth[:, :3]).max())
        rot_err.append((Rotation.from_rotvec(dec.actions[:, 3:6])
                        * Rotation.from_rotvec(truth[:, 3:6]).inv()).magnitude().max())
        grip_err.append(np.abs(dec.actions[:, 6] - truth[:, 6]).max())
    dump["score_pos"] = np.asarray(pos_err)
    dump["score_rot"] = np.asarray(rot_err)
    dump["score_grip"] = np.asarray(grip_err)
    dump["score_spans"] = np.asarray(spans)
    print(f"scored {len(pos_err)}/{len(idx)} random frames, "
          f"p50 pos err {1000 * np.percentile(pos_err, 50):.1f} mm")

    # ------------------------------------------------- baseline ACT (same frames)
    baseline = {}
    if not args.no_baseline:
        del policy, pre, post
        import torch
        torch.cuda.empty_cache()
        bds = LeRobotDataset(args.baseline_root.name, root=args.baseline_root,
                             video_backend="pyav")
        bprobe = bds[ep_start]
        assert int(bprobe["episode_index"]) == args.episode and int(bprobe["frame_index"]) == 0
        bpolicy, bpre, bpost = load_policy(args.baseline_ckpt, args.device)
        n_steps = int(bpolicy.config.n_action_steps)
        print(f"baseline ACT: n_action_steps={n_steps}, chunk_size={bpolicy.config.chunk_size}")

        def chunk_at(frame: int) -> np.ndarray:
            sample = bds[ep_start + int(frame)]
            obs = {k: (v.unsqueeze(0).to(args.device) if torch.is_tensor(v) else v)
                   for k, v in sample.items() if k.startswith("observation") or k == "task"}
            with torch.inference_mode():
                bpolicy.reset()
                obs_p = bpre(obs)
                acts = [bpost(bpolicy.select_action(obs_p))[0].float().cpu().numpy()
                        for _ in range(n_steps)]
            return np.stack(acts)

        b_snaps = [chunk_at(f) for f in snap_frames]
        baseline["base_snap_chunks"] = np.stack(b_snaps)
        b_roll = [chunk_at(f) for f in roll_starts]
        baseline["base_roll_chunks"] = np.stack(b_roll)
        baseline["base_n_steps"] = np.asarray([n_steps])
        print(f"baseline: {len(b_roll)} chunks of {n_steps} actions at the same replan points")

    np.savez_compressed(out / "dump.npz", ep_raw=ep_raw, **dump, **baseline)
    (out / "meta.json").write_text(json.dumps({
        "ckpt": str(args.ckpt),
        "baseline_ckpt": None if args.no_baseline else str(args.baseline_ckpt),
        "dataset_root": str(args.dataset_root),
        "episode": args.episode,
        "episode_frames": T,
        "fps": fps,
        "num_actions": args.num_actions,
        "replan_frames": args.replan_frames,
        "snapshot_frames": snap_frames.tolist(),
        "nonmonotone": int(nonmono),
        "scored": len(pos_err),
        "score_n": int(len(idx)),
        "bspline": meta,
    }, indent=2))
    print(f"\nwrote {out / 'dump.npz'} and {out / 'meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
