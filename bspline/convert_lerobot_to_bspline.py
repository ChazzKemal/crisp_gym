#!/usr/bin/env python3
"""Convert a crisp_gym LeRobot v3.0 dataset into a B-spline-action dataset.

What changes and what does not
------------------------------
Only the ``action`` column is rewritten. Observations, videos and episode
boundaries are untouched -- videos and images are symlinked, so the conversion
costs a few MB rather than a full re-encode.

The new ``action`` at frame *i* is the flattened B-spline parameter matrix
assigned to that frame::

    (chunk_size + 2 * degree, 1 + 10)  ->  flat, row-major

column 0 of each row being the knot (in frames, relative to *i*) and columns
1..10 the control point ``[x, y, z, rot6d(6), gripper]``. A policy trained on
this predicts one parameter matrix per observation; ``decode_bspline_action``
turns it back into as many end-effector waypoints as you want to execute.

``max_first_k`` is fixed at 1 so every source frame keeps a target and the
episode boundaries -- hence the video files -- stay valid.

Normalisation stats are written **per channel**, broadcast across the
``chunk_size + 2 * degree`` rows, mirroring upstream's ``get_normalizer``: the
knot column and each action dimension live on different scales, but rows of the
same column do not.

    conda run -n lerobot-041 python convert_lerobot_to_bspline.py \\
        --src /home/batur/Coding/data/merged_act_finetune_20260528 \\
        --dst /home/batur/Coding/data/merged_bspline_20260528 \\
        --chunk-size 10 --max-error 0.01
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bspline_core.chunk_sampler import BSplineChunkSampler  # noqa: E402
from lerobot_bridge import (  # noqa: E402
    POLICY_ACTION_NAMES,
    load_lerobot_actions,
    to_policy_actions,
)

STAT_KEYS = ("min", "max", "mean", "std")


def bspline_feature_names(n_steps: int) -> list[str]:
    """Row-major names for the flattened parameter matrix."""
    names = []
    for step in range(n_steps):
        names.append(f"knot_{step}")
        for dim in POLICY_ACTION_NAMES:
            names.append(f"c{step}_{dim}")
    return names


def broadcast_channel_stats(sampler: BSplineChunkSampler) -> dict[str, np.ndarray]:
    """Per-channel stats tiled over every row, then flattened to (n_steps*n_ch,)."""
    channel = sampler.get_channel_stats()
    out = {}
    for key in STAT_KEYS:
        tiled = np.broadcast_to(
            channel[key], (1, sampler.n_action_steps, sampler.n_action_channels)
        )
        out[key] = np.asarray(tiled, dtype=np.float64).reshape(-1)
    # A zero std would make mean/std normalisation divide by zero.
    out["std"] = np.maximum(out["std"], 1e-8)
    return out


def _link_or_copy(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src.resolve(), target_is_directory=True)


def convert(
    src: Path,
    dst: Path,
    chunk_size: int,
    degree: int,
    max_error: float,
    stride: int,
    relative_knots: bool,
    overwrite: bool,
) -> dict:
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"{dst} exists (pass --overwrite to replace)")
        shutil.rmtree(dst)

    print(f"Loading {src}")
    data = load_lerobot_actions(src)
    policy_actions = to_policy_actions(data.actions)
    print(f"  {len(data.episode_ends)} episodes, {len(data.actions)} frames @ {data.fps} fps")

    print(f"Fitting B-splines (chunk_size={chunk_size}, degree={degree}, "
          f"max_error={max_error}, stride={stride})")
    sampler = BSplineChunkSampler(
        actions=policy_actions,
        episode_ends=data.episode_ends,
        chunk_size=chunk_size,
        degree=degree,
        max_error=max_error,
        stride=stride,
        relative_knots=relative_knots,
        max_first_k=1,  # keep every frame so episode boundaries stay valid
    )
    n_steps, n_ch = sampler.n_action_steps, sampler.n_action_channels
    flat_dim = n_steps * n_ch

    uncovered = int((sampler.timestep_to_chunk < 0).sum())
    if uncovered:
        raise RuntimeError(f"{uncovered} frames have no chunk; refusing to write a gappy dataset")

    ratios = [s.compression_ratio for s in sampler.fit_stats]
    errors = [s.fit_error for s in sampler.fit_stats]
    n_diverged = sum(1 for s in sampler.fit_stats if not s.converged)
    print(f"  knots/frame: mean {np.mean(ratios):.3f}, max {np.max(ratios):.3f}")
    print(f"  fit error:   max {np.max(errors):.5f}; {n_diverged} episode(s) missed max_error")
    print(f"  action dim:  {n_steps} x {n_ch} = {flat_dim}")

    # ---------------------------------------------------------------- layout
    dst.mkdir(parents=True)
    _link_or_copy(src / "videos", dst / "videos")
    _link_or_copy(src / "images", dst / "images")
    shutil.copytree(src / "meta", dst / "meta")

    # ------------------------------------------------------------ data files
    flat = sampler.all_actions.reshape(len(sampler.all_actions), flat_dim).astype(np.float32)
    per_frame = flat[sampler.timestep_to_chunk]  # (N, flat_dim), frame-ordered

    starts = data.episode_starts
    ep_row = {int(e): (int(a), int(b))
              for e, a, b in zip(data.episode_indices, starts, data.episode_ends)}

    (dst / "data").mkdir()
    n_written = 0
    for parquet in sorted((src / "data").rglob("*.parquet")):
        df = pd.read_parquet(parquet)
        new_actions = np.empty((len(df), flat_dim), dtype=np.float32)
        for ep_id, group in df.groupby("episode_index", sort=True):
            a, b = ep_row[int(ep_id)]
            order = group.sort_values("frame_index").index
            new_actions[df.index.get_indexer(order)] = per_frame[a:b]
        df["action"] = list(new_actions)
        out = dst / "data" / parquet.relative_to(src / "data")
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        n_written += len(df)
    print(f"  wrote {n_written} frames across {len(list((dst / 'data').rglob('*.parquet')))} files")

    # ------------------------------------------------------------ meta/stats
    stats = broadcast_channel_stats(sampler)
    info = json.loads((dst / "meta" / "info.json").read_text())
    info["features"]["action"] = {
        "dtype": "float32",
        "shape": [flat_dim],
        "names": bspline_feature_names(n_steps),
    }
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=4))

    stats_path = dst / "meta" / "stats.json"
    if stats_path.exists():
        all_stats = json.loads(stats_path.read_text())
        all_stats["action"] = {
            "min": stats["min"].tolist(),
            "max": stats["max"].tolist(),
            "mean": stats["mean"].tolist(),
            "std": stats["std"].tolist(),
            "count": [int(len(per_frame))],
        }
        stats_path.write_text(json.dumps(all_stats))

    # Per-episode stats blocks: recompute over that episode's own chunks, but
    # keep the shared per-channel scale so the normaliser stays consistent.
    for parquet in sorted((dst / "meta" / "episodes").rglob("*.parquet")):
        em = pd.read_parquet(parquet)
        for key in STAT_KEYS:
            col = f"stats/action/{key}"
            if col in em.columns:
                em[col] = [stats[key].copy() for _ in range(len(em))]
        for col, val in (("stats/action/count", None),):
            if col in em.columns:
                em[col] = [np.array([int(em["length"].iloc[i])]) for i in range(len(em))]
        for q in ("q01", "q10", "q50", "q90", "q99"):
            col = f"stats/action/{q}"
            if col in em.columns:
                lo, hi = stats["min"], stats["max"]
                frac = {"q01": 0.01, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q99": 0.99}[q]
                em[col] = [lo + frac * (hi - lo) for _ in range(len(em))]
        em.to_parquet(parquet, index=False)

    # ------------------------------------------------------------- sidecar
    meta = {
        "source_dataset": str(src.resolve()),
        "representation": "bspline_policy_action_chunk",
        "upstream": "https://github.com/B-spline-policy/bspline-policy",
        "chunk_size": chunk_size,
        "degree": degree,
        "max_error": max_error,
        "stride": stride,
        "relative_knots": relative_knots,
        "n_action_steps": n_steps,
        "n_action_channels": n_ch,
        "flat_action_dim": flat_dim,
        "control_point_names": POLICY_ACTION_NAMES,
        "raw_action_layout": "xyz(3) + axis_angle(3) + gripper(1)",
        "policy_action_layout": "xyz(3) + rot6d(6) + gripper(1)",
        "fps": data.fps,
        "knot_units": "source frames, relative to the current frame",
        "fit": {
            "knots_per_frame_mean": float(np.mean(ratios)),
            "knots_per_frame_max": float(np.max(ratios)),
            "fit_error_max": float(np.max(errors)),
            "episodes_missing_max_error": int(n_diverged),
        },
    }
    (dst / "meta" / "bspline.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {dst}")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path,
                    default=Path("/home/batur/Coding/data/merged_act_finetune_20260528"))
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--chunk-size", type=int, default=10)
    ap.add_argument("--degree", type=int, default=3)
    ap.add_argument("--max-error", type=float, default=0.01)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--relative-knots", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    convert(args.src, args.dst, args.chunk_size, args.degree, args.max_error,
            args.stride, args.relative_knots, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
