#!/usr/bin/env python3
"""Inspect a labelling run before paying for the conversion.

Reads the sidecar ``label_entropy.py`` wrote and reports, per episode and in
aggregate: how much of the demo was called non-precision, how many frames the
retiming keeps, the resulting speedup, and -- the part that matters for the
real UR10e -- how large the commanded end-effector steps become once the
in-between frames are gone. A 4x stride turns a 20 Hz, 5 mm/frame motion into
20 mm per control period; if that number climbs past what the CRISP cartesian
controller tracks comfortably, lower ``--high-v`` rather than discovering it
on the robot.

    conda run -n lerobot-041 python analyze_labels.py \\
        --dataset-root /home/batur/Coding/data/merged_act_finetune_20260528

    # what would stride 2/3 keep instead?
    conda run -n lerobot-041 python analyze_labels.py --dataset-root ... --high-v 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from demospeedup_core.retiming import HIGH_V, LOW_V, retiming_stats, select_keep_indices  # noqa: E402
from demospeedup_core.segmentation import segment_entropy  # noqa: E402
from lerobot_bridge import episode_actions, episode_ranges, load_info, read_labels  # noqa: E402


def _step_norms(actions: np.ndarray, keep: np.ndarray) -> tuple[float, float]:
    """Mean and max ``||xyz(k+1) - xyz(k)||`` over the kept frames."""
    if len(keep) < 2:
        return 0.0, 0.0
    steps = np.linalg.norm(np.diff(actions[keep, :3], axis=0), axis=-1)
    return float(steps.mean()), float(steps.max())


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--low-v", type=int, default=None, help="override the labelled stride")
    ap.add_argument("--high-v", type=int, default=None, help="override the labelled stride")
    ap.add_argument("--episodes", type=int, default=None, help="only the first N episodes")
    ap.add_argument("--resegment", choices=["threshold", "hdbscan"], default=None,
                    help="re-run segmentation on the stored entropy with this backend "
                         "(compare backends without re-running the policy)")
    ap.add_argument("--min-cluster-size", type=int, default=5)
    ap.add_argument("--fast-prefix", type=int, default=0)
    ap.add_argument("--per-episode", action="store_true", help="print every episode")
    args = ap.parse_args()

    root = args.dataset_root.resolve()
    info = load_info(root)
    frame_labels, config = read_labels(root)
    low_v = args.low_v if args.low_v is not None else int(config.get("low_v", LOW_V))
    high_v = args.high_v if args.high_v is not None else int(config.get("high_v", HIGH_V))

    print(f"dataset  : {root}")
    print(f"labelled : {config.get('policy_path', '?')}")
    print(f"           segmenter={config.get('segmenter', '?')} "
          f"num_samples={config.get('num_samples', '?')} "
          f"bandwidth={config.get('bandwidth', '?')}")
    print(f"stride   : {low_v}x precision / {high_v}x non-precision"
          + (f"   [resegmented with {args.resegment}]" if args.resegment else ""))
    print()

    grouped = {int(e): g.sort_values("frame_index") for e, g in frame_labels.groupby("episode_index")}
    totals = {"frames": 0, "kept": 0, "fast": 0}
    flips = []
    mean_steps, max_steps, speedups = [], [], []
    worst_gap = 0

    for ep in episode_ranges(root, args.episodes):
        group = grouped.get(ep.episode_index)
        if group is None:
            continue
        entropy = group["entropy"].to_numpy()
        labels = group["label"].to_numpy().astype(np.int64)
        if args.resegment:
            reseg = segment_entropy(
                entropy, backend=args.resegment,
                min_cluster_size=args.min_cluster_size, fast_prefix=args.fast_prefix,
            )
            flips.append(float(np.mean(reseg.labels != labels)))
            labels = reseg.labels

        keep = select_keep_indices(labels, low_v, high_v)
        stats = retiming_stats(labels, keep)
        actions = episode_actions(root, ep, info)
        mean_step, max_step = _step_norms(actions, keep)

        totals["frames"] += stats.n_source
        totals["kept"] += stats.n_kept
        totals["fast"] += int(labels.sum())
        speedups.append(stats.speedup)
        mean_steps.append(mean_step)
        max_steps.append(max_step)
        worst_gap = max(worst_gap, stats.max_gap)

        if args.per_episode:
            print(f"ep {ep.episode_index:3d}: {stats.n_source:5d} -> {stats.n_kept:5d} "
                  f"({stats.speedup:.2f}x)  fast {stats.fast_fraction * 100:5.1f}%  "
                  f"step mean {mean_step * 1000:6.1f} mm  max {max_step * 1000:6.1f} mm")

    if not speedups:
        raise SystemExit("no labelled episodes found")

    print(f"{'':11s}{'mean':>10s}{'min':>10s}{'max':>10s}")
    print(f"speedup    {np.mean(speedups):10.2f}{np.min(speedups):10.2f}{np.max(speedups):10.2f}")
    print(f"step (mm)  {np.mean(mean_steps) * 1000:10.1f}{np.min(mean_steps) * 1000:10.1f}"
          f"{np.max(max_steps) * 1000:10.1f}   <- last column is the worst single step")
    print()
    print(f"frames        : {totals['frames']} -> {totals['kept']} "
          f"({totals['frames'] / max(totals['kept'], 1):.2f}x overall)")
    print(f"non-precision : {100 * totals['fast'] / max(totals['frames'], 1):.1f}% of frames")
    print(f"largest gap   : {worst_gap} source frames "
          f"({worst_gap / float(info['fps']) * 1000:.0f} ms of original motion per control step)")
    if flips:
        print(f"resegmentation: {100 * float(np.mean(flips)):.1f}% of frames changed label")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
