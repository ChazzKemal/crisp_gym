#!/usr/bin/env python3
"""Step 2 of DemoSpeedup: label every demonstration frame with its action entropy.

What upstream does (``imitate_episodes.py --label``)
----------------------------------------------------
Replay each *recorded* demonstration through the trained proxy policy -- the
observations come from the demo, not from a rollout -- and at every frame draw
``num_samples = 10`` action chunks from the CVAE prior. With temporal
aggregation on (``--temporal_agg``, query frequency 1) the samples of *every*
still-valid chunk that covers the current frame are pooled, and the KDE
differential entropy of that pooled cloud is the frame's entropy. The trace is
z-normalised per episode, clustered in ``(time, entropy)``, and collapsed to a
binary precision / non-precision label.

What this script does
---------------------
The same, against a LeRobot v3.0 dataset and a LeRobot ``ACTPolicy``
checkpoint. The samples are read in normalised action units (before the
post-processor) exactly as upstream reads them before ``post_process``.
Results go to the sidecar described in :mod:`lerobot_bridge` -- the dataset
itself is never modified.

    conda run -n lerobot-041 python label_entropy.py \\
        --dataset-root /home/batur/Coding/data/merged_act_finetune_20260528 \\
        --policy-path outputs/train/act_cart7_v2_angleaxis_nogrip_chunk100/checkpoints/last/pretrained_model

Cost: one forward pass per frame with the batch tiled ``num_samples`` times.
For 21.5k frames x 10 samples x 2 cameras that is a few GPU-minutes; use
``--episodes N`` for a smoke test first.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from demospeedup_core.entropy import kde_entropy  # noqa: E402
from demospeedup_core.sampling import ACTChunkSampler, TemporalSampleBuffer  # noqa: E402
from demospeedup_core.segmentation import segment_entropy  # noqa: E402
from demospeedup_core.retiming import (  # noqa: E402
    HIGH_V,
    LOW_V,
    retiming_stats,
    select_keep_indices,
)
from lerobot_bridge import episode_ranges, load_info, write_labels  # noqa: E402


def _load_policy(policy_path: Path, device: str):
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    policy = ACTPolicy.from_pretrained(policy_path)
    policy.config.device = device
    policy.to(device)
    policy.eval()
    preprocessor, _ = make_pre_post_processors(
        policy.config, pretrained_path=str(policy_path)
    )
    return policy, preprocessor


def _episode_entropy(
    dataset,
    sampler: ACTChunkSampler,
    start: int,
    stop: int,
    frames_per_forward: int,
    num_workers: int,
    bandwidth: float,
) -> np.ndarray:
    """KDE entropy of the pooled action samples at every frame of one episode."""
    from torch.utils.data import DataLoader, Subset

    loader = DataLoader(
        Subset(dataset, range(start, stop)),
        batch_size=frames_per_forward,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    buffer = TemporalSampleBuffer(sampler.chunk_size)
    entropies: list[float] = []
    for batch in loader:
        samples = sampler.sample(batch)  # (F, S, chunk, dim)
        for frame in range(samples.shape[0]):
            buffer.add(samples[frame])
            pooled = buffer.current().unsqueeze(0)  # (1, n*S, dim)
            entropies.append(float(kde_entropy(pooled, bandwidth).squeeze()))
    return np.asarray(entropies, dtype=np.float64)


def _maybe_plot(out_dir: Path, episode_index: int, entropy: np.ndarray, labels: np.ndarray) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return  # matplotlib is not in lerobot-041; plots are a nicety
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(entropy, marker=".", linewidth=1, label="entropy")
    ax.axhline(entropy.mean(), color="grey", linestyle="--", linewidth=1, label="mean")
    ax.fill_between(
        np.arange(len(labels)), entropy.min(), entropy.max(),
        where=labels == 1, alpha=0.2, color="tab:red", label="non-precision (4x)",
    )
    ax.set_xlabel("frame")
    ax.set_ylabel("KDE entropy")
    ax.set_title(f"episode {episode_index}")
    ax.legend(loc="upper right", fontsize="small")
    fig.tight_layout()
    fig.savefig(out_dir / f"episode_{episode_index:04d}_entropy.png", dpi=110)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--repo-id", default=None, help="defaults to the dataset dir name")
    ap.add_argument("--policy-path", type=Path, required=True,
                    help="proxy ACT checkpoint (.../checkpoints/last/pretrained_model)")
    ap.add_argument("--num-samples", type=int, default=10,
                    help="prior draws per observation (upstream: 10)")
    ap.add_argument("--bandwidth", type=float, default=1.0,
                    help="KDE bandwidth in normalised action units (upstream: 1.0)")
    ap.add_argument("--frames-per-forward", type=int, default=4,
                    help="observations per forward pass; VRAM scales with this x --num-samples")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=None, help="only the first N episodes")
    ap.add_argument("--segmenter", choices=["threshold", "hdbscan"], default="threshold")
    ap.add_argument("--min-cluster-size", type=int, default=5)
    ap.add_argument("--fast-prefix", type=int, default=0,
                    help="force the first N frames to non-precision (upstream: 50)")
    ap.add_argument("--low-v", type=int, default=LOW_V, help="stride in precision phases")
    ap.add_argument("--high-v", type=int, default=HIGH_V, help="stride in non-precision phases")
    ap.add_argument("--video-backend", default="pyav", choices=["pyav", "torchcodec"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=2, help="upstream uses set_seed(2)")
    ap.add_argument("--plots", action="store_true", help="per-episode entropy plots (needs matplotlib)")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    root = args.dataset_root.resolve()
    info = load_info(root)
    repo_id = args.repo_id or root.name
    dataset = LeRobotDataset(repo_id, root=root, video_backend=args.video_backend)
    policy, preprocessor = _load_policy(args.policy_path.resolve(), args.device)
    sampler = ACTChunkSampler(policy, preprocessor, args.num_samples, seed=args.seed)

    print(f"dataset : {root}  ({info['total_episodes']} episodes, "
          f"{info['total_frames']} frames @ {info['fps']} fps)")
    print(f"policy  : {args.policy_path}  (chunk_size={sampler.chunk_size}, "
          f"latent_dim={policy.config.latent_dim})")
    print(f"sampling: {args.num_samples} prior draws/frame, KDE bandwidth {args.bandwidth}")

    episodes = episode_ranges(root, args.episodes)
    rows: list[pd.DataFrame] = []
    per_episode: list[dict] = []
    t0 = time.time()
    for ep in episodes:
        started = time.time()
        entropy = _episode_entropy(
            dataset, sampler, ep.dataset_from_index, ep.dataset_to_index,
            args.frames_per_forward, args.num_workers, args.bandwidth,
        )
        if len(entropy) != ep.length:
            raise RuntimeError(
                f"episode {ep.episode_index}: {len(entropy)} entropies for {ep.length} frames"
            )
        seg = segment_entropy(
            entropy,
            backend=args.segmenter,
            min_cluster_size=args.min_cluster_size,
            fast_prefix=args.fast_prefix,
        )
        keep = select_keep_indices(seg.labels, args.low_v, args.high_v)
        stats = retiming_stats(seg.labels, keep)
        rows.append(pd.DataFrame({
            "episode_index": np.full(ep.length, ep.episode_index, dtype=np.int64),
            "frame_index": np.arange(ep.length, dtype=np.int64),
            "entropy": entropy,
            "entropy_z": seg.entropy_z,
            "label": seg.labels,
        }))
        per_episode.append({
            "episode_index": ep.episode_index,
            "length": ep.length,
            "fast_fraction": stats.fast_fraction,
            "n_kept": stats.n_kept,
            "speedup": stats.speedup,
            "max_gap": stats.max_gap,
        })
        if args.plots:
            _maybe_plot(root / "meta" / "demospeedup" / "plots", ep.episode_index,
                        entropy, seg.labels)
        print(f"ep {ep.episode_index:3d}: {ep.length:4d} frames  "
              f"fast {stats.fast_fraction * 100:5.1f}%  -> {stats.n_kept:4d} kept "
              f"({stats.speedup:.2f}x)  [{time.time() - started:.1f}s]", flush=True)

    frame_labels = pd.concat(rows, ignore_index=True)
    total_kept = sum(e["n_kept"] for e in per_episode)
    total_frames = sum(e["length"] for e in per_episode)
    config = {
        "upstream": "https://github.com/lingxiao-guo/DemoSpeedup",
        "policy_path": str(args.policy_path.resolve()),
        "chunk_size": int(sampler.chunk_size),
        "num_samples": args.num_samples,
        "bandwidth": args.bandwidth,
        "segmenter": args.segmenter,
        "min_cluster_size": args.min_cluster_size,
        "fast_prefix": args.fast_prefix,
        "low_v": args.low_v,
        "high_v": args.high_v,
        "seed": args.seed,
        "fps": info["fps"],
        "n_episodes_labelled": len(episodes),
        "total_frames": total_frames,
        "projected_frames": total_kept,
        "projected_speedup": total_frames / max(total_kept, 1),
        "labelled_at_s": time.time(),
        "per_episode": per_episode,
    }
    labels_path, config_path = write_labels(root, frame_labels, config)
    print(f"\n{len(episodes)} episodes in {time.time() - t0:.0f}s")
    print(f"projected: {total_frames} -> {total_kept} frames "
          f"({config['projected_speedup']:.2f}x average speedup)")
    print(f"labels : {labels_path}")
    print(f"config : {config_path}")
    print(json.dumps({k: v for k, v in config.items() if k != "per_episode"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
