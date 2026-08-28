#!/usr/bin/env python3
"""Step 1 of DemoSpeedup: train the proxy policy on the *original* demos.

The proxy is an ordinary ACT policy -- upstream trains exactly the same model
it later benchmarks, and only uses it as an entropy oracle
(``imitate_episodes.py --label`` reloads this checkpoint). Nothing about the
training run is DemoSpeedup-specific, so this is deliberately a thin wrapper
over ``lerobot-train`` with the same flags and photometric augmentation as the
ACT baselines in ``examples/28`` and ``examples/30``.

**You probably do not need to run this.** Any ACT checkpoint already trained on
the dataset you want to accelerate works as the proxy -- e.g.
``outputs/train/act_cart7_v2_angleaxis_nogrip_chunk100``. Point
``label_entropy.py --policy-path`` at it and skip to step 2. Train a fresh one
only when no checkpoint matches the dataset's feature layout.

    conda run -n lerobot-041 python train_proxy_act.py --wandb

Anything after ``--`` is forwarded verbatim to ``lerobot-train``.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

DEFAULT_ROOT = Path("/home/batur/Coding/data/merged_act_finetune_20260528")
DEFAULT_REPO_ID = "merged_act_finetune_20260528"
DEFAULT_OUTPUT_DIR = Path("outputs/train/demospeedup_proxy_act")
DEFAULT_CHUNK_SIZE = 100
DEFAULT_STEPS = 30_000
DEFAULT_BATCH_SIZE = 32
DEFAULT_WANDB_PROJECT = "demospeedup_proxy_act"

# Identical to the ACT baselines in examples/29, examples/30 and bspline/.
PHOTOMETRIC_TFS = (
    "{brightness: {weight: 1.0, type: ColorJitter, kwargs: {brightness: [0.8, 1.2]}}, "
    "contrast: {weight: 1.0, type: ColorJitter, kwargs: {contrast: [0.8, 1.2]}}, "
    "saturation: {weight: 1.0, type: ColorJitter, kwargs: {saturation: [0.5, 1.5]}}, "
    "hue: {weight: 1.0, type: ColorJitter, kwargs: {hue: [-0.05, 0.05]}}, "
    "sharpness: {weight: 1.0, type: SharpnessJitter, kwargs: {sharpness: [0.5, 1.5]}}}"
)


def build_command(args: argparse.Namespace, extra: list[str]) -> list[str]:
    info_path = args.dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Dataset meta/info.json not found at {info_path}")

    cmd = [
        "lerobot-train",
        "--policy.type=act",
        "--policy.push_to_hub=false",
        # use_vae stays on: the CVAE prior is what label_entropy.py samples.
        "--policy.use_vae=true",
        f"--policy.chunk_size={args.chunk_size}",
        f"--policy.n_action_steps={args.n_action_steps or args.chunk_size}",
        f"--dataset.repo_id={args.repo_id}",
        f"--dataset.root={args.dataset_root}",
        f"--dataset.video_backend={args.video_backend}",
        f"--output_dir={args.output_dir}",
        f"--steps={args.steps}",
        f"--batch_size={args.batch_size}",
        f"--log_freq={args.log_freq}",
        f"--save_freq={args.save_freq}",
        f"--num_workers={args.num_workers}",
        f"--seed={args.seed}",
    ]
    if args.pretrained_path is not None:
        cmd.append(f"--policy.pretrained_path={args.pretrained_path}")
    if not args.no_augment:
        cmd += [
            "--dataset.image_transforms.enable=true",
            "--dataset.image_transforms.max_num_transforms=3",
            f"--dataset.image_transforms.tfs={PHOTOMETRIC_TFS}",
        ]
    if args.no_wandb:
        cmd.append("--wandb.enable=false")
    elif args.wandb:
        cmd += ["--wandb.enable=true", f"--wandb.project={args.wandb_project}"]
    cmd.extend(extra)
    return cmd


def add_common_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    ap.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    ap.add_argument("--n-action-steps", type=int, default=None,
                    help="defaults to --chunk-size")
    ap.add_argument("--pretrained-path", type=Path, default=None,
                    help="warm-start from an existing ACT checkpoint")
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--log-freq", type=int, default=100)
    ap.add_argument("--save-freq", type=int, default=5_000)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--video-backend", default="pyav", choices=["torchcodec", "pyav"])
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--print-only", action="store_true")


def run(build, ap: argparse.ArgumentParser) -> int:
    argv = sys.argv[1:]
    extra: list[str] = []
    if "--" in argv:
        sep = argv.index("--")
        argv, extra = argv[:sep], argv[sep + 1 :]
    args = ap.parse_args(argv)
    cmd = build(args, extra)
    print("\n+ " + " ".join(shlex.quote(c) for c in cmd) + "\n", flush=True)
    if args.print_only:
        return 0
    return subprocess.call(cmd, env=os.environ.copy())


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_arguments(ap)
    return run(build_command, ap)


if __name__ == "__main__":
    raise SystemExit(main())
