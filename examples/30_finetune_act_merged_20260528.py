#!/usr/bin/env python3
"""Fine-tune ACT on the 3-source merged dataset, starting from the 10k
checkpoint of the previous fine-tune (``..._ft_20260525``).

Input/output:
    input  : observation.images.{camera, d405}  +  observation.state (6,)
    output : action (7,)  = [x, y, z, rx, ry, rz, gripper]

Same schema as the parent model (axis-angle rotation, gripper output-only).
Combined dataset is ``data/merged_act_finetune_20260528`` produced by
``merge_datasets_cart6.py`` from:

    - speedup_20260527_234316          (45 ep,  7686 fr)
    - pickplace_cart7_v3               (12 ep,  3017 fr)
    - speedup_20260525_233612_trimmed_nostall  (13 ep, 10857 fr)

Total: 70 ep / 21560 frames (~2x the previous fine-tune's 10857 frames).

Usage (inside the ``lerobot-041`` conda env):
    cd ur10_clearpath/Yunfei/crisp_gym
    conda run -n lerobot-041 python examples/30_finetune_act_merged_20260528.py --wandb

Anything after ``--`` is forwarded verbatim to ``lerobot-train``.
"""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_PRETRAINED_PATH = Path(
    "/home/batur/Coding/ur10_clearpath/Yunfei/crisp_gym/outputs/train/"
    "act_cart7_v2_angleaxis_nogrip_chunk100_ft_20260525/checkpoints/010000/pretrained_model"
)
DEFAULT_REPO_ID = "merged_act_finetune_20260528"
DEFAULT_ROOT = Path("/home/batur/Coding/data/merged_act_finetune_20260528")

DEFAULT_OUTPUT_DIR = Path("outputs/train/act_cart7_v2_angleaxis_nogrip_chunk100_ft_20260528")
DEFAULT_STEPS = 30_000
DEFAULT_BATCH_SIZE = 32
DEFAULT_LOG_FREQ = 100
DEFAULT_SAVE_FREQ = 5_000
DEFAULT_NUM_WORKERS = 4
DEFAULT_VIDEO_BACKEND = "pyav"
DEFAULT_WANDB_PROJECT = "act_cart7_v2_angleaxis_nogrip_chunk100_ft_20260528"

PHOTOMETRIC_TFS = (
    "{brightness: {weight: 1.0, type: ColorJitter, kwargs: {brightness: [0.8, 1.2]}}, "
    "contrast: {weight: 1.0, type: ColorJitter, kwargs: {contrast: [0.8, 1.2]}}, "
    "saturation: {weight: 1.0, type: ColorJitter, kwargs: {saturation: [0.5, 1.5]}}, "
    "hue: {weight: 1.0, type: ColorJitter, kwargs: {hue: [-0.05, 0.05]}}, "
    "sharpness: {weight: 1.0, type: SharpnessJitter, kwargs: {sharpness: [0.5, 1.5]}}}"
)


def _build_command(args: argparse.Namespace, extra: list[str]) -> list[str]:
    if not args.pretrained_path.exists():
        raise FileNotFoundError(f"pretrained_path does not exist: {args.pretrained_path}")
    info_path = args.dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Dataset meta/info.json not found at {info_path}")

    cmd = [
        "lerobot-train",
        "--policy.type=act",
        f"--policy.pretrained_path={args.pretrained_path}",
        "--policy.push_to_hub=false",
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
    if not args.no_augment:
        cmd += [
            "--dataset.image_transforms.enable=true",
            "--dataset.image_transforms.max_num_transforms=3",
            f"--dataset.image_transforms.tfs={PHOTOMETRIC_TFS}",
        ]
    if args.no_wandb:
        cmd.append("--wandb.enable=false")
    elif args.wandb:
        cmd.append("--wandb.enable=true")
        cmd.append(f"--wandb.project={args.wandb_project}")
    cmd.extend(extra)
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pretrained-path", type=Path, default=DEFAULT_PRETRAINED_PATH)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--log-freq", type=int, default=DEFAULT_LOG_FREQ)
    parser.add_argument("--save-freq", type=int, default=DEFAULT_SAVE_FREQ)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--video-backend", default=DEFAULT_VIDEO_BACKEND, choices=["torchcodec", "pyav"])
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--print-only", action="store_true")

    argv = sys.argv[1:]
    extra: list[str] = []
    if "--" in argv:
        sep = argv.index("--")
        argv, extra = argv[:sep], argv[sep + 1 :]
    args = parser.parse_args(argv)
    cmd = _build_command(args, extra)
    print("\n+ " + " ".join(shlex.quote(c) for c in cmd) + "\n", flush=True)
    if args.print_only:
        return 0
    return subprocess.call(cmd, env=os.environ.copy())


if __name__ == "__main__":
    sys.exit(main())
