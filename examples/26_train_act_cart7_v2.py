#!/usr/bin/env python3
"""Warm-start ACT from `act_infinite_cart7` on the merged `pickplace_cart7_v2` set.

Why this script and not `24_train_act.py`
-----------------------------------------
``24_train_act.py`` trains ACT from scratch on the pickplace dataset. This one
points lerobot-train at the most recent `act_infinite_cart7` checkpoint
(`--policy.path=...`), which loads the trained weights and preserves the
normalizer pipeline, so training continues from a known-good initialization on
a slightly different dataset (pickplace_cart7_v2 = infinite_task_trimmed_nostall_cart7
plus two new speedup recordings, all in 7-D cart-pose-plus-gripper form).

Photometric augmentation is half the strength used by `24_train_act.py`: that
script jitters brightness/contrast/saturation by ±20% and hue by ±0.05. Halving
those ranges is the standard "small" colour jitter you find in most BC papers
— enough to harden the warm-started policy against lighting drift, small enough
not to drag it off the prior.

Usage (inside the ``lerobot-041`` conda env):
    cd ur10_clearpath/Yunfei/crisp_gym
    conda run -n lerobot-041 python examples/26_train_act_cart7_v2.py
    conda run -n lerobot-041 python examples/26_train_act_cart7_v2.py --steps 50000

Anything after ``--`` is forwarded verbatim to ``lerobot-train``.
"""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_REPO_ID = "pickplace_cart7_v2"
DEFAULT_ROOT = Path("/home/batur/Coding/data/pickplace_cart7_v2")

# Warm-start checkpoint: the converged act_infinite_cart7 run.
DEFAULT_INIT_CKPT = Path(
    "outputs/train/act_infinite_cart7/checkpoints/last/pretrained_model"
)

DEFAULT_OUTPUT_DIR = Path("outputs/train/act_cart7_v2")
DEFAULT_STEPS = 30_000          # warm-start, ~1/3 of the original 100k run
DEFAULT_BATCH_SIZE = 32
DEFAULT_LOG_FREQ = 100
DEFAULT_SAVE_FREQ = 5_000
DEFAULT_NUM_WORKERS = 4
DEFAULT_VIDEO_BACKEND = "pyav"
DEFAULT_WANDB_PROJECT = "act_cart7_v2"

DEFAULT_CHUNK_SIZE = 32
DEFAULT_N_ACTION_STEPS = 16

# Half-strength photometric jitter (a standard "small" augmentation in BC).
# Brightness / contrast / saturation: ±10%, hue: ±0.025, sharpness: 0.75-1.25.
PHOTOMETRIC_TFS_SMALL = (
    "{brightness: {weight: 1.0, type: ColorJitter, kwargs: {brightness: [0.9, 1.1]}}, "
    "contrast: {weight: 1.0, type: ColorJitter, kwargs: {contrast: [0.9, 1.1]}}, "
    "saturation: {weight: 1.0, type: ColorJitter, kwargs: {saturation: [0.75, 1.25]}}, "
    "hue: {weight: 1.0, type: ColorJitter, kwargs: {hue: [-0.025, 0.025]}}, "
    "sharpness: {weight: 1.0, type: SharpnessJitter, kwargs: {sharpness: [0.75, 1.25]}}}"
)


def _build_command(args: argparse.Namespace, extra: list[str]) -> list[str]:
    dataset_root = args.dataset_root
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(
            f"Dataset meta/info.json not found at {info_path}. "
            f"--dataset-root must point at a directory that contains meta/info.json."
        )
    init_ckpt = args.init_checkpoint
    if not (init_ckpt / "config.json").exists():
        raise FileNotFoundError(
            f"--init-checkpoint must contain config.json (LeRobot pretrained_model layout); "
            f"got {init_ckpt}"
        )

    cmd = [
        "lerobot-train",
        # Warm-start: load weights + processors from the cart7 checkpoint.
        f"--policy.path={init_ckpt}",
        f"--policy.chunk_size={args.chunk_size}",
        f"--policy.n_action_steps={args.n_action_steps}",
        "--policy.push_to_hub=false",
        f"--dataset.repo_id={args.repo_id}",
        f"--dataset.root={dataset_root}",
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
            f"--dataset.image_transforms.tfs={PHOTOMETRIC_TFS_SMALL}",
        ]
    if args.resume:
        cmd.append("--resume=true")
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
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--init-checkpoint", type=Path, default=DEFAULT_INIT_CKPT,
                        help="Pretrained model dir to warm-start from (passed as --policy.path).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--log-freq", type=int, default=DEFAULT_LOG_FREQ)
    parser.add_argument("--save-freq", type=int, default=DEFAULT_SAVE_FREQ)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--video-backend", default=DEFAULT_VIDEO_BACKEND, choices=["torchcodec", "pyav"])
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--n-action-steps", type=int, default=DEFAULT_N_ACTION_STEPS)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable the small photometric augmentation.")
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
