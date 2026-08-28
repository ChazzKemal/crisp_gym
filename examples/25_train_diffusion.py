#!/usr/bin/env python3
"""Train a Diffusion Policy on the merged `pickplace_merged_nostall` dataset.

Thin wrapper around `lerobot-train` (the Draccus-based CLI exposed by the
LeRobot package; run this inside the `lerobot-041` conda env). The Diffusion
counterpart of `24_train_act.py` -- same dataset, same dual-cam UR10e setup:

  /home/batur/Coding/data/pickplace_merged_nostall/
    71 episodes, fps=20  (de-stalled gripper + 232116 recordings, merged)
    observation.images.camera   video, 480x640x3
    observation.images.d405     video, 480x640x3
    observation.state           float32, 7-dim  (xyz + rpy + gripper)  -- input
    action                      float32, 7-dim  (xyz + rpy + gripper)  -- target

Diffusion-specific defaults:
  * `n_obs_steps=2`     -- the policy conditions on 2 stacked observations.
  * `horizon=16`        -- diffusion action-prediction horizon.
  * `n_action_steps=8`  -- actions executed per policy call.
  * `crop_shape=null`   -- disables Diffusion's built-in 84x84 random crop,
    which is sized for tiny PushT images and wrong for 480x640 frames.
    Augmentation comes from the dataset `image_transforms` instead.

By default photometric image augmentation is enabled (brightness / contrast /
saturation / hue / sharpness; geometric affine disabled because the policy
regresses end-effector poses). Pass `--no-augment` to turn it off.

Usage (run inside the `lerobot-041` conda env):
    cd Yunfei/crisp_gym
    conda run -n lerobot-041 python examples/25_train_diffusion.py
    conda run -n lerobot-041 python examples/25_train_diffusion.py --steps 50000 --batch-size 32

Anything after `--` is forwarded verbatim to `lerobot-train`, so you can
override any Diffusion hyperparameter without touching this script::

    conda run -n lerobot-041 python examples/25_train_diffusion.py -- \\
        --policy.num_train_timesteps=50 --num_workers=2

Outputs (under --output-dir, default outputs/train/diffusion_merged):
    checkpoints/<step>/    full pretrained checkpoint per --save-freq
    logs/                  csv + (optional) wandb logs

To resume from the last checkpoint, pass `--resume` to this script.
"""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


# --- Dataset defaults ---
# NOTE: LeRobot's `root` is the *full* path to the dataset directory (the one
# that contains `meta/info.json`), NOT the parent. `repo_id` is just an id.
DEFAULT_REPO_ID = "pickplace_merged_nostall"
DEFAULT_ROOT = Path("/home/batur/Coding/data/pickplace_merged_nostall")

# --- Training defaults ---
DEFAULT_OUTPUT_DIR = Path("outputs/train/diffusion_merged")
DEFAULT_STEPS = 100_000
DEFAULT_BATCH_SIZE = 64
DEFAULT_LOG_FREQ = 100
DEFAULT_SAVE_FREQ = 5_000
DEFAULT_NUM_WORKERS = 4
DEFAULT_VIDEO_BACKEND = "pyav"  # torchcodec is broken in the lerobot-041 env
DEFAULT_WANDB_PROJECT = "diffusion_merged"

# --- Diffusion defaults ---
DEFAULT_HORIZON = 16          # diffusion action-prediction horizon
DEFAULT_N_ACTION_STEPS = 8    # actions executed per policy call
DEFAULT_N_OBS_STEPS = 2       # stacked observations the policy conditions on

# Photometric-only image augmentation. lerobot's `tfs` dict is REPLACED whole
# by this draccus override (a dotted `tfs.affine.weight=0` override is not
# supported), so all five colour/sharpness transforms are listed explicitly
# and the geometric `affine` transform is simply left out — it is unsafe here
# because the policy regresses end-effector poses (moving pixels without
# moving the action label teaches the wrong thing).
PHOTOMETRIC_TFS = (
    "{brightness: {weight: 1.0, type: ColorJitter, kwargs: {brightness: [0.8, 1.2]}}, "
    "contrast: {weight: 1.0, type: ColorJitter, kwargs: {contrast: [0.8, 1.2]}}, "
    "saturation: {weight: 1.0, type: ColorJitter, kwargs: {saturation: [0.5, 1.5]}}, "
    "hue: {weight: 1.0, type: ColorJitter, kwargs: {hue: [-0.05, 0.05]}}, "
    "sharpness: {weight: 1.0, type: SharpnessJitter, kwargs: {sharpness: [0.5, 1.5]}}}"
)


def _build_command(args: argparse.Namespace, extra: list[str]) -> list[str]:
    """Compose the lerobot-train CLI invocation."""
    dataset_root = args.dataset_root
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(
            f"Dataset meta/info.json not found at {info_path}. "
            f"--dataset-root must point at a directory that contains "
            f"meta/info.json (the standard LeRobot v3 layout)."
        )

    cmd = [
        "lerobot-train",
        "--policy.type=diffusion",
        f"--policy.horizon={args.horizon}",
        f"--policy.n_action_steps={args.n_action_steps}",
        f"--policy.n_obs_steps={args.n_obs_steps}",
        # Disable the built-in 84x84 crop (PushT-sized); rely on the dataset
        # image_transforms for augmentation instead.
        "--policy.crop_shape=null",
        # Default to NOT pushing to the HF Hub. Without this, the validator
        # demands --policy.repo_id even when nobody wants to push.
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
        # Photometric-only augmentation on both cameras (see PHOTOMETRIC_TFS).
        cmd += [
            "--dataset.image_transforms.enable=true",
            "--dataset.image_transforms.max_num_transforms=3",
            f"--dataset.image_transforms.tfs={PHOTOMETRIC_TFS}",
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
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repo-id", default=DEFAULT_REPO_ID,
        help=f"LeRobotDataset repo id (default: {DEFAULT_REPO_ID}).",
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=DEFAULT_ROOT,
        help=f"Full path to the dataset directory (contains meta/info.json) "
             f"(default: {DEFAULT_ROOT}).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Checkpoint + log directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--steps", type=int, default=DEFAULT_STEPS,
        help=f"Total optimizer steps (default: {DEFAULT_STEPS}).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Mini-batch size (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--log-freq", type=int, default=DEFAULT_LOG_FREQ,
        help=f"Log scalar metrics every N steps (default: {DEFAULT_LOG_FREQ}).",
    )
    parser.add_argument(
        "--save-freq", type=int, default=DEFAULT_SAVE_FREQ,
        help=f"Save a full checkpoint every N steps (default: {DEFAULT_SAVE_FREQ}).",
    )
    parser.add_argument(
        "--num-workers", type=int, default=DEFAULT_NUM_WORKERS,
        help=f"DataLoader workers (default: {DEFAULT_NUM_WORKERS}).",
    )
    parser.add_argument(
        "--video-backend", default=DEFAULT_VIDEO_BACKEND,
        choices=["torchcodec", "pyav"],
        help=f"Video decoder backend (default: {DEFAULT_VIDEO_BACKEND}).",
    )
    parser.add_argument(
        "--horizon", type=int, default=DEFAULT_HORIZON,
        help=f"Diffusion action-prediction horizon (default: {DEFAULT_HORIZON}).",
    )
    parser.add_argument(
        "--n-action-steps", type=int, default=DEFAULT_N_ACTION_STEPS,
        help=f"Actions executed per policy call (default: {DEFAULT_N_ACTION_STEPS}).",
    )
    parser.add_argument(
        "--n-obs-steps", type=int, default=DEFAULT_N_OBS_STEPS,
        help=f"Stacked observation steps (default: {DEFAULT_N_OBS_STEPS}).",
    )
    parser.add_argument(
        "--seed", type=int, default=1000,
        help="RNG seed (default: 1000).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from the latest checkpoint under --output-dir.",
    )
    parser.add_argument(
        "--no-augment", action="store_true",
        help="Disable photometric image augmentation (enabled by default).",
    )
    parser.add_argument(
        "--wandb", action="store_true",
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb-project", default=DEFAULT_WANDB_PROJECT,
        help=f"W&B project name (default: {DEFAULT_WANDB_PROJECT}).",
    )
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Disable wandb logging (overrides --wandb).",
    )
    parser.add_argument(
        "--print-only", action="store_true",
        help="Print the lerobot-train command that would run and exit.",
    )

    # Split off pass-through args (anything after `--`).
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

    # Inherit env (especially HF_HOME, ROS_*) so the train run sees the same
    # configuration as this wrapper.
    return subprocess.call(cmd, env=os.environ.copy())


if __name__ == "__main__":
    sys.exit(main())
