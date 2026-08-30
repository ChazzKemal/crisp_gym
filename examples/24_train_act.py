#!/usr/bin/env python3
"""Train ACT on the local `speedup_20260515_230303` LeRobot v3 dataset.

Thin wrapper around `lerobot-train` (the Draccus-based CLI exposed by the
LeRobot package installed in the `jazzy-lerobot` pixi env). Sets sane
defaults for the dual-cam UR10e recording the rest of this repo is wired
around:

  /home/ali/.cache/huggingface/lerobot/speedup_20260515_230303/
    64 episodes, 13093 frames, fps=20
    observation.images.camera   video, 480x640x3
    observation.images.d405     video, 480x640x3
    observation.state           float32, 12-dim (6 joints + 6 cartesian)
    action                      float32, 7-dim  (xyz + rpy + gripper)

Usage:
    cd Yunfei/crisp_gym
    pixi run -e jazzy-lerobot python examples/24_train_act.py
    pixi run -e jazzy-lerobot python examples/24_train_act.py --steps 50000 --batch-size 16
    pixi run -e jazzy-lerobot python examples/24_train_act.py --output-dir outputs/train/act_v2

Anything after `--` is forwarded verbatim to `lerobot-train`, so you can
override any ACT hyperparameter without touching this script::

    pixi run -e jazzy-lerobot python examples/24_train_act.py -- \\
        --policy.chunk_size=50 --policy.kl_weight=5.0 --num_workers=2

Outputs (under --output-dir, default outputs/train/act_speedup):
    checkpoints/<step>/    full pretrained checkpoint per --save-freq
    logs/                  csv + (optional) wandb logs

To resume from the last checkpoint, pass `--resume` to this script —
lerobot-train will pick up output_dir's latest checkpoint automatically.

Notes:
  * ACT's `n_obs_steps` is fixed at 1; only `chunk_size` and
    `n_action_steps` matter for the action horizon.
  * The dataset has 13093 frames across 64 episodes. At --batch-size 8
    that is ~1600 steps per pass, so the default --steps 100000 is ~60
    passes. Expect overfitting in the later steps — save checkpoints and
    pick the best by deploying, since BC train-loss alone is not a
    quality signal.
"""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


# --- Dataset defaults ---
# NOTE: LeRobot's `root` is the *full* path to the dataset directory (the
# one that contains `meta/info.json`), NOT the parent of the dataset. The
# `repo_id` is just an identifier — LeRobot does not concatenate them.
# See LeRobotDatasetMetadata.__init__ in lerobot/datasets/lerobot_dataset.py.
DEFAULT_REPO_ID = "speedup_20260515_230303"
DEFAULT_ROOT = Path("/home/ali/.cache/huggingface/lerobot/speedup_20260515_230303")

# --- Training defaults ---
DEFAULT_OUTPUT_DIR = Path("outputs/train/act_speedup")
DEFAULT_STEPS = 100_000
DEFAULT_BATCH_SIZE = 8
DEFAULT_LOG_FREQ = 100
DEFAULT_SAVE_FREQ = 5_000
DEFAULT_NUM_WORKERS = 4

# --- ACT defaults ---
DEFAULT_CHUNK_SIZE = 100         # ACT's max action horizon
DEFAULT_N_ACTION_STEPS = 100     # steps executed per policy call


def _build_command(args: argparse.Namespace, extra: list[str]) -> list[str]:
    """Compose the lerobot-train CLI invocation."""
    dataset_root = args.dataset_root
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(
            f"Dataset meta/info.json not found at {info_path}. "
            f"--dataset-root must point at a directory that contains meta/info.json "
            f"(the standard LeRobot v3 layout)."
        )

    cmd = [
        "lerobot-train",
        "--policy.type=act",
        f"--policy.chunk_size={args.chunk_size}",
        f"--policy.n_action_steps={args.n_action_steps}",
        # Default to NOT pushing to the HF Hub. Without this, the validator
        # demands --policy.repo_id even when nobody actually wants to push.
        # Re-enable by appending `-- --policy.push_to_hub=true --policy.repo_id=...`.
        "--policy.push_to_hub=false",
        f"--dataset.repo_id={args.repo_id}",
        f"--dataset.root={dataset_root}",
        f"--output_dir={args.output_dir}",
        f"--steps={args.steps}",
        f"--batch_size={args.batch_size}",
        f"--log_freq={args.log_freq}",
        f"--save_freq={args.save_freq}",
        f"--num_workers={args.num_workers}",
        f"--seed={args.seed}",
    ]
    if args.resume:
        cmd.append("--resume=true")
    if args.no_wandb:
        cmd.append("--wandb.enable=false")
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
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
        help=f"ACT action chunk size (default: {DEFAULT_CHUNK_SIZE}).",
    )
    parser.add_argument(
        "--n-action-steps", type=int, default=DEFAULT_N_ACTION_STEPS,
        help=f"Actions executed per policy call (default: {DEFAULT_N_ACTION_STEPS}).",
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
        "--no-wandb", action="store_true",
        help="Disable wandb logging.",
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

    # Inherit env (especially PIXI_*, HF_HOME, ROS_*) so the train run sees
    # the same configuration as this wrapper.
    return subprocess.call(cmd, env=os.environ.copy())


if __name__ == "__main__":
    sys.exit(main())
