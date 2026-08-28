#!/usr/bin/env python3
"""Fine-tune SmolVLA on `pickplace_cart7_v2_angleaxis_nogrip` for RTC deploy.

Why this script
---------------
SmolVLA is the smallest of the flow-matching policies in the `lerobot_uncertainty`
fork (alongside Pi0 / Pi0.5) that are compatible with **Real-Time Chunking (RTC)**,
the inference-time technique in `lerobot.policies.rtc`. SmolVLA's SmolVLM2-500M
backbone is ~6x smaller than Pi0/Pi0.5 (PaliGemma 3B), so it is by far the
fastest RTC-compatible policy to deploy on the UR10/Clearpath stack.

Counterpart to `26_train_act_cart7_v2.py`: same wrapper pattern (subprocess to
`lerobot-train`, `--` passthrough), same half-strength photometric augmentation,
but targeting SmolVLA on the angle-axis no-gripper cart7 dataset.

Environment
-----------
This script must run in the `lerobot-rtc` conda env (lerobot 0.4.2 editable from
`/home/batur/lerobot_uncertainty.worktrees/main`, the only env that has both
working `smolvla` and the `rtc` module). NOT `lerobot-041` (no smolvla/rtc) and
NOT `lerobot` (broken smolvla on branch `demospeedup-alpay`).

Usage:
    cd ur10_clearpath/Yunfei/crisp_gym
    conda run -n lerobot-rtc python examples/27_train_smolvla_cart7_v2_angleaxis_nogrip.py
    conda run -n lerobot-rtc python examples/27_train_smolvla_cart7_v2_angleaxis_nogrip.py --steps 50000

Anything after ``--`` is forwarded verbatim to ``lerobot-train``.
"""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_REPO_ID = "pickplace_cart7_v2_angleaxis_nogrip"
DEFAULT_ROOT = Path("/home/batur/Coding/data/pickplace_cart7_v2_angleaxis_nogrip")

# Warm-start from the public SmolVLA base checkpoint on HF.
DEFAULT_INIT_CKPT = "lerobot/smolvla_base"

DEFAULT_OUTPUT_DIR = Path("outputs/train/smolvla_cart7_v2_angleaxis_nogrip")
DEFAULT_STEPS = 30_000
DEFAULT_BATCH_SIZE = 8       # SmolVLA + dual 480x640 cams on a 24 GB GPU
DEFAULT_LOG_FREQ = 100
DEFAULT_SAVE_FREQ = 5_000
DEFAULT_NUM_WORKERS = 4
DEFAULT_VIDEO_BACKEND = "pyav"
DEFAULT_WANDB_PROJECT = "smolvla_cart7_v2_angleaxis_nogrip"

# SmolVLA defaults: chunk_size=50, n_action_steps=50, num_steps=10 (flow ODE).
# Leaving chunk_size large gives RTC headroom (execution_horizon < chunk_size).
DEFAULT_CHUNK_SIZE = 50
DEFAULT_N_ACTION_STEPS = 50

# smolvla_base expects observation.images.camera1/camera2/camera3; our dataset
# has two cameras. Map ours to camera1/camera2 and let camera3 stay missing
# (validate_visual_features_consistency only checks `provided.issubset(expected)`,
# and at runtime `_prepare_images` simply emits no tokens for the missing slot).
RENAME_MAP = (
    '{"observation.images.camera": "observation.images.camera1", '
    '"observation.images.d405": "observation.images.camera2"}'
)

# Half-strength photometric jitter (same as 26_train_act_cart7_v2.py).
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

    cmd = [
        "lerobot-train",
        # Warm-start: load SmolVLA base weights + processors from HF.
        f"--policy.path={args.init_checkpoint}",
        f"--policy.chunk_size={args.chunk_size}",
        f"--policy.n_action_steps={args.n_action_steps}",
        "--policy.push_to_hub=false",
        f"--dataset.repo_id={args.repo_id}",
        f"--dataset.root={dataset_root}",
        f"--dataset.video_backend={args.video_backend}",
        f"--rename_map={RENAME_MAP}",
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
    parser.add_argument("--init-checkpoint", default=DEFAULT_INIT_CKPT,
                        help="HF repo id or local dir to warm-start from (--policy.path).")
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
