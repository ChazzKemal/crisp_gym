#!/usr/bin/env python3
"""Train an ACT policy on a B-spline-action dataset.

Why ACT with ``chunk_size=1``
-----------------------------
Upstream predicts the *whole* B-spline parameter matrix from one observation
window -- their ``DiffusionUnetBSplineImagePolicy`` sets
``n_action_steps = horizon`` and its ``select_action`` returns the full
prediction untouched. The temporal structure lives *inside* the action vector
(knots + control points), not across dataset frames, so there is nothing for a
policy-level action chunk to stack. We therefore flatten the parameter matrix
into one ``(chunk_size + 2 * degree) * 11`` vector per frame and ask ACT for a
single such vector: obs -> parameters, exactly the upstream contract, with a
CVAE where they use a diffusion UNet.

This also keeps the run directly comparable to the existing
``act_cart7_v2_angleaxis_nogrip_chunk100*`` baselines: same dataset, same
cameras, same augmentation -- only the action representation differs.

    conda run -n lerobot-041 python train_bspline_act.py --wandb

Anything after ``--`` is forwarded verbatim to ``lerobot-train``.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

DEFAULT_ROOT = Path("/home/batur/Coding/data/merged_bspline_20260528")
DEFAULT_REPO_ID = "merged_bspline_20260528"
DEFAULT_OUTPUT_DIR = Path("outputs/train/bspline_act_merged_20260528")
DEFAULT_STEPS = 30_000
DEFAULT_BATCH_SIZE = 32
DEFAULT_LOG_FREQ = 100
DEFAULT_SAVE_FREQ = 5_000
DEFAULT_NUM_WORKERS = 4
DEFAULT_WANDB_PROJECT = "bspline_act_merged_20260528"

# Identical to the ACT baselines in examples/29 and examples/30.
PHOTOMETRIC_TFS = (
    "{brightness: {weight: 1.0, type: ColorJitter, kwargs: {brightness: [0.8, 1.2]}}, "
    "contrast: {weight: 1.0, type: ColorJitter, kwargs: {contrast: [0.8, 1.2]}}, "
    "saturation: {weight: 1.0, type: ColorJitter, kwargs: {saturation: [0.5, 1.5]}}, "
    "hue: {weight: 1.0, type: ColorJitter, kwargs: {hue: [-0.05, 0.05]}}, "
    "sharpness: {weight: 1.0, type: SharpnessJitter, kwargs: {sharpness: [0.5, 1.5]}}}"
)


def _build_command(args: argparse.Namespace, extra: list[str]) -> list[str]:
    if args.resume:
        # LeRobot reloads the whole train config (and the optimizer state) from
        # the checkpoint, so nothing else may be passed on the command line.
        # config_path must be the train_config.json FILE: lerobot takes its
        # parent as the model dir and that parent's parent as the checkpoint.
        cfg = args.output_dir / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
        if not cfg.exists():
            raise FileNotFoundError(f"{cfg} not found; nothing to resume")
        step = (args.output_dir / "checkpoints" / "last").resolve().name
        print(f"Resuming {args.output_dir} from step {step}")
        return ["lerobot-train", f"--config_path={cfg}", "--resume=true", *extra]

    info_path = args.dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(
            f"{info_path} not found -- run convert_lerobot_to_bspline.py first"
        )
    sidecar = args.dataset_root / "meta" / "bspline.json"
    if not sidecar.exists():
        raise FileNotFoundError(
            f"{sidecar} not found -- {args.dataset_root} is not a B-spline dataset"
        )
    meta = json.loads(sidecar.read_text())
    action_dim = json.loads(info_path.read_text())["features"]["action"]["shape"][0]
    if action_dim != meta["flat_action_dim"]:
        raise ValueError(
            f"info.json action dim {action_dim} != sidecar {meta['flat_action_dim']}"
        )
    print(
        f"B-spline dataset: chunk_size={meta['chunk_size']} degree={meta['degree']} "
        f"max_error={meta['max_error']} -> action dim {action_dim} "
        f"({meta['n_action_steps']} x {meta['n_action_channels']})"
    )

    cmd = [
        "lerobot-train",
        "--policy.type=act",
        "--policy.push_to_hub=false",
        # One parameter matrix per observation -- see the module docstring.
        "--policy.chunk_size=1",
        "--policy.n_action_steps=1",
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
        cmd += ["--wandb.enable=true", f"--wandb.project={args.wandb_project}"]
    cmd.extend(extra)
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    ap.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--log-freq", type=int, default=DEFAULT_LOG_FREQ)
    ap.add_argument("--save-freq", type=int, default=DEFAULT_SAVE_FREQ)
    ap.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    ap.add_argument("--video-backend", default="pyav", choices=["torchcodec", "pyav"])
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="continue --output-dir from checkpoints/last (config comes from there)")
    ap.add_argument("--print-only", action="store_true")

    argv = sys.argv[1:]
    extra: list[str] = []
    if "--" in argv:
        sep = argv.index("--")
        argv, extra = argv[:sep], argv[sep + 1:]
    args = ap.parse_args(argv)

    cmd = _build_command(args, extra)
    print("\n+ " + " ".join(shlex.quote(c) for c in cmd) + "\n", flush=True)
    if args.print_only:
        return 0
    env = os.environ.copy()
    # lerobot-train is a separate process; without this its stdout is block
    # buffered when piped to a log file and progress appears only in bursts.
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
