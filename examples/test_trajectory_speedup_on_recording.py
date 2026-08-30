#!/usr/bin/env python3
"""Diagnostic for the trajectory-aware kp schedule (no ROS, no robot).

Loads one episode from a LeRobot v3 dataset already on disk under
``~/.cache/huggingface/lerobot/<repo-id>/``, runs ``compute_speed_schedule``
for several lookahead values, and writes diagnostic PNGs so you can
sanity-check the schedule before touching the real robot.

Mirrors the dataset loader path used by ``17_replay_dataset.py`` (same
dataset layout, same action convention: dims 0:3 = xyz, 3:6 = rpy).

Usage (run from the same pixi env as 17_replay_dataset.py):

    cd Yunfei/crisp_gym
    pixi run -e jazzy-lerobot python examples/test_trajectory_speedup_on_recording.py \\
        --repo-id camera_test --episode-idx 0 \\
        --max-speed 4.0 --min-speed 1.0 --clamp-deg 5.0 \\
        --lookahead 0 --lookahead 5 --lookahead 15

Outputs (default to ``../notebooks/``):
    speed_schedule_ep<i>.png   — schedule overlay + per-step pos/ori deltas
    speed_segments_ep<i>.png   — quantized step-function preview (RPC count)
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPLAY_SCRIPT = THIS_DIR / "17_replay_dataset.py"
LEROBOT_CACHE = Path.home() / ".cache/huggingface/lerobot"


def _import_replay_module():
    """Load 17_replay_dataset.py as a module so we can reuse its helpers."""
    spec = importlib.util.spec_from_file_location(
        "replay_module", str(REPLAY_SCRIPT),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not build import spec for {REPLAY_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["replay_module"] = mod
    spec.loader.exec_module(mod)
    return mod


def _per_step_pos_delta_norm(actions: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(np.diff(actions[:, :3], axis=0), axis=-1)
    return np.concatenate([d, d[-1:]])


def _per_step_ori_delta_deg(actions: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(np.diff(actions[:, 3:6], axis=0), axis=-1)
    d = np.concatenate([d, d[-1:]])
    return np.degrees(d)


def _quantize(schedule: np.ndarray, step: float) -> np.ndarray:
    return np.round(schedule / step) * step


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repo-id", required=True,
        help="LeRobot dataset under ~/.cache/huggingface/lerobot/.",
    )
    parser.add_argument("--episode-idx", type=int, default=0)
    parser.add_argument("--max-speed", type=float, default=4.0)
    parser.add_argument("--min-speed", type=float, default=1.0)
    parser.add_argument("--clamp-deg", type=float, default=5.0)
    parser.add_argument(
        "--lookahead", type=int, action="append", default=None,
        help="Lookahead window (forward steps to aggregate). Pass multiple "
             "times to overlay (e.g. --lookahead 0 --lookahead 5). Defaults "
             "to [0, 5, 15] when unset.",
    )
    parser.add_argument(
        "--gain-quantize", type=float, default=0.05,
        help="Snap factors to nearest multiple of this value (default: 0.05). "
             "Bounds the SetParameters RPC count per replay.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=THIS_DIR.parent / "notebooks",
        help="Where to write the diagnostic PNGs (default: ../notebooks/).",
    )
    args = parser.parse_args()

    if args.lookahead is None:
        args.lookahead = [0, 5, 15]

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required: pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    rep = _import_replay_module()

    dataset_dir = LEROBOT_CACHE / args.repo_id
    if not dataset_dir.exists():
        print(f"Dataset not found: {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    info = rep.load_dataset_info(dataset_dir)
    episodes_df = rep.load_episodes_meta(dataset_dir)
    df = rep.load_episode_frames(dataset_dir, info, episodes_df, args.episode_idx)

    if len(df) == 0:
        print(f"Episode {args.episode_idx} has zero frames.", file=sys.stderr)
        sys.exit(1)

    actions = np.stack(
        [np.asarray(a, dtype=np.float64) for a in df["action"].to_numpy()],
        axis=0,
    )
    if actions.shape[1] < 6:
        print(
            f"action_dim={actions.shape[1]} < 6 — cannot compute schedule.",
            file=sys.stderr,
        )
        sys.exit(1)

    fps = float(info.get("fps", 30))
    n_frames = actions.shape[0]
    t_axis = np.arange(n_frames) / fps

    print(
        f"=== {args.repo_id} ep {args.episode_idx} | "
        f"{n_frames} frames @ {fps:.0f} Hz | action_dim={actions.shape[1]} ==="
    )
    print(
        f"max_speed={args.max_speed}  min_speed={args.min_speed}  "
        f"clamp_deg={args.clamp_deg}  lookaheads={args.lookahead}  "
        f"quantize={args.gain_quantize}"
    )

    schedules: dict[int, np.ndarray] = {}
    for n in args.lookahead:
        s = rep.compute_speed_schedule(
            actions[:, :6],
            max_speed=args.max_speed,
            min_speed=args.min_speed,
            clamp_deg=args.clamp_deg,
            n_lookahead=int(n),
        )
        schedules[int(n)] = s
        q = _quantize(s, args.gain_quantize)
        n_segments = int((np.diff(q) != 0).sum()) + 1
        print(
            f"  lookahead={n:>3}  peak={s.max():.3f}  floor={s.min():.3f}  "
            f"mean={s.mean():.3f}  segments@quantize={args.gain_quantize:.3f}: "
            f"{n_segments}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Figure 1: schedule overlay + per-step pos/ori deltas ──
    pos_delta_mm = _per_step_pos_delta_norm(actions) * 1000.0
    ori_delta_deg = _per_step_ori_delta_deg(actions)

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

    ax = axes[0]
    for n, s in schedules.items():
        ax.plot(t_axis, s, label=f"lookahead={n}", linewidth=1.0)
    ax.axhline(args.max_speed, color="k", linestyle=":", linewidth=0.5)
    ax.axhline(args.min_speed, color="k", linestyle=":", linewidth=0.5)
    ax.set_ylabel("kp scale factor")
    ax.set_title(
        f"Speed schedule — {args.repo_id} ep {args.episode_idx} "
        f"(max={args.max_speed}, min={args.min_speed}, "
        f"clamp={args.clamp_deg}°)"
    )
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t_axis, pos_delta_mm, color="tab:blue", linewidth=0.8)
    ax.set_ylabel("‖Δposition‖ (mm/step)")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(t_axis, ori_delta_deg, color="tab:orange", linewidth=0.8)
    ax.axhline(
        args.clamp_deg, color="k", linestyle=":", linewidth=0.5,
        label=f"clamp_deg={args.clamp_deg}",
    )
    ax.set_ylabel("‖Δorientation‖ (deg/step)")
    ax.set_xlabel("time (s)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_a = args.out_dir / f"speed_schedule_ep{args.episode_idx}.png"
    fig.savefig(out_a, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_a}")

    # ── Figure 2: quantized step-function preview ──
    fig, ax = plt.subplots(1, 1, figsize=(11, 4))
    for n, s in schedules.items():
        q = _quantize(s, args.gain_quantize)
        n_segments = int((np.diff(q) != 0).sum()) + 1
        ax.step(
            t_axis, q, where="post",
            label=f"lookahead={n}  segments={n_segments}",
        )
    ax.set_ylabel("quantized kp factor")
    ax.set_xlabel("time (s)")
    ax.set_title(
        f"Quantized step-function (quantize={args.gain_quantize}) — "
        "SetParameters RPCs the segmenter would emit per replay"
    )
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_b = args.out_dir / f"speed_segments_ep{args.episode_idx}.png"
    fig.savefig(out_b, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_b}")


if __name__ == "__main__":
    main()
