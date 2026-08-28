#!/usr/bin/env python3
"""Sweep B-spline fit accuracy vs. compression on a real crisp_gym dataset.

Run before committing to a ``max_error`` / ``chunk_size``: the whole point of
the representation is that a chunk covers many source frames, and that only
holds if the fit actually compresses. A binary gripper channel is the worst
case for a smooth spline, so this reports the pose-only fit separately.

    conda run -n lerobot-041 python analyze_fit.py --root <LEROBOT_DATASET>
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bspline_core.bspline_action import ScipyBSplineCompression  # noqa: E402
from lerobot_bridge import load_lerobot_actions, to_policy_actions  # noqa: E402

DEFAULT_ROOT = "/home/batur/Coding/data/merged_act_finetune_20260528"


def fit_report(actions: np.ndarray, ends: np.ndarray, max_error: float, degree: int,
               n_episodes: int | None) -> dict:
    starts = np.concatenate([[0], ends[:-1]])
    ratios, errors, converged, times = [], [], 0, []
    pairs = list(zip(starts, ends))[:n_episodes]
    for a, b in pairs:
        t0 = time.perf_counter()
        comp = ScipyBSplineCompression(degree=degree)
        comp.compress(actions[a:b], max_error=max_error)
        times.append(time.perf_counter() - t0)
        ratios.append(len(comp.knots) / (b - a))
        errors.append(comp.fit_error)
        converged += int(comp.converged)
    return {
        "n": len(pairs),
        "ratio_mean": float(np.mean(ratios)),
        "ratio_max": float(np.max(ratios)),
        "err_max": float(np.max(errors)),
        "converged": converged,
        "sec_total": float(np.sum(times)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--degree", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=20, help="episodes to sample (0 = all)")
    ap.add_argument("--max-errors", type=float, nargs="+",
                    default=[0.002, 0.005, 0.01, 0.02, 0.05])
    ap.add_argument("--chunk-sizes", type=int, nargs="+", default=[10, 16, 20])
    ap.add_argument("--coverage", action="store_true",
                    help="also report how many frames a decoded chunk spans")
    args = ap.parse_args()

    data = load_lerobot_actions(args.root)
    policy = to_policy_actions(data.actions)
    n_ep = None if args.episodes == 0 else args.episodes
    lengths = np.diff(np.concatenate([[0], data.episode_ends]))
    print(f"{args.root}\n  {len(data.episode_ends)} episodes, {len(data.actions)} frames "
          f"@ {data.fps} fps; lengths {lengths.min()}-{lengths.max()}")
    print(f"  fitting {n_ep or len(lengths)} episodes, degree={args.degree}\n")

    header = f"{'max_error':>10} {'variant':>10} {'ratio_mean':>11} {'ratio_max':>10} " \
             f"{'err_max':>9} {'conv':>7} {'sec':>7}"
    print(header)
    print("-" * len(header))
    for max_error in args.max_errors:
        for name, arr in (("full-10d", policy), ("pose-9d", policy[:, :9])):
            r = fit_report(arr, data.episode_ends, max_error, args.degree, n_ep)
            print(f"{max_error:>10.4f} {name:>10} {r['ratio_mean']:>11.3f} "
                  f"{r['ratio_max']:>10.3f} {r['err_max']:>9.4f} "
                  f"{r['converged']:>3}/{r['n']:<3} {r['sec_total']:>7.1f}")

    if args.coverage:
        print("\nChunk coverage -- how far into the future one chunk reaches.")
        print("A chunk spans `chunk_size` knot intervals, so span ~ chunk_size / ratio.")
        hdr = (f"\n{'max_error':>10} {'chunk':>6} {'span_p50':>9} {'span_p90':>9} "
               f"{'sec_p50':>8} {'speedup':>8} {'params':>7}")
        print(hdr)
        print("-" * (len(hdr) - 1))
        from bspline_core.chunk_sampler import BSplineChunkSampler
        ends = data.episode_ends if n_ep is None else data.episode_ends[:n_ep]
        acts = policy[: ends[-1]]
        for max_error in args.max_errors:
            for chunk_size in args.chunk_sizes:
                sampler = BSplineChunkSampler(
                    actions=acts, episode_ends=ends, chunk_size=chunk_size,
                    degree=args.degree, max_error=max_error, stride=1, max_first_k=2,
                )
                steps = chunk_size + 2 * args.degree
                span = (sampler.all_actions[:, steps - args.degree - 1, 0]
                        - sampler.all_actions[:, args.degree, 0])
                span = span[span > 0]
                p50, p90 = np.percentile(span, [50, 90])
                n_params = steps * (1 + acts.shape[1])
                # frames covered per predicted parameter, vs. a dense chunk of
                # the same parameter count (which would cover n_params/10 frames)
                speedup = p50 / (n_params / acts.shape[1])
                print(f"{max_error:>10.4f} {chunk_size:>6} {p50:>9.1f} {p90:>9.1f} "
                      f"{p50 / data.fps:>8.2f} {speedup:>8.2f} {n_params:>7}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
