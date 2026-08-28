#!/usr/bin/env python3
"""Build the data payload for the B-spline walkthrough page.

Emits ``walkthrough_data.json`` and splices it into ``walkthrough_template.html``
to produce a self-contained ``walkthrough.html``. Everything on the page comes
from one real recorded episode -- nothing is illustrative or synthetic.

    conda run -n lerobot-041 python build_walkthrough.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from bspline_core.bspline_action import (  # noqa: E402
    ScipyBSplineCompression,
    chunk_bspline_trajectory,
    chunk_to_params,
    decode_bspline_action,
)
from bspline_core.chunk_sampler import BSplineChunkSampler  # noqa: E402
from lerobot_bridge import load_lerobot_actions, to_policy_actions  # noqa: E402

SRC = "/home/batur/Coding/data/merged_act_finetune_20260528"
EPISODE = 58
DEGREE = 3
CHUNK_SIZE = 20
MAX_ERROR = 0.01
PLOT_DIMS = [0, 1, 2, 9]  # x, y, z, gripper -- what the page draws
DIM_LABELS = ["x", "y", "z", "gripper"]


def r(a, nd=5):
    """Round for a compact payload; 5 decimals is ~10 micrometres."""
    return np.round(np.asarray(a, dtype=float), nd).tolist()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--episode", type=int, default=EPISODE)
    ap.add_argument("--out", type=Path, default=HERE / "walkthrough.html")
    args = ap.parse_args()

    data = load_lerobot_actions(args.src)
    a = int(data.episode_starts[args.episode])
    b = int(data.episode_ends[args.episode])
    raw = data.actions[a:b]            # (T, 7) xyz + axis-angle + gripper
    policy = to_policy_actions(raw)    # (T, 10) xyz + rot6d + gripper
    T = len(raw)
    print(f"episode {args.episode}: {T} frames @ {data.fps} fps")

    # ---------------------------------------------------------------- the fit
    comp = ScipyBSplineCompression(degree=DEGREE)
    comp.compress(policy, max_error=MAX_ERROR)
    t_full, c_full, _ = comp.spline.tck
    print(f"  knots {len(t_full)} ({len(t_full) / T:.3f}/frame), fit error {comp.fit_error:.5f}")

    fitted = comp.spline(np.arange(T))
    fit_resid_mm = np.linalg.norm(fitted[:, :3] - policy[:, :3], axis=1) * 1000

    # ------------------------------------------------------------- the chunks
    chunks = chunk_bspline_trajectory(comp, chunk_size=CHUNK_SIZE, stride=1)
    sampler = BSplineChunkSampler(
        actions=policy, episode_ends=np.array([T]), chunk_size=CHUNK_SIZE,
        degree=DEGREE, max_error=MAX_ERROR, stride=1, max_first_k=1,
    )
    n_steps = CHUNK_SIZE + 2 * DEGREE

    # Which knot-vector window each frame is served by. The rule is "the last
    # chunk whose valid domain has not started yet", i.e. the smallest window
    # start s with t_full[s + degree] >= ts. Derived independently here and
    # cross-checked against the sampler, so the page and the pipeline agree.
    domain_starts = t_full[DEGREE : DEGREE + len(chunks)]
    win_start, win_lo, win_hi = [], [], []
    for ts in range(T):
        s_idx = int(np.searchsorted(domain_starts, ts, side="left"))
        s_idx = min(s_idx, len(chunks) - 1)
        win_start.append(s_idx)
        p = sampler.chunk_for_timestep(ts)
        win_lo.append(float(p[DEGREE, 0]) + ts)
        win_hi.append(float(p[n_steps - DEGREE - 1, 0]) + ts)
        expected = chunk_to_params(chunks[s_idx], n_steps, 11)
        if not np.allclose(expected[:, 0] - ts, p[:, 0], atol=1e-3):
            raise AssertionError(f"frame {ts}: derived window {s_idx} disagrees with sampler")

    # one fully worked example, shown as a parameter-matrix heatmap
    example_frame = int(T * 0.35)
    example = sampler.chunk_for_timestep(example_frame)

    # ------------------------------------------------- decode-vs-truth error
    frames = np.arange(T)
    pos_mm, rot_deg, spans = [], [], []
    for ts in range(T):
        p = sampler.chunk_for_timestep(ts)
        t_min = float(p[DEGREE, 0])
        t_max = float(p[n_steps - DEGREE - 1, 0])
        spans.append(t_max - t_min)
        if t_max <= t_min or ts + t_max >= T - 1:
            pos_mm.append(np.nan)
            rot_deg.append(np.nan)
            continue
        dec = decode_bspline_action(p, degree=DEGREE, num_actions=16)
        tt = ts + np.linspace(t_min, t_max, 16)
        truth = np.stack([np.interp(tt, frames, raw[:, d]) for d in range(7)], axis=1)
        pos_mm.append(float(np.abs(dec[:, :3] - truth[:, :3]).max() * 1000))
        dec_rot = Rotation.from_matrix(_rot6d_to_matrix(dec[:, 3:9]))
        rot_deg.append(float(np.degrees(
            (dec_rot * Rotation.from_rotvec(truth[:, 3:6]).inv()).magnitude().max())))

    # ------------------------------------------- accuracy / compression sweep
    sweep = []
    for me in (0.002, 0.005, 0.01, 0.02, 0.05):
        c = ScipyBSplineCompression(degree=DEGREE)
        c.compress(policy, max_error=me)
        err = np.abs(c.spline(frames) - policy).max(axis=0)
        sweep.append({
            "max_error": me,
            "knots_per_frame": len(c.knots) / T,
            "pos_mm": float(err[:3].max() * 1000),
            "span": CHUNK_SIZE / (len(c.knots) / T),
        })

    payload = {
        "source": {"dataset": Path(args.src).name, "episode": args.episode,
                   "frames": T, "fps": data.fps,
                   "task": "infinite pick and place, green then blue"},
        "config": {"degree": DEGREE, "chunk_size": CHUNK_SIZE, "max_error": MAX_ERROR,
                   "n_steps": n_steps, "n_channels": 11,
                   "flat_dim": n_steps * 11},
        "dims": DIM_LABELS,
        "raw": {lbl: r(raw[:, d] if d != 9 else raw[:, 6])
                for lbl, d in zip(DIM_LABELS, PLOT_DIMS)},
        "knots": r(t_full, 4),
        "coef": [r(c_full[:, d], 5) for d in range(10)],  # all ten, for the matrix figure
        "fit": {"error": float(comp.fit_error), "resid_mm": r(fit_resid_mm, 3),
                "knots_per_frame": len(t_full) / T,
                "n_control_points": int(len(c_full))},
        "chunks": {"n": len(chunks), "win_start": win_start,
                   "win_lo": r(win_lo, 3), "win_hi": r(win_hi, 3),
                   "spans": r(spans, 2)},
        "example": {"frame": example_frame, "params": r(example, 4)},
        "error": {"pos_mm": r(pos_mm, 2), "rot_deg": r(rot_deg, 3)},
        "sweep": sweep,
    }

    blob = json.dumps(payload, separators=(",", ":"), allow_nan=True).replace("NaN", "null")
    template = (HERE / "walkthrough_template.html").read_text()
    marker = "/*__DATA__*/"
    if marker not in template:
        raise RuntimeError(f"{marker} not found in walkthrough_template.html")
    args.out.write_text(template.replace(marker, blob))
    print(f"  wrote {args.out} ({len(blob) / 1024:.0f} KB of data)")
    return 0


def _rot6d_to_matrix(d6):
    d6 = np.asarray(d6, dtype=np.float64)
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    return np.stack((b1, b2, np.cross(b1, b2)), axis=-2)


if __name__ == "__main__":
    raise SystemExit(main())
