#!/usr/bin/env python3
"""Render the dump from ``run_inference_dump.py`` into figures.

Stage 2 of 2. Runs in any environment with matplotlib (the base conda env);
it never imports lerobot or torch -- everything it needs is in ``dump.npz``.

    /home/batur/miniconda3/bin/python bspline/build_inference_report.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.image import imread  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

# surface/ink from the house style shared with walkthrough.html and
# math_verification.html; series colours from the validated categorical palette
SURFACE = "#FCFDFC"
INK = "#141918"
INK2 = "#3B4544"
GRID = "#D3DAD7"
TRUTH = "#2a78d6"    # slot 1 -- recorded demonstration
PRED = "#eb6834"     # slot 2 -- B-spline policy
BASE = "#1baf7a"     # slot 3 -- ACT chunk-100 baseline

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "grid.color": GRID, "grid.linewidth": 0.6,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.dpi": 130,
})


def rot_err_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.degrees((Rotation.from_rotvec(a) * Rotation.from_rotvec(b).inv()).magnitude())


def set_3d_style(ax, path: np.ndarray) -> None:
    """Equal aspect around the trajectory, muted panes."""
    c = path.mean(axis=0)
    r = max(np.ptp(path, axis=0).max(), 0.05) / 2 * 1.15
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(SURFACE)
        axis.pane.set_edgecolor(GRID)
        axis._axinfo["grid"]["color"] = GRID
    ax.set_xlabel("x [m]", labelpad=-4)
    ax.set_ylabel("y [m]", labelpad=-4)
    ax.set_zlabel("z [m]", labelpad=-4)
    ax.tick_params(labelsize=7, pad=-1)
    ax.view_init(elev=22, azim=-58)


def snapshot_figure(k, d, meta, figdir, framedir):
    fps = meta["fps"]
    ep = d["ep_raw"]
    f = int(d["snap_frame"][k])
    wp, truth = d["snap_waypoints"][k], d["snap_truth"][k]
    times, absf = d["snap_times"][k], d["snap_abs_frames"][k]
    base = d["base_snap_chunks"][k] if "base_snap_chunks" in d else None

    pos_mm = 1000 * np.linalg.norm(wp[:, :3] - truth[:, :3], axis=1)
    rot_d = rot_err_deg(wp[:, 3:6], truth[:, 3:6])

    fig = plt.figure(figsize=(13.6, 6.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.0, 1.35], height_ratios=[1, 1],
                          hspace=0.34, wspace=0.36)

    for j, (cam, label) in enumerate([("camera", "observation.images.camera (scene)"),
                                      ("d405", "observation.images.d405 (wrist)")]):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(imread(framedir / f"snap{k}_{cam}.jpg"))
        ax.set_title(label, color=INK2, fontsize=8.5)
        ax.axis("off")

    # state read-out
    ax = fig.add_subplot(gs[1, 0])
    ax.axis("off")
    st = d["snap_state"][k]
    lines = [
        f"frame       {f} of {len(ep)}   ({f / fps:.2f} s)",
        "",
        "observation.state",
        f"  xyz       {st[0]:+.3f} {st[1]:+.3f} {st[2]:+.3f}  m",
        f"  rotvec    {st[3]:+.3f} {st[4]:+.3f} {st[5]:+.3f}  rad",
        "",
        "policy output",
        f"  {meta['bspline']['flat_action_dim']} numbers"
        f" = {meta['bspline']['n_action_steps']} knots x"
        f" {meta['bspline']['n_action_channels']} channels",
        f"  decodes to {len(wp)} waypoints over",
        f"  {d['snap_span_frames'][k]:.1f} frames = {d['snap_span_frames'][k] / fps:.2f} s ahead",
        f"  knots {'monotone' if d['snap_monotone'][k] else 'NON-monotone -> repaired'}",
        "",
        "error vs the recording",
        f"  position  {pos_mm.mean():5.1f} mean  {pos_mm.max():5.1f} max   mm",
        f"  rotation  {rot_d.mean():5.2f} mean  {rot_d.max():5.2f} max   deg",
    ]
    ax.text(0, 1.02, "\n".join(lines), va="top", ha="left", family="monospace",
            fontsize=7.6, color=INK, linespacing=1.5, transform=ax.transAxes)

    # xyz vs time
    ax = fig.add_subplot(gs[1, 1])
    ax.grid(True, alpha=0.7)
    for dim, name in enumerate("xyz"):
        ax.plot(times, truth[:, dim], color=TRUTH, lw=2,
                label="recorded demo" if dim == 0 else None)
        ax.plot(times, wp[:, dim], color=PRED, lw=2, ls="--", marker="o", ms=3,
                label="B-spline policy" if dim == 0 else None)
        ax.annotate(name, (times[-1], wp[-1, dim]), xytext=(4, -3),
                    textcoords="offset points", color=INK2, fontsize=8)
    ax.set_xlabel("seconds ahead of this observation")
    ax.set_ylabel("position [m]")
    ax.set_title("decoded waypoints vs recording", loc="left")
    ax.legend(loc="best", fontsize=8)

    # 3D
    ax = fig.add_subplot(gs[:, 2], projection="3d")
    ax.plot(*ep[:, :3].T, color=GRID, lw=1.2)
    seg = ep[f:min(len(ep), int(absf[-1]) + 1), :3]
    ax.plot(*seg.T, color=TRUTH, lw=2.6, label="recorded demo (same horizon)")
    if base is not None:
        n_same = max(2, min(len(base), int(round(absf[-1] - f))))
        ax.plot(*base[:n_same, :3].T, color=BASE, lw=1.8, alpha=0.9,
                label=f"ACT chunk-100 (first {n_same} of {len(base)})")
    ax.plot(*wp[:, :3].T, color=PRED, lw=2.2, ls="--", marker="o", ms=3.4,
            label="B-spline policy (16 waypoints)")
    ax.scatter(*ep[f, :3], color=INK, s=42, marker="*", zorder=5, label="current pose")
    set_3d_style(ax, ep[:, :3])
    ax.set_title("end-effector path", loc="left")
    ax.legend(loc="upper left", fontsize=7.2, bbox_to_anchor=(-0.10, 1.04))

    fig.suptitle(f"episode {meta['episode']} - frame {f} - one observation, one prediction",
                 x=0.012, ha="left", fontsize=12, fontweight="bold")
    out = figdir / f"snapshot_{k}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return dict(frame=f, t=f / fps, span_s=float(d["snap_span_frames"][k]) / fps,
                pos_mean=float(pos_mm.mean()), pos_max=float(pos_mm.max()),
                rot_mean=float(rot_d.mean()), rot_max=float(rot_d.max()),
                monotone=bool(d["snap_monotone"][k]), png=out.name)


def rollout_figure(d, meta, figdir):
    fps = meta["fps"]
    ep = d["ep_raw"]
    fig = plt.figure(figsize=(12.6, 5.0))
    gs = fig.add_gridspec(1, 3, wspace=0.12)

    panels = [("recorded demonstration", None, TRUTH),
              (f"B-spline policy\n{len(d['roll_starts'])} predictions x 16 waypoints",
               d["roll_paths"], PRED)]
    if "base_roll_chunks" in d:
        panels.append((f"ACT chunk-100 baseline\n{len(d['roll_starts'])} predictions x "
                       f"{int(d['base_n_steps'][0])} waypoints", d["base_roll_chunks"], BASE))

    for i, (title, chunks, color) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i], projection="3d")
        ax.plot(*ep[:, :3].T, color=GRID if chunks is not None else TRUTH,
                lw=1.6 if chunks is not None else 2.2)
        if chunks is not None:
            for c in chunks:
                ax.plot(*c[:, :3].T, color=color, lw=1.0, alpha=0.55)
        set_3d_style(ax, ep[:, :3])
        ax.set_title(title, loc="left", fontsize=8.8, linespacing=1.4)
    fig.suptitle(f"episode {meta['episode']} - every prediction along the episode, "
                 f"replanned every {meta['replan_frames'] / fps:.1f} s",
                 x=0.012, ha="left", fontsize=12, fontweight="bold")
    out = figdir / "rollout_3d.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out.name


def error_figure(d, meta, figdir):
    fps = meta["fps"]
    pos = 1000 * d["score_pos"]
    rot = np.degrees(d["score_rot"])
    spans = d["score_spans"] / fps

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.4))
    for ax, vals, title, unit, color in [
        (axes[0], pos, "position error vs recording", "mm", TRUTH),
        (axes[1], rot, "rotation error vs recording", "deg", TRUTH),
        (axes[2], spans, "trajectory covered by one prediction", "s", PRED),
    ]:
        ax.hist(vals, bins=24, color=color, alpha=0.85, edgecolor=SURFACE, linewidth=0.8)
        p50, p90 = np.percentile(vals, [50, 90])
        for p, ls, lab in [(p50, "-", "p50"), (p90, ":", "p90")]:
            ax.axvline(p, color=INK, lw=1.2, ls=ls)
            ax.annotate(f"{lab} {p:.1f}", (p, ax.get_ylim()[1]), xytext=(3, -10),
                        textcoords="offset points", fontsize=7.6, color=INK)
        ax.set_title(title, loc="left")
        ax.set_xlabel(unit)
        ax.set_ylabel("frames")
        ax.grid(True, axis="y", alpha=0.7)
    fig.suptitle(f"{meta['scored']} randomly drawn observations from "
                 f"{meta['dataset_episodes']} episodes of the whole dataset",
                 x=0.012, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = figdir / "errors.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out.name, dict(
        pos_p50=float(np.percentile(pos, 50)), pos_p90=float(np.percentile(pos, 90)),
        pos_max=float(pos.max()),
        rot_p50=float(np.percentile(rot, 50)), rot_p90=float(np.percentile(rot, 90)),
        rot_max=float(rot.max()),
        span_p50=float(np.percentile(spans, 50)), span_p90=float(np.percentile(spans, 90)),
    )


def horizon_figure(d, meta, figdir):
    """Does the predicted horizon really stretch where the demo moves slowly?"""
    fps = meta["fps"]
    ep = d["ep_raw"]
    starts = d["roll_starts"]
    spans = d["roll_spans"] / fps
    step = np.linalg.norm(np.diff(ep[:, :3], axis=0), axis=1) * fps  # m/s per frame
    speed = np.array([step[max(0, s - 5):min(len(step), s + 5)].mean() for s in starts])

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 3.6))
    ax = axes[0]
    ax.plot(starts / fps, spans, color=PRED, lw=2, marker="o", ms=4)
    ax.axhline(int(d["base_n_steps"][0]) / fps if "base_n_steps" in d else 5.0,
               color=BASE, lw=2, ls="--")
    ax.annotate("ACT chunk-100: fixed 5.0 s", (0.02, int(d["base_n_steps"][0]) / fps),
                xytext=(2, -12), textcoords="offset points", color=BASE, fontsize=8)
    ax.set_xlabel("time into episode [s]")
    ax.set_ylabel("horizon [s]")
    ax.set_ylim(0, 5.6)
    ax.set_title("how far ahead each prediction reaches", loc="left")
    ax.grid(True, alpha=0.7)

    ax = axes[1]
    ax.scatter(speed, spans, color=PRED, s=26, alpha=0.85, edgecolor=SURFACE, linewidth=0.6)
    if len(speed) > 2:
        r = np.corrcoef(speed, spans)[0, 1]
        k, b = np.polyfit(speed, spans, 1)
        xs = np.linspace(speed.min(), speed.max(), 20)
        ax.plot(xs, k * xs + b, color=INK, lw=1.2, ls=":")
        ax.annotate(f"r = {r:+.2f}", (0.98, 0.94), xycoords="axes fraction",
                    ha="right", fontsize=9, color=INK)
    ax.set_xlabel("demonstrated speed at the observation [m/s]")
    ax.set_ylabel("horizon [s]")
    ax.set_title("adaptive knot spacing: the faster the demo moves, the further\none prediction reaches", loc="left", linespacing=1.4)
    ax.grid(True, alpha=0.7)
    fig.tight_layout()
    out = figdir / "horizon.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out.name, float(np.corrcoef(speed, spans)[0, 1]) if len(speed) > 2 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump-dir", type=Path, default=Path("outputs/inference_report"))
    args = ap.parse_args()

    d = np.load(args.dump_dir / "dump.npz")
    meta = json.loads((args.dump_dir / "meta.json").read_text())
    meta["dataset_episodes"] = json.loads(
        (Path(meta["dataset_root"]) / "meta" / "info.json").read_text())["total_episodes"]
    figdir = args.dump_dir / "figs"
    figdir.mkdir(exist_ok=True)

    snaps = [snapshot_figure(k, d, meta, figdir, args.dump_dir / "frames")
             for k in range(len(d["snap_frame"]))]
    roll = rollout_figure(d, meta, figdir)
    errs, err_stats = error_figure(d, meta, figdir)
    hor, corr = horizon_figure(d, meta, figdir)

    summary = dict(meta=meta, snapshots=snaps, figures=dict(
        rollout=roll, errors=errs, horizon=hor), errors=err_stats,
        horizon_speed_corr=corr)
    (args.dump_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "meta"}, indent=2)[:2000])
    print(f"\nfigures in {figdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
