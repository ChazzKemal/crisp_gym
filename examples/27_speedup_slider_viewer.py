#!/usr/bin/env python3
"""Interactive viewer for the trajectory-aware speed schedule (no ROS, no robot).

Loads one episode from a LeRobot v3 dataset, computes the speed schedule via the
same helpers ``17_replay_dataset.py`` uses on the real robot, and exposes live
sliders for the four headline knobs:

    --max-speed   --min-speed   --clamp-deg   --lookahead

Dragging any slider re-runs ``compute_speed_schedule`` + ``build_speed_queue_arrays``
(both pure numpy, microseconds per call) and redraws:

    - speed-schedule plot (s_raw dashed + cycle-snapped s_eff solid)
    - 3-D trajectory recolored by s_eff
    - stats line (peak/floor/mean s_eff, replay duration, distinct s_eff levels)

A frame slider + Play scrubs through the trajectory; the marker on the speed plot
and the red dot on the 3-D view follow.

The "Print CLI args" button dumps the equivalent ``17_replay_dataset.py`` flags to
stdout so you can paste them into the real replay once you've dialed in good
values.

Backend
-------
Defaults to ``MPLBACKEND=WebAgg`` so SSH'd users get a browser UI without needing
an X server on the laptop side. Export ``MPLBACKEND=TkAgg`` before launching to
use an ssh -X forwarded native window instead.

Usage
-----
On the workstation:

    cd Yunfei/crisp_gym
    pixi run -e jazzy-lerobot python examples/27_speedup_slider_viewer.py \\
        --repo-id camera_test --episode-idx 0

Matplotlib prints ``http://127.0.0.1:8988/`` once it boots. From your laptop, in a
separate terminal:

    ssh -N -L 8988:localhost:8988 <user>@<workstation>

then open ``http://localhost:8988/`` in your browser.
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

# Pick the backend BEFORE matplotlib is imported. WebAgg works from any browser
# via SSH port-forward and supports Slider/Button widgets natively. Respect an
# explicit MPLBACKEND so ssh -X + TkAgg users get the native window they asked
# for.
os.environ.setdefault("MPLBACKEND", "WebAgg")

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
    parser.add_argument(
        "--action-stride", type=int, default=1,
        help="Subsample every Nth recorded action (matches 17_replay_dataset.py "
             "--action-stride). Baked at startup, not a live slider.",
    )
    parser.add_argument("--initial-max-speed", type=float, default=4.0)
    parser.add_argument("--initial-min-speed", type=float, default=1.0)
    parser.add_argument("--initial-clamp-deg", type=float, default=5.0)
    parser.add_argument("--initial-lookahead", type=int, default=0)
    parser.add_argument(
        "--initial-drop-holds", action="store_true",
        help="Start with the drop-holds toggle checked. Held frames "
             "(zero-motion stalls) inherit the speed of the next moving "
             "frame instead of injecting spurious 90 deg angles. Toggleable "
             "live via the checkbox.",
    )
    parser.add_argument(
        "--hold-eps", type=float, default=1e-6,
        help="Minimum per-step position delta (m) below which a frame is "
             "considered 'held' when --initial-drop-holds is checked.",
    )
    args = parser.parse_args()

    # Import matplotlib AFTER setting MPLBACKEND.
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button, CheckButtons

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

    fps = float(info.get("fps", 30))
    stride = max(1, int(args.action_stride))
    if stride > 1:
        df = df.iloc[::stride].reset_index(drop=True)
    n_frames = len(df)
    fps_eff = fps / stride
    # dt_base for adaptive (--scale-kp + --max-speed) mode in 17_replay_dataset.py
    # is 1/fps_eff (line 1973-1976). Match that here so cycle-snap stats line up.
    dt_base = 1.0 / max(fps_eff, 1e-9)

    actions = np.stack(
        [np.asarray(a, dtype=np.float64) for a in df["action"].to_numpy()],
        axis=0,
    )
    if actions.shape[1] < 6:
        print(
            f"Episode action_dim={actions.shape[1]} < 6; the adaptive schedule "
            "needs xyz + rpy. Aborting.",
            file=sys.stderr,
        )
        sys.exit(1)

    t_axis = np.arange(n_frames) / fps_eff
    xs = actions[:, 0].astype(np.float64)
    ys = actions[:, 1].astype(np.float64)
    zs = actions[:, 2].astype(np.float64)

    def recompute(max_speed, min_speed, clamp_deg, lookahead, drop_holds):
        if drop_holds:
            sch = rep.compute_speed_schedule_drop_holds(
                actions[:, :6],
                max_speed=float(max_speed),
                min_speed=float(min_speed),
                clamp_deg=float(clamp_deg),
                n_lookahead=int(lookahead),
                motion_eps=float(args.hold_eps),
            )
        else:
            sch = rep.compute_speed_schedule(
                actions[:, :6],
                max_speed=float(max_speed),
                min_speed=float(min_speed),
                clamp_deg=float(clamp_deg),
                n_lookahead=int(lookahead),
            )
        cycles, dt_eff, s_eff = rep.build_speed_queue_arrays(
            sch, dt_base, n_frames, retime=True,
        )
        return sch, cycles, dt_eff, s_eff

    sch0, cyc0, dte0, seff0 = recompute(
        args.initial_max_speed, args.initial_min_speed,
        args.initial_clamp_deg, args.initial_lookahead,
        args.initial_drop_holds,
    )

    fig = plt.figure(figsize=(15, 8))
    try:
        fig.canvas.manager.set_window_title(
            f"speedup viewer — {args.repo_id} ep {args.episode_idx}"
        )
    except Exception:
        pass

    gs = fig.add_gridspec(
        2, 2, height_ratios=[3, 1],
        left=0.06, right=0.96, top=0.94, bottom=0.32,
        hspace=0.35, wspace=0.20,
    )
    ax_speed = fig.add_subplot(gs[0, 0])
    ax_3d = fig.add_subplot(gs[0, 1], projection="3d")
    ax_stats = fig.add_subplot(gs[1, :])
    ax_stats.set_axis_off()

    # Speed schedule plot.
    line_raw, = ax_speed.plot(
        t_axis, sch0, color="tab:gray", linewidth=1.0, linestyle="--",
        label="s_raw (continuous)",
    )
    line_eff, = ax_speed.plot(
        t_axis, seff0, color="tab:blue", linewidth=1.3,
        label="s_eff (cycle-snapped)",
    )
    speed_marker = ax_speed.axvline(t_axis[0], color="red", linewidth=1.0, alpha=0.7)
    ax_speed.set_xlabel("time (s)")
    ax_speed.set_ylabel("speed factor")
    ax_speed.set_title("Speed schedule")
    ax_speed.set_xlim(t_axis[0], t_axis[-1] if n_frames > 1 else 1.0)
    ax_speed.grid(True, alpha=0.3)
    ax_speed.legend(loc="upper right", fontsize=8)

    # 3-D trajectory.
    ax_3d.plot(xs, ys, zs, color="gray", alpha=0.3, linewidth=0.6)
    sc = ax_3d.scatter(
        xs, ys, zs, c=seff0, cmap="viridis", s=6,
        vmin=float(seff0.min()), vmax=float(seff0.max()),
    )
    cur_dot, = ax_3d.plot([xs[0]], [ys[0]], [zs[0]], "o", color="red", markersize=10)
    ax_3d.set_xlabel("x (m)")
    ax_3d.set_ylabel("y (m)")
    ax_3d.set_zlabel("z (m)")
    ax_3d.set_title("Trajectory (color = s_eff)")
    try:
        fig.colorbar(sc, ax=ax_3d, fraction=0.04, pad=0.02, shrink=0.6)
    except Exception:
        pass

    stats_text = ax_stats.text(
        0.01, 0.5, "", fontsize=10, family="monospace", va="center",
        transform=ax_stats.transAxes,
    )

    def _autorange_speed():
        raw = line_raw.get_ydata()
        eff = line_eff.get_ydata()
        lo = float(min(raw.min(), eff.min(), 0.95))
        hi = float(max(raw.max(), eff.max(), 1.05))
        pad = 0.05 * max(hi - lo, 0.1)
        ax_speed.set_ylim(lo - pad, hi + pad)

    def _refresh_stats(cycles, dt_eff, s_eff):
        # Distinct cycle-snapped s_eff levels (rounded for noise tolerance). This
        # is an approximate RPC ceiling: the producer publishes once whenever
        # s_eff transitions between consecutive frames.
        n_levels = int(np.unique(np.round(s_eff, 3)).size)
        replay_dur = float(np.sum(dt_eff))
        stats_text.set_text(
            f"frames: {n_frames:5d}    "
            f"replay duration: {replay_dur:6.2f} s    "
            f"s_eff: peak {s_eff.max():.2f}  floor {s_eff.min():.2f}  "
            f"mean {s_eff.mean():.2f}    "
            f"cycles: min {int(cycles.min())}  max {int(cycles.max())}    "
            f"distinct s_eff levels: {n_levels}"
        )

    _refresh_stats(cyc0, dte0, seff0)
    _autorange_speed()

    # Sliders.
    s_axes = {
        "max_speed": fig.add_axes((0.08, 0.22, 0.55, 0.018)),
        "min_speed": fig.add_axes((0.08, 0.19, 0.55, 0.018)),
        "clamp_deg": fig.add_axes((0.08, 0.16, 0.55, 0.018)),
        "lookahead": fig.add_axes((0.08, 0.13, 0.55, 0.018)),
        "frame":     fig.add_axes((0.08, 0.09, 0.55, 0.018)),
    }
    sliders = {
        "max_speed": Slider(s_axes["max_speed"], "max_speed", 1.0, 6.0,
                            valinit=args.initial_max_speed, valstep=0.1),
        "min_speed": Slider(s_axes["min_speed"], "min_speed", 0.5, 2.0,
                            valinit=args.initial_min_speed, valstep=0.05),
        "clamp_deg": Slider(s_axes["clamp_deg"], "clamp_deg", 1.0, 30.0,
                            valinit=args.initial_clamp_deg, valstep=0.5),
        "lookahead": Slider(s_axes["lookahead"], "lookahead", 0, 30,
                            valinit=args.initial_lookahead, valstep=1, valfmt="%d"),
        "frame":     Slider(s_axes["frame"], "frame", 0, max(n_frames - 1, 0),
                            valinit=0, valstep=1, valfmt="%d"),
    }

    # Buttons.
    ax_play  = fig.add_axes((0.68, 0.20, 0.07, 0.04))
    ax_reset = fig.add_axes((0.76, 0.20, 0.07, 0.04))
    ax_print = fig.add_axes((0.84, 0.20, 0.12, 0.04))
    btn_play  = Button(ax_play, "Play")
    btn_reset = Button(ax_reset, "Reset")
    btn_print = Button(ax_print, "Print CLI args")

    # Drop-holds checkbox (toggles between compute_speed_schedule and
    # compute_speed_schedule_drop_holds in the recompute path).
    ax_check = fig.add_axes((0.68, 0.09, 0.13, 0.08))
    ax_check.set_axis_off()
    check_drop_holds = CheckButtons(
        ax_check, labels=["drop holds"], actives=[bool(args.initial_drop_holds)],
    )

    state = {"playing": False, "timer": None}

    def _drop_holds_active() -> bool:
        return bool(check_drop_holds.get_status()[0])

    def _recompute_and_redraw(_val=None):
        # Enforce min_speed <= max_speed. set_val will retrigger us with the
        # clamped value; the second pass falls through normally.
        if sliders["min_speed"].val > sliders["max_speed"].val:
            sliders["min_speed"].set_val(sliders["max_speed"].val)
            return
        sch, cycles, dt_eff, s_eff = recompute(
            sliders["max_speed"].val,
            sliders["min_speed"].val,
            sliders["clamp_deg"].val,
            int(sliders["lookahead"].val),
            _drop_holds_active(),
        )
        line_raw.set_ydata(sch)
        line_eff.set_ydata(s_eff)
        sc.set_array(s_eff)
        sc.set_clim(float(s_eff.min()), float(s_eff.max()))
        _refresh_stats(cycles, dt_eff, s_eff)
        _autorange_speed()
        fig.canvas.draw_idle()

    def _on_frame(val):
        i = int(np.clip(int(val), 0, n_frames - 1))
        speed_marker.set_xdata([t_axis[i], t_axis[i]])
        cur_dot.set_data([xs[i]], [ys[i]])
        cur_dot.set_3d_properties([zs[i]])
        fig.canvas.draw_idle()

    sliders["max_speed"].on_changed(_recompute_and_redraw)
    sliders["min_speed"].on_changed(_recompute_and_redraw)
    sliders["clamp_deg"].on_changed(_recompute_and_redraw)
    sliders["lookahead"].on_changed(_recompute_and_redraw)
    sliders["frame"].on_changed(_on_frame)
    check_drop_holds.on_clicked(lambda _label: _recompute_and_redraw())

    def _tick():
        if not state["playing"]:
            return
        cur = int(sliders["frame"].val) + 1
        if cur >= n_frames - 1:
            state["playing"] = False
            btn_play.label.set_text("Play")
            if state["timer"] is not None:
                state["timer"].stop()
            sliders["frame"].set_val(n_frames - 1)
            return
        sliders["frame"].set_val(cur)

    def _toggle_play(_event):
        state["playing"] = not state["playing"]
        btn_play.label.set_text("Pause" if state["playing"] else "Play")
        if state["playing"]:
            if state["timer"] is None:
                interval_ms = max(int(round(1000.0 / max(fps_eff, 1.0))), 16)
                state["timer"] = fig.canvas.new_timer(interval=interval_ms)
                state["timer"].add_callback(_tick)
            state["timer"].start()
        elif state["timer"] is not None:
            state["timer"].stop()

    btn_play.on_clicked(_toggle_play)

    def _reset(_event):
        sliders["max_speed"].set_val(args.initial_max_speed)
        sliders["min_speed"].set_val(args.initial_min_speed)
        sliders["clamp_deg"].set_val(args.initial_clamp_deg)
        sliders["lookahead"].set_val(args.initial_lookahead)
        sliders["frame"].set_val(0)
        if _drop_holds_active() != bool(args.initial_drop_holds):
            check_drop_holds.set_active(0)  # toggles label 0 in/out

    btn_reset.on_clicked(_reset)

    def _print_cli(_event):
        parts = [
            "--scale-kp",
            f"--max-speed {sliders['max_speed'].val:.2f}",
            f"--min-speed {sliders['min_speed'].val:.2f}",
            f"--clamp-deg {sliders['clamp_deg'].val:.1f}",
            f"--lookahead {int(sliders['lookahead'].val)}",
        ]
        if _drop_holds_active():
            parts.append("--drop-holds")
            if abs(args.hold_eps - 1e-6) > 1e-12:
                parts.append(f"--hold-eps {args.hold_eps:g}")
        if stride > 1:
            parts.append(f"--action-stride {stride}")
        print("REPLAY ARGS:  " + " ".join(parts), flush=True)

    btn_print.on_clicked(_print_cli)

    _on_frame(0)

    backend = matplotlib.get_backend().lower()
    if backend == "agg":
        print(
            "MPLBACKEND=agg has no display. Export MPLBACKEND=WebAgg "
            "(browser) or TkAgg (ssh -X).",
            file=sys.stderr,
        )
        sys.exit(1)
    if "webagg" in backend:
        print("=" * 60)
        print("Speedup slider viewer (WebAgg).")
        print("  Matplotlib will print its URL on the next line.")
        print("  From a remote laptop, in a separate terminal:")
        print("    ssh -N -L 8988:localhost:8988 <user>@<workstation>")
        print("  Then open http://localhost:8988/ in your browser.")
        print("=" * 60, flush=True)

    plt.show()


if __name__ == "__main__":
    main()
