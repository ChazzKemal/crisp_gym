#!/usr/bin/env python3
"""Splice the figures and numbers into a self-contained report page.

Stage 3 of 3, after ``run_inference_dump.py`` and ``build_inference_report.py``.
Figures are embedded as data URIs, so the output file stands alone.

    /home/batur/miniconda3/bin/python bspline/build_inference_page.py
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Written after looking at each figure; the numbers in them come from summary.json.
SNAPSHOT_CAPTIONS = {
    5: "Both cups still on their stands, the arm high above the table. From this one frame the "
       "policy commits to the whole descent — {span} of motion — and tracks it to within "
       "{max} at the far end.",
    73: "Wrist camera centred on the green cup, gripper open around it. The prediction anticipates "
        "the lift: z leaves the table earlier than the recording does, which is the whole of the "
        "{max} maximum.",
    141: "Carrying the green cup across the table. The steadiest stretch of the episode and the "
         "steadiest tracking — {mean} mean over {span}.",
    210: "The transfer swing, and the longest horizon of the six at {span}. It is also the widest "
         "miss ({mean} mean): the prediction runs slightly low and inside the demonstrated arc.",
    278: "Gripper closed on the blue cup, a straight vertical lift. The tightest prediction here — "
         "{mean} mean, {rotmax} maximum rotation error.",
    347: "Carrying the blue cup into the final retreat. The predicted lift starts about a tenth of "
         "a second early; that offset is where the {max} maximum sits.",
}

ERR_NOTES = [
    "<b>The representation is not what is being measured.</b> Encoding and decoding a recorded "
    "trajectory through this B-spline chunking reproduces it to within 5.4 mm and 0.78&deg; "
    "(<code>tests/test_converted_dataset.py</code>). At a 27.6 mm median, what you see here is the "
    "policy's error, not the encoding's.",
    "<b>Deviation from the recording is not failure.</b> A policy that takes a different but "
    "perfectly good path to the same cup scores badly on this metric. It is a sanity check on "
    "whether the model learned the demonstrated motion — not a success rate.",
    "<b>24 of 150 predictions (16%) came back with a non-monotone knot column</b> and were repaired "
    "by <code>safer_knots()</code> before decoding. That is the failure mode this representation is "
    "most exposed to: a network is free to emit knots that go backwards in time, and nothing raises "
    "unless you count them.",
    "<b>6 of the 150 sampled frames were discarded</b> — their horizon ran past the end of the "
    "episode, so there is no recording left to compare against.",
]

CAVEATS = [
    "<b>Nothing ran on the robot.</b> Every prediction here was made against a recorded "
    "observation, open loop. There is no success rate, no contact, no closed-loop behaviour in "
    "this page.",
    "<b>The checkpoint is not converged.</b> Training stopped at 30 000 steps with the loss still "
    "drifting down (0.145 over the last thousand steps).",
    "<b>These observations are in-distribution.</b> Episode 58 and the 150 random frames all come "
    "from the training set — no held-out split was made — so the errors are a floor, not a "
    "generalisation estimate.",
    "<b>The ACT baseline is here for scale, not as a contest.</b> It predicts 100 waypoints over a "
    "fixed 5 s from 700 numbers, was fine-tuned on the same demonstrations, and is drawn only to "
    "show what the same observation buys under the standard representation.",
]

COMMANDS = (
    "<b># 1. run the checkpoint on recorded observations, dump predictions + ground truth</b>\n"
    "conda run -n lerobot-041 python bspline/run_inference_dump.py \\\n"
    "    --ckpt outputs/train/bspline_act_merged_20260528/checkpoints/last/pretrained_model \\\n"
    "    --episode 58 --snapshots 6 --replan-frames 10 --score-samples 150\n\n"
    "<b># 2. render the dump (needs matplotlib, which lerobot-041 does not have)</b>\n"
    "/home/batur/miniconda3/bin/python bspline/build_inference_report.py\n\n"
    "<b># 3. build this page</b>\n"
    "/home/batur/miniconda3/bin/python bspline/build_inference_page.py"
)


def data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump-dir", type=Path, default=Path("outputs/inference_report"))
    ap.add_argument("--template", type=Path, default=HERE / "inference_report_template.html")
    ap.add_argument("--out", type=Path, default=HERE / "inference_report.html")
    args = ap.parse_args()

    summary = json.loads((args.dump_dir / "summary.json").read_text())
    m, figs = summary["meta"], summary["figures"]
    figdir = args.dump_dir / "figs"

    snaps = []
    for s in summary["snapshots"]:
        tpl = SNAPSHOT_CAPTIONS.get(s["frame"], "{mean} mean position error over {span}.")
        snaps.append({**s, "caption": tpl.format(
            span=f"{s['span_s']:.2f} s", mean=f"{s['pos_mean']:.1f} mm",
            max=f"{s['pos_max']:.1f} mm", rotmax=f"{s['rot_max']:.2f}&deg;")})

    corr = summary["horizon_speed_corr"]
    report = {
        "meta": {
            # "checkpoints/last" is a symlink to the step directory; report the step
            "ckpt_step": Path(m["ckpt"]).parent.resolve().name,
            "dataset": Path(m["dataset_root"]).name,
            "dataset_episodes": m["dataset_episodes"],
            "episode": m["episode"], "episode_frames": m["episode_frames"], "fps": m["fps"],
            "num_actions": m["num_actions"], "replan_frames": m["replan_frames"],
            "scored": m["scored"], "score_n": m["score_n"],
            "nonmonotone_pct": round(100 * m["nonmonotone"] / m["score_n"]),
            "flat_action_dim": m["bspline"]["flat_action_dim"],
            "n_replans": len(range(0, m["episode_frames"] - 2, m["replan_frames"])),
            "base_n_steps": 100,
            "commands": COMMANDS,
        },
        "errors": summary["errors"],
        "figures": figs,
        "snapshots": snaps,
        "err_notes": ERR_NOTES,
        "caveats": CAVEATS,
        "horizon_intro": (
            f"A chunk is 20 knot intervals wide, so how much wall-clock time one prediction covers "
            f"is decided by how densely the fit had to place knots. Across episode "
            f"{m['episode']} that ranges from 0.4 s to 3.8 s, median "
            f"{summary['errors']['span_p50']:.2f} s. The ACT baseline's horizon is 5 s, every time, "
            f"whatever the arm is doing."),
        "horizon_caption": (
            f"<b>Horizon against demonstrated speed: r = {corr:+.2f}.</b> Where the demonstration "
            f"moves fast it is also smooth, knots are sparse, and one prediction reaches further; "
            f"where it slows down to line up a grasp, knots crowd and the horizon collapses to "
            f"under a second. That is the mechanism the whole speed-up argument rests on, visible "
            f"in a trained policy's own output rather than in the dataset."),
        "footer": (
            f"Generated by <code>run_inference_dump.py</code> &rarr; "
            f"<code>build_inference_report.py</code> &rarr; <code>build_inference_page.py</code>. "
            f"Checkpoint <code>{m['ckpt']}</code>, dataset <code>{m['dataset_root']}</code>."),
        "images": {name: data_uri(figdir / name)
                   for name in [figs["rollout"], figs["errors"], figs["horizon"]]
                   + [s["png"] for s in summary["snapshots"]]},
    }

    tpl = args.template.read_text()
    marker = "/*__REPORT__*/"
    if marker not in tpl:
        raise RuntimeError(f"{marker} not found in {args.template}")
    blob = json.dumps(report, separators=(",", ":"))
    html = tpl.replace(marker, blob)
    args.out.write_text(html)
    print(f"wrote {args.out}  ({len(html) / 1024 / 1024:.1f} MB, "
          f"{len(report['images'])} figures embedded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
