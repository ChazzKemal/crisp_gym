"""End-to-end check of a converted dataset on disk.

Skipped unless ``convert_lerobot_to_bspline.py`` has been run. This is the test
that actually proves the conversion is right: it reads a frame's action out of
the *converted* parquet, decodes it, and compares against the recorded future
actions of the *source* dataset.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from convert_lerobot_to_bspline import bspline_feature_names
from decode_rollout import decode
from lerobot_bridge import load_lerobot_actions

CONVERTED = Path("/home/batur/Coding/data/merged_bspline_20260528")


@pytest.fixture(scope="module")
def converted():
    sidecar = CONVERTED / "meta" / "bspline.json"
    if not sidecar.exists():
        pytest.skip(f"no converted dataset at {CONVERTED}; run convert_lerobot_to_bspline.py")
    meta = json.loads(sidecar.read_text())
    info = json.loads((CONVERTED / "meta" / "info.json").read_text())
    return meta, info, load_lerobot_actions(CONVERTED)


@pytest.fixture(scope="module")
def source(converted):
    meta, _, _ = converted
    return load_lerobot_actions(meta["source_dataset"])


def test_frame_and_episode_counts_are_preserved(converted, source):
    _, info, conv = converted
    assert len(conv.actions) == len(source.actions)
    assert np.array_equal(conv.episode_ends, source.episode_ends)
    assert info["total_frames"] == len(source.actions)


def test_action_feature_declares_the_flat_parameter_matrix(converted):
    meta, info, conv = converted
    feat = info["features"]["action"]
    assert feat["shape"] == [meta["flat_action_dim"]]
    assert feat["shape"][0] == meta["n_action_steps"] * meta["n_action_channels"]
    assert conv.actions.shape[1] == meta["flat_action_dim"]
    assert feat["names"] == bspline_feature_names(meta["n_action_steps"])


def test_observation_features_are_untouched(converted, source):
    """Only the action column is rewritten; videos are symlinked, not re-encoded."""
    meta, info, _ = converted
    src_info = json.loads((Path(meta["source_dataset"]) / "meta" / "info.json").read_text())
    for key, feat in src_info["features"].items():
        if key == "action":
            continue
        assert info["features"][key] == feat, f"{key} changed"
    assert (CONVERTED / "videos").is_symlink()
    assert (CONVERTED / "videos").resolve() == (Path(meta["source_dataset"]) / "videos").resolve()


def test_stats_share_one_scale_per_channel(converted):
    """Upstream collapses the step axis when normalising; every row of a column
    must therefore carry identical stats."""
    meta, _, _ = converted
    stats = json.loads((CONVERTED / "meta" / "stats.json").read_text())["action"]
    n_steps, n_ch = meta["n_action_steps"], meta["n_action_channels"]
    for key in ("min", "max", "mean", "std"):
        grid = np.asarray(stats[key]).reshape(n_steps, n_ch)
        assert np.allclose(grid, grid[0][None, :]), f"{key} varies across rows"
    assert (np.asarray(stats["std"]) > 0).all()


def test_knot_column_is_monotone_everywhere(converted):
    meta, _, conv = converted
    grid = conv.actions.reshape(len(conv.actions), meta["n_action_steps"], meta["n_action_channels"])
    assert (np.diff(grid[:, :, 0], axis=1) >= -1e-4).all()


def test_decoded_actions_match_the_source_recording(converted, source):
    """The claim that matters. Decode converted frames, compare to the source."""
    meta, _, conv = converted
    fps = meta["fps"]
    rng = np.random.default_rng(0)
    starts, ends = source.episode_starts, source.episode_ends

    worst_pos, worst_rot, worst_grip, n = 0.0, 0.0, 0.0, 0
    for ep in rng.choice(len(ends), size=min(20, len(ends)), replace=False):
        a, b = int(starts[ep]), int(ends[ep])
        ep_raw = source.actions[a:b]
        frames = np.arange(len(ep_raw))
        for local in rng.choice(max(b - a - 200, 1), size=min(10, max(b - a - 200, 1)), replace=False):
            local = int(local)
            out = decode(conv.actions[a + local], chunk_size=meta["chunk_size"],
                         degree=meta["degree"], num_actions=16, fps=fps,
                         relative_knots=meta["relative_knots"],
                         n_action_channels=meta["n_action_channels"])
            abs_frames = local + out.times * fps
            if abs_frames[-1] >= len(ep_raw) - 1:
                continue
            truth = np.stack([np.interp(abs_frames, frames, ep_raw[:, d]) for d in range(7)], axis=1)
            worst_pos = max(worst_pos, np.abs(out.actions[:, :3] - truth[:, :3]).max())
            diff = (Rotation.from_rotvec(out.actions[:, 3:6])
                    * Rotation.from_rotvec(truth[:, 3:6]).inv()).magnitude()
            worst_rot = max(worst_rot, diff.max())
            worst_grip = max(worst_grip, np.abs(out.actions[:, 6] - truth[:, 6]).max())
            n += 1

    assert n > 50, f"only {n} samples checked"
    # max_error bounds the fit; linear interpolation of the reference adds a little.
    assert worst_pos < 0.03, f"position {worst_pos * 1000:.1f} mm"
    assert worst_rot < 0.10, f"rotation {worst_rot:.4f} rad"
    assert worst_grip < 0.35, f"gripper {worst_grip:.3f}"
    print(f"\n  checked {n} chunks: pos<={worst_pos * 1000:.1f}mm "
          f"rot<={np.degrees(worst_rot):.2f}deg grip<={worst_grip:.3f}")


def test_chunk_horizon_is_useful(converted):
    """A chunk must reach meaningfully into the future, or the representation
    buys nothing over predicting raw actions."""
    meta, _, conv = converted
    grid = conv.actions.reshape(len(conv.actions), meta["n_action_steps"], meta["n_action_channels"])
    span = grid[:, meta["n_action_steps"] - meta["degree"] - 1, 0] - grid[:, meta["degree"], 0]
    span = span[span > 0]
    p50 = float(np.percentile(span, 50))
    assert p50 > meta["chunk_size"], f"median span {p50:.1f} frames <= chunk_size"
    print(f"\n  median chunk span {p50:.1f} frames = {p50 / meta['fps']:.2f} s "
          f"from {meta['flat_action_dim']} predicted numbers")
