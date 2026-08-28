"""Deployment-side decoding: reshape, repair, decode, convert back to cart7."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from bspline_core.bspline_action import chunk_bspline_trajectory, chunk_to_params
from bspline_core.chunk_sampler import BSplineChunkSampler
from bspline_core.rotation import convert_actions_7d_to_10d
from conftest import smooth_trajectory
from decode_rollout import decode, is_padded, reshape_prediction

DEGREE = 3
CHUNK_SIZE = 10
STEPS = CHUNK_SIZE + 2 * DEGREE
CHANNELS = 11
FPS = 20.0


def _cart7_episode(n=300, seed=0):
    """A plausible cart7 trajectory: smooth pose + binary gripper."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    xyz = np.stack([1.0 + 0.2 * np.sin(2 * np.pi * t),
                    0.15 + 0.1 * np.cos(2 * np.pi * t),
                    0.6 + 0.15 * np.sin(4 * np.pi * t)], axis=1)
    base = rng.normal(size=3)
    rotvec = base[None, :] + 0.15 * np.stack([np.sin(2 * np.pi * t)] * 3, axis=1)
    grip = (t > 0.4).astype(np.float64)
    return np.concatenate([xyz, rotvec, grip[:, None]], axis=1).astype(np.float32)


def _sampler(raw):
    return BSplineChunkSampler(
        actions=convert_actions_7d_to_10d(raw), episode_ends=np.array([len(raw)]),
        chunk_size=CHUNK_SIZE, degree=DEGREE, max_error=0.01, stride=1, max_first_k=1,
    )


def test_reshape_round_trips():
    p = np.arange(STEPS * CHANNELS, dtype=np.float64).reshape(STEPS, CHANNELS)
    assert np.array_equal(reshape_prediction(p.reshape(-1), STEPS, CHANNELS), p)


def test_reshape_rejects_wrong_size():
    with pytest.raises(ValueError, match="expected"):
        reshape_prediction(np.zeros(17), STEPS, CHANNELS)


def test_decode_returns_cart7_waypoints():
    raw = _cart7_episode()
    out = decode(_sampler(raw).chunk_for_timestep(20).reshape(-1),
                 chunk_size=CHUNK_SIZE, degree=DEGREE, num_actions=12, fps=FPS)
    assert out.actions.shape == (12, 7)
    assert out.times.shape == (12,)
    assert np.isfinite(out.actions).all()
    assert (np.diff(out.times) > 0).all()


def test_decoded_waypoints_track_the_source_trajectory():
    """The whole pipeline: cart7 -> rot6d -> spline -> chunk -> decode -> cart7."""
    raw = _cart7_episode()
    s = _sampler(raw)
    frames = np.arange(len(raw))
    worst_pos, worst_rot = 0.0, 0.0
    for ts in range(10, 200, 10):
        out = decode(s.chunk_for_timestep(ts).reshape(-1), chunk_size=CHUNK_SIZE,
                     degree=DEGREE, num_actions=12, fps=FPS)
        abs_frames = ts + out.times * FPS
        if abs_frames[-1] >= len(raw) - 1:
            continue
        truth = np.stack([np.interp(abs_frames, frames, raw[:, d]) for d in range(7)], axis=1)
        worst_pos = max(worst_pos, np.abs(out.actions[:, :3] - truth[:, :3]).max())
        diff = (Rotation.from_rotvec(out.actions[:, 3:6])
                * Rotation.from_rotvec(truth[:, 3:6]).inv()).magnitude()
        worst_rot = max(worst_rot, diff.max())
    assert worst_pos < 0.02, f"position error {worst_pos:.4f} m"
    assert worst_rot < 0.05, f"rotation error {worst_rot:.4f} rad"


def test_speedup_rescales_time_but_not_geometry():
    """The acceleration mechanism: same waypoints, compressed schedule."""
    raw = _cart7_episode()
    flat = _sampler(raw).chunk_for_timestep(20).reshape(-1)
    slow = decode(flat, chunk_size=CHUNK_SIZE, degree=DEGREE, num_actions=12, fps=FPS)
    fast = decode(flat, chunk_size=CHUNK_SIZE, degree=DEGREE, num_actions=12, fps=FPS, speedup=3.0)
    assert np.allclose(slow.actions, fast.actions, atol=1e-6)
    assert np.allclose(slow.times, 3.0 * fast.times, atol=1e-9)


def test_non_monotone_knots_are_repaired_not_raised():
    """A network can emit a knot column that goes backwards; decoding must cope."""
    raw = _cart7_episode()
    p = _sampler(raw).chunk_for_timestep(20)
    p[5, 0], p[6, 0] = p[6, 0], p[5, 0]  # swap two knots
    out = decode(p.reshape(-1), chunk_size=CHUNK_SIZE, degree=DEGREE, num_actions=8, fps=FPS)
    assert np.isfinite(out.actions).all()


def test_padded_chunk_drops_its_zero_sample():
    raw = _cart7_episode(160)
    s = _sampler(raw)
    last_ts = int(s.valid_timesteps[-1])
    p = s.chunk_for_timestep(last_ts)
    if not is_padded(p, DEGREE):
        pytest.skip("final chunk of this fixture is not tail-padded")
    out = decode(p.reshape(-1), chunk_size=CHUNK_SIZE, degree=DEGREE, num_actions=8, fps=FPS)
    assert out.padded
    assert out.actions.shape == (8, 7)
    # a zero row would decode to the origin with an identity-ish rotation
    assert np.linalg.norm(out.actions[-1, :3]) > 0.1


def test_times_are_clamped_to_non_negative():
    """Trailing frames reuse a chunk whose domain starts in the past."""
    raw = _cart7_episode()
    s = _sampler(raw)
    for ts in s.valid_timesteps[-5:]:
        out = decode(s.chunk_for_timestep(int(ts)).reshape(-1), chunk_size=CHUNK_SIZE,
                     degree=DEGREE, num_actions=8, fps=FPS)
        assert (out.times >= 0).all()


def test_span_is_reported_in_frames():
    """``times`` is measured from *now*, so the span is the difference of its
    ends -- not its last value. ``times[0] > 0`` is expected: a chunk's valid
    domain generally starts slightly after the observation frame."""
    raw = _cart7_episode()
    out = decode(_sampler(raw).chunk_for_timestep(20).reshape(-1),
                 chunk_size=CHUNK_SIZE, degree=DEGREE, num_actions=8, fps=FPS)
    assert out.span_frames > 0
    assert out.times[0] >= 0
    assert np.isclose((out.times[-1] - out.times[0]) * FPS, out.span_frames, atol=1e-6)
