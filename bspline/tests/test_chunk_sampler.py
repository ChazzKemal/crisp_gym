"""Per-timestep chunk assignment: coverage, relative-time semantics, and the
end-to-end claim that a timestep's chunk describes that timestep's future."""

import numpy as np
import pytest

from bspline_core.bspline_action import decode_bspline_action
from bspline_core.chunk_sampler import BSplineChunkSampler
from bspline_core.knots import decode_relative_knots
from conftest import smooth_trajectory, steppy_trajectory

DEGREE = 3
CHUNK_SIZE = 10
STEPS = CHUNK_SIZE + 2 * DEGREE
MAX_ERROR = 0.002


def _multi_episode(lengths=(160, 220, 140), n_dims=10, seed=0):
    trajs = [smooth_trajectory(n, n_dims, seed=seed + i) for i, n in enumerate(lengths)]
    return np.concatenate(trajs, axis=0).astype(np.float32), np.cumsum(lengths)


def _sampler(max_first_k=2, relative_knots=False, **kw):
    actions, ends = _multi_episode(**kw)
    return (
        BSplineChunkSampler(
            actions=actions,
            episode_ends=ends,
            chunk_size=CHUNK_SIZE,
            degree=DEGREE,
            max_error=MAX_ERROR,
            stride=1,
            relative_knots=relative_knots,
            max_first_k=max_first_k,
        ),
        actions,
        ends,
    )


def test_chunk_shape_is_steps_by_channels():
    s, actions, _ = _sampler()
    assert s.all_actions.shape[1:] == (STEPS, 1 + actions.shape[1])
    assert s.n_action_steps == STEPS
    assert s.n_action_channels == 11


def test_every_timestep_up_to_the_reserved_tail_is_covered():
    """Upstream reserves ``max_first_k - 1`` frames at the end of each episode
    so the observation history window always fits."""
    for max_first_k in (1, 2, 4):
        s, _, ends = _sampler(max_first_k=max_first_k)
        starts = np.concatenate([[0], ends[:-1]])
        for ep, (a, b) in enumerate(zip(starts, ends)):
            ep_len = b - a
            covered = s.timestep_to_chunk[a:b] >= 0
            expected = ep_len - max_first_k + 1
            assert covered[:expected].all(), f"ep {ep} has a gap"
            assert not covered[expected:].any(), f"ep {ep} covers reserved tail"


def test_one_stored_chunk_per_covered_timestep():
    s, _, _ = _sampler()
    assert len(s.all_actions) == len(s.valid_timesteps)
    assert len(np.unique(s.timestep_to_chunk[s.valid_timesteps])) == len(s.valid_timesteps)


def test_knots_are_relative_to_the_current_frame():
    """The knot column is shifted by the current frame index, so the spline's
    valid domain is expressed as an offset from "now" and stays small."""
    s, _, _ = _sampler()
    first_valid = s.all_actions[:, DEGREE, 0]
    assert first_valid.max() < 200
    # for the overwhelming majority of frames the window starts in the future
    assert (first_valid >= -1e-5).mean() > 0.9


def test_trailing_frames_reuse_the_final_chunk_and_look_backwards():
    """Documented upstream behaviour.

    After the last fitted chunk, the sampler keeps assigning that same chunk to
    the remaining frames with an ever-growing shift, so its domain start goes
    negative -- the chunk begins *before* the current frame. This is confined
    to the last handful of frames of each episode. Callers that care (rollout
    evaluation, replay) should clamp ``t_min`` to 0.
    """
    s, _, ends = _sampler()
    starts = np.concatenate([[0], ends[:-1]])
    first_valid = s.all_actions[:, DEGREE, 0]
    for a, b in zip(starts, ends):
        covered = [t for t in range(a, b) if s.timestep_to_chunk[t] >= 0]
        neg = [t - a for t in covered if first_valid[s.timestep_to_chunk[t]] < 0]
        if not neg:
            continue
        # contiguous, and only at the very end of the episode
        assert neg == list(range(neg[0], neg[-1] + 1))
        assert neg[-1] == len(covered) - 1
        assert neg[0] > 0.85 * (b - a)


def test_knot_vectors_stay_monotone():
    s, _, _ = _sampler()
    assert (np.diff(s.all_actions[:, :, 0], axis=1) >= -1e-5).all()


def test_consecutive_timesteps_shift_knots_by_one():
    """Two neighbouring timesteps served by the same underlying chunk differ
    only by a one-frame translation of the knot column."""
    s, _, _ = _sampler()
    found = 0
    for i in range(len(s.valid_timesteps) - 1):
        a, b = s.all_actions[i], s.all_actions[i + 1]
        if not np.allclose(a[:, 1:], b[:, 1:]):
            continue  # different underlying chunk
        assert np.allclose(a[:, 0] - b[:, 0], 1.0, atol=1e-4)
        found += 1
    assert found > 50


def test_decoded_chunk_matches_the_future_trajectory():
    """The end-to-end claim. For each covered timestep i, decoding its chunk and
    mapping the samples back to absolute frames must reproduce the recorded
    actions there."""
    s, actions, ends = _sampler()
    starts = np.concatenate([[0], ends[:-1]])
    frames_of = {}
    for ep, (a, b) in enumerate(zip(starts, ends)):
        frames_of[ep] = (a, b)

    rng = np.random.default_rng(0)
    sample_ts = rng.choice(s.valid_timesteps, size=200, replace=False)
    worst = 0.0
    n_checked = 0
    for ts in sample_ts:
        ep = int(np.searchsorted(ends, ts, side="right"))
        a, b = frames_of[ep]
        local = ts - a
        p = s.chunk_for_timestep(ts)
        t_min, t_max = float(p[DEGREE, 0]), float(p[STEPS - DEGREE - 1, 0])
        if t_max <= t_min:
            continue
        # absolute frames covered by this chunk, clipped to the episode
        abs_lo, abs_hi = local + t_min, local + t_max
        if abs_hi >= (b - a) - 1:
            continue  # padded tail chunk -- covered by its own test
        decoded = decode_bspline_action(p, degree=DEGREE, num_actions=16)
        t_eval = np.linspace(abs_lo, abs_hi, 16)
        ep_actions = actions[a:b]
        truth = np.stack(
            [np.interp(t_eval, np.arange(len(ep_actions)), ep_actions[:, d])
             for d in range(ep_actions.shape[1])],
            axis=1,
        )
        worst = max(worst, np.abs(decoded - truth).max())
        n_checked += 1
    assert n_checked > 100
    assert worst < 20 * MAX_ERROR, worst


def test_relative_knot_mode_is_recoverable():
    s_abs, _, _ = _sampler(relative_knots=False)
    s_rel, _, _ = _sampler(relative_knots=True)
    assert s_abs.all_actions.shape == s_rel.all_actions.shape
    decoded = decode_relative_knots(s_rel.all_actions[:50].copy(), degree=DEGREE)
    assert np.allclose(decoded, s_abs.all_actions[:50], atol=1e-3)


def test_episode_mask_excludes_episodes():
    actions, ends = _multi_episode()
    mask = np.array([True, False, True])
    s = BSplineChunkSampler(
        actions=actions, episode_ends=ends, chunk_size=CHUNK_SIZE, degree=DEGREE,
        max_error=MAX_ERROR, stride=1, max_first_k=2, episode_mask=mask,
    )
    assert (s.timestep_to_chunk[ends[0]:ends[1]] < 0).all()
    assert (s.timestep_to_chunk[: ends[0] - 1] >= 0).all()
    assert len(s.fit_stats) == 2


def test_fit_stats_are_reported_per_episode():
    s, _, _ = _sampler()
    assert len(s.fit_stats) == 3
    for st in s.fit_stats:
        assert st.converged
        assert st.fit_error < MAX_ERROR
        assert 0 < st.compression_ratio < 1


def test_channel_stats_collapse_the_step_axis():
    """Upstream normalises per channel, not per (step, channel)."""
    s, _, _ = _sampler()
    ch = s.get_channel_stats()
    assert ch["min"].shape == (1, 1, 11)
    full = s.get_action_stats()
    assert np.isclose(ch["min"][0, 0, 0], full["min"][0, :, 0].min())
    assert np.isclose(ch["max"][0, 0, 5], full["max"][0, :, 5].max())


def test_binary_gripper_is_carried_through():
    """The gripper channel is fitted like any other dimension; make sure it
    still lands close to 0/1 after decoding."""
    traj = steppy_trajectory(300)
    s = BSplineChunkSampler(
        actions=traj.astype(np.float32), episode_ends=np.array([300]),
        chunk_size=CHUNK_SIZE, degree=DEGREE, max_error=MAX_ERROR, stride=1, max_first_k=2,
    )
    grip = decode_bspline_action(s.chunk_for_timestep(10), degree=DEGREE, num_actions=16)[:, -1]
    assert np.abs(grip - traj[10:26, -1]).max() < 0.2
