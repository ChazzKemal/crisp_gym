"""The core claim: a chunk is a *window* of the global spline, and decoding it
reproduces the fitted trajectory over that window.

If these pass, the representation is understood correctly:

* ``chunk_bspline_trajectory`` slices knots and control points with the same
  index range, which by the local-support property of B-splines yields a
  spline identical to the global one on ``[t[s+k], t[s+M-k-1]]``.
* ``decode_bspline_action`` takes ``params[:-(degree+1), 1:]`` as control
  points -- exactly the count the windowed knot vector requires.
* the knot column is in frame-index units, so subtracting the current frame
  index makes it relative without changing the decoded values.
"""

import numpy as np
import pytest
from scipy.interpolate import BSpline

from bspline_core.bspline_action import (
    ScipyBSplineCompression,
    chunk_bspline_trajectory,
    chunk_to_params,
    decode_bspline_action,
    extract_unique_knots,
)
from bspline_core.knots import encode_relative_knots
from conftest import smooth_trajectory, steppy_trajectory

DEGREE = 3
CHUNK_SIZE = 10
STEPS = CHUNK_SIZE + 2 * DEGREE
MAX_ERROR = 0.002


def _fit(traj, max_error=MAX_ERROR):
    comp = ScipyBSplineCompression(degree=DEGREE)
    comp.compress(traj, max_error=max_error)
    return comp


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------


def test_fit_reaches_requested_accuracy(smooth_traj):
    comp = _fit(smooth_traj)
    assert comp.converged
    assert comp.fit_error < MAX_ERROR
    assert np.abs(comp.spline(np.arange(len(smooth_traj))) - smooth_traj).max() < MAX_ERROR


def test_fit_compresses(smooth_traj):
    """Fewer knots than frames -- otherwise the representation buys nothing."""
    comp = _fit(smooth_traj)
    assert len(comp.knots) < len(smooth_traj)


def test_knot_vector_layout(smooth_traj):
    """FITPACK layout: len(t) == len(c) + k + 1, with k repeated boundary knots."""
    t_full, c_full, k = _fit(smooth_traj).spline.tck
    assert k == DEGREE
    assert len(t_full) == len(c_full) + DEGREE + 1
    assert len(extract_unique_knots(t_full, DEGREE)) == len(t_full) - 2 * DEGREE
    assert np.all(np.diff(t_full) >= 0)


# --------------------------------------------------------------------------
# chunking
# --------------------------------------------------------------------------


def test_chunk_shapes(smooth_traj):
    comp = _fit(smooth_traj)
    chunks = chunk_bspline_trajectory(comp, chunk_size=CHUNK_SIZE, stride=1)
    assert len(chunks) > 0
    for ch in chunks:
        assert ch["t"].shape == (STEPS,)
        assert ch["c"].shape == (STEPS, smooth_traj.shape[1])
        assert ch["k"] == DEGREE
        assert np.all(np.diff(ch["t"]) >= 0)


def test_params_matrix_shape(smooth_traj):
    comp = _fit(smooth_traj)
    ch = chunk_bspline_trajectory(comp, chunk_size=CHUNK_SIZE, stride=1)[0]
    p = chunk_to_params(ch, STEPS, 1 + smooth_traj.shape[1])
    assert p.shape == (STEPS, 1 + smooth_traj.shape[1])
    assert np.array_equal(p[:, 0], ch["t"].astype(np.float32))


def test_control_point_count_matches_knot_vector(smooth_traj):
    """decode uses params[:-(k+1)] control points; a knot vector of length M
    admits exactly M - k - 1 of them."""
    comp = _fit(smooth_traj)
    ch = chunk_bspline_trajectory(comp, chunk_size=CHUNK_SIZE, stride=1)[0]
    p = chunk_to_params(ch, STEPS, 1 + smooth_traj.shape[1])
    n_ctrl = p[: -(DEGREE + 1), 1:].shape[0]
    assert n_ctrl == STEPS - DEGREE - 1 == CHUNK_SIZE + DEGREE - 1


def test_chunk_is_a_window_of_the_global_spline():
    """THE key property. An interior chunk, evaluated on its own domain, must
    agree with the globally fitted spline to floating-point precision."""
    smooth_traj = smooth_trajectory(800, 10, seed=7)
    comp = _fit(smooth_traj)
    t_full, c_full, _ = comp.spline.tck
    chunks = chunk_bspline_trajectory(comp, chunk_size=CHUNK_SIZE, stride=1)

    checked = 0
    for s, ch in enumerate(chunks):
        if s + STEPS > len(c_full):  # tail chunks are padded; skip them here
            continue
        t_min, t_max = ch["t"][DEGREE], ch["t"][STEPS - DEGREE - 1]
        if t_max <= t_min:
            continue
        t_eval = np.linspace(t_min, t_max, 32, endpoint=False)
        local = BSpline(ch["t"], ch["c"][: -(DEGREE + 1)], DEGREE)(t_eval)
        assert np.allclose(local, comp.spline(t_eval), atol=1e-9), f"chunk {s} diverges"
        checked += 1
    assert checked >= 15, "not enough interior chunks to make the test meaningful"


def test_decode_matches_global_spline(smooth_traj):
    """Same property, through the public decode entrypoint."""
    comp = _fit(smooth_traj)
    c_full = comp.spline.tck[1]
    chunks = chunk_bspline_trajectory(comp, chunk_size=CHUNK_SIZE, stride=1)
    n_channels = 1 + smooth_traj.shape[1]

    for s, ch in enumerate(chunks):
        if s + STEPS > len(c_full):
            continue
        p = chunk_to_params(ch, STEPS, n_channels)
        t_min, t_max = float(p[DEGREE, 0]), float(p[STEPS - DEGREE - 1, 0])
        if t_max <= t_min:
            continue
        decoded = decode_bspline_action(p, degree=DEGREE, num_actions=16)
        expected = comp.spline(np.linspace(t_min, t_max, 16))
        assert np.allclose(decoded, expected, atol=1e-4)


def test_decode_reconstructs_the_original_trajectory(smooth_traj):
    """End to end: chunk -> decode -> compare against the *raw* data, not the fit."""
    comp = _fit(smooth_traj)
    c_full = comp.spline.tck[1]
    chunks = chunk_bspline_trajectory(comp, chunk_size=CHUNK_SIZE, stride=1)
    n_channels = 1 + smooth_traj.shape[1]
    frames = np.arange(len(smooth_traj))

    worst = 0.0
    for s, ch in enumerate(chunks):
        if s + STEPS > len(c_full):
            continue
        p = chunk_to_params(ch, STEPS, n_channels)
        t_min, t_max = float(p[DEGREE, 0]), float(p[STEPS - DEGREE - 1, 0])
        if t_max <= t_min or t_max > frames[-1]:
            continue
        t_eval = np.linspace(t_min, t_max, 24)
        decoded = decode_bspline_action(p, degree=DEGREE, num_actions=24)
        truth = np.stack(
            [np.interp(t_eval, frames, smooth_traj[:, d]) for d in range(smooth_traj.shape[1])],
            axis=1,
        )
        worst = max(worst, np.abs(decoded - truth).max())
    # fit error + linear-interpolation error of the reference; still tiny.
    assert worst < 20 * MAX_ERROR, worst


# --------------------------------------------------------------------------
# time semantics
# --------------------------------------------------------------------------


@pytest.mark.parametrize("offset", [0, 1, 5, 17])
def test_knot_shift_is_a_pure_time_translation(smooth_traj, offset):
    """Subtracting the current frame index from the knot column must not change
    what the spline *is* -- only where its origin sits."""
    comp = _fit(smooth_traj)
    ch = chunk_bspline_trajectory(comp, chunk_size=CHUNK_SIZE, stride=1)[4]
    p = chunk_to_params(ch, STEPS, 1 + smooth_traj.shape[1])
    shifted = p.copy()
    shifted[:, 0] -= offset
    assert np.allclose(
        decode_bspline_action(p, degree=DEGREE, num_actions=16),
        decode_bspline_action(shifted, degree=DEGREE, num_actions=16),
        atol=1e-5,
    )


def test_decoded_span_is_in_frame_units(smooth_traj):
    """t_max - t_min is a duration in source frames, and it is > 0 and finite."""
    comp = _fit(smooth_traj)
    chunks = chunk_bspline_trajectory(comp, chunk_size=CHUNK_SIZE, stride=1)
    spans = [float(ch["t"][STEPS - DEGREE - 1] - ch["t"][DEGREE]) for ch in chunks]
    interior = [s for s in spans[:-2] if s > 0]
    assert len(interior) > 0
    assert min(interior) > 0
    assert max(interior) < len(smooth_traj)


def test_num_actions_only_changes_sampling_density(smooth_traj):
    """Decoding at 8 vs 64 points samples the same curve, not a different one."""
    comp = _fit(smooth_traj)
    ch = chunk_bspline_trajectory(comp, chunk_size=CHUNK_SIZE, stride=1)[4]
    p = chunk_to_params(ch, STEPS, 1 + smooth_traj.shape[1])
    coarse = decode_bspline_action(p, degree=DEGREE, num_actions=8)
    fine = decode_bspline_action(p, degree=DEGREE, num_actions=64)
    assert np.allclose(coarse[0], fine[0], atol=1e-5)
    assert np.allclose(coarse[-1], fine[-1], atol=1e-5)
    assert np.allclose(coarse, fine[:: (63 // 7)][:8], atol=1e-3)


def test_relative_knot_encoding_decodes_identically(smooth_traj):
    comp = _fit(smooth_traj)
    ch = chunk_bspline_trajectory(comp, chunk_size=CHUNK_SIZE, stride=1)[4]
    p = chunk_to_params(ch, STEPS, 1 + smooth_traj.shape[1])
    enc = encode_relative_knots(p, degree=DEGREE)
    assert np.allclose(
        decode_bspline_action(p, degree=DEGREE, num_actions=16),
        decode_bspline_action(enc, degree=DEGREE, num_actions=16, relative_knots=True),
        atol=1e-5,
    )


# --------------------------------------------------------------------------
# edge cases
# --------------------------------------------------------------------------


def test_degenerate_domain_raises():
    p = np.zeros((STEPS, 11), dtype=np.float32)
    with pytest.raises(ValueError, match="Invalid B-spline range"):
        decode_bspline_action(p, degree=DEGREE, num_actions=8)


def test_constant_trajectory_is_representable():
    traj = np.tile(np.linspace(0, 1, 10)[None, :], (120, 1))
    comp = _fit(traj)
    assert comp.fit_error < MAX_ERROR
    ch = chunk_bspline_trajectory(comp, chunk_size=CHUNK_SIZE, stride=1)[0]
    p = chunk_to_params(ch, STEPS, 11)
    decoded = decode_bspline_action(p, degree=DEGREE, num_actions=8)
    # Last sample sits exactly on the padded t_max -- see the test below.
    assert np.allclose(decoded[:-1], traj[0], atol=1e-4)


def test_binary_gripper_channel_costs_knots():
    """A step function is the worst case for a smooth spline: it needs far more
    knots than the pose channels. Documented here so the cost is visible."""
    n = 200
    smooth_only = smooth_trajectory(n, 9)
    with_gripper = steppy_trajectory(n)
    n_smooth = len(_fit(smooth_only).knots)
    n_steppy = len(_fit(with_gripper).knots)
    assert n_steppy > n_smooth


def test_short_episode_still_chunks():
    traj = smooth_trajectory(40, 10)
    comp = _fit(traj)
    chunks = chunk_bspline_trajectory(comp, chunk_size=CHUNK_SIZE, stride=1)
    assert len(chunks) >= 1
    for ch in chunks:
        assert ch["t"].shape == (STEPS,)


def test_padded_tail_chunk_zeroes_its_last_sample():
    """Documented upstream quirk, not a port bug.

    When a chunk runs past the end of the global knot vector,
    ``chunk_bspline_trajectory`` pads by repeating the final knot. The padded
    vector therefore ends with ``t_max`` repeated, so at ``t == t_max`` every
    basis function has already closed its half-open support and the spline
    evaluates to 0. Interior chunks are unaffected because their ``t_max`` is
    strictly below the next knot.

    Consequence for deployment: drop (or ignore) the final decoded action of a
    padded chunk. It only ever occurs at the very end of an episode.
    """
    traj = np.tile(np.linspace(0, 1, 10)[None, :], (120, 1))
    comp = _fit(traj)
    ch = chunk_bspline_trajectory(comp, chunk_size=CHUNK_SIZE, stride=1)[0]
    p = chunk_to_params(ch, STEPS, 11)
    assert p[STEPS - DEGREE - 1, 0] == p[-1, 0], "expected a tail-padded knot vector"
    decoded = decode_bspline_action(p, degree=DEGREE, num_actions=8)
    assert np.allclose(decoded[-1], 0.0)
    assert not np.allclose(decoded[-2], 0.0)


def test_interior_chunk_is_fine_at_t_max():
    """The mirror of the test above: unpadded chunks decode correctly end to end."""
    smooth_traj = smooth_trajectory(800, 10, seed=7)
    comp = _fit(smooth_traj)
    c_full = comp.spline.tck[1]
    chunks = chunk_bspline_trajectory(comp, chunk_size=CHUNK_SIZE, stride=1)
    interior = [ch for s, ch in enumerate(chunks) if s + STEPS <= len(c_full)]
    assert interior
    for ch in interior[:20]:
        p = chunk_to_params(ch, STEPS, 11)
        assert p[STEPS - DEGREE - 1, 0] < p[-1, 0]
        decoded = decode_bspline_action(p, degree=DEGREE, num_actions=8)
        assert np.isfinite(decoded).all()
        assert np.allclose(decoded[-1], comp.spline(float(p[STEPS - DEGREE - 1, 0])), atol=1e-4)
