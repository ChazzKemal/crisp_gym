"""Relative-knot encoding must be exactly invertible, on numpy and torch alike."""

import numpy as np
import torch

from bspline_core.knots import decode_relative_knots, encode_relative_knots, safer_knots

DEGREE = 3
CHUNK, CHANNELS = 10, 11
STEPS = CHUNK + 2 * DEGREE


def _params(seed=0):
    rng = np.random.default_rng(seed)
    p = rng.normal(size=(STEPS, CHANNELS)).astype(np.float32)
    p[:, 0] = np.cumsum(rng.uniform(0.5, 3.0, size=STEPS)) - 5.0  # monotone knots
    return p


def test_encode_decode_round_trip_numpy():
    p = _params()
    assert np.allclose(decode_relative_knots(encode_relative_knots(p, DEGREE), DEGREE), p, atol=1e-5)


def test_encode_decode_round_trip_torch():
    p = torch.from_numpy(_params())
    out = decode_relative_knots(encode_relative_knots(p, DEGREE), DEGREE)
    assert torch.allclose(out, p, atol=1e-5)


def test_encode_does_not_mutate_input():
    p = _params()
    before = p.copy()
    encode_relative_knots(p, DEGREE)
    assert np.array_equal(p, before)


def test_encode_leaves_control_points_untouched():
    p = _params()
    assert np.array_equal(encode_relative_knots(p, DEGREE)[:, 1:], p[:, 1:])


def test_encoded_slot_zero_is_first_valid_knot():
    """Slot 0 carries t[degree] -- the start of the spline's valid domain."""
    p = _params()
    assert np.isclose(encode_relative_knots(p, DEGREE)[0, 0], p[DEGREE, 0], atol=1e-5)


def test_encoded_tail_slots_are_adjacent_differences():
    p = _params()
    enc = encode_relative_knots(p, DEGREE)
    assert np.allclose(enc[1:, 0], np.diff(p[:, 0]), atol=1e-5)


def test_batched_encoding_matches_per_item():
    batch = np.stack([_params(s) for s in range(4)])
    enc = encode_relative_knots(batch, DEGREE)
    for i in range(4):
        assert np.allclose(enc[i], encode_relative_knots(batch[i], DEGREE))


def test_safer_knots_enforces_monotonicity():
    bad = np.array([0.0, 3.0, 1.0, 2.0, 9.0, 8.0])
    fixed = safer_knots(bad)
    assert np.all(np.diff(fixed) >= 0)
    assert np.array_equal(safer_knots(np.arange(6.0)), np.arange(6.0))
