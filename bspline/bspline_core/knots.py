"""Relative-knot encoding helpers.

Vendored from ``bspline_policy/common/knots.py`` (B-spline Policy, Han et al.
2026, https://github.com/B-spline-policy/bspline-policy) so that the encoding
is byte-for-byte the same as upstream. Works on both numpy arrays and torch
tensors; the ``knots`` slice is deliberately a *view* into ``result`` so the
in-place writes land in the returned array.
"""

from __future__ import annotations

import numpy as np

try:  # torch is optional here -- the conversion pipeline is numpy-only.
    import torch

    def _is_tensor(x: object) -> bool:
        return torch.is_tensor(x)

except ImportError:  # pragma: no cover - exercised only without torch

    def _is_tensor(x: object) -> bool:
        return False


def encode_relative_knots(action_data, degree: int = 3):
    """Encode knot values as first valid knot plus adjacent differences.

    Slot 0 receives the first *valid* knot ``t[degree]``; slots ``1:`` receive
    ``t[i] - t[i - 1]``. Exactly invertible by :func:`decode_relative_knots`.
    """
    result = action_data.clone() if _is_tensor(action_data) else action_data.copy()
    knots = result[..., 0]
    original_knots = knots.clone() if _is_tensor(knots) else knots.copy()

    knots[..., 0] = original_knots[..., degree]
    knots[..., 1:] = original_knots[..., 1:] - original_knots[..., :-1]
    return result


def decode_relative_knots(action_data, degree: int = 3):
    """Decode the representation produced by :func:`encode_relative_knots`."""
    result = action_data.clone() if _is_tensor(action_data) else action_data.copy()
    encoded = result[..., 0].clone() if _is_tensor(result) else result[..., 0].copy()
    knots = result[..., 0]
    n_knots = knots.shape[-1]

    knots[..., degree] = encoded[..., 0]
    for knot_idx in range(degree - 1, -1, -1):
        knots[..., knot_idx] = knots[..., knot_idx + 1] - encoded[..., knot_idx + 1]
    for knot_idx in range(degree + 1, n_knots):
        knots[..., knot_idx] = knots[..., knot_idx - 1] + encoded[..., knot_idx]

    return result


def safer_knots(knots: np.ndarray) -> np.ndarray:
    """Force a knot vector to be non-decreasing (upstream inference guard).

    A network's raw prediction can violate monotonicity; upstream repairs it
    before handing the vector to ``scipy.interpolate.BSpline``.
    """
    knots = np.asarray(knots, dtype=np.float64).copy()
    for idx in range(1, len(knots)):
        if knots[idx] < knots[idx - 1]:
            knots[idx] = knots[idx - 1] + 1e-6
    return knots
