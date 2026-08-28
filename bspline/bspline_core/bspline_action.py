"""B-spline action-chunk representation.

Vendored (and de-coupled from ``diffusion_policy``) from
``bspline_policy/common/bspline_action.py`` of the B-spline Policy release,
https://github.com/B-spline-policy/bspline-policy.

Representation
--------------
One action chunk is a dense parameter matrix of shape::

    (chunk_size + 2 * degree, 1 + action_dim)

* column 0 holds the knot vector, expressed in **frame-index units relative to
  the current frame** (so ``knots[degree] >= 0`` is when the spline's valid
  domain starts, measured from "now");
* columns ``1:`` hold the B-spline control points for the action dimensions.

Decoding evaluates the spline at ``num_actions`` points spaced uniformly over
``[knots[degree], knots[-(degree + 1)]]``. Because the fitted knot spacing
adapts to how fast the demonstration is moving, that interval spans a
*variable* number of source frames -- which is what lets the policy replay a
demonstration faster than it was recorded.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.interpolate import BSpline, generate_knots, make_lsq_spline

from .knots import decode_relative_knots, encode_relative_knots


class ScipyBSplineCompression:
    """Fit a multi-dimensional trajectory with a reduced-knot B-spline.

    ``generate_knots`` yields progressively richer knot vectors; the first one
    whose least-squares fit stays within ``max_error`` (max-abs over all
    samples and all dimensions) wins. If none does, the last (richest) one is
    used and a warning is printed -- matching upstream behaviour.
    """

    def __init__(self, degree: int = 3):
        self.degree = int(degree)
        self.spline: Optional[BSpline] = None
        self.knots: Optional[np.ndarray] = None
        self.fit_error: Optional[float] = None
        self.converged: bool = False

    def compress(
        self,
        data: np.ndarray,
        max_error: float = 0.01,
        verbose: bool = False,
        s: float = 1e-12,
    ) -> np.ndarray:
        t = np.arange(len(data))
        last_knots = None
        last_error = None
        for knots in generate_knots(t, data, s=s):
            spl = make_lsq_spline(t, data, knots)
            pred_data = spl(t)
            error = np.abs(pred_data - data).max()
            last_knots = knots
            last_error = error
            if error < max_error:
                self.knots = knots
                self.spline = spl
                self.fit_error = float(error)
                self.converged = True
                break

        if self.knots is None:
            if verbose:
                print(
                    "Failing to compress trajectory with max error "
                    f"{max_error}, use min error we can find. Error is {last_error}. "
                    "You can try to increase the s value."
                )
            self.knots = last_knots
            self.spline = make_lsq_spline(t, data, self.knots)
            self.fit_error = float(last_error)
            self.converged = False

        if verbose:
            print(f"compression ratio: {len(self.knots) / len(t)}")

        return self.knots


def extract_unique_knots(t_full: np.ndarray, degree: int) -> np.ndarray:
    """Strip the repeated boundary knots from FITPACK's full knot vector."""
    return t_full[degree:-degree]


def chunk_bspline_trajectory(
    compressor: ScipyBSplineCompression,
    chunk_size: int = 8,
    stride: Optional[int] = None,
    verbose: bool = False,
) -> list[dict]:
    """Split a fitted B-spline into fixed-size parameter windows.

    Each window takes ``chunk_size + 2 * degree`` consecutive entries of the
    global knot vector together with the *same-indexed* control points. That is
    the local-support property of B-splines: over
    ``[t[s + degree], t[s + M - degree - 1]]`` the windowed spline is identical
    to the global one. Windows that run past the end are padded by repeating
    the last knot / control point (only affecting the region beyond the
    original domain).
    """
    if compressor.spline is None:
        raise ValueError("Please call compress() before chunking")

    if stride is None:
        stride = chunk_size - 1

    degree = compressor.degree
    t_full, c_full, _ = compressor.spline.tck
    unique_t = extract_unique_knots(t_full, degree)
    n_unique = len(unique_t)
    chunks = []

    if verbose:
        print(
            f"B-spline chunking: len(t)={len(t_full)}, len(c)={len(c_full)}, "
            f"degree={degree}, unique_knots={n_unique}, chunk_size={chunk_size}, "
            f"stride={stride}"
        )

    for start_idx in range(0, n_unique - 1, stride):
        first_pos = start_idx + degree
        last_pos = start_idx + chunk_size + degree

        t_start = max(0, first_pos - degree)
        t_end = min(len(t_full), last_pos + degree)

        chunk_t = t_full[t_start:t_end]
        chunk_c = c_full[t_start:t_end]
        expected_len = chunk_size + 2 * degree

        if len(chunk_t) < expected_len:
            chunk_t = np.concatenate([chunk_t, np.full(expected_len - len(chunk_t), chunk_t[-1])])
        if len(chunk_c) < expected_len:
            pad = np.repeat(chunk_c[-1:], expected_len - len(chunk_c), axis=0)
            chunk_c = np.concatenate([chunk_c, pad], axis=0)

        if len(chunk_t) != expected_len:
            raise AssertionError("chunk_t length should equal chunk_size + 2 * degree")
        if len(chunk_c) != expected_len:
            raise AssertionError("chunk_c length should equal chunk_size + 2 * degree")

        chunks.append({"t": chunk_t, "c": chunk_c, "k": degree})

    return chunks


def decode_bspline_action(
    action_params,
    degree: int = 3,
    num_actions: int = 8,
    relative_knots: bool = False,
) -> np.ndarray:
    """Decode one B-spline parameter matrix into regular action vectors.

    Returns ``(num_actions, action_dim)``, sampled uniformly in frame-index
    time across the chunk's valid domain.
    """
    if hasattr(action_params, "detach"):
        action_params = action_params.detach().cpu().numpy()
    action_params = np.asarray(action_params, dtype=np.float64)
    if relative_knots:
        action_params = decode_relative_knots(action_params, degree=degree)

    knots = action_params[:, 0].copy()
    control_points = action_params[: -(degree + 1), 1:].copy()
    t_min = knots[degree]
    t_max = knots[-(degree + 1)]
    if t_max <= t_min:
        raise ValueError(f"Invalid B-spline range: [{t_min}, {t_max}]")

    if num_actions <= 1:
        t_eval = np.asarray([t_min], dtype=np.float64)
    else:
        t_eval = np.linspace(t_min, t_max, int(num_actions), dtype=np.float64)
    return BSpline(knots, control_points, degree, extrapolate=False)(t_eval).astype(np.float32)


def chunk_to_params(chunk: dict, n_action_steps: int, n_action_channels: int) -> np.ndarray:
    """Pack one ``{"t", "c", "k"}`` window into the dense parameter matrix."""
    params = np.zeros((n_action_steps, n_action_channels), dtype=np.float32)
    params[:, 0] = np.asarray(chunk["t"], dtype=np.float32)
    params[:, 1:] = np.asarray(chunk["c"], dtype=np.float32)
    return params


__all__ = [
    "ScipyBSplineCompression",
    "chunk_bspline_trajectory",
    "chunk_to_params",
    "decode_bspline_action",
    "decode_relative_knots",
    "encode_relative_knots",
    "extract_unique_knots",
]
