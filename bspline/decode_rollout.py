"""Turn a policy's flat B-spline prediction into executable waypoints.

This is the deployment-side counterpart of ``convert_lerobot_to_bspline.py``:
the policy emits one flat parameter vector, this module reshapes it, repairs
what a network can get wrong, decodes the spline, and converts back to the
``[x, y, z, rx, ry, rz, gripper]`` layout the crisp_gym cartesian controller
expects.

Two upstream quirks are handled explicitly here (both pinned in ``tests/``):

* a **tail-padded** chunk evaluates to zero exactly at ``t_max``, so its final
  sample is dropped;
* **trailing frames** of an episode reuse the final chunk with a negative
  domain start, so ``t_min`` is clamped to 0 before deciding execution timing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bspline_core.bspline_action import decode_bspline_action
from bspline_core.knots import safer_knots
from bspline_core.rotation import convert_actions_10d_to_7d


@dataclass
class DecodedChunk:
    """Waypoints plus the times, in seconds from now, at which to execute them."""

    actions: np.ndarray  # (n, 7) -> [x, y, z, rx, ry, rz, gripper]
    times: np.ndarray  # (n,) seconds from the observation frame
    span_frames: float
    padded: bool


def reshape_prediction(flat: np.ndarray, n_action_steps: int, n_action_channels: int) -> np.ndarray:
    """Flat policy output -> ``(n_action_steps, n_action_channels)`` parameter matrix."""
    flat = np.asarray(flat, dtype=np.float64).reshape(-1)
    expected = n_action_steps * n_action_channels
    if flat.size != expected:
        raise ValueError(f"expected {expected} values, got {flat.size}")
    return flat.reshape(n_action_steps, n_action_channels)


def is_padded(params: np.ndarray, degree: int) -> bool:
    """A tail-padded chunk has its last knots all equal to ``t_max``."""
    return bool(np.isclose(params[-(degree + 1), 0], params[-1, 0]))


def decode(
    flat_or_params: np.ndarray,
    *,
    chunk_size: int,
    degree: int = 3,
    num_actions: int = 16,
    fps: float = 20.0,
    relative_knots: bool = False,
    n_action_channels: int = 11,
    speedup: float = 1.0,
) -> DecodedChunk:
    """Decode one prediction into waypoints and their execution times.

    ``speedup`` compresses the schedule: 2.0 executes the same geometric path
    in half the wall-clock time. The waypoints themselves do not change --
    only ``times`` -- which is exactly the acceleration mechanism the paper
    describes.
    """
    n_action_steps = chunk_size + 2 * degree
    params = (
        flat_or_params
        if np.ndim(flat_or_params) == 2
        else reshape_prediction(flat_or_params, n_action_steps, n_action_channels)
    )
    params = np.array(params, dtype=np.float64, copy=True)
    padded = is_padded(params, degree)

    # A network's raw knot column need not be monotone; repair it as upstream does.
    params[:, 0] = safer_knots(params[:, 0])

    n = int(num_actions) + (1 if padded else 0)
    decoded10 = decode_bspline_action(
        params, degree=degree, num_actions=n, relative_knots=relative_knots
    )

    t_min = float(params[degree, 0])
    t_max = float(params[n_action_steps - degree - 1, 0])
    times = np.linspace(t_min, t_max, n)
    if padded:  # last sample sits on the repeated knot and decodes to zero
        decoded10 = decoded10[:-1]
        times = times[:-1]

    times = np.maximum(times, 0.0) / (float(fps) * float(speedup))
    return DecodedChunk(
        actions=convert_actions_10d_to_7d(decoded10),
        times=times.astype(np.float64),
        span_frames=t_max - t_min,
        padded=padded,
    )


__all__ = ["DecodedChunk", "decode", "is_padded", "reshape_prediction"]
