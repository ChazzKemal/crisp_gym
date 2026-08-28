"""Dependency-light port of the B-spline Policy action representation.

Upstream: https://github.com/B-spline-policy/bspline-policy
(Han, Xiong, Chen, Liu, Torralba, Zhu, Du -- arXiv:2607.09648)

Only the *action representation* is vendored here -- the pieces needed to turn
a recorded trajectory into B-spline parameter chunks and back. The upstream
training stack (a ``diffusion_policy`` fork pinned to zarr 2.12 / diffusers
0.11 / robomimic 0.2) is deliberately not used; crisp_gym trains through
LeRobot instead.
"""

from .bspline_action import (
    ScipyBSplineCompression,
    chunk_bspline_trajectory,
    chunk_to_params,
    decode_bspline_action,
    extract_unique_knots,
)
from .chunk_sampler import BSplineChunkSampler, EpisodeFitStats
from .knots import decode_relative_knots, encode_relative_knots, safer_knots
from .rotation import convert_actions_7d_to_10d, convert_actions_10d_to_7d

__all__ = [
    "BSplineChunkSampler",
    "EpisodeFitStats",
    "ScipyBSplineCompression",
    "chunk_bspline_trajectory",
    "chunk_to_params",
    "convert_actions_10d_to_7d",
    "convert_actions_7d_to_10d",
    "decode_bspline_action",
    "decode_relative_knots",
    "encode_relative_knots",
    "extract_unique_knots",
    "safer_knots",
]
