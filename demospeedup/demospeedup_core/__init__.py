"""Dependency-light port of DemoSpeedup's entropy-guided acceleration.

Upstream: https://github.com/lingxiao-guo/DemoSpeedup
(Guo, Xue, Xu, Xu -- "DemoSpeedup: Accelerating Visuomotor Policies via
Entropy-Guided Demonstration Acceleration", CoRL 2025, arXiv:2506.05064)

Only the *method* is vendored -- the entropy estimator, the segmentation of a
demonstration into precision / non-precision phases, and the label-driven
variable-stride retiming. Upstream's two training stacks (an Aloha/ACT fork
pinned to MuJoCo 2.1 + a ``diffusion_policy`` fork, and a ``robobase``/Bigym
tree) are deliberately not used: crisp_gym trains through LeRobot, so the
proxy policy and the accelerated policy are both stock ``lerobot-train`` runs.

Nothing in this package imports LeRobot; ``entropy`` and ``sampling`` need
torch, everything else is numpy.
"""

from .entropy import kde_entropy, kozachenko_leonenko_entropy
from .sampling import ACTChunkSampler, TemporalSampleBuffer, prior_latent_sampling
from .retiming import (
    HIGH_V,
    LOW_V,
    RetimingStats,
    process_action_label_upstream,
    retiming_stats,
    select_keep_indices,
)
from .segmentation import (
    NON_PRECISION,
    PRECISION,
    SegmentationResult,
    segment_entropy,
    zscore,
)

__all__ = [
    "ACTChunkSampler",
    "HIGH_V",
    "LOW_V",
    "NON_PRECISION",
    "PRECISION",
    "RetimingStats",
    "SegmentationResult",
    "TemporalSampleBuffer",
    "kde_entropy",
    "kozachenko_leonenko_entropy",
    "prior_latent_sampling",
    "process_action_label_upstream",
    "retiming_stats",
    "segment_entropy",
    "select_keep_indices",
    "zscore",
]
