"""Per-timestep B-spline chunk preprocessing.

Faithful re-implementation of upstream ``BSplineChunkSampler`` with the
``diffusion_policy.ReplayBuffer`` dependency replaced by a plain
``(actions, episode_ends)`` pair. The chunk-construction and
timestep-to-chunk assignment logic is unchanged; see
``tests/test_chunk_sampler.py`` for the equivalence checks.

Assignment rule (upstream, reproduced verbatim)
-----------------------------------------------
Walking the fitted chunks in order, chunk *j* is assigned to every timestep
``i`` up to and including ``floor(t_j[degree])`` that has not already been
claimed by an earlier chunk -- i.e. a timestep uses the last chunk whose valid
domain has not yet started. Each assignment stores its own copy with the knot
column shifted by ``-i`` so knots are relative to the current frame. Trailing
timesteps up to ``ep_length - max_first_k + 1`` reuse the final chunk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .bspline_action import (
    ScipyBSplineCompression,
    chunk_bspline_trajectory,
    chunk_to_params,
)
from .knots import encode_relative_knots


@dataclass
class EpisodeFitStats:
    """Diagnostics for one episode's B-spline fit."""

    episode_index: int
    length: int
    n_unique_knots: int
    n_chunks: int
    fit_error: float
    converged: bool

    @property
    def compression_ratio(self) -> float:
        return self.n_unique_knots / max(self.length, 1)


@dataclass
class BSplineChunkSampler:
    """Preprocess an episode-concatenated action array into B-spline chunks.

    Parameters mirror the upstream sampler. ``actions`` is ``(N, action_dim)``
    with episodes laid out back-to-back and delimited by ``episode_ends``
    (exclusive cumulative ends, as in ``diffusion_policy``'s ReplayBuffer).
    """

    actions: np.ndarray
    episode_ends: np.ndarray
    chunk_size: int = 10
    degree: int = 3
    max_error: float = 0.002
    stride: int = 1
    relative_knots: bool = False
    max_first_k: int = 1
    episode_mask: Optional[np.ndarray] = None
    verbose: bool = False

    all_actions: np.ndarray = field(init=False)
    timestep_to_chunk: np.ndarray = field(init=False)
    valid_timesteps: np.ndarray = field(init=False)
    fit_stats: list[EpisodeFitStats] = field(init=False)

    def __post_init__(self) -> None:
        self.actions = np.asarray(self.actions, dtype=np.float32)
        self.episode_ends = np.asarray(self.episode_ends, dtype=np.int64)
        self.n_action_steps = int(self.chunk_size) + 2 * int(self.degree)
        self.n_control_dims = int(self.actions.shape[1])
        self.n_action_channels = 1 + self.n_control_dims
        if self.max_first_k < 1:
            self.max_first_k = 1
        if self.episode_mask is None:
            self.episode_mask = np.ones(self.episode_ends.shape, dtype=bool)
        self._preprocess()

    def _preprocess(self) -> None:
        all_chunks: list[np.ndarray] = []
        self.fit_stats = []
        self.timestep_to_chunk = np.full(int(self.episode_ends[-1]), -1, dtype=np.int64)

        for ep_idx in range(len(self.episode_ends)):
            if not self.episode_mask[ep_idx]:
                continue

            ep_start = 0 if ep_idx == 0 else int(self.episode_ends[ep_idx - 1])
            ep_end = int(self.episode_ends[ep_idx])
            episode_actions = self.actions[ep_start:ep_end]
            ep_length = len(episode_actions)

            compressor = ScipyBSplineCompression(degree=self.degree)
            compressor.compress(
                episode_actions, max_error=self.max_error, verbose=self.verbose
            )

            chunks = chunk_bspline_trajectory(
                compressor, chunk_size=self.chunk_size, stride=self.stride
            )
            t_full = compressor.spline.tck[0]
            self.fit_stats.append(
                EpisodeFitStats(
                    episode_index=ep_idx,
                    length=ep_length,
                    n_unique_knots=len(t_full) - 2 * self.degree,
                    n_chunks=len(chunks),
                    fit_error=float(compressor.fit_error),
                    converged=bool(compressor.converged),
                )
            )

            chunk_data = None
            local_idx = 0
            for chunk in chunks:
                chunk_data = chunk_to_params(
                    chunk, self.n_action_steps, self.n_action_channels
                )
                t_timesteps = chunk["t"]
                while local_idx <= t_timesteps[self.degree]:
                    all_chunks.append(self._localize(chunk_data, local_idx))
                    self.timestep_to_chunk[local_idx + ep_start] = len(all_chunks) - 1
                    local_idx += 1

            while local_idx < ep_length - self.max_first_k + 1:
                all_chunks.append(self._localize(chunk_data, local_idx))
                self.timestep_to_chunk[local_idx + ep_start] = len(all_chunks) - 1
                local_idx += 1

        if all_chunks:
            self.all_actions = np.asarray(all_chunks, dtype=np.float32)
        else:
            self.all_actions = np.zeros(
                (0, self.n_action_steps, self.n_action_channels), dtype=np.float32
            )
        self.valid_timesteps = np.flatnonzero(self.timestep_to_chunk >= 0)

    def _localize(self, chunk_data: np.ndarray, local_idx: int) -> np.ndarray:
        """Shift knots to be relative to ``local_idx`` and optionally re-encode."""
        local = chunk_data.copy()
        local[:, 0] -= local_idx
        if self.relative_knots:
            local = encode_relative_knots(local, degree=self.degree)
        return local

    def __len__(self) -> int:
        return len(self.valid_timesteps)

    def chunk_for_timestep(self, timestep: int) -> np.ndarray:
        """Parameter matrix assigned to a global timestep index."""
        chunk_idx = self.timestep_to_chunk[timestep]
        if chunk_idx < 0:
            raise KeyError(f"timestep {timestep} has no assigned chunk")
        return self.all_actions[chunk_idx].copy()

    def get_action_stats(self) -> dict:
        """Per-(step, channel) statistics over all chunks, as upstream."""
        if len(self.all_actions) == 0:
            shape = (1, self.n_action_steps, self.n_action_channels)
            return {
                "min": np.zeros(shape, dtype=np.float32),
                "max": np.ones(shape, dtype=np.float32),
                "mean": np.zeros(shape, dtype=np.float32),
                "std": np.ones(shape, dtype=np.float32),
            }
        return {
            "min": np.min(self.all_actions, axis=0, keepdims=True),
            "max": np.max(self.all_actions, axis=0, keepdims=True),
            "mean": np.mean(self.all_actions, axis=0, keepdims=True),
            "std": np.std(self.all_actions, axis=0, keepdims=True),
        }

    def get_channel_stats(self) -> dict:
        """Per-*channel* stats, reduced over the step axis.

        Upstream's ``get_normalizer`` collapses the step axis before building
        the action normalizer, so every row of the parameter matrix shares one
        scale per channel. Reproduced here for the same reason: the knot column
        and each action dimension live on very different scales, but rows of the
        same column do not.
        """
        stats = self.get_action_stats()
        return {
            "min": np.min(stats["min"], axis=1, keepdims=True),
            "max": np.max(stats["max"], axis=1, keepdims=True),
            "mean": np.mean(stats["mean"], axis=1, keepdims=True),
            "std": np.mean(stats["std"], axis=1, keepdims=True),
        }


__all__ = ["BSplineChunkSampler", "EpisodeFitStats"]
