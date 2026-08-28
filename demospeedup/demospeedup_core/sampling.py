"""Draw action-chunk samples from a LeRobot ACT policy's CVAE prior.

Why a patch instead of a fork
-----------------------------
Upstream adds :meth:`DETRVAE.get_samples`, which sets ``mu = logvar = 0`` and
reparametrises ``num_samples`` latents -- i.e. it draws ``z ~ N(0, I)``, the
CVAE *prior*, and decodes one action chunk per draw. The spread of those
chunks is the policy's action entropy at that observation.

LeRobot's ``ACT.forward`` already takes exactly that branch at inference: with
no ``action`` key in the batch it builds ``latent_sample = zeros(B, latent_dim)``
and immediately feeds it to ``encoder_latent_input_proj``. So rather than
vendoring a 100-line ``get_samples``, :class:`PriorLatentSampling` swaps that
one projection for a wrapper that substitutes ``randn`` for the zeros. Nothing
else about the forward pass changes, and the patch is reverted on exit.

The wrapper refuses to fire if the incoming latent is not all-zeros -- that
would mean the VAE-encoder branch ran (training), where replacing the latent
would be wrong.
"""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager

import torch
from torch import Tensor, nn


class _PriorLatentProj(nn.Module):
    """Wraps ``encoder_latent_input_proj``: zeros in, prior samples out."""

    def __init__(self, inner: nn.Module, generator: torch.Generator | None = None):
        super().__init__()
        self.inner = inner
        self.generator = generator
        self.n_calls = 0
        self.n_patched = 0

    def forward(self, latent_sample: Tensor) -> Tensor:
        self.n_calls += 1
        if torch.count_nonzero(latent_sample) == 0:
            latent_sample = torch.randn(
                latent_sample.shape,
                dtype=latent_sample.dtype,
                device=latent_sample.device,
                generator=self.generator,
            )
            self.n_patched += 1
        return self.inner(latent_sample)


@contextmanager
def prior_latent_sampling(policy, generator: torch.Generator | None = None):
    """Temporarily make ``policy``'s ACT model sample ``z ~ N(0, I)``."""
    model = policy.model
    original = model.encoder_latent_input_proj
    wrapper = _PriorLatentProj(original, generator)
    model.encoder_latent_input_proj = wrapper
    try:
        yield wrapper
    finally:
        model.encoder_latent_input_proj = original


class ACTChunkSampler:
    """``num_samples`` action chunks per observation, in normalised units.

    ``policy`` is a LeRobot ``ACTPolicy`` in eval mode and ``preprocessor`` its
    ``PolicyProcessorPipeline`` (normalisation + device placement). The samples
    are taken *before* the post-processor, so they stay in the mean/std space
    the entropy estimator expects -- upstream reads the raw head output too.
    """

    def __init__(self, policy, preprocessor, num_samples: int = 10, seed: int | None = None):
        self.policy = policy
        self.preprocessor = preprocessor
        self.num_samples = num_samples
        self.generator = None
        if seed is not None:
            device = getattr(policy.config, "device", "cpu")
            self.generator = torch.Generator(device=device).manual_seed(seed)

    @property
    def chunk_size(self) -> int:
        return self.policy.config.chunk_size

    @torch.no_grad()
    def sample(self, batch: dict) -> Tensor:
        """``(F, ...)`` observations -> ``(F, num_samples, chunk_size, action_dim)``.

        Each of the ``F`` frames is tiled ``num_samples`` times so every copy
        gets its own prior draw in a single forward pass. Upstream instead
        repeat-interleaves *after* the vision backbone; tiling the input is
        equivalent apart from recomputing the backbone, which costs time but
        keeps us on the stock LeRobot forward.
        """
        # ``action`` must not reach the model: it is what would flip ACT onto
        # the VAE-encoder branch (in training mode) and it is never an input.
        batch = {k: v for k, v in batch.items() if not k.startswith("action")}
        batch = self.preprocessor(batch)
        n_frames = self._batch_size(batch)
        tiled = {
            k: self._tile(v) if isinstance(v, Tensor) else v for k, v in batch.items()
        }
        if self.policy.config.image_features:
            tiled = dict(tiled)
            tiled["observation.images"] = [
                tiled[key] for key in self.policy.config.image_features
            ]
        with prior_latent_sampling(self.policy, self.generator) as patch:
            actions = self.policy.model(tiled)[0]  # (F * S, chunk, dim)
            if patch.n_patched != patch.n_calls:
                raise RuntimeError(
                    "ACT did not take the zero-latent inference branch; refusing "
                    "to report entropy from a non-prior latent"
                )
        return actions.reshape(n_frames, self.num_samples, *actions.shape[1:])

    def _tile(self, value: Tensor) -> Tensor:
        # (F, ...) -> (F * S, ...) with each frame's copies adjacent.
        return value.repeat_interleave(self.num_samples, dim=0)

    @staticmethod
    def _batch_size(batch: dict) -> int:
        for key in ("observation.state", "observation.environment_state"):
            if key in batch:
                return batch[key].shape[0]
        for key, value in batch.items():
            if key.startswith("observation.images.") and isinstance(value, Tensor):
                return value.shape[0]
        raise KeyError("cannot infer batch size: no observation tensor in batch")


class TemporalSampleBuffer:
    """Pools every still-valid chunk's samples for the action at *now*.

    Upstream keeps a dense ``(T, T + chunk, num_samples, dim)`` tensor on the
    GPU and slices column ``t``; with ``chunk_size = 100`` and 300-frame
    episodes that is hundreds of MB for a strictly banded quantity. A ring
    buffer of the last ``chunk_size`` predictions holds exactly the same
    samples: the chunk predicted ``age`` steps ago contributes its element at
    offset ``age``.
    """

    def __init__(self, chunk_size: int):
        self.chunk_size = chunk_size
        self._chunks: deque[Tensor] = deque(maxlen=chunk_size)

    def reset(self) -> None:
        self._chunks.clear()

    def add(self, samples: Tensor) -> None:
        """``samples``: ``(num_samples, chunk_size, action_dim)`` for the current step."""
        self._chunks.appendleft(samples)

    def current(self) -> Tensor:
        """``(n_predictions * num_samples, action_dim)`` samples of the action now."""
        if not self._chunks:
            raise RuntimeError("no predictions buffered yet")
        return torch.cat(
            [chunk[:, age, :] for age, chunk in enumerate(self._chunks)], dim=0
        )

    def __len__(self) -> int:
        return len(self._chunks)


__all__ = ["ACTChunkSampler", "TemporalSampleBuffer", "prior_latent_sampling"]
