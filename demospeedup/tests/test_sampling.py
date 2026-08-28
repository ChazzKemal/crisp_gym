"""The prior-latent patch and the temporal sample buffer."""

import pytest
import torch
from torch import nn

from demospeedup_core.sampling import TemporalSampleBuffer, prior_latent_sampling


class _FakeACT(nn.Module):
    """Just enough of LeRobot's ACT to exercise the latent patch."""

    def __init__(self, latent_dim=4, dim_model=8):
        super().__init__()
        self.encoder_latent_input_proj = nn.Linear(latent_dim, dim_model)
        self.seen: list[torch.Tensor] = []

    def forward(self, latent):
        out = self.encoder_latent_input_proj(latent)
        self.seen.append(out.detach().clone())
        return out


class _FakePolicy:
    def __init__(self):
        self.model = _FakeACT()


def test_patch_replaces_zero_latents_with_prior_draws():
    policy = _FakePolicy()
    zeros = torch.zeros(6, 4)
    with prior_latent_sampling(policy) as patch:
        first = policy.model(zeros)
        second = policy.model(zeros)
        assert patch.n_patched == patch.n_calls == 2
    assert not torch.allclose(first, second)  # different draws, different outputs


def test_patch_is_reverted_on_exit():
    policy = _FakePolicy()
    original = policy.model.encoder_latent_input_proj
    with prior_latent_sampling(policy):
        assert policy.model.encoder_latent_input_proj is not original
    assert policy.model.encoder_latent_input_proj is original


def test_patch_leaves_non_zero_latents_alone():
    """A non-zero latent means the VAE-encoder branch ran -- do not touch it."""
    policy = _FakePolicy()
    latent = torch.ones(3, 4)
    with prior_latent_sampling(policy) as patch:
        out = policy.model(latent)
        assert patch.n_patched == 0
    assert torch.allclose(out, policy.model.encoder_latent_input_proj(latent))


def test_seeded_generator_is_reproducible():
    outs = []
    for _ in range(2):
        torch.manual_seed(0)  # same Linear weights on both passes
        policy = _FakePolicy()
        gen = torch.Generator().manual_seed(7)
        with prior_latent_sampling(policy, gen):
            outs.append(policy.model(torch.zeros(5, 4)))
    assert torch.allclose(outs[0], outs[1])


def test_buffer_pools_the_action_at_now():
    """Chunk added ``age`` steps ago must contribute its offset-``age`` element."""
    buffer = TemporalSampleBuffer(chunk_size=3)
    for t in range(3):
        # sample s of chunk t, offset h, encodes (t, s, h) as t*100 + s*10 + h
        chunk = torch.tensor(
            [[[t * 100 + s * 10 + h] for h in range(3)] for s in range(2)], dtype=torch.float32
        )
        buffer.add(chunk)
    pooled = buffer.current().flatten().tolist()
    # newest chunk (t=2) at offset 0, t=1 at offset 1, t=0 at offset 2
    assert pooled == [200, 210, 101, 111, 2, 12]


def test_buffer_is_bounded_by_chunk_size():
    buffer = TemporalSampleBuffer(chunk_size=4)
    for _ in range(10):
        buffer.add(torch.zeros(5, 4, 3))
    assert len(buffer) == 4
    assert buffer.current().shape == (20, 3)


def test_buffer_grows_during_the_first_frames():
    buffer = TemporalSampleBuffer(chunk_size=8)
    buffer.add(torch.zeros(5, 8, 3))
    assert buffer.current().shape == (5, 3)
    buffer.add(torch.zeros(5, 8, 3))
    assert buffer.current().shape == (10, 3)


def test_buffer_rejects_use_before_any_prediction():
    with pytest.raises(RuntimeError):
        TemporalSampleBuffer(4).current()
