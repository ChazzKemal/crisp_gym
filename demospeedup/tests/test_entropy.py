"""The entropy estimators must order samples by spread, and stay finite."""

import numpy as np
import torch

from demospeedup_core.entropy import (
    kde_entropy,
    kozachenko_leonenko_entropy,
    k_nn_distance,
)


def _cloud(scale: float, n: int = 200, dim: int = 7, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (scale * torch.randn(1, n, dim, generator=g)).float()


def test_kde_entropy_increases_with_spread():
    tight = float(kde_entropy(_cloud(0.05)))
    wide = float(kde_entropy(_cloud(1.0)))
    assert tight < wide


def test_kde_entropy_is_finite_for_a_degenerate_cloud():
    """All samples identical -- upstream's 1e-8 floor must keep this finite."""
    x = torch.zeros(1, 64, 7)
    assert torch.isfinite(kde_entropy(x)).all()


def test_kde_entropy_shape_and_batching():
    x = torch.cat([_cloud(0.1, seed=1), _cloud(2.0, seed=2)], dim=0)
    out = kde_entropy(x)
    assert out.shape == (2, 1)
    assert out[0, 0] < out[1, 0]


def test_kde_entropy_is_permutation_invariant():
    x = _cloud(0.7)
    shuffled = x[:, torch.randperm(x.shape[1]), :]
    assert torch.allclose(kde_entropy(x), kde_entropy(shuffled), atol=1e-5)


def test_smaller_bandwidth_resolves_more_structure():
    x = _cloud(1.0)
    assert float(kde_entropy(x, 0.25)) > float(kde_entropy(x, 4.0))


def test_knn_distances_exclude_self():
    x = _cloud(1.0, n=32)
    d = k_nn_distance(x, k=3)
    assert d.shape == (1, 32, 3)
    assert (d > 0).all()


def test_kozachenko_leonenko_increases_with_spread():
    tight = float(kozachenko_leonenko_entropy(_cloud(0.05)))
    wide = float(kozachenko_leonenko_entropy(_cloud(1.0)))
    assert tight < wide


def test_estimators_agree_on_ordering():
    scales = [0.05, 0.2, 1.0, 5.0]
    kde = [float(kde_entropy(_cloud(s))) for s in scales]
    knn = [float(kozachenko_leonenko_entropy(_cloud(s))) for s in scales]
    assert np.all(np.diff(kde) > 0) and np.all(np.diff(knn) > 0)
