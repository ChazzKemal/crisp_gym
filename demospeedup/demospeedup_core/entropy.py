"""Differential-entropy estimators for a set of action samples.

Vendored from upstream ``aloha/act/detr/models/entropy_utils.py``
(DemoSpeedup, https://github.com/lingxiao-guo/DemoSpeedup). The KDE estimator
is the one the paper actually uses to label demonstrations; the
Kozachenko-Leonenko k-NN estimator ships alongside it upstream and is kept
here because it is the cheap sanity check for the KDE numbers.

All estimators take ``(batch, num_samples, dim)`` and return ``(batch, 1)``.
Inputs are expected in **normalised** action units (mean/std over the dataset,
exactly what the policy's own head emits) -- the KDE bandwidth is hard-coded
to 1.0 upstream, which only makes sense on a unit-ish scale.
"""

from __future__ import annotations

import torch
from scipy.special import digamma


def k_nn_distance(x: torch.Tensor, k: int) -> torch.Tensor:
    """Distances to the ``k`` nearest neighbours of every sample (self excluded)."""
    batch_size, num_samples, _ = x.size()
    x_flat = x.view(batch_size, num_samples, -1)
    distances = torch.cdist(x_flat, x_flat)  # (B, N, N)
    k_distances, _ = torch.topk(distances, k + 1, dim=-1, largest=False)
    return k_distances[:, :, 1:]  # drop the zero self-distance


def kozachenko_leonenko_entropy(x: torch.Tensor, k: int = 5) -> torch.Tensor:
    """k-NN differential entropy estimate.

    ``H ~= psi(N) - psi(k) + d * E[log eps_k]``, up to the constant volume term.

    Deviation from upstream: their line reads ``digamma_n - digamma_k - dim *
    log(...)``, i.e. it *subtracts* the log-distance term, which inverts the
    estimator -- a wider sample cloud would score as lower entropy. The sign is
    corrected here. Nothing in the pipeline depends on it (DemoSpeedup labels
    with :func:`kde_entropy`); it is kept as an independent cross-check, and an
    inverted cross-check is worse than none.
    """
    _, num_samples, dim = x.size()
    k_distances = k_nn_distance(x, k)
    avg_distances = k_distances.mean(dim=2)
    digamma_k = torch.tensor(digamma(k), dtype=torch.float32, device=x.device)
    digamma_n = torch.tensor(digamma(num_samples), dtype=torch.float32, device=x.device)
    return digamma_n - digamma_k + dim * torch.log(avg_distances).mean(dim=1, keepdim=True)


def gaussian_kernel(x: torch.Tensor, bandwidth: float) -> torch.Tensor:
    """Pairwise Gaussian kernel matrix ``exp(-||xi - xj||^2 / (2 h^2))``."""
    x_i = x.unsqueeze(2)  # (B, N, 1, D)
    x_j = x.unsqueeze(1)  # (B, 1, N, D)
    distances = torch.sum((x_i - x_j) ** 2, dim=-1)  # (B, N, N)
    return torch.exp(-distances / (2 * bandwidth**2))


def kde_entropy(x: torch.Tensor, bandwidth: float = 1.0) -> torch.Tensor:
    """Resubstitution KDE entropy: ``-mean_i log p_hat(x_i)``.

    This is DemoSpeedup's entropy signal. Upstream fixes ``bandwidth = 1``
    (``KDE.kde_entropy`` assigns it twice and ignores ``estimate_bandwidth``),
    and drops the ``1 / (h^d (2 pi)^{d/2})`` normaliser -- a constant offset
    that cancels when the per-episode entropy trace is z-normalised before
    segmentation. Both quirks are reproduced deliberately.
    """
    num_samples = x.size(1)
    kernel_values = gaussian_kernel(x, bandwidth)
    density = kernel_values.sum(dim=2) / num_samples  # (B, N)
    log_density = torch.log(density + 1e-8)
    return -log_density.mean(dim=1, keepdim=True)  # (B, 1)


def estimate_bandwidth(x: torch.Tensor, rule: str = "scott") -> float:
    """Scott/Silverman bandwidth for ``(num_samples, dim)`` -- unused by default."""
    num_samples, dim = x.size()
    std = x.std(dim=0).mean().item()
    if rule == "silverman":
        return 1.06 * std * num_samples ** (-1 / 5)
    if rule == "scott":
        return std * num_samples ** (-1 / (dim + 4))
    raise ValueError("Unsupported rule. Choose 'silverman' or 'scott'.")


__all__ = [
    "estimate_bandwidth",
    "gaussian_kernel",
    "k_nn_distance",
    "kde_entropy",
    "kozachenko_leonenko_entropy",
]
