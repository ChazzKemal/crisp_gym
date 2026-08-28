"""Turn a per-frame entropy trace into a binary precision / non-precision label.

Upstream (``imitate_episodes.py::hdbscan_with_custom_merge``) z-normalises the
entropy trace and the frame index, clusters the resulting 2D points with
HDBSCAN(min_cluster_size=5), then collapses every cluster to a binary label::

    label 0 -> "precision"      : replay at 2x    (low entropy, be careful)
    label 1 -> "non-precision"  : replay at 4x    (high entropy, free motion)

Two backends are provided:

``hdbscan``
    A faithful reimplementation of the above. Needs ``sklearn>=1.3`` (or the
    standalone ``hdbscan`` package); neither is installed in ``lerobot-041``,
    so this backend is opt-in.

``threshold`` (default)
    numpy only. Split at the episode mean entropy, then enforce a minimum run
    length of ``min_cluster_size`` frames and re-apply upstream's own
    cluster rule to the merged runs. On a monotone-in-time feature pair like
    ``(t, entropy)`` HDBSCAN's clusters *are* essentially contiguous
    above/below-mean runs, so the two agree closely -- see ``analyze_labels.py
    --compare-backends`` to check on a given dataset.

Upstream's cluster rule is reproduced verbatim, quirk included::

    if np.mean(cluster_points[:, 1] < 0):   # truthy iff *any* point is < 0
        label = 0                           # precision
    else:
        label = -1                          # -> abs() -> 1, non-precision

so a segment is only marked "fast" when *every* one of its frames sits above
the episode's mean entropy. Noise points (HDBSCAN ``-1``) also come out as 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PRECISION = 0
NON_PRECISION = 1


@dataclass
class SegmentationResult:
    labels: np.ndarray  # (T,) int in {0, 1}
    entropy_z: np.ndarray  # (T,) z-normalised entropy
    raw_clusters: np.ndarray  # (T,) backend cluster ids, -1 = noise
    backend: str

    @property
    def fast_fraction(self) -> float:
        return float(np.mean(self.labels == NON_PRECISION))


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    std = x.std()
    if std < 1e-12:
        return np.zeros_like(x)
    return (x - x.mean()) / std


def _cluster_vote(entropy_z: np.ndarray, member_mask: np.ndarray) -> int:
    """Upstream's rule: any below-mean frame in the cluster -> precision."""
    points = entropy_z[member_mask]
    return PRECISION if np.mean(points < 0) else NON_PRECISION


def _runs(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """Contiguous ``(start, stop, value)`` runs of an integer array."""
    if len(labels) == 0:
        return []
    edges = np.flatnonzero(np.diff(labels)) + 1
    bounds = np.concatenate([[0], edges, [len(labels)]])
    return [(int(a), int(b), int(labels[a])) for a, b in zip(bounds[:-1], bounds[1:])]


def _enforce_min_run(labels: np.ndarray, min_run: int) -> np.ndarray:
    """Absorb runs shorter than ``min_run`` into a neighbour, shortest first."""
    labels = labels.copy()
    if min_run <= 1:
        return labels
    while True:
        runs = _runs(labels)
        if len(runs) <= 1:
            return labels
        short = [r for r in runs if r[1] - r[0] < min_run]
        if not short:
            return labels
        a, b, _ = min(short, key=lambda r: r[1] - r[0])
        idx = runs.index((a, b, int(labels[a])))
        prev_len = runs[idx - 1][1] - runs[idx - 1][0] if idx > 0 else -1
        next_len = runs[idx + 1][1] - runs[idx + 1][0] if idx + 1 < len(runs) else -1
        take_prev = prev_len >= next_len
        labels[a:b] = labels[runs[idx - 1][0]] if take_prev else labels[runs[idx + 1][0]]


def segment_threshold(entropy_z: np.ndarray, min_cluster_size: int = 5) -> np.ndarray:
    """numpy-only backend: mean split, min run length, upstream cluster rule."""
    coarse = (entropy_z >= 0).astype(np.int64)
    merged = _enforce_min_run(coarse, min_cluster_size)
    clusters = np.zeros_like(merged)
    for cid, (a, b, _) in enumerate(_runs(merged)):
        clusters[a:b] = cid
    return clusters


def segment_hdbscan(
    entropy_z: np.ndarray, min_cluster_size: int = 5
) -> np.ndarray:  # pragma: no cover - optional dependency
    """Upstream backend: HDBSCAN over ``(z(frame_index), z(entropy))``."""
    try:
        from sklearn.cluster import HDBSCAN

        clusterer = HDBSCAN(min_cluster_size=min_cluster_size)
    except ImportError:
        try:
            import hdbscan as _hdbscan

            clusterer = _hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        except ImportError as exc:
            raise ImportError(
                "the 'hdbscan' backend needs scikit-learn>=1.3 or the hdbscan "
                "package; neither is installed in lerobot-041 -- use "
                "--segmenter threshold, or run the labeller in the 'lerobot' env"
            ) from exc
    features = np.stack([zscore(np.arange(len(entropy_z))), entropy_z], axis=-1)
    clusterer.fit(features)
    return np.asarray(clusterer.labels_, dtype=np.int64)


def segment_entropy(
    entropy: np.ndarray,
    *,
    backend: str = "threshold",
    min_cluster_size: int = 5,
    fast_prefix: int = 0,
) -> SegmentationResult:
    """Entropy trace -> ``{0: precision, 1: non-precision}`` per frame.

    ``fast_prefix`` forces the first N frames to be treated as HDBSCAN noise,
    which upstream hard-codes to 50 (``initial_labels[:50] = -1``) so the
    approach phase of a 400-step Aloha episode is always replayed fast. Left at
    0 here: our episodes are ~300 frames and the approach length varies with
    where the operator started, so the entropy signal decides instead. Set it
    explicitly if the first seconds of every demo are known dead time.
    """
    entropy = np.asarray(entropy, dtype=np.float64)
    entropy_z = zscore(entropy)

    if backend == "threshold":
        clusters = segment_threshold(entropy_z, min_cluster_size)
    elif backend == "hdbscan":
        clusters = segment_hdbscan(entropy_z, min_cluster_size)
    else:
        raise ValueError(f"unknown segmenter backend: {backend!r}")

    clusters = clusters.copy()
    if fast_prefix > 0:
        clusters[:fast_prefix] = -1  # upstream: treat the approach as noise

    labels = np.full(len(entropy_z), NON_PRECISION, dtype=np.int64)
    for cid in np.unique(clusters[clusters >= 0]):
        mask = clusters == cid
        labels[mask] = _cluster_vote(entropy_z, mask)
    return SegmentationResult(
        labels=labels, entropy_z=entropy_z, raw_clusters=clusters, backend=backend
    )


__all__ = [
    "NON_PRECISION",
    "PRECISION",
    "SegmentationResult",
    "segment_entropy",
    "segment_hdbscan",
    "segment_threshold",
    "zscore",
]
