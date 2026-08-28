"""Segmentation must find the precision phase and honour upstream's cluster rule."""

import numpy as np

from conftest import two_phase_entropy
from demospeedup_core.segmentation import (
    NON_PRECISION,
    PRECISION,
    _enforce_min_run,
    segment_entropy,
    zscore,
)


def test_precision_valley_is_labelled_precision(entropy_trace):
    labels = segment_entropy(entropy_trace).labels
    n = len(entropy_trace)
    middle = labels[n // 3 + 5 : 2 * n // 3 - 5]
    ends = np.concatenate([labels[: n // 3 - 5], labels[2 * n // 3 + 5 :]])
    assert np.all(middle == PRECISION)
    assert np.all(ends == NON_PRECISION)


def test_labels_are_binary_and_frame_aligned(entropy_trace):
    result = segment_entropy(entropy_trace)
    assert result.labels.shape == entropy_trace.shape
    assert set(np.unique(result.labels)) <= {PRECISION, NON_PRECISION}
    assert np.allclose(result.entropy_z.mean(), 0.0, atol=1e-9)


def test_constant_entropy_does_not_divide_by_zero():
    result = segment_entropy(np.full(50, 3.14))
    assert np.all(result.entropy_z == 0.0)
    assert result.labels.shape == (50,)


def test_fast_prefix_forces_non_precision():
    trace = two_phase_entropy(150)
    trace[:40] = -5.0  # would otherwise be the most "precise" stretch there is
    labels = segment_entropy(trace, fast_prefix=40).labels
    assert np.all(labels[:40] == NON_PRECISION)


def test_short_runs_are_absorbed():
    noisy = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    merged = _enforce_min_run(noisy, min_run=5)
    assert np.all(merged == 0)


def test_min_cluster_size_suppresses_single_frame_flips():
    trace = np.full(60, -1.0)
    trace[30] = 5.0  # one lone high-entropy frame
    labels = segment_entropy(trace, min_cluster_size=5).labels
    assert labels[30] == PRECISION


def test_upstream_cluster_rule_needs_every_frame_above_the_mean():
    """A segment straddling the mean is precision, even if mostly above it."""
    trace = np.concatenate([np.full(40, 1.0), np.full(40, -1.0)])
    trace[10] = -0.5  # one below-mean frame inside the otherwise-fast run
    result = segment_entropy(trace, min_cluster_size=2)
    assert result.labels[10] == PRECISION


def test_zscore_matches_numpy():
    x = np.array([1.0, 2.0, 4.0, 8.0])
    assert np.allclose(zscore(x), (x - x.mean()) / x.std())
