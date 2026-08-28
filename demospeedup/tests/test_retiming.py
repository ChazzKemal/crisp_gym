"""The stride walk must match upstream and stay executable on the robot."""

import numpy as np
import pytest

from demospeedup_core.retiming import (
    HIGH_V,
    LOW_V,
    process_action_label_upstream,
    retiming_stats,
    select_keep_indices,
)


def test_keep_indices_follow_upstream_walk(rng):
    labels = rng.integers(0, 2, size=300)
    walk = process_action_label_upstream(labels, LOW_V, HIGH_V, start=0)
    assert select_keep_indices(labels, keep_last=False).tolist()[1:] == walk


def test_all_precision_gives_stride_two():
    labels = np.zeros(101, dtype=np.int64)
    keep = select_keep_indices(labels, keep_last=False)
    assert np.all(np.diff(keep) == LOW_V)


def test_all_non_precision_gives_stride_four():
    labels = np.ones(101, dtype=np.int64)
    keep = select_keep_indices(labels, keep_last=False)
    assert np.all(np.diff(keep) == HIGH_V)


@pytest.mark.parametrize("seed", range(8))
def test_gaps_never_exceed_high_v(seed):
    """A gap wider than ``high_v`` would be a jump the controller never saw."""
    labels = np.random.default_rng(seed).integers(0, 2, size=257)
    keep = select_keep_indices(labels)
    assert np.diff(keep).max() <= HIGH_V


@pytest.mark.parametrize("seed", range(8))
def test_indices_are_sorted_unique_and_in_range(seed):
    labels = np.random.default_rng(seed).integers(0, 2, size=193)
    keep = select_keep_indices(labels)
    assert keep[0] == 0
    assert keep[-1] == len(labels) - 1  # keep_last
    assert np.all(np.diff(keep) > 0)
    assert keep.max() < len(labels)


def test_speedup_is_between_the_two_strides(rng):
    labels = rng.integers(0, 2, size=400)
    keep = select_keep_indices(labels)
    stats = retiming_stats(labels, keep)
    assert LOW_V - 0.5 <= stats.speedup <= HIGH_V + 0.5


def test_precision_phases_are_never_skipped_faster_than_low_v():
    """Frames inside a precision run must be sampled at ``low_v``, not ``high_v``."""
    labels = np.concatenate([np.ones(40), np.zeros(40), np.ones(40)]).astype(np.int64)
    keep = select_keep_indices(labels, keep_last=False)
    inside = keep[(keep >= 41) & (keep < 79)]
    assert np.all(np.diff(inside) <= LOW_V)


def test_empty_episode():
    assert len(select_keep_indices(np.zeros(0, dtype=np.int64))) == 0
