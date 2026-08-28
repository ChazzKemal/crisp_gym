"""Label-driven variable-stride retiming -- the actual "speedup" of DemoSpeedup.

Upstream implements this inside the dataset's ``__getitem__``
(``act_utils.py::process_action_label``): the observation at frame ``t`` stays
put, but the action chunk it is trained against is subsampled -- one waypoint
every ``low_v = 2`` frames while the label says *precision*, one every
``high_v = 4`` while it says *non-precision*. Executed at the original control
rate, such a chunk moves the arm 2x / 4x faster.

We apply exactly the same stride walk, but *to the episode itself*: the frames
it lands on are the frames the accelerated dataset keeps
(:mod:`convert_lerobot_to_speedup`). The resulting (observation, action-chunk)
pairs are the same ones upstream trains on -- consecutive frames of the
retimed episode are precisely the subsampled waypoints -- with two practical
differences:

* dropped frames no longer serve as chunk *starts*, so the accelerated dataset
  has 2-4x fewer training samples (upstream keeps all of them);
* what comes out is a stock LeRobot v3.0 dataset, trainable with the same
  ``lerobot-train`` invocation as every other run in this repo, and replayable
  with the existing crisp_gym replay tooling at the unchanged fps.

The acceleration lives in the action deltas, not in the metadata: **keep the
replay/control rate at the source fps**, or the speedup is applied twice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LOW_V = 2  # stride inside precision (label 0) segments
HIGH_V = 4  # stride inside non-precision (label 1) segments


def process_action_label_upstream(
    labels: np.ndarray, low_v: int = LOW_V, high_v: int = HIGH_V, start: int = -1
) -> list[int]:
    """Verbatim numpy port of upstream's ``process_action_label`` index walk.

    ``start = -1`` reproduces upstream exactly (their loop begins at ``i = -1``,
    so the first label consulted is ``labels[-1]``, the *last* frame of the
    chunk -- almost certainly a slip, but it only perturbs the first step).
    Pass ``start = 0`` for the sane variant used by :func:`select_keep_indices`.
    """
    labels = np.asarray(labels)
    horizon = len(labels)
    indices: list[int] = []
    i = start
    while i < horizon:
        if labels[i] == 0 and i + low_v < horizon:
            i += low_v
            indices.append(i)
        elif labels[i] == 1:
            if i + high_v < horizon and np.all(labels[i : i + high_v] == 1):
                i += high_v
                indices.append(i)
            else:
                next_zero = np.flatnonzero(labels[i + 1 :] == 0)
                if len(next_zero) > 0:
                    i = i + 1 + int(next_zero[0])
                    indices.append(i)
                else:
                    break
        else:
            i += 1
    return indices


@dataclass
class RetimingStats:
    n_source: int
    n_kept: int
    max_gap: int
    fast_fraction: float

    @property
    def speedup(self) -> float:
        return self.n_source / max(self.n_kept, 1)


def select_keep_indices(
    labels: np.ndarray,
    low_v: int = LOW_V,
    high_v: int = HIGH_V,
    keep_last: bool = True,
) -> np.ndarray:
    """Frames of one episode to keep, given its per-frame labels.

    Frame 0 is always kept as the episode anchor, then upstream's stride walk
    picks the rest. With ``keep_last`` the final frame is appended too (the
    walk stops ``high_v`` frames short of the end, and dropping the last pose
    of a demonstration would truncate the task); the tail is filled at stride
    ``high_v`` so no gap ever exceeds it.
    """
    labels = np.asarray(labels)
    n = len(labels)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    keep = [0, *process_action_label_upstream(labels, low_v, high_v, start=0)]
    if keep_last and keep[-1] != n - 1:
        keep.extend(range(keep[-1] + high_v, n - 1, high_v))
        keep.append(n - 1)
    out = np.array(sorted(set(int(k) for k in keep)), dtype=np.int64)
    return out[(out >= 0) & (out < n)]


def retiming_stats(labels: np.ndarray, keep: np.ndarray) -> RetimingStats:
    labels = np.asarray(labels)
    gaps = np.diff(keep) if len(keep) > 1 else np.zeros(1, dtype=np.int64)
    return RetimingStats(
        n_source=len(labels),
        n_kept=len(keep),
        max_gap=int(gaps.max()),
        fast_fraction=float(np.mean(labels == 1)) if len(labels) else 0.0,
    )


__all__ = [
    "HIGH_V",
    "LOW_V",
    "RetimingStats",
    "process_action_label_upstream",
    "retiming_stats",
    "select_keep_indices",
]
