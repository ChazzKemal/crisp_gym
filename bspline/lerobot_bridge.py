"""Read a LeRobot v3.0 dataset and turn its actions into B-spline chunks.

The upstream B-spline Policy repo ingests robomimic HDF5 through
``diffusion_policy``'s zarr ReplayBuffer. We skip that entirely: LeRobot
parquet -> numpy -> the vendored chunker in :mod:`bspline_core`. The *action
representation* is identical; only the storage layer differs.

Action layout
-------------
crisp_gym cart7 datasets store ``action = [x, y, z, rx, ry, rz, gripper]``
with an axis-angle rotation -- exactly the raw layout the upstream YAM
single-arm config expects, so the same ``7 -> 10`` (rot6d) conversion applies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from bspline_core.chunk_sampler import BSplineChunkSampler
from bspline_core.rotation import convert_actions_7d_to_10d

RAW_ACTION_NAMES = ["x", "y", "z", "rx", "ry", "rz", "gripper"]
POLICY_ACTION_NAMES = [
    "x", "y", "z",
    "rot6d_0", "rot6d_1", "rot6d_2", "rot6d_3", "rot6d_4", "rot6d_5",
    "gripper",
]


@dataclass
class LeRobotActions:
    """Episode-concatenated actions plus the episode boundaries."""

    actions: np.ndarray  # (N, raw_dim)
    episode_ends: np.ndarray  # (n_episodes,) exclusive cumulative ends
    episode_indices: np.ndarray  # (n_episodes,) original LeRobot episode ids
    fps: float
    info: dict

    @property
    def episode_starts(self) -> np.ndarray:
        return np.concatenate([[0], self.episode_ends[:-1]])

    def episode(self, i: int) -> np.ndarray:
        return self.actions[self.episode_starts[i] : self.episode_ends[i]]


def load_lerobot_actions(root: str | Path, action_key: str = "action") -> LeRobotActions:
    """Load every episode's action array from a LeRobot v3.0 dataset on disk."""
    root = Path(root)
    info = json.loads((root / "meta" / "info.json").read_text())
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"expected LeRobot v3.0, got {info.get('codebase_version')}")

    ep_meta = pd.concat(
        [pd.read_parquet(p) for p in sorted((root / "meta" / "episodes").rglob("*.parquet"))],
        ignore_index=True,
    ).sort_values("episode_index")

    frames = pd.concat(
        [pd.read_parquet(p) for p in sorted((root / "data").rglob("*.parquet"))],
        ignore_index=True,
    ).sort_values(["episode_index", "frame_index"])

    chunks, ends, ep_ids, total = [], [], [], 0
    for ep_id, group in frames.groupby("episode_index", sort=True):
        arr = np.stack(group[action_key].to_numpy()).astype(np.float32)
        expected = int(ep_meta.loc[ep_meta.episode_index == ep_id, "length"].iloc[0])
        if len(arr) != expected:
            raise ValueError(f"episode {ep_id}: {len(arr)} frames, meta says {expected}")
        chunks.append(arr)
        total += len(arr)
        ends.append(total)
        ep_ids.append(int(ep_id))

    return LeRobotActions(
        actions=np.concatenate(chunks, axis=0),
        episode_ends=np.asarray(ends, dtype=np.int64),
        episode_indices=np.asarray(ep_ids, dtype=np.int64),
        fps=float(info["fps"]),
        info=info,
    )


def to_policy_actions(raw: np.ndarray) -> np.ndarray:
    """``(N, 7)`` axis-angle -> ``(N, 10)`` rot6d, as upstream does."""
    return convert_actions_7d_to_10d(raw)


def build_sampler(
    data: LeRobotActions,
    chunk_size: int = 10,
    degree: int = 3,
    max_error: float = 0.002,
    stride: int = 1,
    relative_knots: bool = False,
    max_first_k: int = 2,
    convert_rotation: bool = True,
    verbose: bool = False,
) -> BSplineChunkSampler:
    """Fit and chunk every episode of a loaded LeRobot dataset."""
    actions = to_policy_actions(data.actions) if convert_rotation else data.actions
    return BSplineChunkSampler(
        actions=actions,
        episode_ends=data.episode_ends,
        chunk_size=chunk_size,
        degree=degree,
        max_error=max_error,
        stride=stride,
        relative_knots=relative_knots,
        max_first_k=max_first_k,
        verbose=verbose,
    )


__all__ = [
    "LeRobotActions",
    "POLICY_ACTION_NAMES",
    "RAW_ACTION_NAMES",
    "build_sampler",
    "load_lerobot_actions",
    "to_policy_actions",
]
