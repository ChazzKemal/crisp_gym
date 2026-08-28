"""Shared fixtures and synthetic trajectories for the B-spline test suite."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def smooth_trajectory(n_frames: int = 200, n_dims: int = 10, seed: int = 0) -> np.ndarray:
    """A smooth, band-limited multi-dim trajectory -- easy for a cubic spline."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n_frames)
    out = np.zeros((n_frames, n_dims))
    for d in range(n_dims):
        for k in range(1, 4):
            out[:, d] += rng.normal() / k * np.sin(2 * np.pi * k * t + rng.uniform(0, 2 * np.pi))
    return out


def steppy_trajectory(n_frames: int = 200, seed: int = 0) -> np.ndarray:
    """Smooth pose dims plus a binary gripper channel -- our real action shape."""
    traj = smooth_trajectory(n_frames, 9, seed)
    gripper = np.zeros(n_frames)
    gripper[n_frames // 3 : 2 * n_frames // 3] = 1.0
    return np.concatenate([traj, gripper[:, None]], axis=1)


@pytest.fixture
def smooth_traj():
    return smooth_trajectory()


@pytest.fixture
def real_actions_7d():
    """A short slice of the real merged dataset, if it is on disk."""
    root = Path("/home/batur/Coding/data/merged_act_finetune_20260528")
    parquet = root / "data" / "chunk-000" / "file-000.parquet"
    if not parquet.exists():
        pytest.skip(f"real dataset not available at {root}")
    import pandas as pd

    df = pd.read_parquet(parquet)
    return np.stack(df["action"].to_numpy()).astype(np.float32)
