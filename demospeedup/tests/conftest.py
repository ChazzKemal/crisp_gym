"""Shared fixtures for the DemoSpeedup port's test suite."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def two_phase_entropy(n_frames: int = 200, seed: int = 0) -> np.ndarray:
    """A demo that is confidently fast at the ends and uncertain in the middle.

    Mimics what a real trace looks like: high entropy while the arm flies
    through free space, a low-entropy valley around the grasp.
    """
    rng = np.random.default_rng(seed)
    e = np.full(n_frames, 2.0)
    e[n_frames // 3 : 2 * n_frames // 3] = -1.0
    return e + rng.normal(scale=0.05, size=n_frames)


@pytest.fixture
def entropy_trace():
    return two_phase_entropy()


@pytest.fixture
def real_labels():
    """Labels from a real labelling run, if one is on disk."""
    root = Path("/home/batur/Coding/data/merged_act_finetune_20260528")
    labels = root / "meta" / "demospeedup" / "labels.parquet"
    if not labels.exists():
        pytest.skip(f"no labelling run at {labels}")
    import pandas as pd

    return pd.read_parquet(labels)
