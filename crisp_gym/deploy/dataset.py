"""Reading recorded LeRobot v3 episodes off disk.

Moved out of ``examples/17_replay_dataset.py``. Used by dataset replay and by
``--fake-mode dataset``, which drives the whole sender path from a recorded episode
with no policy and no GPU -- the cheapest way to exercise deploy end to end.
"""

import json
from pathlib import Path

import pandas as pd

# Where the LeRobot v3 datasets live (HF_LEROBOT_HOME).
LEROBOT_CACHE = Path.home() / ".cache/huggingface/lerobot"


# ---------------------------------------------------------------------------
# Dataset loading (LeRobot v3)
# ---------------------------------------------------------------------------

def load_dataset_info(dataset_dir: Path) -> dict:
    """Read meta/info.json and validate it is a v3 dataset."""
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"No info.json at {info_path}")
    info = json.loads(info_path.read_text())
    version = str(info.get("codebase_version", ""))
    if not version.startswith("v3"):
        raise ValueError(
            f"Dataset is {version}, but this script only supports v3."
        )
    return info


def load_episodes_meta(dataset_dir: Path) -> pd.DataFrame:
    """Read meta/episodes/.../file-*.parquet into a DataFrame."""
    parts = sorted((dataset_dir / "meta" / "episodes").rglob("file-*.parquet"))
    if not parts:
        raise FileNotFoundError(
            f"No episode metadata under {dataset_dir / 'meta' / 'episodes'}"
        )
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def load_episode_frames(
    dataset_dir: Path, info: dict, episodes_df: pd.DataFrame, episode_idx: int
) -> pd.DataFrame:
    """Load the per-frame parquet for one episode."""
    match = episodes_df[episodes_df["episode_index"] == episode_idx]
    if match.empty:
        available = sorted(episodes_df["episode_index"].astype(int).tolist())
        raise ValueError(
            f"episode_index {episode_idx} not found. Available: {available}"
        )
    row = match.iloc[0]
    chunk = int(row["data/chunk_index"])
    file_idx = int(row["data/file_index"])
    tmpl = info.get(
        "data_path", "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    )
    data_path = dataset_dir / tmpl.format(chunk_index=chunk, file_index=file_idx)
    df = pd.read_parquet(data_path)
    return df[df["episode_index"] == episode_idx].reset_index(drop=True)


# ---------------------------------------------------------------------------
# --fake-mode dataset helpers: turn a recorded episode into a chunk stream.
# ---------------------------------------------------------------------------

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _load_dataset_actions(repo_id: str, episode_idx: int) -> np.ndarray:
    """Load actions from a recorded episode as a (T, 7) float64 array.

    Mirrors the dataset-loading flow in 17_replay_dataset.py: read meta/info
    + meta/episodes/* via load_*  helpers, stack the action column.
    """
    dataset_dir = LEROBOT_CACHE / repo_id
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_dir}")
    info = load_dataset_info(dataset_dir)
    episodes_df = load_episodes_meta(dataset_dir)
    df = load_episode_frames(dataset_dir, info, episodes_df, episode_idx)
    if len(df) == 0:
        raise ValueError(f"Episode {episode_idx} in {repo_id} has zero frames")
    actions = np.stack(
        [np.asarray(a, dtype=np.float64) for a in df["action"].to_numpy()],
        axis=0,
    )
    if actions.ndim != 2 or actions.shape[1] < 7:
        raise ValueError(
            f"Dataset action shape {actions.shape} not (T, >=7); is this the "
            f"right episode?"
        )
    logger.info(
        "fake dataset source: loaded %d frames from %s ep %d (action_dim=%d)",
        actions.shape[0], repo_id, episode_idx, actions.shape[1],
    )
    return actions[:, :7]


def _strip_held_frames(
    actions: np.ndarray,
    *,
    motion_eps: float,
) -> np.ndarray:
    """Drop runs of held frames from a (T, >=7) recorded trajectory.

    Frame ``i`` is held when ``max(|actions[i, :7] - actions[i-1, :7]|) <=
    motion_eps`` — i.e. xyz, rpy, AND gripper all moved less than the
    threshold. The first frame of every held run is kept as an anchor
    (its predecessor was different, so the inequality fires); subsequent
    identical frames are dropped. Frame 0 is always kept.

    Including the gripper channel preserves the moments where the EE is
    stationary but the gripper opens/closes — those are not "stalls" we
    want to compress.
    """
    if actions.shape[0] <= 1:
        return actions
    deltas = np.abs(np.diff(actions[:, :7], axis=0)).max(axis=1)
    keep = np.concatenate([[True], deltas > motion_eps])
    return actions[keep]
