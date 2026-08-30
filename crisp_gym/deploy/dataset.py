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
