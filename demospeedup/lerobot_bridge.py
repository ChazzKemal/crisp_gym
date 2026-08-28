"""LeRobot v3.0 plumbing for the DemoSpeedup pipeline.

Upstream stores its entropy trace and its binary labels straight back into the
demonstration's HDF5 (``root["/entropy"]``, ``root["/labels"]``). LeRobot v3.0
datasets are parquet + mp4 with checksummed episode metadata, so instead of
rewriting recorded data we drop a **sidecar** next to it::

    <dataset>/meta/demospeedup/labels.parquet   episode_index, frame_index,
                                                entropy, entropy_z, label
    <dataset>/meta/demospeedup.json             how those labels were produced

The source dataset stays byte-identical, several labelling runs can be
compared, and ``convert_lerobot_to_speedup.py`` reads the sidecar back.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LABELS_RELPATH = Path("meta") / "demospeedup" / "labels.parquet"
CONFIG_RELPATH = Path("meta") / "demospeedup.json"
LABEL_COLUMNS = ["episode_index", "frame_index", "entropy", "entropy_z", "label"]


@dataclass
class EpisodeRange:
    """One episode's row range plus the files its data and video live in."""

    episode_index: int
    dataset_from_index: int
    dataset_to_index: int
    length: int
    task: str
    row: pd.Series

    def data_path(self, root: Path, template: str) -> Path:
        return root / template.format(
            chunk_index=int(self.row["data/chunk_index"]),
            file_index=int(self.row["data/file_index"]),
        )

    def video_ids(self, video_key: str) -> tuple[int, int]:
        return (
            int(self.row[f"videos/{video_key}/chunk_index"]),
            int(self.row[f"videos/{video_key}/file_index"]),
        )


def load_info(root: str | Path) -> dict:
    root = Path(root)
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"not a LeRobot dataset: {root}")
    info = json.loads(info_path.read_text())
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"expected LeRobot v3.0, got {info.get('codebase_version')}")
    return info


def episode_table(root: str | Path) -> pd.DataFrame:
    """``meta/episodes/*.parquet`` concatenated, indexed by ``episode_index``."""
    root = Path(root)
    parts = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no episode metadata under {root / 'meta' / 'episodes'}")
    return (
        pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        .set_index("episode_index")
        .sort_index()
    )


def episode_ranges(root: str | Path, limit: int | None = None) -> list[EpisodeRange]:
    table = episode_table(root)
    out = []
    for episode_index in table.index[: limit if limit is not None else len(table)]:
        row = table.loc[episode_index]
        tasks = row.get("tasks", [])
        out.append(
            EpisodeRange(
                episode_index=int(episode_index),
                dataset_from_index=int(row["dataset_from_index"]),
                dataset_to_index=int(row["dataset_to_index"]),
                length=int(row["length"]),
                task=str(tasks[0]) if len(tasks) else "",
                row=row,
            )
        )
    return out


def episode_actions(root: str | Path, ep: EpisodeRange, info: dict) -> np.ndarray:
    """This episode's ``action`` rows, frame-ordered."""
    df = pd.read_parquet(ep.data_path(Path(root), info["data_path"]))
    df = df[df["episode_index"] == ep.episode_index].sort_values("frame_index")
    return np.stack(df["action"].to_numpy()).astype(np.float32)


def write_labels(
    root: str | Path, frame_labels: pd.DataFrame, config: dict
) -> tuple[Path, Path]:
    """Write the label sidecar and the run configuration next to the dataset."""
    root = Path(root)
    missing = set(LABEL_COLUMNS) - set(frame_labels.columns)
    if missing:
        raise ValueError(f"label frame is missing columns: {sorted(missing)}")
    labels_path = root / LABELS_RELPATH
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    frame_labels[LABEL_COLUMNS].to_parquet(labels_path, index=False)
    config_path = root / CONFIG_RELPATH
    config_path.write_text(json.dumps(config, indent=2))
    return labels_path, config_path


def read_labels(root: str | Path) -> tuple[pd.DataFrame, dict]:
    root = Path(root)
    labels_path = root / LABELS_RELPATH
    if not labels_path.exists():
        raise FileNotFoundError(
            f"{labels_path} not found -- run label_entropy.py on this dataset first"
        )
    config_path = root / CONFIG_RELPATH
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    return pd.read_parquet(labels_path), config


def labels_by_episode(frame_labels: pd.DataFrame) -> dict[int, np.ndarray]:
    """``{episode_index: (T,) int labels}``, frame-ordered."""
    out = {}
    for episode_index, group in frame_labels.groupby("episode_index", sort=True):
        ordered = group.sort_values("frame_index")
        out[int(episode_index)] = ordered["label"].to_numpy().astype(np.int64)
    return out


__all__ = [
    "CONFIG_RELPATH",
    "LABELS_RELPATH",
    "LABEL_COLUMNS",
    "EpisodeRange",
    "episode_actions",
    "episode_ranges",
    "episode_table",
    "labels_by_episode",
    "load_info",
    "read_labels",
    "write_labels",
]
