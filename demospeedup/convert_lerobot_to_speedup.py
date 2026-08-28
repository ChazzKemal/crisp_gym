#!/usr/bin/env python3
"""Step 3 of DemoSpeedup: write the accelerated copy of a labelled dataset.

Each episode is walked with DemoSpeedup's variable stride -- one frame every
``--low-v`` while the label says *precision*, one every ``--high-v`` while it
says *non-precision* (see :mod:`demospeedup_core.retiming`) -- and only the
frames it lands on are written out. Parquet rows are dropped and every camera
stream is re-encoded without the dropped frames, so what comes out is a stock
LeRobot v3.0 dataset: ``lerobot-train`` needs no patch to consume it.

The frame *rate* is unchanged. A retimed episode holds the same motion in 2-4x
fewer frames, so replaying it at the source fps is what makes the robot move
faster. Do not additionally raise the replay rate.

This is the structural sibling of ``/home/batur/Coding/crop_stalls.py`` (which
drops zero-motion frames); both layouts LeRobot v3 can produce -- one file per
episode and packed multi-episode files -- are handled the same way, and the
dataset is rebuilt through the LeRobot API so indices, episode metadata and
statistics stay consistent by construction.

    conda run -n lerobot-041 python convert_lerobot_to_speedup.py \\
        --src /home/batur/Coding/data/merged_act_finetune_20260528 \\
        --dst /home/batur/Coding/data/merged_speedup_20260528
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import av
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from demospeedup_core.retiming import HIGH_V, LOW_V, retiming_stats, select_keep_indices  # noqa: E402
from lerobot_bridge import episode_ranges, labels_by_episode, load_info, read_labels  # noqa: E402


class SeqVideoReader:
    """Forward-only sequential reader over a single mp4 file.

    ``take(start, n, keep_set)`` advances to absolute frame ``start``, decodes
    the next ``n`` frames and returns ``{local_index: rgb_frame}`` for the
    local indices in ``keep_set``. Successive calls must ask for non-decreasing
    ``start`` -- true because episodes are stored contiguously and in order --
    so each frame is decoded at most once and only kept frames stay in RAM.

    (Same reader as ``crop_stalls.py``; kept local so this directory is
    self-contained.)
    """

    def __init__(self, path: Path):
        self.path = path
        self._container = av.open(str(path))
        self._gen = self._container.decode(video=0)
        self._pos = 0

    def take(self, start: int, n: int, keep_set: set[int]) -> dict[int, np.ndarray]:
        if start < self._pos:
            raise RuntimeError(
                f"non-monotonic read on {self.path}: asked for frame {start} "
                f"but the reader is already at {self._pos}"
            )
        while self._pos < start:
            next(self._gen)
            self._pos += 1
        kept: dict[int, np.ndarray] = {}
        for local in range(n):
            try:
                frame = next(self._gen)
            except StopIteration as exc:
                raise RuntimeError(
                    f"{self.path}: ran out of frames at absolute index {self._pos} "
                    f"(needed up to {start + n})"
                ) from exc
            if local in keep_set:
                kept[local] = frame.to_ndarray(format="rgb24")
            self._pos += 1
        return kept

    def close(self) -> None:
        self._container.close()


def build_features(info: dict, keep_features: str | None) -> dict:
    """Feature dict for the new dataset: everything LeRobot does not manage itself."""
    from lerobot.datasets.utils import DEFAULT_FEATURES

    features: dict = {}
    for key, ft in info["features"].items():
        if key in DEFAULT_FEATURES:
            continue
        features[key] = {
            "dtype": ft["dtype"],
            "shape": tuple(ft["shape"]),
            "names": ft.get("names", ["height", "width", "channels"])
            if ft["dtype"] == "video"
            else ft.get("names"),
        }
    if keep_features:
        wanted = {k.strip() for k in keep_features.split(",") if k.strip()}
        missing = wanted - set(features)
        if missing:
            raise SystemExit(
                f"--keep-features: unknown feature(s) {sorted(missing)}; "
                f"available: {sorted(features)}"
            )
        features = {k: v for k, v in features.items() if k in wanted}
    return features


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--src", type=Path, required=True, help="labelled LeRobot v3 dataset")
    ap.add_argument("--dst", type=Path, required=True, help="destination dir (must not exist)")
    ap.add_argument("--low-v", type=int, default=None,
                    help="stride in precision phases (default: the value used at labelling)")
    ap.add_argument("--high-v", type=int, default=None,
                    help="stride in non-precision phases (default: as labelled)")
    ap.add_argument("--no-keep-last", action="store_true",
                    help="do not force-keep each episode's final frame")
    ap.add_argument("--limit-episodes", type=int, default=None, help="smoke test on the first N")
    ap.add_argument("--keep-features", default=None,
                    help="comma-separated feature keys to carry over (default: all). "
                         "LeRobot treats every 'observation.*' key as a policy input, so "
                         "drop bookkeeping ones like observation.timestamps.* here.")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    src = args.src.resolve()
    dst = args.dst.resolve()
    if dst.exists():
        raise SystemExit(f"destination already exists, refusing to overwrite: {dst}")

    info = load_info(src)
    fps = int(info["fps"])
    frame_labels, label_config = read_labels(src)
    labels = labels_by_episode(frame_labels)
    low_v = args.low_v if args.low_v is not None else int(label_config.get("low_v", LOW_V))
    high_v = args.high_v if args.high_v is not None else int(label_config.get("high_v", HIGH_V))
    keep_last = not args.no_keep_last

    features = build_features(info, args.keep_features)
    video_keys = [k for k, v in features.items() if v["dtype"] == "video"]
    scalar_keys = [k for k in features if k not in video_keys]

    episodes = episode_ranges(src, args.limit_episodes)
    unlabelled = [e.episode_index for e in episodes if e.episode_index not in labels]
    if unlabelled:
        raise SystemExit(
            f"episodes {unlabelled[:5]}{'...' if len(unlabelled) > 5 else ''} have no "
            f"labels in {src}; re-run label_entropy.py without --episodes"
        )

    print(f"source : {src}")
    print(f"labels : {label_config.get('policy_path', '?')} "
          f"(segmenter={label_config.get('segmenter', '?')})")
    print(f"stride : {low_v}x in precision, {high_v}x in non-precision phases")

    out = LeRobotDataset.create(
        repo_id=dst.name,
        fps=fps,
        features=features,
        root=dst,
        robot_type=info.get("robot_type"),
        use_videos=bool(video_keys),
    )
    # One parquet + one mp4 per episode, as crop_stalls.py does: a sub-kilobyte
    # size cap is below any single episode, so every save_episode() flushes to a
    # fresh file. The cosmetic size fields are restored at the end.
    out.meta.update_chunk_settings(data_files_size_in_mb=1e-3, video_files_size_in_mb=1e-3)

    # Base absolute frame index of each video file: the smallest
    # dataset_from_index among the episodes sharing it.
    video_file_base: dict = {}
    for vkey in video_keys:
        for ep in episodes:
            fkey = (vkey, *ep.video_ids(vkey))
            video_file_base[fkey] = min(
                video_file_base.get(fkey, ep.dataset_from_index), ep.dataset_from_index
            )

    readers: dict[str, SeqVideoReader] = {}
    reader_paths: dict[str, Path] = {}
    data_cache: dict[Path, pd.DataFrame] = {}

    grand_total = grand_kept = 0
    worst_gap = 0
    for ep in episodes:
        data_path = ep.data_path(src, info["data_path"])
        if data_path not in data_cache:
            data_cache[data_path] = pd.read_parquet(data_path)
        full = data_cache[data_path]
        df = (
            full[full["episode_index"] == ep.episode_index]
            .sort_values("frame_index")
            .reset_index(drop=True)
        )
        if len(df) != ep.length:
            raise SystemExit(
                f"ep {ep.episode_index}: parquet yields {len(df)} rows, metadata says {ep.length}"
            )
        ep_labels = labels[ep.episode_index]
        if len(ep_labels) != ep.length:
            raise SystemExit(
                f"ep {ep.episode_index}: {len(ep_labels)} labels for {ep.length} frames -- "
                f"the sidecar was written for a different dataset"
            )

        keep_idx = select_keep_indices(ep_labels, low_v, high_v, keep_last=keep_last)
        keep_set = {int(k) for k in keep_idx}
        stats = retiming_stats(ep_labels, keep_idx)
        worst_gap = max(worst_gap, stats.max_gap)

        vid_kept: dict[str, dict[int, np.ndarray]] = {}
        for vkey in video_keys:
            vchunk, vfile = ep.video_ids(vkey)
            vpath = src / info["video_path"].format(
                video_key=vkey, chunk_index=vchunk, file_index=vfile
            )
            if reader_paths.get(vkey) != vpath:
                if vkey in readers:
                    readers[vkey].close()
                readers[vkey] = SeqVideoReader(vpath)
                reader_paths[vkey] = vpath
            base = video_file_base[(vkey, vchunk, vfile)]
            kept = readers[vkey].take(ep.dataset_from_index - base, len(df), keep_set)
            if len(kept) != len(keep_idx):
                raise SystemExit(
                    f"ep {ep.episode_index}: video {vkey} yielded {len(kept)} kept frames, "
                    f"expected {len(keep_idx)} -- frame/row alignment broken"
                )
            vid_kept[vkey] = kept

        for k in keep_idx:
            frame: dict = {"task": ep.task}
            for vkey in video_keys:
                frame[vkey] = vid_kept[vkey][int(k)]
            for col in scalar_keys:
                value = np.asarray(df[col].iloc[k], dtype=np.dtype(features[col]["dtype"]))
                frame[col] = value.reshape(1) if value.ndim == 0 else value
            # timestamp is deliberately omitted: LeRobot re-derives it as
            # frame_index / fps, which is exactly the accelerated timeline.
            out.add_frame(frame)
        out.save_episode()

        grand_total += ep.length
        grand_kept += len(keep_idx)
        print(f"ep {ep.episode_index:3d}: {ep.length:5d} -> {len(keep_idx):5d} frames "
              f"({stats.speedup:.2f}x, fast {stats.fast_fraction * 100:.0f}%, "
              f"max gap {stats.max_gap})", flush=True)

    for reader in readers.values():
        reader.close()

    out_info_path = dst / "meta" / "info.json"
    out_info = json.loads(out_info_path.read_text())
    out_info["data_files_size_in_mb"] = info.get("data_files_size_in_mb", 100)
    out_info["video_files_size_in_mb"] = info.get("video_files_size_in_mb", 500)
    out_info_path.write_text(json.dumps(out_info, indent=4))

    sidecar = {
        "source_dataset": str(src),
        "method": "demospeedup_entropy_guided_retiming",
        "upstream": "https://github.com/lingxiao-guo/DemoSpeedup",
        "low_v": low_v,
        "high_v": high_v,
        "keep_last": keep_last,
        "fps": fps,
        "n_episodes": len(episodes),
        "source_frames": grand_total,
        "kept_frames": grand_kept,
        "speedup": grand_total / max(grand_kept, 1),
        "max_frame_gap": worst_gap,
        "label_config": label_config,
    }
    (dst / "meta" / "demospeedup_source.json").write_text(json.dumps(sidecar, indent=2))

    print(f"\n{len(episodes)} episodes, {grand_total} -> {grand_kept} frames "
          f"({sidecar['speedup']:.2f}x average speedup, largest source-frame gap {worst_gap}).")
    print(f"Accelerated dataset written to: {dst}")
    print("Replay it at the SOURCE fps -- the acceleration is baked into the action deltas.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
