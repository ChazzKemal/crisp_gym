#!/usr/bin/env python3
"""LeRobot v3 dataset viewer — browse, watch, and time-trim multi-camera datasets.

A single-file Tk GUI for inspecting HuggingFace LeRobot v3.0 datasets:

  * Pick a HuggingFace cache folder and choose a dataset from it.
  * Pick an episode and watch every camera at the same time, synchronised.
  * Transport controls: play/pause, loop, speed, frame step, scrub slider.
  * **Time-trim**: select a frame range on the timeline strip and press "Crop"
    to delete those frames. Repeat as many times as you like; undo / reset are
    available. The camera videos, the observation/state columns and the action
    column are all trimmed together so every modality stays aligned.
  * Save the trimmed result as a brand-new dataset (one button) — re-authored
    through LeRobot's own writer, so videos, parquet, metadata and statistics
    are all regenerated and consistent.
  * Also: save the current frames as PNGs, export an episode as a composite MP4,
    a per-frame state/action inspector and a dataset metadata panel.

Run inside the ``lerobot`` pixi env (it has av, cv2, tkinter, PIL, lerobot):

    cd Yunfei/crisp_gym
    pixi run -e lerobot python examples/28_lerobot_dataset_viewer.py
    pixi run -e lerobot python examples/28_lerobot_dataset_viewer.py ~/.cache/huggingface/lerobot
    pixi run -e lerobot python examples/28_lerobot_dataset_viewer.py --selftest <dataset_dir>

Reading is done by parsing the v3 layout directly (no lerobot import needed);
the ``lerobot`` package is imported lazily, only when saving a trimmed copy.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import queue
import shutil
import sys
import threading
import time
import traceback
from pathlib import Path

import av  # PyAV — frame-accurate AV1 decoding
import cv2
import numpy as np
import pandas as pd

# Bookkeeping columns LeRobot manages itself — never user-facing data.
_BOOKKEEPING = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
JPEG_QUALITY = 92
_NP_DTYPES = {
    "float16": np.float16, "float32": np.float32, "float64": np.float64,
    "int16": np.int16, "int32": np.int32, "int64": np.int64,
    "uint8": np.uint8, "bool": bool,
}


# ======================================================================
#  Data layer — LeRobot v3.0 dataset parsing
# ======================================================================
class DatasetError(Exception):
    """Raised when a folder is not a usable LeRobot v3 dataset."""


def is_dataset_dir(path: Path) -> bool:
    """True if *path* looks like a LeRobot dataset root (has meta/info.json)."""
    return (Path(path) / "meta" / "info.json").is_file()


def find_datasets(folder: Path) -> list[Path]:
    """Return dataset roots inside *folder* (or [folder] if it is one itself)."""
    folder = Path(folder)
    if is_dataset_dir(folder):
        return [folder]
    out: list[Path] = []
    if not folder.is_dir():
        return out
    for child in sorted(folder.iterdir()):
        if child.is_dir() and is_dataset_dir(child):
            out.append(child)
    return out


class LeRobotDatasetV3:
    """Read-only accessor for a LeRobot v3.0 dataset on disk."""

    def __init__(self, root: Path):
        self.root = Path(root)
        info_path = self.root / "meta" / "info.json"
        if not info_path.is_file():
            raise DatasetError(f"Not a LeRobot dataset (missing meta/info.json):\n{self.root}")
        self.info: dict = json.loads(info_path.read_text())

        version = str(self.info.get("codebase_version", "")).lower().lstrip("v")
        if not version.startswith("3"):
            raise DatasetError(
                f"Unsupported codebase_version '{self.info.get('codebase_version')}'.\n"
                "This viewer supports LeRobot v3.x datasets only."
            )

        self.fps = float(self.info.get("fps", 30) or 30)
        self.features: dict = self.info.get("features", {})
        self.camera_keys = [k for k, v in self.features.items() if v.get("dtype") == "video"]
        self.video_path_tmpl = self.info["video_path"]
        self.data_path_tmpl = self.info["data_path"]

        ep_files = sorted((self.root / "meta" / "episodes").glob("**/*.parquet"))
        if not ep_files:
            raise DatasetError("No meta/episodes/*.parquet files found.")
        self.episodes = (
            pd.concat([pd.read_parquet(f) for f in ep_files], ignore_index=True)
            .sort_values("episode_index")
            .reset_index(drop=True)
        )
        self._data_cache: dict[Path, pd.DataFrame] = {}

    # -- basic queries --------------------------------------------------
    @property
    def name(self) -> str:
        return self.root.name

    @property
    def n_episodes(self) -> int:
        return len(self.episodes)

    def episode_row(self, ep: int) -> pd.Series:
        match = self.episodes[self.episodes["episode_index"] == int(ep)]
        if match.empty:
            raise DatasetError(f"Episode {ep} not found.")
        return match.iloc[0]

    def episode_length(self, ep: int) -> int:
        return int(self.episode_row(ep)["length"])

    def episode_tasks(self, ep: int) -> list[str]:
        try:
            return [str(t) for t in list(self.episode_row(ep)["tasks"])]
        except Exception:
            return []

    def camera_size(self, cam: str) -> tuple[int, int] | None:
        """Return (width, height) for *cam* from info.json, or None."""
        feat = self.features.get(cam, {})
        meta = feat.get("info", {})
        w, h = meta.get("video.width"), meta.get("video.height")
        if w and h:
            return int(w), int(h)
        shape = feat.get("shape")  # [height, width, channels]
        if shape and len(shape) >= 2:
            return int(shape[1]), int(shape[0])
        return None

    def camera_codec(self, cam: str) -> str:
        return str(self.features.get(cam, {}).get("info", {}).get("video.codec", "?"))

    # -- file resolution ------------------------------------------------
    def video_file(self, ep: int, cam: str) -> Path:
        row = self.episode_row(ep)
        rel = self.video_path_tmpl.format(
            video_key=cam,
            chunk_index=int(row[f"videos/{cam}/chunk_index"]),
            file_index=int(row[f"videos/{cam}/file_index"]),
        )
        return self.root / rel

    def episode_segment(self, ep: int, cam: str) -> tuple[Path, float, float, int]:
        """Return (video_path, from_timestamp, to_timestamp, n_frames)."""
        row = self.episode_row(ep)
        path = self.video_file(ep, cam)
        from_ts = float(row[f"videos/{cam}/from_timestamp"])
        to_ts = float(row[f"videos/{cam}/to_timestamp"])
        return path, from_ts, to_ts, int(row["length"])

    def episode_data(self, ep: int) -> pd.DataFrame:
        """Return the per-frame data rows for *ep* (state/action/...)."""
        row = self.episode_row(ep)
        rel = self.data_path_tmpl.format(
            chunk_index=int(row["data/chunk_index"]),
            file_index=int(row["data/file_index"]),
        )
        path = self.root / rel
        if path not in self._data_cache:
            self._data_cache[path] = pd.read_parquet(path)
        df = self._data_cache[path]
        return df[df["episode_index"] == int(row["episode_index"])].reset_index(drop=True)


# ======================================================================
#  Video decoding (PyAV) — episode = a [from_ts, to_ts] slice of an mp4
# ======================================================================
def decode_segment(path, from_ts, to_ts, n_frames, fps, progress=None):
    """Decode the frames of one episode from a (possibly concatenated) mp4.

    Seeks to the keyframe at/before *from_ts*, decodes forward, and keeps the
    first *n_frames* frames at/after *from_ts*. Returns a list of RGB uint8
    ndarrays. *progress* (if given) is called with a 0..1 fraction.
    """
    frames: list[np.ndarray] = []
    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        tb = stream.time_base
        tol = 0.5 / fps if fps else 0.02
        if from_ts and from_ts > 0:
            container.seek(int(round(from_ts / tb)), stream=stream,
                           backward=True, any_frame=False)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            t = float(frame.pts * tb)
            if t < from_ts - tol:
                continue
            if to_ts and t > to_ts + tol:
                break
            frames.append(frame.to_ndarray(format="rgb24"))
            if progress and n_frames:
                progress(min(1.0, len(frames) / n_frames))
            if n_frames and len(frames) >= n_frames:
                break
    finally:
        container.close()
    return frames


def jpeg_encode(rgb: np.ndarray) -> bytes:
    """Compress an RGB frame to JPEG bytes (compact in-memory frame cache)."""
    ok, buf = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def jpeg_decode(data: bytes) -> np.ndarray:
    """Inverse of :func:`jpeg_encode` — returns an RGB uint8 ndarray."""
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


# ======================================================================
#  Saving a time-trimmed copy (re-authored via LeRobot's own writer)
# ======================================================================
def build_create_features(info: dict) -> dict:
    """info.json 'features' minus the bookkeeping keys LeRobot adds itself."""
    out: dict = {}
    for key, feat in info.get("features", {}).items():
        if key in _BOOKKEEPING:
            continue
        out[key] = {
            "dtype": feat["dtype"],
            "shape": tuple(feat["shape"]),
            "names": feat.get("names"),
        }
    return out


def save_trimmed_copy(ds: LeRobotDatasetV3, edits: dict, dest: Path, repo_id: str,
                      cancel: threading.Event, progress=None, log=None):
    """Re-author *ds* into a new v3 dataset at *dest*, applying per-episode trims.

    *edits* maps episode_index -> sorted list of kept ORIGINAL frame indices.
    Episodes absent from *edits* are written in full. The new dataset is built
    with LeRobot's own writer so video / parquet / metadata / statistics are
    regenerated and consistent. Honours *cancel*; raises on failure.
    """
    def emit(frac, text):
        if progress:
            progress(frac, text)

    def note(msg):
        if log:
            log(msg)

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except Exception as exc:  # noqa: BLE001
        raise DatasetError(
            "Saving a trimmed copy needs the 'lerobot' package.\n"
            f"Import failed: {exc}\n"
            "Run the viewer inside the 'lerobot' pixi env."
        ) from exc

    dest = Path(dest)
    if dest.exists():
        raise DatasetError(f"Destination already exists:\n{dest}")

    features = build_create_features(ds.info)
    cams = set(ds.camera_keys)
    has_task = "task" in inspect.signature(LeRobotDataset.add_frame).parameters

    # Resolve the keep-list of every episode up front (for total / progress).
    plan: dict[int, list[int]] = {}
    for ep in range(ds.n_episodes):
        keep = edits.get(ep)
        plan[ep] = list(range(ds.episode_length(ep))) if keep is None else list(keep)
    total = max(1, sum(len(v) for v in plan.values()))

    note(f"Creating dataset at {dest}")
    new = LeRobotDataset.create(
        repo_id=repo_id, fps=int(round(ds.fps)), features=features,
        robot_type=ds.info.get("robot_type") or "unknown",
        use_videos=True, root=str(dest),
    )

    done = 0
    try:
        for ep in range(ds.n_episodes):
            if cancel.is_set():
                raise DatasetError("Cancelled.")
            keep = plan[ep]
            if not keep:
                note(f"episode {ep}: all frames removed — skipped")
                continue

            emit(done / total, f"episode {ep}/{ds.n_episodes - 1}: decoding video…")
            data = ds.episode_data(ep)
            task = (ds.episode_tasks(ep) or [""])[0]
            cam_frames: dict[str, list] = {}
            for cam in ds.camera_keys:
                path, ft, tt, length = ds.episode_segment(ep, cam)
                cam_frames[cam] = decode_segment(path, ft, tt, length, ds.fps)
            avail = min([len(v) for v in cam_frames.values()] + [len(data)])
            keep = [i for i in keep if 0 <= i < avail]

            for oi in keep:
                if cancel.is_set():
                    raise DatasetError("Cancelled.")
                frame: dict = {}
                for cam in ds.camera_keys:
                    img = cam_frames[cam][oi]
                    fh, fw = features[cam]["shape"][:2]
                    if img.shape[0] != fh or img.shape[1] != fw:
                        img = cv2.resize(img, (fw, fh))
                    frame[cam] = np.ascontiguousarray(img)
                for fk, fv in features.items():
                    if fk in cams:
                        continue
                    val = np.asarray(data.iloc[oi][fk],
                                     dtype=_NP_DTYPES.get(fv["dtype"], np.float32))
                    frame[fk] = val.reshape(fv["shape"])
                if has_task:
                    new.add_frame(frame, task=task)
                else:
                    frame["task"] = task
                    new.add_frame(frame)
                done += 1
                if done % 8 == 0:
                    emit(done / total,
                         f"episode {ep}/{ds.n_episodes - 1}: writing frames  {done}/{total}")

            emit(done / total, f"episode {ep}/{ds.n_episodes - 1}: encoding video…")
            new.save_episode()
            note(f"episode {ep}: kept {len(keep)}/{ds.episode_length(ep)} frames")

        emit(0.99, "Finalising metadata…")
        new.finalize()
        emit(1.0, "Done.")
    except Exception:
        # An un-finalised dataset is unusable — drop the partial output.
        shutil.rmtree(dest, ignore_errors=True)
        raise


# ======================================================================
#  GUI
# ======================================================================
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    from PIL import Image, ImageTk

    GUI_OK = True
except Exception:  # noqa: BLE001 - headless / no Tk; --selftest still works
    GUI_OK = False

BG = "#1b1d23"
PANEL_BG = "#101014"
ACCENT = "#00b8d4"
SEL_FILL = "#7a2f33"
SEL_LINE = "#ff5d63"
PLAYHEAD = "#00e5ff"
SPEEDS = ["0.25", "0.5", "1.0", "1.5", "2.0", "4.0"]


class CameraPanel:
    """One camera tile: a caption label above a frame-display canvas."""

    def __init__(self, parent, cam_key: str):
        self.cam_key = cam_key
        self._rgb = None
        self._photo = None
        self.frame = ttk.Frame(parent, padding=2)
        self.caption = ttk.Label(self.frame, text=cam_key, style="Cap.TLabel",
                                 anchor="center")
        self.caption.pack(side="top", fill="x")
        self.canvas = tk.Canvas(self.frame, bg=PANEL_BG, highlightthickness=1,
                                highlightbackground="#3a3d46")
        self.canvas.pack(side="top", fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _e: self.redraw())

    def show(self, rgb: np.ndarray):
        self._rgb = rgb
        self.redraw()

    def clear(self):
        self._rgb = None
        self.redraw()

    def set_caption(self, text: str):
        self.caption.configure(text=text)

    def redraw(self):
        cv = self.canvas
        cv.delete("all")
        cw, ch = cv.winfo_width(), cv.winfo_height()
        if self._rgb is None or cw < 8 or ch < 8:
            if cw > 24 and ch > 24:
                cv.create_text(cw // 2, ch // 2, text="no episode loaded",
                               fill="#555", font=("TkDefaultFont", 11))
            return
        ih, iw = self._rgb.shape[:2]
        scale = min(cw / iw, ch / ih)
        dw, dh = max(1, int(iw * scale)), max(1, int(ih * scale))
        self._photo = ImageTk.PhotoImage(Image.fromarray(self._rgb).resize((dw, dh)))
        cv.create_image((cw - dw) // 2, (ch - dh) // 2, anchor="nw", image=self._photo)


class TimelineStrip:
    """A canvas strip showing the edited timeline; drag to select a range to cut."""

    def __init__(self, parent, on_select, on_clear):
        self.on_select = on_select
        self.on_clear = on_clear
        self.n = 0              # number of frames on the (edited) timeline
        self.playhead = 0
        self.sel: tuple[int, int] | None = None
        self.cuts: list[tuple[float, float]] = []  # faint marks of removed spans
        self._anchor = None
        self.canvas = tk.Canvas(parent, height=58, bg="#15171d", highlightthickness=0)
        self.canvas.bind("<Configure>", lambda _e: self.redraw())
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.canvas.bind("<Button-3>", lambda _e: self._clear())

    # -- public API -----------------------------------------------------
    def configure_episode(self, n: int):
        self.n = n
        self.sel = None
        self.cuts = []
        self.playhead = 0
        self.redraw()

    def set_playhead(self, pos: int):
        self.playhead = pos
        self.redraw()

    def set_selection(self, sel):
        self.sel = sel
        self.redraw()

    def set_cuts(self, cuts):
        self.cuts = cuts
        self.redraw()

    # -- geometry -------------------------------------------------------
    @property
    def _margin(self) -> int:
        return 10

    def _x_to_frame(self, x) -> int:
        w = self.canvas.winfo_width()
        m = self._margin
        if w <= 2 * m or self.n <= 1:
            return 0
        return max(0, min(self.n - 1, round((x - m) / (w - 2 * m) * (self.n - 1))))

    def _frame_to_x(self, f) -> float:
        w = self.canvas.winfo_width()
        m = self._margin
        if self.n <= 1:
            return m
        return m + f / (self.n - 1) * (w - 2 * m)

    # -- mouse ----------------------------------------------------------
    def _press(self, e):
        if self.n <= 0:
            return
        self._anchor = self._x_to_frame(e.x)
        self.sel = (self._anchor, self._anchor)
        self.redraw()

    def _drag(self, e):
        if self._anchor is None:
            return
        f = self._x_to_frame(e.x)
        self.sel = (min(self._anchor, f), max(self._anchor, f))
        self.redraw()

    def _release(self, _e):
        if self._anchor is None:
            return
        self._anchor = None
        if self.sel and self.on_select:
            self.on_select(self.sel[0], self.sel[1])

    def _clear(self):
        self.sel = None
        self.redraw()
        if self.on_clear:
            self.on_clear()

    # -- drawing --------------------------------------------------------
    def redraw(self):
        cv = self.canvas
        cv.delete("all")
        w, h = cv.winfo_width(), cv.winfo_height()
        if w < 16 or h < 16:
            return
        m = self._margin
        top, bot = 16, h - 16
        cv.create_rectangle(m, top, w - m, bot, fill="#262a33", outline="#3a3d46")
        if self.n <= 0:
            cv.create_text(w // 2, h // 2, text="no episode", fill="#555")
            return
        # faint marks where earlier cuts removed frames (fractional positions)
        for c0, c1 in self.cuts:
            x0 = m + c0 * (w - 2 * m)
            x1 = m + c1 * (w - 2 * m)
            cv.create_line(max(x0, m + 1), top, min(x1, w - m - 1), top,
                           fill="#d9534f", width=3)
        # current selection band
        if self.sel:
            x0, x1 = self._frame_to_x(self.sel[0]), self._frame_to_x(self.sel[1])
            cv.create_rectangle(x0, top, max(x1, x0 + 1), bot,
                                fill=SEL_FILL, outline=SEL_LINE, width=1)
        # playhead
        px = self._frame_to_x(self.playhead)
        cv.create_polygon(px - 5, top - 6, px + 5, top - 6, px, top,
                          fill=PLAYHEAD, outline="")
        cv.create_line(px, top, px, bot, fill=PLAYHEAD, width=2)
        cv.create_text(m, h - 6, text="0", fill="#7c8696", anchor="w",
                       font=("TkFixedFont", 7))
        cv.create_text(w - m, h - 6, text=str(self.n - 1), fill="#7c8696",
                       anchor="e", font=("TkFixedFont", 7))


class ViewerApp:
    """The main Tk application window."""

    def __init__(self, root: "tk.Tk", start_path: Path | None = None):
        self.root = root
        self.ds: LeRobotDatasetV3 | None = None
        self.dataset_paths: list[Path] = []

        # loaded-episode state -----------------------------------------
        self.ep_idx = -1
        self.frames: dict[str, list[bytes]] = {}   # cam -> jpeg bytes, full episode
        self.data_df: pd.DataFrame | None = None
        self.n_orig = 0                            # original frame count
        self.keep: list[int] = []                  # surviving original indices
        self.n_frames = 0                          # == len(self.keep)
        self.cur = 0                               # position on the edited timeline

        # trim bookkeeping ---------------------------------------------
        self.edits: dict[int, list[int]] = {}      # ep -> keep-list (edited eps only)
        self.undo_hist: dict[int, list[list[int]]] = {}

        # playback ------------------------------------------------------
        self.playing = False
        self.speed = 1.0
        self._play_t0 = 0.0
        self._play_f0 = 0
        self._tick_job = None
        self._slider_sync = False

        # background-job plumbing --------------------------------------
        self.q: queue.Queue = queue.Queue()
        self.panels: dict[str, CameraPanel] = {}
        self._load_gen = 0
        self._busy = False
        self._cancel = threading.Event()

        root.title("LeRobot Dataset Viewer")
        root.geometry("1500x960")
        root.configure(bg=BG)
        root.minsize(1080, 700)

        self._init_style()
        self._build_menu()
        self._build_toolbar()
        self._build_main()
        self._build_transport()
        self._build_timeline()
        self._build_trimbar()
        self._build_framedata()
        self._build_statusbar()
        self._bind_keys()

        root.rowconfigure(1, weight=1)
        root.columnconfigure(0, weight=1)
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.after(40, self._poll)

        if start_path:
            start_path = Path(start_path).expanduser()
            if is_dataset_dir(start_path):
                self._scan_folder(start_path.parent, select=start_path)
            else:
                self._scan_folder(start_path)
        else:
            default = Path("~/.cache/huggingface/lerobot").expanduser()
            if default.is_dir():
                self._scan_folder(default)
        self.status("Ready. Open a cache folder, pick an episode, then trim.")

    # -- styling --------------------------------------------------------
    def _init_style(self):
        style = ttk.Style()
        for theme in ("clam", "alt", "default"):
            if theme in style.theme_names():
                style.theme_use(theme)
                break
        style.configure(".", background=BG, foreground="#e6e6e6")
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground="#e6e6e6")
        style.configure("TLabelframe", background=BG, foreground="#9fb6c8")
        style.configure("TLabelframe.Label", background=BG, foreground="#9fb6c8")
        style.configure("TButton", padding=4)
        style.configure("TCheckbutton", background=BG, foreground="#e6e6e6")
        style.configure("Cap.TLabel", background="#23262e", foreground="#9fd0ff",
                        padding=3, font=("TkDefaultFont", 9))
        style.configure("Sel.TLabel", foreground="#ff9da0", font=("TkFixedFont", 9))
        style.configure("Edit.TLabel", foreground="#9fd0ff", font=("TkFixedFont", 9))
        style.configure("Accent.TButton", foreground="#04121b")
        style.map("Accent.TButton", background=[("active", "#3fd0e8"),
                                                ("!disabled", ACCENT)])
        style.configure("Cut.TButton", foreground="#2a0c0d")
        style.map("Cut.TButton", background=[("active", "#ff7c81"),
                                             ("!disabled", "#ff5d63")])

    # -- menu / toolbar -------------------------------------------------
    def _build_menu(self):
        menubar = tk.Menu(self.root)
        filem = tk.Menu(menubar, tearoff=0)
        filem.add_command(label="Open cache folder…", command=self.open_cache_folder)
        filem.add_command(label="Open single dataset…", command=self.open_dataset)
        filem.add_separator()
        filem.add_command(label="Quit", command=self._on_close)
        menubar.add_cascade(label="File", menu=filem)
        helpm = tk.Menu(menubar, tearoff=0)
        helpm.add_command(label="How to trim / shortcuts", command=self._show_help)
        menubar.add_cascade(label="Help", menu=helpm)
        self.root.configure(menu=menubar)

    def _build_toolbar(self):
        bar = ttk.Frame(self.root, padding=(8, 6))
        bar.grid(row=0, column=0, sticky="ew")
        ttk.Button(bar, text="📂 Open Cache Folder",
                   command=self.open_cache_folder).pack(side="left")
        ttk.Button(bar, text="Open Dataset",
                   command=self.open_dataset).pack(side="left", padx=(6, 12))
        ttk.Label(bar, text="Dataset:").pack(side="left")
        self.dataset_var = tk.StringVar()
        self.dataset_cb = ttk.Combobox(bar, textvariable=self.dataset_var,
                                       state="readonly", width=42)
        self.dataset_cb.pack(side="left", padx=6)
        self.dataset_cb.bind("<<ComboboxSelected>>", self._on_dataset_picked)
        ttk.Button(bar, text="⟳", width=3,
                   command=self._refresh_datasets).pack(side="left")
        self.path_lbl = ttk.Label(bar, text="", foreground="#7c8696")
        self.path_lbl.pack(side="left", padx=12)

    def _build_main(self):
        paned = ttk.PanedWindow(self.root, orient="horizontal")
        paned.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)

        left = ttk.Frame(paned, width=300)
        left.pack_propagate(False)
        paned.add(left, weight=0)

        epbox = ttk.Labelframe(left, text="Episodes", padding=4)
        epbox.pack(side="top", fill="both", expand=True)
        sb = ttk.Scrollbar(epbox, orient="vertical")
        self.ep_list = tk.Listbox(epbox, yscrollcommand=sb.set, activestyle="none",
                                  bg="#15171d", fg="#dfe6ee", selectbackground=ACCENT,
                                  selectforeground="#04121b", highlightthickness=0,
                                  font=("TkFixedFont", 9), exportselection=False)
        sb.config(command=self.ep_list.yview)
        sb.pack(side="right", fill="y")
        self.ep_list.pack(side="left", fill="both", expand=True)
        self.ep_list.bind("<<ListboxSelect>>", self._on_episode_picked)

        infobox = ttk.Labelframe(left, text="Dataset info", padding=4)
        infobox.pack(side="top", fill="x", pady=(6, 0))
        self.info_text = tk.Text(infobox, height=13, bg="#15171d", fg="#cdd6df",
                                 relief="flat", wrap="word", font=("TkFixedFont", 8),
                                 highlightthickness=0)
        self.info_text.pack(fill="x")
        self.info_text.configure(state="disabled")

        right = ttk.Frame(paned)
        paned.add(right, weight=1)
        self.cam_container = ttk.Frame(right)
        self.cam_container.pack(fill="both", expand=True)
        ttk.Label(self.cam_container, anchor="center", foreground="#5d6675",
                  text="Open a dataset and pick an episode.").pack(fill="both", expand=True)

    def _build_transport(self):
        bar = ttk.Frame(self.root, padding=(8, 4))
        bar.grid(row=2, column=0, sticky="ew")
        self.tbtns: list[ttk.Button] = []

        def tb(text, cmd, w=4):
            b = ttk.Button(bar, text=text, width=w, command=cmd)
            b.pack(side="left", padx=1)
            self.tbtns.append(b)
            return b

        tb("⏮", lambda: self.goto(0))
        tb("⏪", lambda: self.step(-10))
        tb("◀", lambda: self.step(-1))
        self.play_btn = tb("▶  Play", self.toggle_play, w=9)
        tb("▶", lambda: self.step(1))
        tb("⏩", lambda: self.step(10))
        tb("⏭", lambda: self.goto(10 ** 9))

        self.slider_var = tk.DoubleVar(value=0)
        self.slider = ttk.Scale(bar, from_=0, to=0, orient="horizontal",
                                variable=self.slider_var, command=self._on_slider)
        self.slider.pack(side="left", fill="x", expand=True, padx=10)
        self.slider.bind("<ButtonPress-1>", lambda _e: self.pause())

        self.frame_lbl = ttk.Label(bar, text="–/–", width=26, font=("TkFixedFont", 9))
        self.frame_lbl.pack(side="left")
        ttk.Label(bar, text="Speed").pack(side="left", padx=(8, 2))
        self.speed_cb = ttk.Combobox(bar, values=SPEEDS, width=5, state="readonly")
        self.speed_cb.set("1.0")
        self.speed_cb.pack(side="left")
        self.speed_cb.bind("<<ComboboxSelected>>", self._on_speed)
        self.loop_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Loop", variable=self.loop_var).pack(side="left", padx=8)

    def _build_timeline(self):
        box = ttk.Labelframe(self.root, text="Timeline  —  drag to select a range to cut",
                             padding=(6, 2))
        box.grid(row=3, column=0, sticky="ew", padx=8)
        self.timeline = TimelineStrip(box, on_select=self._on_strip_select,
                                      on_clear=self._on_strip_clear)
        self.timeline.canvas.pack(fill="x", expand=True)

    def _build_trimbar(self):
        bar = ttk.Frame(self.root, padding=(8, 5))
        bar.grid(row=4, column=0, sticky="ew")
        self.trim_btns: list[ttk.Button] = []

        def add(text, cmd, style=None, w=None):
            kw = {"text": text, "command": cmd}
            if style:
                kw["style"] = style
            if w:
                kw["width"] = w
            b = ttk.Button(bar, **kw)
            b.pack(side="left", padx=2)
            self.trim_btns.append(b)
            return b

        add("⟦ Mark In", lambda: self.mark(in_=True))
        add("Mark Out ⟧", lambda: self.mark(in_=False))
        self.sel_lbl = ttk.Label(bar, text="selection: none", style="Sel.TLabel",
                                 width=30)
        self.sel_lbl.pack(side="left", padx=8)
        self.crop_btn = add("✂  Crop (delete selection)", self.crop_selection,
                            style="Cut.TButton")
        self.undo_btn = add("↶ Undo", self.undo_cut)
        add("⟲ Reset Episode", self.reset_episode)
        self.edit_lbl = ttk.Label(bar, text="", style="Edit.TLabel")
        self.edit_lbl.pack(side="left", padx=14)

        self.mp4_btn = ttk.Button(bar, text="🎬 Export MP4…", command=self.export_mp4)
        self.mp4_btn.pack(side="right", padx=2)
        self.png_btn = ttk.Button(bar, text="🖼 Save PNGs…", command=self.save_frame_png)
        self.png_btn.pack(side="right", padx=2)
        self.save_btn = ttk.Button(bar, text="💾  Save Trimmed Copy…",
                                   style="Accent.TButton", command=self.save_trimmed)
        self.save_btn.pack(side="right", padx=2)
        self._refresh_trim_buttons()

    def _build_framedata(self):
        box = ttk.Labelframe(self.root, text="Frame data (state / action)", padding=2)
        box.grid(row=5, column=0, sticky="ew", padx=8)
        self.data_text = tk.Text(box, height=7, bg="#15171d", fg="#cfe3cf",
                                 relief="flat", wrap="none", font=("TkFixedFont", 9),
                                 highlightthickness=0)
        self.data_text.pack(fill="x")
        self.data_text.configure(state="disabled")

    def _build_statusbar(self):
        bar = ttk.Frame(self.root, padding=(8, 3))
        bar.grid(row=6, column=0, sticky="ew")
        self.status_lbl = ttk.Label(bar, text="", foreground="#9fb6c8")
        self.status_lbl.pack(side="left")
        self.cancel_btn = ttk.Button(bar, text="Cancel", width=8,
                                     command=self._cancel_job, state="disabled")
        self.cancel_btn.pack(side="right")
        self.progress = ttk.Progressbar(bar, length=260, mode="determinate")
        self.progress.pack(side="right", padx=8)

    def _bind_keys(self):
        r = self.root
        r.bind("<space>", lambda _e: self.toggle_play())
        r.bind("<Left>", lambda _e: self.step(-1))
        r.bind("<Right>", lambda _e: self.step(1))
        r.bind("<Prior>", lambda _e: self.step(-10))
        r.bind("<Next>", lambda _e: self.step(10))
        r.bind("<Home>", lambda _e: self.goto(0))
        r.bind("<End>", lambda _e: self.goto(10 ** 9))
        r.bind("<Up>", lambda _e: self._step_episode(-1))
        r.bind("<Down>", lambda _e: self._step_episode(1))
        r.bind("<bracketleft>", lambda _e: self.mark(in_=True))
        r.bind("<bracketright>", lambda _e: self.mark(in_=False))
        r.bind("<Delete>", lambda _e: self.crop_selection())
        r.bind("<BackSpace>", lambda _e: self.crop_selection())
        r.bind("<Control-z>", lambda _e: self.undo_cut())

    def _show_help(self):
        messagebox.showinfo(
            "How to trim",
            "TRIMMING (deletes frames + matching state/action rows):\n"
            "  1. drag on the Timeline strip to select a frame range\n"
            "     (or play to a spot and press [ and ] to mark In/Out)\n"
            "  2. press 'Crop (delete selection)' — those frames are removed\n"
            "  3. repeat as many times as you like; 'Undo' / 'Reset Episode'\n"
            "  4. 'Save Trimmed Copy' writes a new dataset with the edits.\n"
            "Trims are remembered per episode; Save applies them all at once.\n\n"
            "KEYBOARD:\n"
            "  Space  play/pause      ← →  step frame     PgUp/PgDn  ±10\n"
            "  Home/End  first/last   ↑ ↓  prev/next episode\n"
            "  [  mark In   ]  mark Out   Delete  crop   Ctrl+Z  undo\n\n"
            "Right-click the timeline to clear the selection.")

    # -- status helpers -------------------------------------------------
    def status(self, msg: str):
        self.status_lbl.configure(text=msg)

    def _set_busy(self, busy: bool):
        self._busy = busy
        state = "disabled" if busy else "normal"
        for b in (self.save_btn, self.png_btn, self.mp4_btn, *self.trim_btns):
            b.configure(state=state)
        self.cancel_btn.configure(state="normal" if busy else "disabled")
        if not busy:
            self.progress.configure(value=0)
            self._refresh_trim_buttons()

    # -- dataset discovery / loading -----------------------------------
    def open_cache_folder(self):
        folder = filedialog.askdirectory(title="Pick a HuggingFace LeRobot cache folder")
        if folder:
            self._scan_folder(Path(folder))

    def open_dataset(self):
        folder = filedialog.askdirectory(title="Pick a single LeRobot dataset folder")
        if not folder:
            return
        path = Path(folder)
        if not is_dataset_dir(path):
            messagebox.showerror("Not a dataset", "That folder has no meta/info.json.")
            return
        self._scan_folder(path.parent, select=path)

    def _refresh_datasets(self):
        if self.dataset_paths:
            self._scan_folder(self.dataset_paths[0].parent,
                              select=self.ds.root if self.ds else None)

    def _scan_folder(self, folder: Path, select: Path | None = None):
        self.dataset_paths = find_datasets(folder)
        self.path_lbl.configure(text=str(folder))
        if not self.dataset_paths:
            self.dataset_cb["values"] = []
            self.dataset_var.set("")
            self.status(f"No LeRobot datasets found in {folder}")
            return
        self.dataset_cb["values"] = [p.name for p in self.dataset_paths]
        self.status(f"Found {len(self.dataset_paths)} dataset(s) in {folder}")
        target = select if (select and select in self.dataset_paths) else self.dataset_paths[0]
        self.dataset_var.set(target.name)
        self._load_dataset(target)

    def _on_dataset_picked(self, _e):
        for p in self.dataset_paths:
            if p.name == self.dataset_var.get():
                self._load_dataset(p)
                return

    def _load_dataset(self, path: Path):
        self.pause()
        try:
            self.ds = LeRobotDatasetV3(path)
        except DatasetError as exc:
            messagebox.showerror("Cannot open dataset", str(exc))
            self.status(f"Failed to open {path.name}")
            return
        self.edits.clear()
        self.undo_hist.clear()
        self.ep_idx = -1
        self.frames, self.data_df = {}, None
        self.n_orig = self.n_frames = self.cur = 0
        self.keep = []
        self._build_camera_panels()
        self._fill_episode_list()
        self._fill_info()
        self.timeline.configure_episode(0)
        self._update_transport_state()
        self._update_edit_label()
        self.status(f"Loaded {self.ds.name} — {self.ds.n_episodes} episode(s).")
        if self.ds.n_episodes:
            self.ep_list.selection_set(0)
            self.ep_list.see(0)
            self._load_episode(0)

    def _build_camera_panels(self):
        for child in self.cam_container.winfo_children():
            child.destroy()
        self.panels = {}
        cams = self.ds.camera_keys
        if not cams:
            ttk.Label(self.cam_container, text="This dataset has no video cameras.",
                      foreground="#5d6675").pack(fill="both", expand=True)
            return
        cols = max(1, math.ceil(math.sqrt(len(cams))))
        rows = math.ceil(len(cams) / cols)
        for i, cam in enumerate(cams):
            panel = CameraPanel(self.cam_container, cam)
            r, c = divmod(i, cols)
            panel.frame.grid(row=r, column=c, sticky="nsew", padx=2, pady=2)
            self.panels[cam] = panel
        for c in range(cols):
            self.cam_container.columnconfigure(c, weight=1)
        for r in range(rows):
            self.cam_container.rowconfigure(r, weight=1)

    def _episode_label(self, ep: int) -> str:
        orig = self.ds.episode_length(ep)
        tasks = self.ds.episode_tasks(ep)
        task = tasks[0] if tasks else ""
        keep = self.edits.get(ep)
        if keep is not None and len(keep) != orig:
            return f"ep {ep:03d}  {orig}→{len(keep)}f ✂  {task[:18]}"
        return f"ep {ep:03d}  {orig:>5d}f {orig / self.ds.fps:5.1f}s  {task[:18]}"

    def _fill_episode_list(self):
        self.ep_list.delete(0, "end")
        for ep in self.ds.episodes["episode_index"]:
            self.ep_list.insert("end", self._episode_label(int(ep)))

    def _refresh_episode_entry(self, ep: int):
        had = self.ep_list.curselection()
        self.ep_list.delete(ep)
        self.ep_list.insert(ep, self._episode_label(ep))
        if had and had[0] == ep:
            self.ep_list.selection_set(ep)

    def _fill_info(self):
        ds = self.ds
        lines = [
            f"name      {ds.name}",
            f"version   {ds.info.get('codebase_version')}",
            f"robot     {ds.info.get('robot_type')}",
            f"fps       {ds.fps:g}",
            f"episodes  {ds.n_episodes}",
            f"frames    {ds.info.get('total_frames')}",
            "", "cameras:",
        ]
        for cam in ds.camera_keys:
            sz = ds.camera_size(cam)
            lines.append(f"  {cam}")
            lines.append(f"     {sz[0]}x{sz[1]}  {ds.camera_codec(cam)}" if sz
                         else f"     ?  {ds.camera_codec(cam)}")
        other = [k for k in ds.features
                 if ds.features[k].get("dtype") not in ("video", "image")
                 and k not in _BOOKKEEPING]
        if other:
            lines += ["", "features:"]
            lines += [f"  {k} {tuple(ds.features[k].get('shape', ()))}" for k in other]
        self.info_text.configure(state="normal")
        self.info_text.delete("1.0", "end")
        self.info_text.insert("1.0", "\n".join(lines))
        self.info_text.configure(state="disabled")

    # -- episode loading (background) ----------------------------------
    def _on_episode_picked(self, _e):
        sel = self.ep_list.curselection()
        if sel:
            self._load_episode(int(sel[0]))

    def _step_episode(self, delta: int):
        if not self.ds or not self.ds.n_episodes:
            return
        nxt = max(0, min(self.ds.n_episodes - 1, self.ep_idx + delta))
        if nxt != self.ep_idx:
            self.ep_list.selection_clear(0, "end")
            self.ep_list.selection_set(nxt)
            self.ep_list.see(nxt)
            self._load_episode(nxt)

    def _load_episode(self, ep: int):
        if not self.ds or self._busy or ep == self.ep_idx:
            return
        self.pause()
        self.ep_idx = ep
        self._load_gen += 1
        gen = self._load_gen
        self.progress.configure(value=0)
        self.status(f"Loading episode {ep}…")
        for b in self.tbtns:
            b.configure(state="disabled")
        self.slider.configure(state="disabled")
        threading.Thread(target=self._preload_worker, args=(ep, gen), daemon=True).start()

    def _preload_worker(self, ep: int, gen: int):
        try:
            cams = self.ds.camera_keys
            frames: dict[str, list[bytes]] = {}
            for ci, cam in enumerate(cams):
                path, from_ts, to_ts, length = self.ds.episode_segment(ep, cam)
                if not path.is_file():
                    raise DatasetError(f"Missing video file:\n{path}")

                def prog(frac, ci=ci):
                    self.q.put({"kind": "progress", "gen": gen,
                                "frac": (ci + frac) / max(1, len(cams)),
                                "text": f"Decoding {cam}…  {int(frac * 100)}%"})

                arrs = decode_segment(path, from_ts, to_ts, length, self.ds.fps, prog)
                frames[cam] = [jpeg_encode(a) for a in arrs]
            data = self.ds.episode_data(ep)
            self.q.put({"kind": "episode_ready", "gen": gen, "ep": ep,
                        "frames": frames, "data": data})
        except Exception as exc:  # noqa: BLE001
            self.q.put({"kind": "error", "gen": gen, "msg": str(exc),
                        "tb": traceback.format_exc()})

    def _on_episode_ready(self, msg: dict):
        self.frames = msg["frames"]
        self.data_df = msg["data"]
        counts = [len(v) for v in self.frames.values()]
        self.n_orig = min(counts) if counts else 0
        if self.data_df is not None and len(self.data_df):
            self.n_orig = min(self.n_orig, len(self.data_df))
        self.cur = 0
        ep = msg["ep"]
        prev = self.edits.get(ep)
        keep = list(range(self.n_orig)) if prev is None else \
            [i for i in prev if 0 <= i < self.n_orig]
        if not keep:
            keep = list(range(self.n_orig))
        self._set_keep(keep)
        if self.n_orig:
            tasks = self.ds.episode_tasks(ep)
            task = f"  ·  {tasks[0]}" if tasks else ""
            self.status(f"Episode {ep} — {self.n_orig} frames{task}")
        else:
            self.status(f"Episode {ep} decoded 0 frames.")

    def _set_keep(self, keep: list[int]):
        """Install *keep* as the edited timeline for the current episode."""
        self.keep = list(keep)
        self.n_frames = len(self.keep)
        ep = self.ep_idx
        if self.n_frames != self.n_orig:
            self.edits[ep] = list(self.keep)
        else:
            self.edits.pop(ep, None)
        self.cur = max(0, min(self.cur, self.n_frames - 1)) if self.n_frames else 0
        self.timeline.configure_episode(self.n_frames)
        self._update_transport_state()
        self._refresh_episode_entry(ep)
        self._update_edit_label()
        if self.n_frames:
            self.show_frame(self.cur)

    def _update_transport_state(self):
        ready = self.n_frames > 0
        for b in self.tbtns:
            b.configure(state="normal" if ready else "disabled")
        self.slider.configure(state="normal" if ready else "disabled",
                              from_=0, to=max(0, self.n_frames - 1))
        if not ready:
            self.frame_lbl.configure(text="–/–")

    # -- playback -------------------------------------------------------
    def show_frame(self, p: int):
        if not self.n_frames:
            return
        p = max(0, min(self.n_frames - 1, p))
        self.cur = p
        orig = self.keep[p]
        for cam, panel in self.panels.items():
            jpegs = self.frames.get(cam)
            if jpegs and orig < len(jpegs):
                panel.show(jpeg_decode(jpegs[orig]))
        self._slider_sync = True
        self.slider_var.set(p)
        self._slider_sync = False
        self.timeline.set_playhead(p)
        t = p / self.ds.fps if self.ds else 0.0
        self.frame_lbl.configure(
            text=f"{p}/{self.n_frames - 1}  orig {orig}  t={t:6.2f}s")
        self._update_framedata(orig)

    def _update_framedata(self, orig: int):
        self.data_text.configure(state="normal")
        self.data_text.delete("1.0", "end")
        if self.data_df is None or orig >= len(self.data_df):
            self.data_text.configure(state="disabled")
            return
        row = self.data_df.iloc[orig]
        keys = sorted((k for k in self.data_df.columns if k not in _BOOKKEEPING),
                      key=lambda k: (k != "action", k))
        out = []
        for k in keys:
            try:
                arr = np.asarray(row[k]).ravel()
            except Exception:  # noqa: BLE001
                continue
            if arr.dtype.kind in "fc":
                vals = ", ".join(f"{v:+.3f}" for v in arr)
            else:
                vals = ", ".join(str(v) for v in arr)
            out.append(f"{k:<34s} [{vals}]")
        self.data_text.insert("1.0", "\n".join(out))
        self.data_text.configure(state="disabled")

    def goto(self, idx: int):
        if self.n_frames:
            self.pause()
            self.show_frame(idx)

    def step(self, delta: int):
        if self.n_frames:
            self.pause()
            self.show_frame(self.cur + delta)

    def _on_slider(self, val):
        if self._slider_sync or not self.n_frames:
            return
        self.show_frame(int(float(val)))

    def _on_speed(self, _e):
        try:
            self.speed = float(self.speed_cb.get())
        except ValueError:
            self.speed = 1.0
        if self.playing:
            self._play_t0, self._play_f0 = time.monotonic(), self.cur

    def toggle_play(self):
        self.pause() if self.playing else self.play()

    def play(self):
        if not self.n_frames or self.playing:
            return
        self.playing = True
        self.play_btn.configure(text="⏸  Pause")
        self._play_f0 = 0 if self.cur >= self.n_frames - 1 else self.cur
        self._play_t0 = time.monotonic()
        self._tick()

    def pause(self):
        self.playing = False
        if hasattr(self, "play_btn"):
            self.play_btn.configure(text="▶  Play")
        if self._tick_job is not None:
            self.root.after_cancel(self._tick_job)
            self._tick_job = None

    def _tick(self):
        self._tick_job = None
        if not self.playing or not self.n_frames:
            return
        elapsed = time.monotonic() - self._play_t0
        target = self._play_f0 + int(elapsed * self.ds.fps * self.speed)
        if target >= self.n_frames:
            if self.loop_var.get():
                self._play_f0, self._play_t0, target = 0, time.monotonic(), 0
            else:
                self.show_frame(self.n_frames - 1)
                self.pause()
                return
        self.show_frame(target)
        interval = max(8, int(1000.0 / (self.ds.fps * self.speed)))
        self._tick_job = self.root.after(interval, self._tick)

    # -- trimming -------------------------------------------------------
    def _refresh_trim_buttons(self):
        if self._busy:
            return
        has_sel = bool(self.timeline.sel) and self.n_frames > 0
        self.crop_btn.configure(state="normal" if has_sel else "disabled")
        self.undo_btn.configure(
            state="normal" if self.undo_hist.get(self.ep_idx) else "disabled")

    def _refresh_selection(self):
        sel = self.timeline.sel
        if sel and self.n_frames:
            a, b = sel
            n = b - a + 1
            self.sel_lbl.configure(
                text=f"selection: {a}–{b}  ({n}f, {n / self.ds.fps:.1f}s)")
        else:
            self.sel_lbl.configure(text="selection: none")
        self._refresh_trim_buttons()

    def _update_edit_label(self):
        if not self.n_orig:
            self.edit_lbl.configure(text="")
        elif self.n_frames != self.n_orig:
            self.edit_lbl.configure(
                text=f"episode {self.ep_idx}: {self.n_orig} → {self.n_frames}f  "
                     f"({self.n_orig - self.n_frames} removed)")
        else:
            self.edit_lbl.configure(text=f"episode {self.ep_idx}: {self.n_orig}f")
        self._refresh_trim_buttons()

    def _on_strip_select(self, _a, _b):
        self.pause()
        self._refresh_selection()

    def _on_strip_clear(self):
        self._refresh_selection()

    def mark(self, in_: bool):
        """Set the selection In or Out edge to the current playhead frame."""
        if not self.n_frames:
            return
        sel = self.timeline.sel
        a = sel[0] if sel else self.cur
        b = sel[1] if sel else self.cur
        if in_:
            a = self.cur
            b = max(a, b)
        else:
            b = self.cur
            a = min(a, b)
        self.timeline.set_selection((a, b))
        self._refresh_selection()

    def crop_selection(self):
        """Delete the selected frame range from the current episode."""
        if self._busy or not self.n_frames:
            return
        sel = self.timeline.sel
        if not sel:
            self.status("Select a range on the timeline first (drag on it).")
            return
        a, b = sel
        new_keep = self.keep[:a] + self.keep[b + 1:]
        if not new_keep:
            messagebox.showwarning(
                "Cannot crop",
                "That selection covers the whole episode — at least one "
                "frame must remain.")
            return
        removed = b - a + 1
        self.pause()
        self.undo_hist.setdefault(self.ep_idx, []).append(list(self.keep))
        self._set_keep(new_keep)
        self.status(f"Cropped {removed} frame(s). Episode now {self.n_frames} frames.")

    def undo_cut(self):
        if self._busy:
            return
        hist = self.undo_hist.get(self.ep_idx)
        if not hist:
            return
        self.pause()
        self._set_keep(hist.pop())
        self.status(f"Undo — episode restored to {self.n_frames} frames.")

    def reset_episode(self):
        if self._busy or not self.n_orig:
            return
        self.pause()
        self.undo_hist.pop(self.ep_idx, None)
        self._set_keep(list(range(self.n_orig)))
        self.status("Episode reset to full length.")

    # -- save trimmed copy ---------------------------------------------
    def save_trimmed(self):
        if not self.ds or self._busy:
            return
        edited = dict(self.edits)
        if not edited and not messagebox.askyesno(
                "No trims made",
                "You haven't cropped anything yet.\n"
                "Save a full (re-encoded) copy anyway?"):
            return
        dest = filedialog.asksaveasfilename(
            title="Save trimmed dataset copy as…",
            initialdir=str(self.ds.root.parent),
            initialfile=f"{self.ds.name}_trimmed")
        if not dest:
            return
        dest = Path(dest)
        if dest.exists():
            messagebox.showerror("Exists", f"Already exists:\n{dest}")
            return
        n_ep = self.ds.n_episodes
        total_in = sum(self.ds.episode_length(e) for e in range(n_ep))
        total_out = sum(len(edited.get(e, range(self.ds.episode_length(e))))
                        for e in range(n_ep))
        if not messagebox.askyesno(
                "Save trimmed copy",
                f"Write a new dataset to:\n{dest}\n\n"
                f"episodes      : {n_ep}\n"
                f"edited        : {len(edited)}\n"
                f"frames        : {total_in} → {total_out}\n\n"
                "Every video is re-encoded through LeRobot's writer. Continue?"):
            return
        self._cancel.clear()
        self._set_busy(True)
        self.status("Saving trimmed copy…")
        threading.Thread(target=self._save_worker, args=(edited, dest),
                         daemon=True).start()

    def _save_worker(self, edits, dest):
        try:
            save_trimmed_copy(
                self.ds, edits, dest, dest.name, self._cancel,
                progress=lambda f, t: self.q.put(
                    {"kind": "job_progress", "frac": f, "text": t}),
                log=lambda m: self.q.put({"kind": "job_log", "text": m}))
            self.q.put({"kind": "job_done", "what": "save", "dest": str(dest)})
        except Exception as exc:  # noqa: BLE001
            self.q.put({"kind": "job_error", "msg": str(exc),
                        "tb": traceback.format_exc()})

    # -- save frame PNGs / export MP4 ----------------------------------
    def save_frame_png(self):
        if not self.n_frames:
            return
        folder = filedialog.askdirectory(title="Pick a folder for the PNG frames")
        if not folder:
            return
        folder = Path(folder)
        orig = self.keep[self.cur]
        saved = 0
        for cam, jpegs in self.frames.items():
            if orig >= len(jpegs):
                continue
            safe = cam.replace("/", "_").replace(".", "_")
            out = folder / f"{self.ds.name}_ep{self.ep_idx:03d}_f{self.cur:05d}_{safe}.png"
            Image.fromarray(jpeg_decode(jpegs[orig])).save(out)
            saved += 1
        messagebox.showinfo("Saved", f"Saved {saved} PNG(s) to:\n{folder}")
        self.status(f"Saved {saved} PNG frame(s).")

    def export_mp4(self):
        if not self.n_frames or self._busy:
            return
        path = filedialog.asksaveasfilename(
            title="Export episode as MP4", defaultextension=".mp4",
            initialfile=f"{self.ds.name}_ep{self.ep_idx:03d}.mp4",
            filetypes=[("MP4 video", "*.mp4")])
        if not path:
            return
        self.pause()
        self._cancel.clear()
        self._set_busy(True)
        threading.Thread(target=self._export_worker,
                         args=(Path(path), list(self.keep)), daemon=True).start()

    def _export_worker(self, path, keep):
        try:
            cams = self.ds.camera_keys
            writer, n, target_h = None, len(keep), 360
            for i, orig in enumerate(keep):
                if self._cancel.is_set():
                    raise DatasetError("Cancelled.")
                tiles = []
                for cam in cams:
                    rgb = jpeg_decode(self.frames[cam][orig])
                    tw = max(2, int(rgb.shape[1] * target_h / rgb.shape[0]))
                    tiles.append(cv2.resize(rgb, (tw, target_h)))
                comp = np.hstack(tiles) if len(tiles) > 1 else tiles[0]
                if comp.shape[1] % 2:
                    comp = np.pad(comp, ((0, 0), (0, 1), (0, 0)))
                if writer is None:
                    h, w = comp.shape[:2]
                    writer = cv2.VideoWriter(str(path),
                                             cv2.VideoWriter_fourcc(*"mp4v"),
                                             self.ds.fps, (w, h))
                writer.write(cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
                if i % 8 == 0:
                    self.q.put({"kind": "job_progress", "frac": i / max(1, n),
                                "text": f"Exporting MP4…  {i}/{n}"})
            if writer is not None:
                writer.release()
            self.q.put({"kind": "job_done", "what": "mp4", "dest": str(path)})
        except Exception as exc:  # noqa: BLE001
            self.q.put({"kind": "job_error", "msg": str(exc),
                        "tb": traceback.format_exc()})

    # -- background-job polling ----------------------------------------
    def _poll(self):
        try:
            while True:
                self._handle(self.q.get_nowait())
        except queue.Empty:
            pass
        self.root.after(40, self._poll)

    def _handle(self, msg: dict):
        kind = msg["kind"]
        if kind == "progress":
            if msg["gen"] == self._load_gen:
                self.progress.configure(value=msg["frac"] * 100)
                self.status(msg["text"])
        elif kind == "episode_ready":
            if msg["gen"] == self._load_gen:
                self.progress.configure(value=0)
                self._on_episode_ready(msg)
        elif kind == "error":
            if msg["gen"] == self._load_gen:
                self.progress.configure(value=0)
                self.n_frames = 0
                self._update_transport_state()
                messagebox.showerror("Episode load failed", msg["msg"])
                self.status("Episode load failed.")
                sys.stderr.write(msg.get("tb", ""))
        elif kind == "job_progress":
            self.progress.configure(value=msg["frac"] * 100)
            self.status(msg["text"])
        elif kind == "job_log":
            self.status(msg["text"])
        elif kind == "job_done":
            self._set_busy(False)
            dest = msg["dest"]
            if msg["what"] == "save":
                self.status(f"Trimmed copy saved → {dest}")
                if messagebox.askyesno("Done",
                                       f"Trimmed copy saved:\n{dest}\n\nOpen it now?"):
                    self._scan_folder(Path(dest).parent, select=Path(dest))
            else:
                self.status(f"MP4 exported → {dest}")
                messagebox.showinfo("Done", f"Episode exported:\n{dest}")
        elif kind == "job_error":
            self._set_busy(False)
            if "Cancelled" in msg["msg"]:
                self.status("Job cancelled.")
            else:
                messagebox.showerror("Job failed", msg["msg"])
                self.status("Job failed.")
                sys.stderr.write(msg.get("tb", ""))

    def _cancel_job(self):
        self._cancel.set()
        self.status("Cancelling…")

    def _on_close(self):
        self._cancel.set()
        self.pause()
        self.root.destroy()


# ======================================================================
#  Headless self-test
# ======================================================================
def selftest(path: Path) -> int:
    """Load a dataset and decode a few frames without opening a GUI."""
    print(f"[selftest] target: {path}")
    datasets = find_datasets(path)
    if not datasets:
        print("[selftest] FAIL — no LeRobot dataset found.")
        return 1
    ds = LeRobotDatasetV3(datasets[0])
    print(f"[selftest] dataset  : {ds.name}")
    print(f"[selftest] version  : {ds.info.get('codebase_version')}")
    print(f"[selftest] episodes : {ds.n_episodes}   fps={ds.fps:g}")
    print(f"[selftest] cameras  : {ds.camera_keys}")
    if not ds.camera_keys:
        print("[selftest] FAIL — dataset has no video cameras.")
        return 1
    ok = True
    for cam in ds.camera_keys:
        p, ft, tt, length = ds.episode_segment(0, cam)
        t0 = time.monotonic()
        frames = decode_segment(p, ft, tt, length, ds.fps)
        dt = time.monotonic() - t0
        if not frames:
            ok = False
        shape = frames[0].shape if frames else None
        print(f"[selftest]   {cam}: {len(frames)}/{length} frames {shape} "
              f"in {dt:.2f}s  [{'ok' if frames else 'FAIL'}]")
    data = ds.episode_data(0)
    print(f"[selftest] data rows: {len(data)}  cols={len(data.columns)}")
    feats = build_create_features(ds.info)
    print(f"[selftest] writer features ({len(feats)}): {sorted(feats)}")
    print("[selftest] PASS" if ok else "[selftest] FAIL")
    return 0 if ok else 1


def launch_gui(start_path: Path | None) -> int:
    if not GUI_OK:
        print("ERROR: tkinter / PIL not available — cannot open the GUI.\n"
              "Run inside the 'lerobot' pixi env, or use --selftest for a "
              "headless check.", file=sys.stderr)
        return 1
    root = tk.Tk()
    ViewerApp(root, start_path)
    root.mainloop()
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="GUI viewer for LeRobot v3 datasets (browse, watch, time-trim).")
    parser.add_argument("path", nargs="?", default=None,
                        help="cache folder or a single dataset to open on start")
    parser.add_argument("--selftest", metavar="DATASET",
                        help="headless: load a dataset, decode frames, exit")
    args = parser.parse_args(argv)
    if args.selftest:
        return selftest(Path(args.selftest).expanduser())
    return launch_gui(Path(args.path).expanduser() if args.path else None)


if __name__ == "__main__":
    sys.exit(main())
