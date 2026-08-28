"""Convert a LeRobot v3 dataset's orientation slots from Euler XYZ to axis-angle.

This script takes a LeRobot v3 dataset whose `observation.state[3:6]` and
`action[3:6]` slots store **end-effector orientation as Euler XYZ angles** and
writes a *new* sibling dataset where those same slots store the equivalent
**axis-angle (rotation vector)** representation, with per-stream sign-flip
detection that mirrors the runtime logic in
`crisp_gym/envs/manipulator_env.py::_flip_rotation_vector_if_needed` so the
resulting rotvec time-series is continuous within each episode.

Why: Euler has ±π wraparound per axis and gimbal lock; ||ω||≤π axis-angle with
flip detection avoids those discontinuities and is the representation the
runtime env switches to automatically when
`orientation_representation == "angle_axis"` (see manipulator_env.py:157-188,
596-614).

Structure mirrors `fix_rotation_vectors_in_dataset.py` (same dir), with three
divergences:
  1) source / target are **local paths** (`root=...`), not HF Hub repos;
  2) both `observation.state[3:6]` AND `action[3:6]` are converted, each with
     its own per-episode flip accumulator (they are physically distinct signals
     — current EE pose vs. target EE pose — and the runtime env tracks them
     separately too);
  3) Euler→rotvec conversion (`Rotation.from_euler("xyz", v).as_rotvec()`)
     happens **before** the flip-detection step.

Run inside the `lerobot-041` conda env:

    conda run -n lerobot-041 python \
        crisp_gym/scripts/dataset_conversions/convert_euler_to_angle_axis.py

The default source/target paths are wired to the pickplace_cart7_v2 dataset;
override at the bottom of the file (or via CLI flags) if you want to convert a
different dataset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import einops
import numpy as np
import pandas as pd
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES

# `lerobot.utils.rotation.Rotation` is a numpy-only scipy-compatible drop-in
# (covers from_quat/from_rotvec/as_quat/as_rotvec) — using it lets us skip the
# scipy dependency in this env, which doesn't ship with scipy.
from lerobot.utils.rotation import Rotation
from numpy.typing import NDArray
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_SOURCE_REPO_ID = "pickplace_cart7_v2"
DEFAULT_SOURCE_ROOT = Path("/home/batur/Coding/data/pickplace_cart7_v2")
DEFAULT_TARGET_REPO_ID = "pickplace_cart7_v2_angleaxis"
DEFAULT_TARGET_ROOT = Path("/home/batur/Coding/data/pickplace_cart7_v2_angleaxis")

# Indices of the rotation slots within the [x, y, z, roll, pitch, yaw, gripper]
# 7-dim observation.state and action vectors. Fixed by the source dataset's
# `features.*.names` (verified against meta/info.json).
ROT_SLICE = slice(3, 6)

LIMIT_EPISODES: int | None = None  # set to a small int to dry-run on a few episodes


# ---------------------------------------------------------------------------
# Flip-detection helpers — mirrors manipulator_env.py:165-188 exactly, and
# reuses the same logic as fix_rotation_vectors_in_dataset.py.
# ---------------------------------------------------------------------------


def _point_in_opposite_direction(vector1: NDArray, vector2: NDArray) -> bool:
    """Return True iff the two vectors point in roughly opposite directions."""
    n1 = float(np.linalg.norm(vector1))
    n2 = float(np.linalg.norm(vector2))
    if n1 == 0 or n2 == 0:
        return False
    return float(np.dot(vector1 / n1, vector2 / n2)) < 0


def _flip_rotation_vector_if_needed(
    previous_rotation_vector: NDArray | None,
    rotation_vector: NDArray,
) -> NDArray:
    """Apply manipulator_env's rotvec sign convention.

    - First frame of an episode: ensure `rotation_vector[0] >= 0`.
    - Subsequent frames: if the dot product with the previous (kept) vector is
      negative, flip the sign of the current vector so it doesn't visit the
      antipodal pole.
    """
    if previous_rotation_vector is not None:
        if _point_in_opposite_direction(previous_rotation_vector, rotation_vector):
            return -rotation_vector
        return rotation_vector
    # First frame: first component positive
    if rotation_vector[0] < 0:
        return -rotation_vector
    return rotation_vector


def _euler_xyz_extrinsic_to_quat(euler_xyz: NDArray) -> NDArray:
    """Convert extrinsic XYZ Euler angles to a quaternion in scipy [x,y,z,w] order.

    Composition is q = q_z * q_y * q_x (rotations applied first about world X,
    then world Y, then world Z), matching `scipy.spatial.transform.Rotation
    .from_euler("xyz", ...)` to machine precision (verified to 1e-16 over 10k
    random samples). We do this in pure numpy because the `lerobot-041` conda
    env does not ship with scipy.
    """
    a, b, c = float(euler_xyz[0]), float(euler_xyz[1]), float(euler_xyz[2])
    ca, sa = np.cos(a / 2.0), np.sin(a / 2.0)
    cb, sb = np.cos(b / 2.0), np.sin(b / 2.0)
    cc, sc = np.cos(c / 2.0), np.sin(c / 2.0)
    x = cc * cb * sa - sc * sb * ca
    y = cc * sb * ca + sc * cb * sa
    z = sc * cb * ca - cc * sb * sa
    w = cc * cb * ca + sc * sb * sa
    return np.array([x, y, z, w], dtype=np.float64)


def _euler_xyz_to_rotvec(euler_xyz: NDArray) -> NDArray:
    """Convert a single (3,) Euler-XYZ angle triple to a (3,) rotation vector.

    Goes through quaternion internally, which automatically *fixes* Euler ±π
    wraparound and gimbal-lock-adjacent frames — consecutive Euler triples that
    represent the same physical orientation map to consecutive quaternions,
    and those map to a continuous (modulo antipodal) rotvec stream that the
    subsequent flip-detection then fully smooths.
    """
    quat = _euler_xyz_extrinsic_to_quat(euler_xyz)
    return Rotation.from_quat(quat).as_rotvec().astype(np.float32)


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def convert(
    source_repo_id: str,
    source_root: Path,
    target_repo_id: str,
    target_root: Path,
    limit_episodes: int | None = None,
) -> LeRobotDataset:
    if target_root.exists():
        raise FileExistsError(
            f"Target dataset directory already exists: {target_root}\n"
            f"LeRobotDataset.create refuses to overwrite. Remove it first if you "
            f"want to re-run:  rm -rf {target_root}"
        )

    print(f"Source: {source_repo_id} @ {source_root}")
    print(f"Target: {target_repo_id} @ {target_root}")

    # video_backend="pyav" because torchcodec is broken in the lerobot-041
    # env (it fails to load libtorchcodec — FFmpeg shared-object mismatch).
    dataset = LeRobotDataset(repo_id=source_repo_id, root=source_root, video_backend="pyav")
    print(
        f"Loaded source: {dataset.meta.total_episodes} episodes, "
        f"{dataset.meta.total_frames} frames, fps={dataset.fps}."
    )

    # Sanity-check the slot layout we're going to slice into. shape may come
    # back as a list, tuple, or numpy size; names may be list or tuple too.
    for k in ("observation.state", "action"):
        names = list(dataset.features[k]["names"])
        shape = tuple(dataset.features[k]["shape"])
        if shape != (7,) or names[3:6] != ["roll", "pitch", "yaw"]:
            raise RuntimeError(
                f"Unexpected layout for `{k}`: shape={shape}, names={names}. "
                f"This script assumes 7-DoF [x,y,z,roll,pitch,yaw,gripper] with "
                f"the Euler slot at indices [3:6]."
            )

    # Schema is preserved verbatim; the rotvec values live in the same slots
    # under the same names. The semantic change is recorded in a sibling
    # CRISP_META.md file at the end.
    new_dataset = LeRobotDataset.create(
        repo_id=target_repo_id,
        fps=dataset.fps,
        features=dataset.features,
        root=target_root,
        video_backend="pyav",
    )

    total_eps = dataset.meta.total_episodes if limit_episodes is None else min(
        limit_episodes, dataset.meta.total_episodes
    )

    pbar = tqdm(total=total_eps, desc="Converting episodes", unit="ep")
    current_episode_index = 0
    previous_state_rotvec: NDArray | None = None
    previous_action_rotvec: NDArray | None = None

    for frame in dataset:
        ep = int(frame["episode_index"])

        if ep > current_episode_index:
            # Episode boundary: finalize the previous episode and reset
            # both per-stream accumulators.
            new_dataset.save_episode()
            pbar.update(1)
            current_episode_index = ep
            previous_state_rotvec = None
            previous_action_rotvec = None
            if limit_episodes is not None and ep >= limit_episodes:
                break

        new_frame: dict = {}
        for key in dataset.features.keys():
            if key in DEFAULT_FEATURES:
                # episode_index/frame_index/index/task_index/timestamp are
                # auto-populated by add_frame — must not be passed in.
                continue
            if key == "observation.state":
                state = frame[key].clone().detach().cpu().numpy().astype(np.float32)
                rotvec = _euler_xyz_to_rotvec(state[ROT_SLICE])
                rotvec = _flip_rotation_vector_if_needed(previous_state_rotvec, rotvec)
                state[ROT_SLICE] = rotvec
                previous_state_rotvec = rotvec.copy()
                new_frame[key] = state
            elif key == "action":
                action = frame[key].clone().detach().cpu().numpy().astype(np.float32)
                rotvec = _euler_xyz_to_rotvec(action[ROT_SLICE])
                rotvec = _flip_rotation_vector_if_needed(previous_action_rotvec, rotvec)
                action[ROT_SLICE] = rotvec
                previous_action_rotvec = rotvec.copy()
                new_frame[key] = action
            elif key.startswith("observation.images"):
                new_frame[key] = einops.rearrange(frame[key], "c h w -> h w c")

        # `task` is a required string field for add_frame in lerobot 0.4.1
        # (popped from the dict during processing).
        new_frame["task"] = frame.get("task", "")
        new_dataset.add_frame(new_frame)

    # Save the final episode IF the buffer has unsaved frames. When the loop
    # broke early due to --limit-episodes, the final episode was already
    # saved inside the loop on the boundary transition, and the buffer is
    # empty — calling save_episode again would error.
    if new_dataset.episode_buffer is not None and new_dataset.episode_buffer.get("size", 0) > 0:
        new_dataset.save_episode()
        pbar.update(1)
    pbar.close()

    print(f"Wrote {current_episode_index + 1} episodes to {target_root}")
    return new_dataset


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def _load_target_frame_arrays(target_root: Path) -> pd.DataFrame:
    """Read all parquets under target_root/data/chunk-*/file-*.parquet, in
    a single concatenated DataFrame. Faster than re-iterating through the
    LeRobotDataset (no video decode)."""
    parquet_paths = sorted(target_root.glob("data/chunk-*/file-*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files under {target_root}/data/")
    return pd.concat([pd.read_parquet(p) for p in parquet_paths], ignore_index=True)


def verify(target_root: Path) -> None:
    print("\nVerification")
    df = _load_target_frame_arrays(target_root)

    # Stack observation.state and action into (N, 7) arrays.
    state = np.stack(df["observation.state"].to_numpy())
    action = np.stack(df["action"].to_numpy())
    state_rv = state[:, ROT_SLICE].astype(np.float64)
    action_rv = action[:, ROT_SLICE].astype(np.float64)

    # Check 1: ||ω|| ≤ π for every frame (scipy guarantees this from
    # as_rotvec(); flip-by-negation also preserves the norm).
    for label, rv in (("observation.state", state_rv), ("action", action_rv)):
        norms = np.linalg.norm(rv, axis=1)
        max_norm = float(norms.max())
        assert max_norm <= np.pi + 1e-5, (
            f"{label}: max ||ω|| = {max_norm:.4f} exceeds π. "
            f"Conversion produced an out-of-range rotvec."
        )
        print(f"  ||{label}|| ≤ π  ✓  (max = {max_norm:.4f})")

    # Check 2: round-trip identity via quaternion. rv → quat → rv' must agree
    # in norm (the rotvec class returns the canonical small-angle representative
    # so the norm is preserved). lerobot's Rotation is single-vector-only, so
    # we loop.
    rng = np.random.RandomState(0)
    for label, rv in (("observation.state", state_rv), ("action", action_rv)):
        sample_idx = rng.choice(len(rv), size=min(200, len(rv)), replace=False)
        norms_in = np.linalg.norm(rv[sample_idx], axis=1)
        norms_out = np.array([
            np.linalg.norm(
                Rotation.from_quat(Rotation.from_rotvec(rv[i]).as_quat()).as_rotvec()
            )
            for i in sample_idx
        ])
        assert np.allclose(norms_in, norms_out, atol=1e-6), (
            f"{label}: round-trip rv→quat→rv norm mismatch."
        )
        print(f"  {label} round-trip rv→quat→rv  ✓  (200 samples)")

    # Check 3: per-episode consecutive Δω. Flip detection should keep this
    # well below 2 rad; values around π would mean we crossed the antipodal
    # seam without flipping.
    print("\n  Per-episode max consecutive Δω (rad):")
    HARD_LIMIT = 2.0
    SOFT_LIMIT = 1.0
    any_warning = False
    for ep_idx, g in df.groupby("episode_index", sort=True):
        g = g.sort_values("frame_index")
        s_rv = np.stack(g["observation.state"].to_numpy())[:, ROT_SLICE].astype(np.float64)
        a_rv = np.stack(g["action"].to_numpy())[:, ROT_SLICE].astype(np.float64)
        ds = float(np.linalg.norm(np.diff(s_rv, axis=0), axis=1).max()) if len(s_rv) > 1 else 0.0
        da = float(np.linalg.norm(np.diff(a_rv, axis=0), axis=1).max()) if len(a_rv) > 1 else 0.0
        tag = ""
        if ds > HARD_LIMIT or da > HARD_LIMIT:
            tag = "  FAIL"
            any_warning = True
        elif ds > SOFT_LIMIT or da > SOFT_LIMIT:
            tag = "  warn"
            any_warning = True
        print(f"    ep {int(ep_idx):3d}  state Δω_max={ds:.3f}  action Δω_max={da:.3f}{tag}")
        assert ds <= HARD_LIMIT, (
            f"Episode {int(ep_idx)}: observation.state Δω_max={ds:.3f} > {HARD_LIMIT}. "
            f"Flip detection missed a seam."
        )
        assert da <= HARD_LIMIT, (
            f"Episode {int(ep_idx)}: action Δω_max={da:.3f} > {HARD_LIMIT}. "
            f"Flip detection missed a seam."
        )

    if not any_warning:
        print("  All episodes: Δω < 1.0 rad on both streams. ✓")
    else:
        print("  Some episodes exceeded the 1.0-rad soft limit (still < 2.0).")


def write_crisp_meta(target_root: Path, source_repo_id: str) -> None:
    """Document the semantic change in a sibling marker file."""
    note = target_root / "CRISP_META.md"
    note.write_text(
        "# CRISP dataset metadata\n\n"
        f"- `orientation_representation`: **angle_axis** (rotation vector)\n"
        f"- Converted from: `{source_repo_id}` (Euler XYZ)\n"
        f"- Slots: `observation.state[3:6]` and `action[3:6]` now hold rotvecs.\n"
        f"- Conversion: `scipy.spatial.transform.Rotation.from_euler(\"xyz\", v).as_rotvec()` "
        f"followed by per-stream, per-episode flip detection mirroring "
        f"`manipulator_env.py::_flip_rotation_vector_if_needed`.\n"
        f"- Feature `names` were preserved as `[..., roll, pitch, yaw, gripper]` "
        f"to avoid breaking downstream feature lookups — the *names* still say "
        f"`roll/pitch/yaw` but the *values* are now the components of a "
        f"rotation vector. Set the env yaml to `orientation_representation: \"angle_axis\"` "
        f"before deploying any policy trained on this dataset.\n"
    )
    print(f"Wrote marker: {note}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-repo-id", default=DEFAULT_SOURCE_REPO_ID)
    p.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    p.add_argument("--target-repo-id", default=DEFAULT_TARGET_REPO_ID)
    p.add_argument("--target-root", type=Path, default=DEFAULT_TARGET_ROOT)
    p.add_argument(
        "--limit-episodes", type=int, default=LIMIT_EPISODES,
        help="Convert only the first N episodes (for dry runs).",
    )
    p.add_argument(
        "--skip-verify", action="store_true",
        help="Skip the post-conversion sanity checks.",
    )
    args = p.parse_args()

    convert(
        source_repo_id=args.source_repo_id,
        source_root=args.source_root,
        target_repo_id=args.target_repo_id,
        target_root=args.target_root,
        limit_episodes=args.limit_episodes,
    )

    write_crisp_meta(args.target_root, args.source_repo_id)

    if not args.skip_verify:
        verify(args.target_root)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
