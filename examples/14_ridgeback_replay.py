#!/usr/bin/env python3
"""Replay a recorded LeRobot v3 episode on the real UR10e + Ridgeback.

Flow:

  1. Pick a dataset (default: newest ridgeback* in the LeRobot cache; --pick
     forces the InquirerPy picker; --repo-id skips it).
  2. Pick an episode (default: 0; picker if multiple episodes exist).
  3. Slowly move the arm to the FIRST joint configuration of the picked
     episode using joint_trajectory_controller (env.home(home_config=...)).
  4. Switch to cartesian_controller (CRISP).
  5. Replay the recorded /target_pose stream at the recording fps by calling
     env.robot.set_target(...) once per frame; the env's own publisher pushes
     to /target_pose at robot.config.publish_frequency. Gripper targets are
     mirrored via env.gripper.set_target(...) at the same cadence.
  6. After the loop, return to the env's canonical home pose. Skip with
     --no-home if you want to inspect the final replay frame.

Gripper convention WARNING:
    The Ridgeback dataset is internally inconsistent on gripper convention
    (see docs/ridgeback_target_pose_ownership.md):

      - `action[6]`                          uses crisp_py convention
        (1 = open, 0 = closed) — same as `env.gripper.set_target`.
      - `observation.state.gripper`          uses LeRobot convention
        (`1 - gripper.value`, so 0 = open, 1 = closed for the ridgeback YAML
        which has min_value=0.8, max_value=0.0).
      - `observation.state.gripper_target`   same LeRobot flip, AND falls
        back to `gripper.value` for any frame before the first mocap grip
        command lands (see manipulator_env.py:367-378), so its prefix is
        not actually a target.

    We default to `--gripper-source action` because that column is the
    literal command the recorder issued and it can be passed straight to
    `env.gripper.set_target` without any sign flip. The `target` and `obs`
    sources still work but are flipped internally to crisp_py convention
    before being sent to the gripper.

Usage:
    cd Yunfei/crisp_gym
    pixi run -e jazzy-lerobot python examples/14_ridgeback_replay.py
    pixi run -e jazzy-lerobot python examples/14_ridgeback_replay.py --pick
    pixi run -e jazzy-lerobot python examples/14_ridgeback_replay.py \\
        --repo-id ridgeback_223335 --episode-idx 0 --yes
    pixi run -e jazzy-lerobot python examples/14_ridgeback_replay.py --speed 0.5

Prerequisites:
    - Robot up, controller_manager running.
    - cartesian_controller and joint_trajectory_controller both loaded
      (toggle_controller.py manages the switch but make sure both exist).
    - Recording in ~/.cache/huggingface/lerobot/<repo_id>/ written by
      13_ridgeback_mocap_record.py (LeRobot v3.0 layout).
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import qos_profile_system_default
from scipy.spatial.transform import Rotation

from crisp_gym.envs.manipulator_env import make_env
from crisp_gym.util.setup_logger import setup_logging
from crisp_py.utils.geometry import Pose

LEROBOT_CACHE = Path.home() / ".cache/huggingface/lerobot"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset discovery / loading
# ---------------------------------------------------------------------------

def discover_datasets(root: Path) -> list[dict]:
    """Walk the LeRobot cache and return one row per v3 dataset, mtime sorted desc."""
    rows: list[dict] = []
    if not root.exists():
        return rows
    for info_path in sorted(root.rglob("meta/info.json")):
        ds = info_path.parent.parent
        try:
            info = json.loads(info_path.read_text())
        except Exception:
            continue
        n_eps = int(info.get("total_episodes", 0))
        ep_dir = ds / "meta" / "episodes"
        if ep_dir.exists():
            try:
                parts = sorted(ep_dir.rglob("file-*.parquet"))
                if parts:
                    n_eps = sum(
                        len(pd.read_parquet(p, columns=["episode_index"])) for p in parts
                    )
            except Exception:
                pass
        rows.append({
            "repo_id": str(ds.relative_to(root)),
            "version": str(info.get("codebase_version", "v?")),
            "fps": info.get("fps"),
            "n_eps": n_eps,
            "mtime": ds.stat().st_mtime,
        })
    rows.sort(key=lambda r: r["mtime"], reverse=True)
    return rows


def load_episodes_meta(dataset_dir: Path) -> pd.DataFrame:
    """Read meta/episodes/.../file-*.parquet (v3) into a single DataFrame."""
    parts = sorted((dataset_dir / "meta" / "episodes").rglob("file-*.parquet"))
    if not parts:
        raise FileNotFoundError(
            f"No v3 episode metadata under {dataset_dir / 'meta' / 'episodes'}. "
            "This script only supports LeRobot v3 datasets."
        )
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def load_episode_frames(
    dataset_dir: Path, info: dict, episodes_df: pd.DataFrame, episode_idx: int
) -> pd.DataFrame:
    """Load the per-frame parquet for one episode (filtered by episode_index)."""
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
# Interactive pickers (InquirerPy)
# ---------------------------------------------------------------------------

def pick_dataset(rows: list[dict]) -> str:
    from InquirerPy import inquirer
    from InquirerPy.base.control import Choice

    if not rows:
        raise FileNotFoundError(f"No LeRobot datasets in {LEROBOT_CACHE}")

    default = next(
        (r["repo_id"] for r in rows if r["repo_id"].startswith("ridgeback")),
        rows[0]["repo_id"],
    )

    choices = [
        Choice(
            value=r["repo_id"],
            name=(
                f'{r["repo_id"]:<35} '
                f' {r["n_eps"]:>3} ep '
                f' {r["fps"]} Hz '
                f' {r["version"]}'
            ),
        )
        for r in rows
    ]
    return inquirer.select(
        message="Pick a recorded dataset:",
        choices=choices,
        default=default,
    ).execute()


def pick_episode(episodes_df: pd.DataFrame, fps: float) -> int:
    from InquirerPy import inquirer
    from InquirerPy.base.control import Choice

    if len(episodes_df) == 1:
        return int(episodes_df.iloc[0]["episode_index"])

    choices = []
    for _, row in episodes_df.iterrows():
        idx = int(row["episode_index"])
        length = int(row["length"])
        duration = length / fps
        tasks = row.get("tasks", "")
        if hasattr(tasks, "tolist"):
            tasks = tasks.tolist()
        choices.append(
            Choice(
                value=idx,
                name=f"episode {idx:3d}   {length:>5} frames   {duration:6.2f}s   {tasks}",
            )
        )
    return inquirer.select(
        message="Pick an episode:",
        choices=choices,
        default=int(episodes_df.iloc[0]["episode_index"]),
    ).execute()


# ---------------------------------------------------------------------------
# Per-frame extraction
# ---------------------------------------------------------------------------

def first_joint_config(df: pd.DataFrame) -> list[float]:
    arr = np.asarray(df.iloc[0]["observation.state.joints"], dtype=np.float64)
    return arr.tolist()


def ee_target_at(row: pd.Series, source: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (xyz, rpy) for one frame from the chosen field.

    `source="target"` -> observation.state.target (6-vec, x/y/z/r/p/y)
    `source="action"` -> action[0:6]              (7-vec, last is gripper)
    """
    if source == "target":
        arr = np.asarray(row["observation.state.target"], dtype=np.float64)
    elif source == "action":
        arr = np.asarray(row["action"], dtype=np.float64)
    else:
        raise ValueError(f"unknown target source: {source}")
    return arr[:3].copy(), arr[3:6].copy()


def gripper_target_at(row: pd.Series, source: str) -> float:
    """Return the gripper target for one frame in **crisp_py convention**.

    crisp_py: 1 = open, 0 = closed. This is what `env.gripper.set_target`
    consumes directly, with no extra flipping in the replay loop.

    Sources:
      - "action": read `row["action"][6]`. Already crisp_py convention
        (the recorder writes `float(env.gripper.target)` straight in).
      - "target": read `row["observation.state.gripper_target"][0]`. The
        env stores this as `1 - gripper.target` (LeRobot convention), so
        we flip it back to crisp_py here. Note the recorder falls back to
        `gripper.value` before the first mocap grip command lands, so the
        prefix of the column is contaminated with the measured state.
      - "obs": read `row["observation.state.gripper"][0]`. Same LeRobot
        flip as above; this is the *measured* gripper state, not a command.
    """
    if source == "action":
        arr = np.asarray(row["action"], dtype=np.float64)
        return float(arr[6])
    elif source == "target":
        arr = np.asarray(row["observation.state.gripper_target"], dtype=np.float64)
        return float(1.0 - arr.flatten()[0])
    elif source == "obs":
        arr = np.asarray(row["observation.state.gripper"], dtype=np.float64)
        return float(1.0 - arr.flatten()[0])
    else:
        raise ValueError(f"unknown gripper source: {source}")


# ---------------------------------------------------------------------------
# Topic-ownership patch (inverse of 13_ridgeback_mocap_record's silencer)
# ---------------------------------------------------------------------------

def enable_env_target_pose_publishing(env) -> None:
    """Make the env's `crisp_py.Robot` own `/target_pose` again.

    `ur10e_ridgeback_env.yaml` sets `publish_target_pose: false` because the
    mocap recording flow (`13_ridgeback_mocap_record.py`) needs the external
    `track_mocap.py` node to be the sole publisher on `/target_pose`. With
    that flag set, `crisp_py.Robot.__init__` (`crisp_py/robot/robot.py`):

      1. does not create `_target_pose_publisher`,
      2. does not create the 20 Hz publish timer,
      3. instead subscribes to `/target_pose` and writes incoming messages
         into `_target_pose` via `_callback_external_target_pose`.

    For replay we need the inverse: the script calls `env.robot.set_target`
    once per frame and relies on the 20 Hz timer to forward `_target_pose`
    to `cartesian_controller`. Without this patch, `set_target` only updates
    an in-memory buffer that nothing publishes (and the external subscription
    would clobber on the next stray `/target_pose`), so the arm never moves
    while the gripper still does — that is the exact symptom we hit.

    This helper destroys the external subscription, creates the publisher
    and the publish timer that `Robot.__init__` would have created if the
    flag had been `true`, and flips the flag for any code that inspects it.
    Must be called BEFORE `wait_until_ready()` and BEFORE the first
    `set_target` call. See `docs/ridgeback_target_pose_ownership.md`.
    """
    robot = env.robot

    sub = getattr(robot, "_target_pose_subscriber", None)
    if sub is not None:
        robot.node.destroy_subscription(sub)
        robot._target_pose_subscriber = None

    if getattr(robot, "_target_pose_publisher", None) is None:
        robot._target_pose_publisher = robot.node.create_publisher(
            PoseStamped,
            robot.config.target_pose_topic,
            qos_profile_system_default,
        )

    robot.node.create_timer(
        1.0 / robot.config.publish_frequency,
        robot._callback_publish_target_pose,
        ReentrantCallbackGroup(),
    )

    robot.config.publish_target_pose = True


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def confirm(prompt: str, assume_yes: bool) -> bool:
    if assume_yes:
        print(f"  {prompt} [auto-yes]")
        return True
    try:
        ans = input(f"  {prompt} [Y/n] ").strip().lower()
    except EOFError:
        return False
    return ans in ("", "y", "yes")


def print_summary(
    repo_id: str,
    info: dict,
    episode_idx: int,
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    fps_info = float(info.get("fps", 20))
    n = len(df)
    duration = n / fps_info
    first_pos, first_rpy = ee_target_at(df.iloc[0], args.target_source)
    last_pos, last_rpy = ee_target_at(df.iloc[-1], args.target_source)
    first_grip = gripper_target_at(df.iloc[0], args.gripper_source)
    last_grip = gripper_target_at(df.iloc[-1], args.gripper_source)
    first_joints = first_joint_config(df)

    joints_str = ", ".join(f"{j:+.3f}" for j in first_joints)
    print()
    print("=== Replay summary ===")
    print(f"  dataset:        {repo_id}  ({info.get('codebase_version', 'v?')})")
    print(f"  episode_index:  {episode_idx}")
    print(f"  frames:         {n}")
    print(f"  duration:       {duration:.2f} s  @ {fps_info} Hz")
    print(f"  speed:          {args.speed}x  ({duration/args.speed:.2f} s actual)")
    print(f"  target source:  {args.target_source}  (observation.state.target | action[:6])")
    print(f"  gripper source: {args.gripper_source}  (action[6] | observation.state.gripper_target | observation.state.gripper)")
    print(f"  first joints:   [{joints_str}]")
    print(
        f"  first EE pos:   ({first_pos[0]:+.3f}, {first_pos[1]:+.3f}, {first_pos[2]:+.3f})  "
        f"rpy=({first_rpy[0]:+.3f}, {first_rpy[1]:+.3f}, {first_rpy[2]:+.3f})"
    )
    print(
        f"  last  EE pos:   ({last_pos[0]:+.3f}, {last_pos[1]:+.3f}, {last_pos[2]:+.3f})  "
        f"rpy=({last_rpy[0]:+.3f}, {last_rpy[1]:+.3f}, {last_rpy[2]:+.3f})"
    )
    print(
        f"  gripper:        {first_grip:.2f} -> {last_grip:.2f}  (crisp_py: 1=open, 0=closed)"
    )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--repo-id", default=None,
                        help="LeRobot dataset name under ~/.cache/huggingface/lerobot. "
                             "If omitted, picks newest ridgeback* (or opens picker).")
    parser.add_argument("--episode-idx", type=int, default=None,
                        help="Episode index to replay (default: 0 if dataset has one episode, "
                             "otherwise opens picker).")
    parser.add_argument("--pick", action="store_true",
                        help="Force the InquirerPy picker even when --repo-id is given.")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Replay speed multiplier (default 1.0 = real time).")
    parser.add_argument("--target-source", choices=["target", "action"], default="target",
                        help="Which parquet column is the EE target (default: target = "
                             "observation.state.target).")
    parser.add_argument("--gripper-source", choices=["action", "target", "obs"],
                        default="action",
                        help="Which parquet column is the gripper command. "
                             "Default 'action' reads action[6] (the literal command in "
                             "crisp_py convention). 'target' reads observation.state.gripper_target "
                             "(LeRobot convention, flipped internally). 'obs' reads "
                             "observation.state.gripper (the measured state, not a command).")
    parser.add_argument("--no-home", action="store_true",
                        help="Skip the final return-to-home phase.")
    parser.add_argument("--yes", action="store_true",
                        help="Skip the confirm prompt before any motion.")
    parser.add_argument("--env-config", default="ur10e_ridgeback_env",
                        help="crisp_gym env config name (default: ur10e_ridgeback_env).")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    args = parser.parse_args()

    setup_logging(level=args.log_level)

    # ----- Phase 1a: resolve dataset -----
    rows = discover_datasets(LEROBOT_CACHE)
    if not rows:
        logger.error("No LeRobot datasets found in %s", LEROBOT_CACHE)
        sys.exit(1)

    if args.repo_id is None:
        # Default: newest ridgeback*. Picker if --pick or if there's more than one.
        ridgebacks = [r for r in rows if r["repo_id"].startswith("ridgeback")]
        if args.pick or len(ridgebacks) > 1 or not ridgebacks:
            repo_id = pick_dataset(rows)
        else:
            repo_id = ridgebacks[0]["repo_id"]
            logger.info("Default dataset: %s", repo_id)
    else:
        repo_id = args.repo_id
        if args.pick:
            repo_id = pick_dataset(rows)

    dataset_dir = LEROBOT_CACHE / repo_id
    if not dataset_dir.exists():
        logger.error("Dataset not found: %s", dataset_dir)
        sys.exit(1)

    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    if not str(info.get("codebase_version", "")).startswith("v3"):
        logger.error(
            "Dataset %s is codebase_version %r, but this script only supports v3.",
            repo_id, info.get("codebase_version"),
        )
        sys.exit(1)

    episodes_df = load_episodes_meta(dataset_dir)

    # ----- Phase 1b: resolve episode -----
    if args.episode_idx is None:
        if args.pick or len(episodes_df) > 1:
            episode_idx = pick_episode(episodes_df, float(info.get("fps", 20)))
        else:
            episode_idx = int(episodes_df.iloc[0]["episode_index"])
    else:
        episode_idx = args.episode_idx

    # ----- Phase 1c: load frames -----
    df = load_episode_frames(dataset_dir, info, episodes_df, episode_idx)
    if len(df) == 0:
        logger.error("Episode %d has zero frames in %s", episode_idx, repo_id)
        sys.exit(1)

    print_summary(repo_id, info, episode_idx, df, args)
    if not confirm("Replay this episode?", args.yes):
        logger.info("Aborted by user.")
        sys.exit(0)

    # ----- Phase 2..5: drive the robot -----
    logger.info("Creating environment: %s", args.env_config)
    env = make_env(env_type=args.env_config, control_type="cartesian", namespace="")

    # ur10e_ridgeback_env.yaml has `publish_target_pose: false` (mocap owns
    # the topic during recording). Replay needs the env's Robot client to
    # own /target_pose instead, so re-enable the publisher + 20 Hz timer
    # before anything else touches the robot.
    # See docs/ridgeback_target_pose_ownership.md.
    logger.info("Enabling env's /target_pose publisher (replay owns the topic)")
    enable_env_target_pose_publishing(env)

    logger.info("Waiting for robot to be ready...")
    env.wait_until_ready()
    logger.info("Robot ready.")

    fps_info = float(info.get("fps", 20))
    publish_rate = fps_info * float(args.speed)
    if publish_rate <= 0:
        logger.error("Computed publish rate is %.3f Hz; pick a positive --speed", publish_rate)
        env.close()
        rclpy.shutdown()
        sys.exit(1)

    interrupted = False
    try:
        # Phase 2: slow JTC move to first joint config
        first_joints = first_joint_config(df)
        logger.info("Phase 2: slow move to first joint config via JTC")
        env.home(home_config=first_joints, blocking=True)
        logger.info("Phase 2: at first joint config")

        # Phase 3: switch to cartesian controller
        logger.info("Phase 3: switching to cartesian controller")
        env.switch_controller("cartesian")

        # Phase 4: replay loop
        logger.info(
            "Phase 4: replaying %d frames at %.2f Hz (%.2f x recording fps %.1f)",
            len(df), publish_rate, args.speed, fps_info,
        )
        rate = env.robot.node.create_rate(publish_rate)
        last_log = time.monotonic()
        for i, row in df.iterrows():
            pos, rpy = ee_target_at(row, args.target_source)
            pose = Pose(
                position=np.asarray(pos, dtype=np.float64),
                orientation=Rotation.from_euler("xyz", rpy),
            )
            env.robot.set_target(pose=pose)

            if env.gripper is not None:
                grip_crisp = float(np.clip(gripper_target_at(row, args.gripper_source), 0.0, 1.0))
                env.gripper.set_target(grip_crisp)

            now = time.monotonic()
            if now - last_log > 1.0:
                logger.info("  frame %d / %d", i + 1, len(df))
                last_log = now
            rate.sleep()
        logger.info("Phase 4: replay loop done")

    except KeyboardInterrupt:
        interrupted = True
        logger.warning(
            "Interrupted by user. Robot frozen at last commanded pose. "
            "Skipping return-to-home."
        )
    except Exception:
        logger.exception("Replay failed")
        raise
    else:
        # Phase 5: return home (only on a clean replay)
        if not args.no_home:
            try:
                logger.info("Phase 5: returning to home pose")
                env.home(blocking=True)
                logger.info("Replay complete. Robot homed.")
            except KeyboardInterrupt:
                logger.warning("Interrupted during return-to-home.")
        else:
            logger.info("Replay complete. Skipping return-to-home (--no-home).")
    finally:
        try:
            env.close()
        except Exception:
            logger.exception("env.close() raised")
        if rclpy.ok():
            rclpy.shutdown()

    sys.exit(1 if interrupted else 0)


if __name__ == "__main__":
    main()
