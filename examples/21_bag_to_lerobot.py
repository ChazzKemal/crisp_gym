#!/usr/bin/env python3
"""Convert a rosbag (recorded with 20_rosbag_record.py or `ros2 bag record`)
into a LeRobot v3 dataset, using actual ROS message timestamps.

Why: live recording via 16_camera_only_record.py samples the camera's
shared buffer at a Python tick rate, which creates duplicate frames when
the tick happens to read mid-decode (~10% duplicate rate observed). This
offline converter reads every camera message from the bag, picks the
nearest one per target tick using its real timestamp, and emits a
LeRobot v3 dataset with the same feature schema as the live recorder —
so existing training code just works. Duplicate rate with this path is
typically <1%.

Feature schema mirrors 16_camera_only_record.py (single-camera) and extends
naturally to N cameras (multi-camera, e.g. ur10e_ridgeback_dual_cam_env):
    observation.images.<name>                 video           (one per camera)
    observation.timestamps.wall               float64[1]      (bag recv time)
    observation.timestamps.<name>_header      float64[1]      (per-camera msg.header.stamp)
    observation.state.joints                  float32[nq]     (if --no-joints off)
    observation.state.cartesian               float32[6]      (if --no-pose   off)
    observation.state.gripper                  float32[1]      (if --no-gripper-state off)
    observation.state                         float32[sum]
    action                                    float32[7]      if --with-action
                                              float32[1]      otherwise (dummy zero)

Multi-camera behaviour:
  - When --camera-topic is omitted (default), the converter scans the bag's
    topic metadata for every `sensor_msgs/msg/CompressedImage` topic and
    converts them all. Camera names are derived from the topic's first
    path segment (e.g. `/d405/...` → `d405`).
  - To explicitly pick a subset, pass `--camera-topic /A/... /B/...`.
  - To override the derived names, pass `--camera-name a b` aligned with
    `--camera-topic`.
  - A single-camera bag with topic `/camera/...` resolves to name `"camera"`,
    yielding `observation.images.camera` + `observation.timestamps.camera_header`
    — byte-identical to the pre-multi-camera schema.

Usage:
    pixi run -e jazzy-lerobot python examples/21_bag_to_lerobot.py \\
        --bag ./rosbag_recordings/bag_20260417_214616 \\
        --repo-id bag_converted_demo --fps 30

    # Include teleop action (requires /target_pose + /target_gripper_state
    # to be present in the bag):
    pixi run -e jazzy-lerobot python examples/21_bag_to_lerobot.py \\
        --bag ./rosbag_recordings/bag_X --repo-id teleop_converted \\
        --with-action --fps 30

    # Append to an existing dataset as another episode:
    pixi run -e jazzy-lerobot python examples/21_bag_to_lerobot.py \\
        --bag ./rosbag_recordings/bag_Y --repo-id teleop_converted \\
        --with-action --fps 30 --resume
"""

import argparse
import bisect
import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


logger = logging.getLogger(__name__)


DEFAULT_JOINT_NAMES = [
    "arm_0_shoulder_pan_joint",
    "arm_0_shoulder_lift_joint",
    "arm_0_elbow_joint",
    "arm_0_wrist_1_joint",
    "arm_0_wrist_2_joint",
    "arm_0_wrist_3_joint",
]

# Gripper normalization — matches 16_camera_only_record.py:275-283.
# track_mocap.py publishes RAW joint angles on /target_gripper_state:
# 0.0 = trigger released (physical gripper OPEN),
# 0.8 = trigger pressed  (physical gripper CLOSED).
# Normalized convention in the dataset: 0.0 = closed, 1.0 = open
# (same as env.gripper.set_target's docstring and the replay script).
# Mapping:  raw 0.0 -> normalized 1.0 (open)
#           raw 0.8 -> normalized 0.0 (close)
GRIP_RAW_CLOSED = 0.8  # raw value that corresponds to normalized 0.0 (closed)
GRIP_RAW_OPEN = 0.0    # raw value that corresponds to normalized 1.0 (open)

# Default normalized value before the first /target_gripper_state event
# in the bag — physical teleop default is trigger released = gripper open,
# which is normalized 1.0 in this convention.
GRIP_NORMALIZED_DEFAULT = 1.0


def _normalize_grip(raw: float) -> float:
    return float(
        np.clip((raw - GRIP_RAW_CLOSED) / (GRIP_RAW_OPEN - GRIP_RAW_CLOSED), 0.0, 1.0)
    )


# ---------- Camera topic / name helpers ----------

def _camera_name_from_topic(topic: str) -> str:
    """Derive a stable camera key from a topic path.

    Convention: take the first non-empty path segment.
      `/camera/color/image_raw/compressed`             → 'camera'
      `/d405/camera/color/image_rect_raw/compressed`   → 'd405'
      `/right_wrist_camera/color/image_rect_raw/compressed` → 'right_wrist_camera'

    Falls back to 'camera' for empty / malformed input. Caller is
    responsible for de-duplicating collisions (e.g. two topics that share
    the same first segment).
    """
    parts = [p for p in topic.split("/") if p]
    return parts[0] if parts else "camera"


def _dedupe_camera_names(names: list[str]) -> list[str]:
    """Return a list of unique names, suffixing duplicates with _2 / _3 / …

    Stable: first occurrence keeps the bare name; subsequent collisions get
    a 1-based counter suffix.
    """
    seen: dict[str, int] = {}
    out: list[str] = []
    for n in names:
        if n not in seen:
            seen[n] = 1
            out.append(n)
        else:
            seen[n] += 1
            out.append(f"{n}_{seen[n]}")
    return out


def _autodetect_camera_topics(types: dict[str, str]) -> list[str]:
    """Return every RGB CompressedImage topic in the bag, sorted alphabetically.

    Filters on both message type (`sensor_msgs/msg/CompressedImage`) AND the
    `/compressed` topic suffix. The suffix check is what separates RGB-JPEG
    streams from depth-PNG (`/compressedDepth`) and other non-RGB siblings —
    both share the same msg type, only differing in the `.format` field of
    the message body (which we can't inspect cheaply from topic metadata).

    Users who want a non-`/compressed` variant in the dataset can still
    force it via an explicit `--camera-topic ...` list.
    """
    return sorted(
        t for t, ty in types.items()
        if ty == "sensor_msgs/msg/CompressedImage" and t.endswith("/compressed")
    )


# ---------- Bag reading ----------

def _open_reader(bag_dir: Path):
    import rosbag2_py
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader.open(storage_options, converter_options)
    return reader


def _topic_types(bag_dir: Path) -> dict[str, str]:
    reader = _open_reader(bag_dir)
    return {t.name: t.type for t in reader.get_all_topics_and_types()}


def _collect_topic(bag_dir: Path, topic: str) -> list[tuple[int, bytes]]:
    """Return list of (bag_ts_ns, serialized_bytes) sorted by bag_ts_ns."""
    import rosbag2_py
    reader = _open_reader(bag_dir)
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    out: list[tuple[int, bytes]] = []
    while reader.has_next():
        tn, data, ts = reader.read_next()
        if tn == topic:
            out.append((ts, data))
    out.sort(key=lambda r: r[0])
    return out


def _nearest_index(ts_list_sorted: list[int], target_ts: int) -> int:
    """Index of the element in ts_list_sorted closest to target_ts."""
    i = bisect.bisect_left(ts_list_sorted, target_ts)
    if i == 0:
        return 0
    if i >= len(ts_list_sorted):
        return len(ts_list_sorted) - 1
    before, after = ts_list_sorted[i - 1], ts_list_sorted[i]
    return i if (after - target_ts) < (target_ts - before) else i - 1


# ---------- Image resize (matches crisp_py Camera._resize_with_aspect_ratio) ----------

def _fit_crop(image: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    h, w = image.shape[:2]
    if h == target_h and w == target_w:
        return image
    scale = max(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    start_x = (new_w - target_w) // 2
    start_y = (new_h - target_h) // 2
    return resized[start_y : start_y + target_h, start_x : start_x + target_w]


# ---------- Conversion ----------

def convert(args: argparse.Namespace) -> None:
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import CompressedImage, JointState
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import Float32
    from cv_bridge import CvBridge
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    try:
        # Newer lerobot versions accept `task` kwarg on add_frame.
        import inspect
        _ADD_FRAME_HAS_TASK = "task" in inspect.signature(
            LeRobotDataset.add_frame
        ).parameters
    except Exception:
        _ADD_FRAME_HAS_TASK = False

    bag_dir: Path = args.bag
    if not bag_dir.exists():
        logger.error("Bag directory %s does not exist.", bag_dir)
        sys.exit(1)

    types = _topic_types(bag_dir)
    logger.info("Topics in bag: %s", sorted(types))

    # --- Validate topic availability ---
    def _require(topic: str, ros_type: str, name: str) -> None:
        if topic not in types:
            raise SystemExit(f"Bag missing {name} topic {topic!r}")
        if types[topic] != ros_type:
            raise SystemExit(
                f"{name} topic {topic!r} is {types[topic]}, expected {ros_type}"
            )

    use_camera = not args.no_camera
    use_joints = not args.no_joints
    use_pose = not args.no_pose
    use_gripper_state = not args.no_gripper_state
    use_action = args.with_action

    # --- Resolve camera topics + names ---
    # Three modes:
    #   1. --no-camera               → no image features written.
    #   2. --camera-topic A [B ...]  → use the explicit list (require each).
    #   3. neither                   → auto-detect every CompressedImage topic
    #                                  in the bag.
    # Names come from --camera-name (one per topic, same order) if given,
    # otherwise derived from each topic's first path segment.
    camera_topics: list[str] = []
    camera_names: list[str] = []
    if use_camera:
        if args.camera_topic:
            camera_topics = list(args.camera_topic)
        else:
            camera_topics = _autodetect_camera_topics(types)
            if not camera_topics:
                logger.warning(
                    "No sensor_msgs/msg/CompressedImage topics found in bag — "
                    "disabling camera. Pass --camera-topic to override."
                )
                use_camera = False

    if use_camera:
        if args.camera_name:
            if len(args.camera_name) != len(camera_topics):
                raise SystemExit(
                    f"--camera-name has {len(args.camera_name)} entries but "
                    f"--camera-topic has {len(camera_topics)}; they must align."
                )
            camera_names = _dedupe_camera_names(list(args.camera_name))
        else:
            camera_names = _dedupe_camera_names(
                [_camera_name_from_topic(t) for t in camera_topics]
            )
        for topic, name in zip(camera_topics, camera_names):
            _require(topic, "sensor_msgs/msg/CompressedImage", f"camera/{name}")
        logger.info(
            "Cameras (%d): %s",
            len(camera_topics),
            ", ".join(f"{n} ← {t}" for n, t in zip(camera_names, camera_topics)),
        )

    if use_joints:
        _require(args.joint_topic, "sensor_msgs/msg/JointState", "joints")
    if use_pose:
        _require(args.pose_topic, "geometry_msgs/msg/PoseStamped", "pose")
    if use_action:
        _require(args.target_pose_topic, "geometry_msgs/msg/PoseStamped", "target pose")
        _require(args.gripper_target_topic, "std_msgs/msg/Float32", "gripper target")

    # Gripper joint state is optional: bags recorded before /gripper/joint_states
    # was in the recorder won't have it — skip gracefully rather than hard-fail.
    if use_gripper_state and args.gripper_state_topic not in types:
        logger.warning(
            "Gripper-state topic %s not in bag — observation.state.gripper omitted "
            "(older bag, or gripper driver not running during recording).",
            args.gripper_state_topic,
        )
        use_gripper_state = False
    if use_gripper_state and types[args.gripper_state_topic] != "sensor_msgs/msg/JointState":
        raise SystemExit(
            f"gripper-state topic {args.gripper_state_topic!r} is "
            f"{types[args.gripper_state_topic]}, expected sensor_msgs/msg/JointState"
        )

    # --- Collect messages per topic ---
    # Topics split by time-window policy:
    #   "continuous": assumed to publish for the full session. Their time
    #                 ranges constrain the dataset window (intersection).
    #   "sparse"    : may only publish on changes (e.g. gripper). We don't let
    #                 them shrink the window; ticks before the first message
    #                 get a default value, later ticks nearest-sample.
    logger.info("Reading messages from bag...")
    records: dict[str, list[tuple[int, bytes]]] = {}
    continuous_keys: list[str] = []
    # Each camera lives under its own "camera:<name>" key in records so the
    # per-frame loop can iterate cameras independently and each contributes
    # to the continuous-window intersection.
    if use_camera:
        for topic, name in zip(camera_topics, camera_names):
            key = f"camera:{name}"
            records[key] = _collect_topic(bag_dir, topic)
            continuous_keys.append(key)
    if use_joints:
        records["joints"] = _collect_topic(bag_dir, args.joint_topic)
        continuous_keys.append("joints")
    if use_pose:
        records["pose"] = _collect_topic(bag_dir, args.pose_topic)
        continuous_keys.append("pose")
    if use_gripper_state:
        # Measured gripper joint angle. Nearest-sample (not a continuous /
        # window-defining topic) so a short or late gripper stream can't
        # shrink the dataset window or hard-fail the conversion.
        records["gripper_state"] = _collect_topic(bag_dir, args.gripper_state_topic)
    if use_action:
        records["target_pose"] = _collect_topic(bag_dir, args.target_pose_topic)
        continuous_keys.append("target_pose")
        # /target_gripper_state is sparse — only fires on grip changes.
        records["gripper"] = _collect_topic(bag_dir, args.gripper_target_topic)

    for k, r in records.items():
        logger.info("  %s: %d msgs", k, len(r))
        if k in continuous_keys and len(r) < 2:
            raise SystemExit(f"Not enough messages on {k!r} to build a dataset.")
    if use_action and len(records.get("gripper", [])) == 0:
        logger.warning(
            "No gripper messages on %s; using default 1.0 (open) for every frame.",
            args.gripper_target_topic,
        )
    if use_gripper_state and len(records.get("gripper_state", [])) == 0:
        logger.warning(
            "No messages on %s; omitting observation.state.gripper.",
            args.gripper_state_topic,
        )
        use_gripper_state = False

    # --- Determine time range from continuous topics only ---
    if not continuous_keys:
        raise SystemExit("No continuous topics enabled; cannot determine time range.")
    starts = [records[k][0][0] for k in continuous_keys]
    ends = [records[k][-1][0] for k in continuous_keys]
    t_start_ns = max(starts)
    t_end_ns = min(ends)
    span_s = (t_end_ns - t_start_ns) * 1e-9
    if span_s <= 1.0 / args.fps:
        raise SystemExit(
            f"Overlap window too short: {span_s:.3f}s. Some topic(s) may not cover the full recording."
        )
    logger.info("Overlap window: %.3f s (from %s)", span_s, ", ".join(continuous_keys))
    logger.info("Target fps: %d  (expected frames: %d)", args.fps, int(span_s * args.fps))

    # --- Pre-compute sorted timestamp arrays for O(log n) lookup ---
    ts_arrays: dict[str, list[int]] = {k: [t for t, _ in v] for k, v in records.items()}

    # --- Build features dict ---
    h, w = args.resolution

    state_components = []
    if use_joints:
        state_components.append(("joints", len(args.joint_names), list(args.joint_names)))
    if use_pose:
        state_components.append(("cartesian", 6, ["x", "y", "z", "roll", "pitch", "yaw"]))
    if use_gripper_state:
        state_components.append(("gripper", 1, ["gripper"]))
    if not state_components:
        state_components.append(("dummy", 1, ["dummy_state"]))

    state_dim = sum(dim for _, dim, _ in state_components)
    all_state_names = [n for _, _, names in state_components for n in names]

    features: dict[str, Any] = {}
    if use_camera:
        for name in camera_names:
            features[f"observation.images.{name}"] = {
                "dtype": "video",
                "shape": (h, w, 3),
                "names": ["height", "width", "channels"],
                "video_info": {
                    "video.fps": args.fps,
                    "video.codec": "av1",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            }
    features["observation.timestamps.wall"] = {
        "dtype": "float64",
        "shape": (1,),
        "names": ["seconds"],
    }
    if use_camera:
        for name in camera_names:
            features[f"observation.timestamps.{name}_header"] = {
                "dtype": "float64",
                "shape": (1,),
                "names": ["seconds"],
            }
    for comp_name, comp_dim, comp_names in state_components:
        features[f"observation.state.{comp_name}"] = {
            "dtype": "float32",
            "shape": (comp_dim,),
            "names": comp_names,
        }
    features["observation.state"] = {
        "dtype": "float32",
        "shape": (state_dim,),
        "names": all_state_names,
    }
    if use_action:
        features["action"] = {
            "dtype": "float32",
            "shape": (7,),
            "names": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
        }
    else:
        features["action"] = {
            "dtype": "float32",
            "shape": (1,),
            "names": ["dummy"],
        }

    # --- Open or create dataset ---
    root = None  # lerobot chooses default cache under ~/.cache/huggingface/lerobot/<repo_id>
    if args.resume:
        logger.info("Loading existing dataset %r to append episode", args.repo_id)
        dataset = LeRobotDataset(repo_id=args.repo_id, root=root)
    else:
        logger.info("Creating new dataset %r at %d fps", args.repo_id, args.fps)
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            fps=args.fps,
            robot_type="camera",
            features=features,
            use_videos=True,
        )

    # --- Generate tick grid and emit frames ---
    bridge = CvBridge()
    tick_period_ns = int(1e9 / args.fps)
    n_ticks = int(span_s * args.fps)
    logger.info("Emitting %d frames...", n_ticks)

    # Per-camera duplicate tracking — when target ticks line up against a
    # camera publishing slower than fps we'll repeat the nearest sample.
    dup_last_camera_idx: dict[str, int | None] = {n: None for n in camera_names}
    duplicate_camera: dict[str, int] = {n: 0 for n in camera_names}

    for tick in range(n_ticks):
        target_ts = t_start_ns + tick * tick_period_ns

        frame: dict[str, Any] = {}
        wall_seconds = target_ts * 1e-9
        frame["observation.timestamps.wall"] = np.array([wall_seconds], dtype=np.float64)

        # ---- Cameras ----
        if use_camera:
            for name in camera_names:
                key = f"camera:{name}"
                cam_idx = _nearest_index(ts_arrays[key], target_ts)
                if cam_idx == dup_last_camera_idx[name]:
                    duplicate_camera[name] += 1
                dup_last_camera_idx[name] = cam_idx
                cam_msg = deserialize_message(records[key][cam_idx][1], CompressedImage)
                header_s = (
                    cam_msg.header.stamp.sec + cam_msg.header.stamp.nanosec * 1e-9
                )
                rgb = bridge.compressed_imgmsg_to_cv2(cam_msg, desired_encoding="rgb8")
                rgb = _fit_crop(rgb, (h, w))
                frame[f"observation.images.{name}"] = rgb
                frame[f"observation.timestamps.{name}_header"] = np.array(
                    [header_s], dtype=np.float64
                )
        # When --no-camera the camera_header keys are simply absent from the
        # features schema, so no NaN-sentinel placeholders are needed.

        # ---- Joints ----
        if use_joints:
            j_idx = _nearest_index(ts_arrays["joints"], target_ts)
            jmsg = deserialize_message(records["joints"][j_idx][1], JointState)
            vals = np.zeros(len(args.joint_names), dtype=np.float32)
            name_to_idx = {n: i for i, n in enumerate(args.joint_names)}
            for name, pos in zip(jmsg.name, jmsg.position):
                if name in name_to_idx:
                    vals[name_to_idx[name]] = pos
            frame["observation.state.joints"] = vals

        # ---- Pose ----
        if use_pose:
            p_idx = _nearest_index(ts_arrays["pose"], target_ts)
            pmsg = deserialize_message(records["pose"][p_idx][1], PoseStamped)
            p = pmsg.pose.position
            q = pmsg.pose.orientation
            rpy = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")
            frame["observation.state.cartesian"] = np.array(
                [p.x, p.y, p.z, rpy[0], rpy[1], rpy[2]], dtype=np.float32
            )

        # ---- Gripper state ----
        if use_gripper_state:
            g_idx = _nearest_index(ts_arrays["gripper_state"], target_ts)
            gmsg = deserialize_message(records["gripper_state"][g_idx][1], JointState)
            # Robotiq 2F-85 gripper joint is position[0] (gripper_config.index: 0).
            # Normalized to the action convention: 1.0 = open, 0.0 = closed.
            raw_grip = float(gmsg.position[0]) if len(gmsg.position) else GRIP_RAW_OPEN
            frame["observation.state.gripper"] = np.array(
                [_normalize_grip(raw_grip)], dtype=np.float32
            )

        if not use_joints and not use_pose and not use_gripper_state:
            frame["observation.state.dummy"] = np.zeros(1, dtype=np.float32)

        # ---- Concatenated state (insertion order: joints, cartesian, gripper) ----
        parts = []
        if use_joints:
            parts.append(frame["observation.state.joints"])
        if use_pose:
            parts.append(frame["observation.state.cartesian"])
        if use_gripper_state:
            parts.append(frame["observation.state.gripper"])
        if not parts:
            parts.append(frame["observation.state.dummy"])
        frame["observation.state"] = np.concatenate(parts).astype(np.float32)

        # ---- Action ----
        if use_action:
            tp_idx = _nearest_index(ts_arrays["target_pose"], target_ts)
            tp_msg = deserialize_message(records["target_pose"][tp_idx][1], PoseStamped)
            p = tp_msg.pose.position
            q = tp_msg.pose.orientation
            rpy = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")
            tp_arr = np.array(
                [p.x, p.y, p.z, rpy[0], rpy[1], rpy[2]], dtype=np.float32
            )
            # /target_gripper_state is sparse and event-driven. Use last-known
            # value at or before target_ts; before the first message we default
            # to 1.0 = open (matches 16_camera_only_record.py's default).
            g_ts_list = ts_arrays.get("gripper", [])
            if g_ts_list:
                i = bisect.bisect_right(g_ts_list, target_ts)
                if i == 0:
                    grip = GRIP_NORMALIZED_DEFAULT
                else:
                    g_msg = deserialize_message(records["gripper"][i - 1][1], Float32)
                    grip = _normalize_grip(float(g_msg.data))
            else:
                grip = GRIP_NORMALIZED_DEFAULT
            frame["action"] = np.concatenate(
                [tp_arr, np.array([grip], dtype=np.float32)]
            ).astype(np.float32)
        else:
            frame["action"] = np.zeros(1, dtype=np.float32)

        # ---- Commit frame ----
        if _ADD_FRAME_HAS_TASK:
            dataset.add_frame(frame, task=args.task)
        else:
            frame["task"] = args.task
            dataset.add_frame(frame)

    logger.info("Finalising episode (video encoding, parquet write)...")
    dataset.save_episode()

    if use_camera:
        dup_summary = ", ".join(
            f"{name}: {duplicate_camera[name]} "
            f"({100.0 * duplicate_camera[name] / max(1, n_ticks):.1f}%)"
            for name in camera_names
        )
        logger.info(
            "Done. Wrote %d frames to dataset %r. "
            "Per-camera duplicate samples at target fps: %s.",
            n_ticks, args.repo_id, dup_summary,
        )
    else:
        logger.info(
            "Done. Wrote %d frames to dataset %r. (No camera streams.)",
            n_ticks, args.repo_id,
        )
    logger.info(
        "Dataset cached at: ~/.cache/huggingface/lerobot/%s",
        args.repo_id,
    )


# ---------- CLI ----------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--bag", type=Path, required=True, help="Input bag directory.")
    parser.add_argument("--repo-id", type=str, required=True, help="LeRobot dataset name.")
    parser.add_argument("--task", type=str, default="observe scene",
                        help="Task label for the episode.")
    parser.add_argument("--fps", type=int, default=30, help="Target sampling rate.")
    parser.add_argument(
        "--resolution", type=int, nargs=2, default=[480, 640], metavar=("H", "W"),
        help="Target image resolution height width (default: 480 640).",
    )

    parser.add_argument(
        "--camera-topic", type=str, nargs="*", default=None,
        help="CompressedImage topic(s) to convert. When omitted, the converter "
             "auto-detects every sensor_msgs/msg/CompressedImage topic in the "
             "bag. Pass an explicit list to restrict the set, e.g. "
             "`--camera-topic /camera/color/image_raw/compressed "
             "/d405/camera/color/image_rect_raw/compressed`.",
    )
    parser.add_argument(
        "--camera-name", type=str, nargs="*", default=None,
        help="Camera names aligned with --camera-topic (same count, same "
             "order); these become the `observation.images.<name>` keys. "
             "When omitted, each name is derived from its topic's first "
             "path segment (e.g. `/d405/...` → 'd405').",
    )
    parser.add_argument("--no-camera", action="store_true", default=False)

    parser.add_argument("--joint-topic", type=str, default="/joint_states")
    parser.add_argument("--joint-names", type=str, nargs="+",
                        default=DEFAULT_JOINT_NAMES)
    parser.add_argument("--no-joints", action="store_true", default=False)

    parser.add_argument("--pose-topic", type=str, default="/current_pose")
    parser.add_argument("--no-pose", action="store_true", default=False)

    parser.add_argument("--gripper-state-topic", type=str, default="/gripper/joint_states")
    parser.add_argument("--no-gripper-state", action="store_true", default=False,
                        help="Skip observation.state.gripper even if the topic is present.")

    parser.add_argument("--with-action", action="store_true", default=False,
                        help="Include action from target pose + gripper topics.")
    parser.add_argument("--target-pose-topic", type=str, default="/target_pose")
    parser.add_argument("--gripper-target-topic", type=str, default="/target_gripper_state")

    parser.add_argument("--resume", action="store_true", default=False,
                        help="Append to an existing LeRobot dataset as a new episode.")

    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    convert(args)


if __name__ == "__main__":
    main()
