# Scripted calibration recording — `16_calibration_record.py`

Deterministic record/replay smoke test for the UR10e + Ridgeback + Robotiq 2F-85
recording stack. Drives the arm itself along a fixed cartesian sweep with
constant orientation and logs each commanded frame as a LeRobot v3 dataset.
No mocap, no SpaceMouse, no human input — the setpoints are computed in code,
so any mismatch between disk and ground truth is unambiguously a bug in the
recording path (not mocap jitter, not controller lag).

**Motivation and acceptance checks:** see
[`ridgeback_calibration_recording_plan.md`](./ridgeback_calibration_recording_plan.md).
The plan also covers related bugs in the replay / orientation-representation
path that this calibration was designed to isolate.

**Source:** `examples/16_calibration_record.py` — standalone, does not import
from or modify any of the existing `12_` / `13_` / `14_` / `15_` scripts.

---

## 1. What it does

On startup:

1. **Parses CLI.** Computes the frame budget from `--velocity`,
   `--displacement`, `--dwell-seconds`, and `--fps`.
2. **Creates the env** (`make_env("ur10e_ridgeback_env", control_type="cartesian")`).
3. **Takes ownership of `/target_pose`.** `ur10e_ridgeback_env.yaml` has
   `publish_target_pose: false` for the mocap flow, which makes `crisp_py.Robot`
   *subscribe* to `/target_pose` instead of publishing it. The script inlines
   `enable_env_target_pose_publishing` (the same patch `14_ridgeback_replay.py`
   uses): destroys the subscription, creates the publisher and the 20 Hz
   publish timer, flips the flag. This runs **before** `wait_until_ready()`.
4. **Waits for the robot**, creates the `KeyboardRecordingManager`, homes the
   arm via JTC, then samples `env.robot.end_effector_pose` as the anchor
   `P0 = (x0, y0, z0, R0)`.
5. **Builds the trajectory in memory** as a list of `(Pose, grip)` tuples
   (410 frames by default — see §3).
6. **Validates workspace limits** against `env.config.safety_box` (derived
   from `min_x/max_x/min_y/max_y/min_z/max_z` in the env YAML). Aborts with
   a clear error naming the violating waypoint and axis if any waypoint is
   outside the box — before the arm moves.
7. **Auto-starts recording.** Flips `recording_manager.state = "recording"`
   immediately after entering the context manager. No `r` keypress required.
8. **Runs the trajectory.** `record_episode` calls `data_fn` at `fps` (20 Hz).
   `data_fn` pops the next `(pose, grip)` pair, calls
   `env.robot.set_target(pose=...)` + `env.gripper.set_target(grip)`, grabs
   `env.get_obs()`, packs `action = [pose_euler(6), grip(1)]`, and returns
   `(obs, action)`.
9. **Auto-saves.** When `data_fn` emits the final frame, it sets
   `recording_manager.state = "to_be_saved"`. The recorder's while loop
   exits on the next tick and `_handle_post_episode` routes the episode
   directly into `SAVE_EPISODE` — no `s` keypress required.
10. **Exits cleanly.** `on_end` runs `reset_targets()` + `home(blocking=False)`
    + `gripper.open()`. The script then calls `env.home()` once more
    (blocking) and shuts down ROS.

The recording runs unattended from step 7 onwards — ~20 s at defaults.

---

## 2. How to use it

### 2.1 Prerequisites

- Robot up, controller_manager running, `cartesian_controller` (CRISP) active
  and `joint_state_broadcaster` + `pose_broadcaster` publishing.
- Orbbec camera streaming (e.g. `pixi run orbbec` from `clearpath_remote_ws`).
- **`track_mocap.py` must NOT be running.** This script publishes
  `/target_pose` itself; if the mocap tracker is also publishing, the two
  will fight for the controller and the trajectory will not match what is
  recorded.

The cleanest bring-up is `master_launch.sh` **without** `--track`:

```bash
./tools/master_launch.sh up --controller crisp
```

### 2.2 Run the recorder

```bash
cd Yunfei/crisp_gym
pixi run -e jazzy-lerobot python examples/16_calibration_record.py \
    --repo-id calib_axis_001
```

At defaults you should see:

- a log line reporting the trajectory budget
  (`8 × 40 + 9 × 10 = 410 frames ≈ 20.5 s at 20 Hz`),
- a log line printing `P0` position + Euler XYZ,
- `Auto-starting recording (no 'r' keypress required).`,
- the dataset save message at the end.

**Do not press `r` / `s` / `d` / `q` during the run.** The keyboard listener
is still live (so you can abort with `q` if something goes wrong), but the
state machine is driven by the script; a stray keypress will pause, stop, or
discard the episode.

### 2.3 CLI options

| Flag | Default | Description |
| --- | --- | --- |
| `--repo-id` | `calib_axis_001` | LeRobot dataset ID (directory name under `~/.cache/huggingface/lerobot/`). |
| `--task` | `"calibration sweep"` | Task label stored on every frame. |
| `--fps` | `20` | Recording frame rate. Must match the cartesian controller's effective publish rate. |
| `--velocity` | `0.05` | Cartesian speed during motion phases, m/s. Slower = less controller lag = cleaner diff at the cost of a longer episode. |
| `--displacement` | `0.10` | Half-range of each axis sweep, metres. |
| `--dwell-seconds` | `0.5` | Dwell duration at the anchor / extrema between phases. |
| `--include-gripper-toggle / --no-include-gripper-toggle` | on | Exercise the gripper action column with one close / open cycle (§3.2). Off = gripper held open for the entire trajectory. |
| `--env-config` | `ur10e_ridgeback_env` | Env YAML name passed to `make_env`. |
| `--resume` | off | Append to an existing dataset with the same `--repo-id` instead of erroring out. |
| `--push-to-hub / --no-push-to-hub` | off | Upload to Hugging Face Hub after saving (requires login). |
| `--log-level` | `INFO` | Python logging level. |

### 2.4 Adjusting the trajectory

Frame counts are derived at runtime:

```
n_motion = round(displacement / velocity * fps)   # 40 at defaults
n_dwell  = round(dwell_seconds * fps)              # 10 at defaults
total    = 8 * n_motion + 9 * n_dwell              # 410 at defaults
```

Two handy examples:

```bash
# Slower sweep (2 cm/s → 5 s / 100 frames per phase) — cleaner record vs. replay.
python examples/16_calibration_record.py \
    --repo-id calib_axis_slow --velocity 0.02

# Smaller box (±5 cm) if the workspace is tight.
python examples/16_calibration_record.py \
    --repo-id calib_axis_small --displacement 0.05
```

### 2.5 Aborting mid-run

Press `q` at any time to tell the recorder to exit. The currently recorded
episode is discarded (not saved).

---

## 3. The trajectory

### 3.1 Structure

Starting from `P0 = env.robot.end_effector_pose` (captured after
`env.home()`), the script sweeps each cartesian axis through `±displacement`
with linear interpolation at constant speed, returning to the anchor between
sweeps. The sequence is:

| # | Segment | Frames @ defaults | End position (relative to `P0`) |
| --- | --- | --- | --- |
| 0 | dwell at anchor | 10 | `(0, 0, 0)` |
| 1 | move `+z` | 40 | `(0, 0, +0.10)` |
| 2 | dwell at `+z` extremum | 10 | `(0, 0, +0.10)` |
| 3 | move back to anchor | 40 | `(0, 0, 0)` |
| 4 | dwell at anchor | 10 | `(0, 0, 0)` |
| 5 | move `−z` | 40 | `(0, 0, −0.10)` |
| 6 | dwell at `−z` extremum | 10 | `(0, 0, −0.10)` |
| 7 | move back to anchor | 40 | `(0, 0, 0)` |
| 8 | dwell at anchor | 10 | `(0, 0, 0)` |
| 9 | move `+y` (left) | 40 | `(0, +0.10, 0)` |
| 10 | dwell at `+y` extremum | 10 | `(0, +0.10, 0)` |
| 11 | move back to anchor | 40 | `(0, 0, 0)` |
| 12 | dwell at anchor | 10 | `(0, 0, 0)` |
| 13 | move `−y` (right) | 40 | `(0, −0.10, 0)` |
| 14 | dwell at `−y` extremum | 10 | `(0, −0.10, 0)` |
| 15 | move back to anchor | 40 | `(0, 0, 0)` |
| 16 | dwell at anchor | 10 | `(0, 0, 0)` |
| — | (out-of-episode) `env.home()` | — | joint-space safety return |

Alternating `dwell / motion / dwell / … / motion / dwell` gives 9 dwells
between segment boundaries, matching the §3.3 budget in the planning doc.

**Orientation is held constant at `R0`** for every frame (taken from the
anchor `Pose.orientation` and reused, not re-derived from Euler). Any nonzero
variation in the recorded orientation columns is a proven representation
bug — the script guarantees the commanded orientation is literally the same
`scipy.spatial.transform.Rotation` object.

**"Left" / "right" convention.** `+y` is called "left" and `−y` "right" in
the logs and trajectory phase table, matching the plan doc's default. This
is the `arm_0_base_link` frame; adjust if your operator stands on the other
side.

### 3.2 Gripper toggle

With `--include-gripper-toggle` (default), the gripper action column is
exercised:

- **Open** (`1.0` in crisp_py convention) for dwell 0 through segment 8
  (everything up to and including the anchor dwell before the y-axis sweep).
- **Close** (`0.0`) for segments 9 through 15 (the first y-axis outbound move
  through the final return to anchor).
- **Open** again for dwell 16 (final dwell at the anchor).

With `--no-include-gripper-toggle`, the gripper stays open (`1.0`) for the
entire trajectory. If the env has no gripper (`env.gripper is None`), the
toggle is skipped with a warning and the script logs everything as if
`--no-include-gripper-toggle` had been passed.

### 3.3 Workspace safety

Before the arm moves, `validate_against_workspace` checks every waypoint
against `env.config.safety_box`. The default env YAML has
`x, y ∈ [−0.9, 0.9]`, `z ∈ [0.05, 1.0]` (metres, `arm_0_base_link`), which
easily contains `P0 ± 0.10` for the standard home pose (`z0 ≈ 0.86`). If any
waypoint fails, the script **aborts with a clear error** naming the waypoint
index, position, and the violated bound — it does **not** clip silently.

Beyond the workspace check, the script does not override CRISP safety
(joint limits, velocity limits, safety_observer_controller). Those still
apply as usual and can still halt a run if e.g. the elbow approaches a
limit.

---

## 4. What it saves and where

### 4.1 Location

Datasets land under the standard LeRobot cache:

```
~/.cache/huggingface/lerobot/<repo-id>/
```

At defaults that is `~/.cache/huggingface/lerobot/calib_axis_001/`. The path
is governed by `lerobot.utils.constants.HF_LEROBOT_HOME`, not by anything
in this script.

**If the directory already exists**, `RecordingManager._create_dataset`
refuses to overwrite it: you will see a
`FileExistsError` telling you to either pass `--resume` (append) or
`rm -r` the directory manually.

### 4.2 Format

LeRobot v3 dataset (parquet + video + meta). Created via
`LeRobotDataset.create(..., use_videos=True, fps=<fps>, robot_type="ur10e",
features=<dict>)`. One episode per script run. The schema is whatever
`crisp_gym.util.lerobot_features.get_features(env)` returns for
`ur10e_ridgeback_env` with `control_type="cartesian"`, which for the
default YAML is:

| Feature | Shape | `dtype` | Names | Source |
| --- | --- | --- | --- | --- |
| `action` | `(7,)` | `float32` | `x, y, z, roll, pitch, yaw, gripper` | **script-generated** (see §4.3) |
| `observation.state.cartesian` | `(6,)` | `float32` | `x, y, z, roll, pitch, yaw` | `env.robot.end_effector_pose.to_array(env.config.orientation_representation)` — *measured* EE pose. |
| `observation.state.joints` | `(6,)` | `float32` | `joint_0 … joint_5` | `env.robot.joint_values`. |
| `observation.state.gripper` | `(1,)` | `float32` | `gripper` | `env.gripper.value` mapped to LeRobot convention `1 - value` (so `0 = open`, `1 = closed`). |
| `observation.state.gripper_target` | `(1,)` | `float32` | `gripper_target` | `env.gripper.target` (LeRobot-flipped, and falls back to current gripper value before the first grip command — the same prefix-quirk `manipulator_env.py:367-378` has for the other recorders). |
| `observation.state.target` | `(6,)` | `float32` | `target_x … target_yaw` | `env.robot.target_pose` — what the env-side `crisp_py.Robot` *thinks* its target pose is. Since this script owns the publisher, this mirrors what `set_target` just wrote. |
| `observation.state` | `(sum,)` | `float32` | concat of the above in iteration order | built by `concatenate_state_features`. |
| `observation.images.camera` | `(H, W, 3)` | `video` (av1) | `height, width, channels` | Orbbec color stream as configured in the YAML (`resolution: [640, 480]`). |
| *(per-episode metadata)* | — | — | — | `fps`, `robot_type=ur10e`, `task`, `episode_index`, `frame_index`, `timestamp`, `index`, and the crisp_gym metadata from `env.get_metadata()` (crisp_gym + crisp_py versions, control type, env config). |

A later `LeRobotDataset` schema change may shift this list. Run
`python -c "from crisp_gym.envs.manipulator_env import make_env; from
crisp_gym.util.lerobot_features import get_features; import rich;
rich.print(get_features(make_env('ur10e_ridgeback_env',
control_type='cartesian')))"` on the live env to print the current truth.

### 4.3 Action column — script-generated, ground-truth

This is the whole point of the calibration. `data_fn` writes:

```python
pose_vec = target_pose.to_array(
    representation=env.config.orientation_representation   # "euler" for ridgeback
).astype(np.float32)
action = np.concatenate(
    [pose_vec, np.array([target_grip], dtype=np.float32)]
)
```

So for the default ridgeback YAML (`orientation_representation: "euler"`)
the action is:

```
action[0:3] = target_pose.position               # metres, arm_0_base_link frame
action[3:6] = target_pose.orientation.as_euler("xyz", degrees=False)   # radians, constant
action[6]   = target_grip                        # crisp_py convention: 1=open, 0=closed
```

The action vector at every frame is exactly what the script computed — it
is the same `Pose` object that was handed to `env.robot.set_target(...)`,
passed through `Pose.to_array(representation=...)`. There is no round trip
through any subscription, no measurement, no estimator. This is the
ground truth that §4 of the plan doc diffs against.

> **Gripper convention mismatch.** `action[6]` uses crisp_py convention
> (`1.0 = open`, `0.0 = closed`) but `observation.state.gripper` uses
> LeRobot convention (`0 = open`, `1 = closed`, because
> `gripper_config.min_value=0.8, max_value=0.0`). This matches the existing
> `13_` / `15_` recorder layout, so `14_ridgeback_replay.py
> --gripper-source action` replays calibration datasets correctly without
> needing a sign flip. Notebook-side comparisons must flip one or the other
> before subtracting.

### 4.4 Typical disk layout after a single run

Approximate layout of `~/.cache/huggingface/lerobot/calib_axis_001/` after
one run (exact filenames are LeRobot v3 internals and may change):

```
calib_axis_001/
├── meta/
│   ├── info.json                    # total_episodes, total_frames, fps, features schema, robot_type…
│   ├── episodes/                    # per-episode parquet files with frame data
│   │   └── chunk-000/file-000.parquet
│   ├── episodes_stats/              # per-episode feature statistics
│   └── tasks.jsonl                  # one row per unique task string (here: "calibration sweep")
└── videos/
    └── chunk-000/observation.images.camera/
        └── episode_000000.mp4       # av1-encoded camera stream, aligned to frames
```

At defaults (410 frames × 20 Hz) the episode is ~20 s long, and the mp4 is
usually a few megabytes.

---

## 5. Verification (post-run)

See §4 of `ridgeback_calibration_recording_plan.md` for the full list. The
quick checks you can run on a saved `calib_axis_001` dataset without ever
needing the mocap stream:

- **Orientation invariance.** The script holds orientation constant, so
  `max(std(action[3:6], axis=0))` must be **exactly zero** and
  `max(std(observation.state.target[3:6], axis=0))` should be
  (near-)zero. If either is nonzero, there is a representation / buffer
  bug in the `set_target → to_array` path.
- **Action vs. recorded target.** `max | action[:6] −
  observation.state.target[:6] |` should be sub-mm / sub-mrad.
- **Record-side position fidelity.** For each motion phase, `action[:3]`
  is a known straight line; compare with `observation.state.cartesian[:3]`
  and expect only a small controller-lag offset.
- **Replay round-trip.** Feed the dataset to
  `examples/14_ridgeback_replay.py --repo-id calib_axis_001 --target-source
  action` and then again with `--target-source target`. Both should trace
  the same up / down / left / right pattern with **no wrist flips** at any
  phase boundary.
- **Symmetry.** Phases 1+3 (z) and 9+13 (y) should mirror around the anchor.

A dedicated verification notebook can live under
`Yunfei/crisp_gym/notebooks/` alongside `inspect_dataset.ipynb` — this
document does not prescribe its layout.

---

## 6. Known caveats

- **The script does not verify which controller is active.** It assumes
  `cartesian_controller` is loaded and running. If a different controller
  is active when you start it, `env.robot.set_target(...)` silently
  updates an internal buffer that the wrong controller ignores and you
  get a static episode. Always bring the robot up via `master_launch.sh up
  --controller crisp` first.
- **No dry-run mode.** Every run moves the real arm. If you want to
  inspect the trajectory offline, import `build_trajectory(...)` from the
  script in a notebook and plot the result.
- **The dataset directory must not already exist** unless `--resume` is
  passed. `RecordingManager._create_dataset` raises `FileExistsError`
  otherwise — this is a standard LeRobot guard, not specific to this
  script.
- **Keyboard listener is still live during auto-run.** The
  `KeyboardRecordingManager` listener thread is started on context entry,
  so pressing `r` / `s` / `d` / `q` during the sweep will still perturb
  the state machine. `q` is the safe way to abort.
- **Action orientation-representation lock-in.** The action column is
  written in whatever `orientation_representation` the env YAML declares
  (Euler-XYZ by default). Changing that YAML field between record and
  replay will break the round trip; this is the representation bug that
  motivated the calibration in the first place. Re-record whenever the
  representation changes.
