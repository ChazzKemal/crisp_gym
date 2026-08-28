# Scripted geometric calibration recording — `18_calibration_record_geo.py`

Deterministic record/replay smoke test on a **closed geometric shape**
(circle or square) traced around the anchor pose. Sibling of
`16_calibration_record.py`, which does axis-aligned sweeps; together they
cover both orthogonal translations and continuous / polygonal paths.

Same invariants as script 16: no mocap, no SpaceMouse, no human input. The
setpoints are computed in code, so any mismatch between disk and ground
truth is unambiguously a bug in the recording path (not mocap jitter, not
controller lag).

**Motivation and acceptance checks:** see
[`ridgeback_calibration_recording_plan.md`](./ridgeback_calibration_recording_plan.md)
(general plan) and
[`ridgeback_calibration_recording.md`](./ridgeback_calibration_recording.md)
(script 16 — axis-aligned variant). The plan's §4 verification catalogue
applies here too with minor wording changes (see §5 below).

**Source:** `examples/18_calibration_record_geo.py` — standalone, does not
import from or modify any of the existing `12_` / `13_` / `14_` / `15_` /
`16_` scripts.

---

## 1. What it does

On startup:

1. **Parses CLI.** Shape (`circle` or `square`) and size are **required**.
   The approximate frame budget is logged from `--velocity`, `--size`,
   `--loops`, `--fps`, and `--dwell-seconds`; the exact budget is finalised
   once the anchor is known.
2. **Creates the env** (`make_env("ur10e_ridgeback_env", control_type="cartesian")`).
3. **Takes ownership of `/target_pose`.** Same inline patch as script 16 /
   script 14: destroys the env subscription, creates the publisher + 20 Hz
   publish timer, flips `publish_target_pose = True`. Must run **before**
   `wait_until_ready()`.
4. **Waits for the robot**, creates the `KeyboardRecordingManager`, homes
   the arm via JTC, then samples `env.robot.end_effector_pose` as the
   anchor `P0 = (x0, y0, z0, R0)`. The shape is centred at `P0`.
5. **Builds the trajectory in memory** as a list of `(Pose, grip)` tuples
   (see §3). Every frame uses the same orientation `R0` — exactly the
   same `scipy.spatial.transform.Rotation` object handed to every `Pose`.
6. **Validates workspace limits** against `env.config.safety_box` (derived
   from `min_x/max_x/min_y/max_y/min_z/max_z` in the env YAML). Aborts with
   a clear error naming the violating waypoint and axis if any waypoint is
   outside the box — before the arm moves.
7. **Auto-starts recording.** Flips `recording_manager.state = "recording"`
   immediately after entering the context manager. No `r` keypress
   required.
8. **Runs the trajectory.** `record_episode` calls `data_fn` at `fps`
   (20 Hz). `data_fn` pops the next `(pose, grip)` pair, calls
   `env.robot.set_target(pose=...)` + `env.gripper.set_target(grip)`, grabs
   `env.get_obs()`, packs `action = [pose_euler(6), grip(1)]`, and returns
   `(obs, action)`.
9. **Auto-saves.** When `data_fn` emits the final frame it sets
   `recording_manager.state = "to_be_saved"`. The recorder's while loop
   exits on the next tick and `_handle_post_episode` routes the episode
   directly into `SAVE_EPISODE` — no `s` keypress required.
10. **Exits cleanly.** `on_end` runs `reset_targets()` + `home(blocking=False)`
    + `gripper.open()`. The script then calls `env.home()` once more
    (blocking) and shuts down ROS.

The recording runs unattended from step 7 onwards.

---

## 2. How to use it

### 2.1 Prerequisites

- Robot up, controller_manager running, `cartesian_controller` (CRISP) active
  and `joint_state_broadcaster` + `pose_broadcaster` publishing.
- Orbbec camera streaming (e.g. `pixi run orbbec` from `clearpath_remote_ws`).
- **`track_mocap.py` must NOT be running.** This script publishes
  `/target_pose` itself; if the mocap tracker is also publishing, the two
  will fight for the controller and the trajectory will not match what is
  recorded. (Same dual-publisher issue as documented in
  `ridgeback_target_pose_ownership.md`.)

The cleanest bring-up is `master_launch.sh` **without** `--track`:

```bash
./tools/master_launch.sh up --controller crisp
```

### 2.2 Run the recorder

```bash
cd Yunfei/crisp_gym

# Horizontal 5 cm-radius circle centred at the home pose.
pixi run -e jazzy-lerobot python examples/18_calibration_record_geo.py \
    --repo-id calib_geo_circle_001 \
    --shape circle --size 0.05

# Horizontal 5 cm square centred at the home pose.
pixi run -e jazzy-lerobot python examples/18_calibration_record_geo.py \
    --repo-id calib_geo_square_001 \
    --shape square --size 0.05
```

At defaults you should see:

- a log line reporting the approximate shape frame count and the finalised
  budget after the anchor is captured,
- a log line printing `P0` position + Euler XYZ and the shape parameters,
- `Auto-starting recording (no 'r' keypress required).`,
- the dataset save message at the end.

**Do not press `r` / `s` / `d` / `q` during the run.** The keyboard listener
is still live (so you can abort with `q` if something goes wrong), but the
state machine is driven by the script; a stray keypress will pause, stop,
or discard the episode.

### 2.3 CLI options

| Flag | Default | Description |
| --- | --- | --- |
| `--repo-id` | `calib_geo_001` | LeRobot dataset ID under `~/.cache/huggingface/lerobot/`. |
| `--task` | `"calibration geometric sweep"` | Task label stored on every frame. |
| `--fps` | `20` | Recording frame rate. Must match the cartesian controller's effective publish rate. |
| `--shape` | *(required)* | `circle` or `square`. |
| `--size` | *(required)* | Circle **radius** OR square **side length**, metres. |
| `--plane` | `xy` | Plane of the shape in `arm_0_base_link` frame: `xy` / `xz` / `yz`. |
| `--velocity` | `0.05` | Cartesian speed along the shape, m/s. Slower = less controller lag = cleaner diff at the cost of a longer episode. |
| `--loops` | `1` | Number of full shape repetitions. |
| `--dwell-seconds` | `0.5` | Dwell duration at the anchor before and after the shape trace (seconds). |
| `--include-gripper-toggle / --no-include-gripper-toggle` | on | Exercise the gripper action column with one close / open cycle (§3.2). Off = gripper held open for the entire trajectory. |
| `--env-config` | `ur10e_ridgeback_env` | Env YAML name passed to `make_env`. |
| `--resume` | off | Append to an existing dataset with the same `--repo-id` instead of erroring out. |
| `--push-to-hub / --no-push-to-hub` | off | Upload to Hugging Face Hub after saving (requires login). |
| `--yes`, `-y` | off | Auto-confirm the `rm -rf` prompt for a stale dataset directory. |
| `--skip-workspace-check` | off | Bypass the pre-flight workspace-box validation. |
| `--log-level` | `INFO` | Python logging level. |

### 2.4 Adjusting the trajectory

Frame counts are derived at runtime from the shape geometry and
`--velocity`:

```
# circle
n_per_loop    = max(8, round((2·π) / (velocity / (size · fps))))
n_shape       = n_per_loop · loops

# square (side = --size)
n_per_edge    = max(1, round((size/2) / velocity · fps))
n_shape       = 8 · n_per_edge · loops

# common
n_approach    = max(1, round(distance_center_to_first_vertex / velocity · fps))
n_dwell       = max(1, round(dwell_seconds · fps))
total         = 2 · n_approach + 2 · n_dwell + n_shape
```

Handy worked examples at defaults (`velocity=0.05 m/s`, `fps=20`,
`dwell-seconds=0.5`):

- `--shape circle --size 0.05 --loops 1`
  → `n_per_loop ≈ round(2π·r/(v/fps)) = round(2π·0.05/0.0025) ≈ 126` frames,
  `n_approach = round(0.05/0.0025) = 20`, `n_dwell = 10`,
  total ≈ `2·20 + 2·10 + 126 ≈ 186` frames ≈ **9.3 s**.
- `--shape square --size 0.05 --loops 1`
  → `n_per_edge = round(0.025/0.0025) = 10`, `n_shape = 80`,
  `n_approach = 10`, `n_dwell = 10`, total ≈ `2·10 + 2·10 + 80 = 120`
  frames ≈ **6.0 s**.

Two handy invocations:

```bash
# Slower circle (2 cm/s) — cleaner record vs. replay.
python examples/18_calibration_record_geo.py \
    --repo-id calib_geo_circle_slow \
    --shape circle --size 0.05 --velocity 0.02

# Larger square across 3 loops in the XY plane.
python examples/18_calibration_record_geo.py \
    --repo-id calib_geo_square_triple \
    --shape square --size 0.10 --loops 3 --velocity 0.05

# Sagittal-plane circle (XZ).
python examples/18_calibration_record_geo.py \
    --repo-id calib_geo_circle_xz \
    --shape circle --size 0.05 --plane xz
```

### 2.5 Aborting mid-run

Press `q` at any time to tell the recorder to exit. The currently recorded
episode is discarded (not saved).

---

## 3. The trajectory

### 3.1 Structure

Starting from `P0 = env.robot.end_effector_pose` (captured after
`env.home()`), the script builds:

```
dwell(P0)               —  n_dwell frames at the anchor, gripper OPEN
approach(P0 → A0)       —  n_approach linear frames, gripper OPEN
shape trace             —  n_shape frames on the shape (circle or square)
return(An → P0)         —  n_approach linear frames, gripper OPEN or CLOSED
dwell(P0)               —  n_dwell frames at the anchor, gripper OPEN
```

`A0` is the shape's first vertex (`(+size, 0)` from `P0` in the chosen
plane for `circle`, `(+size/2, 0)` for `square`). `An` is the shape's
final frame, which for both shapes coincides with `A0` at closure.

#### Circle

The circle has radius `--size` in the `--plane` with `P0` at its centre.
Positions are generated at constant tangential speed `--velocity`: the
angle advances by `velocity / (size · fps)` radians per recorded frame, so
the cartesian speed along the circumference is exactly `velocity` m/s.
First frame of the trace is one step off `A0 = P0 + (size, 0)` so the
approach segment (which ends at `A0`) hands off cleanly. Traversed CCW in
the `(u, v)` plane axes documented in §3.4.

#### Square (8 vertices)

Vertices, CCW starting at `(+size/2, 0)`:

| idx | offset from P0  | name               |
| --- | --------------- | ------------------ |
| 0   | `(+s/2, 0)`     | right midpoint     |
| 1   | `(+s/2, +s/2)`  | top-right corner   |
| 2   | `(0, +s/2)`     | top midpoint       |
| 3   | `(-s/2, +s/2)`  | top-left corner    |
| 4   | `(-s/2, 0)`     | left midpoint      |
| 5   | `(-s/2, -s/2)`  | bottom-left corner |
| 6   | `(0, -s/2)`     | bottom midpoint    |
| 7   | `(+s/2, -s/2)`  | bottom-right corner |

Each edge between consecutive vertices has length `s/2` (midpoint →
corner or corner → midpoint) and is linearly interpolated at speed
`--velocity` with `n_per_edge` frames. Corners reach `s·√2/2` from the
centre, so check your workspace bounds accordingly. After the last edge
(`7 → 0`) of each loop, the next loop re-starts at vertex 0 — there is no
per-loop dwell on the shape itself, only the pre- and post-trace dwells
at the anchor.

### 3.2 Gripper toggle

With `--include-gripper-toggle` (default), the gripper action column is
exercised once:

- **Open** (`1.0` in crisp_py convention) for the pre-dwell, the approach
  segment, and the **first half** of the shape trace (by frame count).
- **Close** (`0.0`) for the second half of the shape trace and the return
  segment.
- **Open** again for the final post-dwell.

With `--no-include-gripper-toggle`, the gripper stays open (`1.0`) for the
entire trajectory. If the env has no gripper (`env.gripper is None`), the
toggle is skipped with a warning and the script logs everything as if
`--no-include-gripper-toggle` had been passed.

### 3.3 Workspace safety

Before the arm moves, `validate_against_workspace` checks every waypoint
against `env.config.safety_box`. The default env YAML has
`x, y ∈ [−0.9, 0.9]`, `z ∈ [0.05, 1.0]` (metres, `arm_0_base_link`), which
comfortably contains `P0 ± 0.10` in any axis for the standard home pose
(`z0 ≈ 0.86`). Corner distance for a square of side `s` is `s·√2/2` from
the centre, which still fits for `s ≤ 0.10` at the standard home.

If any waypoint fails, the script **aborts with a clear error** naming
the waypoint index, position, and violated bound — it does **not** clip
silently. Pass `--skip-workspace-check` only if you have verified the
trajectory is physically safe but the env YAML's limits are narrower than
your actual reachable box (rare).

Beyond the workspace check, the script does not override CRISP safety
(joint limits, velocity limits, `safety_observer_controller`). Those still
apply as usual and can still halt a run if e.g. the elbow approaches a
limit.

### 3.4 Plane and orientation conventions

`--plane` selects a pair of orthonormal axes `(u, v)` in `arm_0_base_link`
and the shape is traced in that plane around `P0`:

| `--plane` | u (first axis) | v (second axis) | Typical use |
|-----------|----------------|-----------------|-------------|
| `xy`      | `+x` (forward) | `+y` (left)     | horizontal shape (default) |
| `xz`      | `+x` (forward) | `+z` (up)       | sagittal / vertical-forward |
| `yz`      | `+y` (left)    | `+z` (up)       | frontal / vertical-sideways |

The shape's first vertex is always at `+u` from the centre, and the trace
goes CCW in the `(u, v)` basis. `"Left" / "right"` in the gripper-toggle
narrative above refers to `+y` / `−y` in the `xy` plane for the default
home stance.

**Orientation is held constant at `R0`** for every frame (taken from
`anchor.orientation` once and reused as the same `Rotation` object on
every `Pose`, deep-copied once via quaternion round-trip to guarantee it
does not share state with the anchor). Any nonzero variation in the
recorded orientation columns is therefore a proven representation bug —
the script guarantees the commanded orientation is literally the same
rotation value.

---

## 4. What it saves and where

### 4.1 Location

Datasets land under the standard LeRobot cache:

```
~/.cache/huggingface/lerobot/<repo-id>/
```

At defaults that is `~/.cache/huggingface/lerobot/calib_geo_001/`. The path
is governed by `lerobot.utils.constants.HF_LEROBOT_HOME`, not by anything
in this script.

If the directory already exists and `--resume` is **not** set, the script
offers to `rm -rf` it before starting (same prompt as script 16). Pass
`--yes` / `-y` to bypass the confirmation.

### 4.2 Format

LeRobot v3 dataset, identical schema to what `16_calibration_record.py`
produces — the feature dict is built by
`crisp_gym.util.lerobot_features.get_features(env)` for the same
`ur10e_ridgeback_env` config, so a verification notebook written against
script 16's output will load script 18's output without changes.

Key features (for `ur10e_ridgeback_env.yaml` with
`orientation_representation: "euler"`):

| Feature | Shape | dtype | Source |
| --- | --- | --- | --- |
| `action` | `(7,)` | `float32` | **script-generated** (see §4.3) |
| `observation.state.cartesian` | `(6,)` | `float32` | `env.robot.end_effector_pose.to_array(...)` — *measured* EE pose |
| `observation.state.joints` | `(6,)` | `float32` | `env.robot.joint_values` |
| `observation.state.gripper` | `(1,)` | `float32` | `env.gripper.value` in LeRobot convention (`0=open, 1=closed`) |
| `observation.state.gripper_target` | `(1,)` | `float32` | `env.gripper.target`, LeRobot-flipped, same prefix-quirk as scripts 13/15 |
| `observation.state.target` | `(6,)` | `float32` | `env.robot.target_pose` — env-side mirror of what `set_target` just wrote |
| `observation.state` | `(sum,)` | `float32` | concat of the above |
| `observation.images.camera` | `(H, W, 3)` | video (av1) | Orbbec color stream per the env YAML |

See §4.2 of
[`ridgeback_calibration_recording.md`](./ridgeback_calibration_recording.md)
for the exhaustive table, since the schema is shared.

### 4.3 Action column — script-generated, ground-truth

Identical to script 16. `data_fn` writes:

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
action[0:3] = target_pose.position              # metres, arm_0_base_link frame
action[3:6] = target_pose.orientation.as_euler("xyz", degrees=False)  # radians, constant
action[6]   = target_grip                       # crisp_py convention: 1=open, 0=closed
```

The action vector at every frame is exactly what the script computed — it
is the same `Pose` object that was handed to `env.robot.set_target(...)`,
passed through `Pose.to_array(representation=...)`. There is no round trip
through any subscription, no measurement, no estimator. This is the
ground truth that §5 diffs against.

> **Gripper convention mismatch.** `action[6]` uses crisp_py convention
> (`1.0 = open`, `0.0 = closed`) but `observation.state.gripper` uses
> LeRobot convention. Same as scripts 13/15/16 — replay with
> `14_ridgeback_replay.py --gripper-source action` Just Works without any
> sign flip.

### 4.4 Typical disk layout after a single run

Approximate layout of `~/.cache/huggingface/lerobot/calib_geo_circle_001/`
after one `--shape circle --size 0.05` run at defaults:

```
calib_geo_circle_001/
├── meta/
│   ├── info.json
│   ├── episodes/chunk-000/file-000.parquet
│   ├── episodes_stats/
│   └── tasks.jsonl
└── videos/
    └── chunk-000/observation.images.camera/
        └── episode_000000.mp4
```

At defaults the circle episode is ~9 s long and the square episode is
~6 s long; both mp4s are typically a few megabytes.

---

## 5. Verification (post-run)

See §4 of `ridgeback_calibration_recording_plan.md` for the full list.
The quick checks you can run on a saved `calib_geo_*` dataset without
ever needing the mocap stream:

- **Orientation invariance.** The script holds orientation constant, so
  `max(std(action[3:6], axis=0))` must be **exactly zero** and
  `max(std(observation.state.target[3:6], axis=0))` should be
  (near-)zero. If either is nonzero there is a representation / buffer
  bug in the `set_target → to_array` path. This is the cleanest bug
  signal in the dataset — any nonzero std at all is newsworthy because
  the script literally wrote the same rotation on every frame.

- **Action vs. recorded target.** `max | action[:6] −
  observation.state.target[:6] |` should be sub-mm / sub-mrad.

- **Record-side position fidelity.** For the circle, `action[:3]` should
  trace a planar closed curve with RMS radius equal to `--size` and
  plane normal perpendicular to the `(u, v)` basis. For the square,
  `action[:3]` should trace 8 straight edges with the expected corner
  coordinates. Overlay `observation.state.cartesian[:3]` and expect only
  a small controller-lag offset on the curved / polygonal path.

- **Replay round-trip.** Feed the dataset to
  `examples/14_ridgeback_replay.py --repo-id calib_geo_circle_001 --target-source action`
  and then again with `--target-source target`. Both should retrace the
  same closed shape with **no wrist flips** at any frame boundary. The
  closed-loop geometry makes any replay glitch (jitter, frame drop,
  representation wraparound) visually obvious on the recorded video.

- **Closure check (unique to geometric variant).** For each loop, the
  final shape frame and the first shape frame of the next loop differ by
  one frame-velocity step; the recorded path should be closed to within
  controller tracking error. If the visual trace **does not close**, the
  record or replay path is losing data in a way that axis-aligned sweeps
  cannot surface.

A dedicated verification notebook can live under
`Yunfei/crisp_gym/notebooks/` alongside `inspect_dataset.ipynb` — this
document does not prescribe its layout.

---

## 6. Known caveats

- **The script does not verify which controller is active.** It assumes
  `cartesian_controller` is loaded and running. If a different controller
  is active when you start it, `env.robot.set_target(...)` silently
  updates an internal buffer that the wrong controller ignores and you
  get a static episode. Always bring the robot up via
  `master_launch.sh up --controller crisp` first.
- **No dry-run mode.** Every run moves the real arm. If you want to
  inspect the trajectory offline, import `build_shape_trajectory(...)`
  from the script in a notebook and plot the result.
- **The dataset directory must not already exist** unless `--resume` is
  passed or the script's `rm -rf` prompt is accepted.
- **Keyboard listener is still live during auto-run.** The
  `KeyboardRecordingManager` listener thread is started on context entry,
  so pressing `r` / `s` / `d` / `q` during the sweep will still perturb
  the state machine. `q` is the safe way to abort.
- **Square corner reach.** A square of side `s` reaches `s·√2/2 ≈ 0.707·s`
  from the centre at its corners, not `s/2`. Size your `--size` with that
  in mind — a `--size 0.10` square extends `±7.07 cm` from `P0` in both
  `u` and `v` axes at its corners.
- **Circle angular resolution snapping.** `n_per_loop` is rounded to the
  nearest integer at `max(8, ...)`. For very small circles or very slow
  velocities the realised tangential speed may differ slightly from
  `--velocity` due to rounding. The action column always reflects the
  *realised* positions exactly, so this is not a bug — just a detail
  when comparing against a nominal analytic circle.
- **Action orientation-representation lock-in.** The action column is
  written in whatever `orientation_representation` the env YAML declares
  (Euler-XYZ by default). Changing that YAML field between record and
  replay will break the round trip; this is the representation bug that
  motivated the calibration in the first place. Re-record whenever the
  representation changes.
- **Shared topic-ownership patch with script 16.** Both scripts inline
  the same `enable_env_target_pose_publishing` helper. If that helper is
  fixed in the future (e.g. to add proper teardown), the fix needs to
  land in all copies — grep for the function name before editing.
