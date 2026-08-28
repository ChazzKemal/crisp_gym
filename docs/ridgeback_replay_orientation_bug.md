# Ridgeback replay does not match the recorded video — Euler orientation bug

**Status:** diagnosed, not yet fixed.
**Affected dataset(s):** `ridgeback_223335` (and any other Ridgeback dataset recorded with the current `ur10e_ridgeback_env.yaml`, which sets `orientation_representation: "euler"`).
**Affected scripts:** `examples/13_ridgeback_mocap_record.py`, `examples/15_ridgeback_mocap_record.py` (recording), `examples/14_ridgeback_replay.py` (replay).
**Related doc:** `docs/ridgeback_target_pose_ownership.md` (covers the `/target_pose` ownership problem; this doc covers an independent problem in the *same* recording flow).

---

## 1. Symptom

Data was collected with `13_ridgeback_mocap_record.py` and saved as the LeRobot v3 dataset `ridgeback_223335`. When the same episodes were replayed on the real robot via `14_ridgeback_replay.py`, the arm reached the correct first joint configuration (because replay uses `joint_trajectory_controller` for that initial move) but **the subsequent cartesian replay phase did not reproduce the motion seen in the recorded `observation.images.camera` video**. The wrist made unexpected ~360° flips, the end-effector drifted off the demonstrated trajectory, and the final pose did not match the last frame of the recording.

The gripper open/close events replayed correctly. The mismatch is in the **arm pose**, not the gripper.

---

## 2. Dataset facts (`ridgeback_223335`)

```
~/.cache/huggingface/lerobot/ridgeback_223335/
├── meta/
│   ├── info.json            codebase_version=v3.0, robot_type=ur10e, fps=20
│   ├── tasks.parquet
│   ├── stats.json
│   └── episodes/chunk-000/file-000.parquet
├── data/chunk-000/file-000.parquet     1987 frames total
└── videos/observation.images.camera/chunk-000/file-000.mp4
```

| Episode | Frames | Duration @20 Hz |
| ---: | ---: | ---: |
| 0 | 287  | 14.3 s |
| 1 | 1700 | 85.0 s |

### Feature schema (from `meta/info.json`)

| Feature | dtype | shape | Notes |
| --- | --- | --- | --- |
| `observation.images.camera`        | video    | 640×480×3 av1 | 20 Hz |
| `observation.state.cartesian`      | float32  | (6,)  | `[x, y, z, roll, pitch, yaw]` in **EULER xyz** |
| `observation.state.target`         | float32  | (6,)  | same layout, sourced from `robot.target_pose` |
| `observation.state.joints`         | float32  | (6,)  | order matches `ur10e_ridgeback_env.yaml:39-45` (`elbow, shoulder_lift, shoulder_pan, wrist_1, wrist_2, wrist_3`) — **NOT** the URDF order |
| `observation.state.gripper`        | float32  | (1,)  | LeRobot convention: `1 - gripper.value`, so `0=open, 1=closed` for the Ridgeback yaml (`min_value=0.8, max_value=0.0`) |
| `observation.state.gripper_target` | float32  | (1,)  | LeRobot convention; **prefix is contaminated**: before the first `/target_gripper_state` arrives the recorder writes `1 - gripper.value` instead of an actual command (`manipulator_env.py:367-378`) |
| `action`                           | float32  | (7,)  | `[0:6]` = mocap target pose in **EULER xyz**, `[6]` = gripper command in **crisp_py convention** (`1=open, 0=closed`, no flip) |

The recorded data shape and contents are exactly what the `13_ridgeback_mocap_record.py` `data_fn` produces — see Section 4 for how that function builds each frame.

---

## 3. The smoking gun

Per-axis `max | action[:6] − observation.state.target |` over each episode:

```
ep 0   x:0.011  y:0.017  z:0.027   roll:6.245   pitch:0.034  yaw:0.031     ← roll  ≈ 2π
ep 1   x:0.016  y:0.021  z:0.007   roll:0.033   pitch:0.227  yaw:3.739     ← yaw   ≈ 2π−2.5
```

Position channels match to ~1–2 cm — that's expected sub-step jitter between the mocap subscription that `data_fn` reads (`mocap_capture.get()`) and the env's own `/target_pose` subscriber that `_get_obs` reads (`robot.target_pose`). They're both samples of the same `/target_pose` stream taken at slightly different instants in the same loop iteration.

Orientation channels carry **2π wraparound discontinuities**. And the recorded ranges:

```
ep 0   cart roll:[-3.14, +3.14]    target roll:[-3.14, +3.14]
ep 1   cart yaw :[-2.23, -1.45]    target yaw :[-2.35, +3.08]
```

In ep 1 the actual end-effector yaw never crossed +π in reality (the cartesian column shows `[-2.23, -1.45]`), but the *Euler representation of the target* jumps between `−π+ε` and `+π−ε` for the same physical orientation.

This is not a recording-system bug, it is a representation bug: **`Rotation.as_euler("xyz")` is non-unique near gimbal lock and discontinuous at the ±π wrap.** Two consecutive frames that physically describe the same wrist orientation can be encoded as two Euler triples that differ by ~2π.

---

## 4. Where the bug lives in the code

### 4.1 Recorder side — `13_ridgeback_mocap_record.py` and `15_ridgeback_mocap_record.py`

The `data_fn` (`13:266-297`, identical in script 15) sources the cartesian half of `action` from the latest mocap target pose:

```python
mocap_pose = mocap_capture.get()
target_pose = mocap_pose.to_array(
    representation=env.config.orientation_representation
).astype(np.float32)
```

`env.config.orientation_representation` is the string `"euler"` from `ur10e_ridgeback_env.yaml:34`. `Pose.to_array(EULER)` calls `to_pos_euler_array` which is:

```python
# crisp_py/utils/geometry.py:97
def to_pos_euler_array(self) -> np.ndarray:
    euler = self.orientation.as_euler("xyz", degrees=False)
    return np.concatenate([self.position, euler], axis=0)
```

Raw `as_euler("xyz")`. No unwrapping, no continuity tracking.

The same conversion path is used by `manipulator_env.ManipulatorCartesianEnv._get_obs` for both `observation.state.cartesian` (`manipulator_env.py:342-360`) and `observation.state.target` (`manipulator_env.py:702-710`):

```python
target_pose_array = self.robot.target_pose.to_array(
    representation=self.config.orientation_representation
)
```

So **all three orientation columns** in the dataset (`cartesian`, `target`, `action[3:6]`) go through `as_euler("xyz")` independently per frame, with no continuity guarantee.

### 4.2 The continuity helper that exists but does not run

`manipulator_env.py:165-188` defines `_flip_rotation_vector_if_needed`, which compares the current orientation 3-vec with the previous one and flips its sign if they have negative dot product. This would handle the angle-axis sign ambiguity correctly. But the gate at `manipulator_env.py:157-163`:

```python
def _should_check_proper_orientation_representation(self) -> bool:
    return self.config.orientation_representation == OrientationRepresentation.ANGLE_AXIS
```

**only returns True for `ANGLE_AXIS`**. With `"euler"`, the flip helper is never called, and even if it were, it would be the wrong fix — Euler XYZ is not just sign-ambiguous, it is multi-valued and gimbal-locked.

### 4.3 Replay side — `14_ridgeback_replay.py`

The replay loop (`14:512-528`) reconstructs each commanded pose by inverting the conversion:

```python
pos, rpy = ee_target_at(row, args.target_source)
pose = Pose(
    position=np.asarray(pos, dtype=np.float64),
    orientation=Rotation.from_euler("xyz", rpy),
)
env.robot.set_target(pose=pose)
```

`Rotation.from_euler("xyz", rpy)` will faithfully round-trip *any single* `as_euler("xyz")` output back to the same rotation matrix. That is not the problem. The problem is that **consecutive recorded frames** can encode two near-identical physical orientations as Euler triples that differ by ~2π. When `from_euler` is called per frame, the resulting rotation sequence has discontinuities that physically don't exist, and these are sent straight to the cartesian impedance controller (`crisp_controllers`).

CRISP's cartesian impedance loop interpolates internally between successive targets. A target that flips ~2π between two 50 ms frames is interpreted as a real command — the wrist tries to follow it, makes a large unwanted motion, and the rest of the trajectory desynchronizes from the mocap demonstration.

**`--target-source action` does not help.** The action column was written through the same `Pose.to_array(EULER)` path, and the data confirms it: `max |action[:6] − target| = 6.245` on roll for ep 0. Whichever column you replay from, you replay the same Euler discontinuities.

### 4.4 Why script 15 fixes a *different* problem and is not relevant here

Scripts 13 and 15 are byte-for-byte identical except for `silence_env_target_publishers`:

- **13** unconditionally rewrites `env.robot._target_pose_publisher.publish` (`13:130`).
- **15** guards with `if env.robot._target_pose_publisher is not None` (`15:153`).

`ur10e_ridgeback_env.yaml:67` sets `publish_target_pose: false`, so `crisp_py.Robot.__init__` *never creates* `_target_pose_publisher`. Script 13 raises `AttributeError: 'NoneType' object has no attribute 'publish'` on startup with the current yaml. Script 15 is the working fix.

But once 13 (or 15) has actually started recording, the data shape and values it writes are identical. **The orientation discontinuity bug is in the env / `data_fn` path, which is the same in both scripts.** Switching from 13 to 15 does not fix `ridgeback_223335` and does not fix future replays.

---

## 5. Secondary issues observed (not the root cause, but worth noting)

These do **not** explain the replay mismatch on `ridgeback_223335`, but they are real and deserve cleanup at some point:

1. **`observation.state.gripper_target` prefix is contaminated.** Before the first `/target_gripper_state` arrives the recorder writes `1 − gripper.value` (the *measured* state, not a command) — `manipulator_env.py:367-378`. Script 14 already documents this and defaults to `--gripper-source action`, which reads `action[6]` (the literal command), so replay is unaffected by default. The inspect notebook's "action vs observation" sanity check in `notebooks/inspect_dataset.ipynb` cell 23 should ignore the prefix.

2. **`enable_env_target_pose_publishing` in script 14 leaks a timer.** It calls `robot.node.create_timer(...)` (`14:309`) on every invocation. For a single-shot script this is fine. If anyone ever adapts the helper for a long-running process or repeated replays in the same Python interpreter, you'd accumulate publish callbacks.

3. **Replay timer cadence vs replay loop cadence.** The replay loop sends `set_target` at the recording fps (20 Hz) and the new timer republishes at `robot.config.publish_frequency` (also 20 Hz). They are not phase-locked, so the controller actually receives 20–40 Hz of `/target_pose` messages depending on the relative phase. Slight jitter, not catastrophic.

4. **Joint name ordering in `ur10e_ridgeback_env.yaml:39-45`** is `[elbow, shoulder_lift, shoulder_pan, wrist_1, wrist_2, wrist_3]` — *not* the URDF / `/joint_states` canonical order. `crisp_py.Robot` reorders incoming joint states to match this list, so record and replay are self-consistent as long as both go through the same yaml. Anyone loading `ridgeback_223335` against a different env config will silently get the wrong arm pose.

5. **`make_env(... namespace="")`** in both 13 and 14. The yaml docstring (`ur10e_ridgeback_env.yaml:14-20, 56-58`) flags that on the real robot, topic names may need to be namespaced under `/r100_0207/manipulators/...`. If the real robot is ever brought up with namespaced topics, both record and replay break together — symmetric, but easy to forget.

---

## 6. Plan — fix options

The five options below are not mutually exclusive. The recommendation is in §7.

### Option A — switch the env to a continuous orientation representation

**Change:** set `orientation_representation: "angle_axis"` in `ur10e_ridgeback_env.yaml:34`.

**Why it helps:** the env's existing `_flip_rotation_vector_if_needed` (`manipulator_env.py:165-188`) already runs in this mode and keeps the rotation 3-vector on a single sign sheet across consecutive frames. `Pose.to_pos_angle_axis_array` (`crisp_py/utils/geometry.py`) and `Rotation.from_rotvec` round-trip cleanly without gimbal lock. Future recordings replay correctly with no further changes.

**Drawbacks:**
- Any policy already trained on `"euler"`-format data will not load — the `action` and `observation.state.cartesian/target` semantics change.
- `ridgeback_223335` itself remains broken; it must be re-recorded.
- The `_flip_rotation_vector_if_needed` helper does **not** unwrap angle-axis vectors that legitimately cross a 2π boundary in magnitude (it only handles the sign-flip ambiguity at ±π). For the Ridgeback pick-and-place task this is fine because the wrist never makes a full revolution, but for tasks that do, the same class of bug returns in a different form.

**Effort:** trivial code change, plus re-collection of every existing dataset.

### Option B — add Euler-continuity unwrap inside `_get_obs` for the EULER branch

**Change:**
1. Extend `_should_check_proper_orientation_representation` (`manipulator_env.py:157-163`) to also return True for `EULER`.
2. Add a new method `_unwrap_euler_if_needed(previous_euler, current_euler)` that, for each of the three Euler components, applies an `np.unwrap`-style branch correction so consecutive frames never differ by more than π on any axis.
3. Call it from both `_get_obs` cartesian path (`manipulator_env.py:346-350`) and the cartesian env target path (`manipulator_env.py:705-709`), maintaining a `_previous_target_euler` analogous to `_previous_target_rotation_vector`.
4. Apply the same unwrap to the `mocap_capture` cartesian half inside `data_fn` of the recorder scripts, so `action[3:6]` stays on the same branch as the env's `target` column. The simplest way is to delete the script-side `mocap_capture` plumbing entirely and source `action[3:6]` from `env.robot.target_pose` (the same place `observation.state.target` already comes from). They differ only by a few-millisecond phase offset and the env path will go through the unwrap helper for free.

**Why it helps:** keeps the on-disk format identical, fixes recording continuity, makes replay (Section 4.3) round-trip correctly *as long as* the recorded Euler sequence stays continuous frame-to-frame. Backward compatible with policies trained on `"euler"` data because the *format* (3 floats per frame, xyz Euler) does not change — only the per-frame branch selection changes, and only on frames where the previous version was discontinuous (which were unusable for training anyway).

**Drawbacks:**
- Does **not** fix `ridgeback_223335` retroactively. The unwrap has to run live during recording; you cannot recover the missing branch information from a frame in isolation.
- Gimbal lock (pitch ≈ ±π/2) still produces ambiguous Euler triples even with unwrap. The Ridgeback pick task is mostly tool-down with pitch near 0, so this is unlikely to bite, but it is a latent footgun.
- Requires touching the env, not just the script.

**Effort:** ~30 lines in `manipulator_env.py`, plus optional simplification of the recorder scripts.

### Option C — save orientation as a quaternion in the recording layer

**Change:** make `crisp_gym/util/lerobot_features.get_features` declare the cartesian/target/action shapes as 7-vec / 8-vec quaternion regardless of the env's `orientation_representation`, and have `_get_obs` always emit quaternions on the wire while keeping the env's internal representation flexible.

**Why it helps:** quaternions are continuous, unique up to a global sign (which is easy to fix with a `dot < 0` flip exactly like the existing helper), and round-trip losslessly. Best long-term solution; eliminates the entire class of bug.

**Drawbacks:**
- Most invasive change. Breaks every existing dataset and every consumer (training scripts, the inspect notebook, replay script 14, any policy infrastructure that consumes `action`).
- The action space dimension changes from 7 to 8, which means action-norm regularizers, network output heads, and dataset stats all need to be recomputed.
- Inconsistent with how other crisp_gym envs are configured today.

**Effort:** large, multi-file, plus full retraining and dataset re-collection.

### Option D — fix only the replay script with a quaternion shortest-path guard

**Change:** in `14_ridgeback_replay.py:512-528`, keep a running `previous_quat`. On each frame:

```python
q = Rotation.from_euler("xyz", rpy).as_quat()
if previous_quat is not None and np.dot(previous_quat, q) < 0:
    q = -q
pose = Pose(position=pos, orientation=Rotation.from_quat(q))
previous_quat = q
```

**Why it helps:** removes the per-frame ~2π flip on the replay side, so the cartesian impedance controller sees a continuous target stream and follows it smoothly. Lets you re-run replay against `ridgeback_223335` *without* re-recording, as a quick sanity check.

**Drawbacks:**
- Only fixes the **sign-flip** discontinuity, not gimbal-lock ambiguity. A frame where `as_euler` legitimately picked the "other" branch (e.g. roll = +π−ε followed by roll = −π+ε for the same rotation) round-trips back to the same rotation, so the dot-product flip is a no-op. But two genuinely different gimbal-locked Euler triples for the *same* rotation will still produce the same quaternion and behave correctly.
- Does not address the recording-side bug, so future datasets keep accumulating discontinuities.
- The recorded data is also degraded for **training** purposes (a learner sees a yaw jumping from −π to +π as a regression target with infinite gradient), and option D doesn't help training at all — it only helps replay.

**Effort:** ~10 lines in script 14, fully reversible.

### Option E — re-record `ridgeback_223335`

**Change:** delete `~/.cache/huggingface/lerobot/ridgeback_223335` (after backing up), re-run the recorder against the same task with whichever recording-side fix (A or B) has been merged.

**Why it helps:** the only way to get a clean dataset; required if `ridgeback_223335` is going to be used for training.

**Drawbacks:** human time on the robot.

**Effort:** ~30 minutes to re-collect 2 episodes once the recording fix is in.

---

## 7. Recommendation

**Apply B + D + E.**

- **B fixes the recording path** with the smallest possible change, keeps the on-disk format identical, and is fully backward compatible with existing `"euler"`-format consumers and any policy already trained on that format.
- **D is a defensive guard on the replay side** that costs nothing, makes script 14 robust against any *other* Euler discontinuities that might sneak in (e.g. from a different mocap source, or from a third-party dataset), and lets us verify the fix immediately by running the patched script 14 against the unfixed `ridgeback_223335` to confirm the symptom is rotation-discontinuity-driven.
- **E is unavoidable** if `ridgeback_223335` will be used for training. It is fine to skip if the dataset was only ever a smoke test for the recording infrastructure.

A is reasonable if the team has decided that angle-axis is a better representation in general — but there is no functional reason to switch unless we also want the slightly cleaner numerics it offers near gimbal lock. C is the most "correct" answer in absolute terms but is too disruptive for what is currently a single-task pilot dataset.

Whichever option is taken, also clean up the secondary issues from §5 — in particular, document the joint name ordering in the env config and mark the gripper-target prefix contamination as a known caveat in the recorder docstring (it is already documented in script 14, just not in 13/15).

---

## 8. Open questions for the operator

1. Are there any policies already trained on `"euler"`-format Ridgeback data that we would break by switching to `"angle_axis"` (Option A)?
2. Is `ridgeback_223335` precious, or is it OK to re-record it once the fix is in (Option E)?
3. Should the immediate next step be:
   - **(a)** apply Option D to script 14 and re-run replay against `ridgeback_223335` to confirm the diagnosis end-to-end on hardware, *then* apply Option B to the env and re-record; or
   - **(b)** apply Option B straight away, re-record, and skip the diagnostic replay because the `max |action[:6] − target|` numbers in §3 are evidence enough?
4. Should `examples/13_ridgeback_mocap_record.py` be deleted in favour of `15_ridgeback_mocap_record.py` once the env fix is in, or kept as a historical reference? Right now it still crashes on startup against the current yaml (§4.4), so it is dead code on disk.
