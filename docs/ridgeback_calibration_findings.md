# Ridgeback calibration findings — `calib_axis_001`

**Date:** 2026-04-15
**Dataset:** `~/.cache/huggingface/lerobot/calib_axis_001`
**Recorder:** `examples/16_calibration_record.py` (defaults: 5 cm/s, 10 cm displacement, 0.5 s dwell, 20 Hz, gripper toggle on)
**Analyzer:** `examples/17_calibration_report.py` (pure numpy, no ROS)
**Generated report:** `analysis/calib_axis_001_calibration_report.md`

**Related docs:**
- `docs/ridgeback_calibration_recording_plan.md` — motivation and acceptance criteria
- `docs/ridgeback_calibration_recording.md` — how the recorder works and what it saves
- `docs/ridgeback_replay_orientation_bug.md` — the Euler-XYZ issue this calibration was designed to isolate
- `docs/ridgeback_target_pose_ownership.md` — the `/target_pose` ownership model used by recorder / replay

---

## 1. TL;DR

The scripted calibration trajectory ran end-to-end on the real UR10e + Ridgeback and saved a 410-frame LeRobot v3 episode. Running `17_calibration_report.py` on the dataset:

**The four record-path invariants all hold.** The recorder is clean — there is no representation bug between `env.robot.set_target(...)`, `env.get_obs()`, and the parquet file:

| Invariant | Result |
| --- | --- |
| `action[:6]` equals the ground-truth commanded setpoint at every frame | ✅ max diff 3.8 × 10⁻⁸ m, 0 rad |
| `std(action[3:6])` (commanded orientation is constant) | ✅ exactly 0 |
| `std(observation.state.target[3:6])` (env's internal target orientation) | ✅ exactly 0 |
| `action[:6]` equals `observation.state.target[:6]` every frame | ✅ exactly 0 |

**Two real bugs surfaced, neither in the calibration script:**

1. **Euler-XYZ branch flip in the *measured* cartesian column** — the linear std of the roll angle is `2.60 rad` while the circular std is `0.04 rad` and there are 2 frame-to-frame `|Δ| > π` jumps. The physical rotation is essentially constant; the linear std is a representation artifact of `Rotation.as_euler('xyz')` flipping sign at ±π on a stationary input. Matches `ridgeback_replay_orientation_bug.md`.
2. **crisp_py gripper self-subscription echo race** — `action[6]` disagrees with `observation.state.gripper_target` on 25 scattered non-transition frames (transitions themselves are clean). Root cause: `Gripper.set_target` stores a raw hardware value in `self._target` while the same instance's `_callback_target_state` subscription overwrites it with a normalized value, and the `gripper.target` property then reinterprets it with the wrong convention. The actual gripper command was correct; only the recorded `gripper_target` column flickers.

Cartesian tracking lag is consistent with a moderate cartesian impedance stiffness at 5 cm/s (**RMS 29 mm, max 49 mm**) and is not a bug.

---

## 2. How the dataset was produced

```bash
# Bring up the robot (NO --track, the recorder owns /target_pose itself):
./tools/master_launch.sh up --controller crisp

# Hotfix: load the missing joint_state_broadcaster into /gripper/controller_manager
# (see §5 below for the permanent fix).

# Run the recorder:
cd Yunfei/crisp_gym
pixi run -e jazzy-lerobot python examples/16_calibration_record.py --repo-id calib_axis_001
```

The recorder homes the arm with `env.home()`, captures `env.robot.end_effector_pose` as the anchor `P0`, and drives a scripted ±10 cm sweep on `z` then `y` at 5 cm/s with 0.5 s dwells between phases, holding `R0` constant for every frame. Episode saved auto-magically when the 410-frame trajectory ends; no keyboard input needed.

Anchor pose captured at run time:

```
position (m)  : [+0.9311, +0.1697, +0.8658]
orientation   : [-3.1408, -0.0037, -1.5739]   # Euler XYZ, rad
```

The roll is essentially at `-π` — the exact condition where `as_euler('xyz')` is prone to branch flips.

### Prerequisites that tripped us first

- **Gripper bringup was incomplete.** `/gripper/controller_manager` only had `gripper_controller` loaded — the `joint_state_broadcaster` was never spawned. `/gripper/joint_states` had Publisher count = 0, so `env.wait_until_ready()` timed out on `gripper.wait_until_ready(...)`. Hotfix: three service calls (`load_controller`, `configure_controller`, `switch_controller`) against `/gripper/controller_manager` to bring the broadcaster up. **Permanent fix belongs in the gripper bringup launch file in `clearpath_robot_ws` — its spawner list is missing `joint_state_broadcaster`.**
- **Env yaml `max_x: 0.9` was stale.** The home pose lands the EE at x ≈ 0.931, outside the declared safety box. Bumped to `1.0` in `Yunfei/crisp_gym/crisp_gym/config/envs/ur10e_ridgeback_env.yaml` (still well inside the UR10e's ~1.3 m reach envelope). Also added `--skip-workspace-check` to `16_calibration_record.py` as an escape hatch.
- **Stale dataset directory.** `RecordingManager._create_dataset` refuses to overwrite an existing `repo_id`. Added an interactive prompt + `--yes` flag to `16_calibration_record.py` so the user can delete a previous stub without manual `rm -rf`.

---

## 3. Raw calibration report — all metrics

### 3.1 Dataset summary

| Field | Value |
| --- | --- |
| Frames | 410 (expected 410) |
| Nominal fps | 20 |
| Robot type | ur10e |
| Episodes | 1 |
| Video | av1, 640 × 480, 20 fps (Orbbec color) |

### 3.2 Section 1 — Action vs ground truth

| Metric | Value |
| --- | ---: |
| max ‖Δpos‖ | 3.814697 × 10⁻⁸ m |
| mean ‖Δpos‖ | 9.761351 × 10⁻⁹ m |
| max ‖Δrot‖ | 0 rad |
| mean ‖Δrot‖ | 0 rad |

**Verdict: ✅** action column is bit-identical to ground truth (float32 rounding only). This means the entire `script → Pose(position, rotation) → Pose.to_array('euler') → np.float32 → parquet` path is lossless, and the ground-truth trajectory reconstructed offline by `17_calibration_report.py` matches the recorder's commanded trajectory exactly.

### 3.3 Section 2 — Orientation invariance

The recorder commands a literally constant orientation `R0` for every frame, so the commanded columns must have std exactly zero. The measured column is analysed with both linear and circular std to distinguish physical motion from Euler-XYZ wraparound artifacts.

| Column | std kind | roll (rad) | pitch (rad) | yaw (rad) |
| --- | --- | ---: | ---: | ---: |
| `action[3:6]` | linear | 0 | 0 | 0 |
| `observation.state.target[3:6]` | linear | 0 | 0 | 0 |
| `observation.state.cartesian[3:6]` | linear | **2.597** | 0.040 | 0.021 |
| `observation.state.cartesian[3:6]` | **circular** | **0.039** | 0.040 | 0.021 |

**Euler-XYZ branch flips per axis** (`|Δ|` between consecutive frames > π): **roll = 2**, pitch = 0, yaw = 0.

**Verdict:** ✅ commanded columns are exactly constant. ⚠ measured column has an **artifact** — huge linear std but tiny circular std on roll, explained entirely by **2 sign flips at ±π**. See §4.

### 3.4 Section 3 — Action vs recorded env target

| Metric | Value |
| --- | ---: |
| max `|action[:3] − observation.state.target[:3]|` per-axis | 0 m |
| max `|action[3:6] − observation.state.target[3:6]|` per-axis | 0 rad |

**Verdict: ✅** `env.robot.target_pose` is a faithful mirror of what the script passed to `set_target(pose=...)`. No mutation anywhere between the `Pose` object created in `build_trajectory` and the serialized `observation.state.target` column.

### 3.5 Section 4 — Position fidelity per phase

Euclidean distance between commanded `action[:3]` and measured `observation.state.cartesian[:3]`, in millimetres. This is **not a bug detector** — it measures the cartesian impedance controller's tracking lag against a 5 cm/s step input.

| Phase | Frames | RMS err (mm) | Max err (mm) | GT Δ end (mm) |
| --- | ---: | ---: | ---: | ---: |
| `dwell_0_anchor` | 0..10 (10) | 0.05 | 0.06 | [0, 0, 0] |
| `move_1_plus_z` | 10..50 (40) | 28.05 | 40.26 | [0, 0, +100] |
| `dwell_1_plus_z` | 50..60 (10) | 26.16 | 31.99 | [0, 0, +100] |
| `move_2_return` | 60..100 (40) | 21.05 | 30.84 | [0, 0, 0] |
| `dwell_2_anchor` | 100..110 (10) | 20.65 | 24.76 | [0, 0, 0] |
| `move_3_minus_z` | 110..150 (40) | 32.35 | 39.46 | [0, 0, −100] |
| `dwell_3_minus_z` | 150..160 (10) | 26.72 | 37.39 | [0, 0, −100] |
| `move_4_return` | 160..200 (40) | 21.31 | 32.97 | [0, 0, 0] |
| `dwell_4_anchor` | 200..210 (10) | 22.22 | 26.49 | [0, 0, 0] |
| `move_5_plus_y` | 210..250 (40) | 36.06 | 45.31 | [0, +100, 0] |
| `dwell_5_plus_y` | 250..260 (10) | 37.97 | 40.43 | [0, +100, 0] |
| `move_6_return` | 260..300 (40) | 28.18 | 37.44 | [0, 0, 0] |
| `dwell_6_anchor` | 300..310 (10) | 28.52 | 34.08 | [0, 0, 0] |
| `move_7_minus_y` | 310..350 (40) | **40.12** | **48.94** | [0, −100, 0] |
| `dwell_7_minus_y` | 350..360 (10) | 34.77 | 43.05 | [0, −100, 0] |
| `move_8_return` | 360..400 (40) | 27.60 | 36.09 | [0, 0, 0] |
| `dwell_8_anchor` | 400..410 (10) | 28.68 | 36.09 | [0, 0, 0] |

**Overall: RMS cartesian error 29.38 mm, max 48.94 mm.** Final-dwell cartesian offset from anchor at frame 409 is **26.57 mm** — the arm had not yet settled back to `P0` when the 0.5 s final dwell ended.

**Observations:**
- `dwell_0_anchor` is sub-tenth-of-a-mm tracking (0.05 mm RMS) — the initial dwell is the only phase where the measured pose really matches the command, because the arm was settled at home before recording started.
- Y-axis tracking (36–40 mm RMS on the `±y` phases) is noticeably worse than z-axis (21–32 mm). The `move_7_minus_y` phase has the largest error of any segment; worth re-running at lower velocity to confirm whether this is asymmetric friction / gravity effects or a one-off.
- All dwell phases after the first show 20–38 mm RMS — the controller never fully settles before the next motion starts. This is the expected behaviour of a cartesian impedance controller with moderate stiffness against a 10-frame (0.5 s) dwell, not a bug.

**Not a bug finding — but two suggestions for a cleaner next baseline:**
- `--velocity 0.02` (2 cm/s) → controller lag would be ~2 / 5 ≈ 40% of the current values, so ~12 mm RMS.
- `--dwell-seconds 1.5` → gives the controller time to actually reach the anchor before the next phase.

### 3.6 Section 5 — Gripper consistency

| Comparison | max `|Δ|` | frames with `|Δ|` > 0.05 |
| --- | ---: | ---: |
| `action[6]` vs `(1 − gripper_target)` | **1.0000** | **25** |
| `action[6]` vs `(1 − gripper)` | 1.0000 | (measurement lag, not a bug) |

**Transition-frame check:** ✅ `action[6]` == `gripper_target` at all 2 transition boundaries (open → closed at frame 210, closed → open at frame 400) *and* their follow-up frames.

**Verdict: ⚠** Transitions are clean. The 25 bad frames are scattered throughout both the "open" region (frames 0–209) and the "closed" region (frames 210–399), not clustered at transitions — classic signature of a **race** between `set_target` and its own echo callback. See §5 for the root cause.

### 3.7 Section 6 — Timing (synthetic ⚠)

| Metric | Value |
| --- | ---: |
| Span | 20.450 s |
| Mean dt | 50.00 ms |
| Median dt | 50.00 ms |
| Max dt | 50.00 ms |
| Min dt | 50.00 ms |
| Stddev dt | 0.001 ms |
| Effective fps (from mean dt) | 20.00 |

**The `timestamp` column is synthetic** — LeRobot v3 stores `frame_index / fps`, not wall-clock. The recording emitted dozens of `Frame processing took too long` warnings during the run (ranging from ~3 ms to ~545 ms over the 50 ms budget), so the **real effective fps was far below 20 Hz at several moments**, but the saved timestamps look perfectly uniform. True per-frame jitter is only visible in the live log, not the parquet.

**Wall-clock evidence from the live log:** first `data_fn` call was logged at `21:42:16`, last at `21:43:02` — that's **~46 s of wall-clock for 410 "20 Hz" frames**, an effective rate of ~9 Hz. That means the real cartesian velocity was ~2.2 cm/s, not 5 cm/s, and the tracking errors above are against an even-slower-than-intended motion.

**Suggestion for a future iteration of the recorder:** have `data_fn` log `time.monotonic()` into a side-channel list, and save it next to the parquet (e.g. `analysis/<repo_id>_walltime.npy`). `17_calibration_report.py` can then compute real per-frame jitter and an actual effective fps. This does not modify existing scripts; it's an additive optional feature on `16_calibration_record.py`.

### 3.8 Summary verdicts (straight from the report)

- ✅ `action = ground truth`
- ✅ action orientation is constant
- ✅ recorded target orientation is constant
- ✅ `action == env.robot.target_pose`
- ⚠ measured cartesian orientation is stable angularly (circular std 0.04 rad) but linear std is 2.60 rad and there are 2 Euler-XYZ branch flips — **representation artifact**, see §4
- ⚠ gripper: transitions clean but 25 non-transition frames flicker — **crisp_py gripper echo race**, see §5
- ℹ RMS cartesian tracking error 29.38 mm (not a bug, controller lag)
- ℹ final dwell cartesian offset from anchor: 26.57 mm

---

## 4. Bug #1 — Euler-XYZ branch flip in the measured cartesian column

### 4.1 Symptom

`std(observation.state.cartesian[3])` (roll) = **2.597 rad** over the recording. The physical robot was not rotating — it was holding a constant `R0` and executing pure translation in `y` and `z` directly from `env.home()`. A 2.6 rad standard deviation on roll would mean an average swing of ~150°, which clearly did not happen.

The circular std of the same column is **0.039 rad** (2.2°). The raw min/max are **−3.1410 / +3.1389**, straddling ±π. There are exactly **2 frame-to-frame jumps with `|Δroll| > π`**. These are the signature of sign flips at the branch cut, not physical motion.

### 4.2 Root cause

`crisp_py/utils/geometry.py:97` defines `Pose.to_pos_euler_array` as:

```python
def to_pos_euler_array(self) -> np.ndarray:
    euler = self.orientation.as_euler("xyz", degrees=False)
    return np.concatenate([self.position, euler], axis=0)
```

`scipy.spatial.transform.Rotation.as_euler('xyz')` returns angles in the ranges `roll ∈ (−π, +π]`, `pitch ∈ [−π/2, +π/2]`, `yaw ∈ (−π, +π]`. Near `roll = ±π` the sign is ambiguous — a rotation that is within numerical noise of roll = `+π` and a rotation within noise of `−π` are physically the same rotation, but SciPy can return either convention depending on the underlying quaternion's `w` sign. On consecutive ROS messages, rounding can flip the sign, producing a step of `2π` in the saved parquet column while the underlying physical rotation didn't move.

`manipulator_env.py:get_obs → obs['observation.state.cartesian'] = env.robot.end_effector_pose.to_array(env.config.orientation_representation)`. With `orientation_representation: "euler"` in `ur10e_ridgeback_env.yaml`, the env calls `to_pos_euler_array()` once per frame, which means this sign-flip happens **inside the recording pipeline**. The commanded columns don't suffer because `16_calibration_record.py` builds `Pose(position, Rotation)` once from `anchor_pose.orientation.as_quat()` and reuses the same `Rotation` object for every frame — that single conversion is stable. The measured column rebuilds the euler tuple on every single `Pose.from_ros_msg → as_euler('xyz')` call and each call can pick a different branch.

### 4.3 Why it matters

Any consumer that takes `observation.state.cartesian` at face value will see spurious 2π jumps in roll whenever the EE is parked near the singular orientation of the home pose (roll ≈ π, yaw ≈ −π/2). Specifically:

- **Policy learning** on top of this column will see a discontinuous feature and try to learn it, producing flaky policies whose outputs depend on which branch happened to win during recording.
- **Replay** using `14_ridgeback_replay.py --target-source obs` (reading the measured cartesian as if it were a target) would command the arm through a 2π wrist flip at every branch jump. `--target-source action` and `--target-source target` both avoid this because the commanded columns are clean — see `ridgeback_replay_orientation_bug.md` for the full replay-side story.
- **Offline metrics** (including the naive linear std used in an earlier version of the calibration report) will report a non-existent "wobble" of ~150° on roll.

### 4.4 Proposed fixes

**(a) Short-term (workaround that does not break any policy / dataset):** use `--target-source action` or `--target-source target` in `14_ridgeback_replay.py`. These read the *commanded* columns, which pass `Rotation → as_euler('xyz')` exactly once at command time and are therefore bit-stable. This is what `14_ridgeback_replay.py` already defaults to for replay. No code changes required.

**(b) Medium-term (canonicalize the measured stream):** in `crisp_py/utils/geometry.py:97`, unwrap euler output against a running reference so consecutive calls can't cross the branch cut by more than ~π. Something like:

```python
# crisp_py/utils/geometry.py
class Pose:
    _last_euler: np.ndarray | None = None  # class-level cache

    def to_pos_euler_array(self) -> np.ndarray:
        euler = self.orientation.as_euler("xyz", degrees=False)
        if Pose._last_euler is not None:
            diff = euler - Pose._last_euler
            euler -= 2 * np.pi * np.round(diff / (2 * np.pi))
        Pose._last_euler = euler.copy()
        return np.concatenate([self.position, euler], axis=0)
```

**Caveat:** this is not thread-safe, pollutes `Pose` with hidden state, and the unwrap behaviour depends on call order. A cleaner shape is to move unwrapping to `manipulator_env._get_obs` with state scoped to the env instance (so each env tracks its own reference). This is a one-liner on the consumer side:

```python
# manipulator_env._get_obs (sketch)
euler = self.robot.end_effector_pose.orientation.as_euler("xyz")
if self._last_euler is not None:
    euler -= 2 * np.pi * np.round((euler - self._last_euler) / (2 * np.pi))
self._last_euler = euler
obs['observation.state.cartesian'] = np.concatenate([pos, euler])
```

Either variant keeps the serialized column continuous whenever the arm doesn't physically cross π between frames.

**(c) Long-term (correct):** switch the env's `orientation_representation` from `"euler"` to `"quaternion"` (or `"angle_axis"`) in `ur10e_ridgeback_env.yaml`. This removes the singular representation entirely. **Breaking change** — any existing policy that assumes 6-dim pose-euler actions needs to be re-trained, and any existing dataset is pinned to the representation that was in effect when it was recorded. Do this on a clean branch, migrate all scripts that hardcode `action[3:6] = euler` (including `14_ridgeback_replay.py`'s source-reading helpers), and bump the env config version. Scripts `16_calibration_record.py` and `17_calibration_report.py` will also need to be taught about the alternative representation.

### 4.5 Reproducing the finding

```bash
cd Yunfei/crisp_gym
pixi run -e jazzy-lerobot -- python -c "
import pandas as pd, numpy as np
df = pd.read_parquet('/home/ali/.cache/huggingface/lerobot/calib_axis_001/data/chunk-000/file-000.parquet')
cart = np.stack(df['observation.state.cartesian'].to_numpy())
roll = cart[:, 3]
print('linear  std:', roll.std())
print('circular std:', np.sqrt(-2 * np.log(np.abs(np.exp(1j * roll).mean()))))
print('branch flips:', int(np.sum(np.abs(np.diff(roll)) > np.pi)))
print('min:', roll.min(), 'max:', roll.max())
"
```

Expected output:
```
linear  std: 2.597
circular std: 0.039
branch flips: 2
min: -3.141 max: 3.139
```

---

## 5. Bug #2 — `crisp_py.Gripper` self-subscription echo race

### 5.1 Symptom

`action[6]` (crisp_py convention: 1=open, 0=closed) disagrees with `1 − observation.state.gripper_target` (the LeRobot-flipped `gripper_target` column) on **25 non-transition frames**. The frames are scattered across both the "open" region (frames 0–209) and the "closed" region (frames 210–399) — not clustered near the two transition boundaries (frames 210 and 400), which are clean. Exact indices of the bad frames in this run: `[1, 21, 29, 43, 78, 94, 122, 127, 134, 136, 141, 185, 206, 292, 308, 309, 317, 321, 325, 339, …]` (25 total).

At bad frames:

| frame | action[6] | obs.gripper_target | flip diff |
| ---: | ---: | ---: | ---: |
| 0 | 1.0 | 0.0 | 0 ✓ |
| **1** | 1.0 | **1.0** | 1.0 ❌ |
| 5 | 1.0 | 0.0 | 0 ✓ |
| **21** | 1.0 | **1.0** | 1.0 ❌ |
| … | … | … | … |

The failure mode is always the same magnitude (`|Δ| = 1.0`), i.e. the recorded target flips fully to the opposite convention, never a partial value.

### 5.2 Root cause

`crisp_py/gripper/gripper.py` has an "echo" subscription on its own `target_state_topic`:

```python
# crisp_py/gripper/gripper.py:105
self._target_state_publisher = self.node.create_publisher(
    Float32, self.config.target_state_topic, qos_profile_system_default
)
self.node.create_subscription(
    Float32,
    self.config.target_state_topic,
    self._callback_target_state,
    qos_profile_system_default,
    callback_group=ReentrantCallbackGroup(),
)
```

The idea (per the docstring of `_callback_target_state`) is that a mocap teleop publisher can push gripper targets on `target_state_topic` and any subscribing `Gripper` instance will pick them up. The problem is that **the same instance also publishes to that topic from `set_target`** and then receives its own echo, and `set_target` and `_callback_target_state` write to `self._target` using **different value conventions**:

```python
# crisp_py/gripper/gripper.py:328
def _callback_target_state(self, msg: Float32):
    """Update the gripper target from the target_state topic.

    This allows any Gripper instance to track the commanded target
    published by another instance (e.g. a mocap recording script).
    The message data is a raw joint position (same units as _value).
    """
    self._target = float(msg.data)   # ← assumes msg.data is RAW

# crisp_py/gripper/gripper.py:337
def set_target(self, target: float, *, epsilon: float = 0.1):
    ...
    self._target = self._unnormalize(target)   # ← writes RAW
    msg = Float32()
    msg.data = float(target)                   # ← publishes NORMALIZED
    self._target_state_publisher.publish(msg)
```

The callback's docstring asserts `msg.data` is a raw joint position. `set_target` publishes `float(target)` — the *normalized* crisp_py value in `[0, 1]`. The two halves disagree.

With the Ridgeback 2F-85 YAML (`min_value: 0.8, max_value: 0.0`):

```python
# crisp_py/gripper/gripper.py:352-358
def _normalize(self, raw):
    return (raw - self.min_value) / (self.max_value - self.min_value)
    # = (raw - 0.8) / -0.8

def _unnormalize(self, norm):
    return (self.max_value - self.min_value) * norm + self.min_value
    # = -0.8 * norm + 0.8
```

Tracing `set_target(1.0)` (open, crisp_py convention):

1. `self._target = _unnormalize(1.0) = 0.0` ← raw hardware "fully open"
2. Publish `msg.data = 1.0` on `target_state_topic`
3. Sometime later (same thread / next executor tick), `_callback_target_state` fires:
4. `self._target = float(msg.data) = 1.0` ← **interpreted as raw**, but it's really the normalized value

Now when `env.get_obs()` reads `env.gripper.target`:

```python
# crisp_py/gripper/gripper.py:241
@property
def target(self) -> float:
    return np.clip(self._normalize(self._target), 0.0, 1.0)
```

With the echoed `_target = 1.0`: `_normalize(1.0) = (1.0 − 0.8) / −0.8 = −0.25 → clip → 0.0`. So `gripper.target` returns `0.0` (crisp_py: closed), even though `set_target(1.0)` had just been called (crisp_py: open).

`manipulator_env._get_obs` then stores `observation.state.gripper_target = 1.0 − 0.0 = 1.0` (LeRobot: closed). Flipping back in the report: `1.0 − 1.0 = 0.0`, compared against `action[6] = 1.0` → diff = 1.0.

**Whether a given frame is "good" or "bad" depends entirely on whether the ROS executor scheduled `_callback_target_state` between the `set_target` call and the next `env.get_obs()` on that same tick.** With a `ReentrantCallbackGroup` and a loopback subscription, the timing is non-deterministic, which is exactly the scattered pattern we see.

### 5.3 Why the actual gripper command was still correct

`gripper.set_target` both:
- writes `self._target = _unnormalize(target)` (raw)
- and publishes a `GripperCommand` action goal to `self.config.command_topic` via `self._callback_publish_target` (timer-driven, 20 Hz)

The `command_topic` action is the thing the Robotiq 2F-85 driver actually listens to. That path receives the *raw* `_target` value (through `_normalize → _denormalize` or equivalent) and correctly sends the open/close command. The 25 flickery frames are entirely a logging artifact — the physical gripper opened and closed on the intended transition frames, and `observation.state.gripper` (the measured position) confirms it (open ≈ 0.99, closed ≈ 0.00, as expected in LeRobot convention).

This is important to say out loud: **the recorded action column `action[6]` is the authoritative gripper command for this dataset**. The `gripper_target` observation column is unreliable. Downstream consumers (policy training, replay) should use `action[6]` for the gripper command, not `observation.state.gripper_target`.

### 5.4 Proposed fixes

All three edits live in `crisp_py/gripper/gripper.py` — no change to `crisp_gym` or any recorder script.

**(a) Cleanest:** make the callback consistent with the publisher by applying `_unnormalize`:

```python
# crisp_py/gripper/gripper.py:328
def _callback_target_state(self, msg: Float32):
    """Update the gripper target from the target_state topic.

    The message convention matches what `set_target` publishes: a
    normalized crisp_py value in [0, 1]. Convert to raw before storing.
    """
    self._target = self._unnormalize(float(msg.data))
```

This is one line and is backwards-compatible with any external publisher that already uses crisp_py-normalized values on `target_state_topic`. Any external publisher that was publishing *raw* values (which the docstring claimed was the convention) would break — but given that `set_target` is the only thing in the repo that actually publishes on this topic, such an external publisher is unlikely to exist.

**(b) More invasive but more correct:** drop the self-subscription entirely. A `Gripper` does not need to subscribe to its own echo; the point of the subscription (per its docstring) is to track commands from *another* publisher, which is useful for mocap teleop but bad for single-instance use. Options:

- Remove lines 108–114 of `gripper.py` unconditionally.
- Or gate the subscription behind a `Gripper.__init__` argument: `subscribe_target: bool = False`, defaulting to off.

**(c) Defensive / conservative (if (a) breaks someone):** filter echoes by tracking the last published value and ignoring a callback whose data matches it. Fragile — two near-identical commands in quick succession will still break.

My recommendation is **(a)**. One line, backwards-compatible in practice, preserves the external-publisher use case, and eliminates the race entirely.

**Testing the fix:**

```bash
# After editing crisp_py/gripper/gripper.py:
cd Yunfei/crisp_gym
rm -rf ~/.cache/huggingface/lerobot/calib_axis_gripper_fix
pixi run -e jazzy-lerobot python examples/16_calibration_record.py \
    --repo-id calib_axis_gripper_fix -y
pixi run -e jazzy-lerobot python examples/17_calibration_report.py \
    --repo-id calib_axis_gripper_fix | grep -A3 "action\[6\] vs"
```

Expected: `max |Δ| = 0.0000, frames with |Δ| > 0.05 = 0`.

### 5.5 Reproducing the finding

```bash
cd Yunfei/crisp_gym
pixi run -e jazzy-lerobot -- python -c "
import pandas as pd, numpy as np
df = pd.read_parquet('/home/ali/.cache/huggingface/lerobot/calib_axis_001/data/chunk-000/file-000.parquet')
act = np.stack(df['action'].to_numpy())[:, 6]
tgt = np.array([float(np.asarray(v).flatten()[0]) for v in df['observation.state.gripper_target'].to_numpy()])
diff = act - (1.0 - tgt)
print('max diff:', np.max(np.abs(diff)))
print('bad frames:', int(np.sum(np.abs(diff) > 0.05)))
print('first 20 bad frame indices:', np.where(np.abs(diff) > 0.05)[0][:20].tolist())
# Confirm transitions themselves are clean
trans = np.where(np.abs(np.diff(act)) > 0.5)[0]
print('transitions at frames:', trans.tolist())
print('diff at transitions:', diff[trans].tolist(), diff[trans + 1].tolist())
"
```

Expected output:
```
max diff: 1.0
bad frames: 25
first 20 bad frame indices: [1, 21, 29, 43, 78, 94, 122, 127, 134, 136, 141, 185, 206, ...]
transitions at frames: [209, 399]
diff at transitions: [0.0, 0.0] [0.0, 0.0]
```

---

## 6. Not-bugs

These are also in the calibration output but should not be classified as bugs, for context:

### 6.1 ~29 mm RMS cartesian tracking lag

Expected behaviour of the cartesian impedance controller against a 5 cm/s step input. RMS 29 mm / max 49 mm corresponds roughly to a 600 ms time constant at 5 cm/s. Lower velocity will linearly reduce lag. The `move_7_minus_y` phase is marginally worse than its mirror (`move_5_plus_y`: 36 mm vs 40 mm RMS); a slower re-run (`--velocity 0.02`) would let you decide whether the asymmetry is real or noise.

### 6.2 `frame-too-long` warnings during recording

The live log of the recording session showed dozens of `Frame processing took too long` warnings ranging from ~3 ms to ~545 ms over the 50 ms nominal budget. The total wall-clock of the recording was ~46 s for 410 frames, which is ~9 Hz effective, not 20 Hz. The saved `timestamp` column does not show this because LeRobot v3 stores synthetic `frame_index / fps` timestamps.

Probable cause: `env.get_obs()` pulls a 640 × 480 camera frame from the Orbbec ROS topic each tick, and the image copy + `np.asarray` conversion occasionally blows the 50 ms budget. The recorder does not drop frames — it just logs a warning and runs late, which is correct behaviour for an index-keyed calibration trajectory.

This is tangentially relevant: the "commanded" velocity was 5 cm/s in the script, but the *actual* velocity was ~2.2 cm/s due to the slow frame cadence, so the tracking lag numbers above are against a 2.2 cm/s motion. Re-running with a lower image resolution or without the camera would give a cleaner timing baseline but loses visual data needed for imitation learning. Out of scope for this calibration.

### 6.3 Missing `joint_state_broadcaster` in `/gripper/controller_manager`

A prerequisite bug — not in any recording or env file, but in the gripper bringup on the real robot's `clearpath_robot_ws`. Its spawner list is missing `joint_state_broadcaster`, so `/gripper/joint_states` has Publisher count = 0 until the broadcaster is manually loaded. Fix belongs in the gripper bringup launch file on the onboard computer.

---

## 7. Action items

Ordered by value vs. effort.

| # | Item | Effort | Breaking? |
| --- | --- | --- | --- |
| 1 | **Fix gripper echo race** — one-line edit to `crisp_py/gripper/gripper.py:328` (option (a)), then re-run `16_calibration_record.py` + `17_calibration_report.py` and confirm 0 bad gripper frames. | Trivial | No (in practice) |
| 2 | **Add `matplotlib` to `jazzy-lerobot` pixi env** so `17_calibration_report.py --plot` emits position/orientation/tracking plots. | Trivial | No |
| 3 | **Euler unwrap in `manipulator_env._get_obs`** — per-env cache, short-circuits the branch flip for all future datasets without changing representation. | Small | No |
| 4 | **Fix `clearpath_robot_ws` gripper bringup** to spawn `joint_state_broadcaster` automatically, so `env.wait_until_ready()` no longer needs a hotfix. | Small | No |
| 5 | **Re-record calibration at `--velocity 0.02 --dwell-seconds 1.5`** for a cleaner tracking baseline without controller-lag confounds. Save as `calib_axis_002`. | Trivial | No |
| 6 | **Switch `orientation_representation` to `"quaternion"`** in `ur10e_ridgeback_env.yaml`. Removes the singular representation permanently. | Medium | **Yes** — breaks existing policies and recorded datasets |
| 7 | **Drop the gripper self-subscription** (option (b)) — cleaner long-term architecture. | Small | Potentially (if external publishers exist) |

---

## 8. Files and where they live

- `Yunfei/crisp_gym/examples/16_calibration_record.py` — the recorder (new, standalone).
- `Yunfei/crisp_gym/examples/17_calibration_report.py` — the offline analyzer (new, pure numpy, no ROS).
- `Yunfei/crisp_gym/docs/ridgeback_calibration_recording_plan.md` — original motivation.
- `Yunfei/crisp_gym/docs/ridgeback_calibration_recording.md` — how the recorder works.
- `Yunfei/crisp_gym/docs/ridgeback_calibration_findings.md` — **this document.**
- `Yunfei/crisp_gym/analysis/calib_axis_001_calibration_report.md` — the raw report output from `17_calibration_report.py` for this dataset.
- `~/.cache/huggingface/lerobot/calib_axis_001/` — the LeRobot dataset itself (410 frames + video).
