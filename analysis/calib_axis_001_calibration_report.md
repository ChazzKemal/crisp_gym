# Calibration report — `calib_axis_001`

- Dataset path: `/home/ali/.cache/huggingface/lerobot/calib_axis_001`
- Robot type: `ur10e`
- Frames: **410**   (spec expected 410)
- Nominal fps: **20**

## Trajectory spec used for ground truth

| param | value |
| --- | --- |
| velocity | 0.05 m/s |
| displacement | 0.1 m |
| dwell_seconds | 0.5 s |
| fps | 20 |
| include_gripper_toggle | True |
| n_motion (frames / phase) | 40 |
| n_dwell (frames / dwell) | 10 |
| n_total (8·n_motion + 9·n_dwell) | 410 |

**If these parameters don't match the recording, every metric below is misaligned. Pass the same values to this script and the recorder.**

## Anchor pose (frame 0 action[:6])

```
position (m)  : [  +0.9311,   +0.1697,   +0.8658]
orientation   : [  -3.1408,   -0.0037,   -1.5739]  (Euler XYZ, rad)
```

## 1. Action vs ground truth

The recorder's `action[:6]` should *exactly* equal the commanded setpoint at every frame, since both come from the same `Pose` object in the script. Non-zero diff means either the spec passed to this analyzer doesn't match the recording, or `Pose.to_array` is mangling the command on its way to disk.

| metric | value |
| --- | --- |
| max ‖Δpos‖ (m) | 3.814697e-08 |
| mean ‖Δpos‖ (m) | 9.761351e-09 |
| max ‖Δrot‖ (rad) | 0.000000e+00 |
| mean ‖Δrot‖ (rad) | 0.000000e+00 |

OK — action exactly matches ground truth (float64 noise only).

## 2. Orientation invariance

The recorder holds orientation constant at `R0` for every frame. `action[3:6]` and `observation.state.target[3:6]` should have std **exactly zero** (linear) — the recorder writes a literally constant value. For the *measured* column `observation.state.cartesian[3:6]` we need two statistics: linear std catches physical wobble, but it also blows up near Euler singularities (±π) because `Rotation.as_euler('xyz')` can flip sign on a stationary rotation. **Circular std** is the right metric for measured angles — if it is small while linear std is huge, the physical rotation is constant and you are staring at a representation artifact, not a hardware wobble.

| column | std kind | roll | pitch | yaw |
| --- | --- | --- | --- | --- |
| action[3:6]                      | linear   | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| observation.state.target[3:6]    | linear   | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| observation.state.cartesian[3:6] | linear   | 2.597e+00 | 3.978e-02 | 2.137e-02 |
| observation.state.cartesian[3:6] | circular | 3.907e-02 | 3.978e-02 | 2.137e-02 |

**Euler-XYZ branch flips per axis** (|Δ| > π between consecutive frames): roll=2, pitch=0, yaw=0.

OK — `action[3:6]` is exactly constant (good).
OK — `observation.state.target[3:6]` is exactly constant (good).
⚠  Measured cartesian orientation has huge linear std (2.60 rad) but tiny circular std (3.98e-02 rad), and 2 Euler-XYZ branch flips. The physical rotation is essentially constant; the linear std is a **representation artifact** of `Rotation.as_euler('xyz')` flipping sign at ±π. This is the bug documented in `docs/ridgeback_replay_orientation_bug.md` — the *measured* column is unreliable, but the *commanded* column is clean, so replay via `--target-source action` or `--target-source target` is the right workaround until the representation is switched away from Euler-XYZ (quaternion or angle-axis).

## 3. Action vs recorded env target

`observation.state.target[:6]` is what `env.robot.target_pose` returned at each frame — i.e. what the env-side `crisp_py.Robot` thinks its internal target is. Since `16_calibration_record.py` calls `env.robot.set_target(pose=...)` directly, this should equal `action[:6]` to sub-mm / sub-mrad. Any divergence means `set_target` is mutating the target, or `target_pose` is stale.

| metric | value |
| --- | --- |
| max |Δpos| per-axis (m) | 0.000000e+00 |
| max |Δrot| per-axis (rad) | 0.000000e+00 |

OK — action matches recorded env target to sub-mm / sub-mrad.

## 4. Position fidelity per phase

Euclidean distance between commanded `action[:3]` and measured `observation.state.cartesian[:3]`. This is **not a bug detector** — it measures cartesian impedance controller lag, which is proportional to speed / stiffness. Use it to see whether the trajectory was tracked, and whether the symmetry check in §4.5 of the plan doc passes.

| phase | frames | RMS err (mm) | max err (mm) | GT Δ start (mm) | GT Δ end (mm) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dwell_0_anchor` | 0..10 (10) |    0.05 |    0.06 | [  +0.0,   +0.0,   +0.0] | [  +0.0,   +0.0,   +0.0] |
| `move_1_plus_z` | 10..50 (40) |   28.05 |   40.26 | [  +0.0,   +0.0,   +2.5] | [  +0.0,   +0.0, +100.0] |
| `dwell_1_plus_z` | 50..60 (10) |   26.16 |   31.99 | [  +0.0,   +0.0, +100.0] | [  +0.0,   +0.0, +100.0] |
| `move_2_return` | 60..100 (40) |   21.05 |   30.84 | [  +0.0,   +0.0,  +97.5] | [  +0.0,   +0.0,   +0.0] |
| `dwell_2_anchor` | 100..110 (10) |   20.65 |   24.76 | [  +0.0,   +0.0,   +0.0] | [  +0.0,   +0.0,   +0.0] |
| `move_3_minus_z` | 110..150 (40) |   32.35 |   39.46 | [  +0.0,   +0.0,   -2.5] | [  +0.0,   +0.0, -100.0] |
| `dwell_3_minus_z` | 150..160 (10) |   26.72 |   37.39 | [  +0.0,   +0.0, -100.0] | [  +0.0,   +0.0, -100.0] |
| `move_4_return` | 160..200 (40) |   21.31 |   32.97 | [  +0.0,   +0.0,  -97.5] | [  +0.0,   +0.0,   +0.0] |
| `dwell_4_anchor` | 200..210 (10) |   22.22 |   26.49 | [  +0.0,   +0.0,   +0.0] | [  +0.0,   +0.0,   +0.0] |
| `move_5_plus_y` | 210..250 (40) |   36.06 |   45.31 | [  +0.0,   +2.5,   +0.0] | [  +0.0, +100.0,   +0.0] |
| `dwell_5_plus_y` | 250..260 (10) |   37.97 |   40.43 | [  +0.0, +100.0,   +0.0] | [  +0.0, +100.0,   +0.0] |
| `move_6_return` | 260..300 (40) |   28.18 |   37.44 | [  +0.0,  +97.5,   +0.0] | [  +0.0,   +0.0,   +0.0] |
| `dwell_6_anchor` | 300..310 (10) |   28.52 |   34.08 | [  +0.0,   +0.0,   +0.0] | [  +0.0,   +0.0,   +0.0] |
| `move_7_minus_y` | 310..350 (40) |   40.12 |   48.94 | [  +0.0,   -2.5,   +0.0] | [  +0.0, -100.0,   +0.0] |
| `dwell_7_minus_y` | 350..360 (10) |   34.77 |   43.05 | [  +0.0, -100.0,   +0.0] | [  +0.0, -100.0,   +0.0] |
| `move_8_return` | 360..400 (40) |   27.60 |   36.09 | [  +0.0,  -97.5,   +0.0] | [  +0.0,   +0.0,   +0.0] |
| `dwell_8_anchor` | 400..410 (10) |   28.68 |   36.09 | [  +0.0,   +0.0,   +0.0] | [  +0.0,   +0.0,   +0.0] |

Overall: RMS cartesian error = **29.38 mm**, max = **48.94 mm**.

## 5. Gripper consistency

`action[6]` uses crisp_py convention (1=open, 0=closed) while `observation.state.gripper` and `.gripper_target` use LeRobot convention (0=open, 1=closed). This section flips the obs to crisp_py convention before subtracting.

| comparison | max |Δ| | frames with |Δ| > 0.05 |
| --- | ---: | ---: |
| action[6] vs (1 − gripper_target) | 1.0000 | 25 |
| action[6] vs (1 − gripper)        | 1.0000 | (measurement lag, not a bug) |

✅ Transitions: `action[6]` == `gripper_target` at all 2 transition boundaries and their follow-up frames.

⚠  `action[6]` ≠ `gripper_target` on 25 **non-transition** frames, scattered (not clustered). Transitions themselves are clean. This is the **crisp_py gripper echo race**: `Gripper.set_target(v)` writes `_target = _unnormalize(v)` (raw hardware value), publishes to `target_gripper_state`, and the same instance's `_callback_target_state` subscription receives the echo and writes back `_target = float(msg.data)` (normalized crisp_py value). The two conventions collide, and depending on whether the echo callback fires before the next `env.get_obs()`, the `gripper.target` property reads a raw or a normalized value and the obs column flickers. It does **not** mean the actual gripper command was wrong — `action[6]` in this dataset is the authoritative commanded value. Fix belongs in `crisp_py/gripper/gripper.py`: make `_callback_target_state` also apply `_unnormalize`, or drop the self-subscription entirely.

## 6. Timing — ⚠ synthetic timestamps

**The `timestamp` column in LeRobot v3 parquet is `frame_index / fps`, not wall-clock.** It does NOT reflect how long the recorder actually spent in each frame. If the recorder logged `frame-too-long` warnings during the run, those are invisible here — the saved timestamps will still look perfect.

| metric | value |
| --- | --- |
| span | 20.450 s |
| mean dt | 50.00 ms |
| median dt | 50.00 ms |
| max dt | 50.00 ms |
| min dt | 50.00 ms |
| stddev dt | 0.001 ms |
| effective fps (1/mean dt) | 20.00 |

Timestamps are uniformly spaced (synthetic). True wall-clock timing was not captured. To measure real per-frame jitter you would need to log `time.monotonic()` inside `data_fn` during recording and diff it post-hoc.

## Summary

- ✅ action = ground truth
- ✅ action orientation is constant
- ✅ recorded target orientation is constant
- ✅ action == env.robot.target_pose
- ⚠  measured cartesian orientation is stable angularly (circular std 3.98e-02) but linear std is 2.60 rad and there are 2 Euler-XYZ branch flips — representation artifact, see `docs/ridgeback_replay_orientation_bug.md`
- ⚠  gripper: transitions clean but 25 non-transition frames flicker — crisp_py gripper echo race (see §5)
- ℹ  RMS cartesian tracking error 29.38 mm (not a bug, controller lag)
- ℹ  final dwell cartesian offset from anchor: 26.57 mm
