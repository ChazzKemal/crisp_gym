# Scripted calibration recording for record/replay verification

**Status:** plan only — no code written yet.
**Goal:** validate the LeRobot recording pipeline end-to-end with a *deterministic* arm trajectory (no mocap, no SpaceMouse, no human input), so that any mismatch between commanded / recorded / replayed motion is unambiguously a bug in the recording or replay path rather than mocap-quality variability.
**Related docs:**
- `docs/ridgeback_target_pose_ownership.md` — `/target_pose` ownership during recording vs. replay
- `docs/ridgeback_replay_orientation_bug.md` — the Euler-XYZ orientation discontinuity bug that motivated this calibration

---

## 1. Why a scripted calibration is the right test

The current mocap recording flow has three independent moving parts: `track_mocap.py` (publisher), `crisp_py.Robot` + `manipulator_env._get_obs` (recording-side observers), and `cartesian_controller` (executor). When `ridgeback_223335` failed to replay correctly, we could not initially tell whether:

1. the **mocap stream** was already noisy / discontinuous before recording,
2. the **env's `_get_obs`** was distorting the orientation columns on the way to disk,
3. the **replay script** was reconstructing the wrong rotation from disk, or
4. the **cartesian controller** was simply lagging behind a fast demonstration.

A scripted calibration trajectory removes (1) and (4) as confounders. The script knows *exactly* what setpoints it sent — they are computed in code, not measured from a physical input device — and the trajectory is slow and smooth enough that the controller has no excuse to lag. Whatever ends up on disk can be diffed against the ground-truth setpoint sequence directly. If the diff is nonzero, the recording path is broken. If the diff is zero but replay still desyncs, the replay path is broken. The two failure modes become independently observable for the first time.

This calibration is also reusable as a smoke test after every change to the env, the recorder, or `crisp_py`.

---

## 2. Architecture

### 2.1 New script

A new example script, e.g. `examples/16_calibration_record.py`, that **drives the arm itself** along a fixed trajectory and records each commanded frame as a LeRobot v3 dataset using the same `RecordingManager` + `get_features` infrastructure as `13_ridgeback_mocap_record.py` / `15_ridgeback_mocap_record.py`. No mocap, no external publisher.

### 2.2 Topic ownership

The script needs `env.robot` to **own** `/target_pose` (publish, not subscribe). The current `ur10e_ridgeback_env.yaml:67` has `publish_target_pose: false` for the mocap flow, which causes `crisp_py.Robot.__init__` to *not* create the publisher and instead subscribe.

We re-use the runtime patch already implemented in `14_ridgeback_replay.py:269` (`enable_env_target_pose_publishing`):

1. Destroy `robot._target_pose_subscriber` if present.
2. Create `robot._target_pose_publisher` against `target_pose_topic`.
3. Create the 20 Hz publish timer that calls `robot._callback_publish_target_pose`.
4. Flip `robot.config.publish_target_pose = True`.

This is the inverse of `silence_env_target_publishers` from script 13/15 and must run **before** `wait_until_ready()` and **before** the first `set_target` call.

**Alternative considered:** add a separate yaml `ur10e_ridgeback_calib_env.yaml` with `publish_target_pose: true`. Rejected — duplicating the yaml means another file to keep in sync. The runtime patch is fine because it is already proven by script 14.

### 2.3 Recorder data flow

The recorder's `data_fn` will both **issue the next setpoint** and **return the (obs, action) row** for that frame:

```python
def data_fn():
    target_pose, target_grip = trajectory.step()    # advance pre-built setpoint sequence
    env.robot.set_target(pose=target_pose)
    env.gripper.set_target(target_grip)
    obs = env.get_obs()
    action = pack(target_pose, target_grip)         # crisp_py gripper convention
    return obs, action
```

`RecordingManager` ticks `data_fn` at `fps` (20 Hz), so the trajectory is materialized as a list of length `n_frames` and stepped once per call. The episode length is fixed (no keyboard prompts), and `RecordingManager` should close the episode automatically once the trajectory ends. We will need to look at `RecordingManager`'s API for the right hook — `record_episode` already takes `data_fn`; we just need to either signal "done" from inside `data_fn` (e.g. by raising `StopIteration`) or pick a recorder type that records a fixed number of frames. **Design decision deferred to implementation time after a quick read of `crisp_gym/record/recording_manager.py`.**

---

## 3. Trajectory definition

### 3.1 Anchor and orientation

Starting from `env.home()`, capture the resulting cartesian pose as the **anchor** `P0 = (x0, y0, z0, R0)`. The orientation `R0` is held **constant** for the entire test — pure translation, no wrist motion.

This is deliberate: it isolates the orientation-discontinuity bug we already diagnosed (`docs/ridgeback_replay_orientation_bug.md`). If the recorded yaw / roll / pitch wobble at all on this dataset, that proves the bug is in the env representation path independent of mocap.

### 3.2 Phases (translations in `arm_0_base_link` frame, `+z = world up`)

| Phase | Motion | End pose | Reason |
| --- | --- | --- | --- |
| 0 | Dwell at home, gripper open | `P0` | Capture a clean reference frame at the start |
| 1 | Up   10 cm in `+z` | `P0 + (0, 0, +0.10)` | Forward Z |
| 2 | Down 10 cm back to anchor | `P0` | Return |
| 3 | Down 10 cm in `-z` | `P0 + (0, 0, -0.10)` | Reverse Z |
| 4 | Up   10 cm back to anchor | `P0` | Return |
| 5 | Left  10 cm in `+y` | `P0 + (0, +0.10, 0)` | Forward Y |
| 6 | Right 10 cm back to anchor | `P0` | Return |
| 7 | Right 10 cm in `-y` | `P0 + (0, -0.10, 0)` | Reverse Y |
| 8 | Left  10 cm back to anchor | `P0` | Return |
| 9 | Dwell at anchor | `P0` | Capture a clean reference frame at the end |
| 10 | Joint-space `env.home()` outside the recorded episode | home | Safety, matches `on_end` in 13/15 |

### 3.3 Sampling

Each motion phase is interpolated **linearly** in cartesian space at constant velocity, sampled at `fps = 20 Hz`. With `v = 5 cm/s`, each 10 cm phase takes `2.0 s = 40 frames`. A `0.5 s` dwell (`10 frames`) is inserted at the anchor between phases for clean segment boundaries.

Approximate budget:

```
8 motion phases × 40 frames  = 320 frames
9 dwells       × 10 frames   =  90 frames
                  --------------------
total                         ≈ 410 frames ≈ 20.5 s
```

Adjustable via CLI flags `--velocity`, `--dwell-seconds`, `--displacement`.

### 3.4 Constant orientation

The orientation in every phase is the **same** `R0` quaternion held constant. We send it through `Pose(orientation=R0)` so the env's Euler conversion gets a stationary input. Any nonzero variation in `observation.state.cartesian[3:6]` or `observation.state.target[3:6]` across frames is then a clean measurement of representation noise versus solver noise.

### 3.5 Optional gripper exercise

A binary gripper toggle is added at one frame in the middle (e.g. close at phase 5 start, open at phase 8 end) so that `gripper_target` and `action[6]` get exercised against `observation.state.gripper`. Optional — see Q4 in §7.

---

## 4. Verification we can run on the resulting dataset

After recording `calib_axis_001`, we open it with `notebooks/inspect_dataset.ipynb` (or a small dedicated verification notebook) and check, **without ever needing the mocap stream**:

### 4.1 Position fidelity (record path)

For each phase, the commanded `action[:3]` is a known straight line in time. Compare against:

- `observation.state.target[:3]` (env's own target snapshot via `robot.target_pose`) → should be equal modulo a 1-frame lag.
- `observation.state.cartesian[:3]` (measured EE) → should track within the cartesian impedance controller's typical lag (a few mm at 5 cm/s, depending on stiffness).

The straight-line geometry makes outliers obvious — any deviation off the line is either controller noise or a bug.

### 4.2 Orientation invariance (record path)

The script holds orientation constant. So:

- `max( std(observation.state.cartesian[3:6], axis=0) )` should be tiny. If it isn't, the env's `as_euler('xyz')` is producing branch flips on a literally stationary input → representation bug confirmed.
- `max( std(observation.state.target[3:6], axis=0) )` should be **exactly zero** if `robot.target_pose` is faithfully echoing what `set_target` wrote. If it's nonzero, `crisp_py.Robot.set_target` is mutating the orientation somewhere.
- `max( std(action[3:6], axis=0) )` should also be **exactly zero**, since we wrote a constant.

### 4.3 Action vs. target consistency

`max | action[:6] − observation.state.target[:6] |` should be sub-millimetre / sub-milliradian. Anything bigger means the env writes a different target than the script intends.

### 4.4 Replay round-trip

Run `examples/14_ridgeback_replay.py --repo-id calib_axis_001 --episode-idx 0` twice:

- Once with `--target-source target`.
- Once with `--target-source action`.

For both:

- Record the joint trajectory of the replayed run via `ros2 bag` or a small subscriber, *or* extend the replay script with a per-frame dump of `env.robot.current_pose`.
- The replayed cartesian trajectory should reproduce the recorded `observation.state.cartesian` track within the same controller-lag envelope as the original.
- Visually: the EE traces the same up/down/left/right pattern. **No wrist flips at any phase.**

If replay drifts even on this dataset, the bug is conclusively in the replay / representation path. If replay is clean here but mocap data is still broken, the bug is in the mocap path.

### 4.5 Symmetry test

Phase 1+2 and phase 3+4 should have mirror-symmetric position traces around `z = z0`. Phase 5+6 and phase 7+8 likewise around `y = y0`. If they don't, there's a controller asymmetry (gravity comp, friction, cable pull) — useful to know but unrelated to the recording bug.

---

## 5. Risks and things to be careful about

- **Workspace limits.** `ur10e_ridgeback_env.yaml:102-107` gives `z ∈ [0.05, 1.0]`, `x,y ∈ [-0.9, 0.9]`. The home cartesian pose for `ridgeback_223335` was around `z ≈ 0.86`, so `±0.10` stays inside. The script should still query the actual home pose at runtime and clip / abort if necessary.
- **Cartesian controller lag.** CRISP cartesian impedance does not perfectly track straight lines — there will be a position lag proportional to velocity divided by stiffness. 5 cm/s is gentle enough that the lag is small. We could optionally use `joint_trajectory_controller` for the calibration motion and only use cartesian for the recording-side comparison, but that complicates the script. Default: cartesian + slow.
- **`env.home()` vs. trajectory end.** `env.home()` does a joint-space move via JTC. For safety we keep phase 10 *outside* the recorded episode (matches the `on_end` pattern in 13/15). Inside the episode, the trajectory ends with a dwell at the anchor.
- **Gripper convention bookkeeping.** As documented in script 14, `action[6]` is crisp_py convention (`1=open, 0=closed`) and `observation.state.gripper` is LeRobot-flipped (`1 - gripper.value`, so `0=open, 1=closed`). The calibration script must write `action[6]` in **crisp_py convention** for consistency with the existing format. The verification step has to flip when comparing to `observation.state.gripper`.
- **Single-shot recording manager.** `RecordingManager` is keyboard-driven by default. We need to either pick a non-keyboard recorder type or signal completion from inside `data_fn`. To be confirmed at implementation time.

---

## 6. Recommendation

Implement `examples/16_calibration_record.py` as a single-episode recorder that runs the trajectory in §3 with `velocity=5 cm/s`, `dwell=0.5 s`, `displacement=10 cm`, constant orientation, optional gripper toggle. Save to `~/.cache/huggingface/lerobot/calib_axis_001`. Run the verification checks in §4 from a small notebook (or extend `notebooks/inspect_dataset.ipynb` with a "calibration mode" branch).

Run this calibration **before** applying the §6 fixes from `ridgeback_replay_orientation_bug.md`. That gives us a baseline showing exactly which columns are wrong. Then apply the fix and re-run — same calibration script gives a clean dataset to compare against. The script itself is fix-agnostic and remains useful as a regression test forever after.

---

## 7. Open questions before implementation

1. **Frame for "left/right".** From the operator's standpoint (human standing where `track_mocap.py` user normally stands), is "left" `+y` in `arm_0_base_link`, or `-y`? Default assumption: `+y = left`. Confirm.

2. **Single combined trajectory or one episode per axis?**
   - **(a)** One episode containing all 8 segments → easier to inspect as a single timeline, single video.
   - **(b)** One episode per axis pair (z, then y) → simpler to debug if one direction fails.

   Default: **(a)**.

3. **Phase 10 — final return to home.**
   - **(a)** Recorded as a final cartesian segment (collapses to a dwell because anchor == home).
   - **(b)** Done outside the recorded episode via `env.home()` joint move, like 13/15 do in `on_end`.

   Default: **(b)** — safer, matches existing pattern.

4. **Gripper toggle.** Include a single open → close → open cycle somewhere in the middle of the trajectory (so we exercise the gripper logging path), or leave the gripper at "open" the whole time (cleanest possible isolation of arm behaviour)?

   Default: include it — almost free, exercises another data column.

5. **Velocity / segment duration.** Proposed 5 cm/s → 2 s per 10 cm segment. Slower (e.g. 2 cm/s, 5 s/segment) reduces controller lag and gives a cleaner record-vs-replay comparison at the cost of a longer episode. Faster goes the other way.

   Default: 5 cm/s.

6. **Dataset name.** Default `--repo-id calib_axis_001`. Easy to grep for in `~/.cache/huggingface/lerobot/`.

7. **Where to put the calibration recorder.**
   - **(a)** `examples/16_calibration_record.py` (next to 13/14/15, parallel with the existing numbered scripts).
   - **(b)** `crisp_gym/scripts/` (it is more of a tool than a tutorial example).

   Default: **(a)** for discoverability and parallelism.

8. **Run the calibration before or after applying the orientation-bug fixes?**

   Default: **before**, so we have a baseline that demonstrates the bug on this dataset, then re-run after the fix to confirm it disappears. The script itself is fix-agnostic.
