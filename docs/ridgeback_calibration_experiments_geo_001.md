# Geometric calibration experiments — first run (2026-04-15)

First pass of `18_calibration_record_geo.py` on the real Ridgeback stack.
Two runs were attempted; neither produced a usable calibration dataset,
but both surfaced distinct failure modes that need fixing before the next
attempt. This document is the experiment log — evidence, diagnosis,
actions.

**Related docs:**
- [`ridgeback_calibration_recording_geo.md`](./ridgeback_calibration_recording_geo.md) — operator guide for the recorder (script 18).
- [`ridgeback_calibration_recording.md`](./ridgeback_calibration_recording.md) — operator guide for the axis-aligned sibling (script 16).
- [`ridgeback_calibration_recording_plan.md`](./ridgeback_calibration_recording_plan.md) — motivation and acceptance checks.

**Source scripts:**
- Recorder: `examples/18_calibration_record_geo.py`
- Analyzer: `examples/19_calibration_report_geo.py` (used to generate every metric below)

---

## 1. Experiments attempted

| # | Repo ID                | Shape  | Size (m) | Plane | Loops | Result                                      |
|---|------------------------|--------|----------|-------|-------|---------------------------------------------|
| 1 | `calib_geo_square_001` | square | **0.05** | xy    | 1     | **Wrote dataset, physical motion degraded** |
| 2 | `calib_geo_square_002` | square | **0.5**  | xy    | 1     | **Workspace validation aborted before motion** |

Commands, verbatim:

```bash
pixi run -e jazzy-lerobot python examples/18_calibration_record_geo.py \
    --repo-id calib_geo_square_001 --shape square --size 0.05

pixi run -e jazzy-lerobot python examples/18_calibration_record_geo.py \
    --repo-id calib_geo_square_002 --shape square --size 0.5
```

Both runs used default `--velocity 0.05`, `--plane xy`, `--loops 1`,
`--dwell-seconds 0.5`, `--include-gripper-toggle`.

Anchor pose (identical within 0.05 mm across both runs, since both called
`env.home()` from the same base pose):

```
position    : [+0.9311, +0.1697, +0.8658]  m
orientation : [-3.1408, -0.0036, -1.5739]  Euler XYZ, rad
```

---

## 2. Run 1 — `calib_geo_square_001` (`--size 0.05`)

### 2.1 What was on screen

The script built a 120-frame trajectory and auto-started recording. During
the 120-frame run, `recording_manager` printed 12 separate
`Frame processing took too long` warnings:

```
Frame processing took too long: 0.019 s too long i.e. 14.50 FPS
Frame processing took too long: 0.131 s too long i.e.  5.53 FPS
Frame processing took too long: 0.169 s too long i.e.  4.56 FPS
Frame processing took too long: 0.123 s too long i.e.  5.80 FPS
Frame processing took too long: 0.166 s too long i.e.  4.64 FPS
Frame processing took too long: 0.081 s too long i.e.  7.66 FPS
Frame processing took too long: 0.011 s too long i.e. 16.33 FPS
Frame processing took too long: 0.018 s too long i.e. 14.73 FPS
Frame processing took too long: 0.130 s too long i.e.  5.57 FPS
Frame processing took too long: 0.410 s too long i.e.  2.17 FPS
Frame processing took too long: 0.119 s too long i.e.  5.90 FPS
Frame processing took too long: 0.325 s too long i.e.  2.67 FPS
```

Target was 20 Hz; actual instantaneous rate at the slowest frames was
**2.17 FPS** (a single `data_fn` took 0.46 s). Wall-clock duration of the
episode (from `Started recording episode.` at 22:10:28 to `Saving` at
22:10:41) was ~13 s, where the nominal 120 frames at 20 Hz would be 6 s.
Effective average rate: **~9 Hz**, roughly half of target.

The dataset saved successfully. Parquet, video, and metadata all present
at `~/.cache/huggingface/lerobot/calib_geo_square_001/`.

### 2.2 What the analyzer says

`19_calibration_report_geo.py` against this dataset:

```bash
pixi run -e jazzy-lerobot python examples/19_calibration_report_geo.py \
    --repo-id calib_geo_square_001 --shape square --size 0.05
```

Report summary (abridged):

```
- ✅ action = ground truth
- ✅ action orientation is constant
- ✅ recorded target orientation is constant
- ✅ action == env.robot.target_pose
- ⚠  measured cartesian orientation stable angularly (circular std 1.91e-02)
     but linear std 2.93 rad and 2 Euler-XYZ branch flips — representation artifact
- ❌ measured path length is only 25.1% of commanded — the arm did not follow.
- ℹ  shape closes within 8.53 mm (controller settling)
- ⚠  gripper: transitions clean but 7 non-transition frames flicker
- ℹ  RMS cartesian tracking error 19.60 mm (controller lag)
```

Key quantitative findings:

**Recording-layer invariants are all clean.** The recording path itself
(what the script writes to disk) is correct — no bugs there.

| invariant | result |
|---|---|
| `action[:6]` vs ground truth, max ‖Δpos‖ | `0.000000e+00` m ✓ |
| `action[:6]` vs ground truth, max ‖Δrot‖ | `0.000000e+00` rad ✓ |
| `action[3:6]` linear std (orientation held constant) | `[0, 0, 0]` ✓ |
| `observation.state.target[3:6]` linear std            | `[0, 0, 0]` ✓ |
| `action[:6]` vs `observation.state.target[:6]`, max abs | `0` m, `0` rad ✓ |
| commanded square geometry (u, v ranges)               | `[±0.0250, ±0.0250]` m ✓ (exact) |
| commanded corner distance                             | `0.035355` m = `size·√2/2` ✓ |
| commanded closure gap                                 | `2.5` mm — one frame-step off by construction |

**But the physical motion is only 25 % of the commanded motion.**

| metric | commanded | measured |
|---|---:|---:|
| total path length (inc. approach + return) | **0.2500 m** | **0.0627 m** (**25.1 %**) |
| shape-only path length                     | 0.1975 m     | 0.0538 m (27.3 %) |
| u range (`±side/2` = `±25 mm`)             | **±25.0 mm** | **−7.0 / +6.7 mm** (**27.4 %**) |
| v range (`±side/2` = `±25 mm`)             | **±25.0 mm** | **−8.4 / +8.3 mm** (**33.3 %**) |
| RMS cartesian tracking error (all phases)  | —            | **19.60 mm** |
| max cartesian tracking error               | —            | **30.03 mm** |
| per-phase max err, `approach`              | —            | 21.13 mm |
| per-phase max err, `shape_open`            | —            | 30.03 mm |
| per-phase max err, `shape_closed`          | —            | 29.91 mm |
| per-phase max err, `return`                | —            | 17.90 mm |

The arm was commanded to trace a 5 cm square. It actually traced a ~1.4 cm
square-ish shape with ~2 cm RMS offset. Figure-of-merit: at a 5 cm / s
commanded tangential speed on a ~5 cm perimeter, an impedance controller
at healthy gains should finish within ~1 cm lag. 3 cm lag on every frame
is not controller softness — it is a systematic timing failure.

### 2.3 Root cause

The very first line of the recorder's log says it:

```
[crisp_gym setup_robot_env] WARNING: net.core.rmem_max is small —
Orbbec FPS may suffer. Fix once with:
  sudo sysctl -w net.core.rmem_max=2147483647
  sudo sysctl -w net.core.wmem_max=2147483647
```

The sysctl fix was **not applied**. Consequences:

1. Orbbec camera is capped below 20 Hz by kernel UDP buffer limits.
2. `env.get_obs()` inside `data_fn` blocks waiting for a fresh camera
   frame, occasionally for up to 0.4 s.
3. `data_fn` therefore runs at **~9 Hz average** instead of 20 Hz, with
   individual frames between 2.2 Hz and 20 Hz.
4. `env.robot.set_target(...)` is called at the same degraded rate. The
   env-side `crisp_py.Robot` publish timer keeps ticking at 20 Hz but
   re-publishes the same target during the gaps.
5. `cartesian_controller` receives a slow, jittery reference stream.
   Each target is held for 2–4 control ticks before the next arrives, so
   the controller never gets a clean 20 Hz command sequence to follow
   the 5 cm/s reference. The impedance dynamics filter the step
   responses and the net effect is a heavily attenuated realised path.
6. **LeRobot writes synthetic timestamps** (`frame_index / fps`), so the
   parquet's `timestamp` column looks like a perfect 20 Hz recording
   even though wall-clock timing was irregular. The degradation is
   invisible to anything that only inspects timestamps.

The analyzer's §4 verdict fires automatically on this condition:

> ❌  arm tracked only 25.1 % of the commanded path length. The
> recording is not usable for calibration. Root cause is almost
> certainly a slow `data_fn` (Orbbec blocking) — re-run after
> `sudo sysctl -w net.core.rmem_max=2147483647;`
> `sudo sysctl -w net.core.wmem_max=2147483647` and verify the
> recorder logs zero `Frame processing took too long` warnings.

### 2.4 Secondary findings

**Orientation representation artifact** — exactly as documented in
[`ridgeback_replay_orientation_bug.md`](./ridgeback_replay_orientation_bug.md):

- `action[3:6]` linear std = `[0, 0, 0]` ✓ (command side is clean)
- `observation.state.cartesian[3:6]` linear std = **2.93 rad on roll**
- `observation.state.cartesian[3:6]` circular std = **1.82e-02 rad** ✓
- 2 Euler-XYZ branch flips on roll, 0 on pitch/yaw

The measured arm orientation is physically constant (1.82e-02 rad
circular std = noise floor). The 2.93 rad linear std is `scipy`'s
`as_euler("xyz")` flipping between `+π - ε` and `−π + ε` on the same
physical rotation. This is a display / representation bug in the *measured*
column, not a physical wobble. Not a recorder bug. Not a blocker for this
calibration.

**Gripper echo race** — 7 non-transition frames where `action[6]` disagreed
with `gripper_target` by > 0.05. Transitions themselves (2 of them) were
clean. This is the `crisp_py` gripper self-subscription race already
documented in `17_calibration_report.py`'s §5. Not specific to this script,
not a recording bug, and doesn't affect replay because `action[6]` is the
authoritative commanded value.

### 2.5 Verdict on run 1

**The dataset is not usable for calibration.** The `action` column is
correct (matches ground truth exactly), so a policy trained on it would
learn the commanded intent, but the `observation.state.cartesian` column
(the *actual* arm pose) is a ~⅓-scale wobble of what was commanded. Any
verification that requires measured cartesian to match commanded
cartesian fails by construction.

Keep the dataset as evidence of the pre-fix state if you want a before /
after comparison. Otherwise delete it:

```bash
rm -rf ~/.cache/huggingface/lerobot/calib_geo_square_001
```

---

## 3. Run 2 — `calib_geo_square_002` (`--size 0.5`)

### 3.1 What was on screen

The workspace pre-flight **caught the bad trajectory before any motion**
and raised:

```
ValueError: Calibration waypoint 37 at position [1.00115356 0.16961628 0.86570393]
violates workspace limits: x=+1.001 > max_x=+1.000. Check the anchor pose,
--size, and --plane, or widen the limits in ur10e_ridgeback_env.yaml.
```

### 3.2 Geometric reasoning

Square of side 0.5 m in the `xy` plane centred at `P0 = (0.931, …)`
requires reaching `x = 0.931 + 0.5/2 = 1.181` m on the right edge.
Current env YAML limits (`ur10e_ridgeback_env.yaml:109-114`, updated
2026-04-15):

```yaml
min_z: 0.05
max_z: 1.0
min_x: -0.9
max_x: 1.0        # raised from 0.9 on 2026-04-15
min_y: -0.9
max_y: 0.9
```

`+x` slack at the standard home pose is `1.0 − 0.931 = 0.069 m` —
**6.9 cm**, nowhere near the 25 cm the trajectory asked for.

The check fired at waypoint **37**, which sits inside the *approach*
segment (pre-dwell = 10 frames, approach = 100 frames, shape trace
starts at frame 110). The approach linearly interpolates from `P0` to
`P0 + 0.25·u` in 100 steps; frame 27 of the approach (absolute frame 37)
is at fraction `28/100 = 0.28` along the way, giving
`x = 0.931 + 0.28·0.25 = 1.001` m — which matches the reported position
to the millimetre. The validator is doing exactly what it is supposed
to do.

### 3.3 Practical ceilings for `--size` at the current anchor

Under the current `xy` limits, max sizes before the workspace check fires:

| shape    | plane | max `--size`      | bound by          |
|----------|-------|-------------------|-------------------|
| circle   | xy    | **≈ 0.069 m**     | `+x` slack 6.9 cm |
| square   | xy    | **≈ 0.138 m**     | `+x` slack / `±s/2` |
| circle   | xz    | **≈ 0.134 m**     | `+z` slack 13.4 cm |
| square   | xz    | **≈ 0.268 m**     | `+z` slack / `±s/2` |
| circle   | yz    | **≈ 0.134 m**     | `+z` slack 13.4 cm |
| square   | yz    | **≈ 0.268 m**     | `+z` slack / `±s/2` |

All the above assume the standard home pose gives `P0 ≈ (0.93, 0.17, 0.87)`.
If you want bigger shapes, either (a) start the arm from a pose further
from the `+x` wall, (b) use the `xz` or `yz` plane, or (c) widen the env
YAML limits — but that is a robot-safety change, not a calibration change.

### 3.4 Side effects

Run 2 created an empty stub at
`~/.cache/huggingface/lerobot/calib_geo_square_002/meta/` before the
pre-flight fired (this is just the LeRobot dataset writer initialising).
No `data/` directory, no frames. Safe to delete:

```bash
rm -rf ~/.cache/huggingface/lerobot/calib_geo_square_002
```

The next run of script 18 with that same `--repo-id` will offer to do
this for you automatically.

### 3.5 Teardown noise (cosmetic only)

After the `ValueError` propagated up, the `finally: env.close(); rclpy.shutdown()`
block fired while `crisp_py.Robot`, `crisp_py.Gripper`, and
`crisp_py.Camera` spin threads were still running. They tried to submit
new work to the already-shutting-down Python thread-pool executor and
printed a few `RuntimeError: cannot schedule new futures after interpreter
shutdown` lines. This is a teardown-ordering bug in `crisp_py`'s spin-thread
lifecycle, not a bug in the calibration script or the trajectory. The
actual error (the `ValueError` that stopped the run) is printed **before**
this noise and is what matters.

### 3.6 Verdict on run 2

**Pre-flight did its job.** The trajectory was infeasible for the
current workspace and the script refused to move. No robot motion, no
dataset, no recovery needed. Change `--size` (or `--plane`) and re-run.

---

## 4. What needs to happen before the next run

In order of importance:

### 4.1 Fix the Orbbec socket buffers — blocking issue

Run **once** per machine (the setting persists until reboot):

```bash
sudo sysctl -w net.core.rmem_max=2147483647
sudo sysctl -w net.core.wmem_max=2147483647
```

To make it permanent across reboots, add to `/etc/sysctl.d/99-ros.conf`:

```
net.core.rmem_max = 2147483647
net.core.wmem_max = 2147483647
```

**Acceptance test** after the fix: re-run `calib_geo_square_001` and
confirm the recorder logs **zero** `Frame processing took too long`
warnings. Then run the analyzer and expect:

- `measured path length ≥ 90 % of commanded`
- `u, v tracking ratios ≥ 90 %`
- `RMS cartesian error ≤ 2 mm` at 5 cm/s
- `max cartesian error ≤ 5 mm`

Anything else is still broken.

### 4.2 Clean up the bad datasets

```bash
rm -rf ~/.cache/huggingface/lerobot/calib_geo_square_001
rm -rf ~/.cache/huggingface/lerobot/calib_geo_square_002
```

Or keep `_001` as a pre-fix baseline for side-by-side comparison —
your call.

### 4.3 Pick reasonable `--size` values for the current anchor

For `xy` plane at the standard home: `--size 0.05` to `0.10` for square,
`--size 0.04` to `0.06` for circle. For larger shapes, use `--plane xz`
where `+z` slack gives ~13 cm of vertical room.

Suggested first post-fix runs:

```bash
# 5 cm square in xy — baseline smoke test.
pixi run -e jazzy-lerobot python examples/18_calibration_record_geo.py \
    --repo-id calib_geo_square_005cm --shape square --size 0.05

# 5 cm circle in xy — test continuous curved path.
pixi run -e jazzy-lerobot python examples/18_calibration_record_geo.py \
    --repo-id calib_geo_circle_005cm --shape circle --size 0.05

# 10 cm square in xz — test the second axis pair with a larger footprint.
pixi run -e jazzy-lerobot python examples/18_calibration_record_geo.py \
    --repo-id calib_geo_square_10cm_xz --shape square --size 0.10 --plane xz
```

### 4.4 Run the analyzer on every dataset

```bash
pixi run -e jazzy-lerobot python examples/19_calibration_report_geo.py \
    --repo-id <name> --shape <circle|square> --size <m> [--plane ...] \
    --out analysis --plot
```

`--plot` writes four PNGs (`position_traces.png`,
`orientation_traces.png`, `tracking_error.png`, `shape_overlay.png`)
into the `analysis/` directory. `shape_overlay.png` is the top-down view
of ground-truth vs commanded-action vs measured-cartesian in the `(u, v)`
plane — the fastest visual sanity check for shape fidelity.

---

## 5. What the calibration DID successfully prove

Even though the physical motion was broken, the **recording-layer
invariants that script 18 is designed to check all passed**:

- `action[:6]` is bit-exact equal to the ground-truth trajectory
  (max diff = 0).
- Commanded orientation is exactly constant (linear std = 0).
- `observation.state.target[:6]` mirrors `action[:6]` exactly
  (max diff = 0) — `set_target` → `env.robot.target_pose` → recorded
  target round-trip is clean.
- The commanded square geometry is exact: u, v both span `±0.0250 m`,
  max in-plane distance is `0.035355 m = 0.05·√2/2`, out-of-plane
  component is exactly 0.

These invariants would have caught **any** bug in script 18's trajectory
generation, topic ownership, `Pose.to_array` path, or gripper schedule.
None of those exist. The recorder is doing its half of the job
correctly. The next run, with the socket fix, should flip §4 of the
analyzer from `❌` to `✅` with everything else unchanged.

---

## 6. Timeline

| wall-clock | event |
|---|---|
| 22:10:19 | script 18 launched for run 1, env setup, warning about Orbbec buffers printed |
| 22:10:27 | arm homed, anchor captured, trajectory built |
| 22:10:28 | auto-start recording |
| 22:10:28 – 22:10:41 | 13 s wall-clock for a nominal 6 s / 120-frame episode (→ ~9 Hz avg) |
| 22:10:41 | dataset saved, arm homed |
| 22:11:02 | script 18 launched for run 2 (`--size 0.5`) |
| 22:11:12 | anchor captured |
| 22:11:12 | trajectory built (1020 frames), `validate_against_workspace` raised `ValueError` |
| 22:11:12 | cosmetic thread-pool shutdown errors from `crisp_py` spin threads |

Total elapsed: ~1 min. No robot damage, no operator intervention
required. The calibration workflow itself is working — it caught a
real problem (Orbbec socket buffers) and a real geometric mistake
(50 cm square at the `+x` wall), and in both cases the next step is
obvious.

---

## 7. Follow-up: report generator (`19_calibration_report_geo.py`)

Every number in §2.2 of this document came from the analyzer. The
analyzer is the single source of truth for calibration assessment and
should be run against every future calibration dataset. Its verdicts
map directly to action items:

| verdict | action |
|---|---|
| `❌ action ≠ ground truth`         | Spec args don't match recording — fix CLI |
| `❌ action orientation wobbles`    | Representation bug in `Pose.to_array` — file issue |
| `❌ action ≠ env.robot.target_pose`| `set_target` mutation — file issue |
| `❌ measured path < 70 % of commanded` | Orbbec socket buffers, frame-rate warnings |
| `⚠  measured path < 90 %`          | Impedance gains soft, velocity too high, or early-stage slowdowns |
| `⚠  cartesian linear std huge, circular std small` | Representation artifact (known), not a bug |
| `⚠  gripper non-transition flicker` | `crisp_py` echo race (known), doesn't affect replay |
| `✅` all the way                    | Dataset is ready to use for calibration |
