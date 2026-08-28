# B-spline deployment + DemoSpeedup train-code compatibility

Two separate questions, answered from the code as it stands on
`ur10_clearpath@speedup-replay` (commit `4749a4a`) and
`lerobot_uncertainty@hk-speed` (commit `95367d78` + uncommitted speedup edits).

**Nothing has been changed.** Part 1 is an audit; Part 2 is a change plan
awaiting a go/no-go. Open questions are collected at the end.

---

## Part 1 — Is the xVLA DemoSpeedup train code compatible with our real-data runs?

### 1.1 What we actually ran, and where

There are **two independent implementations of DemoSpeedup** in play. They were
never used together, and the real-robot run used the one *without* the xVLA
code path.

| | crisp_gym port (`Yunfei/crisp_gym/demospeedup/`) | lerobot_uncertainty fork (`--use_speedup`) |
|---|---|---|
| lerobot | `lerobot-041` env, stock **0.4.1** (site-packages) | `lerobot` env, fork **0.4.3** (`/home/batur/lerobot_uncertainty/src`) |
| where retiming happens | **offline**, dropping frames from the dataset (`convert_lerobot_to_speedup.py`) | **online**, per batch (`processor/speedup_processor.py`) |
| training entry | stock `lerobot-train` on the retimed dataset | `lerobot-train --use_speedup=true ...` |
| labels artifact | `<dataset>/meta/demospeedup/labels.parquet` (+ `demospeedup.json`) | `<label_dir>/speedup_labels/episode_<i>.npy` |
| entropy sampler | vendored `demospeedup_core/sampling.py` (wraps a projection module to substitute `randn`) | `ACTPolicy.sample_action_chunks` → `ACTTemporalEnsembler.forward_with_latent` (fork-only method) |
| segmentation | `threshold` (default) or `hdbscan` | `hdbscan_with_custom_merge` only |

Provenance, from the wandb metadata of the two real runs:

- `outputs/train/demospeedup_act` — `lerobot-041/bin/lerobot-train --policy.type=act
  --policy.chunk_size=50 --policy.n_action_steps=50 --dataset.repo_id=merged_speedup_20260528`.
  Its `train_config.json` contains **no `use_speedup` key at all**: the speedup was
  baked into the dataset before training.
- `outputs/train/bspline_act_merged_20260528` — same env, resumed from `train_config.json`.

So the short answer to "is the xVLA train code compatible with our real run": **it
was not used for it, and one flag in it will crash on ACT.** Details below.

### 1.2 What *does* line up (verified, not assumed)

- **Checkpoint loading across versions.** The lerobot-041-trained proxy
  (`act_cart7_v2_angleaxis_nogrip_chunk100_ft_20260528/checkpoints/030000/pretrained_model`)
  loads cleanly under the fork's 0.4.3 and exposes both entropy hooks:
  `sample_action_chunks: True`, `forward_with_latent: True`, `chunk_size=100`,
  `use_vae=True`, output `action: (7,)`, preprocessor steps
  `[Rename, AddBatchDimension, Device, Normalizer]`.
- **Dataset loading.** `merged_act_finetune_20260528` (v3.0, 70 eps / 21560 frames /
  20 fps) opens under the fork with `video_backend=pyav`. The batch carries
  `frame_index` as the **episode-local** index (`ds[5] → frame_index=5,
  episode_index=0`), which is exactly what `SpeedupDownsampleProcessor.__call__`
  indexes labels with. No adapter needed there.
- **Action space.** Real action is `[x, y, z, rx, ry, rz, gripper]` — **absolute**
  pose in axis-angle, not deltas. This is the precondition the xVLA script calls
  out (`run_demospeedup_xvla_ee6d.sh` header: retiming a *delta* action space
  silently discards the dropped motion). Our data is safe.
- **The stride walk is the same algorithm.** `demospeedup_core/retiming.py::
  process_action_label_upstream(start=0)` and
  `lerobot/utils/entropy.py::downsample_with_labels` are the same loop with the
  same `i + low_v < horizon` boundary and the same `torch.all(labels[i:i+high_v] == 1)`
  guard. The crisp_gym side additionally seeds `[0]` and appends a `keep_last`
  tail at stride `high_v`; the fork side seeds `[0]` and just stops. Labels
  produced by either would be interpreted identically up to that tail.
- **Chunk convention matches.** `speedup_halve_chunk` would take our ACT's
  `chunk_size=100 → 50`, which is exactly the 50 that `train_speedup_act.py`
  used by hand for the `demospeedup_act` run.
- **Labels already exist** for the real dataset:
  `meta/demospeedup/labels.parquet`, 70/70 episodes, `segmenter=threshold`,
  `low_v=2 high_v=4 fast_prefix=0 chunk_size=100 seed=2`, projected speedup
  **2.09×** (21560 → 10302 frames). They do not have to be recomputed.

### 1.3 Why the chunk gets halved (and whether it has to)

`chunk_size` counts **waypoints**, and after retiming one waypoint is worth 2-4
source frames. So the number is not a time budget any more, it is a motion
budget, and two things follow:

*Geometry.* A 100-waypoint chunk on retimed data spans ~210 source frames of
motion instead of 100. Halving to 50 puts it back at `50 x 2.09 = 105` source
frames, i.e. the same physical lookahead the baseline had. That is upstream's
"maintain geometrical consistency", and it is why `train_speedup_act.py` used 50.

*Executability, which matters more.* In the online processor the walk over a
100-frame window yields **~48 valid waypoints** and pads the rest. If the model
still predicts 100, roughly 52 of its outputs have no supervision at all (ACT
masks them via `action_is_pad`) and are not executable — at deploy you would
have to set `n_action_steps ~= 48` by hand or publish garbage. Halving makes the
output length match the number of real waypoints.

So halving is neither arbitrary nor sacred. The principled number is
`ceil(horizon / mean_stride) = ceil(100 / 2.09) = 48`, and 50 is that rounded up.
The `//2` hard-code is only correct because `low_v=2` sets the floor: the walk
emits at most `horizon / low_v` waypoints (verified: **50** for an all-precision
100-frame window, **25** for an all-fast one). Change `low_v` to 3 and halving
over-provisions; change it to 1 and halving silently truncates real waypoints.
A `speedup_chunk_divisor` (or deriving it from the measured per-dataset speedup)
would be more honest than `//2`.

For ACT specifically, training does not *need* the halving — the loss is masked,
so the padded tail costs nothing but decoder queries. It is the deploy side that
needs the output length to mean something.

#### The one hard blocker: `--speedup_halve_chunk=true` crashes ACT

`lerobot_train.py:218-230` halves `chunk_size` *after* `make_dataset`, so the
batch still carries the full-length action sequence, and `downsample_with_labels`
pads its output back to the **original** horizon
(`entropy.py:339 new_actions = torch.zeros_like(actions)`). xVLA survives only
because `modeling_xvla.py:569 _prepare_action_targets` calls
`pad_tensor_along_dim(actions, chunk_size, dim=1)`, and that helper **truncates**
when the input is longer (`modeling_xvla.py:1138-1141`). ACT has no such step —
`modeling_act.py:192` does `F.l1_loss(batch[ACTION], actions_hat)` directly:

```
batch[ACTION] (B,100,7)  vs  actions_hat (B,50,7)
RuntimeError: The size of tensor a (100) must match the size of tensor b (50)
              at non-singleton dimension 1
```

(verified by running the exact call). This is why the flag has only ever been
exercised on xVLA — `logs/demospeedup_xvla_ee6d_train.log:335` shows
`Speedup: halved chunk_size to 15` and the run proceeds to 17k steps.

The fix is to truncate the downsampled chunk and `action_is_pad` to the halved
horizon inside `downsample_with_labels`. At `low_v=2` that discards nothing (the
walk cannot emit more than 50 waypoints from a 100-frame window), so it is
lossless at our settings and correct at every other setting too.

### 1.4 Segmentation: threshold vs HDBSCAN on our data

Agreed on standardising on HDBSCAN, and it is cheap to adopt: `labels.parquet`
already stores the raw `entropy` and `entropy_z` for all 21560 frames, so
re-segmenting is a CPU pass over stored data — **no policy re-run, no GPU**.
Measured just now on `merged_act_finetune_20260528`, all 70 episodes:

| | `threshold` (what `demospeedup_act` trained on) | `hdbscan` |
|---|---|---|
| non-precision frames | 12.0% | **27.5%** |
| overall speedup | 2.09x | 2.17x |
| retimed frames | 10302 | 9927 |
| worst single step | 146.5 mm | **166.7 mm** |
| frames whose label flips | — | **14.1%** |

HDBSCAN marks 2.3x more of each episode as fast, and buys **+4% throughput for
+14% on the worst commanded step** (146.5 -> 166.7 mm per 20 Hz period = 2.93 ->
3.33 m/s). That is a real change in what the arm is asked to do, so it is worth
deciding deliberately rather than inheriting.

Two caveats on "use HDBSCAN everywhere":

- **The env forces the current default.** `lerobot-041` has neither `sklearn` nor
  `hdbscan` installed (checked); the `lerobot` env has `sklearn 1.7.2`. That is
  the only reason `threshold` is the default — labelling has to move to the
  `lerobot` env for HDBSCAN to be available at all.
- **"HDBSCAN" is not one algorithm here.** crisp_gym's `segment_hdbscan` is plain
  `sklearn.cluster.HDBSCAN` over `(z(frame_index), z(entropy))` — the Aloha
  upstream. The fork's `hdbscan_with_custom_merge` (`utils/entropy.py:142`) adds
  isolation-forest outlier removal and splits any cluster larger than 25 — the
  RoboBase upstream. Standardising means picking one of *those* two as well.

Change needed either way: `analyze_labels.py --resegment hdbscan` only **prints**
the flip rate, it does not write. Adopting HDBSCAN labels needs either a
`--write` flag on it (cheap, reuses the stored entropy) or a re-run of
`label_entropy.py --segmenter hdbscan` in the `lerobot` env (~25 GPU-min, and it
would also re-draw the entropy, so the comparison above would no longer be
apples-to-apples).

### 1.5 Why the offline route trains on fewer samples

One training sample is one chunk *start*, and one chunk start is one frame of the
dataset the trainer loads:

- **offline** — the retimed dataset physically contains only the kept frames:
  `merged_speedup_20260528` has **10302** frames (vs 21560 in the source). So
  10302 possible chunk starts.
- **online** — the dataset still has all **21560** frames; every one is a chunk
  start, and only its *target chunk* is retimed.

The dropped frames' observations are not wrong or unusable — they are just never
used as a starting point. The practical difference is phase coverage: online sees
chunk starts at every offset of the retiming lattice, offline only at lattice
points, and at deploy the policy is queried at whatever phase the control loop
lands on. Whether that matters at 20 Hz, where adjacent frames are near
duplicates, is empirical and untested here. Note that raw data quantity was not
the binding constraint on the run we have: `demospeedup_act` did 30000 steps at
batch 32 over 10302 frames = **~93 epochs**.

### 1.6 Label file naming — narrower than I made it sound

This only exists if we run the fork's processor: `SpeedupDownsampleProcessor`
globs `episode_*.npy` and looks them up by the batch's `episode_index`, while
`lerobot_label.py` names them by *position* in the loaded dataset. The two agree
whenever the labelled set is `0..N-1` contiguous, which is the normal case. On
the offline route it is irrelevant. Treat it as a footnote on subset labelling,
not a gap.

### 1.7 Recommendation

If we keep the offline route (what produced `demospeedup_act`): the only change
worth making is the segmenter — add a write path for `--resegment hdbscan`,
re-convert, retrain. Nothing about the fork is involved.

If we also want the fork's `--use_speedup` on real data: additionally patch
`downsample_with_labels` to truncate to the halved horizon (section 1.2 (a)),
export the labels to per-episode `.npy`, and run under the `lerobot` env with
`--speedup_pad_mode=zero`.

---

## Part 2 — Deploying B-spline through `19_deploy_policy.py`

### 2.1 What we are deploying

`outputs/train/bspline_act_merged_20260528/checkpoints/030000/pretrained_model`:
an ACT with `chunk_size=1`, `n_action_steps=1`, `n_obs_steps=1`, inputs
`observation.images.camera`, `observation.images.d405`, `observation.state (6)`,
output `action: (286,)`.

That 286-vector is **not** a trajectory. Per
`/home/batur/Coding/data/merged_bspline_20260528/meta/bspline.json` it is a
`26 × 11` parameter matrix — `chunk_size=20`, `degree=3`, `relative_knots=false`,
column 0 the knot vector in *source frames relative to now*, columns 1: the
control points for `[x, y, z, rot6d(6), gripper]`. Decoding it gives ~16
waypoints spanning a **median 45 source frames ≈ 2.25 s**.

The good news: `bspline/decode_rollout.py::decode()` already returns exactly what
deployment needs — `DecodedChunk(actions (n,7) = [x,y,z,axis-angle,gripper],
times (n,) seconds from now, span_frames, padded)` — and it already handles the
two upstream quirks (drops the zero-valued last sample of a tail-padded chunk;
clamps negative `t_min`). `convert_actions_10d_to_7d` emits **axis-angle**, which
matches `env.action_to_rotation` when the env yaml has
`orientation_representation: "angle_axis"`.

### 2.2 What breaks today, unchanged

Run as-is, `19_deploy_policy.py` reaches `main()`'s chunk handling with
`chunk.shape == (1, 286)`:

- `19_deploy_policy.py:2337` — the guard is `chunk.shape[1] < 7`, so **286 passes**.
  Nothing raises.
- `19_deploy_policy.py:2384` — `_pre_compute_chunk_arrays` reads `actions[:, :3]`
  as xyz, `actions[k, 3:6]` as a rotation and `actions[k, 6]` as the gripper.
  Those are **knot values and control-point coordinates**, not a pose. The arm
  would be commanded to a meaningless target at full rate.

This is the silent-failure case, so the branch must be explicit and must refuse
to fall through.

### 2.3 The timing question (the one real design decision)

A B-spline chunk carries its own clock. `times[i]` says when waypoint *i* is due,
in seconds, at the **demonstration's** speed. The deploy pipeline instead builds
its schedule from a per-frame speed factor:

```
19_deploy_policy.py:2377   s_raw  = _build_chunk_speed_schedule(chunk, args)
19_deploy_policy.py:2380   cycles, dt_eff, s_eff = build_speed_queue_arrays(s_raw, dt_base, K, retime=True)
17_replay_dataset.py:1145  dt_raw = dt_base / s_raw ;  cycles = ceil(dt_raw / 0.002) ;  s_eff = dt_base / dt_eff
```

`s_eff` is then what `ReplayScaler.step_to()` uses to scale the controller:
`kp = kp_base * s_eff**2`, `kd = kd_base * s_eff`, gripper speed linear in `s_eff`.

Two modes are worth having, and they are not the same experiment:

- **`spline` (faithful / safe first run)** — execute waypoint *i* at
  `times[i]`, i.e. `dt[i] = df[i] * dt_base` where `df[i]` is the waypoint's span
  in source frames. This replays at the demonstrated speed. It is the right
  first deployment: it isolates "does the decoded spline track at all" from any
  speedup claim.
- **`uniform` (the paper's speedup)** — execute every waypoint at `dt_base`
  regardless of `df[i]`. Sparse-knot stretches (easy motion) then run `df[i]`×
  faster and dense-knot stretches slow down. This *is* the B-spline policy's
  acceleration mechanism; at our fit (`0.42 knots/frame`) it averages ~2.8× on
  the easy segments.

**Important: `decode(..., speedup=...)` is the wrong knob for either mode.**
It compresses `times` only. The deploy loop would still see `s_eff = 1.0` and
would never raise kp/kd, so the OSC controller would lag exactly where the
motion got faster. Decode at `speedup=1.0` and let the speed factor carry it.

**And `build_speed_queue_arrays` cannot express what we need as written.** It
couples timing and gains through the single `s_raw`: `retime=True` forces
`dt_eff = dt_base / s_raw`, `retime=False` forces `dt_eff = dt_base`. For
`uniform` mode we want *uniform `dt_base` timing* **and** *gains scaled by
`df[i]`* — which `retime=False` happens to give (`s_eff = s_raw`). For `spline`
mode we want *non-uniform `df[i]·dt_base` timing* **and** `s_eff = 1` — which
neither branch gives.

The clean fix is a small bspline-local helper that decouples them and derives
the gain factor from the physically meaningful quantity, *source frames of
motion per wall-clock base period*:

```
cycles[i] = max(1, ceil(dt_target[i] / CONTROL_DT))
dt_eff[i] = cycles[i] * CONTROL_DT
s_eff[i]  = df[i] * dt_base / dt_eff[i]
```

This reduces to the existing semantics when `df ≡ 1`, and it fixes the same
under-scaling that `--stride` has today (striding by 2 makes each step twice as
long but leaves `s_eff` at 1, so kp/kd never compensate — see the `--stride`
docstring at `19_deploy_policy.py:2350`).

### 2.4 Proposed changes, file by file

All additive and behind a flag; the ACT/diffusion path stays byte-identical when
`--bspline` is off.

**`examples/19_deploy_policy.py`**

1. *(~line 88, next to `_load_replay17`)* — `sys.path.insert(0, <crisp_gym>/bspline)`
   and import `decode` from `decode_rollout`, guarded so the import cost is only
   paid when `--bspline` is passed. Mirrors how `eval_bspline_checkpoint.py:28-30`
   does it.
2. *(argparse, new group)*
   - `--bspline` — enable the branch.
   - `--bspline-meta PATH` — path to `bspline.json`. Default: auto-discover from
     the checkpoint's `train_config.json → dataset.root` + `/meta/bspline.json`
     (verified present: `dataset.root = /home/batur/Coding/data/merged_bspline_20260528`).
   - `--bspline-num-actions N` (default 16) — waypoints decoded per chunk.
   - `--bspline-time-mode {spline,uniform}` (default `spline`).
   - `--bspline-max-waypoints N` (default 0 = all) — truncate the decoded chunk to
     force earlier replanning; see 2.5.
3. *(after `chunk = chunk_source.request(...)`, ~line 2334, i.e. after the
   `--record-trace` capture and before the shape guard)* — if `--bspline`:
   reshape `(1, 286) → (26, 11)`, `decode(...)`, replace `chunk` with
   `decoded.actions` `(n, 7)` and carry `decoded.times` forward. Doing it here
   keeps it source-agnostic (async and `--sync` both land in the same place).
4. *(shape guard, line 2337)* — when `--bspline` is on, assert
   `chunk.shape[1] == n_action_channels * ...` **before** decode and fail loudly
   on mismatch, so a non-bspline checkpoint passed with `--bspline` (or the
   reverse) cannot reach `_pre_compute_chunk_arrays`.
5. *(speed schedule, line 2377-2382)* — branch to the helper from 2.3 instead of
   `_build_chunk_speed_schedule` + `build_speed_queue_arrays`. `--max-speed` /
   `--lookahead` curvature braking can still compose on top by multiplying into
   `df` — but I would leave that off for the first run.
6. *(line 1890)* — the `args.lookahead >= n_act` warning reads `n_act` from the
   chunk source, which is **1** for this checkpoint. Compare against the decoded
   waypoint count instead.
7. *(line 2427)* — `frame_idx=(chunk_count - 1) * K + i` assumes constant `K`;
   the decoded count can differ by one between padded and interior chunks. Use a
   running counter.
8. *(`--record-trace`, ~line 2265)* — record the raw `(26, 11)` params **and** the
   decoded waypoints; the raw matrix alone is not inspectable after the fact.

**No changes needed** in `crisp_gym/policy/async_lerobot_policy.py` — it is
action-dim agnostic (`inference_worker` reads `n_action_steps` from the config
and returns `chunk.squeeze(0).numpy()`), and `_SyncLeRobotChunkSource` likewise.
**No changes needed** in `_pre_compute_chunk_arrays` — after decode the chunk is
the standard 7-dim layout, and the gripper channel gets the existing 0.5
binarisation, which is the right treatment for a spline-smoothed gripper.
**No env/yaml changes** — the bspline checkpoint's `input_features` are identical
to the ACT baselines'.

### 2.5 Two risks worth naming before the first run

- **Replan cadence.** With `n_action_steps=1` the policy emits one parameter
  matrix per observation, and the producer waits for the queue to drain to
  `--overlap-threshold` before requesting the next
  (`19_deploy_policy.py:2461`). One chunk covers ~2.25 s, so the arm would run
  **open-loop for over two seconds** — far longer than the ACT baseline's
  chunk-100 (5 s at 20 Hz, but consumed at `n_action_steps=50`). This is what
  `--bspline-max-waypoints` is for: execute the first N of 16 and replan.
  Suggest starting at 4–6.
- **Knot monotonicity.** `decode()` repairs a non-monotone predicted knot column
  with `safer_knots`, silently. `eval_bspline_checkpoint.py` already reports how
  often that fires for this checkpoint — worth reading that number *before*
  putting it on the arm, and worth logging per-chunk during deploy.

### 2.6 Suggested order of work

1. Offline: run `eval_bspline_checkpoint.py` on `checkpoints/030000` and record
   the mm/deg error and the non-monotone rate. No robot.
2. Implement 2.4 items 1-4 (decode plumbing) and dry-run with `--dry-run` +
   `--fake-mode`, checking the logged waypoints against the offline decode.
3. Implement item 5 (timing helper), still `--dry-run`, and verify `dt_eff` /
   `s_eff` traces in `chunks.csv` match the intended mode.
4. First robot run: `--bspline-time-mode spline --bspline-max-waypoints 6`,
   `--max-speed 1.0 --min-speed 1.0`, `--scale-kp` off. Source speed, no gain
   bump, nothing to blame but the decode.
5. Then `uniform` with `--scale-kp` on.

---

## Open questions

1. **Segmenter** — adopt HDBSCAN on the real data? It costs +14% on the worst
   commanded step (146.5 -> 166.7 mm) for +4% throughput. If yes: re-segment the
   stored entropy (free) or re-label from scratch (~25 GPU-min)? And which
   HDBSCAN variant — crisp_gym's plain sklearn one, or the fork's
   outlier-removal + cluster-splitting one?
2. **Chunk divisor** — leave `//2`, or replace it with an explicit
   `speedup_chunk_divisor` / a value derived from the measured speedup? Only
   matters if we ever move off `low_v=2`.
3. **Part 1 scope** — do we want the fork's `--use_speedup` path working on real
   data at all, or is the offline route the one we keep? If the latter, the ACT
   truncation fix and the label export are both unnecessary.
4. **Part 2 timing** — is `spline` (source speed) the right default for the first
   robot run, or do you want `uniform` (the actual speedup) straight away?
5. **Part 2 gains** — the `s_eff` redefinition in 2.3 would also fix the
   under-scaling `--stride > 1` has today. Keep it scoped to the bspline branch,
   or apply it to the stride path too (which changes existing replay behaviour)?
6. **Where the bspline code lives** — inline in `examples/19_deploy_policy.py`
   per the note at line 84, or a small `bspline/deploy_bridge.py` that `19_*`
   imports? The decode + timing helper is ~80 lines and is testable in isolation.
