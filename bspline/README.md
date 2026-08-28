# B-spline action representation for crisp_gym datasets

A port of the action representation from **B-spline Policy: Accelerating
Manipulation Policies via B-spline Action Representations**
([paper](https://arxiv.org/abs/2607.09648),
[code](https://github.com/B-spline-policy/bspline-policy)), wired up to the
LeRobot v3.0 datasets this repo records.

Upstream ships a `diffusion_policy` fork pinned to zarr 2.12 / diffusers 0.11 /
robomimic 0.2 / numpy 1.26, which cannot coexist with `lerobot-041`. Rather
than build that environment, only the **action representation** is vendored
here (`bspline_core/`) and training goes through LeRobot as usual.

## What the representation is

One action chunk is a dense parameter matrix

```
(chunk_size + 2 * degree, 1 + action_dim)
```

* **column 0** — the B-spline knot vector, in *source-frame units, relative to
  the current frame*.
* **columns 1:** — the B-spline control points for the action dimensions
  (`[x, y, z, rot6d(6), gripper]`, i.e. 10 dims for our cart7 data).

Per episode, `scipy.interpolate.generate_knots` grows a knot vector until a
least-squares fit of the whole action trajectory is within `max_error`
(max-abs over all frames and dimensions). That global spline is then sliced
into fixed-size windows: `chunk_size + 2 * degree` consecutive knots together
with the **same-indexed** control points. By the local-support property of
B-splines, such a window reproduces the global spline exactly on
`[t[s+degree], t[s+M-degree-1]]`.

Every frame is assigned the last chunk whose valid domain has not yet started,
with the knot column shifted by `-frame_index` so it reads as "time from now".

Decoding evaluates the spline at `num_actions` points spaced uniformly over
`[knots[degree], knots[-(degree+1)]]`. **That interval is a variable number of
source frames** — knot spacing adapts to how fast the demonstration is moving.
That is the whole point: one prediction can cover a long, easy stretch of
trajectory, so the same policy replays it faster.

## Layout

```
bspline/
  bspline_core/               vendored, dependency-light (numpy + scipy only)
    knots.py                  relative-knot encode/decode, safer_knots
    bspline_action.py         fit, chunk, decode
    chunk_sampler.py          per-frame chunk assignment
    rotation.py               axis-angle <-> rot6d without pytorch3d
  lerobot_bridge.py           LeRobot v3.0 -> numpy actions -> chunks
  analyze_fit.py              accuracy/compression sweep (run this first)
  convert_lerobot_to_bspline.py   dataset conversion CLI
  train_bspline_act.py        training entry point
  decode_rollout.py           parameter matrix -> executable waypoints
  eval_bspline_checkpoint.py  decode a checkpoint's predictions, score in mm/deg
  build_walkthrough.py        rebuilds walkthrough.html from one real episode
  walkthrough_template.html   its markup/JS; data is spliced in at build time
  walkthrough.html            generated -- interactive explainer of the method
  worked_example.py           one episode through every stage, with real numbers
  worked_example_template.html / build_worked_example_page.py
  worked_example.html         generated -- the numbers as a readable page
  verify_math.py              24 machine-checked mathematical claims
  build_verification_page.py  renders those results into a report page
  math_verification.html      generated -- the derivations plus the evidence
  policy_io.py                the ONLY correct way to load a checkpoint
  tests/                      the correctness suite
```

## Verifying the maths

`verify_math.py` states each mathematical claim the pipeline depends on and
checks it against an *independent* reference -- a Cox-de Boor recursion written
from the definition, scipy, the walkthrough page's own JavaScript decoder run
under node, and real recorded episodes. It exits non-zero if any check fails.

```bash
conda run -n lerobot-041 python bspline/verify_math.py
conda run -n lerobot-041 python bspline/build_verification_page.py
```

Covered: the B-spline axioms on our own knot vectors; the windowing theorem
(a chunk equals the global spline on its interior); knot translation invariance;
the relative-knot bijection; agreement between four evaluators; the padded-tail
degeneracy (partition of unity fails at `t_max`); the rotation conventions
against scipy; the fit tolerance; the frame->chunk assignment rule against a
closed form; and the normalisation identity.

## Walkthrough

`walkthrough.html` works the whole method through episode 58 of
`merged_act_finetune_20260528`: the recording, the fit and its adaptive knot
placement, one chunk as a window of the global spline, an interactive scrubber
over the episode, the parameter matrix, and the decode error. Every number on
the page is computed at build time -- nothing is illustrative.

```bash
conda run -n lerobot-041 python bspline/build_walkthrough.py --episode 58
```

## Workflow

```bash
cd ur10_clearpath/Yunfei/crisp_gym

# 1. how well does *our* data compress?
conda run -n lerobot-041 python bspline/analyze_fit.py --episodes 20 --coverage

# 2. convert
conda run -n lerobot-041 python bspline/convert_lerobot_to_bspline.py \
    --src /home/batur/Coding/data/merged_act_finetune_20260528 \
    --dst /home/batur/Coding/data/merged_bspline_20260528 \
    --chunk-size 10 --max-error 0.01

# 3. verify the conversion round-trips
cd bspline && conda run -n lerobot-041 python -m pytest tests/ -q && cd ..

# 4. train (run from crisp_gym/, not bspline/)
conda run -n lerobot-041 python bspline/train_bspline_act.py --wandb

# 5. score a checkpoint in physical units
conda run -n lerobot-041 python bspline/eval_bspline_checkpoint.py \
    --ckpt outputs/train/bspline_act_merged_20260528/checkpoints/last/pretrained_model
```

## Chosen settings, and the measurements behind them

`analyze_fit.py` over the 70-episode `merged_act_finetune_20260528` dataset
(20 Hz, positions in metres):

| `max_error` | knots/frame | per-dim max error | chunk-20 span |
|---|---|---|---|
| 0.005 | 0.66 | pos ≤1.3 mm, rot ≤0.005 | ~30 fr (1.5 s) |
| **0.01** | **0.48** | **pos ≤4.5 mm, rot ≤0.010** | **~45 fr (2.25 s)** |
| 0.02  | 0.37 | pos ≤7.4 mm, rot ≤0.019 | ~54 fr (2.7 s) |

Upstream's `max_error = 0.002` gives 0.90 knots/frame on this data — almost no
compression, so the representation would buy nothing. `0.01` was chosen as the
point where positions stay sub-5 mm while a chunk still reaches ~2 s ahead.

`chunk_size = 20` (not upstream's 10) because our data is 20 Hz where theirs is
10 Hz; 20 knot intervals give the same ~2 s horizon in wall-clock terms.

Verified end to end on the converted dataset
(`tests/test_converted_dataset.py`): decoding a stored action and comparing to
the source recording gives **≤5.4 mm position, ≤0.78° rotation**, with a median
chunk span of **45 frames (2.25 s) from 286 predicted numbers** — versus the
ACT chunk-100 baseline's 100 frames from 700.

## Deviations from upstream, and why

| Upstream | Here | Why |
|---|---|---|
| robomimic HDF5 + zarr ReplayBuffer | LeRobot v3.0 parquet | our recording pipeline already emits v3.0; no re-encode needed |
| `pytorch3d` rotation transforms | numpy/scipy in `rotation.py` | avoids a heavy pinned dependency; conventions verified in `tests/test_rotation.py` |
| Diffusion UNet predicts `(16, 11)` | ACT predicts a flat `176` vector | the temporal axis lives *inside* the action, so LeRobot's action-chunk stacking has nothing to stack; `chunk_size=1` is the faithful mapping |
| `max_first_k = n_obs_steps` (2) | `max_first_k = 1` | keeps every source frame labelled so episode boundaries — and therefore the existing video files — stay valid |
| `max_error = 0.002` | see `analyze_fit.py` | at 0.002 our 20 Hz data barely compresses (0.90 knots/frame); the representation only pays off at a looser tolerance |

## The normalisation trap (cost me a wrong diagnosis)

LeRobot 0.4.x keeps normalisation **outside** the policy, in pre/post-processor
pipelines saved beside the weights. `ACTPolicy.from_pretrained()` restores only
the network, so this looks right and is silently wrong:

```python
policy = ACTPolicy.from_pretrained(ckpt)      # WRONG -- no normalisation
action = policy.select_action(obs)            # returns a NORMALISED action
```

Nothing raises. Decoded waypoints came out 1.6 m off for a model that was
training perfectly well. Always go through `policy_io.load_policy`:

```python
policy, pre, post = load_policy(ckpt, device)
action = post(policy.select_action(pre(obs)))
```

`tests/test_policy_io.py` guards this: it asserts the checkpoint ships the
processor files, that a postprocessed action lands inside the dataset's range,
and that the postprocessor is not a no-op.

## Two upstream behaviours the tests pin down

Both are real, both are reproduced faithfully, and both are asserted in
`tests/` so a future refactor cannot silently change them.

**Tail-padded chunks zero their last sample.** When a window runs past the end
of the global knot vector, `chunk_bspline_trajectory` pads by repeating the
final knot. The padded vector then ends with `t_max` repeated, so at
`t == t_max` every basis function has already closed its half-open support and
the spline evaluates to `0`. Interior chunks are unaffected. *Drop the last
decoded action of a padded chunk.*

**Trailing frames look backwards.** After the last fitted chunk, the sampler
keeps assigning that same chunk to the remaining frames with a growing shift,
so its domain start goes negative — the chunk begins *before* "now". This is
confined to the last ~5% of each episode. *Clamp `t_min` to 0 when rolling
out.*

## Cost note

`generate_knots` grows the knot vector one knot at a time and refits at every
step, so fitting is superlinear in episode length. Short episodes (~200 frames)
fit in milliseconds; multi-thousand-frame episodes take minutes. It is a
one-off cost paid during conversion, not during training.
