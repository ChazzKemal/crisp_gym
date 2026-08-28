# DemoSpeedup for crisp_gym datasets

A port of **DemoSpeedup: Accelerating Visuomotor Policies via Entropy-Guided
Demonstration Acceleration** ([paper](https://arxiv.org/abs/2506.05064),
[code](https://github.com/lingxiao-guo/DemoSpeedup), CoRL 2025 oral), wired up
to the LeRobot v3.0 datasets this repo records.

Upstream ships two full training stacks — an Aloha/ACT fork pinned to MuJoCo
2.1 + a `diffusion_policy` fork, and a `robobase`/Bigym tree — neither of which
can coexist with `lerobot-041`. As with `../bspline/`, only the **method** is
vendored here (`demospeedup_core/`): the entropy estimator, the phase
segmentation, and the retiming rule. Training is stock `lerobot-train`.

## What the method is

A demonstration is not uniformly hard. Where the policy is *confident* about
what comes next, the demo can be replayed faster without losing anything; where
it is uncertain — around contacts, grasps, insertions — it must not be.
DemoSpeedup measures that confidence with the policy itself:

1. **Proxy policy.** Train an ordinary ACT on the original demos.
2. **Label.** Replay each recorded demo through it. At every frame, draw
   `num_samples = 10` action chunks from the CVAE **prior** (`z ~ N(0, I)`),
   pool the samples of every still-valid chunk covering that frame (temporal
   aggregation), and take the KDE differential entropy of the pooled cloud.
   High entropy = the policy has many equally good options = free motion.
3. **Segment.** Z-normalise the entropy trace per episode and split it into
   *precision* (label 0) and *non-precision* (label 1) phases.
4. **Accelerate.** Walk the episode with a variable stride: one frame every
   `low_v = 2` in precision phases, one every `high_v = 4` in non-precision
   ones. Train on the result.

The accelerated policy emits waypoints spaced 2–4 source frames apart. Run at
the unchanged control rate, that *is* the speedup — nothing about the runtime
changes. **Do not also raise the replay fps**, or the acceleration is applied
twice.

## What is ported, and how it differs

| upstream | here |
| --- | --- |
| `detr/models/entropy_utils.py` (KDE, k-NN) | `demospeedup_core/entropy.py`, verbatim except the k-NN sign (see below) |
| `DETRVAE.get_samples` — a forked forward pass that reparametrises `mu = logvar = 0` | `demospeedup_core/sampling.py` — LeRobot's ACT already builds a zero latent at inference, so one projection module is temporarily wrapped to substitute `randn`. Stock forward, reverted on exit |
| dense `(T, T+chunk, 10, dim)` GPU tensor for temporal aggregation | `TemporalSampleBuffer`, a ring buffer of the last `chunk_size` predictions — identical samples, ~100x less memory |
| `hdbscan_with_custom_merge` | `demospeedup_core/segmentation.py`, two backends (below) |
| `act_utils.py::process_action_label` — subsamples the action chunk inside `__getitem__` | `demospeedup_core/retiming.py` + `convert_lerobot_to_speedup.py` — the same stride walk, applied to the episode |
| labels written into the demo's HDF5 | a sidecar next to the dataset; recorded data is never modified |

Three deviations are worth knowing about:

**Retiming the dataset instead of the chunk.** Upstream keeps every frame as a
training observation and subsamples only the action chunk it is paired with. We
drop the in-between frames outright. Consecutive frames of a retimed episode
*are* the subsampled waypoints of upstream's stride rule: that rule is
memoryless — the next step depends only on the labels at the current position,
and its `i + low_v < horizon` boundary is start-invariant — so a walk launched
from any frame already on the global walk reproduces the global walk's own
continuation (`tests/test_upstream_parity.py::test_stride_walk_is_memoryless`,
checked at every kept start frame). What we lose is chunk *starts*: the
accelerated dataset has 2–4x fewer training samples from the same demos. What
we gain is a stock LeRobot v3.0 dataset — `lerobot-train` needs no patch, and
the existing crisp_gym replay tooling can play it back to watch what the policy
is being asked to imitate.

One caveat, found by running upstream's own function: `process_action_label`
starts its loop at `i = -1`, so the first label it consults is
`current_label[-1]` — the *last* frame of the chunk, an arbitrary distance
ahead. Its first emitted waypoint is therefore 1 or 3 frames out (3 because
`label[-1:3]` slices to empty and `torch.all` of an empty tensor is `True`),
whatever the labels near the start of the chunk say; every later step follows
the normal rule. So we match upstream's *intended* walk and differ from its
literal first step, which we treat as a slip rather than a design choice.

**Segmentation backend.** Upstream clusters `(z(frame_index), z(entropy))` with
HDBSCAN(`min_cluster_size=5`). Neither `scikit-learn` nor `hdbscan` is
installed in `lerobot-041`, so the default backend is `threshold`: split at the
episode mean, enforce a minimum run length, then apply upstream's own cluster
rule to the merged runs. `--segmenter hdbscan` reproduces upstream exactly and
runs in the `lerobot` conda env (sklearn 1.7).

The two are *not* interchangeable. On the first three episodes of
`merged_act_finetune_20260528` they disagree on **15-20% of frames**, and on one
episode they differ on how much of it is fast at all (28.7% vs 12.7%). Both
labellings are defensible — the disagreement sits at phase boundaries and on
frames near the mean — but pick one deliberately and record which. Compare them
on your own data with `analyze_labels.py --resegment hdbscan`, which
re-segments the stored entropy without re-running the policy.

Upstream's cluster rule is kept quirk and all: `if np.mean(cluster_points[:, 1]
< 0)` is truthy whenever *any* frame of the cluster sits below the mean, so a
phase is only marked fast when **every** one of its frames is above-mean
entropy. This biases the labelling towards caution, which is the right
direction to be wrong in on a real UR10e.

**k-NN entropy sign.** Upstream's Kozachenko-Leonenko estimator subtracts the
log-distance term, which inverts it — a wider sample cloud would score as
*lower* entropy. Corrected here. Nothing in the pipeline depends on it (the
labelling uses the KDE estimator, as upstream does); it is kept as an
independent cross-check, and an inverted cross-check is worse than none.

## Layout

```
demospeedup/
  demospeedup_core/               vendored, dependency-light (numpy + torch + scipy)
    entropy.py                    KDE / k-NN differential entropy
    sampling.py                   CVAE-prior sampling from a LeRobot ACT + temporal aggregation
    segmentation.py               entropy trace -> precision / non-precision labels
    retiming.py                   labels -> the frames the accelerated episode keeps
  lerobot_bridge.py               LeRobot v3.0 episode ranges + the label sidecar
  train_proxy_act.py              step 1 (usually skippable — see below)
  label_entropy.py                step 2
  analyze_labels.py               inspect a labelling run before converting
  convert_lerobot_to_speedup.py   step 3
  train_speedup_act.py            step 4
  walkthrough.ipynb               all of the above, one step per cell, with figures
  walkthrough.html                the same run as a self-contained page (published artifact)
  tests/                          the correctness suite
    test_upstream_parity.py       numerical parity against a DemoSpeedup clone
```

Nothing in `demospeedup_core/` imports LeRobot.

## Workflow

`walkthrough.ipynb` runs this whole sequence one cell at a time on a handful of
episodes, plots the entropy trace and the surviving frames, sweeps the strides,
and verifies the conversion — start there if you want to *see* what each step
does. It shells the pipeline steps out to `conda run -n lerobot-041`, so any
kernel with numpy/pandas/av works (use the `lerobot` kernel for the figures).

```bash
cd ur10_clearpath/Yunfei/crisp_gym/demospeedup
CG="conda run -n lerobot-041"

# 0. the suite should be green before you trust any of this
$CG python -m pytest tests/ -q                       # 46 tests

# 0b. numerical parity against upstream itself (37 more; skipped without a clone)
git clone --depth 1 https://github.com/lingxiao-guo/DemoSpeedup /tmp/DemoSpeedup
DEMOSPEEDUP_UPSTREAM=/tmp/DemoSpeedup $CG python -m pytest tests/ -q

# 1. proxy policy -- SKIP THIS if you already have an ACT checkpoint trained on
#    the dataset you want to accelerate. Any such checkpoint is a valid proxy.
$CG python train_proxy_act.py --wandb

# 2. label (~65 ms/frame on a 4090: ~25 min for 21.5k frames; --episodes 2 first)
$CG python label_entropy.py \
    --dataset-root /home/batur/Coding/data/merged_act_finetune_20260528 \
    --policy-path ../outputs/train/act_cart7_v2_angleaxis_nogrip_chunk100_ft_20260528/checkpoints/last/pretrained_model

# 3. check what the labels imply BEFORE re-encoding 20k video frames
$CG python analyze_labels.py \
    --dataset-root /home/batur/Coding/data/merged_act_finetune_20260528 --per-episode

# 4. write the accelerated dataset
$CG python convert_lerobot_to_speedup.py \
    --src /home/batur/Coding/data/merged_act_finetune_20260528 \
    --dst /home/batur/Coding/data/merged_speedup_20260528

# 5. train on it
$CG python train_speedup_act.py --wandb
```

Step 2 writes `<dataset>/meta/demospeedup/labels.parquet` (per frame: entropy,
z-score, label) and `<dataset>/meta/demospeedup.json` (how they were produced).
The source dataset is otherwise untouched, so several labelling runs can be
compared by moving the sidecar aside.

## Choosing the strides

`analyze_labels.py` reports the number that decides whether this is safe on the
real arm: **how far the end-effector is commanded to move per control period**
once the in-between frames are gone. Measured on the first three episodes of
`merged_act_finetune_20260528`:

| `low_v` | `high_v` | speedup | worst step | worst speed |
| --- | --- | --- | --- | --- |
| 1 | 1 | 1.00x (source) | 87.2 mm | 1.74 m/s |
| 2 | 3 | 2.09x | 99.5 mm | 1.99 m/s |
| 2 | 4 | 2.16x | 106.8 mm | 2.14 m/s |
| 3 | 6 | 3.20x | 141.9 mm | 2.84 m/s |

The source data already peaks at 87 mm per 20 Hz period, so upstream's `(2, 4)`
raises the worst case by only ~23% while halving the episode length — the fast
phases it accelerates are the ones that were already sparse. `(3, 6)` is a
different proposition. If the CRISP cartesian controller cannot track the number
in the last column, lower `--high-v` (the flag exists on both
`label_entropy.py` and `convert_lerobot_to_speedup.py`; the converter defaults
to whatever the labelling run used) rather than finding out on the robot.

`--fast-prefix` forces the first N frames to non-precision. Upstream hard-codes
50 (`initial_labels[:50] = -1`) so the approach phase of a 400-step Aloha
episode is always fast. It defaults to 0 here — our episodes are ~300 frames
and the approach length varies with where the operator started, so the entropy
signal decides. Set it if the first seconds of every demo are known dead time.

## Chunk size

Upstream halves it for the accelerated run — 50 → 25 for ACT, 48 → 24 for DP —
"to maintain geometrical consistency": one accelerated frame covers 2–4 source
frames, so half the chunk length still spans about as much *motion*. Our ACT
baselines use `chunk_size=100`, hence 50 in `train_speedup_act.py`. The proxy
in step 1 keeps 100 — a longer chunk gives the temporal aggregation more
predictions to pool per frame, which is what the entropy is computed over.
