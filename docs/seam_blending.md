# Chunk-seam blending: how the handoff works, and why it is split in two

A policy is queried every K actions, but the arm never stops. Each new chunk is a
fresh plan that starts from wherever the policy thinks the arm is, and if we simply
concatenated plans the arm would step from the end of one prediction to the start of
the next. Seam blending is the handoff that makes that transition continuous.

## The mechanism

At the end of every chunk we deliberately **do not send the last N rows**. We keep
them (`blend_carry`). When the next chunk arrives, those kept rows are folded into
its first N rows, ramping the weight from old to new, so the arm eases out of the
committed plan and into the fresh one.

```
chunk n     [ 0 .............. K-N-1 ][ K-N ... K-1 ]
                    emitted               held back ──┐
                                                      │  blend_carry
chunk n+1   [ 0 ... N-1 ][ N .............. K-N-1 ][ ...]
              ▲ folded ──────────────────────────────┘
              in, ramped old→new
```

`--blend-overlap N` sets the width, `--blend-mode` picks how:

- **linear** — per-row weighted average, `w = (j+1)/(n_blend+1)`. This is genuine
  temporal ensembling: the new prediction's first rows influence the result.
- **hermite** — a cubic bridge from the last emitted pose *and its velocity* to the
  first verbatim new row. Matches position **and** velocity, so there is no kink at
  the seam, but the new chunk's first N rows are discarded rather than averaged.

The gripper channel `[6]` is **never** blended in either mode — it is binary, and an
averaged gripper command is meaningless. It always takes the new chunk's value.

## Why the block is split across steps 2d and 3c

The two halves run at different points in the loop, and that is deliberate:

| half | step | what it does |
|---|---|---|
| fold in | **2d** | mix the *previous* chunk's carry into this chunk's head |
| hold back | **3c** | keep this chunk's last N *emitted* rows as the next carry |

They used to be one block at 2d. That was correct only as long as nothing between
2d and the queue push changed the row set — true for the baseline, false the moment
methods arrived. A method's steps reshape rows:

- `PaceSpeed` truncates the chunk to `n_action_steps`
- `GripperReplicate` (demospeedup) *inserts* rows to give the gripper stroke time

So at 2d the loop does not yet know which rows will be sent. Capturing the carry
there held back rows from a stretch of the plan that the pipeline then discarded.

**Measured cost of getting this wrong** (2026-08-31, PACE at 2x, ACT `chunk_size=100`,
`n_action_steps=32`, `--blend-overlap 4`): the policy returned 100 rows, 2d held back
rows 96–99, and the pipeline then cut the chunk to rows 0–31. The next chunk's head
was therefore blended toward a pose **65 frames further along the trajectory** than
where the arm actually was.

| gap | mean | p90 |
|---|---|---|
| row 31 → row 96 (what the blend pulled toward) | **236 mm** | 394 mm |
| genuine motion across the 4-frame blend window | 22 mm | 40 mm |

An order of magnitude of spurious displacement injected into the first commands of
every chunk, roughly once per second. On hardware it read as continuous jitter. The
baseline was unaffected because with `method: none` nothing reshapes rows, so rows
96–99 really were the next four frames.

## Placement rules (do not reorder without reading this)

**3c must run after step 3.** Step 3 is where the method pipeline runs, so it is the
first point at which `chunk` is the set of rows that will actually be sent.

**3c must run before step 3b.** 3b is the gripper-grab slowdown, whose
`close_slow_remaining` window carries across chunk boundaries. If it saw rows that
are subsequently held back, its carry would advance by more frames than the arm
executes and the window would expire early after a seam.

**`s_raw` is sliced alongside `chunk`.** From step 3 onward every row carries a
speed. The two arrays must stay the same length or cycle-snap sees a mismatch — this
is the one line the old placement did not need, because it truncated before speeds
existed.

## Consequences worth knowing

- **The baseline is bit-identical.** Deploy runs use `max_speed == min_speed == 1.0`,
  and `_build_chunk_speed_schedule` short-circuits to `np.ones(K)` before any
  curvature math, so computing the schedule on K rows and slicing to K-N gives the
  same array as computing it on K-N rows. On the legacy heuristic path
  (`max_speed > 1`) the schedule now sees N more rows of look-ahead at its right
  edge, which is strictly more information, not less.

- **`prev_emitted_tail` is now a real velocity anchor.** Hermite's `v_start` is the
  difference of the last two *emitted* rows. Before the split it was the difference
  of two rows that, under a truncating method, were never sent.

- **Residual caveat on the method path.** `GripperHold` runs *inside* the pipeline,
  so it still sees rows that 3c may subsequently hold back, and its cross-chunk carry
  can over-advance by up to N frames. This affects only method-driven runs and only
  when a grasp straddles a seam. Fixing it properly means telling the pipeline how
  many rows will be emitted, which is not worth the coupling until it is measured to
  matter.

- **Blend width is defined in rows, not seconds.** Its physical length therefore
  scales with `s_eff`: a width that is safe at 1.0x covers twice the distance at 2.0x.
  Open question, tracked in the plan.

## Regression test

`tests/test_deploy_seam.py` drives `run_producer_loop` end-to-end with a ramp source
(chunk n is the ramp `n*1000 .. n*1000+19`, so every row is identifiable) and a
`_TruncateTo(8)` step standing in for `PaceSpeed`. It asserts the exact value of the
blended head, which pins down *which* rows the carry came from:

    emitted chunk 1 = 1000..1005,  carry = [1006, 1007]
    chunk 2 head[0] = (2/3)*1006 + (1/3)*2000

and asserts it is **not** the value implied by the old ordering (carry `[1018, 1019]`).
The end-to-end shape is the point: every stage was individually correct, the bug was
only in the ordering between them. A third case runs with `steps=None` and passes
against both the old and new code, which is what makes the "baseline is untouched"
claim checkable rather than asserted.
