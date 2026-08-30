"""The producer loop: observation in, scheduled targets out.

Moved out of ``main()`` in ``examples/19_deploy_policy.py``. Each iteration asks the
chunk source for actions, computes a speed schedule, snaps it to whole controller
cycles, anchors deadlines, and pushes one :class:`TargetItem` per action onto the
sender's queue -- then blocks until the queue drains to ``--overlap-threshold`` so
the next inference overlaps with actions still in flight.

The loop is deliberately policy-agnostic: it knows only the
:class:`~crisp_gym.deploy.sources.ChunkSource` protocol, which is what lets a method
pipeline or a different runner drive the same hardware path.

Telemetry lives on the caller's :class:`~crisp_gym.deploy.trace.RunRecord`. The seven
collections are mutated in place; the five scalars are written back in a ``finally``
so a run that ends by exception still reports how far it got.
"""

import logging
import time

import numpy as np

from crisp_gym.deploy.pipeline import (
    _build_chunk_speed_schedule,
    _inpaint_blend_into_history,
)
from crisp_gym.deploy.gripper import GripperCloseWindow
from crisp_gym.deploy.obs import _get_obs_zerofill
from crisp_gym.deploy.sender import TargetItem
from crisp_gym.deploy.sources import DatasetExhausted
from crisp_gym.deploy.timing import _pre_compute_chunk_arrays, build_speed_queue_arrays

logger = logging.getLogger(__name__)


def run_producer_loop(
    *,
    env,
    chunk_source,
    q,
    args,
    rec,
    dt_base: float,
    obs_schema,
    gripper_enabled: bool,
    gripper_unnormalize_fn,
    obs_buf,
    last_obs,
    lookbehind_buf,
    shadow_policy=None,
    # Cross-chunk carry. These are loop state, but they are seeded by the caller
    # because the first chunk has no predecessor: the blend has nothing to bridge
    # from, the gripper edge detector has no previous level, and the deadline
    # anchor has no prior chunk to follow. Dropping any of them silently degrades
    # a seam rather than raising.
    blend_carry=None,
    prev_emitted_tail=None,
    close_slow_remaining: int = 0,
    prev_grip_closed=None,
    last_pushed_deadline=None,
    dt_eff_mean_prev: float | None = None,
) -> None:
    """Run until the chunk source is exhausted, --max-chunks is hit, or Ctrl-C."""
    chunk_count = rec.chunk_count
    stopped_by = rec.stopped_by
    starvation_event_count = rec.starvation_event_count
    shadow_inpaint_blend_total = rec.shadow_inpaint_blend_total
    shadow_inpaint_delta_sum = rec.shadow_inpaint_delta_sum
    chunk_rows = rec.chunk_rows
    pred_dt_samples = rec.pred_dt_samples
    pred_dt_samples_shadow = rec.pred_dt_samples_shadow
    stage_samples_producer = rec.stage_samples_producer
    trace_records = rec.trace_records
    trace_images_buf = rec.trace_images_buf
    shadow_action_history = rec.shadow_action_history
    if dt_eff_mean_prev is None:
        dt_eff_mean_prev = dt_base

    # One detector for the whole run; it owns the cross-chunk carry that used to
    # live in two loop locals. Seeded from the caller so the contract is unchanged.
    _n_slow = int(getattr(args, "gripper_slowdown_frames", 0))
    grip_window = None
    if _n_slow > 0 and gripper_enabled:
        grip_window = GripperCloseWindow(_n_slow, invert=bool(args.invert_gripper))
        grip_window.remaining = int(close_slow_remaining)
        grip_window.prev_closed = prev_grip_closed

    try:
        logger.info(
            "Phase 4: deploying — Ctrl-C to stop. Overlap threshold = %d "
            "(next inference fires when queue <= %d).",
            args.overlap_threshold, args.overlap_threshold,
        )
        while True:
            if args.max_chunks > 0 and chunk_count >= args.max_chunks:
                logger.info("Reached --max-chunks=%d, stopping", args.max_chunks)
                stopped_by = "normal"
                break

            # 1. Refresh obs buffer (cameras / joints / ee / gripper). This
            #    is the ONLY hot-path touch of env._get_obs(); the sender
            #    thread never reads obs. Cameras live behind their own
            #    daemon spinners — the get_obs call below is cheap (dict
            #    copy of the latest cached frame) but does momentarily
            #    grab the GIL. That's fine here, off the publish path.
            #    Wrapped in _get_obs_zerofill so a missing/silent sensor
            #    substitutes a zero array of the right shape instead of
            #    raising RuntimeError mid-deploy.
            _t_stage = time.perf_counter()
            obs_buf.append(_get_obs_zerofill(env, obs_schema, last_obs))
            get_obs_ms = (time.perf_counter() - _t_stage) * 1000.0
            stage_samples_producer["get_obs_ms"].append(get_obs_ms)

            # 2. Request a chunk. For a real policy this blocks for inference
            #    latency (~10-50 ms typical, ACT can be <10 ms, diffusion
            #    50-200 ms). For fake sources it returns immediately.
            #    Pre-inference snapshot: how many items are queued AND what
            #    drain budget that buys us at the previous chunk's cadence.
            #    If inference latency exceeds the budget, the sender will
            #    block on q.get() and we'll observe an underrun cluster.
            q_before_inf = q.qsize()
            tail_budget_ms = q_before_inf * dt_eff_mean_prev * 1000.0
            t_send = time.monotonic()
            try:
                chunk = chunk_source.request(obs_buf)
            except DatasetExhausted as e:
                logger.info("Fake dataset exhausted (%s) — exiting cleanly.", e)
                stopped_by = "dataset_exhausted"
                break
            except (BrokenPipeError, EOFError):
                logger.error("Chunk source pipe closed; exiting loop.")
                failed = True
                stopped_by = "chunk_source_pipe_closed"
                break
            inf_dt = time.monotonic() - t_send
            inf_dt_ms = inf_dt * 1000.0
            pred_dt_samples.append(inf_dt)
            stage_samples_producer["synth_ms"].append(inf_dt_ms)
            chunk_count += 1

            # --record-trace capture. Done right after the chunk arrives,
            # BEFORE the speed-schedule/cycle-snap step modifies the
            # publish cadence. The action vectors themselves aren't
            # modified by the scaler (only timing is), so the chunk we
            # capture here is exactly what the sender will publish.
            if args.record_trace and (chunk_count - 1) % max(1, args.record_trace_every) == 0:
                obs_now = obs_buf[-1]
                record = {
                    "chunk_idx": int(chunk_count - 1),  # match chunk_rows
                    "wall_ns": int(time.time_ns()),
                    "mono_ns": int(time.monotonic_ns()),
                    "chunk": np.asarray(chunk, dtype=np.float32),
                }
                for k, v in obs_now.items():
                    if k.startswith("observation.state."):
                        record[k] = np.asarray(v, dtype=np.float32).reshape(-1)
                task = obs_now.get("task", "")
                if task:
                    record["task"] = str(task)
                trace_records.append(record)

                # Buffer JPEG-encodable image arrays for shutdown-time disk
                # write. crisp_py returns HxWxC uint8 RGB; cv2.imwrite needs
                # BGR. We do the cheap channel-flip in-line here and defer
                # the actual write to keep the hot loop fast.
                if not args.record_trace_no_images:
                    for k, v in obs_now.items():
                        if not k.startswith("observation.images."):
                            continue
                        img = np.asarray(v)
                        if img.ndim != 3 or img.shape[-1] != 3:
                            continue
                        cam = k.rsplit(".", 1)[-1]
                        fname = f"chunk_{chunk_count - 1:06d}_{cam}.jpg"
                        # Channel-flip view; cv2 will encode at write time.
                        trace_images_buf.append((fname, img[..., ::-1].copy()))

            if inf_dt_ms > tail_budget_ms:
                starvation_event_count += 1
                logger.warning(
                    "chunk %d: inference (%.1fms) > queue tail budget "
                    "(%.1fms, q_before_inf=%d * dt_eff_mean_prev=%.1fms). "
                    "Sender likely starved; bump --overlap-threshold or "
                    "accept underruns.",
                    chunk_count, inf_dt_ms, tail_budget_ms,
                    q_before_inf, dt_eff_mean_prev * 1000.0,
                )

            # 2b. Run the shadow ACT forward pass on the same obs the fake
            #     source just saw. We discard the output for execution — only
            #     the wall time matters — but we DO route it into the shadow
            #     history deque and optionally inpaint-blend it there
            #     (--shadow-inpaint-tail). The shadow history is never
            #     consumed by the robot; it's purely a smoke test of the
            #     blending math, exercising the code path a real RTC-enabled
            #     producer would run.
            if shadow_policy is not None:
                t_shadow = time.monotonic()
                try:
                    shadow_chunk = shadow_policy.predict(obs_buf[-1])
                except Exception:
                    logger.exception(
                        "shadow predict() raised at chunk %d; disabling shadow.",
                        chunk_count,
                    )
                    shadow_policy = None
                else:
                    pred_dt_samples_shadow.append(time.monotonic() - t_shadow)
                    if args.shadow_inpaint_tail > 0:
                        n_blended, mean_delta = _inpaint_blend_into_history(
                            shadow_action_history,
                            shadow_chunk,
                            args.shadow_inpaint_tail,
                        )
                        if n_blended > 0:
                            shadow_inpaint_blend_total += n_blended
                            shadow_inpaint_delta_sum += mean_delta * n_blended
                    else:
                        # No blending — still track the chunk in history so
                        # the size of the history reflects real usage.
                        for action in shadow_chunk:
                            shadow_action_history.append(np.asarray(action).copy())

            if not isinstance(chunk, np.ndarray) or chunk.ndim != 2:
                logger.warning("Chunk %d: unexpected payload %r — skipping",
                               chunk_count, type(chunk).__name__)
                continue
            K_raw = chunk.shape[0]
            if K_raw == 0 or chunk.shape[1] < 7:
                logger.warning("Chunk %d: bad shape %s — skipping",
                               chunk_count, chunk.shape)
                continue

            # 2c. Stride: decimate the chunk before speed schedule. Each
            #     remaining frame still gets one dt_eff worth of sender time,
            #     so the trajectory advances `stride` times faster per
            #     published target. The speed schedule + cycle-snap run on
            #     the strided chunk; deltas between consecutive entries are
            #     `stride` times larger, which compute_speed_schedule sees
            #     directly. Combine with --max-speed for total speedup =
            #     stride × s_eff at dt_eff = dt_base / s_eff cadence.
            if args.stride > 1:
                chunk = chunk[::args.stride].copy()
            K = chunk.shape[0]
            if K == 0:
                logger.warning(
                    "Chunk %d: stride=%d produced empty chunk from K_raw=%d; "
                    "skipping", chunk_count, args.stride, K_raw,
                )
                continue

            # 2d. Chunk-seam blending (temporal ensembling). Hold back the
            #     last N raw frames of this chunk; average them with the next
            #     chunk's first N frames, ramping the weight old->new so the
            #     seam stays continuous with what's executing but converges to
            #     the fresher prediction. Operates on the RAW action array
            #     (xyz + rotvec) BEFORE pose/quat conversion; the gripper
            #     channel [6] is NEVER averaged (binary) — it takes the new
            #     chunk's value. Producer-side → applies to both senders. N is
            #     clamped to K//2 so the blended head [0:N] and held-back tail
            #     [K-N:] never overlap. --blend-overlap 0 keeps head-to-tail.
            if args.blend_overlap > 0 and K >= 2:
                N = min(int(args.blend_overlap), K // 2)
                if blend_carry is not None:
                    if args.blend_mode == "hermite" and prev_emitted_tail is not None and K > N + 1:
                        # Cubic Hermite bridge from (p_start, v_start) at
                        # the last actually-emitted frame to (p_end, v_end)
                        # at the first verbatim new-chunk frame after the
                        # blend zone. The blend slots chunk[0:N] are filled
                        # with N interior samples of the cubic. Bridges
                        # both position AND velocity -> no boundary kink.
                        #
                        # Parameterization: cubic on s in [0, 1], with N+1
                        # equal subdivisions (slot 0 at s=1/(N+1), slot N-1
                        # at s=N/(N+1)). Frame-step deltas (no dt scaling)
                        # because dt cancels between v and T in the
                        # standard Hermite form.
                        p_start = prev_emitted_tail[-1, :6].astype(np.float64)
                        v_start = (
                            prev_emitted_tail[-1, :6].astype(np.float64)
                            - prev_emitted_tail[-2, :6].astype(np.float64)
                        )
                        p_end = chunk[N, :6].astype(np.float64)
                        v_end = (
                            chunk[N + 1, :6].astype(np.float64)
                            - chunk[N, :6].astype(np.float64)
                        )
                        T_frames = float(N + 1)
                        s_vec = (np.arange(N) + 1) / T_frames   # (N,)
                        h00 = 2 * s_vec ** 3 - 3 * s_vec ** 2 + 1
                        h10 = s_vec ** 3 - 2 * s_vec ** 2 + s_vec
                        h01 = -2 * s_vec ** 3 + 3 * s_vec ** 2
                        h11 = s_vec ** 3 - s_vec ** 2
                        bridge = (
                            h00[:, None] * p_start
                            + (h10[:, None] * T_frames) * v_start
                            + h01[:, None] * p_end
                            + (h11[:, None] * T_frames) * v_end
                        )
                        chunk[:N, :6] = bridge.astype(chunk.dtype)
                        # Gripper [6] left as the new chunk's value
                        # (NEVER interpolated, matches linear mode).
                    else:
                        # Linear path (existing behaviour). Per-frame
                        # weighted average; skips the first `skip` frames
                        # as committed.
                        n = min(len(blend_carry), N)
                        skip = min(max(0, int(args.blend_skip)), n)
                        n_blend = n - skip  # frames actually averaged
                        for i in range(n):
                            if i < skip:
                                # Commit horizon: execute the previous chunk's
                                # prediction VERBATIM (pose AND gripper) for these
                                # already-in-flight timesteps; blending starts
                                # after them.
                                chunk[i, :] = blend_carry[i, :]
                            else:
                                # Ramp restarted across the (n_blend) blended
                                # frames: frame `skip` stays close to the committed
                                # old plan (w small) and converges to new by the
                                # end. Gripper [6] is left as the new chunk's value
                                # (never averaged).
                                j = i - skip
                                w = (j + 1) / (n_blend + 1)  # old-heavy -> new-heavy
                                chunk[i, :6] = (
                                    (1.0 - w) * blend_carry[i, :6] + w * chunk[i, :6]
                                )
                blend_carry = chunk[K - N:].copy()   # hold back for next seam
                chunk = chunk[: K - N].copy()         # emit the rest now
                K = chunk.shape[0]
                # Save the last 2 actually-emitted frames for the next
                # iteration's Hermite v_start. Only needed in hermite mode,
                # but the cost is one ndarray copy of shape (2, 7) per
                # chunk so we do it unconditionally to keep the code paths
                # symmetric. K >= 2 by the outer `if K >= 2` guard above
                # (post-emit K is K_orig - N, which is >= K_orig // 2 >= 1;
                # for K_orig >= 4 it's >= 2).
                if K >= 2:
                    prev_emitted_tail = chunk[K - 2:K].copy()

            # 3. Speed schedule on the (possibly strided) chunk.
            _t_stage = time.perf_counter()
            past = (
                np.asarray(lookbehind_buf, dtype=np.float64)
                if len(lookbehind_buf) > 0 else None
            )
            s_raw = _build_chunk_speed_schedule(
                chunk.astype(np.float64), args, past_buffer=past,
            )

            # 3b. Gripper-grab slowdown (--gripper-slowdown-frames). On each
            #     open→close transition, force s_raw = 1.0 (real-time) for that
            #     frame + the next N-1, so the arm runs at normal speed *while it
            #     grabs* and resumes speedup for the carry. Edge-triggered on the
            #     CLOSE transition, NOT the level — staying closed during the
            #     carry fires nothing, so transport keeps the speedup. The window
            #     can straddle chunk boundaries (close_slow_remaining carries the
            #     leftover). No-op when N=0, and a no-op anyway with no speedup
            #     (s_raw already 1.0). Baked into s_raw → flows through cycle-snap
            #     into dt_eff/deadlines, so it also works with --cpp-sender.
            if grip_window is not None:
                slow_mask = grip_window.mask(chunk)
                if slow_mask.any():
                    s_raw[slow_mask] = 1.0

            # 4. Cycle-snap.
            cycles, dt_eff, s_eff = build_speed_queue_arrays(
                s_raw, dt_base, K, retime=True,
            )

            # 5. Pre-compute pose / gripper for each frame.
            target_xyz, target_quat, grip_raw, actions_f32 = _pre_compute_chunk_arrays(
                chunk,
                args=args,
                gripper_enabled=gripper_enabled,
                gripper_unnormalize_fn=gripper_unnormalize_fn,
                rotation_from_action=env.action_to_rotation,
            )
            build_ms = (time.perf_counter() - _t_stage) * 1000.0
            stage_samples_producer["build_ms"].append(build_ms)

            # 6. Anchor deadlines. Two regimes:
            #    (a) Queue is empty / first chunk → anchor at now. Sender
            #        publishes frame 0 at now + dt_eff[0].
            #    (b) Items still in queue (overlap append) → anchor at the
            #        last-pushed item's deadline. New frame 0 publishes
            #        dt_eff[0] AFTER the last in-flight item finishes,
            #        giving the controller a clean dt_eff[0] window. No
            #        deadline collisions, no payload overwrites.
            # Capture queue depth pre-push for the log AFTER the q.put loop
            # (we log *after* pushing so the logger.info doesn't sit
            # between now_mono and the first q.put — that gap was costing
            # ~30 ms of Rich-rendering and pushing item 0's deadline into
            # the past, causing cascading underruns).
            q_before_push = q.qsize()

            now_mono = time.monotonic()
            if last_pushed_deadline is None or last_pushed_deadline < now_mono:
                # Empty queue, or last deadline already passed — anchor at now.
                anchor = now_mono
                anchor_mode = "fresh"
            else:
                anchor = last_pushed_deadline
                anchor_mode = "overlap"
            deadlines = anchor + np.cumsum(dt_eff)

            # 7. Push K TargetItems onto the queue IMMEDIATELY after the
            #    anchor decision — no logging in between. Always pushes
            #    (even in --dry-run); sender's dry_run flag handles the
            #    ROS-side gating.
            _t_stage = time.perf_counter()
            for i in range(K):
                grip = float(grip_raw[i]) if gripper_enabled else None
                item = TargetItem(
                    pose_xyz=target_xyz[i],
                    pose_quat=target_quat[i],
                    grip_raw=grip,
                    action=actions_f32[i],
                    deadline_mono=float(deadlines[i]),
                    frame_idx=(chunk_count - 1) * K + i,
                    s_eff=float(s_eff[i]),
                    cycles=int(cycles[i]),
                )
                q.put(item)
            push_ms = (time.perf_counter() - _t_stage) * 1000.0
            stage_samples_producer["push_ms"].append(push_ms)
            last_pushed_deadline = float(deadlines[-1])
            # Carry the chunk's mean dt_eff into the next iteration so the
            # pre-inference budget check is accurate.
            dt_eff_mean_prev = float(np.mean(dt_eff))

            # Feed the emitted action rows into the lookbehind buffer so the
            # next chunk's speed schedule can see real past motion at its
            # left boundary. Uses the post-blend, post-emit chunk (what was
            # actually queued for publish) — the truncated `K-N` slice when
            # --blend-overlap is active, the full K rows otherwise. deque
            # maxlen enforces the window size; appends are no-ops when
            # --lookbehind == 0.
            if lookbehind_buf.maxlen and lookbehind_buf.maxlen > 0:
                for i in range(K):
                    lookbehind_buf.append(
                        np.asarray(chunk[i, :6], dtype=np.float64)
                    )

            # NOW it's safe to log — sender has had a chance to pick up
            # item 0 from a deadline that's still genuinely in the future.
            logger.info(
                "Chunk %d: shape=%s inf=%.1fms anchor=%s  s_raw[%.2f-%.2f] "
                "s_eff[%.2f-%.2f] dt_eff[%.0f-%.0f]ms  "
                "q_before_inf=%d budget=%.0fms q_before_push=%d",
                chunk_count, tuple(chunk.shape), inf_dt_ms, anchor_mode,
                float(s_raw.min()), float(s_raw.max()),
                float(s_eff.min()), float(s_eff.max()),
                float(dt_eff.min()) * 1000, float(dt_eff.max()) * 1000,
                q_before_inf, tail_budget_ms, q_before_push,
            )

            # 8. Wait for the queue to drain to the overlap threshold, then
            #    loop back to request the next chunk. With threshold=2,
            #    inference fires when 2 items are still in flight — if
            #    inference latency < 2 * dt_eff (~66 ms at 30 Hz), the
            #    sender thread never sees an empty queue at chunk
            #    boundaries. Threshold=0 reverts to "wait for full drain"
            #    (sequential replan, ~inf_dt_ms gap per chunk).
            _t_stage = time.perf_counter()
            thresh = max(0, int(args.overlap_threshold))
            while q.qsize() > thresh:
                time.sleep(0.005)
            drain_wait_ms = (time.perf_counter() - _t_stage) * 1000.0
            stage_samples_producer["drain_wait_ms"].append(drain_wait_ms)

            # Per-chunk row for chunks.csv. Captures everything observable
            # at the producer level for post-hoc analysis. Negative
            # drain_wait_ms is impossible; near-zero means the sender was
            # already at/below threshold by the time we got back here.
            chunk_rows.append({
                "chunk_idx": chunk_count,
                "q_before_inf": q_before_inf,
                "q_before_push": q_before_push,
                "anchor_mode": anchor_mode,
                "K": K,
                "dt_eff_mean_ms": float(np.mean(dt_eff)) * 1000.0,
                "get_obs_ms": get_obs_ms,
                "synth_ms": inf_dt_ms,
                "build_ms": build_ms,
                "push_ms": push_ms,
                "drain_wait_ms": drain_wait_ms,
            })
    finally:
        # Scalars do not propagate through the record by mutation, so hand them
        # back explicitly -- including on the exception paths, which is how this
        # loop normally ends (Ctrl-C, DatasetExhausted, --max-chunks).
        rec.chunk_count = chunk_count
        rec.stopped_by = stopped_by
        rec.starvation_event_count = starvation_event_count
        rec.shadow_inpaint_blend_total = shadow_inpaint_blend_total
        rec.shadow_inpaint_delta_sum = shadow_inpaint_delta_sum
        rec.shadow_action_history = shadow_action_history
        rec.shadow_policy = shadow_policy
