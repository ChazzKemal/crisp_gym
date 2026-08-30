"""Durable record of a deploy run: summary.json, chunks.csv, trace.npz.

Moved out of ``main()``'s ``finally`` block in ``examples/19_deploy_policy.py``.
Reporting is separated from teardown because they fail differently: teardown must
always run (a live sender thread and scaled controller gains outlive a crash),
whereas a failed report should never prevent the robot from being left safe.

The block read 23 names straight out of ``main()``'s scope. Nineteen of them are
telemetry the producer loop accumulates, so they are gathered into
:class:`RunRecord`; only ``args`` and the two live objects it inspects stay as
separate parameters.
"""

import csv
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from crisp_gym.deploy.obs import _ZEROFILL_COUNTS
from crisp_gym.deploy.timing import CONTROL_DT

logger = logging.getLogger(__name__)


@dataclass
class RunRecord:
    """What a run accumulated, as one object instead of nineteen locals."""

    out_dir: Path
    run_started_at: str
    duration_s: float
    n_obs: int
    n_act: int
    chunk_count: int
    stopped_by: str
    starvation_event_count: int = 0
    chunk_rows: list[dict] = field(default_factory=list)
    pred_dt_samples: list[float] = field(default_factory=list)
    pred_dt_samples_shadow: list[float] = field(default_factory=list)
    stage_samples_producer: dict[str, list[float]] = field(default_factory=dict)
    sender_stage_samples: dict[str, Any] = field(default_factory=dict)
    trace_records: list[Any] = field(default_factory=list)
    trace_images_buf: list[Any] = field(default_factory=list)
    shadow_action_history: Any = None
    shadow_inpaint_blend_total: int = 0
    shadow_inpaint_delta_sum: float = 0.0


def write_run_artifacts(rec: RunRecord, args, sender, shadow_policy) -> None:
    """Write summary.json (+ chunks.csv, trace.npz) for one run.

    Never raises: a run that finished on the robot should not be reported as
    failed because its bookkeeping could not be written.
    """
    # Unpacked so the moved body reads exactly as it did inside main().
    out_dir = rec.out_dir
    run_started_at = rec.run_started_at
    duration_s = rec.duration_s
    n_obs, n_act = rec.n_obs, rec.n_act
    chunk_count = rec.chunk_count
    stopped_by = rec.stopped_by
    starvation_event_count = rec.starvation_event_count
    chunk_rows = rec.chunk_rows
    pred_dt_samples = rec.pred_dt_samples
    pred_dt_samples_shadow = rec.pred_dt_samples_shadow
    stage_samples_producer = rec.stage_samples_producer
    sender_stage_samples = rec.sender_stage_samples
    trace_records = rec.trace_records
    trace_images_buf = rec.trace_images_buf
    shadow_action_history = rec.shadow_action_history
    shadow_inpaint_blend_total = rec.shadow_inpaint_blend_total
    shadow_inpaint_delta_sum = rec.shadow_inpaint_delta_sum

    try:
        run_ended_at = datetime.now().isoformat(timespec="seconds")
        # duration_s already computed above (right after sender.join)

        def _percentiles(samples: list[float]) -> dict | None:
            if not samples:
                return None
            a = np.asarray(samples, dtype=np.float64) * 1000.0
            return {
                "n": int(a.size),
                "mean_ms": float(a.mean()),
                "median_ms": float(np.median(a)),
                "p90_ms": float(np.percentile(a, 90)),
                "p99_ms": float(np.percentile(a, 99)),
                "max_ms": float(a.max()),
            }

        def _ms_percentiles(samples_ms: list[float]) -> dict | None:
            """Like _percentiles but takes samples already in ms — used
            for stage timers captured via time.perf_counter * 1000.
            """
            if not samples_ms:
                return None
            a = np.asarray(samples_ms, dtype=np.float64)
            return {
                "n": int(a.size),
                "mean_ms": float(a.mean()),
                "median_ms": float(np.median(a)),
                "p90_ms": float(np.percentile(a, 90)),
                "p99_ms": float(np.percentile(a, 99)),
                "max_ms": float(a.max()),
            }

        def _slack_stats(samples_ms: list[float]) -> dict | None:
            """Slack samples are already in ms and span both signs —
            reuse the percentile shape but expose low percentiles (p1, p10)
            since the failure mode is 'we got popped after the deadline'.
            """
            if not samples_ms:
                return None
            a = np.asarray(samples_ms, dtype=np.float64)
            return {
                "n": int(a.size),
                "mean_ms": float(a.mean()),
                "median_ms": float(np.median(a)),
                "p10_ms": float(np.percentile(a, 10)),
                "p1_ms": float(np.percentile(a, 1)),
                "min_ms": float(a.min()),
                "max_ms": float(a.max()),
            }

        def _arg_value(v):
            if isinstance(v, Path):
                return str(v)
            return v

        summary: dict = {
            "started_at": run_started_at,
            "ended_at": run_ended_at,
            "duration_s": duration_s,
            "stopped_by": stopped_by,
            "chunks_run": int(chunk_count),
            "n_act": int(n_act),
            "n_obs": int(n_obs),
            "args": {k: _arg_value(v) for k, v in vars(args).items()},
            "sender": {
                "n_processed": int(sender.n_published),
                "underrun_count": int(sender.underrun_count),
                "gripper_dedupe_count": int(
                    getattr(sender, "gripper_dedupe_count", 0)
                ),
                "gripper_latch_blocked_count": int(
                    getattr(sender, "gripper_latch_blocked_count", 0)
                ),
                "n_late_frames": int(sender.n_late_frames),
                "late_frame_pct": (
                    100.0 * sender.n_late_frames
                    / max(len(sender.slack_samples_ms), 1)
                ),
                "queue_depth_min": (
                    int(sender.queue_depth_min)
                    if sender.queue_depth_min != 2 ** 31 else None
                ),
                "queue_depth_max": int(sender.queue_depth_max),
            },
            "fps_baseline": float(args.fps),
            "control_dt_ms": float(CONTROL_DT * 1000.0),
            "starvation_events": int(starvation_event_count),
            "slack_ms": _slack_stats(sender.slack_samples_ms),
            "zerofill": {
                "n_substitutions": int(sum(_ZEROFILL_COUNTS.values())),
                "by_error": dict(_ZEROFILL_COUNTS),
            },
            "inference_ms": _percentiles(pred_dt_samples),
            "shadow_ms": _percentiles(pred_dt_samples_shadow),
            "publish_ms": _percentiles(sender.pub_dt_samples),
            "producer_stages_ms": {
                stage: _ms_percentiles(stage_samples_producer.get(stage, []))
                for stage in ("get_obs_ms", "synth_ms", "build_ms", "push_ms", "drain_wait_ms")
            },
            "sender_stages_ms": {
                stage: _ms_percentiles(sender_stage_samples.get(stage, []))
                for stage in (
                    "pop_ms", "scaler_rpc_ms", "sleep_overshoot_ms",
                    "pub_pose_ms", "pub_grip_ms", "loop_total_ms",
                )
            },
            "shadow_inpaint": (
                {
                    "tail_per_chunk": int(args.shadow_inpaint_tail),
                    "n_action_frames_blended": int(shadow_inpaint_blend_total),
                    "mean_l2_delta": (
                        shadow_inpaint_delta_sum
                        / max(shadow_inpaint_blend_total, 1)
                    ),
                    "history_len_final": len(shadow_action_history),
                }
                if args.shadow_inpaint_tail > 0 and shadow_inpaint_blend_total > 0
                else None
            ),
            "overlap_budget_ms": (
                args.overlap_threshold * 1000.0 / max(args.fps, 1e-9)
            ),
            "shadow_flavor": (
                shadow_policy.flavor if shadow_policy is not None else None
            ),
        }

        # out_dir + ts_dir were computed up-front (right after
        # run_started_at) so the video writer subprocess can stream
        # into the same folder during the run. Reuse them here.
        summary_path = out_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info("summary written to %s", summary_path)

        # Per-chunk CSV: one row per producer iteration, all the stage
        # timings + queue state. Useful for plotting drift over time
        # or for spotting outlier chunks the percentile summary hides.
        if chunk_rows:
            chunks_csv = out_dir / "chunks.csv"
            try:
                with open(chunks_csv, "w", newline="") as f:
                    writer = csv.DictWriter(
                        f, fieldnames=list(chunk_rows[0].keys()),
                    )
                    writer.writeheader()
                    writer.writerows(chunk_rows)
                logger.info("chunks.csv written to %s (N=%d)",
                            chunks_csv, len(chunk_rows))
            except Exception:
                logger.exception("failed to write chunks.csv")

        # --record-trace dump. trace.npz holds the obs→chunk pairing
        # (numerical only); per-chunk camera frames are written as
        # JPEGs under trace_images/ for visual review.
        if args.record_trace and trace_records:
            trace_npz = out_dir / "trace.npz"
            try:
                # Discover the union of keys across records. Most should
                # be present in every record (uniform env schema), but
                # we guard with np.stack-with-fallback per key.
                all_keys: list[str] = []
                seen: set = set()
                for r in trace_records:
                    for k in r:
                        if k not in seen:
                            seen.add(k)
                            all_keys.append(k)

                bundles: dict = {}
                for k in all_keys:
                    if k == "task":
                        bundles[k] = np.array(
                            [r.get(k, "") for r in trace_records], dtype=object,
                        )
                        continue
                    try:
                        arrs = [np.asarray(r[k]) for r in trace_records if k in r]
                        if len(arrs) == len(trace_records):
                            bundles[k] = np.stack(arrs)
                        else:
                            # Sparse key (not in every record); save as
                            # an object array of variable entries.
                            bundles[k] = np.array(
                                [r.get(k) for r in trace_records], dtype=object,
                            )
                    except Exception:
                        logger.exception("trace: failed to stack key %r", k)

                np.savez(trace_npz, **bundles)
                logger.info("trace.npz written to %s (N=%d)",
                            trace_npz, len(trace_records))
            except Exception:
                logger.exception("failed to write trace.npz")

            # Flush JPEGs. cv2 imports through crisp_py.camera already,
            # so this is just a re-use of the loaded module.
            if trace_images_buf:
                try:
                    import cv2  # noqa: PLC0415
                    img_dir = out_dir / "trace_images"
                    img_dir.mkdir(exist_ok=True)
                    n_ok = 0
                    for fname, bgr in trace_images_buf:
                        path = img_dir / fname
                        if cv2.imwrite(
                            str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 85]
                        ):
                            n_ok += 1
                    logger.info(
                        "trace_images: %d JPEGs written to %s (of %d buffered)",
                        n_ok, img_dir, len(trace_images_buf),
                    )
                except Exception:
                    logger.exception("failed to flush trace_images JPEGs")

        # Per-frame CSV from the sender thread. Pair with chunks.csv
        # to correlate producer-side stalls with sender-side underruns.
        sender_frame_rows = getattr(sender, "frame_rows", []) or []
        if sender_frame_rows:
            frames_csv = out_dir / "frames.csv"
            try:
                with open(frames_csv, "w", newline="") as f:
                    writer = csv.DictWriter(
                        f, fieldnames=list(sender_frame_rows[0].keys()),
                    )
                    writer.writeheader()
                    writer.writerows(sender_frame_rows)
                logger.info("frames.csv written to %s (N=%d)",
                            frames_csv, len(sender_frame_rows))
            except Exception:
                logger.exception("failed to write frames.csv")
    except Exception:
        logger.exception("failed to write summary.json")
