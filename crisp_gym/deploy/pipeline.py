"""Chunk shaping between the policy and the sender.

Moved out of ``examples/19_deploy_policy.py``. Everything here rewrites the chunk's
*contents* -- its actions or its per-step speeds -- which is the line that separates
this module from the loop: the loop owns time, queueing, and when to ask the policy
for more.

``_build_chunk_speed_schedule`` is the seam a method plugs into. Its signature,
``(actions, args, past_buffer) -> s_raw (K,)``, is already the shape a method's
speed decision takes, which is why PACE can be dropped in here without the loop
knowing anything about it.
"""

import numpy as np

from crisp_gym.deploy.timing import (
    compute_speed_schedule,
    compute_speed_schedule_cumangle,
)


# ---------------------------------------------------------------------------
# Zero-fill for missing sensor data
#
# If a camera / joint_state / gripper topic never receives a message,
# crisp_py raises RuntimeError out of `current_image` / `joint_values` /
# `Gripper.value`. For smoke tests and partial sensor setups (e.g. one
# camera down) we substitute a zero-filled array of the right shape so
# the chunk source still sees a well-formed obs dict and the pipeline
# keeps running. Always-on: the count of substitutions per error message
# is surfaced in summary.json, so a real deploy that's missing a sensor
# is still visible after the fact.
# ---------------------------------------------------------------------------
# Shadow policy — real LeRobot ACT (or a torchvision stub) run alongside the
# fake source to exercise the inference path & measure realistic latency.
#
# Architecture: producer thread calls `_ShadowACTPolicy.predict(obs)` after
# each `chunk_source.request(obs_buf)`. The shadow's output is RANDOM and is
# NEVER queued for execution; only its wall-clock latency is recorded into
# `pred_dt_samples_shadow`. The cycle-snap queue still drains the dataset
# chunk, so the robot is safe.
#
# RTC note: with `temporal_ensemble_coeff` set, ACTConfig forces
# `n_action_steps = 1`. Our producer is per-chunk (calls predict_action_chunk,
# not select_action), so the temporal ensembler is constructed but its
# `.update()` is never invoked — i.e., RTC is *configured* and the model
# *runs* under the RTC architecture, but the blending step is dormant
# because we don't query per-step. Strict-RTC exercising would require a
# per-step shadow loop on a separate thread (deferred follow-up).
# ---------------------------------------------------------------------------


def _inpaint_blend_into_history(
    history,  # collections.deque[np.ndarray]
    new_chunk: np.ndarray,
    n_blend: int,
    weight_old: float = 0.5,
) -> tuple[int, float]:
    """Weighted-blend the last n_blend items in `history` with new_chunk[0:n_blend].

    The blend (`weight_old * old + (1 - weight_old) * new`) replaces the
    history's tail in-place; the remaining new_chunk items (index n_blend
    onward) are then appended. Mirrors xVLA-style 'inpainting' but applied
    here to the SHADOW history only — never the cycle-snap queue.

    Returns (n_blended, mean_l2_delta). mean_l2_delta = the average L2 norm
    between (blended action) and (raw new action) across the blended tail —
    a cheap measure of "how much did the blend move the action vs taking
    the raw new prediction." Non-zero values mean the smoothing did
    something. Zero (or NaN) means history was empty, no blending happened.

    Pure smoke-test utility. Output is never executed.
    """
    n = min(n_blend, len(history))
    diffs: list[float] = []
    for i in range(n):
        old = history[-(n - i)]
        new = new_chunk[i]
        blended = weight_old * old + (1.0 - weight_old) * new
        history[-(n - i)] = blended
        diffs.append(float(np.linalg.norm(blended - new)))
    for i in range(n, len(new_chunk)):
        history.append(np.asarray(new_chunk[i]).copy())
    return n, (float(np.mean(diffs)) if diffs else 0.0)


def _build_chunk_speed_schedule(
    actions: np.ndarray, args, past_buffer: np.ndarray | None = None,
):
    """Per-chunk speed factor with optional adaptive look-ahead / look-behind.

    Returns s_raw (K,). When --min-speed == --max-speed (flat), every entry
    equals that value and no curvature math runs. Otherwise:
    - ``n_lookahead`` pulls factors from the chunk's tail to inform earlier
      actions (slow-before-curve, bounded by chunk boundary).
    - ``n_lookbehind`` extends the window backwards using already-published
      action rows held in ``past_buffer`` (shape ``(M, >=6)``, absolute pose
      in the same frame as ``actions``). The buffer is concatenated in front
      of the chunk, the schedule is computed on the stitched array, and the
      first ``M`` factors are sliced off so the return value still has
      length ``K``. ``past_buffer=None`` (cold start) falls back to the
      centered window's edge-pad at the left boundary — fine for the very
      first chunk, less informative than feeding real history.
    """
    K = actions.shape[0]
    if args.max_speed <= 1.0 and args.min_speed <= 1.0:
        return np.ones(K, dtype=np.float64)

    M = max(0, int(getattr(args, "lookbehind", 0)))
    if M > 0 and past_buffer is not None and len(past_buffer) > 0:
        m = min(M, len(past_buffer))
        stitched = np.concatenate(
            [np.asarray(past_buffer[-m:, :6], dtype=np.float64), actions[:, :6]],
            axis=0,
        )
        offset = m
    else:
        stitched = actions[:, :6]
        offset = 0

    if args.cum_lookahead > 0:
        # Cumulative-angle path (matches the viewer's cum_lookahead slider).
        # Wins over --lookahead when both are > 0.
        sched = compute_speed_schedule_cumangle(
            stitched,
            max_speed=args.max_speed,
            min_speed=args.min_speed,
            clamp_deg=args.clamp_deg,
            cum_window=int(args.cum_lookahead),
            n_lookbehind=M,
        )
    else:
        sched = compute_speed_schedule(
            stitched,
            max_speed=args.max_speed,
            min_speed=args.min_speed,
            clamp_deg=args.clamp_deg,
            n_lookahead=args.lookahead,
            n_lookbehind=M,
        )
    return sched[offset:]


# ---------------------------------------------------------------------------
# --save-video helper: spawn the C++ crisp_video_recorder as a subprocess.
#
# Why a C++ binary and not in-process Python:
#   The Python in-process recorder (rclpy callback + cv2.VideoWriter in a
#   subprocess via mp.Queue) lost frames mid-stream under the same
#   rclpy executor / GIL contention that motivated crisp_camera_bridge.cpp.
#   The C++ recorder subscribes via rclcpp (no GIL), decompresses with
#   cv_bridge, and writes straight to disk — same architecture as the
#   camera bridge and crisp_sender.
#
# The binary lives at clearpath_remote_ws/install/tum09_custom/lib/
#   tum09_custom/crisp_video_recorder after `colcon build`. We discover it
#   the same way crisp_gym.deploy.cpp_sender does: try the known install path,
#   back to `ros2 run` if that fails.
# ---------------------------------------------------------------------------


import subprocess  # noqa: E402  (close to point of use; rest of imports at top)
