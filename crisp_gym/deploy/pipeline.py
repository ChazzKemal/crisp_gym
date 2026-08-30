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

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from crisp_gym.deploy.gripper import GripperCloseWindow, GripperMotionRun
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


# ---------------------------------------------------------------------------
# The method seam: an ordered list of steps that reshape a chunk.
#
# A method contributes steps; the loop runs them and knows nothing else about it.
# The dividing line is: a step transforms the chunk's *contents*, while the loop
# owns time, queueing, and when to ask the policy for more. Seam blending and the
# speed schedule are steps; the overlap threshold and deadline anchoring are not.
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """One chunk on its way from the policy to the sender.

    ``speeds`` travels with ``actions`` rather than beside it because steps change
    the *row count* -- replication inserts rows, striding and blending remove them --
    and every such step must keep per-row data aligned. Splitting them across two
    containers makes that an invariant somebody has to remember; here it is a local
    operation on one object. An off-by-one between the two would mean every action
    executing at its neighbour's speed, silently.

    ``speeds`` is deliberately not optional. 1.0 is a positive statement -- run at
    the nominal control period -- not a placeholder for "unset", and for demospeedup
    it is a *required* value: the demonstration is already compressed in the weights,
    so also compressing time applies the speedup twice.
    """

    actions: np.ndarray          # (K, D), absolute [x, y, z, r0, r1, r2, grip]
    speeds: np.ndarray           # (K,), multiplier on the nominal control period

    def __post_init__(self):
        if self.actions.ndim != 2:
            raise ValueError(f"actions must be (K, D); got {self.actions.shape}")
        if self.speeds.shape != (self.actions.shape[0],):
            raise ValueError(
                f"speeds {self.speeds.shape} must be (K,) for actions "
                f"{self.actions.shape} -- a mismatch means actions execute at the "
                "wrong speeds rather than raising later"
            )

    @classmethod
    def nominal(cls, actions: np.ndarray) -> "Chunk":
        """A chunk at speed 1.0 throughout -- what a method receives."""
        actions = np.asarray(actions)
        return cls(actions=actions, speeds=np.ones(actions.shape[0], dtype=np.float64))

    def __len__(self) -> int:
        return self.actions.shape[0]


@runtime_checkable
class DeployStep(Protocol):
    """Anything that reshapes a chunk. Methods are lists of these."""

    def __call__(self, chunk: Chunk) -> Chunk: ...


def run_pipeline(chunk: Chunk, steps) -> Chunk:
    """Apply steps in order. The loop's entire knowledge of methods."""
    for step in steps:
        chunk = step(chunk)
    return chunk


class HeuristicSpeed:
    """The curvature schedule the deploy path has always used -- method ``none``.

    Wraps :func:`_build_chunk_speed_schedule` unchanged, so the default behaviour of
    a method-driven run is bit-identical to the pre-method one.
    """

    def __init__(self, args, past_buffer=None):
        self.args = args
        self.past_buffer = past_buffer

    def __call__(self, chunk: Chunk) -> Chunk:
        s = _build_chunk_speed_schedule(
            chunk.actions.astype(np.float64), self.args, past_buffer=self.past_buffer,
        )
        return Chunk(actions=chunk.actions, speeds=np.asarray(s, dtype=np.float64))


class GripperHold:
    """Pin speed to 1.0 near an open->close edge -- pays for the grasp in *time*.

    Used by ``none`` and ``pace``, which compress time; ``demospeedup`` compressed
    waypoints instead and so pays in rows (see :class:`GripperReplicate`). Same
    physical operation, each method's own currency.
    """

    def __init__(self, n_frames: int, *, invert: bool = False):
        self.window = GripperCloseWindow(n_frames, invert=invert)

    def __call__(self, chunk: Chunk) -> Chunk:
        mask = self.window.mask(chunk.actions)
        if not mask.any():
            return chunk
        speeds = chunk.speeds.copy()
        speeds[mask] = 1.0
        return Chunk(actions=chunk.actions, speeds=speeds)


class GripperReplicate:
    """Repeat **each** row of a gripper-motion run ``low_v`` times.

    Not one row repeated ``low_v`` times: that would freeze the arm at a single pose
    and then jump. The arm is usually still moving during a grasp, so repeating every
    row keeps it on the identical path at ``1/low_v`` pace -- which is exactly the
    inverse of the stride demospeedup applied to those frames at training time.

    ``low_v`` rather than ``high_v`` because the entropy labelling marks grasp moments
    as *precision* regions, and precision regions were strided by ``low_v``. What was
    removed there is what gets put back.

    The chunk grows. That is intentional: the inference deadline is
    ``overlap_threshold x dt_eff``, which does not contain K, so a longer chunk costs
    nothing there -- while truncating back to K would pay for gripper time by dropping
    arm motion from the tail, and would compound with the blend's own hold-back.
    """

    def __init__(self, low_v: int, *, eps: float = 1e-3):
        if low_v < 1:
            raise ValueError(f"low_v must be >= 1; got {low_v}")
        self.low_v = int(low_v)
        self.run = GripperMotionRun(eps=eps)

    def __call__(self, chunk: Chunk) -> Chunk:
        if self.low_v == 1:
            return chunk
        mask = self.run.mask(chunk.actions)
        if not mask.any():
            return chunk
        reps = np.where(mask, self.low_v, 1)
        idx = np.repeat(np.arange(len(chunk)), reps)
        return Chunk(actions=chunk.actions[idx], speeds=chunk.speeds[idx])
