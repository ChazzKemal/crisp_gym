"""Pure-numpy timing and speed-schedule math for the deploy path.

Moved verbatim out of ``examples/17_replay_dataset.py`` (the speed-schedule cluster
and ``build_speed_queue_arrays``). Nothing here touches ROS, torch, or an env -- it
is arrays in, arrays out -- which is the whole point of separating it: this is the
part of the deploy path that can be unit-tested on a laptop with no robot attached.

Consumers: ``examples/17_replay_dataset.py`` (dataset replay),
``examples/19_deploy_policy.py`` (policy deploy), and the method pipeline, which
turns a chunk into per-step speeds before the sender converts them into deadlines.
"""

import numpy as np

# CRISP cartesian controller runs at 500 Hz on the real UR10e. dt_eff for any
# replay frame must be an integer multiple of this so the controller swallows
# whole cycles -- no sub-cycle waypoints, no dropped commands.
CONTROL_DT = 0.002


# ---------------------------------------------------------------------------
# Trajectory-aware speed schedule (port of xVLA b915f95 — see plan)
#
# Math is a numpy mirror of speedup_lerobot/src/lerobot/policies/xvla/
# modeling_xvla.py L619-864. Operates on absolute pose actions of shape
# (T, >=6) where dims 0:3 = xyz and dims 3:6 = orientation (xVLA: axis-angle;
# our recording: euler-xyz radians — for typical replay rates the consecutive-
# delta math is indistinguishable). Returns a per-frame speed factor in
# [min_speed, max_speed] that ReplayScaler maps to scaled controller kp.
#
# Aggregator is fixed to xVLA's "cumulative_bending" mode (sum the per-step
# direction-change angles over the next n+1 steps, then re-derive the speed
# factor from the cumulative bend) so the robot slows down *before* curvy
# regions instead of reactively at the curve.
# ---------------------------------------------------------------------------


def _per_step_angle(traj_xyz: np.ndarray) -> np.ndarray:
    """Per-step direction-change angle in degrees on absolute positions.

    Mirror of xVLA's _per_step_angle (modeling_xvla.py L619-631). Last two
    entries replicate the previous angle (matches xVLA's edge padding).
    """
    deltas = traj_xyz[1:] - traj_xyz[:-1]                       # (T-1, 3)
    a = deltas[:-1]
    b = deltas[1:]
    a_norm = np.linalg.norm(a, axis=-1)
    b_norm = np.linalg.norm(b, axis=-1)
    denom = np.maximum(a_norm * b_norm, 1e-8)
    cos = np.sum(a * b, axis=-1) / denom                        # (T-2,)
    cos = np.clip(cos, -1.0, 1.0)
    angles = np.degrees(np.arccos(cos))                          # (T-2,)
    if angles.size == 0:
        return np.zeros(traj_xyz.shape[0], dtype=np.float64)
    # Pad to length T by replicating the last value (xVLA L628).
    return np.concatenate([angles, angles[-1:], angles[-1:]])


def _speed_from_angle_factors(
    angles_deg: np.ndarray, max_speed: float, min_speed: float,
) -> np.ndarray:
    """Per-step speed factor from direction-change angle (xVLA L633-642)."""
    factors = np.clip(90.0 - angles_deg, 0.0, None) / 90.0
    return min_speed + (max_speed - min_speed) * factors


def _speed_from_orientation_factors(
    traj_ori: np.ndarray, max_speed: float, min_speed: float, clamp_deg: float,
) -> np.ndarray:
    """Per-step speed factor from orientation-delta magnitude (xVLA L695-717).

    Below ``clamp_deg`` of rotation per step the factor stays at max_speed;
    above it, factor decays linearly toward min_speed within an additional
    ``clamp_deg`` of excess (so rotations >= 2 * clamp_deg/step give min_speed).
    """
    ori_d = traj_ori[1:] - traj_ori[:-1]                         # (T-1, 3)
    ori_d = np.concatenate([ori_d, ori_d[-1:]], axis=0) if ori_d.size else np.zeros_like(traj_ori)
    rot_degs = np.degrees(np.linalg.norm(ori_d, axis=-1))         # (T,)
    if clamp_deg is None or clamp_deg <= 0:
        return np.full_like(rot_degs, float(max_speed))
    excess = np.clip(rot_degs - clamp_deg, 0.0, None)
    factors = np.clip((clamp_deg - excess) / clamp_deg, 0.0, 1.0)
    return min_speed + (max_speed - min_speed) * factors


def _forward_window_sum(values: np.ndarray, n: int) -> np.ndarray:
    """Forward-window sum of length n+1 with edge replication.

    Narrowed port of xVLA's _apply_lookahead (modeling_xvla.py L670-693) to
    agg="sum". For each timestep t, returns sum(values[t..t+n]) where the
    chunk tail is edge-padded so it isn't dragged down by missing future steps.
    """
    if n <= 0:
        return values.astype(np.float64, copy=True)
    shifted = [values.astype(np.float64, copy=True)]
    for k in range(1, n + 1):
        s = np.empty_like(values, dtype=np.float64)
        s[:-k] = values[k:]
        s[-k:] = values[-1]                                       # edge-pad
        shifted.append(s)
    return np.stack(shifted, axis=0).sum(axis=0)                  # (T,)


def _centered_window_sum(
    values: np.ndarray, n_past: int, n_future: int,
) -> np.ndarray:
    """Centered-window sum of length n_past + n_future + 1, edge-padded both ends.

    For each timestep t, returns sum(values[t - n_past .. t + n_future]) with
    edge replication outside the array. n_past=0 reduces exactly to
    _forward_window_sum(values, n_future); n_future=0 is the time-reversed
    counterpart. Used by compute_speed_schedule to symmetrize the
    lookahead window so the arm stays slow on the EXIT of a curve, not just
    the entry.
    """
    n_past = max(0, int(n_past))
    n_future = max(0, int(n_future))
    if n_past == 0 and n_future == 0:
        return values.astype(np.float64, copy=True)
    shifted = [values.astype(np.float64, copy=True)]
    for k in range(1, n_future + 1):
        s = np.empty_like(values, dtype=np.float64)
        s[:-k] = values[k:]
        s[-k:] = values[-1]                                       # edge-pad right
        shifted.append(s)
    for k in range(1, n_past + 1):
        s = np.empty_like(values, dtype=np.float64)
        s[k:] = values[:-k]
        s[:k] = values[0]                                         # edge-pad left
        shifted.append(s)
    return np.stack(shifted, axis=0).sum(axis=0)                  # (T,)


def compute_speed_schedule(
    actions: np.ndarray,
    *,
    max_speed: float,
    min_speed: float = 1.0,
    clamp_deg: float = 5.0,
    n_lookahead: int = 0,
    n_lookbehind: int = 0,
) -> np.ndarray:
    """Per-frame speed factor in [min_speed, max_speed] for an absolute trajectory.

    ``actions`` is shape ``(T, >=6)`` with dims 0:3 = xyz and dims 3:6 =
    orientation. Combines speed_from_angle (positions) with speed_from_
    orientation (rotation magnitude) via element-wise min, mirroring xVLA
    select_action (L770-837). When ``n_lookahead > 0`` the angle channel is
    replaced by the cumulative_bending factor over the next ``n_lookahead+1``
    steps (xVLA L811-824); the orientation channel is left raw because
    rotation magnitude has no per-step bending interpretation (L825-827).
    ``n_lookbehind > 0`` extends the window symmetrically backwards so the
    arm stays slow on the EXIT of a curve, not just the entry — the window
    length becomes ``n_lookbehind + n_lookahead + 1`` and the normalization
    grows in lockstep.

    Designed to be called once per recorded episode (replay) or once per
    generated chunk (future live policy).
    """
    if actions.ndim != 2 or actions.shape[1] < 6:
        raise ValueError(
            f"compute_speed_schedule expects (T, >=6); got shape {actions.shape}"
        )
    if max_speed < min_speed:
        raise ValueError(
            f"max_speed ({max_speed}) must be >= min_speed ({min_speed})"
        )
    actions = np.asarray(actions, dtype=np.float64)
    pos = actions[:, :3]
    ori = actions[:, 3:6]

    angles = _per_step_angle(pos)
    speeds_ori = _speed_from_orientation_factors(ori, max_speed, min_speed, clamp_deg)

    if n_lookahead > 0 or n_lookbehind > 0:
        # cumulative_bending on the angle channel, symmetric window.
        cum = _centered_window_sum(angles, n_lookbehind, n_lookahead)
        denom = 90.0 * (n_lookbehind + n_lookahead + 1)
        factors = np.clip(denom - cum, 0.0, None) / denom
        speeds_coord = min_speed + (max_speed - min_speed) * factors
    else:
        speeds_coord = _speed_from_angle_factors(angles, max_speed, min_speed)

    return np.minimum(speeds_coord, speeds_ori)


def compute_speed_schedule_cumangle(
    actions: np.ndarray,
    *,
    max_speed: float,
    min_speed: float = 1.0,
    clamp_deg: float = 5.0,
    cum_window: int = 0,
    n_lookbehind: int = 0,
) -> np.ndarray:
    """Cumulative-angle variant: factor = clip(90 - cum, 0) / 90.

    Differs from ``compute_speed_schedule(n_lookahead=N)`` only in the
    normalization. The existing formula divides the cumulative angle by
    ``90 * (N+1)`` so AVERAGE angle >= 90 deg drives the factor to zero.
    This variant divides by 90 only — CUMULATIVE angle >= 90 deg drives
    it to zero. Net effect: a longer window slows down much more
    aggressively as small bends add up, instead of being averaged away.

    Useful for chunks (policy deploy) where many sub-threshold direction
    changes within the window collectively warrant a slowdown that the
    averaging-lookahead path under-weights.

    ``n_lookbehind > 0`` extends the cumulative window symmetrically into
    the past so already-executed bends keep weighing on the current speed
    factor; the denominator stays 90 by design (cumulative threshold, not
    average).
    """
    if actions.ndim != 2 or actions.shape[1] < 6:
        raise ValueError(
            f"compute_speed_schedule_cumangle expects (T, >=6); "
            f"got shape {actions.shape}"
        )
    if max_speed < min_speed:
        raise ValueError(
            f"max_speed ({max_speed}) must be >= min_speed ({min_speed})"
        )
    actions = np.asarray(actions, dtype=np.float64)
    pos = actions[:, :3]
    ori = actions[:, 3:6]

    angles = _per_step_angle(pos)
    speeds_ori = _speed_from_orientation_factors(
        ori, max_speed, min_speed, clamp_deg,
    )

    cum = _centered_window_sum(
        angles, max(0, int(n_lookbehind)), max(0, int(cum_window)),
    )
    factors = np.clip(90.0 - cum, 0.0, None) / 90.0
    speeds_coord = min_speed + (max_speed - min_speed) * factors

    return np.minimum(speeds_coord, speeds_ori)


def compute_speed_schedule_drop_holds(
    actions: np.ndarray,
    *,
    max_speed: float,
    min_speed: float = 1.0,
    clamp_deg: float = 5.0,
    n_lookahead: int = 0,
    n_lookbehind: int = 0,
    cum_window: int = 0,
    motion_eps: float = 1e-6,
) -> np.ndarray:
    """Like ``compute_speed_schedule`` but ignore zero-motion (held) frames.

    A frame is "held" iff ``||pos[i] - pos[i-1]|| <= motion_eps``. Held
    frames inject a spurious 90 deg fallback into ``_per_step_angle``
    (``cos = 0 / eps``), which pins the position channel to ``min_speed``
    at every transition in / out of a hold. For teleop-recorded datasets
    that artifact dominates the schedule.

    Strategy: filter actions down to moving frames only, run the chosen
    schedule fn (cumangle when ``cum_window > 0``, else
    ``compute_speed_schedule``) on the moving sub-trajectory, then
    broadcast each moving-frame speed back so each held frame inherits
    the speed of the next moving frame at-or-after it (conservative —
    matches the upcoming motion rather than always using max_speed).
    """
    if actions.ndim != 2 or actions.shape[1] < 6:
        raise ValueError(
            f"compute_speed_schedule_drop_holds expects (T, >=6); "
            f"got shape {actions.shape}"
        )
    actions = np.asarray(actions, dtype=np.float64)
    n = len(actions)
    if n == 0:
        return np.array([], dtype=np.float64)

    step = np.linalg.norm(np.diff(actions[:, :3], axis=0), axis=-1)
    is_moving = np.empty(n, dtype=bool)
    is_moving[0] = True  # always anchor on the first frame
    is_moving[1:] = step > float(motion_eps)

    # _per_step_angle needs >= 3 moving frames to produce a single real angle.
    if int(is_moving.sum()) < 3:
        return np.full(n, float(max_speed), dtype=np.float64)

    moving_idx = np.where(is_moving)[0]
    moving_actions = actions[moving_idx]

    if int(cum_window) > 0:
        s_mov = compute_speed_schedule_cumangle(
            moving_actions,
            max_speed=max_speed,
            min_speed=min_speed,
            clamp_deg=clamp_deg,
            cum_window=int(cum_window),
            n_lookbehind=int(n_lookbehind),
        )
    else:
        s_mov = compute_speed_schedule(
            moving_actions,
            max_speed=max_speed,
            min_speed=min_speed,
            clamp_deg=clamp_deg,
            n_lookahead=int(n_lookahead),
            n_lookbehind=int(n_lookbehind),
        )

    # For each original i, find the first moving index >= i. Held frames
    # bracketed by two motions inherit the speed of the upcoming motion.
    target = np.searchsorted(moving_idx, np.arange(n), side="left")
    target = np.minimum(target, len(moving_idx) - 1)
    return s_mov[target]



# ---------------------------------------------------------------------------
# Speed queue + producer/consumer target sender
#
# All per-frame numbers are derived once at startup from compute_speed_schedule.
# The dataset producer pre-fills a bounded queue with TargetItem(s); a dedicated
# TargetSenderThread pops them at item.deadline_mono and publishes to /target_pose
# (+ optional gripper). rclpy publishes release the GIL inside the C extension,
# so the sender thread does not freeze main during DDS work, and policy-driven
# producers (future work) can stall arbitrarily without disturbing publish cadence.
#
# Cycle-snap quantization: dt_eff is forced to an integer multiple of CONTROL_DT
# (500 Hz CRISP controller cycle). The back-computed s_eff = dt_base / dt_eff
# is the single scalar that drives all four scaling channels (time, kp, kd,
# gripper speed) — consistent everywhere. See docs/variable_impedance_design.md
# and the plan file.
# ---------------------------------------------------------------------------


def build_speed_queue_arrays(
    s_raw: np.ndarray | None,
    dt_base: float,
    n_frames: int,
    *,
    retime: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pre-compute (cycles, dt_eff, s_eff) for the whole episode.

    s_raw    : (T,) per-frame factor from compute_speed_schedule, or None.
    dt_base  : base inter-frame period in seconds (1 / (fps * args.speed)).
    retime   : if True and s_raw is provided, time-warp via cycle-snap; if
               False, time stays uniform at dt_base regardless of s_raw.

    Returns:
        cycles : (T,) int, number of CONTROL_DT cycles per frame (>=1).
        dt_eff : (T,) float, effective inter-frame period.
        s_eff  : (T,) float, effective speed driving the kp/kd/gripper knobs.
                 When retime is on and s_raw is set, s_eff = dt_base / dt_eff
                 (always <= s_raw — conservative ceil snap). When retime is
                 off but s_raw is set, s_eff = s_raw (gains still scale; time
                 does not). When s_raw is None, s_eff = 1.0 everywhere.
    """
    base_cycles = max(1, int(round(dt_base / CONTROL_DT)))
    if s_raw is None:
        cycles = np.full(n_frames, base_cycles, dtype=np.int64)
        dt_eff = cycles.astype(np.float64) * CONTROL_DT
        s_eff = np.ones(n_frames, dtype=np.float64)
        return cycles, dt_eff, s_eff

    s_raw_arr = np.asarray(s_raw, dtype=np.float64)
    if not retime:
        cycles = np.full(n_frames, base_cycles, dtype=np.int64)
        dt_eff = cycles.astype(np.float64) * CONTROL_DT
        # Gains still scale with the raw factor; time stays uniform.
        s_eff = s_raw_arr.copy()
        return cycles, dt_eff, s_eff

    dt_raw = dt_base / s_raw_arr
    cycles = np.maximum(1, np.ceil(dt_raw / CONTROL_DT)).astype(np.int64)
    dt_eff = cycles.astype(np.float64) * CONTROL_DT
    s_eff = dt_base / dt_eff
    return cycles, dt_eff, s_eff


# ---------------------------------------------------------------------------
# Producer loop
# ---------------------------------------------------------------------------


def _pre_compute_chunk_arrays(
    chunk: np.ndarray,
    *,
    args,
    gripper_enabled: bool,
    gripper_unnormalize_fn,
    rotation_from_action,
):
    """Same conversions as DatasetProducer._build_arrays, but for a live chunk.

    Returns (target_xyz, target_quat, grip_raw, actions_f32) all length K.
    Action convention (matches recorded datasets): [x, y, z, <rot>, grip]
    with grip in [0, 1] (crisp_py: 1=open, 0=closed). The grip channel is
    binarized (snapped to 0.0 / 1.0 at the 0.5 midpoint) before
    unnormalization so deployment never commands a partial grip.

    ``rotation_from_action`` maps the action's rotation slots (``action[3:6]``)
    to a scipy Rotation. It is ``env.action_to_rotation``, so the orientation
    representation is read from the env config rather than hardcoded — a
    policy trained on angle-axis just needs the env yaml set to
    ``orientation_representation: "angle_axis"``.

    NOTE: this assumes a 3-element rotation (euler OR angle_axis), so the
    action layout is [x, y, z, r0, r1, r2, grip] (7 dims). The QUATERNION
    representation has a 4-element rotation (8-dim action, grip at index 7);
    supporting it here would need the rotation slice + gripper index widened.
    Not handled because no quaternion-action policy exists in this repo yet.
    """
    K = chunk.shape[0]
    actions = chunk.astype(np.float64, copy=False)
    target_xyz = actions[:, :3].copy()
    target_quat = np.zeros((K, 4), dtype=np.float64)
    grip_raw = np.zeros(K, dtype=np.float64)
    actions_f32 = actions.astype(np.float32)
    for k in range(K):
        target_quat[k] = rotation_from_action(actions[k, 3:6]).as_quat()
        if gripper_enabled and gripper_unnormalize_fn is not None:
            g = float(np.clip(actions[k, 6], 0.0, 1.0))
            if args.invert_gripper:
                g = 1.0 - g
            # Binarize the gripper command: the policy's continuous output is
            # snapped to fully open / fully closed so deployment never holds a
            # partial grip. Threshold is the 0.5 midpoint of the [0, 1] range;
            # g >= 0.5 -> 1.0 (open), else 0.0 (closed). Applied after
            # --invert-gripper so the open/close direction stays correct.
            g = 1.0 if g >= 0.5 else 0.0
            grip_raw[k] = float(gripper_unnormalize_fn(g))
    return target_xyz, target_quat, grip_raw, actions_f32
