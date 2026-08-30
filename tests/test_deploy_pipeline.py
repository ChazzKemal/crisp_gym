"""Tests for the method pipeline: Chunk, and the steps methods contribute.

The properties checked here are the ones whose violation is *silent on hardware*:
speeds drifting out of alignment with actions after a row-count change, and
GripperReplicate freezing the arm instead of slowing it.
"""

import numpy as np
import pytest

from crisp_gym.deploy.pipeline import (
    Chunk,
    DeployStep,
    GripperHold,
    GripperReplicate,
    HeuristicSpeed,
    run_pipeline,
)


def chunk(k=10, grip=None):
    a = np.zeros((k, 7))
    a[:, 0] = np.arange(k) * 0.01          # move along +x
    if grip is not None:
        a[:, 6] = grip
    return Chunk.nominal(a)


# --------------------------------------------------------------------------
# Chunk: the alignment invariant
# --------------------------------------------------------------------------

def test_nominal_is_speed_one_everywhere():
    c = chunk(5)
    np.testing.assert_array_equal(c.speeds, np.ones(5))
    assert len(c) == 5


def test_misaligned_speeds_are_rejected_at_construction():
    """The failure this prevents: every action executing at its neighbour's speed."""
    with pytest.raises(ValueError, match=r"must be \(K,\)"):
        Chunk(actions=np.zeros((10, 7)), speeds=np.ones(9))


def test_non_2d_actions_are_rejected():
    with pytest.raises(ValueError, match=r"\(K, D\)"):
        Chunk(actions=np.zeros(10), speeds=np.ones(10))


# --------------------------------------------------------------------------
# GripperReplicate -- the semantics that were explicitly corrected
# --------------------------------------------------------------------------

def test_replicate_repeats_each_row_not_one_row():
    """low_v=2 over a 3-frame ramp: g0 g0 g1 g1 g2 g2, NOT g0 g0 g0 g0 g0 g1 g2."""
    a = np.zeros((6, 7))
    a[:, 0] = np.arange(6)                      # distinguishable rows
    a[:, 6] = [0.0, 0.25, 0.5, 0.75, 1.0, 1.0]  # gripper moves over frames 1..4
    out = GripperReplicate(low_v=2)(Chunk.nominal(a))
    xs = out.actions[:, 0].tolist()
    assert xs == [0, 1, 1, 2, 2, 3, 3, 4, 4, 5], xs
    # the arm keeps advancing through the grasp rather than freezing on one pose
    assert len(set(xs)) == 6, "every original row survives; none is replaced by a hold"


def test_replicate_keeps_speeds_aligned_after_growing():
    a = np.zeros((5, 7)); a[2, 6] = 1.0
    c = Chunk(actions=a, speeds=np.linspace(1.0, 2.0, 5))
    out = GripperReplicate(low_v=3)(c)
    assert out.actions.shape[0] == out.speeds.shape[0], "alignment is the invariant"
    assert len(out) > len(c), "replication grows the chunk"


def test_replicate_grows_by_exactly_low_v_minus_one_per_moving_frame():
    a = np.zeros((8, 7)); a[4:, 6] = 1.0       # exactly one transition frame
    for low_v in (2, 3, 4):
        out = GripperReplicate(low_v=low_v)(Chunk.nominal(a))
        assert len(out) == 8 + (low_v - 1), f"low_v={low_v}"


def test_replicate_is_a_no_op_without_gripper_motion():
    a = np.zeros((7, 7)); a[:, 6] = 1.0        # constant, closed
    out = GripperReplicate(low_v=4)(Chunk.nominal(a))
    assert len(out) == 7
    np.testing.assert_array_equal(out.actions, a)


def test_replicate_low_v_one_is_identity():
    c = chunk(6, grip=np.linspace(0, 1, 6))
    assert GripperReplicate(low_v=1)(c) is c


def test_replicate_rejects_low_v_below_one():
    with pytest.raises(ValueError, match="low_v"):
        GripperReplicate(low_v=0)


# --------------------------------------------------------------------------
# GripperHold -- pays in time, so it must not touch actions
# --------------------------------------------------------------------------

def test_hold_pins_speed_and_leaves_actions_untouched():
    a = np.zeros((8, 7)); a[:, 0] = np.arange(8); a[3:, 6] = 0.0; a[:3, 6] = 1.0
    c = Chunk(actions=a, speeds=np.full(8, 2.0))
    out = GripperHold(3)(c)
    np.testing.assert_array_equal(out.actions, a), "GripperHold pays in time, not rows"
    assert len(out) == len(c), "row count must not change"
    assert (out.speeds == 1.0).any(), "the grasp window runs at nominal speed"
    assert (out.speeds == 2.0).any(), "and the rest keeps its speedup"


def test_hold_and_replicate_are_complementary():
    """Same physical intent, different currency: one writes speeds, one writes rows."""
    a = np.zeros((8, 7)); a[:3, 6] = 1.0
    c = Chunk(actions=a, speeds=np.full(8, 2.0))
    h, r = GripperHold(3)(c), GripperReplicate(3)(c)
    assert len(h) == len(c) and not np.array_equal(h.speeds, c.speeds)   # time
    assert len(r) > len(c) and np.array_equal(r.speeds[:3], c.speeds[:3])  # rows


def test_hold_zero_frames_is_a_no_op():
    c = Chunk(actions=np.zeros((5, 7)), speeds=np.full(5, 2.0))
    np.testing.assert_array_equal(GripperHold(0)(c).speeds, np.full(5, 2.0))


# --------------------------------------------------------------------------
# HeuristicSpeed and composition
# --------------------------------------------------------------------------

class Args:
    max_speed = 2.0; min_speed = 1.0; clamp_deg = 5.0
    lookahead = 0; lookbehind = 0; cum_lookahead = 0


def test_heuristic_speed_matches_the_function_it_wraps():
    """method `none` must be bit-identical to the pre-method behaviour."""
    from crisp_gym.deploy.pipeline import _build_chunk_speed_schedule
    a = np.zeros((12, 7)); a[:6, 0] = np.arange(6)*0.01; a[6:, 1] = np.arange(6)*0.01
    expected = _build_chunk_speed_schedule(a.astype(np.float64), Args(), past_buffer=None)
    np.testing.assert_array_equal(HeuristicSpeed(Args())(Chunk.nominal(a)).speeds, expected)


def test_steps_satisfy_the_protocol():
    for s in (HeuristicSpeed(Args()), GripperHold(3), GripperReplicate(2)):
        assert isinstance(s, DeployStep), type(s)


def test_pipeline_applies_steps_in_order_and_stays_aligned():
    a = np.zeros((12, 7)); a[:, 0] = np.arange(12)*0.01; a[6:, 6] = 1.0
    out = run_pipeline(Chunk.nominal(a), [HeuristicSpeed(Args()), GripperHold(3),
                                          GripperReplicate(2)])
    assert out.actions.shape[0] == out.speeds.shape[0]
    assert len(out) >= 12


def test_empty_pipeline_is_identity():
    c = chunk(5)
    assert run_pipeline(c, []) is c


# --------------------------------------------------------------------------
# The claim that lets the loop take a `steps` list at all
# --------------------------------------------------------------------------

def test_method_none_matches_the_inline_path():
    """`[HeuristicSpeed, GripperHold]` must equal what the loop does inline.

    The loop keeps its built-in schedule for the None case, so this is the property
    that makes the two interchangeable -- and therefore the property that lets a
    method-driven runner reuse the hardware-proven loop rather than fork it. Checked
    over trajectories with real gripper edges, since that is where the two paths
    could plausibly diverge.
    """
    from crisp_gym.deploy.gripper import GripperCloseWindow
    from crisp_gym.deploy.pipeline import _build_chunk_speed_schedule

    rng = np.random.default_rng(0)
    for trial in range(8):
        k = 16
        a = np.zeros((k, 7))
        a[:, :3] = np.cumsum(rng.normal(scale=0.01, size=(k, 3)), axis=0)
        a[:, 3:6] = np.cumsum(rng.normal(scale=0.01, size=(k, 3)), axis=0)
        g, col = 1.0, np.empty(k)
        for i in range(k):
            if rng.random() < 0.2:
                g = 1.0 - g
            col[i] = g
        a[:, 6] = col

        # inline, exactly as the loop does it when steps is None
        s_inline = _build_chunk_speed_schedule(a.astype(np.float64), Args(), past_buffer=None)
        mask = GripperCloseWindow(5, invert=False).mask(a)
        if mask.any():
            s_inline = s_inline.copy()
            s_inline[mask] = 1.0

        # pipeline
        out = run_pipeline(Chunk.nominal(a.astype(np.float64)),
                           [HeuristicSpeed(Args()), GripperHold(5)])

        np.testing.assert_array_equal(out.speeds, s_inline, err_msg=f"trial {trial}")
        np.testing.assert_array_equal(out.actions, a.astype(np.float64))
