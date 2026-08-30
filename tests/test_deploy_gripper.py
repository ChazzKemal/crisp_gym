"""Differential tests: the extracted detectors vs the original inline algorithm.

The gripper close-window logic lived inline in 19_deploy_policy.py's producer loop,
carrying its state (`prev_grip_closed`, `close_slow_remaining`) in loop locals. That
code was proven on hardware, so the only question worth asking of the extraction is
whether it is *the same function*. These tests answer that directly: the original is
reproduced verbatim below as a reference, and both are run over randomised chunk
sequences -- including sequences engineered to straddle chunk boundaries, which is
where a naive extraction breaks and where the failure is silent (the window simply
stops at the seam and the gripper gets less time than intended).
"""

import numpy as np
import pytest

from crisp_gym.deploy.gripper import GripperCloseWindow, GripperMotionRun


def reference_close_window(chunks, n_frames, invert=False):
    """The original inline algorithm, verbatim from 19_deploy_policy.py's loop.

    State lives in the enclosing scope exactly as it did in main().
    """
    close_slow_remaining = 0
    prev_grip_closed = None
    out = []
    for chunk in chunks:
        K = chunk.shape[0]
        slow_mask = np.zeros(K, dtype=bool)
        if n_frames > 0:
            g_norm = np.clip(chunk[:, 6], 0.0, 1.0)
            if invert:
                g_norm = 1.0 - g_norm
            closed = g_norm < 0.5
            if close_slow_remaining > 0:
                c = min(close_slow_remaining, K)
                slow_mask[:c] = True
                close_slow_remaining -= c
            was_closed = bool(prev_grip_closed) if prev_grip_closed is not None else False
            for i in range(K):
                if closed[i] and not was_closed:
                    end = i + n_frames
                    slow_mask[i:min(end, K)] = True
                    if end > K:
                        close_slow_remaining = max(close_slow_remaining, end - K)
                was_closed = bool(closed[i])
            prev_grip_closed = bool(closed[-1])
        out.append(slow_mask)
    return out


def make_chunks(rng, n_chunks, k, p_flip=0.15):
    """A sequence of chunks whose gripper channel flips at random frames."""
    chunks, g = [], 0.0
    for _ in range(n_chunks):
        a = rng.normal(size=(k, 7))
        col = np.empty(k)
        for i in range(k):
            if rng.random() < p_flip:
                g = 1.0 - g
            col[i] = g
        a[:, 6] = col
        chunks.append(a)
    return chunks


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("n_frames,k", [(1, 8), (3, 8), (5, 8), (12, 8), (5, 3), (20, 5)])
def test_close_window_matches_the_original_exactly(seed, n_frames, k):
    """Extraction is faithful iff it is the same function on every input."""
    rng = np.random.default_rng(seed)
    chunks = make_chunks(rng, n_chunks=6, k=k)
    expected = reference_close_window(chunks, n_frames)
    det = GripperCloseWindow(n_frames)
    for i, (c, exp) in enumerate(zip(chunks, expected)):
        np.testing.assert_array_equal(det.mask(c), exp, err_msg=f"chunk {i}")


@pytest.mark.parametrize("seed", range(6))
def test_close_window_matches_with_inverted_gripper(seed):
    rng = np.random.default_rng(seed)
    chunks = make_chunks(rng, n_chunks=5, k=7)
    expected = reference_close_window(chunks, 4, invert=True)
    det = GripperCloseWindow(4, invert=True)
    for c, exp in zip(chunks, expected):
        np.testing.assert_array_equal(det.mask(c), exp)


def test_window_survives_a_chunk_boundary():
    """The property a stateless extraction would silently lose."""
    a = np.zeros((4, 7)); a[3, 6] = 0.0; a[:3, 6] = 1.0   # open,open,open,close at last frame
    b = np.ones((4, 7))                                    # next chunk: stays "open" valued
    det = GripperCloseWindow(3)
    m1, m2 = det.mask(a), det.mask(b)
    assert m1[3], "the close edge itself must be in the window"
    assert m2[0] and m2[1], "the remaining 2 frames must carry into the next chunk"
    assert not m2[2], "and stop after exactly n_frames total"


def test_edge_exactly_on_the_seam_is_detected():
    """Needs the previous chunk's final level; a fresh detector would miss it."""
    a = np.ones((3, 7))          # all "open" (g=1 -> closed=False)
    b = np.zeros((3, 7))         # all "closed" -> edge at frame 0 of chunk b
    det = GripperCloseWindow(2)
    det.mask(a)
    assert det.mask(b)[0], "open->close across the seam is an edge"


def test_staying_closed_fires_nothing():
    """Edge-triggered, not level-triggered: the carry keeps its speedup."""
    det = GripperCloseWindow(3)
    det.mask(np.zeros((5, 7)))                  # first chunk: edge at frame 0
    assert not det.mask(np.zeros((5, 7))).any(), "no new edge while it stays closed"


def test_zero_frames_is_a_no_op():
    det = GripperCloseWindow(0)
    assert not det.mask(np.zeros((6, 7))).any()


# --------------------------------------------------------------------------
# GripperMotionRun
# --------------------------------------------------------------------------

def test_motion_run_marks_only_the_transition():
    a = np.zeros((6, 7)); a[3:, 6] = 1.0        # steps at frame 3
    m = GripperMotionRun().mask(a)
    assert m.tolist() == [False, False, False, True, False, False]


def test_motion_run_does_not_cover_the_settled_carry():
    """The failure mode of an 'agrees with the new value' definition."""
    a = np.zeros((10, 7)); a[2:, 6] = 1.0       # closes at 2, stays closed
    m = GripperMotionRun().mask(a)
    assert m.sum() == 1, "only the change is motion; the carry is not"


def test_motion_run_spans_a_ramp():
    a = np.zeros((6, 7)); a[:, 6] = [0.0, 0.25, 0.5, 0.75, 1.0, 1.0]
    assert GripperMotionRun().mask(a).sum() == 4, "each changing frame counts"


def test_motion_run_detects_a_change_across_the_seam():
    det = GripperMotionRun()
    det.mask(np.zeros((3, 7)))
    b = np.ones((3, 7))
    assert det.mask(b)[0], "frame 0 changed relative to the previous chunk's last frame"
