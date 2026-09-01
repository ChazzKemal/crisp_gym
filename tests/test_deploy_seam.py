"""The seam handoff must be taken from the rows that are actually emitted.

Regression test for the 2026-08-31 jitter: the carry was captured before the
method pipeline ran, so when a step reshaped the row set -- PaceSpeed truncating
to ``n_action_steps``, GripperReplicate inserting rows -- the held-back frames
came from a stretch of the plan that was then discarded. The next chunk's head
was blended toward a pose ~65 frames further down the trajectory (236 mm mean on
pickplace_cart7_v2, against 22 mm of genuine motion across the window), which the
arm executed as a jerk at every chunk boundary.

The check is end-to-end through ``run_producer_loop`` rather than on a helper,
because the bug was one of *ordering between stages* -- every stage in isolation
was correct.
"""

import queue
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation

from crisp_gym.deploy.cli import build_parser
from crisp_gym.deploy.loop import run_producer_loop
from crisp_gym.deploy.pipeline import Chunk
from crisp_gym.deploy.trace import RunRecord

ROWS = 20        # rows the "policy" returns
KEEP = 8         # rows the method keeps -- mimics PaceSpeed's n_action_steps cut
N = 2            # --blend-overlap


class _TruncateTo:
    """Stand-in for PaceSpeed: keeps the first `keep` rows, drops the rest."""

    def __init__(self, keep):
        self.keep = keep

    def __call__(self, chunk):
        return Chunk(actions=chunk.actions[: self.keep],
                     speeds=chunk.speeds[: self.keep])


class _RampSource:
    """Chunk n is the ramp n*1000 .. n*1000+ROWS-1, so any row is identifiable."""

    n_obs = 1
    n_act = ROWS

    def __init__(self):
        self.calls = 0

    def request(self, obs_buf):
        base = (self.calls + 1) * 1000.0
        self.calls += 1
        a = np.zeros((ROWS, 7), dtype=np.float64)
        a[:, 0] = base + np.arange(ROWS)      # x carries the identity
        return a

    def shutdown(self):
        pass


class _Env:
    def _get_obs(self):
        return {"observation.state": np.zeros(6, dtype=np.float32)}

    @staticmethod
    def action_to_rotation(r):
        return Rotation.from_rotvec(np.asarray(r, dtype=np.float64))


def _run(steps, *, max_chunks=2):
    args = build_parser().parse_args([])
    args.blend_overlap, args.blend_mode, args.blend_skip = N, "linear", 0
    args.max_chunks, args.dry_run, args.stride = max_chunks, True, 1
    args.gripper_slowdown_frames = 0
    args.overlap_threshold = 10_000   # no sender here; never wait to drain
    q = queue.Queue(maxsize=1024)
    rec = RunRecord(out_dir=None, run_started_at="t", duration_s=0.0,
                    n_obs=1, n_act=ROWS, chunk_count=0, stopped_by="init")
    run_producer_loop(
        env=_Env(), chunk_source=_RampSource(), q=q, args=args, rec=rec,
        dt_base=0.05, obs_schema={}, gripper_enabled=False,
        gripper_unnormalize_fn=None, obs_buf=[], last_obs=[None],
        lookbehind_buf=deque(maxlen=0), steps=steps,
    )
    return [q.get().action[0] for _ in range(q.qsize())]


def test_carry_comes_from_the_emitted_rows_not_the_discarded_tail():
    """The decisive assertion: which rows the next chunk's head blends toward."""
    x = _run([_TruncateTo(KEEP)])

    # Chunk 1: 20 rows -> step keeps 8 (1000..1007) -> hold back N -> emit 6.
    emitted_1 = KEEP - N
    assert x[:emitted_1] == [1000.0 + i for i in range(emitted_1)]

    # The held-back carry is therefore rows 1006, 1007 -- the two frames that
    # directly follow the last one sent. Chunk 2's head is the linear ramp
    # w = (j+1)/(n_blend+1) between that carry and the fresh 2000, 2001.
    got_first, got_second = x[emitted_1], x[emitted_1 + 1]
    assert got_first == float(np.float32((2 / 3) * 1006.0 + (1 / 3) * 2000.0))
    assert got_second == float(np.float32((1 / 3) * 1007.0 + (2 / 3) * 2001.0))

    # If the carry had been taken before the step (the bug), it would have been
    # rows 1018/1019 -- the far end of a plan that was never executed.
    assert got_first != float(np.float32((2 / 3) * 1018.0 + (1 / 3) * 2000.0))


def test_speeds_stay_aligned_when_the_tail_is_held_back():
    """s_raw is sliced with the chunk; a mismatch would raise in cycle-snap."""
    assert len(_run([_TruncateTo(KEEP)])) == 2 * (KEEP - N)


def test_holding_back_still_works_with_no_pipeline():
    """The inline path is unchanged: 20 rows in, 20-N emitted."""
    assert len(_run(None)) == 2 * (ROWS - N)
