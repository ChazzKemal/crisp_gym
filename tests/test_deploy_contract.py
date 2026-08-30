"""The contract crisp_gym.deploy offers an external runner.

Pace's run_real.py depends on these names and shapes. They are checked here, in
crisp_gym, because that is where a change would break them -- and the breakage would
otherwise surface only when someone tries to deploy a policy on the robot.
"""

import inspect

import numpy as np

from crisp_gym.deploy.loop import run_producer_loop
from crisp_gym.deploy.pipeline import Chunk, DeployStep, GripperHold, GripperReplicate
from crisp_gym.deploy.sources import ChunkSource
from crisp_gym.deploy.trace import RunRecord, write_run_artifacts


def test_loop_accepts_a_steps_pipeline():
    """The parameter Pace's runner passes its method's steps through."""
    p = inspect.signature(run_producer_loop).parameters
    assert "steps" in p, "run_producer_loop must accept a method pipeline"
    assert p["steps"].default is None, "None must keep the built-in inline path"


def test_loop_takes_a_run_record():
    assert "rec" in inspect.signature(run_producer_loop).parameters


def test_chunk_source_protocol_is_checkable():
    """A runner must be able to assert its source satisfies the contract."""
    class Fake:
        n_obs = 1
        n_act = 8
        def request(self, obs_buf): return np.zeros((8, 7))
        def shutdown(self): pass
    assert isinstance(Fake(), ChunkSource)


def test_incomplete_source_is_rejected():
    class Missing:
        n_obs = 1
        n_act = 8
    assert not isinstance(Missing(), ChunkSource)


def test_steps_are_structurally_interchangeable():
    """Any DeployStep can be substituted for any other by the loop."""
    a = np.zeros((6, 7)); a[3:, 6] = 1.0
    c = Chunk.nominal(a)
    for step in (GripperHold(3), GripperReplicate(2)):
        assert isinstance(step, DeployStep)
        out = step(c)
        assert isinstance(out, Chunk)
        assert out.actions.shape[0] == out.speeds.shape[0], "alignment is preserved"


def test_run_record_defaults_allow_partial_construction():
    """A runner builds the record before the loop; only identity is required."""
    from pathlib import Path
    r = RunRecord(out_dir=Path("/tmp/x"), run_started_at="t", duration_s=0.0,
                  n_obs=1, n_act=8, chunk_count=0, stopped_by="init")
    assert r.chunk_rows == [] and r.trace_records == []


def test_write_run_artifacts_signature_is_stable():
    assert list(inspect.signature(write_run_artifacts).parameters) == \
        ["rec", "args", "sender", "shadow_policy"]
