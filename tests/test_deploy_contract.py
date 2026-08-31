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


def test_cpp_sender_startup_error_surfaces_the_reason(tmp_path):
    """A failed handshake must say why, not just 'exited with code 1'."""
    from crisp_gym.deploy.cpp_sender import _tail_stderr
    log = tmp_path / "e.log"
    log.write_text("noise\n\ncrisp_sender fatal: shared-memory layout version "
                   "mismatch: producer speaks v2, this binary was built for v1\n")
    out = _tail_stderr(log)
    assert "layout version mismatch" in out
    assert "v2" in out and "v1" in out


def test_tail_stderr_never_raises_on_the_failure_path(tmp_path):
    from crisp_gym.deploy.cpp_sender import _tail_stderr
    assert "unavailable" in _tail_stderr(tmp_path / "missing.log")
    empty = tmp_path / "empty.log"; empty.write_text("")
    assert "nothing" in _tail_stderr(empty)


def test_cli_defaults_are_usable_as_a_deploy_namespace():
    """Pace seeds crisp_gym's ~60 deploy flags from this parser's own defaults.

    Restating them in draccus would let the two drift, and a flag nobody thought
    about would silently get a fresh value instead of the one proven on this rig. So
    the parser is asked for its defaults and only the exposed flags are overridden --
    which only works if parse_args([]) succeeds with no arguments at all.
    """
    from crisp_gym.deploy.cli import build_parser

    args = build_parser().parse_args([])
    # every attribute the session phases and the loop read
    for name in ("env_config", "fps", "dry_run", "offline", "yes", "max_chunks",
                 "scale_kp", "max_speed", "min_speed", "clamp_deg", "lookahead",
                 "lookbehind", "cum_lookahead", "invert_gripper",
                 "gripper_slowdown_frames", "cpp_sender", "startup_delay",
                 "overlap_threshold", "stride", "kp_exp", "kd_exp", "n_act"):
        assert hasattr(args, name), f"deploy namespace is missing {name!r}"

    # the defaults a method-driven run relies on being safe
    assert args.dry_run is False and args.offline is False
    assert args.max_speed == 1.0 and args.min_speed == 1.0, "no speedup unless asked"
    assert args.scale_kp is False, "gains untouched unless asked"


def test_session_exposes_every_phase_the_runner_calls():
    """Pace's run_on_robot() calls these by name; renaming one breaks deploy."""
    from crisp_gym.deploy import session
    for phase in ("build_env", "phase_home", "phase_switch_controller",
                  "phase_scaler", "phase_pin_gripper_speed", "phase_gil_hygiene",
                  "phase_publish_channels", "phase_start_sender",
                  "phase_video_and_delay"):
        assert callable(getattr(session, phase, None)), f"session.{phase} missing"


def test_publish_channels_carries_what_the_sender_needs():
    from crisp_gym.deploy.session import PublishChannels
    ch = PublishChannels()
    for f in ("base_frame_id", "target_pose_pub", "pose_msg", "gripper_raw_pub",
              "gripper_action_client", "gripper_max_effort",
              "gripper_unnormalize_fn", "gripper_enabled"):
        assert hasattr(ch, f), f


def test_run_record_ships_the_buckets_the_loop_writes_to():
    """The loop indexes these directly; a bare {} KeyErrors on the first chunk.

    That failure lands after the robot is homed, the controller switched and the
    sender running — the most expensive point at which to discover a missing dict
    key — so the default carries them.
    """
    from pathlib import Path

    from crisp_gym.deploy.trace import PRODUCER_STAGES, RunRecord

    r = RunRecord(out_dir=Path("/tmp/x"), run_started_at="t", duration_s=0.0,
                  n_obs=1, n_act=8, chunk_count=0, stopped_by="init")
    for stage in PRODUCER_STAGES:
        assert stage in r.stage_samples_producer, f"missing bucket {stage!r}"
        r.stage_samples_producer[stage].append(1.0)   # must not raise


def test_two_records_do_not_share_buckets():
    """default_factory, not a shared default — one run's timings must not leak."""
    from pathlib import Path

    from crisp_gym.deploy.trace import RunRecord
    mk = lambda: RunRecord(out_dir=Path("/tmp/x"), run_started_at="t", duration_s=0.0,
                           n_obs=1, n_act=8, chunk_count=0, stopped_by="init")
    a, b = mk(), mk()
    a.stage_samples_producer["get_obs_ms"].append(1.0)
    assert b.stage_samples_producer["get_obs_ms"] == []
