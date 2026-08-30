"""Unit tests for crisp_gym.deploy.timing.

These are the first tests the deploy path has ever had. They were impossible
before the extraction: the code lived inside ``examples/17_replay_dataset.py``,
whose leading digit makes it unimportable, and which pulls in rclpy at module
scope. Now it is arrays in, arrays out, so the invariants that actually matter
on hardware can be checked on a laptop.

The most important of those is the cycle-snap invariant: the CRISP cartesian
controller runs at 500 Hz, so every dt_eff must be an integer multiple of
CONTROL_DT. A fractional dt means the controller either drops a command or
swallows a partial cycle, which on the robot shows up as jitter, not an error.
"""

import numpy as np
import pytest

from crisp_gym.deploy.timing import (
    CONTROL_DT,
    build_speed_queue_arrays,
    compute_speed_schedule,
    compute_speed_schedule_cumangle,
)


def straight_line(n=20, step=0.01):
    """Constant-velocity path along +x, zero rotation: no bending anywhere."""
    a = np.zeros((n, 6))
    a[:, 0] = np.arange(n) * step
    return a


def right_angle(n=20, step=0.01):
    """Constant speed along +x, then a hard 90 deg turn onto +y at the midpoint."""
    a = np.zeros((n, 6))
    half = n // 2
    a[:half, 0] = np.arange(half) * step
    a[half:, 0] = (half - 1) * step
    a[half:, 1] = np.arange(n - half) * step
    return a


# --------------------------------------------------------------------------
# build_speed_queue_arrays -- the cycle-snap invariant
# --------------------------------------------------------------------------

@pytest.mark.parametrize("dt_base", [0.05, 1 / 30, 0.1])
@pytest.mark.parametrize("speed", [0.5, 1.0, 1.7, 2.0, 3.3])
def test_dt_eff_is_always_an_integer_number_of_control_cycles(dt_base, speed):
    n = 12
    s_raw = np.full(n, speed)
    cycles, dt_eff, s_eff = build_speed_queue_arrays(s_raw, dt_base, n, retime=True)

    assert cycles.shape == dt_eff.shape == s_eff.shape == (n,)
    assert np.all(cycles >= 1), "a frame may never occupy zero controller cycles"
    assert cycles.dtype.kind in "iu", "cycles must be integral"
    # the invariant the controller depends on
    np.testing.assert_allclose(dt_eff, cycles * CONTROL_DT, rtol=0, atol=1e-15)
    # s_eff is the speed actually achieved after snapping, not the one requested
    np.testing.assert_allclose(s_eff, dt_base / dt_eff, rtol=1e-12)


def test_speed_one_reproduces_the_base_period():
    n = 8
    dt_base = 0.05
    _, dt_eff, s_eff = build_speed_queue_arrays(np.ones(n), dt_base, n, retime=True)
    np.testing.assert_allclose(dt_eff, dt_base, rtol=1e-12)
    np.testing.assert_allclose(s_eff, 1.0, rtol=1e-12)


def test_faster_requested_speed_never_yields_a_longer_frame():
    dt_base, n = 0.05, 10
    _, slow, _ = build_speed_queue_arrays(np.full(n, 1.0), dt_base, n, retime=True)
    _, fast, _ = build_speed_queue_arrays(np.full(n, 2.0), dt_base, n, retime=True)
    assert np.all(fast <= slow)


def test_none_schedule_is_uniform_base_rate():
    n, dt_base = 6, 0.05
    cycles, dt_eff, s_eff = build_speed_queue_arrays(None, dt_base, n, retime=True)
    assert len(dt_eff) == n
    np.testing.assert_allclose(dt_eff, dt_base, rtol=1e-12)
    np.testing.assert_allclose(s_eff, 1.0, rtol=1e-12)


# --------------------------------------------------------------------------
# compute_speed_schedule -- bounds and the shape it is defined on
# --------------------------------------------------------------------------

@pytest.mark.parametrize("traj", [straight_line(), right_angle()])
@pytest.mark.parametrize("lo,hi", [(1.0, 1.0), (1.0, 2.0), (0.5, 3.0)])
def test_schedule_stays_within_its_bounds(traj, lo, hi):
    s = compute_speed_schedule(traj, max_speed=hi, min_speed=lo)
    assert s.shape == (len(traj),)
    assert np.all(s >= lo - 1e-12) and np.all(s <= hi + 1e-12)


def test_flat_bounds_give_a_flat_schedule():
    s = compute_speed_schedule(right_angle(), max_speed=1.0, min_speed=1.0)
    np.testing.assert_allclose(s, 1.0, rtol=1e-12)


def test_a_corner_is_taken_slower_than_a_straight_line():
    kw = dict(max_speed=2.0, min_speed=1.0)
    assert compute_speed_schedule(right_angle(), **kw).min() < \
           compute_speed_schedule(straight_line(), **kw).min()


def test_lookahead_brakes_earlier_and_more_gradually():
    """The point of lookahead: brake *before* the bend, spread over the approach.

    Without it the schedule holds max_speed right up to the corner and then drops
    in a single frame; with it the deceleration starts several frames earlier. Note
    this *raises* the minimum (braking is distributed rather than concentrated), so
    "slower" is the wrong thing to assert -- "earlier and smoother" is the property.
    """
    traj = right_angle(n=24)
    kw = dict(max_speed=2.0, min_speed=1.0)
    plain = compute_speed_schedule(traj, **kw)
    ahead = compute_speed_schedule(traj, n_lookahead=4, **kw)

    first_below = lambda s: int(np.argmax(s < kw["max_speed"] - 1e-9))
    assert first_below(ahead) < first_below(plain), "lookahead must start braking sooner"

    biggest_drop = lambda s: float(np.max(-np.diff(s), initial=0.0))
    assert biggest_drop(ahead) < biggest_drop(plain), "lookahead must smooth the drop"


def test_wrong_shape_is_rejected():
    with pytest.raises(ValueError, match=r"\(T, >=6\)"):
        compute_speed_schedule(np.zeros((10, 3)), max_speed=2.0)
    with pytest.raises(ValueError, match=r"\(T, >=6\)"):
        compute_speed_schedule(np.zeros(10), max_speed=2.0)


def test_inverted_bounds_are_rejected():
    with pytest.raises(ValueError, match="must be >="):
        compute_speed_schedule(straight_line(), max_speed=1.0, min_speed=2.0)


def test_cumangle_variant_also_respects_bounds():
    s = compute_speed_schedule_cumangle(
        right_angle(), max_speed=2.5, min_speed=1.0, cum_window=3,
    )
    assert np.all(s >= 1.0 - 1e-12) and np.all(s <= 2.5 + 1e-12)
