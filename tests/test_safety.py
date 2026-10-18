"""SR1: the safety filter clips to joint limits and rate-limits every setpoint."""

from __future__ import annotations

import numpy as np
import pytest

from robotokens.robot.safety import SafetyFilter, TargetInterpolator, minimum_jerk

LOWER = np.array([-1.0, -1.0, -1.0, 0.0, -1.0, -1.0, -1.0, 0.0])
UPPER = np.array([1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 0.044])
VMAX = np.array([1.0] * 7 + [0.022])


def make_filter(margin=0.02):
    return SafetyFilter(LOWER, UPPER, VMAX, margin)


def test_step_is_rate_limited():
    f = make_filter()
    f.reset(np.array([0, 0, 0, 1.0, 0, 0, 0, 0.0]))
    sp = f.step(np.array([0.5, 0, 0, 1.0, 0, 0, 0, 0.044]), dt=0.01)
    assert sp[0] == pytest.approx(0.01)            # 1 rad/s * 10 ms
    assert sp[7] == pytest.approx(0.022 * 0.01)    # gripper speed limit


def test_target_clipped_to_limits_minus_margin():
    f = make_filter(margin=0.02)
    f.reset(np.array([0.9, 0, 0, 1.0, 0, 0, 0, 0.0]))
    for _ in range(1000):
        sp = f.step(np.array([5.0, 0, 0, 1.0, 0, 0, 0, 1.0]), dt=0.01)
    assert sp[0] == pytest.approx(0.98)            # upper 1.0 - margin
    assert sp[7] == pytest.approx(0.044)           # no margin on the gripper


def test_start_inside_margin_is_allowed_but_not_beyond():
    # Joint 4 hangs at its lower limit 0.0 (inside the 0.02 margin): it may stay / return there.
    f = make_filter(margin=0.02)
    f.reset(np.zeros(8))
    sp = f.step(np.array([0, 0, 0, -0.5, 0, 0, 0, 0.0]), dt=1.0)
    assert sp[3] == pytest.approx(0.0)


def test_step_before_reset_raises():
    with pytest.raises(RuntimeError):
        make_filter().step(np.zeros(8), 0.01)


def test_minimum_jerk_endpoints():
    assert minimum_jerk(0.0) == 0.0 and minimum_jerk(1.0) == 1.0 and minimum_jerk(0.5) == pytest.approx(0.5)


def test_interpolator_spreads_target_over_period():
    it = TargetInterpolator(dim=8)
    it.reset(np.zeros(8), now=0.0)
    it.set_goal(np.ones(8), now=0.0)                # first goal: passes as a step
    it.set_goal(2 * np.ones(8), now=0.02)           # 20 ms period
    assert it.value(0.03)[0] == pytest.approx(1.5)  # half way
    assert it.value(0.05)[0] == pytest.approx(2.0)
