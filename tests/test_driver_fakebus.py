"""SR2 / SR3 / SR5 on the real 250 Hz driver thread, with a FakeBus instead of CAN."""

from __future__ import annotations

import time

import numpy as np
import pytest

from robotokens.robot import make_arm


def wait_until(cond, timeout=2.0):
    t_end = time.monotonic() + timeout
    while time.monotonic() < t_end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_start_holds_measured_pose(make_fake_arm):
    q0 = np.array([0.1, -0.2, 0.0, 0.5, 0.0, 0.0, 0.0])
    arm = make_fake_arm(q0=q0)
    st = arm.state()
    np.testing.assert_allclose(st.q_setpoint, q0, atol=1e-9)
    assert st.fault is None


def test_refuses_start_outside_limits(cfg, model):  # SR2
    arm = make_arm(cfg, fake=True, q0=np.array([0, 0, 0, -0.5, 0, 0, 0]))  # joint 4 limit is [0, 2.44]
    with pytest.raises(RuntimeError, match="outside"):
        arm.start()


def test_refuses_start_when_motor_silent(cfg, model):  # SR2
    arm = make_arm(cfg, fake=True, silent=True)
    with pytest.raises(RuntimeError, match="did not reply"):
        arm.start()


def test_move_joints_reaches_target(make_fake_arm):
    arm = make_fake_arm()
    goal = np.array([0.0, -0.3, 0.0, 0.6, 0.0, 0.0, 0.0])
    arm.move_joints(goal, duration=0.8)
    arm.advance(0.3)
    np.testing.assert_allclose(arm.state().q, goal, atol=0.01)


@pytest.mark.parametrize("fault", ["motor_error", "silent", "bus", "tracking"])
def test_faults_latch_and_freeze(make_fake_arm, fault):  # SR3
    arm = make_fake_arm()
    bus = arm.bus
    if fault == "motor_error":
        bus.error_codes[2] = 0xA
    elif fault == "silent":
        bus.silent = True
    elif fault == "bus":
        bus.healthy = False
    else:
        bus.frozen = True  # motors stop following: sustained tracking error
        arm.set_targets(q=np.array([0.0, -0.9, 0.0, 0.9, 0.0, 0.0, 0.0]))
    assert wait_until(lambda: arm.fault is not None, timeout=3.0)
    held = arm.state().q_setpoint.copy()
    arm.set_targets(q=np.array([0.0, -0.5, 0.0, 0.5, 0.0, 0.0, 0.0]))  # ignored while faulted
    time.sleep(0.1)
    np.testing.assert_allclose(arm.state().q_setpoint, held, atol=1e-9)
    with pytest.raises(RuntimeError):
        arm.move_joints(np.zeros(7), duration=0.2)


def test_gripper_torque_is_capped_and_ramped(make_fake_arm, cfg):  # SR5
    arm = make_fake_arm()
    arm.set_gripper_torque(5.0)  # above max_torque_nm
    time.sleep(0.05)
    early = arm.state().gripper_torque_cmd
    assert 0 < early < cfg.gripper.max_torque_nm  # ramping (4 Nm/s)
    assert wait_until(lambda: arm.state().gripper_torque_cmd == pytest.approx(cfg.gripper.max_torque_nm))
    arm.set_gripper_torque(None)
    assert wait_until(lambda: arm.state().gripper_torque_cmd is None)


def test_stop_moves_to_rest_then_disables(cfg, model):
    arm = make_arm(cfg, fake=True, q0=np.array([0.0, -0.2, 0.0, 0.4, 0.0, 0.0, 0.0]))
    arm.start()
    arm.stop()
    # Joint 4 started outside the limit margin, so it returns only to the margin edge (SR1).
    np.testing.assert_allclose(arm.bus.q, cfg.robot.rest_qpos, atol=cfg.hardware.joint_limit_margin_rad + 1e-3)
    assert not arm.bus.enabled
