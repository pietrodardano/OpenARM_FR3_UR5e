"""Checked Cartesian / joint motion on the FakeBus driver (SR4)."""

from __future__ import annotations

import numpy as np
import pytest

from robotokens.robot.motion import Motion, Target, UnsafeMotion


@pytest.fixture
def motion(make_fake_arm, model, cfg):
    arm = make_fake_arm(q0=model.home)
    return Motion(arm, model, cfg)


def test_move_to_tracks_straight_line(motion, model, cfg):
    p0, _ = motion.tcp_command()
    target = Target(p0[0] - 0.03, p0[1] + 0.03, p0[2] - 0.03, yaw=0.0, pitch=0.0)
    motion.move_to(target)
    motion.arm.advance(0.3)
    p, _ = model.fk(motion.arm.state().q)
    np.testing.assert_allclose(p, [target.x, target.y, target.z], atol=0.006)


def test_line_through_table_is_refused_before_moving(motion, cfg):
    q_before = motion.arm.targets()[0].copy()
    p0, _ = motion.tcp_command()
    with pytest.raises(UnsafeMotion):
        motion.move_to(Target(p0[0], p0[1], cfg.table_z - 0.02, pitch=0.0))
    np.testing.assert_allclose(motion.arm.targets()[0], q_before)


def test_target_outside_workspace_is_refused(motion):
    with pytest.raises(UnsafeMotion, match="workspace"):
        motion.check_line(Target(1.5, 0.2, 0.5))


def test_route_from_rest_uses_side_route(model, cfg, make_fake_arm):
    arm = make_fake_arm()  # at rest (hanging, hand below the tabletop)
    m = Motion(arm, model, cfg)
    with pytest.raises(UnsafeMotion):
        m.check_joint_path(np.asarray(cfg.robot.rest_qpos), model.home)  # direct move sweeps the table
    chain = m.plan_route(model.home)
    assert len(chain) > 2  # via the side route
    for a, b in zip(chain[:-1], chain[1:]):
        m.check_joint_path(a, b)


def test_route_from_near_rest_inside_table_guard_settles_to_rest_first(model, cfg, make_fake_arm):
    # Arm hanging a bit forward after being moved by hand: its start pose is inside the
    # conservative table guard, so no route could leave it. It first settles onto exact rest.
    q0 = np.array([-0.05, 0.0, 0.0, 0.03, 0.0, 0.0, 0.0])
    assert model.blocked(q0) is not None
    m = Motion(make_fake_arm(q0=q0), model, cfg)
    chain = m.plan_route(model.home)
    np.testing.assert_allclose(chain[1], cfg.robot.rest_qpos)
    for a, b in zip(chain[1:-1], chain[2:]):
        m.check_joint_path(a, b)


def test_route_far_from_rest_inside_guard_is_still_refused(model, cfg, make_fake_arm):
    q0 = np.array([-0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # hand on the tabletop: not "near rest"
    assert model.blocked(q0) is not None
    with pytest.raises(UnsafeMotion):
        Motion(make_fake_arm(q0=q0), model, cfg).plan_route(model.home)
