"""FR7: the MuJoCo twin follows the robot's measured state and never commands it."""

from __future__ import annotations

import numpy as np

from robotokens.twin.mirror import Twin


class SpyArm:
    """Records every method call; serves a fixed state."""

    def __init__(self, state):
        self._state = state
        self.calls = []

    def __getattr__(self, name):
        def method(*args, **kwargs):
            self.calls.append(name)
            if name == "state":
                return self._state
            raise AssertionError(f"twin called arm.{name}()")

        return method


def test_twin_mirrors_measured_joints(make_fake_arm, model, cfg):
    arm = make_fake_arm(q0=model.home)
    twin = Twin(cfg, arm, objects=[(0.21, 0.13), (0.21, 0.27)])
    assert twin.update()
    p_twin = twin.tcp()
    p_fk, _ = model.fk(arm.state().q)
    np.testing.assert_allclose(p_twin, p_fk, atol=1e-6)
    np.testing.assert_allclose(twin.object_position(0)[:2], [0.21, 0.13])


def test_twin_only_reads_state(make_fake_arm, model, cfg):
    arm = make_fake_arm(q0=model.home)
    spy = SpyArm(arm.state())
    twin = Twin(cfg, spy, objects=[(0.21, 0.13)])
    for _ in range(3):
        twin.update()
    assert set(spy.calls) == {"state"}


def test_held_object_follows_the_hand(make_fake_arm, model, cfg):
    from robotokens.robot.ik import IK

    # Hand at the grasp pose of an object at (0.25, 0.20), fingers open.
    p = np.array([0.25, 0.20, cfg.table_z + cfg.task.grasp.height_above_table_m])
    q = IK(model).solve(p, model.hand_rotation(0.0, np.radians(cfg.task.grasp.pitch_deg)), model.home, 300).q
    arm = make_fake_arm(q0=q)
    arm.set_targets(gripper=cfg.gripper.open_m)
    arm.advance(2.2)
    twin = Twin(cfg, arm, objects=[(0.25, 0.20)])
    twin.update()
    obj0 = twin.object_position(0).copy()
    # Squeeze in torque mode (fingers stop on the object), then lift the forearm.
    arm.bus.gripper_block_at = arm.gripper_to_motor(0.033)
    arm.set_gripper_torque(0.5)
    arm.advance(0.5)
    twin.update()
    q_up = q.copy()
    q_up[3] -= 0.3
    arm.move_joints(q_up, duration=0.6)
    arm.advance(0.2)
    twin.update()
    assert np.linalg.norm(twin.object_position(0) - obj0) > 0.02
    # Released: the object stays where it was let go.
    arm.set_gripper_torque(None)
    arm.advance(0.1)
    twin.update()
    left_at = twin.object_position(0).copy()
    arm.move_joints(q, duration=0.6)
    twin.update()
    np.testing.assert_allclose(twin.object_position(0), left_at)
