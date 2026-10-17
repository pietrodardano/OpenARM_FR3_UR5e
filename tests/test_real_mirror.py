"""
Tests for the read-only MuJoCo twin that follows the robot (src/simulation/real_mirror.py).
The leader is the real driver (OpenArmBimanual) on the fake CAN bus.
"""

import os
import sys
import unittest

import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HAS_MODEL = os.path.exists(os.path.join(REPO, "third_party/openarm/openarm_mujoco/v1/scene.xml"))


class SpyBackend:
    """Forwards attribute access to a robot and records every method called on it."""

    def __init__(self, robot):
        self._robot = robot
        self.called = []

    def __getattr__(self, name):
        attr = getattr(self._robot, name)
        if callable(attr):
            self.called.append(name)
        return attr


@unittest.skipUnless(HAS_MODEL, "official OpenArm sources not in third_party/openarm")
class TestRealRobotMirror(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from scripts.jog_joints import make_robot
        from src.planning.openarm_kinematics import OpenArmKinematics

        with open(os.path.join(REPO, "configs/task_config.yaml"), "r", encoding="utf-8") as f:
            cls.cfg = yaml.safe_load(f)
        cls.robot = make_robot(cls.cfg, "fake", False)  # real driver thread, simulated CAN bus
        cls.kin = OpenArmKinematics(cls.cfg["robot"]["mjcf_path"])

    def _mirror(self, robot):
        from scripts.run_weigh_demo import ensure_demo_scene
        from src.simulation.real_mirror import MirrorConfig, RealRobotMirror

        ensure_demo_scene(self.cfg)
        mcfg = MirrorConfig.from_config(self.cfg)
        mcfg.scene_path = os.path.join(REPO, mcfg.scene_path)
        return RealRobotMirror(robot, mcfg, objects=self.cfg["demo"]["manual_objects"])

    def test_waits_for_driver_start(self):
        from src.hardware import FakeOpenArmBus, OpenArmBimanual

        idle = OpenArmBimanual.from_config(self.cfg, bus_factory=lambda _c: FakeOpenArmBus(dt=0.004))
        self.assertFalse(self._mirror(idle).update())  # not started: no state, no exception

    def test_follows_measured_state_read_only(self):
        spy = SpyBackend(self.robot)
        mirror = self._mirror(spy)
        self.robot.start()
        try:
            q = np.radians([10.0, 5.0, -5.0, 60.0, 5.0, -10.0, 5.0])
            self.robot.move_joints({"left": q}, gripper={"left": 0.03}, duration=1.5)
            self.robot.advance(0.5)
            self.assertTrue(mirror.update())
            st = self.robot.get_state()
            for side in ("left", "right"):
                np.testing.assert_allclose(mirror.data.qpos[mirror._qadr[side]], st.arms[side].q, atol=1e-6)
            np.testing.assert_allclose(mirror.data.qpos[mirror._gadr["left"]], [st.arms["left"].gripper] * 2, atol=1e-6)
            fk = self.kin.fk({s: st.arms[s].q for s in ("left", "right")})
            np.testing.assert_allclose(mirror.tcp("left"), fk["left"][0], atol=0.003)  # leader's measured TCP
            fk_cmd = self.kin.fk({s: st.arms[s].q_setpoint for s in ("left", "right")})
            np.testing.assert_allclose(mirror.tcp("left", commanded=True), fk_cmd["left"][0], atol=0.003)
            self.assertEqual(len(mirror.overlay()), 2)  # commanded + measured TCP of the active arm
            obj = self.cfg["demo"]["manual_objects"][0]
            np.testing.assert_allclose(mirror.can_position("can_1")[:2], [obj["x"], obj["y"]])
        finally:
            self.robot.stop()
        self.assertEqual(set(spy.called), {"get_state"})  # the twin never commands the robot


if __name__ == "__main__":
    unittest.main()
