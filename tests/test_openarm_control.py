"""
Tests for the OpenArm deployment interface.

- Safety filter: pure numpy.
- Hardware driver: the real threaded control loop against FakeOpenArmBus (no CAN needed).
- Controllers: joint / IK control on the official MuJoCo model (skipped if
  third_party/openarm is missing, see docs/openarm_v1_environment.md).
"""

import os
import sys
import time
import unittest
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.control.robot_interface import CMD_DIM, SIDES, JointLimits, TargetInterpolator, arm_slice, gripper_index
from src.control.safety import JointSafetyFilter

REPO = Path(__file__).resolve().parents[1]
HAS_MODEL = (REPO / "third_party/openarm/openarm_mujoco/v1/openarm_bimanual.xml").exists()
CFG = yaml.safe_load(open(REPO / "configs/task_config.yaml"))


def _limits() -> JointLimits:
    return JointLimits(lower=np.full(CMD_DIM, -1.0), upper=np.full(CMD_DIM, 1.0))


class TestSafetyFilter(unittest.TestCase):
    def test_rate_limit_and_clip(self):
        f = JointSafetyFilter(_limits(), max_joint_velocity=1.0, max_gripper_velocity=0.1, joint_limit_margin=0.1)
        f.reset(np.zeros(CMD_DIM))
        sp = f.step(np.full(CMD_DIM, 5.0), dt=0.01)
        self.assertAlmostEqual(sp[0], 0.01)  # 1 rad/s * 10 ms
        self.assertAlmostEqual(sp[gripper_index("left")], 0.001)  # 0.1 m/s * 10 ms
        for _ in range(1500):  # 15 s: enough for the gripper at 0.1 m/s
            sp = f.step(np.full(CMD_DIM, 5.0), dt=0.01)
        self.assertAlmostEqual(sp[0], 0.9)  # upper limit minus margin
        self.assertAlmostEqual(sp[gripper_index("right")], 1.0)  # margin does not apply to grippers

    def test_joint_inside_margin_is_not_pulled(self):
        f = JointSafetyFilter(_limits(), 1.0, 0.1, joint_limit_margin=0.1)
        q = np.zeros(CMD_DIM)
        q[0] = -1.0  # at the hard lower limit, inside the 0.1 margin
        f.reset(q)
        for _ in range(100):
            sp = f.step(q, 0.01)  # holding: must not move
        self.assertAlmostEqual(sp[0], -1.0)
        target = q.copy()
        target[0] = -2.0  # further out: refused
        self.assertAlmostEqual(f.step(target, 0.01)[0], -1.0)
        target[0] = 0.0  # inwards: allowed
        for _ in range(100):
            f.step(target, 0.01)
        self.assertAlmostEqual(f.setpoint[0], 0.0)
        for _ in range(200):  # and back to the start pose (inside the margin): allowed
            sp = f.step(q, 0.01)
        self.assertAlmostEqual(sp[0], -1.0)

    def test_reset_anchors_at_measured(self):
        f = JointSafetyFilter(_limits(), 1.0, 0.1)
        q = np.linspace(-0.5, 0.5, CMD_DIM)
        np.testing.assert_allclose(f.reset(q), q)
        np.testing.assert_allclose(f.step(q, 0.01), q)


class TestTargetInterpolator(unittest.TestCase):
    def test_streamed_targets_are_spread_over_the_period(self):
        it = TargetInterpolator(max_period=0.1)
        it.reset(np.zeros(CMD_DIM), now=0.0)
        it.set_goal(np.full(CMD_DIM, 1.0), now=1.0)  # after a pause: passes as a step
        np.testing.assert_allclose(it.value(1.0), 1.0)
        it.set_goal(np.full(CMD_DIM, 2.0), now=1.02)  # streaming at 50 Hz: period 20 ms
        np.testing.assert_allclose(it.value(1.02), 1.0)
        np.testing.assert_allclose(it.value(1.03), 1.5)
        np.testing.assert_allclose(it.value(1.05), 2.0)  # holds the goal after one period
        it.set_goal(np.full(CMD_DIM, 3.0), now=1.04)  # next target mid-way: continues from where it is
        np.testing.assert_allclose(it.value(1.04), 2.0)
        self.assertGreater(it.value(1.05)[0], 2.0)
        it.set_entry(0, -1.0)  # override (torque-mode gripper): immediate, no interpolation
        self.assertEqual(it.value(1.05)[0], -1.0)


@unittest.skipUnless(HAS_MODEL, "official OpenArm sources not in third_party/openarm")
class TestHardwareDriverWithFakeBus(unittest.TestCase):
    def _make(self, q0=None, active_sides=("left", "right")):
        from src.hardware import FakeOpenArmBus, OpenArmBimanual

        self.buses = {}

        def factory(bus_cfg):
            side = "left" if bus_cfg.can_interface == CFG["hardware"]["arms"]["left"]["can_interface"] else "right"
            bus = FakeOpenArmBus(q0=None if q0 is None else q0[side], dt=1.0 / CFG["hardware"]["control_rate_hz"])
            self.buses[side] = bus
            return bus

        cfg = yaml.safe_load(open(REPO / "configs/task_config.yaml"))
        cfg["hardware"]["rest_duration_s"] = 0.5
        cfg["hardware"]["active_sides"] = list(active_sides)
        return OpenArmBimanual.from_config(cfg, bus_factory=factory)

    def test_left_only_never_touches_right_bus(self):
        q0 = {"left": np.array([0.1, -0.2, 0.0, 0.3, 0.0, 0.0, 0.0]), "right": np.zeros(7)}
        robot = self._make(q0, active_sides=("left",))
        self.assertEqual(set(self.buses), {"left"})
        robot.start()
        try:
            time.sleep(0.05)
            state = robot.get_state()
            self.assertIsNone(state.fault)
            self.assertTrue(state.arms["left"].active)
            self.assertFalse(state.arms["right"].active)
            np.testing.assert_allclose(state.arms["right"].q, CFG["hardware"]["rest_qpos"]["right"])
            goal = state.command_vector()
            goal[arm_slice("left")][3] = 0.4
            goal[arm_slice("right")][3] = 1.0  # ignored: right arm inactive
            robot.set_target(goal)
            time.sleep(0.3)
            state = robot.get_state()
            self.assertAlmostEqual(state.arms["left"].q_setpoint[3], 0.4, places=6)
            np.testing.assert_allclose(state.arms["right"].q_setpoint, CFG["hardware"]["rest_qpos"]["right"])
            with self.assertRaises(ValueError):
                robot.set_gripper_torque({"right": 0.2})
        finally:
            robot.stop()
        self.assertFalse(self.buses["left"].enabled)

    def test_gripper_torque_mode_ramps_and_returns_bumpless(self):
        q0 = {"left": np.array([0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0]), "right": np.zeros(7)}
        robot = self._make(q0, active_sides=("left",))
        robot.start()
        try:
            bus = self.buses["left"]
            robot.set_joint_targets(gripper={"left": 0.03})
            time.sleep(1.6)  # opening at 22 mm/s
            bus.gripper_block_at = -0.5  # an object stops the fingers at ~21 mm
            robot.set_gripper_torque({"left": 0.5})
            time.sleep(0.05)
            cmd = bus.last_command
            self.assertEqual(cmd["gripper_kp"], 0.0)  # pure torque (current) control
            self.assertGreater(cmd["gripper_tau_ff"], 0.0)  # motor sign of closing: + (open is -1.0472 rad)
            self.assertLess(cmd["gripper_tau_ff"], 0.45)  # still ramping (4 Nm/s)
            time.sleep(0.8)
            arm = robot.get_state().arms["left"]
            self.assertAlmostEqual(arm.gripper_torque_cmd, 0.5, places=6)
            self.assertAlmostEqual(arm.gripper_tau, 0.5, places=6)
            self.assertAlmostEqual(arm.gripper, 0.5 / 1.0472 * 0.044, places=4)  # held by the object
            robot.set_joint_targets(gripper={"left": 0.0})  # ignored while in torque mode
            robot.set_gripper_torque({"left": None})
            time.sleep(0.1)
            arm = robot.get_state().arms["left"]
            self.assertIsNone(arm.gripper_torque_cmd)
            self.assertEqual(bus.last_command["gripper_kp"], CFG["hardware"]["gripper"]["kp"])
            self.assertAlmostEqual(arm.gripper_setpoint, arm.gripper, places=4)  # holds where it is
        finally:
            robot.stop()

    def test_grip_approaches_fast_then_holds_grip_torque(self):
        from scripts.grip_torque_test import grip

        q0 = {"left": np.array([0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0]), "right": np.zeros(7)}
        robot = self._make(q0, active_sides=("left",))
        robot.start()
        try:
            bus = self.buses["left"]
            bus.gripper_rad_s_per_nm = 0.5  # sluggish fingers: 0.1 Nm alone would take ~20 s
            robot.set_joint_targets(gripper={"left": 0.044})
            time.sleep(2.2)
            bus.gripper_block_at = -0.5  # object at ~21 mm
            t0 = time.monotonic()
            g = grip(robot, "left", 0.1, CFG)
            self.assertLess(time.monotonic() - t0, 4.0)
            self.assertAlmostEqual(g, 0.5 / 1.0472 * 0.044, places=3)
            self.assertAlmostEqual(robot.get_state().arms["left"].gripper_torque_cmd, 0.1, places=6)
        finally:
            robot.stop()

    def test_start_holds_measured_pose_then_tracks(self):
        q0 = {"left": np.array([0.1, -0.2, 0.0, 0.3, 0.0, 0.0, 0.0]), "right": np.array([-0.1, 0.2, 0.0, 0.3, 0.0, 0.0, 0.0])}
        robot = self._make(q0)
        robot.start()
        try:
            time.sleep(0.05)
            state = robot.get_state()
            self.assertIsNone(state.fault)
            np.testing.assert_allclose(state.arms["left"].q_setpoint, q0["left"], atol=1e-9)  # no jump at start
            # A far target is approached at the joint velocity limit, not in one step.
            goal = state.command_vector()
            goal[arm_slice("left")][3] = 1.3
            robot.set_target(goal)
            time.sleep(0.2)
            sp = robot.get_state().arms["left"].q_setpoint[3]
            vmax = CFG["hardware"]["max_joint_velocity_rad_s"][3]
            self.assertGreater(sp, 0.3 + 0.05)
            self.assertLess(sp, 0.3 + vmax * 0.35)
            # The gripper is commanded in motor radians: 0.044 m open == -1.0472 rad.
            robot.set_joint_targets(gripper={"right": 0.044})
            time.sleep(2.3)  # full stroke takes 2 s at max_gripper_velocity
            self.assertAlmostEqual(self.buses["right"].last_command["gripper_q"], -1.0472, places=3)
        finally:
            robot.stop()
        self.assertFalse(self.buses["left"].enabled)  # disabled after the rest move

    def test_silent_motor_latches_fault_and_holds(self):
        robot = self._make()
        robot.start()
        try:
            self.buses["left"].silent = True
            time.sleep(0.3)
            state = robot.get_state()
            self.assertIn("stopped replying", state.fault or "")
            held = state.arms["right"].q_setpoint.copy()
            robot.set_target(state.command_vector() + 0.3)  # ignored while faulted
            time.sleep(0.05)
            np.testing.assert_allclose(robot.get_state().arms["right"].q_setpoint, held, atol=1e-9)
            self.buses["left"].silent = False
            robot.clear_fault()
            time.sleep(0.05)
            self.assertIsNone(robot.get_state().fault)
        finally:
            robot.stop()

    def test_refuses_to_start_outside_limits(self):
        q0 = {"left": np.array([3.0, 0, 0, 0.3, 0, 0, 0]), "right": np.zeros(7)}  # left j1 upper limit is 1.396
        robot = self._make(q0)
        with self.assertRaisesRegex(RuntimeError, "zero calibration"):
            robot.start()
        self.assertFalse(self.buses["left"].enabled)


@unittest.skipUnless(HAS_MODEL, "official OpenArm sources not in third_party/openarm")
class TestControllersInSim(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.control.ik_controller import IKController
        from src.simulation.openarm_sim import OpenArmSim

        cfg = yaml.safe_load(open(REPO / "configs/task_config.yaml"))
        cfg["hardware"]["active_sides"] = list(SIDES)  # bimanual controller tests
        cls.robot = OpenArmSim.from_config(cfg)
        cls.robot.start()
        cls.ik = IKController.from_config(cfg, cls.robot)
        cls.home = {s: np.asarray(CFG["robot"]["home_qpos"][s]) for s in SIDES}

    @classmethod
    def tearDownClass(cls):
        cls.robot.stop()

    def setUp(self):
        self.robot.move_joints(self.home, gripper={s: 0.044 for s in SIDES}, duration=2.0)
        self.robot.advance(0.5)
        self.ik.reset()

    def test_move_joints_reaches_home(self):
        np.testing.assert_allclose(self.robot.get_joints(), np.concatenate([self.home[s] for s in SIDES]), atol=0.03)

    def test_move_l_reaches_pose(self):
        p0, R0 = self.ik.get_tcp_pose()["left"]
        target = p0 + np.array([0.05, 0.0, -0.08])
        self.ik.move_l({"left": (target, R0)}, duration=2.0)
        self.robot.advance(0.5)
        p, R = self.ik.get_tcp_pose()["left"]
        self.assertLess(np.linalg.norm(p - target), 0.01)
        self.assertLess(np.linalg.norm(R - R0), 0.05)

    def test_ik_command_delta_moves_tcp(self):
        p0 = self.ik.get_tcp_pose()["right"][0]
        for _ in range(10):  # 10 x 1 cm up at 20 Hz
            self.ik.command_delta({"right": np.array([0.0, 0.0, 0.01])})
            self.robot.advance(0.05)
        self.robot.advance(0.5)
        dz = self.ik.get_tcp_pose()["right"][0][2] - p0[2]
        self.assertAlmostEqual(dz, 0.10, delta=0.005)

    def test_ik_step_is_capped(self):
        p0, R0 = self.ik.pose_cmd["left"]
        self.ik.command_poses({"left": (p0 + np.array([0.0, 0.0, -0.2]), R0)})
        moved = np.linalg.norm(self.ik.pose_cmd["left"][0] - p0)
        self.assertAlmostEqual(moved, CFG["controllers"]["ik"]["max_pos_step_m"], delta=1e-3)

    def test_joint_command_delta_and_gripper(self):
        from src.control.joint_position_controller import JointPositionController

        jc = JointPositionController(self.robot)
        jc.reset()
        dq = np.zeros(7)
        dq[6] = 0.05  # left joint 7
        for _ in range(5):
            jc.command_delta({"left": dq}, gripper_m={"right": 0.0})
            self.robot.advance(0.05)
        self.robot.advance(2.5)  # grippers need 2 s for a full stroke
        state = self.robot.get_state()
        self.assertAlmostEqual(state.arms["left"].q[6], 0.25, delta=0.01)
        self.assertLess(state.arms["right"].gripper, 0.005)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(HAS_MODEL, "official OpenArm sources not in third_party/openarm")
class TestPayloadAndGripperTorqueInSim(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.planning.openarm_kinematics import OpenArmKinematics
        from src.simulation.openarm_sim import OpenArmSim

        cfg = yaml.safe_load(open(REPO / "configs/task_config.yaml"))
        cfg["hardware"]["active_sides"] = ["left"]
        cls.robot = OpenArmSim.from_config(cfg)
        cls.robot.start()
        cls.kin = OpenArmKinematics(cfg["robot"]["mjcf_path"])
        cls.robot.move_joints({"left": np.asarray(cfg["robot"]["home_qpos"]["left"])},
                              gripper={"left": 0.03}, duration=2.0)
        cls.robot.advance(1.0)

    @classmethod
    def tearDownClass(cls):
        cls.robot.stop()

    def test_inactive_arm_ignores_targets(self):
        self.robot.set_joint_targets({"right": np.full(7, 0.3)})
        self.robot.advance(0.2)
        np.testing.assert_allclose(self.robot.get_state().arms["right"].q_setpoint, 0.0)

    def test_gripper_torque_closes_and_releases_in_place(self):
        self.robot.set_gripper_torque({"left": 0.3})
        self.robot.advance(1.5)
        arm = self.robot.get_state().arms["left"]
        self.assertLess(arm.gripper, 0.002)  # nothing in between: closes fully
        self.assertAlmostEqual(arm.gripper_torque_cmd, 0.3)
        self.robot.set_gripper_torque({"left": None})
        self.robot.advance(0.5)
        arm = self.robot.get_state().arms["left"]
        self.assertIsNone(arm.gripper_torque_cmd)
        self.assertLess(arm.gripper, 0.002)  # stays where it was
        self.robot.set_joint_targets(gripper={"left": 0.03})
        self.robot.advance(2.0)

    def test_heavier_load_is_detected(self):
        from src.estimation.payload import compare, measure_load

        robot, hand = self.robot, self.robot.model.body("openarm_left_hand").id
        kw = dict(amplitude_deg=1.0, period_s=1.0, cycles=2)
        light = measure_load(robot, self.kin, "left", "light", **kw)
        robot.data.xfrc_applied[hand, 2] = -0.3 * 9.81  # 300 g hanging on the hand
        try:
            robot.advance(0.5)
            heavy = measure_load(robot, self.kin, "left", "heavy", **kw)
        finally:
            robot.data.xfrc_applied[hand, :] = 0.0
        self.assertEqual(light.joints, [0, 3, 6])  # shoulder pitch, elbow, wrist pitch
        self.assertLess(abs(light.mass), 0.05)  # sim torques match the model
        c = compare(light, heavy)
        self.assertEqual(c.heavier.label, "heavy")
        self.assertGreater(c.diff_kg, 0.2)  # the hand body sits a bit inboard of the TCP
        self.assertLess(c.diff_kg, 0.35)
