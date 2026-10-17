"""
Unit & Integration Tests for Bimanual OpenArms Pipeline.
"""

import os
import sys
import unittest
import numpy as np

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vision.hand_tracker_listener import (
    HandPose,
    MockHandTrackerStream,
    EgocentricToRobotMapper,
)
from src.planning.ompl_planner import (
    BimanualOMPLPlanner,
    PlanningResultStatus,
)
from src.simulation.mujoco_env import BimanualMuJoCoEnv
from src.control.bimanual_controller import BimanualController, ControllerState


class TestHandTrackerAndMapper(unittest.TestCase):
    def test_mock_stream_generation(self):
        stream = MockHandTrackerStream(fps=30)
        self.assertTrue(stream.connect())
        frame = stream.get_latest_frame()
        self.assertIsNotNone(frame)
        self.assertIsNotNone(frame.left_hand)
        self.assertIsNotNone(frame.right_hand)
        self.assertEqual(frame.left_hand.handedness, "left")
        self.assertEqual(frame.right_hand.handedness, "right")
        self.assertEqual(len(frame.left_hand.wrist_position), 3)
        self.assertTrue(0.0 <= frame.left_hand.grasp_strength <= 1.0)
        stream.disconnect()

    def test_coordinate_mapping_and_clamping(self):
        mapper = EgocentricToRobotMapper(
            scale_factor=1.0,
            alpha=1.0,  # No smoothing for direct test
            left_offset=np.array([0.35, 0.20, 0.50]),
            workspace_bounds={
                "x_min": 0.20,
                "x_max": 0.50,
                "y_min": -0.30,
                "y_max": 0.30,
                "z_min": 0.40,
                "z_max": 0.60,
            },
        )

        # Baseline pose
        pose1 = HandPose(
            handedness="left",
            wrist_position=np.array([0.0, 0.0, 0.5]),
            grasp_strength=0.0,
        )
        res1 = mapper.map_hand_pose(pose1)
        self.assertTrue(np.allclose(res1["cartesian_pos"], [0.35, 0.20, 0.50]))

        # Extreme movement that exceeds max bounds
        pose_extreme = HandPose(
            handedness="left",
            wrist_position=np.array([10.0, -10.0, 10.0]),
            grasp_strength=1.0,
        )
        res_extreme = mapper.map_hand_pose(pose_extreme)
        pos = res_extreme["cartesian_pos"]

        # Check that result is clamped strictly within safety bounds
        self.assertTrue(0.20 <= pos[0] <= 0.50)
        self.assertTrue(-0.30 <= pos[1] <= 0.30)
        self.assertTrue(0.40 <= pos[2] <= 0.60)
        self.assertAlmostEqual(res_extreme["gripper_target"], -0.035)


class TestOMPLPlanner(unittest.TestCase):
    def test_joint_path_planner(self):
        planner = BimanualOMPLPlanner(max_planning_time=0.1, interpolation_steps=15)

        q_left_start = np.zeros(7)
        q_right_start = np.zeros(7)
        q_left_goal = np.array([0.2, -0.1, -0.4, 0.1, 0.0, 0.0, 0.0])
        q_right_goal = np.array([-0.2, 0.1, -0.4, 0.1, 0.0, 0.0, 0.0])

        traj = planner.plan_bimanual_joint_path(
            q_left_start, q_right_start, q_left_goal, q_right_goal, duration=0.5
        )

        self.assertEqual(traj.status, PlanningResultStatus.SUCCESS)
        self.assertEqual(len(traj.timestamps), 15)
        self.assertEqual(traj.left_trajectory.shape, (15, 7))
        self.assertEqual(traj.right_trajectory.shape, (15, 7))
        # Start and end waypoint match
        self.assertTrue(np.allclose(traj.left_trajectory[0], q_left_start))
        self.assertTrue(np.allclose(traj.left_trajectory[-1], q_left_goal))

    def test_collision_detection_in_planner(self):
        planner = BimanualOMPLPlanner()
        # Inward collision pose: arms severely crossing each other
        q_left_cross = np.array([-1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        q_right_cross = np.array([1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        traj = planner.plan_bimanual_joint_path(
            np.zeros(7), np.zeros(7), q_left_cross, q_right_cross
        )
        self.assertEqual(traj.status, PlanningResultStatus.COLLISION)


class TestSimulationEnv(unittest.TestCase):
    def test_env_initialization_and_step(self):
        xml_path = "third_party/openarm/openarm_mujoco/v1/openarm_bimanual_physical_scene.xml"
        env = BimanualMuJoCoEnv(xml_path=xml_path, render=False)

        state = env.reset()
        self.assertEqual(len(state.left.joint_positions), 7)
        self.assertEqual(len(state.right.joint_positions), 7)
        self.assertEqual(len(state.left.ee_position), 3)
        self.assertEqual(len(state.right.ee_position), 3)

        # Step with new target
        step_state = env.step(
            left_joint_targets=np.array([0.1, -0.1, 0.1, 0.1, 0.0, 0.0, 0.0]),
            right_joint_targets=np.array([-0.1, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0]),
        )
        self.assertFalse(step_state.in_collision)
        env.close()


class TestBimanualController(unittest.TestCase):
    def test_controller_pipeline_step(self):
        xml_path = "third_party/openarm/openarm_mujoco/v1/openarm_bimanual_physical_scene.xml"
        env = BimanualMuJoCoEnv(xml_path=xml_path, render=False)
        planner = BimanualOMPLPlanner()
        mapper = EgocentricToRobotMapper()

        controller = BimanualController(env, planner, mapper)
        controller.start()
        self.assertEqual(controller.state, ControllerState.TRACKING)

        stream = MockHandTrackerStream()
        stream.connect()
        frame = stream.get_latest_frame()

        state = controller.update(frame)
        self.assertIsNotNone(state)
        self.assertIn(controller.state, [ControllerState.TRACKING, ControllerState.EXECUTING, ControllerState.PLANNING])

        stream.disconnect()
        env.close()


if __name__ == "__main__":
    unittest.main()
