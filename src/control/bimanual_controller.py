"""
Bimanual Coordination & Safety Controller.

Orchestrates teleoperation commands from egocentric vision, triggers OMPL planning
upon large goal deltas, enforces workspace bounds and dual-arm clearance,
and executes smooth trajectories on the OpenArms platform.
"""

from __future__ import annotations
import time
import logging
from enum import Enum
from typing import Optional, Dict, Any, Tuple
import numpy as np

from ..vision.hand_tracker_listener import BimanualHandFrame, EgocentricToRobotMapper
from ..planning.ompl_planner import BimanualOMPLPlanner, BimanualTrajectory, PlanningResultStatus
from ..simulation.mujoco_env import BimanualMuJoCoEnv, BimanualState

logger = logging.getLogger("RBS.BimanualController")


class ControllerState(Enum):
    IDLE = "IDLE"
    CALIBRATING = "CALIBRATING"
    TRACKING = "TRACKING"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    ESTOP = "ESTOP"


class BimanualController:
    """
    Main closed-loop coordinator integrating vision, OMPL motion planning, and MuJoCo simulation.
    """

    def __init__(
        self,
        env: BimanualMuJoCoEnv,
        planner: BimanualOMPLPlanner,
        mapper: EgocentricToRobotMapper,
        replan_delta_threshold: float = 0.04,  # 4 cm trigger for OMPL replan
        min_ee_distance: float = 0.08,  # 8 cm minimum dual end-effector spacing
    ):
        self.env = env
        self.planner = planner
        self.mapper = mapper
        self.replan_delta_threshold = replan_delta_threshold
        self.min_ee_distance = min_ee_distance

        self.state = ControllerState.IDLE
        self.current_trajectory: Optional[BimanualTrajectory] = None
        self._traj_step_idx: int = 0

        self.last_planned_left_target: Optional[np.ndarray] = None
        self.last_planned_right_target: Optional[np.ndarray] = None

    def start(self) -> None:
        """Start tracking mode."""
        self.state = ControllerState.TRACKING
        logger.info("Bimanual controller started in TRACKING mode.")

    def estop(self, reason: str = "") -> None:
        """Emergency stop: locks joints and halts motion immediately."""
        self.state = ControllerState.ESTOP
        self.current_trajectory = None
        logger.error(f"EMERGENCY STOP TRIGGERED: {reason}")

    def update(self, hand_frame: Optional[BimanualHandFrame]) -> BimanualState:
        """
        Main cycle called at control frequency (e.g. 50-100Hz).
        Reads hand frame, checks safety, plans trajectories when necessary,
        and steps the simulation.
        """
        if self.state == ControllerState.ESTOP:
            # Hold current position
            return self.env.get_bimanual_state()

        current_robot_state = self.env.get_bimanual_state()

        # Check for unexpected physical collision
        if current_robot_state.in_collision:
            self.estop("Self-collision detected between arm links in simulation")
            return current_robot_state

        if hand_frame is None or (hand_frame.left_hand is None and hand_frame.right_hand is None):
            return current_robot_state

        # Map egocentric coordinates to robot base frame
        left_cmd = self.mapper.map_hand_pose(hand_frame.left_hand) if hand_frame.left_hand else None
        right_cmd = self.mapper.map_hand_pose(hand_frame.right_hand) if hand_frame.right_hand else None

        left_target_pos = left_cmd["cartesian_pos"] if left_cmd else current_robot_state.left.ee_position
        right_target_pos = right_cmd["cartesian_pos"] if right_cmd else current_robot_state.right.ee_position

        left_grip = left_cmd["gripper_target"] if left_cmd else 0.0
        right_grip = right_cmd["gripper_target"] if right_cmd else 0.0

        # Safety Check: Dual-arm clearance distance
        ee_dist = float(np.linalg.norm(left_target_pos - right_target_pos))
        if ee_dist < self.min_ee_distance:
            logger.warning(f"Dual-arm proximity warning: EE distance {ee_dist*100:.1f} cm < {self.min_ee_distance*100:.1f} cm. Clamping targets.")
            # Separate them along Y axis
            mid_y = 0.5 * (left_target_pos[1] + right_target_pos[1])
            left_target_pos[1] = mid_y + (self.min_ee_distance / 2.0)
            right_target_pos[1] = mid_y - (self.min_ee_distance / 2.0)

        # Determine if we should trigger an OMPL trajectory plan
        needs_replan = False
        if self.last_planned_left_target is None or self.last_planned_right_target is None:
            needs_replan = True
        else:
            d_left = np.linalg.norm(left_target_pos - self.last_planned_left_target)
            d_right = np.linalg.norm(right_target_pos - self.last_planned_right_target)
            if max(d_left, d_right) > self.replan_delta_threshold:
                needs_replan = True

        # Replan trajectory if needed or if currently idle
        if needs_replan or (self.current_trajectory is None or self._traj_step_idx >= len(self.current_trajectory.timestamps)):
            traj = self.planner.plan_cartesian_goals(
                env=self.env,
                left_ee_target=left_target_pos,
                right_ee_target=right_target_pos,
                current_state=current_robot_state,
            )

            if traj.is_valid:
                self.current_trajectory = traj
                self._traj_step_idx = 0
                self.last_planned_left_target = left_target_pos.copy()
                self.last_planned_right_target = right_target_pos.copy()
                self.state = ControllerState.EXECUTING
            else:
                logger.warning(f"Planner rejected motion: {traj.message}")

        # Execute trajectory waypoint or direct IK tracking step
        if self.current_trajectory is not None and self._traj_step_idx < len(self.current_trajectory.timestamps):
            q_left_tgt = self.current_trajectory.left_trajectory[self._traj_step_idx]
            q_right_tgt = self.current_trajectory.right_trajectory[self._traj_step_idx]
            self._traj_step_idx += 1
        else:
            # Fall back to fine-grained Jacobian IK step for smooth servoing
            q_left_tgt = self.env.compute_ik_step("left", left_target_pos)
            q_right_tgt = self.env.compute_ik_step("right", right_target_pos)

        # Advance physics
        next_state = self.env.step(
            left_joint_targets=q_left_tgt,
            left_gripper_target=left_grip,
            right_joint_targets=q_right_tgt,
            right_gripper_target=right_grip,
            decimation=self.env.timestep and 5,
        )

        return next_state
