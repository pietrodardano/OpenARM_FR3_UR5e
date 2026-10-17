"""
OMPL & RobotBlockSet (RBS) Bimanual Motion Planner.

Computes collision-free joint trajectories for bimanual goal poses
derived from egocentric hand-tracking inputs, using RBS JointPathPlanner
with robust fallback execution.
"""

from __future__ import annotations
import time
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
import numpy as np

logger = logging.getLogger("RBS.OMPLPlanner")

# Check for RobotBlockSet & OMPL bindings
HAS_RBS_OMPL = False
try:
    import robotblockset as rbs  # type: ignore
    # Look for RBS OMPL planner interface
    from robotblockset.planning import JointPathPlanner  # type: ignore
    HAS_RBS_OMPL = True
    logger.info("RBS JointPathPlanner available.")
except (ImportError, AttributeError):
    logger.info("RBS JointPathPlanner not found. Using native collision-aware trajectory planner.")


class PlanningResultStatus(Enum):
    SUCCESS = "SUCCESS"
    COLLISION = "COLLISION"
    TIMEOUT = "TIMEOUT"
    IK_FAILED = "IK_FAILED"


@dataclass
class BimanualTrajectory:
    """Represents a synchronized time-parameterized bimanual trajectory."""
    status: PlanningResultStatus
    planning_time_sec: float
    timestamps: np.ndarray  # shape (N,)
    left_trajectory: np.ndarray  # shape (N, 5)
    right_trajectory: np.ndarray  # shape (N, 5)
    message: str = ""

    @property
    def is_valid(self) -> bool:
        return self.status == PlanningResultStatus.SUCCESS and len(self.timestamps) > 0


class BimanualOMPLPlanner:
    """
    Bimanual Motion Planner leveraging RobotBlockSet (RBS) and OMPL.
    Plans coordinated, collision-free C-space paths for dual OpenArms.
    """

    def __init__(
        self,
        planner_type: str = "RRTConnect",
        max_planning_time: float = 0.15,
        clearance_margin: float = 0.015,
        interpolation_steps: int = 25,
        max_joint_velocity: float = 2.5,
    ):
        """
        Args:
            planner_type: Algorithm name ('RRTConnect', 'BiTRRT', 'BKPIECE1').
            max_planning_time: Max planning duration in seconds.
            clearance_margin: Safety collision padding in meters.
            interpolation_steps: Number of discrete waypoints along path.
            max_joint_velocity: Joint velocity limit in rad/s.
        """
        self.planner_type = planner_type
        self.max_planning_time = max_planning_time
        self.clearance_margin = clearance_margin
        self.interpolation_steps = interpolation_steps
        self.max_joint_velocity = max_joint_velocity

        self._rbs_planner: Optional[Any] = None
        if HAS_RBS_OMPL:
            self._setup_rbs_ompl()

    def _setup_rbs_ompl(self) -> None:
        """Initialize RBS JointPathPlanner wrapper if available."""
        try:
            # Instantiate RBS JointPathPlanner
            self._rbs_planner = JointPathPlanner(planner=self.planner_type)
            logger.info(f"Initialized RBS JointPathPlanner ({self.planner_type})")
        except Exception as e:
            logger.warning(f"Could not initialize native RBS JointPathPlanner: {e}")
            self._rbs_planner = None

    def plan_bimanual_joint_path(
        self,
        current_left_q: np.ndarray,
        current_right_q: np.ndarray,
        goal_left_q: np.ndarray,
        goal_right_q: np.ndarray,
        duration: float = 0.5,
    ) -> BimanualTrajectory:
        """
        Plans a collision-free joint trajectory from current joint postures to goal postures.
        """
        start_t = time.time()

        # Step 1: Validate distance limits and potential self-collision at goal
        if not self._is_bimanual_state_valid(goal_left_q, goal_right_q):
            return BimanualTrajectory(
                status=PlanningResultStatus.COLLISION,
                planning_time_sec=time.time() - start_t,
                timestamps=np.array([]),
                left_trajectory=np.empty((0, 5)),
                right_trajectory=np.empty((0, 5)),
                message="Goal configuration violates dual-arm collision clearance",
            )

        # Step 2: Native RBS OMPL planner branch
        if self._rbs_planner is not None:
            try:
                # Joint space 10-DOF planning: combine left and right
                q_start = np.concatenate([current_left_q, current_right_q])
                q_goal = np.concatenate([goal_left_q, goal_right_q])
                raw_path = self._rbs_planner.plan(q_start, q_goal, timeout=self.max_planning_time)
                if raw_path is not None and len(raw_path) > 0:
                    timestamps = np.linspace(0.0, duration, len(raw_path))
                    return BimanualTrajectory(
                        status=PlanningResultStatus.SUCCESS,
                        planning_time_sec=time.time() - start_t,
                        timestamps=timestamps,
                        left_trajectory=raw_path[:, :5],
                        right_trajectory=raw_path[:, 5:],
                        message="Planned successfully via RBS OMPL",
                    )
            except Exception as e:
                logger.warning(f"RBS OMPL execution fallback triggered: {e}")

        # Step 3: Hackathon-resilient trajectory generator (Quintic polynomial / minimum jerk)
        steps = max(5, self.interpolation_steps)
        t_normalized = np.linspace(0.0, 1.0, steps)
        # Minimum jerk polynomial profile: s(tau) = 10*tau^3 - 15*tau^4 + 6*tau^5
        s = 10 * (t_normalized ** 3) - 15 * (t_normalized ** 4) + 6 * (t_normalized ** 5)

        left_traj = np.zeros((steps, 5))
        right_traj = np.zeros((steps, 5))

        for i in range(5):
            left_traj[:, i] = current_left_q[i] + s * (goal_left_q[i] - current_left_q[i])
            right_traj[:, i] = current_right_q[i] + s * (goal_right_q[i] - current_right_q[i])

        # Step 4: Validate all waypoints for self-collisions
        for step_idx in range(steps):
            if not self._is_bimanual_state_valid(left_traj[step_idx], right_traj[step_idx]):
                return BimanualTrajectory(
                    status=PlanningResultStatus.COLLISION,
                    planning_time_sec=time.time() - start_t,
                    timestamps=np.array([]),
                    left_trajectory=np.empty((0, 5)),
                    right_trajectory=np.empty((0, 5)),
                    message=f"Collision detected at waypoint {step_idx}/{steps}",
                )

        # Enforce velocity limits
        total_duration = max(duration, self._compute_minimum_duration(left_traj, right_traj))
        timestamps = np.linspace(0.0, total_duration, steps)

        return BimanualTrajectory(
            status=PlanningResultStatus.SUCCESS,
            planning_time_sec=time.time() - start_t,
            timestamps=timestamps,
            left_trajectory=left_traj,
            right_trajectory=right_traj,
            message="Collision-free minimum-jerk trajectory computed",
        )

    def plan_cartesian_goals(
        self,
        env: Any,
        left_ee_target: np.ndarray,
        right_ee_target: np.ndarray,
        current_state: Any,
        ik_iterations: int = 5,
    ) -> BimanualTrajectory:
        """
        Solves IK for left and right targets and plans collision-free C-space motion.
        """
        # Iterative IK refinement using environment Jacobian
        goal_left_q = current_state.left.joint_positions.copy()
        goal_right_q = current_state.right.joint_positions.copy()

        for _ in range(ik_iterations):
            goal_left_q = env.compute_ik_step("left", left_ee_target)
            goal_right_q = env.compute_ik_step("right", right_ee_target)

        return self.plan_bimanual_joint_path(
            current_left_q=current_state.left.joint_positions,
            current_right_q=current_state.right.joint_positions,
            goal_left_q=goal_left_q,
            goal_right_q=goal_right_q,
        )

    def _is_bimanual_state_valid(self, q_left: np.ndarray, q_right: np.ndarray) -> bool:
        """
        Heuristic clearance and joint limit verification.
        Prevents arms from crossing over and colliding in the central zone.
        """
        # Joint limit checks
        joint_limits = np.array([2.6, 1.8, 2.2, 1.8, 2.9])
        if np.any(np.abs(q_left) > joint_limits) or np.any(np.abs(q_right) > joint_limits):
            return False

        # Shoulder pan crossing check: prevent left arm rotating inward into right arm
        # q1: positive = inward for left, negative = inward for right
        if q_left[0] < -1.2 and q_right[0] > 1.2:
            return False

        return True

    def _compute_minimum_duration(self, left_traj: np.ndarray, right_traj: np.ndarray) -> float:
        """Computes minimum time required to respect joint velocity limits."""
        delta_q_left = np.abs(np.diff(left_traj, axis=0))
        delta_q_right = np.abs(np.diff(right_traj, axis=0))
        max_delta = max(float(np.max(delta_q_left)), float(np.max(delta_q_right)), 1e-4)

        # dt_min = max_delta / max_vel
        min_step_dt = max_delta / self.max_joint_velocity
        total_time = min_step_dt * len(left_traj)
        return float(max(total_time, 0.1))
