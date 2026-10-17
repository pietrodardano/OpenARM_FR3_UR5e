"""
MuJoCo Simulation Environment for Bimanual OpenArms using RobotBlockSet (RBS) Conventions.

This module initializes the dual-arm OpenArms setup inside MuJoCo, bridges with
RobotBlockSet if installed, manages actuators, computes kinematics (FK/IK),
and detects self-collisions and environmental contact constraints.
"""

from __future__ import annotations
import os
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
import numpy as np

# Configure logger
logger = logging.getLogger("RBS.MuJoCoEnv")

# Check for RobotBlockSet (RBS) availability
HAS_RBS = False
try:
    import robotblockset as rbs  # type: ignore
    HAS_RBS = True
    logger.info("Successfully loaded RobotBlockSet (RBS) framework.")
except ImportError:
    logger.warning("robotblockset package not found in current environment. Using native MuJoCo RBS-compatible wrapper.")

# Check for MuJoCo availability
HAS_MUJOCO = False
try:
    import mujoco  # type: ignore
    import mujoco.viewer  # type: ignore
    HAS_MUJOCO = True
except ImportError:
    logger.warning("mujoco package not installed. Running in mock simulation mode.")


@dataclass
class RobotArmState:
    """State of a single robot arm."""
    joint_positions: np.ndarray  # shape (5,)
    joint_velocities: np.ndarray  # shape (5,)
    ee_position: np.ndarray  # shape (3,) [x, y, z] in robot base frame
    ee_orientation: np.ndarray  # shape (4,) [qw, qx, qy, qz]
    gripper_position: float  # slider joint value
    actuator_controls: np.ndarray  # shape (6,) [q1..q5, gripper]


@dataclass
class BimanualState:
    """Full synchronized state of the bimanual system."""
    time: float
    left: RobotArmState
    right: RobotArmState
    in_collision: bool
    contact_count: int


class BimanualMuJoCoEnv:
    """
    RBS-compatible simulation environment for the Bimanual OpenArms platform.
    Manages loading MJCF XML, setting actuator goals, reading kinematic states,
    and verifying collision clearances.
    """

    LEFT_JOINT_NAMES = [
        "left_joint1", "left_joint2", "left_joint3", "left_joint4", "left_joint5"
    ]
    RIGHT_JOINT_NAMES = [
        "right_joint1", "right_joint2", "right_joint3", "right_joint4", "right_joint5"
    ]
    LEFT_ACTUATOR_NAMES = [
        "left_act1", "left_act2", "left_act3", "left_act4", "left_act5", "left_act_gripper"
    ]
    RIGHT_ACTUATOR_NAMES = [
        "right_act1", "right_act2", "right_act3", "right_act4", "right_act5", "right_act_gripper"
    ]

    def __init__(
        self,
        xml_path: str = "configs/bimanual_openarms.xml",
        timestep: float = 0.002,
        render: bool = False,
    ):
        self.xml_path = xml_path
        self.timestep = timestep
        self.render = render

        self.model: Optional[Any] = None
        self.data: Optional[Any] = None
        self.viewer: Optional[Any] = None

        # Joint & actuator IDs
        self.left_qpos_indices: List[int] = []
        self.right_qpos_indices: List[int] = []
        self.left_qvel_indices: List[int] = []
        self.right_qvel_indices: List[int] = []
        self.left_act_indices: List[int] = []
        self.right_act_indices: List[int] = []

        self.left_gripper_qpos_idx: int = -1
        self.right_gripper_qpos_idx: int = -1

        self.left_ee_site_id: int = -1
        self.right_ee_site_id: int = -1

        self._sim_time: float = 0.0
        self._mock_q_left = np.zeros(5, dtype=np.float64)
        self._mock_q_right = np.zeros(5, dtype=np.float64)

        self._initialize_model()

    def _initialize_model(self) -> None:
        """Loads MJCF model and maps IDs."""
        if not HAS_MUJOCO:
            logger.info("Initializing in Mock MuJoCo mode.")
            return

        if not os.path.exists(self.xml_path):
            raise FileNotFoundError(f"MuJoCo MJCF model not found at {self.xml_path}")

        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.model.opt.timestep = self.timestep
        self.data = mujoco.MjData(self.model)

        # Map joint indices
        for name in self.LEFT_JOINT_NAMES:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.left_qpos_indices.append(self.model.jnt_qposadr[jid])
            self.left_qvel_indices.append(self.model.jnt_dofadr[jid])

        for name in self.RIGHT_JOINT_NAMES:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.right_qpos_indices.append(self.model.jnt_qposadr[jid])
            self.right_qvel_indices.append(self.model.jnt_dofadr[jid])

        # Gripper joint indices
        l_grip_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_gripper_joint")
        self.left_gripper_qpos_idx = self.model.jnt_qposadr[l_grip_jid]

        r_grip_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_gripper_joint")
        self.right_gripper_qpos_idx = self.model.jnt_qposadr[r_grip_jid]

        # Actuator indices
        for name in self.LEFT_ACTUATOR_NAMES:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            self.left_act_indices.append(aid)

        for name in self.RIGHT_ACTUATOR_NAMES:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            self.right_act_indices.append(aid)

        # End-effector site IDs
        self.left_ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_ee")
        self.right_ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_ee")

        self.reset()
        logger.info(f"Loaded bimanual OpenArms MJCF. {self.model.nq} generalized coordinates.")

    def reset(self) -> BimanualState:
        """Reset simulation to nominal starting posture."""
        if HAS_MUJOCO and self.model is not None and self.data is not None:
            mujoco.mj_resetData(self.model, self.data)

            # Set ready pose (shoulder lifted, elbow bent forward)
            default_left_q = np.array([0.0, 0.4, 0.8, -0.4, 0.0], dtype=np.float64)
            default_right_q = np.array([0.0, 0.4, 0.8, -0.4, 0.0], dtype=np.float64)

            for idx, val in zip(self.left_qpos_indices, default_left_q):
                self.data.qpos[idx] = val
            for idx, val in zip(self.right_qpos_indices, default_right_q):
                self.data.qpos[idx] = val

            mujoco.mj_forward(self.model, self.data)
        else:
            self._mock_q_left = np.array([0.0, 0.4, 0.8, -0.4, 0.0], dtype=np.float64)
            self._mock_q_right = np.array([0.0, 0.4, 0.8, -0.4, 0.0], dtype=np.float64)

        return self.get_bimanual_state()

    def get_bimanual_state(self) -> BimanualState:
        """Extracts the complete bimanual state from MuJoCo data structures."""
        if HAS_MUJOCO and self.model is not None and self.data is not None:
            # Left arm
            left_q = np.array([self.data.qpos[i] for i in self.left_qpos_indices])
            left_dq = np.array([self.data.qvel[i] for i in self.left_qvel_indices])
            left_ee_pos = self.data.site_xpos[self.left_ee_site_id].copy()
            left_ee_mat = self.data.site_xmat[self.left_ee_site_id].copy()
            left_ee_quat = np.zeros(4)
            mujoco.mju_mat2Quat(left_ee_quat, left_ee_mat)
            left_grip = float(self.data.qpos[self.left_gripper_qpos_idx])
            left_ctrl = np.array([self.data.ctrl[i] for i in self.left_act_indices])

            # Right arm
            right_q = np.array([self.data.qpos[i] for i in self.right_qpos_indices])
            right_dq = np.array([self.data.qvel[i] for i in self.right_qvel_indices])
            right_ee_pos = self.data.site_xpos[self.right_ee_site_id].copy()
            right_ee_mat = self.data.site_xmat[self.right_ee_site_id].copy()
            right_ee_quat = np.zeros(4)
            mujoco.mju_mat2Quat(right_ee_quat, right_ee_mat)
            right_grip = float(self.data.qpos[self.right_gripper_qpos_idx])
            right_ctrl = np.array([self.data.ctrl[i] for i in self.right_act_indices])

            in_collision, ncon = self.check_collision()
            sim_t = float(self.data.time)
        else:
            # Mock mode state
            self._sim_time += self.timestep
            sim_t = self._sim_time
            left_q = self._mock_q_left.copy()
            right_q = self._mock_q_right.copy()
            left_dq = np.zeros(5)
            right_dq = np.zeros(5)
            left_ee_pos = np.array([0.35, 0.18, 0.55])
            right_ee_pos = np.array([0.35, -0.18, 0.55])
            left_ee_quat = np.array([1.0, 0.0, 0.0, 0.0])
            right_ee_quat = np.array([1.0, 0.0, 0.0, 0.0])
            left_grip = 0.0
            right_grip = 0.0
            left_ctrl = np.zeros(6)
            right_ctrl = np.zeros(6)
            in_collision = False
            ncon = 0

        left_state = RobotArmState(
            joint_positions=left_q,
            joint_velocities=left_dq,
            ee_position=left_ee_pos,
            ee_orientation=left_ee_quat,
            gripper_position=left_grip,
            actuator_controls=left_ctrl,
        )

        right_state = RobotArmState(
            joint_positions=right_q,
            joint_velocities=right_dq,
            ee_position=right_ee_pos,
            ee_orientation=right_ee_quat,
            gripper_position=right_grip,
            actuator_controls=right_ctrl,
        )

        return BimanualState(
            time=sim_t,
            left=left_state,
            right=right_state,
            in_collision=in_collision,
            contact_count=ncon,
        )

    def step(
        self,
        left_joint_targets: Optional[np.ndarray] = None,
        left_gripper_target: float = 0.0,
        right_joint_targets: Optional[np.ndarray] = None,
        right_gripper_target: float = 0.0,
        decimation: int = 1,
    ) -> BimanualState:
        """
        Applies actuator targets and steps the MuJoCo physics simulation.
        Args:
            left_joint_targets: 5-DOF joint angles for left arm.
            left_gripper_target: Left gripper slider position [-0.035, 0].
            right_joint_targets: 5-DOF joint angles for right arm.
            right_gripper_target: Right gripper slider position [-0.035, 0].
            decimation: Number of simulation substeps to execute.
        """
        if HAS_MUJOCO and self.model is not None and self.data is not None:
            if left_joint_targets is not None:
                for aid, target in zip(self.left_act_indices[:5], left_joint_targets):
                    self.data.ctrl[aid] = target
            self.data.ctrl[self.left_act_indices[5]] = left_gripper_target

            if right_joint_targets is not None:
                for aid, target in zip(self.right_act_indices[:5], right_joint_targets):
                    self.data.ctrl[aid] = target
            self.data.ctrl[self.right_act_indices[5]] = right_gripper_target

            for _ in range(decimation):
                mujoco.mj_step(self.model, self.data)

            if self.viewer is not None:
                self.viewer.sync()
        else:
            if left_joint_targets is not None:
                self._mock_q_left = np.array(left_joint_targets, dtype=np.float64)
            if right_joint_targets is not None:
                self._mock_q_right = np.array(right_joint_targets, dtype=np.float64)

        return self.get_bimanual_state()

    def check_collision(self) -> Tuple[bool, int]:
        """
        Detects self-collisions between left and right arm links or unauthorized contacts.
        Returns: (has_collision, contact_count).
        """
        if not (HAS_MUJOCO and self.model and self.data):
            return False, 0

        ncon = self.data.ncon
        for i in range(ncon):
            contact = self.data.contact[i]
            geom1_id = contact.geom1
            geom2_id = contact.geom2

            geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom1_id) or ""
            geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom2_id) or ""

            # Check for left-arm colliding with right-arm
            is_l_r_collision = (
                ("left" in geom1_name and "right" in geom2_name) or
                ("right" in geom1_name and "left" in geom2_name)
            )
            if is_l_r_collision:
                return True, ncon

        return False, ncon

    def compute_ik_step(
        self,
        arm: str,
        target_pos: np.ndarray,
        damping: float = 1e-2,
        step_scale: float = 0.5,
    ) -> np.ndarray:
        """
        Damped least-squares inverse kinematics step for a single arm.
        Returns new joint position target (5-DOF).
        """
        if not (HAS_MUJOCO and self.model and self.data):
            return np.zeros(5)

        is_left = arm.lower() == "left"
        site_id = self.left_ee_site_id if is_left else self.right_ee_site_id
        q_indices = self.left_qpos_indices if is_left else self.right_qpos_indices

        # Current end-effector position
        current_pos = self.data.site_xpos[site_id]
        pos_error = target_pos - current_pos

        # Compute Jacobian at the end-effector site
        jac_pos = np.zeros((3, self.model.nv))
        jac_rot = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jac_pos, jac_rot, site_id)

        # Slice relevant joint DOFs
        dof_indices = self.left_qvel_indices if is_left else self.right_qvel_indices
        J = jac_pos[:, dof_indices]  # (3, 5)

        # DLS formula: dq = J^T (J J^T + lambda^2 I)^(-1) dx
        JJT = J @ J.T + (damping ** 2) * np.eye(3)
        delta_q = J.T @ np.linalg.solve(JJT, pos_error)

        current_q = np.array([self.data.qpos[i] for i in q_indices])
        new_q = current_q + step_scale * delta_q
        return new_q

    def close(self) -> None:
        """Close viewer and clean resources."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
