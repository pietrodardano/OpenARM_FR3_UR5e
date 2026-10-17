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
from typing import Optional, Any, Tuple, List
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
    joint_positions: np.ndarray  # shape (7,)
    joint_velocities: np.ndarray  # shape (7,)
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

    ARM_DOF = 7
    LEFT_JOINT_NAMES = [f"openarm_left_joint{i}" for i in range(1, ARM_DOF + 1)]
    RIGHT_JOINT_NAMES = [f"openarm_right_joint{i}" for i in range(1, ARM_DOF + 1)]
    LEFT_ACTUATOR_NAMES = [
        *(f"left_joint{i}_ctrl" for i in range(1, ARM_DOF + 1)), "left_finger1_ctrl"
    ]
    RIGHT_ACTUATOR_NAMES = [
        *(f"right_joint{i}_ctrl" for i in range(1, ARM_DOF + 1)), "right_finger1_ctrl"
    ]
    LEFT_TCP_BODY_NAME = "openarm_left_hand_tcp"
    RIGHT_TCP_BODY_NAME = "openarm_right_hand_tcp"
    LEFT_GRIPPER_JOINT_NAME = "openarm_left_finger_joint1"
    RIGHT_GRIPPER_JOINT_NAME = "openarm_right_finger_joint1"

    def __init__(
        self,
        xml_path: str = "third_party/openarm/openarm_mujoco/v1/openarm_bimanual_physical_scene.xml",
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

        self.left_tcp_body_id: int = -1
        self.right_tcp_body_id: int = -1

        self._sim_time: float = 0.0
        self._mock_q_left = np.zeros(self.ARM_DOF, dtype=np.float64)
        self._mock_q_right = np.zeros(self.ARM_DOF, dtype=np.float64)

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

        # The official parallel-link gripper actuator drives finger joint 1.
        l_grip_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.LEFT_GRIPPER_JOINT_NAME)
        self.left_gripper_qpos_idx = self.model.jnt_qposadr[l_grip_jid]

        r_grip_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.RIGHT_GRIPPER_JOINT_NAME)
        self.right_gripper_qpos_idx = self.model.jnt_qposadr[r_grip_jid]

        # Actuator indices
        for name in self.LEFT_ACTUATOR_NAMES:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            self.left_act_indices.append(aid)

        for name in self.RIGHT_ACTUATOR_NAMES:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            self.right_act_indices.append(aid)

        # Use the physical hand TCP, not the wrist-level control-point sites.
        self.left_tcp_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.LEFT_TCP_BODY_NAME)
        self.right_tcp_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.RIGHT_TCP_BODY_NAME)

        self.reset()
        logger.info("Loaded official OpenArm v1 bimanual MJCF: %d arm DOF, %d total controls.", self.ARM_DOF, self.model.nu)

    def reset(self) -> BimanualState:
        """Reset simulation to nominal starting posture."""
        if HAS_MUJOCO and self.model is not None and self.data is not None:
            mujoco.mj_resetData(self.model, self.data)

            # A lifted, symmetric ready pose within official v1 joint limits.
            default_left_q = np.array([0.0, -0.8, 0.0, 0.7, 0.0, 0.0, 0.0])
            default_right_q = np.array([0.0, 0.8, 0.0, 0.7, 0.0, 0.0, 0.0])

            for idx, val in zip(self.left_qpos_indices, default_left_q):
                self.data.qpos[idx] = val
            for idx, val in zip(self.right_qpos_indices, default_right_q):
                self.data.qpos[idx] = val

            mujoco.mj_forward(self.model, self.data)
        else:
            self._mock_q_left = np.array([0.0, -0.8, 0.0, 0.7, 0.0, 0.0, 0.0])
            self._mock_q_right = np.array([0.0, 0.8, 0.0, 0.7, 0.0, 0.0, 0.0])

        return self.get_bimanual_state()

    def get_bimanual_state(self) -> BimanualState:
        """Extracts the complete bimanual state from MuJoCo data structures."""
        if HAS_MUJOCO and self.model is not None and self.data is not None:
            # Left arm
            left_q = np.array([self.data.qpos[i] for i in self.left_qpos_indices])
            left_dq = np.array([self.data.qvel[i] for i in self.left_qvel_indices])
            left_ee_pos = self.data.xpos[self.left_tcp_body_id].copy()
            left_ee_mat = self.data.xmat[self.left_tcp_body_id].copy()
            left_ee_quat = np.zeros(4)
            mujoco.mju_mat2Quat(left_ee_quat, left_ee_mat)
            left_grip = float(self.data.qpos[self.left_gripper_qpos_idx])
            left_ctrl = np.array([self.data.ctrl[i] for i in self.left_act_indices])

            # Right arm
            right_q = np.array([self.data.qpos[i] for i in self.right_qpos_indices])
            right_dq = np.array([self.data.qvel[i] for i in self.right_qvel_indices])
            right_ee_pos = self.data.xpos[self.right_tcp_body_id].copy()
            right_ee_mat = self.data.xmat[self.right_tcp_body_id].copy()
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
            left_dq = np.zeros(self.ARM_DOF)
            right_dq = np.zeros(self.ARM_DOF)
            left_ee_pos = np.array([0.35, 0.18, 0.55])
            right_ee_pos = np.array([0.35, -0.18, 0.55])
            left_ee_quat = np.array([1.0, 0.0, 0.0, 0.0])
            right_ee_quat = np.array([1.0, 0.0, 0.0, 0.0])
            left_grip = 0.0
            right_grip = 0.0
            left_ctrl = np.zeros(self.ARM_DOF + 1)
            right_ctrl = np.zeros(self.ARM_DOF + 1)
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
            left_joint_targets: Seven official OpenArm v1 left-arm joint angles.
            left_gripper_target: Parallel-link finger-1 position in metres.
            right_joint_targets: Seven official OpenArm v1 right-arm joint angles.
            right_gripper_target: Parallel-link finger-1 position in metres.
            decimation: Number of simulation substeps to execute.
        """
        if HAS_MUJOCO and self.model is not None and self.data is not None:
            if left_joint_targets is not None:
                self._validate_joint_targets(left_joint_targets, "left")
                for aid, target in zip(self.left_act_indices[:self.ARM_DOF], left_joint_targets):
                    self.data.ctrl[aid] = target
            self.data.ctrl[self.left_act_indices[self.ARM_DOF]] = left_gripper_target

            if right_joint_targets is not None:
                self._validate_joint_targets(right_joint_targets, "right")
                for aid, target in zip(self.right_act_indices[:self.ARM_DOF], right_joint_targets):
                    self.data.ctrl[aid] = target
            self.data.ctrl[self.right_act_indices[self.ARM_DOF]] = right_gripper_target

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

            # Finger-to-finger contact within the same parallel-link gripper is
            # expected. Detect only contacts whose OpenArm side prefixes differ.
            geom1_is_left = geom1_name.startswith("openarm_left_")
            geom1_is_right = geom1_name.startswith("openarm_right_")
            geom2_is_left = geom2_name.startswith("openarm_left_")
            geom2_is_right = geom2_name.startswith("openarm_right_")
            is_l_r_collision = (
                (geom1_is_left and geom2_is_right) or
                (geom1_is_right and geom2_is_left)
            )
            if is_l_r_collision:
                return True, ncon

        return False, ncon

    def _validate_joint_targets(self, targets: np.ndarray, side: str) -> None:
        """Reject malformed targets before writing controls to the official model."""
        values = np.asarray(targets, dtype=np.float64)
        if values.shape != (self.ARM_DOF,):
            raise ValueError(f"{side} OpenArm target must have shape ({self.ARM_DOF},), got {values.shape}.")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{side} OpenArm target contains non-finite values.")

    def solve_position_ik(
        self,
        arm: str,
        target_position: np.ndarray,
        seed_joint_positions: np.ndarray,
        tolerance_m: float = 0.005,
        max_iterations: int = 150,
        damping: float = 0.03,
    ) -> Optional[np.ndarray]:
        """Solve a position-only IK target using the official model Jacobian.

        This utility is simulation-only. It deliberately leaves the live
        simulation state unchanged and returns ``None`` when no solution meets
        the requested Cartesian tolerance.
        """
        if not (HAS_MUJOCO and self.model is not None and self.data is not None):
            return None

        side = arm.lower()
        if side not in {"left", "right"}:
            raise ValueError("arm must be 'left' or 'right'.")
        self._validate_joint_targets(seed_joint_positions, side)
        target = np.asarray(target_position, dtype=np.float64)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError("target_position must contain three finite values.")

        q_indices = self.left_qpos_indices if side == "left" else self.right_qpos_indices
        dof_indices = self.left_qvel_indices if side == "left" else self.right_qvel_indices
        body_id = self.left_tcp_body_id if side == "left" else self.right_tcp_body_id
        joint_names = self.LEFT_JOINT_NAMES if side == "left" else self.RIGHT_JOINT_NAMES
        joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
        limits = self.model.jnt_range[joint_ids]
        saved_qpos = self.data.qpos.copy()
        saved_qvel = self.data.qvel.copy()
        q = np.asarray(seed_joint_positions, dtype=np.float64).copy()

        try:
            for _ in range(max_iterations):
                self.data.qpos[q_indices] = q
                self.data.qvel[:] = 0.0
                mujoco.mj_forward(self.model, self.data)
                error = target - self.data.xpos[body_id]
                if np.linalg.norm(error) <= tolerance_m:
                    return q
                jac_pos = np.zeros((3, self.model.nv))
                jac_rot = np.zeros((3, self.model.nv))
                mujoco.mj_jacBody(self.model, self.data, jac_pos, jac_rot, body_id)
                jacobian = jac_pos[:, dof_indices]
                step = jacobian.T @ np.linalg.solve(
                    jacobian @ jacobian.T + damping**2 * np.eye(3), error
                )
                q = np.clip(q + np.clip(step, -0.08, 0.08), limits[:, 0], limits[:, 1])
            return None
        finally:
            self.data.qpos[:] = saved_qpos
            self.data.qvel[:] = saved_qvel
            mujoco.mj_forward(self.model, self.data)

    def solve_top_grasp_ik(
        self,
        arm: str,
        target_position: np.ndarray,
        seed_joint_positions: np.ndarray,
        tolerance_m: float = 0.006,
        tolerance_rad: float = 0.08,
        max_iterations: int = 250,
        damping: float = 0.05,
    ) -> Optional[np.ndarray]:
        """Solve a yaw-free top-down pose with local tool +Z facing downward."""
        if not (HAS_MUJOCO and self.model is not None and self.data is not None):
            return None
        side = arm.lower()
        if side not in {"left", "right"}:
            raise ValueError("arm must be 'left' or 'right'.")
        self._validate_joint_targets(seed_joint_positions, side)
        target = np.asarray(target_position, dtype=np.float64)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError("target_position must contain three finite values.")

        target_axis = np.array([0.0, 0.0, -1.0])
        q_indices = self.left_qpos_indices if side == "left" else self.right_qpos_indices
        dof_indices = self.left_qvel_indices if side == "left" else self.right_qvel_indices
        body_id = self.left_tcp_body_id if side == "left" else self.right_tcp_body_id
        joint_names = self.LEFT_JOINT_NAMES if side == "left" else self.RIGHT_JOINT_NAMES
        joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
        limits = self.model.jnt_range[joint_ids]
        saved_qpos = self.data.qpos.copy()
        saved_qvel = self.data.qvel.copy()
        q = np.asarray(seed_joint_positions, dtype=np.float64).copy()

        try:
            for _ in range(max_iterations):
                self.data.qpos[q_indices] = q
                self.data.qvel[:] = 0.0
                mujoco.mj_forward(self.model, self.data)
                position_error = target - self.data.xpos[body_id]
                current_axis = self.data.xmat[body_id].reshape(3, 3)[:, 2]
                # Cross-product error aligns the tool axis while leaving yaw free.
                rotation_error = np.cross(current_axis, target_axis)
                if (
                    np.linalg.norm(position_error) <= tolerance_m
                    and np.linalg.norm(rotation_error) <= tolerance_rad
                    and np.dot(current_axis, target_axis) > 0.0
                ):
                    return q
                jac_pos = np.zeros((3, self.model.nv))
                jac_rot = np.zeros((3, self.model.nv))
                mujoco.mj_jacBody(self.model, self.data, jac_pos, jac_rot, body_id)
                jacobian = np.vstack((jac_pos[:, dof_indices], jac_rot[:, dof_indices]))
                error = np.concatenate((position_error, rotation_error))
                step = jacobian.T @ np.linalg.solve(
                    jacobian @ jacobian.T + damping**2 * np.eye(6), error
                )
                q = np.clip(q + np.clip(step, -0.06, 0.06), limits[:, 0], limits[:, 1])
            return None
        finally:
            self.data.qpos[:] = saved_qpos
            self.data.qvel[:] = saved_qvel
            mujoco.mj_forward(self.model, self.data)

    def is_collision_free_configuration(self, arm: str, joint_positions: np.ndarray) -> bool:
        """Check a candidate arm posture against table and opposite-arm contacts.

        Contacts with the grasp cube are permitted so the top-down end pose can
        make contact. This validity test is for offline simulation planning.
        """
        if not (HAS_MUJOCO and self.model is not None and self.data is not None):
            return False
        side = arm.lower()
        if side not in {"left", "right"}:
            raise ValueError("arm must be 'left' or 'right'.")
        self._validate_joint_targets(joint_positions, side)
        q_indices = self.left_qpos_indices if side == "left" else self.right_qpos_indices
        arm_prefix = f"openarm_{side}_"
        other_prefix = "openarm_right_" if side == "left" else "openarm_left_"
        saved_qpos = self.data.qpos.copy()
        saved_qvel = self.data.qvel.copy()
        try:
            self.data.qpos[q_indices] = joint_positions
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)
            for contact_index in range(self.data.ncon):
                contact = self.data.contact[contact_index]
                first = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or ""
                second = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or ""
                touches_active_arm = first.startswith(arm_prefix) or second.startswith(arm_prefix)
                touches_table_or_other_arm = (
                    first == "work_table_top"
                    or second == "work_table_top"
                    or first.startswith(other_prefix)
                    or second.startswith(other_prefix)
                )
                if touches_active_arm and touches_table_or_other_arm:
                    return False
            return True
        finally:
            self.data.qpos[:] = saved_qpos
            self.data.qvel[:] = saved_qvel
            mujoco.mj_forward(self.model, self.data)

    def compute_ik_step(
        self,
        arm: str,
        target_pos: np.ndarray,
        damping: float = 1e-2,
        step_scale: float = 0.5,
    ) -> np.ndarray:
        """
        Damped least-squares inverse kinematics step for a single arm.
        Returns new joint position target (7-DOF).
        """
        if not (HAS_MUJOCO and self.model and self.data):
            return np.zeros(self.ARM_DOF)

        is_left = arm.lower() == "left"
        body_id = self.left_tcp_body_id if is_left else self.right_tcp_body_id
        q_indices = self.left_qpos_indices if is_left else self.right_qpos_indices

        # Current end-effector position
        current_pos = self.data.xpos[body_id]
        pos_error = target_pos - current_pos

        # Compute Jacobian at the end-effector site
        jac_pos = np.zeros((3, self.model.nv))
        jac_rot = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jac_pos, jac_rot, body_id)

        # Slice relevant joint DOFs
        dof_indices = self.left_qvel_indices if is_left else self.right_qvel_indices
        J = jac_pos[:, dof_indices]  # (3, 7)

        # DLS formula: dq = J^T (J J^T + lambda^2 I)^(-1) dx
        JJT = J @ J.T + (damping ** 2) * np.eye(3)
        delta_q = J.T @ np.linalg.solve(JJT, pos_error)

        current_q = np.array([self.data.qpos[i] for i in q_indices])
        new_q = current_q + step_scale * delta_q
        joint_names = self.LEFT_JOINT_NAMES if is_left else self.RIGHT_JOINT_NAMES
        joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
        limits = self.model.jnt_range[joint_ids]
        return np.clip(new_q, limits[:, 0], limits[:, 1])

    def close(self) -> None:
        """Close viewer and clean resources."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
