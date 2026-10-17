"""
Kinematics of the official OpenArm v1 bimanual model (``openarm_mujoco``).

Loads ``third_party/openarm/openarm_mujoco/v1/openarm_bimanual.xml`` (robot only,
no scene objects) and adds one ``<side>_tcp`` site per arm at the official
``openarm_<side>_hand_tcp`` body, i.e. the point between the fingertips. All
poses are expressed in the model's world frame, which is the robot base frame:
origin on the floor under the torso, +Z up.

This is the single source of truth for joint limits, forward kinematics and the
gravity model used by the IK controller and (optionally) the hardware driver.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import mujoco
import numpy as np
from numpy.typing import NDArray

from ..control.robot_interface import (
    ARM_DOF,
    CMD_DIM,
    GRIPPER_CLOSED_M,
    GRIPPER_OPEN_M,
    SIDES,
    JointLimits,
    Side,
    arm_slice,
    gripper_index,
)

DEFAULT_MJCF = "third_party/openarm/openarm_mujoco/v1/openarm_bimanual.xml"

Pose = Tuple[NDArray[np.float64], NDArray[np.float64]]  # (position (3,), rotation matrix (3, 3))


def tcp_site_name(side: Side) -> str:
    return f"{side}_tcp"


def resolve_mjcf(path: Optional[str] = None) -> Path:
    """Resolves the official MJCF path (relative paths are taken from the repo root)."""
    p = Path(path or DEFAULT_MJCF)
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[2] / p
    if not p.exists():
        raise FileNotFoundError(
            f"Official OpenArm MJCF not found at {p}. Clone the official sources into "
            "third_party/openarm/ (see docs/openarm_v1_environment.md)."
        )
    return p


def load_openarm_spec(mjcf_path: Optional[str] = None) -> mujoco.MjSpec:
    """Official bimanual MJCF as an editable spec, with ``<side>_tcp`` sites added."""
    spec = mujoco.MjSpec.from_file(str(resolve_mjcf(mjcf_path)))
    for side in SIDES:
        body = spec.body(f"openarm_{side}_hand_tcp")
        body.add_site(name=tcp_site_name(side), size=[0.01, 0, 0], rgba=[1, 0, 0, 1], group=4)
    return spec


class OpenArmKinematics:
    """Forward kinematics, joint limits and gravity torques of both arms."""

    def __init__(self, mjcf_path: Optional[str] = None) -> None:
        self.spec = load_openarm_spec(mjcf_path)
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        self.arm_joint_names: Dict[Side, list] = {
            side: [f"openarm_{side}_joint{i}" for i in range(1, ARM_DOF + 1)] for side in SIDES
        }
        self.qpos_adr: Dict[Side, NDArray[np.int64]] = {}
        self.dof_adr: Dict[Side, NDArray[np.int64]] = {}
        for side in SIDES:
            jids = [self.model.joint(n).id for n in self.arm_joint_names[side]]
            self.qpos_adr[side] = self.model.jnt_qposadr[jids].copy()
            self.dof_adr[side] = self.model.jnt_dofadr[jids].copy()
        self.site_id = {side: self.model.site(tcp_site_name(side)).id for side in SIDES}
        self.limits = self._joint_limits()
        # Collision geoms used by clearance(): each arm from link2 outwards (link0/1 sit
        # on the torso), and the torso itself.
        self._arm_geoms = {
            side: self._collision_geoms(
                [f"openarm_{side}_link{i}" for i in range(2, ARM_DOF + 1)]
                + [f"openarm_{side}_hand", f"openarm_{side}_left_finger", f"openarm_{side}_right_finger"]
            )
            for side in SIDES
        }
        self._torso_geoms = self._collision_geoms(["openarm_body_link0"])
        key = self.model.key("home").id
        self.home_qpos: Dict[Side, NDArray[np.float64]] = {
            side: self.model.key_qpos[key][self.qpos_adr[side]].copy() for side in SIDES
        }

    def _joint_limits(self) -> JointLimits:
        lower = np.zeros(CMD_DIM)
        upper = np.zeros(CMD_DIM)
        for side in SIDES:
            jids = [self.model.joint(n).id for n in self.arm_joint_names[side]]
            lower[arm_slice(side)] = self.model.jnt_range[jids, 0]
            upper[arm_slice(side)] = self.model.jnt_range[jids, 1]
            lower[gripper_index(side)] = GRIPPER_CLOSED_M
            upper[gripper_index(side)] = GRIPPER_OPEN_M
        return JointLimits(lower=lower, upper=upper)

    def _collision_geoms(self, bodies: list) -> list:
        ids = {self.model.body(b).id for b in bodies}
        return [
            g for g in range(self.model.ngeom)
            if self.model.geom_bodyid[g] in ids and (self.model.geom_contype[g] or self.model.geom_conaffinity[g])
        ]

    def clearance(self, side: Side, q_arm: Mapping[Side, NDArray[np.float64]], max_dist: float = 0.5) -> float:
        """Minimum distance (m) between ``side``'s arm and the torso or the other arm.

        ``q_arm`` should contain both arms (an omitted arm keeps its last pose).
        Uses the official collision meshes; negative values mean penetration.
        """
        self._set_arms(q_arm)
        mujoco.mj_kinematics(self.model, self.data)
        other = "right" if side == "left" else "left"
        fromto = np.zeros(6)
        return min(
            mujoco.mj_geomDistance(self.model, self.data, a, b, max_dist, fromto)
            for a in self._arm_geoms[side]
            for b in self._torso_geoms + self._arm_geoms[other]
        )

    def _set_arms(self, q_arm: Mapping[Side, NDArray[np.float64]]) -> None:
        for side, q in q_arm.items():
            self.data.qpos[self.qpos_adr[side]] = q

    def fk(self, q_arm: Mapping[Side, NDArray[np.float64]]) -> Dict[Side, Pose]:
        """TCP poses (base frame) for the given arm joint positions."""
        self._set_arms(q_arm)
        mujoco.mj_kinematics(self.model, self.data)
        return {
            side: (
                self.data.site_xpos[self.site_id[side]].copy(),
                self.data.site_xmat[self.site_id[side]].reshape(3, 3).copy(),
            )
            for side in q_arm
        }

    def gravity_torques(self, q_arm: Mapping[Side, NDArray[np.float64]]) -> Dict[Side, NDArray[np.float64]]:
        """Joint torques (Nm) that cancel gravity at ``q_arm`` according to the official model.

        Positive torque follows the joint axis, which matches the raw Damiao motor
        convention used by ``openarm_can`` (motor angle == URDF joint angle).
        """
        self.data.qvel[:] = 0.0
        self._set_arms(q_arm)
        mujoco.mj_forward(self.model, self.data)
        return {side: self.data.qfrc_bias[self.dof_adr[side]].copy() for side in q_arm}
