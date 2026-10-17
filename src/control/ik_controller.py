"""
IK (end-effector pose) controller: Cartesian actions on top of joint position control.

OpenArm has no built-in Cartesian control (no ``moveL``/``servoL`` like UR), so
this class provides it: TCP pose commands -> workspace clamp -> step limit ->
:class:`~src.planning.openarm_ik.BimanualIK` (official model) -> joint targets
-> :class:`~src.control.joint_position_controller.JointPositionController`.

- :meth:`command_poses` / :meth:`command_delta`: streaming (servoL-like), one
  call per control step, absolute or relative TCP targets.
- :meth:`move_l`: blocking straight-line TCP motion (moveL-like), for scripted skills.
- :meth:`get_tcp_pose`: measured TCP poses via forward kinematics.

All poses are in the robot base frame of the official model: +X forward,
+Y left, +Z up, origin at the bottom of the torso. Delta actions are applied to
the previous *commanded* pose; if IK cannot fully reach a target, the commanded
pose is re-anchored at what was actually reached so errors do not wind up.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation, Slerp

from ..planning.openarm_ik import BimanualIK
from ..planning.openarm_kinematics import OpenArmKinematics, Pose
from .joint_position_controller import JointPositionController
from .robot_interface import SIDES, RobotBackend, Side, arm_slice, gripper_index, minimum_jerk

logger = logging.getLogger("RoboTokens.IKController")


@dataclass
class WorkspaceBox:
    """Axis-aligned TCP position bounds (base frame, m)."""

    low: Sequence[float] = (0.15, -0.55, 0.15)
    high: Sequence[float] = (0.65, 0.55, 0.90)

    def clip(self, p: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.clip(p, self.low, self.high)


@dataclass
class IKControllerConfig:
    max_pos_step: float = 0.02  # m, cap on TCP translation per streaming command
    max_rot_step: float = 0.10  # rad, cap on TCP rotation per streaming command
    workspace: Dict[Side, WorkspaceBox] = field(
        default_factory=lambda: {
            "left": WorkspaceBox((0.15, -0.15, 0.15), (0.65, 0.55, 0.90)),
            "right": WorkspaceBox((0.15, -0.55, 0.15), (0.65, 0.15, 0.90)),
        }
    )
    ik_iters: int = 30  # differential-IK iterations per streaming step
    max_ik_pos_error: float = 0.02  # m; warn if a step ends further than this from its target


def _limit_step(p_prev, R_prev, p, R, max_dp: float, max_dr: float):
    """Caps the translation / rotation between two poses."""
    dp = p - p_prev
    n = np.linalg.norm(dp)
    if n > max_dp:
        p = p_prev + dp * (max_dp / n)
    drot = Rotation.from_matrix(R) * Rotation.from_matrix(R_prev).inv()
    angle = np.linalg.norm(drot.as_rotvec())
    if angle > max_dr:
        R = (Rotation.from_rotvec(drot.as_rotvec() * (max_dr / angle)) * Rotation.from_matrix(R_prev)).as_matrix()
    return p, R


class IKController:
    """Streams TCP pose targets for both arms through IK.

    Args:
        backend: Real or simulated robot.
        kinematics: Official model wrapper (FK, limits). Not shared with the hardware thread.
        cfg: Step limits and workspace.
        ik: Solver (built from ``kinematics`` if omitted).
        joint_controller: Downstream joint controller (built if omitted).
    """

    def __init__(
        self,
        backend: RobotBackend,
        kinematics: OpenArmKinematics,
        cfg: Optional[IKControllerConfig] = None,
        ik: Optional[BimanualIK] = None,
        joint_controller: Optional[JointPositionController] = None,
    ) -> None:
        self.backend = backend
        self.kin = kinematics
        self.cfg = cfg or IKControllerConfig()
        self.ik = ik or BimanualIK(kinematics)
        self.joints = joint_controller or JointPositionController(backend)
        self.q_ref: Optional[Dict[Side, NDArray[np.float64]]] = None  # last IK solution (seed)
        self.pose_cmd: Optional[Dict[Side, Pose]] = None  # last commanded (reached) TCP poses
        self.gripper_cmd: Dict[Side, float] = {}

    @classmethod
    def from_config(
        cls, cfg: Dict[str, Any], backend: RobotBackend, kinematics: Optional[OpenArmKinematics] = None
    ) -> "IKController":
        """Builds the controller (and its joint controller) from ``configs/task_config.yaml``."""
        ik = cfg["controllers"]["ik"]
        ik_cfg = IKControllerConfig(
            max_pos_step=ik["max_pos_step_m"],
            max_rot_step=ik["max_rot_step_rad"],
            ik_iters=ik["ik_iters"],
            workspace={s: WorkspaceBox(tuple(ik["workspace"][s]["low"]), tuple(ik["workspace"][s]["high"])) for s in SIDES},
        )
        kin = kinematics or OpenArmKinematics(cfg["robot"]["mjcf_path"])
        return cls(backend, kin, ik_cfg)

    # ------------------------------------------------------------- state

    def reset(self) -> None:
        """Re-anchors commands at the backend's current targets (call after start / move_joints)."""
        self.joints.reset()
        q_cmd = self.joints.q_cmd
        self.q_ref = {s: q_cmd[arm_slice(s)].copy() for s in SIDES}
        self.pose_cmd = self.kin.fk(self.q_ref)
        self.gripper_cmd = {s: float(q_cmd[gripper_index(s)]) for s in SIDES}

    def get_tcp_pose(self) -> Dict[Side, Pose]:
        """Measured TCP poses (FK of the measured joints)."""
        state = self.backend.get_state()
        return self.kin.fk({s: state.arms[s].q for s in SIDES})

    # --------------------------------------------------------- streaming

    def command_poses(
        self,
        poses: Mapping[Side, Pose],
        gripper_m: Optional[Mapping[Side, float]] = None,
        limit_step: bool = True,
    ) -> Dict[Side, float]:
        """Absolute TCP targets for any subset of arms. Returns per-arm IK position error (m)."""
        if self.pose_cmd is None:
            self.reset()
        targets: Dict[Side, Pose] = {}
        for side, (p, R) in poses.items():
            p = self.cfg.workspace[side].clip(np.asarray(p, dtype=np.float64))
            R = np.asarray(R, dtype=np.float64)
            if limit_step:
                p, R = _limit_step(*self.pose_cmd[side], p, R, self.cfg.max_pos_step, self.cfg.max_rot_step)
            targets[side] = (p, R)

        result = self.ik.solve(targets, self.q_ref, max_iters=self.cfg.ik_iters)
        for side in targets:
            self.q_ref[side] = result.q_arm[side]
        self.pose_cmd.update(self.kin.fk({s: self.q_ref[s] for s in targets}))
        for side, err in result.pos_error.items():
            if err > self.cfg.max_ik_pos_error:
                logger.warning("%s TCP target not reachable (IK error %.1f mm)", side, err * 1e3)
        if gripper_m:
            self.gripper_cmd.update(gripper_m)
        self.joints.command({s: self.q_ref[s] for s in targets}, gripper_m)
        return result.pos_error

    def command_delta(
        self,
        dpos: Mapping[Side, NDArray[np.float64]],
        drot: Optional[Mapping[Side, NDArray[np.float64]]] = None,
        gripper_m: Optional[Mapping[Side, float]] = None,
    ) -> Dict[Side, float]:
        """Relative TCP motion per arm: translation (3,) m and rotation vector (3,) rad, both
        in the base frame, applied to the previous *commanded* pose."""
        if self.pose_cmd is None:
            self.reset()
        poses: Dict[Side, Pose] = {}
        for side, dp in dpos.items():
            p0, R0 = self.pose_cmd[side]
            dr = np.zeros(3) if drot is None or side not in drot else np.asarray(drot[side])
            poses[side] = (p0 + np.asarray(dp), (Rotation.from_rotvec(dr) * Rotation.from_matrix(R0)).as_matrix())
        return self.command_poses(poses, gripper_m)

    # ---------------------------------------------------------- blocking

    def move_l(
        self,
        poses: Mapping[Side, Pose],
        duration: float = 2.0,
        gripper_m: Optional[Mapping[Side, float]] = None,
        rate_hz: float = 50.0,
    ) -> Dict[Side, float]:
        """Blocking straight-line TCP motion (minimum-jerk timing), like UR ``moveL``.

        Starts from the current *commanded* pose. Returns the final IK position errors.
        """
        if self.pose_cmd is None:
            self.reset()
        start = {s: self.pose_cmd[s] for s in poses}
        slerps = {
            s: Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([start[s][1], np.asarray(poses[s][1])])))
            for s in poses
        }
        g_start = {s: self.gripper_cmd.get(s, 0.0) for s in (gripper_m or {})}
        steps = max(1, int(duration * rate_hz))
        errors: Dict[Side, float] = {}
        for k in range(1, steps + 1):
            a = minimum_jerk(k / steps)
            interp = {
                s: (start[s][0] + a * (np.asarray(poses[s][0]) - start[s][0]), slerps[s](a).as_matrix())
                for s in poses
            }
            grip = {s: g_start[s] + a * (gripper_m[s] - g_start[s]) for s in (gripper_m or {})}
            errors = self.command_poses(interp, grip or None, limit_step=False)
            self.backend.advance(1.0 / rate_hz)
            if self.backend.fault:
                raise RuntimeError(f"move_l aborted: {self.backend.fault}")
        return errors

    def joint_targets(self) -> Dict[Side, NDArray[np.float64]]:
        """Latest IK joint solution per arm."""
        return {s: self.joints.q_cmd[arm_slice(s)].copy() for s in SIDES}
