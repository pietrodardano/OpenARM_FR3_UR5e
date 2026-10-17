"""Model-based kinematics (FK, Jacobian, IK, CLIK servo) for any serial arm in a MuJoCo scene.

One code path serves the 7-DoF OpenArm and FR3 and the 6-DoF UR5e. The MJCF is
the kinematic source of truth; ``ArmModel`` works on a private ``MjData`` so it
never disturbs a live simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import mujoco
import numpy as np


@dataclass
class IKResult:
    q: np.ndarray
    position_error_m: float
    axis_error_rad: float
    converged: bool
    iterations: int


class ArmModel:
    """A serial arm identified by joint names and a TCP body or site."""

    def __init__(
        self,
        model: "mujoco.MjModel",
        joint_names: Sequence[str],
        tcp_name: str,
        tcp_type: str = "body",
    ) -> None:
        if tcp_type not in {"body", "site"}:
            raise ValueError("tcp_type must be 'body' or 'site'.")
        self.model = model
        self.data = mujoco.MjData(model)
        self.joint_names = list(joint_names)
        joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.joint_names]
        if any(j < 0 for j in joint_ids):
            missing = [n for n, j in zip(self.joint_names, joint_ids) if j < 0]
            raise ValueError(f"Unknown joints: {missing}")
        self.joint_ids = np.array(joint_ids)
        self.qpos_idx = model.jnt_qposadr[self.joint_ids]
        self.dof_idx = model.jnt_dofadr[self.joint_ids]
        self.limits = model.jnt_range[self.joint_ids].copy()
        self.n = len(self.joint_names)
        self.tcp_type = tcp_type
        obj = mujoco.mjtObj.mjOBJ_BODY if tcp_type == "body" else mujoco.mjtObj.mjOBJ_SITE
        self.tcp_id = mujoco.mj_name2id(model, obj, tcp_name)
        if self.tcp_id < 0:
            raise ValueError(f"Unknown TCP {tcp_type} '{tcp_name}'.")
        self.q_center = self.limits.mean(axis=1)
        mujoco.mj_resetData(model, self.data)

    # ------------------------------------------------------------------ FK
    def _set(self, q: np.ndarray) -> None:
        self.data.qpos[self.qpos_idx] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def set_environment_qpos(self, qpos: np.ndarray) -> None:
        """Copy the full scene qpos so a mounted arm sees a consistent world (rarely needed)."""
        self.data.qpos[:] = qpos

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """TCP world position (3,) and rotation matrix (3, 3)."""
        self._set(np.asarray(q, dtype=np.float64))
        if self.tcp_type == "body":
            return self.data.xpos[self.tcp_id].copy(), self.data.xmat[self.tcp_id].reshape(3, 3).copy()
        return self.data.site_xpos[self.tcp_id].copy(), self.data.site_xmat[self.tcp_id].reshape(3, 3).copy()

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """6 x n Jacobian [linear; angular] of the TCP in the world frame."""
        self._set(np.asarray(q, dtype=np.float64))
        jp = np.zeros((3, self.model.nv))
        jr = np.zeros((3, self.model.nv))
        if self.tcp_type == "body":
            mujoco.mj_jacBody(self.model, self.data, jp, jr, self.tcp_id)
        else:
            mujoco.mj_jacSite(self.model, self.data, jp, jr, self.tcp_id)
        return np.vstack((jp[:, self.dof_idx], jr[:, self.dof_idx]))

    # ------------------------------------------------------------------ IK
    @staticmethod
    def _rotation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Axis-angle vector rotating ``current`` onto ``target`` (world frame)."""
        rel = target @ current.T
        cos_angle = np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        skew = np.array([rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]])
        if angle < 1e-8:
            return 0.5 * skew
        if np.pi - angle < 1e-5:  # near 180 deg: use the symmetric part
            axis = np.sqrt(np.maximum((np.diag(rel) + 1.0) / 2.0, 0.0))
            return angle * axis / (np.linalg.norm(axis) + 1e-12)
        return angle * skew / (2.0 * np.sin(angle))

    def task_error(
        self,
        q: np.ndarray,
        target_pos: np.ndarray,
        target_rot: Optional[np.ndarray] = None,
        target_axis: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """6-vector error. Orientation is full (target_rot), tool-z-axis only (target_axis) or off."""
        pos, rot = self.fk(q)
        error = np.zeros(6)
        error[:3] = np.asarray(target_pos) - pos
        if target_rot is not None:
            error[3:] = self._rotation_error(rot, np.asarray(target_rot))
        elif target_axis is not None:
            axis = np.asarray(target_axis, dtype=np.float64)
            error[3:] = np.cross(rot[:, 2], axis / np.linalg.norm(axis))
        return error

    def _dls_step(
        self,
        q: np.ndarray,
        error: np.ndarray,
        orientation_weight: float,
        damping: float,
        null_gain: float,
        q_posture: np.ndarray,
    ) -> np.ndarray:
        weights = np.array([1.0, 1.0, 1.0] + [orientation_weight] * 3)
        jac = self.jacobian(q)
        jw = weights[:, None] * jac
        ew = weights * error
        jjt = jw @ jw.T + damping**2 * np.eye(6)
        pinv = jw.T @ np.linalg.inv(jjt)
        step = pinv @ ew
        if self.n > 6 and null_gain > 0.0:
            # Null-space posture term: pull towards q_posture (joint-limit centring by default).
            span = np.maximum(self.limits[:, 1] - self.limits[:, 0], 1e-6)
            gradient = (q_posture - q) / span**2
            step += (np.eye(self.n) - pinv @ jw) @ (null_gain * gradient)
        return step

    def solve_ik(
        self,
        target_pos: np.ndarray,
        q_seed: np.ndarray,
        target_rot: Optional[np.ndarray] = None,
        target_axis: Optional[np.ndarray] = None,
        orientation_weight: float = 0.5,
        pos_tol_m: float = 1e-3,
        ori_tol_rad: float = 0.02,
        max_iterations: int = 200,
        damping: float = 0.05,
        max_step_rad: float = 0.1,
        null_gain: float = 0.5,
        q_posture: Optional[np.ndarray] = None,
        restarts: int = 0,
        rng: Optional[np.random.Generator] = None,
    ) -> IKResult:
        """Warm-started weighted DLS IK with joint-limit clipping and a null-space posture term.

        The first attempt starts from ``q_seed`` so that solutions stay on the
        same branch as the measured arm. Random restarts are only used if that fails.
        """
        q_seed = np.clip(np.asarray(q_seed, dtype=np.float64), self.limits[:, 0], self.limits[:, 1])
        posture = self.q_center if q_posture is None else np.asarray(q_posture, dtype=np.float64)
        rng = rng or np.random.default_rng(0)
        best: Optional[IKResult] = None
        for attempt in range(restarts + 1):
            q = q_seed.copy()
            if attempt > 0:
                q = np.clip(
                    q_seed + rng.normal(0.0, 0.6, self.n), self.limits[:, 0], self.limits[:, 1]
                )
            for it in range(max_iterations):
                error = self.task_error(q, target_pos, target_rot, target_axis)
                pos_err = float(np.linalg.norm(error[:3]))
                ori_err = float(np.linalg.norm(error[3:]))
                oriented = target_rot is not None or target_axis is not None
                if pos_err <= pos_tol_m and (not oriented or ori_err <= ori_tol_rad):
                    return IKResult(q, pos_err, ori_err, True, it)
                step = self._dls_step(q, error, orientation_weight if oriented else 0.0, damping, null_gain, posture)
                norm = np.max(np.abs(step))
                if norm > max_step_rad:
                    step *= max_step_rad / norm
                q = np.clip(q + step, self.limits[:, 0], self.limits[:, 1])
            error = self.task_error(q, target_pos, target_rot, target_axis)
            candidate = IKResult(q, float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:])), False, max_iterations)
            if best is None or candidate.position_error_m < best.position_error_m:
                best = candidate
        assert best is not None
        return best

    # --------------------------------------------------------------- CLIK
    def servo_step(
        self,
        q_meas: np.ndarray,
        target_pos: np.ndarray,
        dt: float,
        target_rot: Optional[np.ndarray] = None,
        target_axis: Optional[np.ndarray] = None,
        gain: float = 8.0,
        orientation_weight: float = 0.5,
        damping: float = 0.05,
        null_gain: float = 0.5,
        max_joint_vel: float = 1.5,
        q_posture: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """DEPRECATED (see docs/control_stack.md): proportional-only, lags speed/gain; replaced
        by the mink servo with feedforward (plan step 2).

        One resolved-rate closed-loop step anchored on the *measured* joints.

        Returns the next joint setpoint ``q_meas + dq*dt``. Because the error is
        recomputed from measured q every cycle, tracking error of the plant is
        absorbed instead of accumulating as an open-loop offset.
        """
        q_meas = np.asarray(q_meas, dtype=np.float64)
        error = self.task_error(q_meas, target_pos, target_rot, target_axis)
        oriented = target_rot is not None or target_axis is not None
        posture = self.q_center if q_posture is None else np.asarray(q_posture, dtype=np.float64)
        dq = self._dls_step(q_meas, gain * error, orientation_weight if oriented else 0.0, damping, null_gain, posture)
        peak = np.max(np.abs(dq))
        if peak > max_joint_vel:
            dq *= max_joint_vel / peak
        return np.clip(q_meas + dq * dt, self.limits[:, 0], self.limits[:, 1])
