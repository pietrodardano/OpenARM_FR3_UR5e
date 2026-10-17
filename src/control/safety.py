"""
Joint-space safety filter shared by every robot backend.

Runs inside the backend's low-level loop (hardware control thread or physics
loop), so no matter what a policy commands, the setpoint that reaches the motors
is always (1) inside the official joint limits minus a margin (a joint that
started inside the margin may return to, but not go beyond, its start pose) and (2) moving no
faster than the configured per-joint velocity limit. A policy that jumps its
target therefore produces a bounded-velocity motion, not a step.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray

from .robot_interface import ARM_DOF, CMD_DIM, SIDES, JointLimits, gripper_index


class JointSafetyFilter:
    """Clips targets to joint limits and rate-limits the setpoint.

    Args:
        limits: Position limits of the 16-dim command vector.
        max_joint_velocity: Arm joint velocity limit (rad/s), scalar or (7,) per joint.
        max_gripper_velocity: Finger opening speed limit (m/s).
        joint_limit_margin: Distance (rad) kept from the arm joint limits.
    """

    def __init__(
        self,
        limits: JointLimits,
        max_joint_velocity: Union[float, Sequence[float]],
        max_gripper_velocity: float,
        joint_limit_margin: float = 0.0,
    ) -> None:
        self.limits = limits
        self.margin = joint_limit_margin
        arm_vmax = np.broadcast_to(np.asarray(max_joint_velocity, dtype=np.float64), (ARM_DOF,))
        self.vmax = np.zeros(CMD_DIM)
        for k, side in enumerate(SIDES):
            base = k * (ARM_DOF + 1)
            self.vmax[base : base + ARM_DOF] = arm_vmax
            self.vmax[gripper_index(side)] = max_gripper_velocity
        self._setpoint: Optional[NDArray[np.float64]] = None
        self._anchor: Optional[NDArray[np.float64]] = None  # pose at the last reset()

    @property
    def setpoint(self) -> Optional[NDArray[np.float64]]:
        return None if self._setpoint is None else self._setpoint.copy()

    def reset(self, measured: NDArray[np.float64]) -> NDArray[np.float64]:
        """Re-anchors the setpoint at the measured pose (clipped into the limits).

        Must be called before the first :meth:`step` and after any discontinuity
        (start-up, fault recovery) so the motors never see a jump.
        """
        self._setpoint = self.limits.clip(np.asarray(measured, dtype=np.float64), 0.0)
        self._anchor = self._setpoint.copy()
        return self._setpoint.copy()

    def step(self, target: NDArray[np.float64], dt: float) -> NDArray[np.float64]:
        """Advances the setpoint towards ``target`` by at most ``vmax * dt``."""
        if self._setpoint is None:
            raise RuntimeError("JointSafetyFilter.step() called before reset()")
        # Keep targets inside the limits minus the margin, widened to include the pose at
        # the last reset(): a joint that started inside the margin (joint 4 at 0 rad in the
        # hanging pose) is not pulled away and can always return there, but never beyond.
        soft_lo = self.limits.clip(self.limits.lower, self.margin)
        soft_hi = self.limits.clip(self.limits.upper, self.margin)
        lo = np.minimum(soft_lo, self._anchor)
        hi = np.maximum(soft_hi, self._anchor)
        goal = np.clip(np.asarray(target, dtype=np.float64), lo, hi)
        max_step = self.vmax * dt
        self._setpoint = self._setpoint + np.clip(goal - self._setpoint, -max_step, max_step)
        return self._setpoint.copy()
