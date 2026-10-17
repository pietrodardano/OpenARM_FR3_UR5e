"""
Joint position controller for the OpenArm v1 bimanual robot.

Sends absolute or delta joint targets (rad) and gripper openings (m) to a
:class:`~src.control.robot_interface.RobotBackend`. Joint-limit enforcement and
velocity limiting happen in the backend's safety filter at the control-thread
rate; this class clips to limits so the stored command stays reachable, and
keeps that command as the reference for delta commands.
"""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
from numpy.typing import NDArray

from .robot_interface import RobotBackend, Side, arm_slice, gripper_index


class JointPositionController:
    """Absolute / delta joint position commands for both arms."""

    def __init__(self, backend: RobotBackend) -> None:
        self.backend = backend
        self.q_cmd: Optional[NDArray[np.float64]] = None  # last 16-dim command

    def reset(self) -> None:
        """Re-anchors the command at the backend's current targets (call after start / move_joints).

        Targets, not measured joints: the measured pose includes gravity sag and
        tracking lag, which would otherwise be baked into every following delta.
        """
        self.q_cmd = self.backend.get_joint_targets()

    def command(
        self,
        q_arm: Optional[Mapping[Side, NDArray[np.float64]]] = None,
        gripper_m: Optional[Mapping[Side, float]] = None,
    ) -> NDArray[np.float64]:
        """Absolute joint targets (7,) per arm and gripper openings (0 closed .. 0.044 m open).

        Arms / grippers that are omitted keep their previous command.
        """
        if self.q_cmd is None:
            self.reset()
        vec = self.q_cmd.copy()
        for side, q in (q_arm or {}).items():
            vec[arm_slice(side)] = q
        for side, g in (gripper_m or {}).items():
            vec[gripper_index(side)] = g
        vec = self.backend.limits.clip(vec)
        self.q_cmd = vec
        self.backend.set_target(vec)
        return vec.copy()

    def command_delta(
        self,
        dq_arm: Mapping[Side, NDArray[np.float64]],
        gripper_m: Optional[Mapping[Side, float]] = None,
    ) -> NDArray[np.float64]:
        """Joint deltas (7,) per arm, added to the previous command (not the measured pose)."""
        if self.q_cmd is None:
            self.reset()
        q_arm = {side: self.q_cmd[arm_slice(side)] + np.asarray(dq) for side, dq in dq_arm.items()}
        return self.command(q_arm, gripper_m)
