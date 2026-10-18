"""What the demo needs from the world: a :class:`Detector` and :class:`Skills`.

The state machine (:mod:`.demo`) depends only on these two protocols. :class:`RobotSkills`
implements the skills on the real (or fake-bus) arm; :class:`ManualDetector` serves object
positions from the config when vision is not used.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from ..config import Config
from ..robot import gripper
from ..robot.driver import Arm
from ..robot.model import ArmModel
from ..robot.motion import Motion, Target
from ..sensing.payload import MassEstimate, measure_load

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    """An object on the table, robot base frame."""

    x: float
    y: float
    confidence: float = 1.0
    label: str = ""


class Detector(Protocol):
    def detect(self, prompt: str) -> List[Detection]: ...


class Skills(Protocol):
    def go_home(self) -> None: ...
    def move_to(self, target: Target) -> None: ...
    def grasp(self) -> bool: ...
    def release(self) -> None: ...
    def to_weigh_pose(self) -> None: ...
    def measure(self, label: str) -> MassEstimate: ...
    def read_torques(self) -> NDArray: ...
    def wait(self, dt: float) -> None: ...
    def abort(self) -> None: ...


class ManualDetector:
    """Fixed object positions (``task.manual_objects`` or ``--objects``)."""

    def __init__(self, objects: Sequence[Detection]) -> None:
        self.objects = list(objects)

    def detect(self, prompt: str) -> List[Detection]:
        return [Detection(o.x, o.y, o.confidence, o.label or prompt) for o in self.objects]


class RobotSkills:
    """:class:`Skills` on one arm: checked motion, torque-mode gripper, torque-based weighing."""

    def __init__(self, arm: Arm, model: ArmModel, cfg: Config, require_object: bool = True) -> None:
        self.arm = arm
        self.model = model
        self.cfg = cfg
        self.motion = Motion(arm, model, cfg)
        self.require_object = require_object  # False: rehearse motions without objects

    def go_home(self) -> None:
        self.motion.go_home()

    def move_to(self, target: Target) -> None:
        self.motion.move_to(target)

    def grasp(self) -> bool:
        """Torque-mode grip. False (and opened again) if the fingers closed on nothing."""
        opening = gripper.grip(self.arm, self.cfg.gripper)
        if opening < self.cfg.gripper.min_held_opening_m and self.require_object:
            self.release()
            return False
        return True

    def release(self) -> None:
        gripper.release(self.arm, self.cfg.gripper)
        self.motion.reset()

    def to_weigh_pose(self) -> None:
        w = self.cfg.task.weigh.pose
        self.move_to(Target(w.x, w.y, self.cfg.table_z + w.above_table_m, 0.0, float(np.radians(w.pitch_deg))))

    def measure(self, label: str) -> MassEstimate:
        return measure_load(self.arm, self.model, self.cfg.task.weigh, label)

    def read_torques(self) -> NDArray:
        st = self.arm.state()
        if st.fault:
            raise RuntimeError(f"robot fault: {st.fault}")
        return st.tau.copy()

    def wait(self, dt: float) -> None:
        self.arm.advance(dt)
        if self.arm.fault:
            raise RuntimeError(f"robot fault: {self.arm.fault}")

    def abort(self) -> None:
        """Best effort, never raises: lift 10 cm if low over the table, then park beside it."""
        try:
            if self.arm.fault:
                return
            p, R = self.motion.tcp_command()
            if p[2] < self.cfg.table_z + 0.20:
                finger = R[:, 2]  # tool z = finger direction
                pitch = float(np.arcsin(np.clip(-finger[2], -1.0, 1.0)))
                yaw = float(np.arctan2(finger[1], finger[0]))
                self.move_to(Target(float(p[0]), float(p[1]), float(p[2]) + 0.10, yaw, pitch))
        except Exception as exc:  # noqa: BLE001
            logger.error("abort: lift failed: %s", exc)
        try:
            if not self.arm.fault:
                self.motion.park()
        except Exception as exc:  # noqa: BLE001
            logger.error("abort: park failed: %s", exc)
