"""
Backend-agnostic robot interface for the OpenArm v1 bimanual robot.

Every backend (real hardware over SocketCAN, MuJoCo with the official model, the
fake bus used in tests) implements :class:`RobotBackend`. Controllers and
policies only ever talk to this interface, so the same code runs in simulation
and on the physical robot.

Joint conventions follow the official OpenArm sources (``openarm_description``
URDF == ``openarm_mujoco`` v1 MJCF == raw ``openarm_can`` motor angles):

- 7 arm joints per side, ``openarm_<side>_joint1..7``, radians.
- 1 gripper per side, expressed as the parallel-link finger opening in metres,
  ``0.0`` = closed, ``0.044`` = fully open (``openarm_<side>_finger_joint1``).

Flat "joint command vectors" have 16 entries, ordered like the MJCF actuators:
``[left j1..j7, left gripper, right j1..j7, right gripper]``.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Literal, Mapping, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

Side = Literal["left", "right"]
SIDES: Tuple[Side, Side] = ("left", "right")
ARM_DOF = 7
CMD_DIM = 2 * (ARM_DOF + 1)  # 16

GRIPPER_OPEN_M = 0.044
GRIPPER_CLOSED_M = 0.0

JOINT_NAMES = [
    name
    for side in SIDES
    for name in [f"openarm_{side}_joint{i}" for i in range(1, ARM_DOF + 1)] + [f"openarm_{side}_finger_joint1"]
]


def side_slice(side: Side) -> slice:
    """Slice of a side's 8 entries (7 joints + gripper) in a 16-dim command vector."""
    start = 0 if side == "left" else ARM_DOF + 1
    return slice(start, start + ARM_DOF + 1)


def arm_slice(side: Side) -> slice:
    """Slice of a side's 7 arm joints in a 16-dim command vector."""
    start = 0 if side == "left" else ARM_DOF + 1
    return slice(start, start + ARM_DOF)


def gripper_index(side: Side) -> int:
    """Index of a side's gripper entry in a 16-dim command vector."""
    return ARM_DOF if side == "left" else 2 * ARM_DOF + 1


def pack_command(q_arm: Mapping[Side, NDArray[np.float64]], gripper: Mapping[Side, float]) -> NDArray[np.float64]:
    """Packs per-side arm joints (7,) and gripper openings (m) into a 16-dim vector."""
    vec = np.zeros(CMD_DIM, dtype=np.float64)
    for side in SIDES:
        vec[arm_slice(side)] = q_arm[side]
        vec[gripper_index(side)] = gripper[side]
    return vec


@dataclass
class ArmState:
    """Measured and commanded state of one arm."""

    q: NDArray[np.float64]  # (7,) measured joint positions, rad
    dq: NDArray[np.float64]  # (7,) measured joint velocities, rad/s
    tau: NDArray[np.float64]  # (7,) measured joint torques, Nm
    gripper: float  # measured finger opening, m
    q_setpoint: NDArray[np.float64]  # (7,) position setpoint sent this tick (after safety filter)
    gripper_setpoint: float  # gripper setpoint sent this tick, m
    motor_errors: Tuple[int, ...] = ()  # raw Damiao status nibble per motor (0/1 = OK, >=8 = fault)


@dataclass
class RobotState:
    """Synchronized snapshot of both arms."""

    timestamp: float  # time.monotonic() of the control tick that produced it
    arms: Dict[Side, ArmState]
    fault: Optional[str] = None
    stats: Dict[str, float] = field(default_factory=dict)

    @property
    def q(self) -> NDArray[np.float64]:
        """(14,) measured arm joints, left then right."""
        return np.concatenate([self.arms[s].q for s in SIDES])

    @property
    def dq(self) -> NDArray[np.float64]:
        """(14,) measured arm joint velocities, left then right."""
        return np.concatenate([self.arms[s].dq for s in SIDES])

    @property
    def gripper(self) -> NDArray[np.float64]:
        """(2,) measured gripper openings in metres, left then right."""
        return np.array([self.arms[s].gripper for s in SIDES])

    def command_vector(self) -> NDArray[np.float64]:
        """Measured state laid out as a 16-dim joint command vector."""
        return pack_command({s: self.arms[s].q for s in SIDES}, {s: self.arms[s].gripper for s in SIDES})


@dataclass
class JointLimits:
    """Position limits for the 16-dim command vector (rad for arm joints, m for grippers)."""

    lower: NDArray[np.float64]
    upper: NDArray[np.float64]

    def clip(self, vec: NDArray[np.float64], margin: float = 0.0) -> NDArray[np.float64]:
        """Clips to ``[lower + margin, upper - margin]`` (margin applies to arm joints only)."""
        m = np.full(CMD_DIM, margin)
        for side in SIDES:
            m[gripper_index(side)] = 0.0
        return np.clip(vec, self.lower + m, self.upper - m)


def minimum_jerk(s: float) -> float:
    """Minimum-jerk time scaling: maps normalized time [0, 1] to progress [0, 1]."""
    s = min(max(s, 0.0), 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


class RobotBackend(ABC):
    """Common interface of the real OpenArm driver and the MuJoCo backend.

    Backends run their own low-level loop (a Python thread for hardware and
    real-time simulation) that tracks the latest joint targets through a
    :class:`~src.control.safety.JointSafetyFilter`. Callers only set targets and
    read state; both calls are thread-safe and non-blocking.
    """

    limits: JointLimits

    # -------------------------------------------------------------- lifecycle

    @abstractmethod
    def start(self) -> None:
        """Connects, enables actuators and starts holding the current pose."""

    @abstractmethod
    def stop(self) -> None:
        """Safe shutdown (hardware: move to the rest pose, then disable motors)."""

    def __enter__(self) -> "RobotBackend":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # ------------------------------------------------------------------ I/O

    @abstractmethod
    def get_state(self) -> RobotState:
        """Latest synchronized state of both arms."""

    @abstractmethod
    def set_joint_targets(
        self,
        q_arm: Optional[Mapping[Side, NDArray[np.float64]]] = None,
        gripper: Optional[Mapping[Side, float]] = None,
    ) -> None:
        """Sets new position targets. Sides / grippers that are omitted keep their target."""

    @abstractmethod
    def get_joint_targets(self) -> NDArray[np.float64]:
        """Current 16-dim target vector (before the safety filter)."""

    def advance(self, dt: float) -> None:
        """Lets ``dt`` seconds of robot time pass.

        Real-time backends sleep until the caller's next ``dt`` deadline (so a
        policy loop calling this keeps a fixed rate); the synchronous MuJoCo
        backend steps physics for ``dt`` instead.
        """
        now = time.monotonic()
        deadline = getattr(self, "_advance_deadline", None)
        if deadline is None or deadline < now - dt:  # first call, or fell behind: resync
            deadline = now
        deadline += dt
        self._advance_deadline = deadline
        time.sleep(max(0.0, deadline - time.monotonic()))

    # --------------------------------------------------------------- faults

    @property
    def fault(self) -> Optional[str]:
        return self.get_state().fault

    def clear_fault(self) -> None:
        """Clears a latched fault; the robot keeps holding its measured pose."""

    # -------------------------------------------------------------- helpers

    def set_target(self, vec: NDArray[np.float64]) -> None:
        """Sets all 16 targets at once (``[left j1..j7, left grip, right j1..j7, right grip]``)."""
        self._set_vector(np.asarray(vec, dtype=np.float64))

    def get_joints(self) -> NDArray[np.float64]:
        """(14,) measured arm joints, left then right."""
        return self.get_state().q

    def move_joints(
        self,
        q_arm: Mapping[Side, NDArray[np.float64]],
        gripper: Optional[Mapping[Side, float]] = None,
        duration: float = 3.0,
        rate_hz: float = 100.0,
    ) -> None:
        """Blocking minimum-jerk joint-space move from the current setpoint.

        Starts from the current targets rather than the measured pose: the measured
        pose includes gravity sag, and starting there would make the first tick jump.
        """
        start = self.get_joint_targets()
        goal = start.copy()
        for side, q in q_arm.items():
            goal[arm_slice(side)] = q
        for side, g in (gripper or {}).items():
            goal[gripper_index(side)] = g
        goal = self.limits.clip(goal)
        steps = max(1, int(duration * rate_hz))
        for k in range(1, steps + 1):
            vec = start + minimum_jerk(k / steps) * (goal - start)
            self._set_vector(vec)
            self.advance(1.0 / rate_hz)
            if self.fault:
                raise RuntimeError(f"move_joints aborted: {self.fault}")

    def _set_vector(self, vec: NDArray[np.float64]) -> None:
        self.set_joint_targets(
            {s: vec[arm_slice(s)] for s in SIDES},
            {s: float(vec[gripper_index(s)]) for s in SIDES},
        )
