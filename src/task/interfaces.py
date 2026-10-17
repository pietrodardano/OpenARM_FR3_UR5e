"""
Contracts between the weigh-and-sort task and the components that implement it.

The task layer never imports the vision pipeline or the robot driver directly. It
talks to three structural interfaces (``typing.Protocol``):

- :class:`ObjectDetector`: the vision branch (GroundingDINO + SAM2, optional VLM).
  It returns object positions on the table plane in the robot base frame.
- :class:`ManipulationSkills`: the real-robot controller (:mod:`src.task.openarm_adapter`):
  Cartesian ``move_l`` motions, gripper torque mode, joint-torque readout.
- :class:`PayloadEstimator`: the torque-based load measurement (``src/estimation/payload.py``).

Mocks for all three live in :mod:`src.task.mocks`; the OpenArm implementation is
:mod:`src.task.openarm_adapter`. Vision is mocked until its branch is merged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ObjectDetection:
    """One object found by the vision pipeline, on the table plane.

    Attributes:
        label: Open-vocabulary class from the detector (e.g. ``"soda can"``).
        x: Object center, robot base frame (+X forward), meters.
        y: Object center, robot base frame (+Y left), meters.
        yaw: Object rotation about base +Z, radians (0 if rotationally symmetric).
        confidence: Detector confidence in [0, 1].
        frame: Frame of ``x``/``y``. The contract is ``"base"``; anything else is rejected.
    """

    label: str
    x: float
    y: float
    yaw: float = 0.0
    confidence: float = 1.0
    frame: str = "base"

    @property
    def xy(self) -> NDArray[np.float64]:
        """Object center as a (2,) array in meters."""
        return np.array([self.x, self.y], dtype=np.float64)


@dataclass(frozen=True)
class TcpTarget:
    """Cartesian TCP goal in the robot base frame.

    Orientation = the hand-forward orientation of ``robot.home_qpos`` (fingers along +X,
    closing axis along Y), pitched down by ``pitch`` about Y, then turned by ``yaw`` about +Z.
    Pitch keeps the closing axis horizontal: still a lateral squeeze of an upright can.
    """

    x: float
    y: float
    z: float
    yaw: float = 0.0
    pitch: float = 0.0

    @property
    def position(self) -> NDArray[np.float64]:
        """TCP position as a (3,) array in meters."""
        return np.array([self.x, self.y, self.z], dtype=np.float64)


@dataclass
class MassEstimate:
    """Equivalent payload mass from joint torques.

    The absolute value is rough (model and friction errors); the difference between
    two estimates taken the same way is what the task decides on.

    Attributes:
        label: Name of the measured object.
        mass_kg: Equivalent payload mass, kg.
        mass_se_kg: Standard error of ``mass_kg``, kg.
        raw: Backend-specific sample (e.g. ``LoadSample`` from ``src.estimation.payload``).
    """

    label: str
    mass_kg: float
    mass_se_kg: float
    raw: Any = None


@dataclass
class ComparisonResult:
    """Which of two estimates is heavier, and how confidently.

    Attributes:
        heavier: Estimate with the larger mass.
        lighter: Estimate with the smaller mass.
        diff_kg: ``heavier.mass_kg - lighter.mass_kg`` (>= 0).
        z: ``diff_kg`` divided by the combined standard error.
        conclusive: ``z >= min_z``: safe to act on.
    """

    heavier: MassEstimate
    lighter: MassEstimate
    diff_kg: float
    z: float
    conclusive: bool


@runtime_checkable
class ObjectDetector(Protocol):
    """Vision pipeline: finds objects on the table from the RGB camera."""

    def detect(self, prompt: str) -> List[ObjectDetection]:
        """Returns every object matching ``prompt``, positions in the robot base frame."""
        ...


@runtime_checkable
class ManipulationSkills(Protocol):
    """Single-arm manipulation primitives. Every motion call blocks until done.

    Implementations raise an exception on any fault; the task then calls :meth:`abort`.
    """

    def go_home(self) -> None:
        """Moves to the home pose."""
        ...

    def move_to(self, target: TcpTarget) -> None:
        """Straight-line Cartesian TCP motion to ``target`` (moveL)."""
        ...

    def grasp(self) -> bool:
        """Closes the gripper in torque mode at the current pose. True if an object is held."""
        ...

    def release(self) -> None:
        """Opens the gripper."""
        ...

    def lift_to_weigh_pose(self) -> None:
        """Moves the held object to the fixed pose where every load is measured."""
        ...

    def read_joint_torques(self) -> NDArray[np.float64]:
        """Measured arm joint torques (7,), Nm."""
        ...

    def wait(self, dt: float) -> None:
        """Holds the current targets for ``dt`` seconds (keeps the loop rate)."""
        ...

    def abort(self) -> None:
        """Brings the arm to a safe state after a failure. Must not raise."""
        ...


@runtime_checkable
class PayloadEstimator(Protocol):
    """Torque-based load measurement of the currently held object."""

    def measure(self, label: str) -> MassEstimate:
        """Measures the held load at the current (weigh) pose. Blocking."""
        ...
