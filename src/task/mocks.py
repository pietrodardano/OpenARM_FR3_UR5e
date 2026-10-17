"""
Offline stand-ins for the vision pipeline, the robot skills and the payload estimator.

They let the weigh-and-sort state machine run and be tested without the camera, the
robot, or the not-yet-merged branches. None of them models physics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .interfaces import MassEstimate, ObjectDetection, TcpTarget


class MockObjectDetector:
    """Returns a fixed list of detections, whatever the prompt (stand-in until vision is merged)."""

    def __init__(self, detections: List[ObjectDetection]) -> None:
        """Args:
            detections: Detections returned by every :meth:`detect` call.
        """
        self.detections = list(detections)
        self.prompts: List[str] = []

    @classmethod
    def from_config(cls, objects: List[Mapping[str, Any]]) -> "MockObjectDetector":
        """Builds the detector from a list like ``demo.manual_objects``."""
        dets = [ObjectDetection(label=str(d["label"]), x=float(d["x"]), y=float(d["y"]),
                                yaw=float(d.get("yaw", 0.0)), confidence=float(d.get("confidence", 1.0)))
                for d in objects]
        return cls(dets)

    def detect(self, prompt: str) -> List[ObjectDetection]:
        """Returns the fixed detections."""
        self.prompts.append(prompt)
        return list(self.detections)


class MockManipulationSkills:
    """Records every skill call and synthesizes holding torques with an optional user grasp.

    The mock clock restarts on every :meth:`move_to`. When ``contact_at_s`` is set, the
    torques shift by ``contact_delta_nm`` on all joints from that time after the last move:
    a person taking the object at the handover pose.
    """

    def __init__(
        self,
        grasp_succeeds: bool = True,
        torque_noise_nm: float = 0.02,
        contact_at_s: Optional[float] = None,
        contact_delta_nm: float = 0.6,
        seed: int = 0,
    ) -> None:
        """Args:
            grasp_succeeds: Return value of :meth:`grasp`.
            torque_noise_nm: Standard deviation of the synthetic torque noise.
            contact_at_s: Time after the last move at which the user grasps (None = never).
            contact_delta_nm: Torque shift caused by the user's grasp.
            seed: Random seed of the torque noise.
        """
        self.grasp_succeeds = grasp_succeeds
        self.torque_noise_nm = torque_noise_nm
        self.contact_at_s = contact_at_s
        self.contact_delta_nm = contact_delta_nm
        self.calls: List[Tuple[str, Any]] = []
        self.holding = False
        self.tcp: Optional[TcpTarget] = None
        self.released_at: List[Optional[TcpTarget]] = []
        self._clock = 0.0
        self._rng = np.random.default_rng(seed)
        self._base_tau = np.array([-1.5, 0.2, 0.1, 2.0, 0.05, 0.1, -0.6])

    @classmethod
    def from_config(cls, mock_cfg: Mapping[str, Any]) -> "MockManipulationSkills":
        """Builds the mock from ``demo.mock``."""
        contact = mock_cfg.get("contact_at_s")
        return cls(torque_noise_nm=float(mock_cfg.get("torque_noise_nm", 0.02)),
                   contact_at_s=None if contact is None else float(contact),
                   contact_delta_nm=float(mock_cfg.get("contact_delta_nm", 0.6)))

    def go_home(self) -> None:
        """Records the call."""
        self.calls.append(("go_home", None))

    def move_to(self, target: TcpTarget) -> None:
        """Teleports the mock TCP and restarts the mock clock."""
        self.calls.append(("move_to", target))
        self.tcp = target
        self._clock = 0.0

    def grasp(self) -> bool:
        """Returns ``grasp_succeeds``."""
        self.calls.append(("grasp", None))
        self.holding = self.grasp_succeeds
        return self.grasp_succeeds

    def release(self) -> None:
        """Opens the mock gripper at the current TCP."""
        self.calls.append(("release", self.tcp))
        self.released_at.append(self.tcp)
        self.holding = False

    def lift_to_weigh_pose(self) -> None:
        """Records the call."""
        self.calls.append(("lift_to_weigh_pose", None))

    def read_joint_torques(self) -> NDArray[np.float64]:
        """Holding torques + noise, shifted once the scheduled user grasp has happened."""
        tau = self._base_tau + self._rng.normal(0.0, self.torque_noise_nm, 7)
        if self.contact_at_s is not None and self._clock >= self.contact_at_s:
            tau = tau - self.contact_delta_nm
        return tau

    def wait(self, dt: float) -> None:
        """Advances the mock clock."""
        self._clock += dt

    def abort(self) -> None:
        """Records the call."""
        self.calls.append(("abort", None))

    def call_names(self) -> List[str]:
        """Names of all recorded calls, in order."""
        return [name for name, _ in self.calls]


class MockPayloadEstimator:
    """Returns configured masses with synthetic per-cycle noise."""

    def __init__(self, masses_kg: Mapping[str, float], noise_kg: float = 0.01, cycles: int = 3, seed: int = 0) -> None:
        """Args:
            masses_kg: True mass per object name (``object_1``, ``object_2``, ...).
            noise_kg: Standard deviation of each simulated dither-cycle mass.
            cycles: Simulated dither cycles per measurement.
            seed: Random seed.
        """
        self.masses_kg: Dict[str, float] = dict(masses_kg)
        self.noise_kg = noise_kg
        self.cycles = cycles
        self.measured: List[str] = []
        self._rng = np.random.default_rng(seed)

    @classmethod
    def from_config(cls, mock_cfg: Mapping[str, Any]) -> "MockPayloadEstimator":
        """Builds the mock from ``demo.mock``: ``masses_kg[i]`` belongs to ``object_{i+1}``."""
        masses = {f"object_{i + 1}": float(m) for i, m in enumerate(mock_cfg["masses_kg"])}
        return cls(masses, noise_kg=float(mock_cfg.get("noise_kg", 0.01)))

    def measure(self, label: str) -> MassEstimate:
        """Simulates one dithered load measurement of ``label``."""
        self.measured.append(label)
        cycle_mass = self.masses_kg[label] + self._rng.normal(0.0, self.noise_kg, self.cycles)
        se = float(np.std(cycle_mass, ddof=1) / np.sqrt(self.cycles)) if self.cycles > 1 else float("nan")
        return MassEstimate(label=label, mass_kg=float(np.mean(cycle_mass)), mass_se_kg=se, raw=cycle_mass)
