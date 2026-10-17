"""
External-contact detection from joint torques, for the handover release decision.

While the arm holds the object still at the handover pose, the measured joint torques
are steady: arm gravity + payload + the impedance controller's small holding effort.
When a person takes the object, they support its weight or pull it, and the torques of
the load-bearing joints shift. The gripper opens only after that shift is seen. A hand
that only looks close to the object does not count.

Detection rule, per used joint ``j``:

    baseline:   mu_j, sigma_j  over ``baseline_s`` while holding still
    threshold:  thr_j = max(k_sigma * sigma_j, min_delta_nm)
    contact:    |tau_j - mu_j| > thr_j on any used joint, for ``hold_s`` without a break

No contact before ``timeout_s`` means "do not release".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, List, Mapping, Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class ContactConfig:
    """Tunables of :class:`ExternalContactDetector` (``demo.contact`` in task_config.yaml)."""

    baseline_s: float = 1.0
    k_sigma: float = 4.0
    min_delta_nm: float = 0.25
    hold_s: float = 0.15
    timeout_s: float = 20.0
    rate_hz: float = 100.0
    joints: Optional[List[int]] = field(default=None)  # 0-based; None = all joints

    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any]) -> "ContactConfig":
        """Builds the config from the ``demo.contact`` mapping."""
        joints = cfg.get("joints")
        return cls(
            baseline_s=float(cfg.get("baseline_s", cls.baseline_s)),
            k_sigma=float(cfg.get("k_sigma", cls.k_sigma)),
            min_delta_nm=float(cfg.get("min_delta_nm", cls.min_delta_nm)),
            hold_s=float(cfg.get("hold_s", cls.hold_s)),
            timeout_s=float(cfg.get("timeout_s", cls.timeout_s)),
            rate_hz=float(cfg.get("rate_hz", cls.rate_hz)),
            joints=None if joints is None else [int(j) for j in joints],
        )


class ExternalContactDetector:
    """Latches when the joint torques leave their holding baseline for long enough."""

    def __init__(self, config: ContactConfig) -> None:
        """Creates a detector without a baseline.

        Args:
            config: Thresholds and timing.
        """
        self.config = config
        self._hold_samples = max(1, int(math.ceil(config.hold_s * config.rate_hz)))
        self.reset()

    def reset(self) -> None:
        """Clears the baseline and the contact latch."""
        self.mean: Optional[NDArray[np.float64]] = None
        self.threshold: Optional[NDArray[np.float64]] = None
        self._streak = 0
        self.in_contact = False

    def set_baseline(self, samples: NDArray[np.float64]) -> None:
        """Sets the holding baseline from torque samples.

        Args:
            samples: (n, n_joints) torques recorded while holding still, Nm. n >= 2.
        """
        samples = np.asarray(samples, dtype=np.float64)
        if samples.ndim != 2 or samples.shape[0] < 2:
            raise ValueError("baseline needs at least two (n_joints,) torque samples")
        self.mean = samples.mean(axis=0)
        sigma = samples.std(axis=0, ddof=1)
        self.threshold = np.maximum(self.config.k_sigma * sigma, self.config.min_delta_nm)
        self._streak = 0
        self.in_contact = False

    def residual(self, tau: NDArray[np.float64]) -> NDArray[np.float64]:
        """Torque deviation from the baseline on the used joints, Nm."""
        if self.mean is None:
            raise RuntimeError("set_baseline() must be called first")
        res = np.asarray(tau, dtype=np.float64) - self.mean
        return res if self.config.joints is None else res[self.config.joints]

    def update(self, tau: NDArray[np.float64]) -> bool:
        """Feeds one torque sample. Returns True once contact is latched."""
        if self.in_contact:
            return True
        assert self.threshold is not None
        thr = self.threshold if self.config.joints is None else self.threshold[self.config.joints]
        exceeded = bool(np.any(np.abs(self.residual(tau)) > thr))
        self._streak = self._streak + 1 if exceeded else 0
        self.in_contact = self._streak >= self._hold_samples
        return self.in_contact


def wait_for_contact(
    read_torques: Callable[[], NDArray[np.float64]],
    wait: Callable[[float], None],
    config: ContactConfig,
    detector: Optional[ExternalContactDetector] = None,
) -> bool:
    """Records a baseline, then blocks until external contact or timeout.

    The arm must already be settled at the handover pose, holding the object.

    Args:
        read_torques: Returns the current (n_joints,) joint torques, Nm.
        wait: Advances one control period (e.g. ``robot.advance``).
        config: Detection thresholds and timing.
        detector: Optional detector instance (exposed for logging); a new one by default.

    Returns:
        True if contact was felt before ``config.timeout_s``; False means do not release.
    """
    det = detector if detector is not None else ExternalContactDetector(config)
    dt = 1.0 / config.rate_hz
    n_baseline = max(2, int(round(config.baseline_s * config.rate_hz)))
    samples = []
    for _ in range(n_baseline):
        wait(dt)
        samples.append(np.asarray(read_torques(), dtype=np.float64))
    det.set_baseline(np.asarray(samples))
    for _ in range(int(round(config.timeout_s * config.rate_hz))):
        wait(dt)
        if det.update(read_torques()):
            return True
    return False
