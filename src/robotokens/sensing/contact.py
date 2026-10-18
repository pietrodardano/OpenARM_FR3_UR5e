"""External-contact detection from joint torques: the handover release decision (FR6).

While the arm holds the object still at the handover pose the torques are steady. When a person
takes the object they support or pull it, and the load-bearing joint torques shift. The gripper
opens only after that shift is felt; a hand that is merely close does not count.

    baseline:  mean_j, std_j over ``baseline_s`` while holding still
    threshold: max(k_sigma * std_j, min_delta_nm)
    contact:   |tau_j - mean_j| > threshold on any joint, for ``hold_s`` without a break
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray

from ..config import ContactConfig


class ContactDetector:
    def __init__(self, cfg: ContactConfig) -> None:
        self.cfg = cfg
        self._hold = max(1, int(math.ceil(cfg.hold_s * cfg.rate_hz)))
        self.mean: Optional[NDArray] = None
        self.threshold: Optional[NDArray] = None
        self._streak = 0
        self.in_contact = False

    def set_baseline(self, samples: NDArray) -> None:
        """(n, 7) torques recorded while holding still, n >= 2."""
        samples = np.asarray(samples, dtype=np.float64)
        if samples.ndim != 2 or samples.shape[0] < 2:
            raise ValueError("the baseline needs at least two torque samples")
        self.mean = samples.mean(axis=0)
        self.threshold = np.maximum(self.cfg.k_sigma * samples.std(axis=0, ddof=1), self.cfg.min_delta_nm)
        self._streak, self.in_contact = 0, False

    def update(self, tau: NDArray) -> bool:
        """Feeds one sample; True once contact is latched."""
        if self.in_contact:
            return True
        if self.mean is None:
            raise RuntimeError("set_baseline() first")
        exceeded = bool(np.any(np.abs(np.asarray(tau) - self.mean) > self.threshold))
        self._streak = self._streak + 1 if exceeded else 0
        self.in_contact = self._streak >= self._hold
        return self.in_contact


def wait_for_contact(read_torques: Callable[[], NDArray], wait: Callable[[float], None], cfg: ContactConfig) -> bool:
    """Records a baseline, then blocks until contact (True) or ``timeout_s`` (False: do not release)."""
    det = ContactDetector(cfg)
    dt = 1.0 / cfg.rate_hz
    samples = []
    for _ in range(max(2, int(round(cfg.baseline_s * cfg.rate_hz)))):
        wait(dt)
        samples.append(np.asarray(read_torques(), dtype=np.float64))
    det.set_baseline(np.asarray(samples))
    for _ in range(int(round(cfg.timeout_s * cfg.rate_hz))):
        wait(dt)
        if det.update(read_torques()):
            return True
    return False
