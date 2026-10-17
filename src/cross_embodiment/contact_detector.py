"""Detect an external contact from a change in joint torque (the hand-over release trigger).

The detector watches the change of the measured joint torque against a baseline window taken while
the arm holds still at the hand-over pose. It does not need the gravity model, only the change
against the baseline, so it is robust to model error. It never fires on a timer: the caller opens
the gripper only when ``update`` returns True.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class ContactDetector:
    def __init__(self, baseline: np.ndarray, baseline_std: np.ndarray, sigmas: float = 8.0, min_delta_nm: float = 0.5, persist: int = 5) -> None:
        self.baseline = np.asarray(baseline, dtype=float)
        self.threshold = np.maximum(sigmas * np.asarray(baseline_std, dtype=float), min_delta_nm)
        self.persist = persist
        self._count = 0
        self.score = 0.0

    @classmethod
    def from_window(cls, window: np.ndarray, **kwargs) -> "ContactDetector":
        """Build from an (N, n) array of torque samples taken while holding still."""
        return cls(window.mean(axis=0), window.std(axis=0), **kwargs)

    def update(self, tau: np.ndarray) -> bool:
        """Feed one torque sample; True once the change has persisted for ``persist`` samples."""
        ratio = np.abs(np.asarray(tau, dtype=float) - self.baseline) / self.threshold
        self.score = float(ratio.max())
        self._count = self._count + 1 if self.score > 1.0 else 0
        return self._count >= self.persist
