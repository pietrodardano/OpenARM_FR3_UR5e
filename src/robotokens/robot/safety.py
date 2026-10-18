"""Joint-space safety filter and setpoint helpers (SR1), used inside the 250 Hz servo loop.

Vectors are 8-dim: 7 arm joints (rad) + gripper opening (m). Whatever a caller commands,
the setpoint that reaches the motors is (1) inside the official joint limits minus a margin
(a joint that started inside the margin may return to, but not go beyond, its start pose) and
(2) moving no faster than the per-joint velocity limit.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

MIN_JERK_PEAK = 1.875  # peak / average speed of a minimum-jerk profile


def minimum_jerk(s: float) -> float:
    """Minimum-jerk time scaling: normalized time [0, 1] -> progress [0, 1]."""
    s = min(max(s, 0.0), 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


class SafetyFilter:
    """Clips targets to the joint limits (minus ``margin`` on the arm joints) and rate-limits them."""

    def __init__(self, lower: NDArray, upper: NDArray, vmax: NDArray, margin: float) -> None:
        self.lower = np.asarray(lower, dtype=np.float64)
        self.upper = np.asarray(upper, dtype=np.float64)
        self.vmax = np.asarray(vmax, dtype=np.float64)
        m = np.full(self.lower.shape, margin)
        m[-1] = 0.0  # gripper: no margin
        self.soft_lower = np.minimum(self.lower + m, self.upper)
        self.soft_upper = np.maximum(self.upper - m, self.lower)
        self._setpoint: Optional[NDArray] = None
        self._anchor: Optional[NDArray] = None  # pose at the last reset()

    @property
    def setpoint(self) -> NDArray:
        if self._setpoint is None:
            raise RuntimeError("SafetyFilter used before reset()")
        return self._setpoint.copy()

    def reset(self, measured: NDArray) -> NDArray:
        """Re-anchors at the measured pose (start-up, fault): the motors never see a jump."""
        self._setpoint = np.clip(np.asarray(measured, dtype=np.float64), self.lower, self.upper)
        self._anchor = self._setpoint.copy()
        return self._setpoint.copy()

    def reset_entry(self, index: int, value: float) -> None:
        """Re-anchors one entry (a gripper tracked in torque mode); the others keep going."""
        if self._setpoint is None:
            raise RuntimeError("SafetyFilter used before reset()")
        self._setpoint[index] = np.clip(value, self.lower[index], self.upper[index])

    def step(self, target: NDArray, dt: float) -> NDArray:
        """Moves the setpoint towards ``target`` by at most ``vmax * dt``."""
        if self._setpoint is None:
            raise RuntimeError("SafetyFilter used before reset()")
        lo = np.minimum(self.soft_lower, self._anchor)
        hi = np.maximum(self.soft_upper, self._anchor)
        goal = np.clip(np.asarray(target, dtype=np.float64), lo, hi)
        max_step = self.vmax * dt
        self._setpoint = self._setpoint + np.clip(goal - self._setpoint, -max_step, max_step)
        return self._setpoint.copy()


class TargetInterpolator:
    """Spreads each new target over the caller's command period.

    A caller streaming targets at 10-100 Hz into the 250 Hz loop would otherwise produce a
    staircase setpoint. Each new target is reached linearly over the measured period between
    calls (one period of latency). After a pause longer than ``max_period`` a target passes
    as a step (the safety filter ramps it).
    """

    def __init__(self, dim: int, max_period: float = 0.1) -> None:
        self.max_period = max_period
        self._start = np.zeros(dim)
        self._goal = np.zeros(dim)
        self._t_goal = 0.0
        self._period = 0.0

    def reset(self, vec: NDArray, now: float) -> None:
        self._start = np.array(vec, dtype=np.float64)
        self._goal = self._start.copy()
        self._t_goal = now
        self._period = 0.0

    def set_goal(self, vec: NDArray, now: float) -> None:
        gap = now - self._t_goal
        if gap > self.max_period:
            self._period = 0.0
        else:
            self._period = gap if self._period == 0.0 else 0.7 * self._period + 0.3 * gap
        self._start = self.value(now)
        self._goal = np.array(vec, dtype=np.float64)
        self._t_goal = now

    def set_entry(self, index: int, value: float) -> None:
        """Overrides one entry immediately."""
        self._start[index] = value
        self._goal[index] = value

    def value(self, now: float) -> NDArray:
        if self._period <= 0.0:
            return self._goal.copy()
        a = min(1.0, max(0.0, (now - self._t_goal) / self._period))
        return self._start + a * (self._goal - self._start)
