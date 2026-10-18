"""Gripper skills in torque mode (ported from ``scripts/grip_torque_test.py`` on ``main``).

Linkage friction makes a gentle grip torque close very slowly, so :func:`approach` regulates
the closing speed: the torque ramps up until the fingers close at ``approach_speed_mm_s``,
capped at ``approach_max_torque_nm``. Contact = the fingers were moving and (nearly) stopped,
or never got going at full approach torque (object already touching). :func:`grip` then squeezes
with ``grasp_torque_nm``. Squeeze force ~ 12 N per Nm (ideal linkage).
"""

from __future__ import annotations

import logging
import time

from ..config import GripperConfig
from .driver import Arm

logger = logging.getLogger(__name__)

RATE_HZ = 50.0


def approach(arm: Arm, g: GripperConfig, timeout_s: float = 8.0) -> str:
    """Closes in torque mode until the fingers stop: "contact", "closed" (nothing held) or "timeout"."""
    v_ref = g.approach_speed_mm_s / 1e3
    ramp = g.approach_ramp_nm_s / RATE_HZ
    dt = 1.0 / RATE_HZ
    g0 = g_prev = arm.state().gripper
    tau, v, slow_s, stuck_s, moved = 0.0, 0.0, 0.0, 0.0, False
    arm.set_gripper_torque(tau)
    for _ in range(int(timeout_s * RATE_HZ)):
        arm.advance(dt)
        st = arm.state()
        if st.fault:
            raise RuntimeError(st.fault)
        opening = st.gripper
        v = 0.7 * v + 0.3 * (g_prev - opening) / dt  # closing speed, m/s (filtered)
        g_prev = opening
        if opening < 0.5e-3:
            return "closed"
        moved = moved or g0 - opening > 2e-3
        slow_s = slow_s + dt if moved and v < 0.3 * v_ref else 0.0
        stuck_s = stuck_s + dt if not moved and tau >= g.approach_max_torque_nm else 0.0
        if slow_s >= 0.1 or stuck_s >= 0.3:
            return "contact"
        tau = min(g.approach_max_torque_nm, tau + ramp) if v < v_ref else max(0.0, tau - ramp)
        arm.set_gripper_torque(tau)
    return "timeout"


def wait_settled(arm: Arm, timeout_s: float = 6.0, still_mm_s: float = 0.5, still_s: float = 0.3) -> float:
    """Waits until the opening stops changing; returns it (m)."""
    t_end = time.monotonic() + timeout_s
    prev = arm.state().gripper
    still = 0.0
    while time.monotonic() < t_end:
        arm.advance(1.0 / RATE_HZ)
        opening = arm.state().gripper
        still = still + 1.0 / RATE_HZ if abs(opening - prev) * RATE_HZ * 1e3 < still_mm_s else 0.0
        if still >= still_s:
            return opening
        prev = opening
    return arm.state().gripper


def grip(arm: Arm, g: GripperConfig) -> float:
    """Closes on an object (regulated approach, then ``grasp_torque_nm``); returns the settled opening (m)."""
    result = approach(arm, g)
    arm.set_gripper_torque(g.grasp_torque_nm)
    opening = wait_settled(arm)
    logger.info("grip: approach %s, holding %.2f Nm at %.1f mm", result, g.grasp_torque_nm, opening * 1e3)
    return opening


def release(arm: Arm, g: GripperConfig, open_m: float | None = None) -> None:
    """Torque mode off (holds the measured opening, no squeeze), then opens."""
    target = g.open_m if open_m is None else open_m
    arm.set_gripper_torque(None)
    arm.advance(0.1)
    arm.set_targets(gripper=target)
    arm.advance(abs(target - arm.state().gripper) / g.max_velocity_m_s + 0.3)
