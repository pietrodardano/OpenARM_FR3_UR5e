"""
Thin adapter over the official ``openarm_can`` SDK for ONE arm on ONE CAN bus.

This is the only module that imports ``openarm_can``. Its Python binding is
marked experimental upstream, so if the binding API changes, this file is the
only one that needs updating. Everything above it works with plain numpy arrays.

Units at this boundary are raw motor units: arm motors in rad (== URDF joint
angle for OpenArm v1), the gripper motor in rad (NOT finger metres; the
conversion lives in the bimanual driver).

:class:`FakeOpenArmBus` has the same API and simulates first-order motors, so
the threaded driver can be tested without hardware.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

ARM_MOTORS = 7


@dataclass
class ArmBusConfig:
    """CAN layout of one OpenArm v1 arm (defaults = official v1 / openarm_ros2 values)."""

    can_interface: str
    can_fd: bool = True
    motor_types: Sequence[str] = ("DM8009", "DM8009", "DM4340", "DM4340", "DM4310", "DM4310", "DM4310")
    send_can_ids: Sequence[int] = (0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07)
    recv_can_ids: Sequence[int] = (0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17)
    gripper_motor_type: str = "DM4310"
    gripper_send_can_id: int = 0x08
    gripper_recv_can_id: int = 0x18


@dataclass
class BusReading:
    """Latest feedback of the 7 arm motors + gripper motor of one bus."""

    q: NDArray[np.float64]  # (7,) rad
    dq: NDArray[np.float64]  # (7,) rad/s
    tau: NDArray[np.float64]  # (7,) Nm
    gripper_q: float  # gripper motor angle, rad
    error_codes: List[int] = field(default_factory=list)  # 8 raw status nibbles (7 arm + gripper)
    responses: List[int] = field(default_factory=list)  # 8 cumulative reply counters
    bus_healthy: bool = True


class OpenArmBus:
    """One arm: 7 Damiao arm motors + 1 gripper motor, MIT (impedance) control."""

    def __init__(self, cfg: ArmBusConfig) -> None:
        self.cfg = cfg
        self._oa = None
        self._dev = None

    def open(self) -> None:
        """Opens the SocketCAN interface and registers the motors (no motion)."""
        import openarm_can as oa  # deferred: only needed on the real robot

        self._oa = oa
        dev = oa.OpenArm(self.cfg.can_interface, self.cfg.can_fd)
        dev.init_arm_motors(
            [getattr(oa.MotorType, t) for t in self.cfg.motor_types],
            list(self.cfg.send_can_ids),
            list(self.cfg.recv_can_ids),
        )
        dev.init_gripper_motor(
            getattr(oa.MotorType, self.cfg.gripper_motor_type),
            self.cfg.gripper_send_can_id,
            self.cfg.gripper_recv_can_id,
        )
        dev.set_callback_mode_all(oa.CallbackMode.STATE)
        self._dev = dev

    def enable(self) -> None:
        self._dev.enable_all()
        self._dev.recv_all(2000)  # enabling is slow; give the motors 2 ms to answer

    def disable(self) -> None:
        self._dev.disable_all()
        self._dev.recv_all(2000)

    def refresh(self, timeout_us: int = 2000) -> BusReading:
        """Requests feedback without commanding anything (works with motors disabled)."""
        self._dev.refresh_all()
        return self.receive(timeout_us)

    def command(
        self,
        q: NDArray[np.float64],
        kp: NDArray[np.float64],
        kd: NDArray[np.float64],
        tau_ff: NDArray[np.float64],
        gripper_q: float,
        gripper_kp: float,
        gripper_kd: float,
    ) -> None:
        """Sends one MIT frame per motor: tau = kp (q - q_meas) - kd dq_meas + tau_ff."""
        mit = self._oa.MITParam
        self._dev.get_arm().mit_control_all(
            [mit(float(kp[i]), float(kd[i]), float(q[i]), 0.0, float(tau_ff[i])) for i in range(ARM_MOTORS)]
        )
        self._dev.get_gripper().mit_control_all([mit(float(gripper_kp), float(gripper_kd), float(gripper_q), 0.0, 0.0)])

    def receive(self, timeout_us: int) -> BusReading:
        """Drains replies (waits up to ``timeout_us`` for the first) and returns the latest state."""
        self._dev.recv_all(int(timeout_us))
        arm, grip = self._dev.get_arm(), self._dev.get_gripper()
        motors = arm.get_motors()
        gmotor = grip.get_motors()[0]
        return BusReading(
            q=np.array([m.get_position() for m in motors]),
            dq=np.array([m.get_velocity() for m in motors]),
            tau=np.array([m.get_torque() for m in motors]),
            gripper_q=float(gmotor.get_position()),
            error_codes=[int(m.get_error_code()) for m in motors] + [int(gmotor.get_error_code())],
            responses=[int(arm.get_link_stats(i).responses) for i in range(ARM_MOTORS)]
            + [int(grip.get_link_stats(0).responses)],
            bus_healthy=bool(self._dev.is_bus_healthy()),
        )

    def close(self) -> None:
        self._dev = None


class FakeOpenArmBus:
    """Drop-in stand-in for :class:`OpenArmBus`: motors track MIT targets with a first-order lag.

    Args:
        q0: Initial arm motor angles (7,).
        gripper_q0: Initial gripper motor angle.
        tau_s: Time constant of the simulated position response (s).
        dt: Assumed time between ``command`` calls (s).
    """

    def __init__(
        self,
        q0: Optional[NDArray[np.float64]] = None,
        gripper_q0: float = 0.0,
        tau_s: float = 0.02,
        dt: float = 0.002,
    ) -> None:
        self.q = np.zeros(ARM_MOTORS) if q0 is None else np.array(q0, dtype=np.float64)
        self.dq = np.zeros(ARM_MOTORS)
        self.gripper_q = gripper_q0
        self.alpha = min(1.0, dt / tau_s)
        self.dt = dt
        self.enabled = False
        self.responses = [0] * (ARM_MOTORS + 1)
        self.error_codes = [0] * (ARM_MOTORS + 1)
        self.silent = False  # set True to simulate a motor/bus that stopped answering
        self.last_command: Optional[dict] = None

    def open(self) -> None:
        pass

    def enable(self) -> None:
        self.enabled = True
        self.error_codes = [1] * (ARM_MOTORS + 1)

    def disable(self) -> None:
        self.enabled = False
        self.error_codes = [0] * (ARM_MOTORS + 1)

    def refresh(self, timeout_us: int = 2000) -> BusReading:
        self._count_replies()
        return self.receive(timeout_us)

    def command(self, q, kp, kd, tau_ff, gripper_q, gripper_kp, gripper_kd) -> None:  # noqa: D102
        self.last_command = dict(q=np.array(q), kp=np.array(kp), kd=np.array(kd), tau_ff=np.array(tau_ff), gripper_q=gripper_q)
        if not self.enabled:
            return
        q_prev = self.q.copy()
        self.q = self.q + self.alpha * (np.asarray(q) - self.q)
        self.dq = (self.q - q_prev) / self.dt
        self.gripper_q += self.alpha * (gripper_q - self.gripper_q)
        self._count_replies()

    def _count_replies(self) -> None:
        if not self.silent:
            self.responses = [r + 1 for r in self.responses]

    def receive(self, timeout_us: int) -> BusReading:
        return BusReading(
            q=self.q.copy(),
            dq=self.dq.copy(),
            tau=np.zeros(ARM_MOTORS),
            gripper_q=self.gripper_q,
            error_codes=list(self.error_codes),
            responses=list(self.responses),
            bus_healthy=True,
        )

    def close(self) -> None:
        pass
