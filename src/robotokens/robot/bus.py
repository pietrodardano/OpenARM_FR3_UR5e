"""One OpenArm arm (7 Damiao motors + gripper motor) on one CAN bus.

:class:`CanBus` is the only code that imports ``openarm_can`` (NFR3); its Python binding is
experimental upstream, so an API change stays in this file. :class:`FakeBus` has the same API
with first-order motors, so the real driver thread runs without hardware (tests, ``--fake``).

Units here are raw motor units: arm motors in rad (= joint angle), gripper motor in rad.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
from numpy.typing import NDArray

from ..config import GripperConfig, HardwareConfig

N_JOINTS = 7


@dataclass
class BusReading:
    """Latest feedback of the 7 arm motors + gripper motor."""

    q: NDArray  # (7,) rad
    dq: NDArray  # (7,) rad/s
    tau: NDArray  # (7,) Nm, from phase current
    gripper_q: float  # motor rad
    gripper_dq: float = 0.0
    gripper_tau: float = 0.0  # Nm, motor sign
    error_codes: List[int] = field(default_factory=list)  # 8 status nibbles (0/1 OK, >= 0x8 fault)
    responses: List[int] = field(default_factory=list)  # 8 cumulative reply counters
    bus_healthy: bool = True


class CanBus:
    """Real arm over SocketCAN, MIT (impedance) control: tau = kp (q - q_meas) - kd dq + tau_ff."""

    def __init__(self, hw: HardwareConfig, gripper: GripperConfig) -> None:
        self.hw = hw
        self.gripper = gripper
        self._oa = None
        self._dev = None

    def open(self) -> None:
        """Opens the interface and registers the motors (no motion)."""
        import openarm_can as oa  # only on the real robot

        self._oa = oa
        dev = oa.OpenArm(self.hw.can_interface, self.hw.can_fd)
        dev.init_arm_motors([getattr(oa.MotorType, t) for t in self.hw.motor_types],
                            list(self.hw.send_can_ids), list(self.hw.recv_can_ids))
        dev.init_gripper_motor(getattr(oa.MotorType, self.gripper.motor_type),
                               self.gripper.send_can_id, self.gripper.recv_can_id)
        dev.set_callback_mode_all(oa.CallbackMode.STATE)
        self._dev = dev

    def enable(self) -> None:
        self._dev.enable_all()
        self._dev.recv_all(2000)

    def disable(self) -> None:
        self._dev.disable_all()
        self._dev.recv_all(2000)

    def refresh(self, timeout_us: int = 2000) -> BusReading:
        """Requests feedback without commanding (works with motors disabled)."""
        self._dev.refresh_all()
        return self.receive(timeout_us)

    def command(self, q: NDArray, kp: NDArray, kd: NDArray, tau_ff: NDArray,
                gripper_q: float, gripper_kp: float, gripper_kd: float, gripper_tau_ff: float = 0.0) -> None:
        """One MIT frame per motor. ``gripper_kp = 0`` = gripper torque (current) mode."""
        mit = self._oa.MITParam
        self._dev.get_arm().mit_control_all(
            [mit(float(kp[i]), float(kd[i]), float(q[i]), 0.0, float(tau_ff[i])) for i in range(N_JOINTS)])
        self._dev.get_gripper().mit_control_all(
            [mit(float(gripper_kp), float(gripper_kd), float(gripper_q), 0.0, float(gripper_tau_ff))])

    def receive(self, timeout_us: int) -> BusReading:
        """Drains replies (waits up to ``timeout_us`` for the first) and returns the latest state."""
        self._dev.recv_all(int(timeout_us))
        arm, grip = self._dev.get_arm(), self._dev.get_gripper()
        motors, gm = arm.get_motors(), grip.get_motors()[0]
        return BusReading(
            q=np.array([m.get_position() for m in motors]),
            dq=np.array([m.get_velocity() for m in motors]),
            tau=np.array([m.get_torque() for m in motors]),
            gripper_q=float(gm.get_position()),
            gripper_dq=float(gm.get_velocity()),
            gripper_tau=float(gm.get_torque()),
            error_codes=[int(m.get_error_code()) for m in motors] + [int(gm.get_error_code())],
            responses=[int(arm.get_link_stats(i).responses) for i in range(N_JOINTS)]
            + [int(grip.get_link_stats(0).responses)],
            bus_healthy=bool(self._dev.is_bus_healthy()),
        )

    def close(self) -> None:
        self._dev = None


TauFn = Callable[[NDArray, NDArray, float], NDArray]  # (q, dq, gripper motor rad) -> tau (7,)


class FakeBus:
    """Stand-in for :class:`CanBus`: motors follow MIT targets with a first-order lag.

    Test hooks: ``silent`` (no replies), ``error_codes`` (motor faults), ``healthy`` (bus-off),
    ``frozen`` (motors ignore commands: tracking error), ``gripper_block_at`` (motor angle where
    an object stops the closing fingers), ``tau_fn`` (measured joint torques).
    """

    def __init__(self, q0: Optional[NDArray] = None, dt: float = 0.004, tau_fn: Optional[TauFn] = None,
                 motor_open_rad: float = -1.0472, silent: bool = False) -> None:
        self.q = np.zeros(N_JOINTS) if q0 is None else np.array(q0, dtype=np.float64)
        self.dq = np.zeros(N_JOINTS)
        self.gripper_q = 0.0
        self.gripper_dq = 0.0
        self.gripper_tau = 0.0
        self.gripper_range = (min(motor_open_rad, 0.0), max(motor_open_rad, 0.0))
        self.close_sign = -1.0 if motor_open_rad > 0 else 1.0
        self.gripper_rad_s_per_nm = 2.0  # torque mode: closing speed per Nm until a stop
        self.gripper_block_at: Optional[float] = None
        self.tau_fn = tau_fn
        self.alpha = min(1.0, dt / 0.02)
        self.dt = dt
        self.enabled = False
        self.silent = silent
        self.frozen = False
        self.healthy = True
        self.responses = [0] * (N_JOINTS + 1)
        self.error_codes = [0] * (N_JOINTS + 1)

    def open(self) -> None:
        pass

    def enable(self) -> None:
        self.enabled = True
        self.error_codes = [1] * (N_JOINTS + 1)

    def disable(self) -> None:
        self.enabled = False
        self.error_codes = [0] * (N_JOINTS + 1)

    def refresh(self, timeout_us: int = 2000) -> BusReading:
        self._reply()
        return self.receive(timeout_us)

    def command(self, q, kp, kd, tau_ff, gripper_q, gripper_kp, gripper_kd, gripper_tau_ff=0.0) -> None:  # noqa: D102
        if self.enabled and not self.frozen:
            q_prev = self.q.copy()
            self.q = self.q + self.alpha * (np.asarray(q) - self.q)
            self.dq = (self.q - q_prev) / self.dt
            g_prev = self.gripper_q
            if gripper_kp > 0:
                self.gripper_q += self.alpha * (gripper_q - self.gripper_q)
                self.gripper_tau = gripper_kp * (gripper_q - self.gripper_q)
            else:  # torque mode: close (or open) until a stop or the grasped object
                lo, hi = self.gripper_range
                if self.gripper_block_at is not None:
                    lo, hi = (lo, self.gripper_block_at) if self.close_sign > 0 else (self.gripper_block_at, hi)
                step = self.gripper_rad_s_per_nm * gripper_tau_ff * self.dt
                self.gripper_q = float(np.clip(self.gripper_q + step, lo, hi))
                self.gripper_tau = gripper_tau_ff
            self.gripper_dq = (self.gripper_q - g_prev) / self.dt
        else:
            self.dq = np.zeros(N_JOINTS)
        self._reply()

    def _reply(self) -> None:
        if not self.silent:
            self.responses = [r + 1 for r in self.responses]

    def receive(self, timeout_us: int) -> BusReading:
        tau = np.zeros(N_JOINTS) if self.tau_fn is None else np.asarray(self.tau_fn(self.q, self.dq, self.gripper_q))
        return BusReading(q=self.q.copy(), dq=self.dq.copy(), tau=tau, gripper_q=self.gripper_q,
                          gripper_dq=self.gripper_dq, gripper_tau=self.gripper_tau,
                          error_codes=list(self.error_codes), responses=list(self.responses),
                          bus_healthy=self.healthy)

    def close(self) -> None:
        pass
