"""Real-time driver of one OpenArm v1 arm: one Python thread is the servo loop (no ROS).

Ported from the hardware-validated ``OpenArmBimanual`` on ``main``, reduced to one arm.
Every tick (250 Hz) the thread

1. takes the caller's latest target (lock-protected, non-blocking), spread over the caller's
   command period by :class:`TargetInterpolator`,
2. passes it through :class:`SafetyFilter` (joint limits minus margin, velocity limits; SR1),
3. sends one MIT frame per motor, with gravity feedforward from the official model,
4. reads the replies and publishes an :class:`ArmState`,
5. checks faults (SR3): motor error codes, motors that stop replying, bus-off, sustained
   tracking error. A fault latches and freezes the arm at its measured pose; motions abort.

The gripper runs in position mode (MIT, gripper kp/kd) or, after :meth:`Arm.set_gripper_torque`,
in torque mode: MIT with kp = 0, the motor current loop outputs the commanded torque, capped
and ramped (SR5). Torque mode survives a fault, so a grasped object is not dropped.

Vectors are 8-dim internally: 7 joints (rad) + gripper opening (m, 0 = closed).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ..config import Config
from .bus import N_JOINTS, BusReading, CanBus, FakeBus
from .safety import SafetyFilter, TargetInterpolator, minimum_jerk

logger = logging.getLogger(__name__)

MOTOR_ERRORS = {0x8: "OVERVOLTAGE", 0x9: "UNDERVOLTAGE", 0xA: "OVERCURRENT", 0xB: "MOS_OVERHEAT",
                0xC: "COIL_OVERHEAT", 0xD: "COMMUNICATION_LOST", 0xE: "OVERLOAD"}
G = N_JOINTS  # index of the gripper in the 8-dim vectors


@dataclass(frozen=True)
class ArmState:
    """Measured and commanded state of the arm at one control tick."""

    q: NDArray  # (7,) measured joints, rad
    dq: NDArray  # (7,) rad/s
    tau: NDArray  # (7,) measured joint torques, Nm
    gripper: float  # measured opening, m
    q_setpoint: NDArray  # (7,) setpoint sent this tick (after the safety filter)
    gripper_setpoint: float  # m
    gripper_tau: float  # measured gripper motor torque, Nm, + = closing
    gripper_torque_cmd: Optional[float]  # Nm sent in torque mode, None = position mode
    motor_errors: Tuple[int, ...]
    fault: Optional[str]
    timestamp: float


class Arm:
    """One arm over a bus (:class:`CanBus` or :class:`FakeBus`), streamed from a thread.

    Args:
        cfg: Full config (``hardware``, ``gripper``, ``robot.rest_qpos``).
        bus: The arm's bus.
        lower, upper: Official joint limits (7,).
        gravity_fn: ``q (7,) -> tau (7,)`` model gravity, called from the servo thread
            (give it a dedicated model instance).
    """

    def __init__(self, cfg: Config, bus: CanBus | FakeBus, lower: NDArray, upper: NDArray,
                 gravity_fn: Optional[Callable[[NDArray], NDArray]] = None) -> None:
        self.cfg = cfg
        self.hw = cfg.hardware
        self.grip = cfg.gripper
        self.bus = bus
        self.lower = np.asarray(lower, dtype=np.float64)
        self.upper = np.asarray(upper, dtype=np.float64)
        self.gravity_fn = gravity_fn
        if self.hw.gravity_compensation_scale > 0 and gravity_fn is None:
            raise ValueError("gravity_compensation_scale > 0 needs a gravity_fn")
        self.filter = SafetyFilter(np.append(self.lower, 0.0), np.append(self.upper, self.grip.open_m),
                                   np.append(self.hw.max_joint_velocity_rad_s, self.grip.max_velocity_m_s),
                                   self.hw.joint_limit_margin_rad)
        self._kp = np.asarray(self.hw.kp, dtype=np.float64)
        self._kd = np.asarray(self.hw.kd, dtype=np.float64)
        self._dt = 1.0 / self.hw.control_rate_hz
        self._close_sign = -1.0 if self.grip.motor_open_rad > 0 else 1.0  # motor sign of "closing"

        self._lock = threading.Lock()
        self._target = np.zeros(N_JOINTS + 1)
        self._interp = TargetInterpolator(N_JOINTS + 1)
        self._hold = np.zeros(N_JOINTS + 1)  # frozen setpoint while faulted
        self._state: Optional[ArmState] = None
        self._fault: Optional[str] = None
        self._clear_fault_req = False
        self._tau_target: Optional[float] = None  # requested gripper torque (None = position mode)
        self._tau_cmd: Optional[float] = None  # ramped gripper torque actually sent
        self._last: Optional[BusReading] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._first_tick = threading.Event()
        self._opened = False
        self._enabled = False
        self._advance_deadline: Optional[float] = None

    # ------------------------------------------------------------------ units
    def gripper_to_motor(self, opening_m: float) -> float:
        return opening_m / self.grip.open_m * self.grip.motor_open_rad

    def gripper_from_motor(self, motor_rad: float) -> float:
        return motor_rad / self.grip.motor_open_rad * self.grip.open_m

    # -------------------------------------------------------------- lifecycle
    def _open(self) -> None:
        if not self._opened:
            self.bus.open()
            self._opened = True

    def _read_until_all_reply(self, tries: int) -> BusReading:
        # Replies show up one call late: retry until every motor has answered.
        for _ in range(tries):
            r = self.bus.refresh()
            if all(n > 0 for n in r.responses):
                return r
            time.sleep(0.01)
        silent = [i + 1 for i, n in enumerate(r.responses) if n == 0]
        raise RuntimeError(f"motors {silent} did not reply (1-7 = joints, 8 = gripper) on {self.hw.can_interface}")

    def read_passive(self) -> ArmState:
        """Reads the joints without enabling or moving anything."""
        self._open()
        r = self._read_until_all_reply(10)
        return self._make_state(r, None, None)

    def start(self) -> None:
        """Enables the motors and holds the *measured* pose (no jump)."""
        if self._thread is not None:
            return
        self._open()
        self.bus.enable()
        self._enabled = True
        try:
            r = self._read_until_all_reply(5)
            measured = self._measured(r)
            self._check_startup_limits(measured[:N_JOINTS])
        except Exception:
            self._disable()
            raise
        self._last = r
        with self._lock:
            self._target = self.filter.reset(measured)
            self._interp.reset(self._target, time.monotonic())
            self._fault = None
            self._tau_target = self._tau_cmd = None
            self._state = self._make_state(r, self._target, None)
        self._stop_evt.clear()
        self._first_tick.clear()
        self._thread = threading.Thread(target=self._loop, name="openarm_servo", daemon=True)
        self._thread.start()
        if not self._first_tick.wait(timeout=1.0):
            self.stop(rest=False)
            raise RuntimeError("servo thread did not start")
        logger.info("%s arm streaming at %.0f Hz on %s", self.cfg.robot.side, self.hw.control_rate_hz,
                    self.hw.can_interface)

    def stop(self, rest: bool = True) -> None:
        """Moves to the rest pose (if healthy and ``rest``), stops the thread, disables the motors."""
        if self._thread is not None and self._fault is None and rest:
            try:
                if self._tau_target is not None:
                    logger.info("gripper torque mode off (releases the squeeze)")
                    self.set_gripper_torque(None)
                    time.sleep(2 * self._dt)
                logger.info("moving to the rest pose before disabling")
                self.move_joints(np.asarray(self.cfg.robot.rest_qpos), duration=self.hw.rest_duration_s)
            except Exception as exc:  # noqa: BLE001 - shutdown must continue
                logger.error("rest move failed: %s", exc)
        if self._thread is not None:
            self._stop_evt.set()
            self._thread.join(timeout=2.0)
            self._thread = None
        self._disable()

    def _disable(self) -> None:
        if self._enabled:
            try:
                self.bus.disable()
            except Exception as exc:  # noqa: BLE001
                logger.error("disable failed: %s", exc)
            self._enabled = False

    def __enter__(self) -> "Arm":
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    # -------------------------------------------------------------------- I/O
    def state(self) -> ArmState:
        with self._lock:
            if self._state is None:
                raise RuntimeError("arm not started")
            return self._state

    @property
    def fault(self) -> Optional[str]:
        return self.state().fault

    def targets(self) -> Tuple[NDArray, float]:
        """Current caller targets: joints (7,) and gripper opening (m), before the safety filter."""
        with self._lock:
            return self._target[:N_JOINTS].copy(), float(self._target[G])

    def set_targets(self, q: Optional[NDArray] = None, gripper: Optional[float] = None) -> None:
        """New position targets (non-blocking). In torque mode the gripper target is ignored."""
        with self._lock:
            if q is not None:
                self._target[:N_JOINTS] = q
            if gripper is not None and self._tau_target is None:
                self._target[G] = gripper
            self._interp.set_goal(self._target, time.monotonic())

    def set_gripper_torque(self, torque_nm: Optional[float]) -> None:
        """Torque mode with this squeeze (Nm, + closes), or ``None`` = position mode holding the
        measured opening, which RELEASES the squeeze: only while the object is supported."""
        lim = self.grip.max_torque_nm
        with self._lock:
            self._tau_target = None if torque_nm is None else float(np.clip(torque_nm, -lim, lim))

    def clear_fault(self) -> None:
        with self._lock:
            self._clear_fault_req = True

    def advance(self, dt: float) -> None:
        """Sleeps until the caller's next ``dt`` deadline (a fixed-rate caller loop)."""
        now = time.monotonic()
        deadline = self._advance_deadline
        if deadline is None or deadline < now - dt:  # first call, or fell behind: resync
            deadline = now
        deadline += dt
        self._advance_deadline = deadline
        time.sleep(max(0.0, deadline - time.monotonic()))

    def move_joints(self, q: NDArray, gripper: Optional[float] = None, duration: float = 3.0,
                    rate_hz: float = 100.0) -> None:
        """Blocking minimum-jerk joint move from the current *targets* (not the sagging measured pose)."""
        q0, g0 = self.targets()
        q1 = np.clip(np.asarray(q, dtype=np.float64), self.lower, self.upper)
        g1 = g0 if gripper is None else gripper
        steps = max(1, int(duration * rate_hz))
        for k in range(1, steps + 1):
            a = minimum_jerk(k / steps)
            self.set_targets(q0 + a * (q1 - q0), g0 + a * (g1 - g0) if gripper is not None else None)
            self.advance(1.0 / rate_hz)
            if self.fault:
                raise RuntimeError(f"move aborted: {self.fault}")

    # ----------------------------------------------------------- servo loop
    def _loop(self) -> None:
        if self.hw.realtime_priority > 0:
            try:
                os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(self.hw.realtime_priority))
            except (PermissionError, AttributeError, OSError) as exc:
                logger.warning("SCHED_FIFO not available (%s)", exc)
        dt = self._dt
        last_resp: Optional[NDArray] = None
        last_reply_t = np.full(N_JOINTS + 1, time.monotonic())
        track_since: Optional[float] = None
        next_tick = time.monotonic()

        while not self._stop_evt.is_set():
            with self._lock:
                if self._clear_fault_req:
                    self._clear_fault_req = False
                    if self._fault is not None and self._state is not None:
                        logger.warning("fault cleared: %s", self._fault)
                        self._fault = None
                        self._target = self.filter.reset(np.append(self._state.q, self._state.gripper))
                        self._interp.reset(self._target, time.monotonic())
                target = self._hold.copy() if self._fault is not None else self._interp.value(time.monotonic())
                tau_target = self._tau_target

            setpoint = self.filter.step(target, dt)
            tau_ff = np.zeros(N_JOINTS)
            if self.hw.gravity_compensation_scale > 0:
                tau_ff = self.hw.gravity_compensation_scale * self.gravity_fn(setpoint[:N_JOINTS])
            gq, gkp, gkd, gtau = self._gripper_command(setpoint, tau_target, dt)
            self.bus.command(setpoint[:N_JOINTS], self._kp, self._kd, tau_ff, gq, gkp, gkd, gtau)
            r = self.bus.receive(self.hw.recv_timeout_us)
            now = time.monotonic()
            self._last = r

            # Torque mode: keep the gripper setpoint on the measured opening, so leaving torque
            # mode holds the current opening instead of jumping.
            if self._tau_cmd is not None:
                g_meas = self.gripper_from_motor(r.gripper_q)
                self.filter.reset_entry(G, g_meas)
                setpoint[G] = self.filter.setpoint[G]
                with self._lock:
                    self._hold[G] = setpoint[G]
                    if self._tau_target is not None:
                        self._target[G] = setpoint[G]
                        self._interp.set_entry(G, setpoint[G])

            new_fault = self._check_faults(r, setpoint, now, last_resp, last_reply_t)
            resp = np.asarray(r.responses)
            if last_resp is not None:
                last_reply_t[resp > last_resp] = now
            last_resp = resp
            err = float(np.max(np.abs(setpoint[:N_JOINTS] - r.q)))
            if err > self.hw.max_tracking_error_rad:
                track_since = track_since or now
                if now - track_since > self.hw.tracking_error_time_s:
                    new_fault = new_fault or f"tracking error {err:.2f} rad (collision or blocked joint?)"
            else:
                track_since = None

            with self._lock:
                if new_fault and self._fault is None:
                    self._fault = new_fault
                    logger.error("FAULT: %s -> holding the measured pose", new_fault)
                    self._hold = self.filter.reset(self._measured(r))
                    setpoint = self._hold.copy()
                    track_since = None
                self._state = self._make_state(r, setpoint, self._fault)
            self._first_tick.set()

            next_tick += dt
            remaining = next_tick - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            else:
                next_tick = time.monotonic()  # overrun: resync instead of bursting

    def _check_faults(self, r: BusReading, setpoint: NDArray, now: float, last_resp: Optional[NDArray],
                      last_reply_t: NDArray) -> Optional[str]:
        if not r.bus_healthy:
            return f"CAN bus unhealthy ({self.hw.can_interface})"
        for i, code in enumerate(r.error_codes):
            if code >= 0x8:
                name = "gripper" if i == N_JOINTS else f"joint{i + 1}"
                return f"{name}: motor error {MOTOR_ERRORS.get(code, hex(code))}"
        if last_resp is not None:
            got = np.asarray(r.responses) > last_resp
            stale = (now - last_reply_t > self.hw.missing_reply_time_s) & ~got
            if stale.any():
                names = ["gripper" if i == N_JOINTS else f"joint{i + 1}" for i in np.flatnonzero(stale)]
                return f"{names} stopped replying"
        return None

    def _gripper_command(self, setpoint: NDArray, tau_target: Optional[float], dt: float):
        """Gripper MIT frame ``(q_motor, kp, kd, tau_motor)`` for this tick."""
        q_motor = self.gripper_to_motor(setpoint[G])
        if tau_target is None:
            self._tau_cmd = None
            return q_motor, self.grip.kp, self.grip.kd, 0.0
        cmd = self._tau_cmd
        if cmd is None:  # entering torque mode: start from the torque the motor outputs now
            cmd = 0.0 if self._last is None else self._close_sign * self._last.gripper_tau
            cmd = float(np.clip(cmd, -self.grip.max_torque_nm, self.grip.max_torque_nm))
        step = self.grip.torque_rate_nm_s * dt
        cmd += float(np.clip(tau_target - cmd, -step, step))
        self._tau_cmd = cmd
        return q_motor, 0.0, self.grip.torque_kd, self._close_sign * cmd

    # ---------------------------------------------------------------- helpers
    def _measured(self, r: BusReading) -> NDArray:
        return np.append(r.q, self.gripper_from_motor(r.gripper_q))

    def _check_startup_limits(self, q: NDArray) -> None:
        tol = self.hw.startup_limit_tolerance_rad
        bad = [f"joint{i + 1}={v:+.3f} rad (limits [{lo:+.3f}, {hi:+.3f}])"
               for i, (v, lo, hi) in enumerate(zip(q, self.lower, self.upper)) if v < lo - tol or v > hi + tol]
        if bad:
            raise RuntimeError(
                "measured joints are outside the official limits: the motor zero calibration is likely off. "
                "Run `openarm-can-zero-position-calibration` (docs/hardware_bringup.md) first:\n  "
                + "\n  ".join(bad))

    def _make_state(self, r: BusReading, setpoint: Optional[NDArray], fault: Optional[str]) -> ArmState:
        sp = self._measured(r) if setpoint is None else setpoint
        return ArmState(q=r.q.copy(), dq=r.dq.copy(), tau=np.asarray(r.tau, dtype=np.float64).copy(),
                        gripper=self.gripper_from_motor(r.gripper_q), q_setpoint=sp[:N_JOINTS].copy(),
                        gripper_setpoint=float(sp[G]), gripper_tau=self._close_sign * r.gripper_tau,
                        gripper_torque_cmd=self._tau_cmd, motor_errors=tuple(r.error_codes), fault=fault,
                        timestamp=time.monotonic())
