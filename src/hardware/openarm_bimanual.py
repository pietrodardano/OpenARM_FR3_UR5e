"""
Real OpenArm v1 bimanual driver: plain Python + one streaming thread (no ROS).

Equivalent of a ``ur_rtde`` servoJ streamer, except that OpenArm has no arm
controller box: this thread *is* the servo loop. Every tick (default 250 Hz) it

1. copies the latest 16-dim target set by the caller (``set_target`` /
   ``set_joint_targets``, lock-protected, non-blocking), spread over the caller's
   command period by :class:`~src.control.robot_interface.TargetInterpolator` so a
   10-50 Hz caller yields a smooth setpoint instead of a 50 Hz staircase,
2. passes it through :class:`~src.control.safety.JointSafetyFilter` (official
   joint limits minus a margin + per-joint velocity limit),
3. sends one MIT impedance frame per motor on ``can1`` (left) and ``can0``
   (right) via ``openarm_can``, optionally with gravity feedforward from the
   official MuJoCo model,
4. reads the replies and publishes a :class:`~src.control.robot_interface.RobotState`,
5. checks faults: motor error codes, motors that stop replying, bus-off, and
   sustained tracking error (collision / blocked joint). A fault latches and
   freezes the robot at its measured pose until :meth:`clear_fault`.

Only the arms in ``active_sides`` are opened and commanded (e.g. ``("left",)``
while the right arm is broken); the other arm is reported frozen at its rest pose.

Grippers run in position mode (MIT, gripper kp/kd) or, after
:meth:`set_gripper_torque`, in torque mode: MIT with kp = 0, so the motor's
current loop outputs the commanded torque minus ``gripper_torque_kd * dq``. The
torque is capped at ``gripper_max_torque`` and ramped at ``gripper_torque_rate``.
Torque mode survives a fault (a grasped object is not dropped).

The Damiao motors run their own fast PD loop on the last MIT setpoint, so a few
milliseconds of Python jitter only delays setpoint updates; it does not
destabilize the joints.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from ..control.robot_interface import (
    ARM_DOF,
    CMD_DIM,
    GRIPPER_OPEN_M,
    SIDES,
    ArmState,
    JointLimits,
    RobotBackend,
    RobotState,
    Side,
    TargetInterpolator,
    arm_slice,
    gripper_index,
    pack_command,
)
from ..control.safety import JointSafetyFilter
from .openarm_bus import ArmBusConfig, BusReading, OpenArmBus

logger = logging.getLogger("RoboTokens.OpenArmHW")

_MOTOR_ERRORS = {0x8: "OVERVOLTAGE", 0x9: "UNDERVOLTAGE", 0xA: "OVERCURRENT", 0xB: "MOS_OVERHEAT",
                 0xC: "COIL_OVERHEAT", 0xD: "COMMUNICATION_LOST", 0xE: "OVERLOAD"}


@dataclass
class OpenArmHardwareConfig:
    """Driver parameters (loaded from ``configs/task_config.yaml`` -> ``hardware``)."""

    arms: Dict[Side, ArmBusConfig] = field(
        default_factory=lambda: {"left": ArmBusConfig("can1"), "right": ArmBusConfig("can0")}
    )
    # MIT gains, official v1 defaults (openarm_description control_gains.yaml).
    kp: Sequence[float] = (70.0, 70.0, 70.0, 60.0, 10.0, 10.0, 10.0)
    kd: Sequence[float] = (2.75, 2.5, 2.0, 2.0, 0.7, 0.6, 0.5)
    gripper_kp: float = 5.0
    gripper_kd: float = 0.1
    # Gripper motor angle at full opening (0.044 m); 0 rad == closed (openarm_ros2 mapping).
    gripper_motor_open_rad: float = -1.0472
    control_rate_hz: float = 250.0
    recv_timeout_us: int = 300
    max_joint_velocity: Sequence[float] = (1.0, 1.0, 1.5, 1.5, 2.0, 2.0, 2.0)  # rad/s
    max_gripper_velocity: float = 0.022  # m/s, full 0.044 m stroke in 2 s
    joint_limit_margin: float = 0.02  # rad
    # Refuse to start if a measured joint is further than this outside its limits:
    # it means the motor zero calibration is wrong and every command would be off.
    startup_limit_tolerance: float = 0.15  # rad
    max_tracking_error: float = 0.5  # rad, |setpoint - measured| ...
    tracking_error_time: float = 0.3  # ... sustained this long => fault
    missing_reply_time: float = 0.1  # s without any reply from a motor => fault
    rest_qpos: Dict[Side, Sequence[float]] = field(
        default_factory=lambda: {s: (0.0,) * ARM_DOF for s in SIDES}  # arms hanging straight down
    )
    rest_duration: float = 4.0  # s, move to rest before disabling on stop()
    disable_on_stop: bool = True
    gravity_compensation_scale: float = 0.0  # 0 = off, 1 = full model gravity feedforward
    realtime_priority: int = 0  # >0: request SCHED_FIFO with this priority for the loop thread
    active_sides: Tuple[Side, ...] = SIDES  # arms to drive; the others are never opened
    # Gripper torque mode (set_gripper_torque).
    gripper_max_torque: float = 1.0  # Nm on the motor, |command| is clipped to this
    gripper_torque_kd: float = 0.3  # Nm s/rad damping: free closing speed ~ torque / kd
    gripper_torque_rate: float = 1.0  # Nm/s ramp of the torque command


class OpenArmBimanual(RobotBackend):
    """Both OpenArm v1 arms over SocketCAN, streamed from a Python thread.

    Args:
        cfg: Driver configuration.
        limits: Official joint limits (from :class:`~src.planning.openarm_kinematics.OpenArmKinematics`).
        gravity_fn: Optional ``q_arm -> {side: tau (7,)}`` used when
            ``cfg.gravity_compensation_scale > 0`` (normally ``OpenArmKinematics.gravity_torques``
            of a dedicated instance: it is called from the control thread).
        bus_factory: ``ArmBusConfig -> bus``; tests pass a fake bus here.
    """

    def __init__(
        self,
        cfg: OpenArmHardwareConfig,
        limits: JointLimits,
        gravity_fn: Optional[Callable[[Mapping[Side, NDArray[np.float64]]], Dict[Side, NDArray[np.float64]]]] = None,
        bus_factory: Callable[[ArmBusConfig], OpenArmBus] = OpenArmBus,
    ) -> None:
        self.cfg = cfg
        self.limits = limits
        self.gravity_fn = gravity_fn
        if cfg.gravity_compensation_scale > 0 and gravity_fn is None:
            raise ValueError("gravity_compensation_scale > 0 needs a gravity_fn")
        if not cfg.active_sides or any(s not in SIDES for s in cfg.active_sides):
            raise ValueError(f"active_sides must be a non-empty subset of {SIDES}, got {cfg.active_sides}")
        self.active_sides = tuple(s for s in SIDES if s in cfg.active_sides)
        self.buses = {side: bus_factory(cfg.arms[side]) for side in self.active_sides}
        # Inactive arms: a constant reading at the rest pose stands in for the bus.
        self._frozen = {
            side: BusReading(q=np.asarray(cfg.rest_qpos[side], dtype=np.float64), dq=np.zeros(ARM_DOF),
                             tau=np.zeros(ARM_DOF), gripper_q=0.0,
                             error_codes=[0] * (ARM_DOF + 1), responses=[0] * (ARM_DOF + 1))
            for side in SIDES if side not in self.active_sides
        }
        self.filter = JointSafetyFilter(
            limits, cfg.max_joint_velocity, cfg.max_gripper_velocity, cfg.joint_limit_margin
        )
        self._kp = np.asarray(cfg.kp, dtype=np.float64)
        self._kd = np.asarray(cfg.kd, dtype=np.float64)
        self._dt = 1.0 / cfg.control_rate_hz

        self._lock = threading.Lock()
        self._target = np.zeros(CMD_DIM)  # latest caller target (the interpolator's goal)
        self._interp = TargetInterpolator()
        self._hold = np.zeros(CMD_DIM)  # frozen setpoint while a fault is latched
        self._state: Optional[RobotState] = None
        self._fault: Optional[str] = None
        self._clear_fault_req = False
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._first_tick = threading.Event()
        self._opened = False
        self._enabled = False
        # Gripper torque mode: requested torque (None = position mode) and the ramped command.
        self._close_sign = -1.0 if cfg.gripper_motor_open_rad > 0 else 1.0  # motor sign of "closing"
        self._grip_tau_target: Dict[Side, Optional[float]] = {s: None for s in SIDES}
        self._grip_tau_cmd: Dict[Side, Optional[float]] = {s: None for s in SIDES}
        self._last_readings: Optional[Dict[Side, BusReading]] = None

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], bus_factory: Callable[[ArmBusConfig], OpenArmBus] = OpenArmBus) -> "OpenArmBimanual":
        """Builds the driver from ``configs/task_config.yaml`` (sections ``robot`` + ``hardware``)."""
        from ..planning.openarm_kinematics import OpenArmKinematics

        hw, g = cfg["hardware"], cfg["hardware"]["gripper"]
        arms = {
            side: ArmBusConfig(
                can_interface=hw["arms"][side]["can_interface"],
                can_fd=hw["can_fd"],
                motor_types=tuple(hw["motor_types"]),
                send_can_ids=tuple(hw["send_can_ids"]),
                recv_can_ids=tuple(hw["recv_can_ids"]),
                gripper_motor_type=g["motor_type"],
                gripper_send_can_id=g["send_can_id"],
                gripper_recv_can_id=g["recv_can_id"],
            )
            for side in SIDES
        }
        hw_cfg = OpenArmHardwareConfig(
            arms=arms,
            kp=tuple(hw["kp"]),
            kd=tuple(hw["kd"]),
            gripper_kp=g["kp"],
            gripper_kd=g["kd"],
            gripper_motor_open_rad=g["motor_open_rad"],
            control_rate_hz=hw["control_rate_hz"],
            recv_timeout_us=hw["recv_timeout_us"],
            max_joint_velocity=tuple(hw["max_joint_velocity_rad_s"]),
            max_gripper_velocity=hw["max_gripper_velocity_m_s"],
            joint_limit_margin=hw["joint_limit_margin_rad"],
            startup_limit_tolerance=hw["startup_limit_tolerance_rad"],
            max_tracking_error=hw["max_tracking_error_rad"],
            tracking_error_time=hw["tracking_error_time_s"],
            missing_reply_time=hw["missing_reply_time_s"],
            rest_qpos={s: tuple(hw["rest_qpos"][s]) for s in SIDES},
            rest_duration=hw["rest_duration_s"],
            gravity_compensation_scale=hw["gravity_compensation_scale"],
            realtime_priority=hw["realtime_priority"],
            active_sides=tuple(hw.get("active_sides", SIDES)),
            gripper_max_torque=g.get("max_torque_nm", 1.0),
            gripper_torque_kd=g.get("torque_kd", 0.3),
            gripper_torque_rate=g.get("torque_rate_nm_s", 1.0),
        )
        limits = OpenArmKinematics(cfg["robot"]["mjcf_path"]).limits
        gravity_fn = None
        if hw_cfg.gravity_compensation_scale > 0:
            # Dedicated model instance: it is called from the control thread.
            gravity_fn = OpenArmKinematics(cfg["robot"]["mjcf_path"]).gravity_torques
        return cls(hw_cfg, limits, gravity_fn=gravity_fn, bus_factory=bus_factory)

    # -------------------------------------------------------------- units

    def _gripper_to_motor(self, opening_m: float) -> float:
        return opening_m / GRIPPER_OPEN_M * self.cfg.gripper_motor_open_rad

    def _gripper_from_motor(self, motor_rad: float) -> float:
        return motor_rad / self.cfg.gripper_motor_open_rad * GRIPPER_OPEN_M

    # ---------------------------------------------------------- lifecycle

    def open(self) -> None:
        """Opens both CAN buses and registers the motors. Does not enable or move anything."""
        if not self._opened:
            for bus in self.buses.values():
                bus.open()
            self._opened = True

    def read_passive(self) -> RobotState:
        """Reads joint positions with the motors left as they are (no enable, no motion)."""
        self.open()
        # Replies land after the first refresh returns (they show up one call late): retry until
        # every motor has answered, otherwise the first read is the all-zero default.
        for _ in range(10):
            readings = self._read_all(lambda bus: bus.refresh())
            if all(r > 0 for side in self.active_sides for r in readings[side].responses):
                break
            time.sleep(0.01)
        silent = {side: [i + 1 for i, r in enumerate(readings[side].responses) if r == 0] for side in self.active_sides}
        silent = {side: ids for side, ids in silent.items() if ids}
        if silent:
            raise RuntimeError(f"motors did not reply (1-7 = joints, 8 = gripper): {silent}")
        return self._make_state(readings, setpoint=None, fault=None)

    def start(self) -> None:
        """Enables the motors and starts holding the *measured* pose (no jump)."""
        if self._thread is not None:
            return
        self.open()
        for bus in self.buses.values():
            bus.enable()
        self._enabled = True
        # recv_all() only waits for the first reply, so late replies can be missed: retry a few times.
        for _ in range(5):
            readings = self._read_all(lambda bus: bus.refresh())
            if all(r > 0 for side in self.active_sides for r in readings[side].responses):
                break
            time.sleep(0.01)
        for side in self.active_sides:
            if not all(r > 0 for r in readings[side].responses):
                silent = [i + 1 for i, r in enumerate(readings[side].responses) if r == 0]
                self._shutdown_buses()
                raise RuntimeError(f"{side} arm ({self.cfg.arms[side].can_interface}): motors {silent} did not reply "
                                   "(drive only the other arm with hardware.active_sides)")
        measured = self._measured_vector(readings)
        self._check_startup_limits(measured)
        self._last_readings = readings

        with self._lock:
            self._target = self.filter.reset(measured)
            self._interp.reset(self._target, time.monotonic())
            self._fault = None
            self._grip_tau_target = {s: None for s in SIDES}
            self._grip_tau_cmd = {s: None for s in SIDES}
            self._state = self._make_state(readings, self._target, None)
        self._stop_evt.clear()
        self._first_tick.clear()
        self._thread = threading.Thread(target=self._loop, name="openarm_servo_loop", daemon=True)
        self._thread.start()
        if not self._first_tick.wait(timeout=1.0):
            self.stop()
            raise RuntimeError("OpenArm control thread did not start")
        logger.info("OpenArm streaming at %.0f Hz (%s)", self.cfg.control_rate_hz,
                    ", ".join(f"{s}={self.cfg.arms[s].can_interface}" for s in self.active_sides))

    def stop(self) -> None:
        """Moves to the rest pose (if healthy), stops the thread, disables the motors."""
        if self._thread is not None and self._fault is None and self.cfg.disable_on_stop:
            try:
                if any(t is not None for t in self._grip_tau_target.values()):
                    logger.info("Gripper torque mode off (releasing the squeeze)")
                    self.set_gripper_torque({s: None for s in self.active_sides})
                    time.sleep(2 * self._dt)
                logger.info("Moving to rest pose before disabling motors...")
                self.move_joints(
                    {s: np.asarray(self.cfg.rest_qpos[s]) for s in SIDES}, duration=self.cfg.rest_duration
                )
            except Exception as exc:  # noqa: BLE001 - shutdown must continue
                logger.error("Rest move failed: %s", exc)
        if self._thread is not None:
            self._stop_evt.set()
            self._thread.join(timeout=2.0)
            self._thread = None
        self._shutdown_buses()

    def _shutdown_buses(self) -> None:
        if self._enabled and self.cfg.disable_on_stop:
            for bus in self.buses.values():
                try:
                    bus.disable()
                except Exception as exc:  # noqa: BLE001
                    logger.error("disable failed: %s", exc)
            self._enabled = False

    # ---------------------------------------------------------------- I/O

    def get_state(self) -> RobotState:
        with self._lock:
            if self._state is None:
                raise RuntimeError("OpenArmBimanual not started")
            return self._state

    def set_joint_targets(
        self,
        q_arm: Optional[Mapping[Side, NDArray[np.float64]]] = None,
        gripper: Optional[Mapping[Side, float]] = None,
    ) -> None:
        with self._lock:
            for side, q in (q_arm or {}).items():
                if side in self.active_sides:  # an inactive arm stays at its rest pose
                    self._target[arm_slice(side)] = q
            for side, g in (gripper or {}).items():
                # In torque mode the loop keeps the gripper target at the measured opening.
                if side in self.active_sides and self._grip_tau_target[side] is None:
                    self._target[gripper_index(side)] = g
            self._interp.set_goal(self._target, time.monotonic())

    def set_gripper_torque(self, torque_nm: Mapping[Side, Optional[float]]) -> None:
        lim = self.cfg.gripper_max_torque
        with self._lock:
            for side, tau in torque_nm.items():
                if side not in self.active_sides:
                    raise ValueError(f"{side} arm is not active (hardware.active_sides = {list(self.active_sides)})")
                self._grip_tau_target[side] = None if tau is None else float(np.clip(tau, -lim, lim))

    def get_joint_targets(self) -> NDArray[np.float64]:
        with self._lock:
            return self._target.copy()

    def clear_fault(self) -> None:
        with self._lock:
            self._clear_fault_req = True

    # ------------------------------------------------------- control loop

    def _loop(self) -> None:
        if self.cfg.realtime_priority > 0:
            try:
                os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(self.cfg.realtime_priority))
            except (PermissionError, AttributeError, OSError) as exc:
                logger.warning("SCHED_FIFO not available (%s); running at normal priority", exc)

        dt = self._dt
        last_responses = {s: None for s in SIDES}
        last_reply_time = {s: np.full(ARM_DOF + 1, time.monotonic()) for s in SIDES}
        track_err_since: Optional[float] = None
        ticks, overruns, max_late = 0, 0, 0.0
        next_tick = time.monotonic()

        while not self._stop_evt.is_set():
            with self._lock:
                if self._clear_fault_req:
                    self._clear_fault_req = False
                    if self._fault is not None and self._state is not None:
                        logger.warning("Fault cleared: %s", self._fault)
                        self._fault = None
                        self._target = self.filter.reset(self._state.command_vector())
                        self._interp.reset(self._target, time.monotonic())
                # While faulted, new targets are ignored and the frozen pose is held.
                target = self._hold.copy() if self._fault is not None else self._interp.value(time.monotonic())
                grip_tau_target = dict(self._grip_tau_target)

            setpoint = self.filter.step(target, dt)
            tau_ff = {s: np.zeros(ARM_DOF) for s in SIDES}
            if self.cfg.gravity_compensation_scale > 0:
                g = self.gravity_fn({s: setpoint[arm_slice(s)] for s in SIDES})
                tau_ff = {s: self.cfg.gravity_compensation_scale * g[s] for s in SIDES}

            for side in self.active_sides:
                gq, gkp, gkd, gtau = self._gripper_command(side, setpoint, grip_tau_target[side], dt)
                self.buses[side].command(
                    setpoint[arm_slice(side)], self._kp, self._kd, tau_ff[side], gq, gkp, gkd, gtau,
                )
            readings = self._read_all(lambda bus: bus.receive(self.cfg.recv_timeout_us))
            now = time.monotonic()
            self._last_readings = readings

            # Torque-mode grippers: keep their position setpoint/target on the measured opening,
            # so leaving torque mode holds the current opening instead of jumping.
            for side in self.active_sides:
                if self._grip_tau_cmd[side] is not None:
                    gi = gripper_index(side)
                    g_meas = self._gripper_from_motor(readings[side].gripper_q)
                    self.filter.reset_entry(gi, g_meas)
                    setpoint[gi] = self.filter.setpoint[gi]
                    with self._lock:
                        self._hold[gi] = setpoint[gi]
                        if self._grip_tau_target[side] is not None:  # not just switched back by the caller
                            self._target[gi] = setpoint[gi]
                            self._interp.set_entry(gi, setpoint[gi])

            # ---- fault checks (only the first fault is latched) ----
            new_fault = None
            for side in self.active_sides:
                r = readings[side]
                if not r.bus_healthy:
                    new_fault = new_fault or f"{side} CAN bus unhealthy ({self.cfg.arms[side].can_interface})"
                for i, code in enumerate(r.error_codes):
                    if code >= 0x8:
                        name = "gripper" if i == ARM_DOF else f"joint{i + 1}"
                        new_fault = new_fault or f"{side} {name}: motor error {_MOTOR_ERRORS.get(code, hex(code))}"
                resp = np.asarray(r.responses)
                if last_responses[side] is not None:
                    last_reply_time[side][resp > last_responses[side]] = now
                last_responses[side] = resp
                stale = now - last_reply_time[side] > self.cfg.missing_reply_time
                if stale.any():
                    names = ["gripper" if i == ARM_DOF else f"joint{i + 1}" for i in np.flatnonzero(stale)]
                    new_fault = new_fault or f"{side} {names} stopped replying"
            measured = self._measured_vector(readings)
            arm_err = max(np.max(np.abs(setpoint[arm_slice(s)] - measured[arm_slice(s)])) for s in self.active_sides)
            if arm_err > self.cfg.max_tracking_error:
                track_err_since = track_err_since or now
                if now - track_err_since > self.cfg.tracking_error_time:
                    new_fault = new_fault or f"tracking error {arm_err:.2f} rad (collision or blocked joint?)"
            else:
                track_err_since = None

            ticks += 1
            stats = {"ticks": ticks, "overruns": overruns, "max_late_ms": max_late * 1e3}
            with self._lock:
                if new_fault and self._fault is None:
                    self._fault = new_fault
                    logger.error("FAULT: %s -> holding measured pose", new_fault)
                    # Freeze: hold where the robot actually is, ignore new targets until cleared.
                    self._hold = self.filter.reset(measured)
                    setpoint = self._hold.copy()
                    track_err_since = None
                self._state = self._make_state(readings, setpoint, self._fault, stats)
            self._first_tick.set()

            next_tick += dt
            remaining = next_tick - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            else:
                overruns += 1
                max_late = max(max_late, -remaining)
                next_tick = time.monotonic()  # slip: resync instead of bursting

    # ------------------------------------------------------------ helpers

    def _read_all(self, read: Callable[[OpenArmBus], BusReading]) -> Dict[Side, BusReading]:
        """One reading per side: from the bus for active arms, the frozen rest reading otherwise."""
        return {s: read(self.buses[s]) if s in self.buses else self._frozen[s] for s in SIDES}

    def _gripper_command(
        self, side: Side, setpoint: NDArray[np.float64], tau_target: Optional[float], dt: float
    ) -> Tuple[float, float, float, float]:
        """Gripper MIT frame ``(q_motor, kp, kd, tau_motor)`` for this tick (position or torque mode)."""
        if tau_target is None:
            self._grip_tau_cmd[side] = None
            return self._gripper_to_motor(setpoint[gripper_index(side)]), self.cfg.gripper_kp, self.cfg.gripper_kd, 0.0
        cmd = self._grip_tau_cmd[side]
        if cmd is None:  # entering torque mode: start from the torque the motor outputs now (no jump)
            last = self._last_readings
            cmd = 0.0 if last is None else self._close_sign * last[side].gripper_tau
            cmd = float(np.clip(cmd, -self.cfg.gripper_max_torque, self.cfg.gripper_max_torque))
        step = self.cfg.gripper_torque_rate * dt
        cmd += float(np.clip(tau_target - cmd, -step, step))
        self._grip_tau_cmd[side] = cmd
        return self._gripper_to_motor(setpoint[gripper_index(side)]), 0.0, self.cfg.gripper_torque_kd, self._close_sign * cmd

    def _measured_vector(self, readings: Mapping[Side, BusReading]) -> NDArray[np.float64]:
        return pack_command(
            {s: readings[s].q for s in SIDES},
            {s: self._gripper_from_motor(readings[s].gripper_q) for s in SIDES},
        )

    def _check_startup_limits(self, measured: NDArray[np.float64]) -> None:
        tol = self.cfg.startup_limit_tolerance
        bad = []
        for side in self.active_sides:
            sl = arm_slice(side)
            for i, (q, lo, hi) in enumerate(zip(measured[sl], self.limits.lower[sl], self.limits.upper[sl])):
                if q < lo - tol or q > hi + tol:
                    bad.append(f"{side} joint{i + 1}={q:+.3f} rad (limits [{lo:+.3f}, {hi:+.3f}])")
        if bad:
            self._shutdown_buses()
            raise RuntimeError(
                "Measured joints are outside the official limits, motor zero calibration is likely off. "
                "Run `openarm-can-zero-position-calibration --robot-version v1 --arm-side <right_arm|left_arm>` "
                "(see docs/real_robot_deployment.md) before streaming:\n  " + "\n  ".join(bad)
            )

    def _make_state(
        self,
        readings: Mapping[Side, BusReading],
        setpoint: Optional[NDArray[np.float64]],
        fault: Optional[str],
        stats: Optional[dict] = None,
    ) -> RobotState:
        arms = {}
        for side in SIDES:
            r = readings[side]
            sp = setpoint if setpoint is not None else self._measured_vector(readings)
            arms[side] = ArmState(
                q=r.q.copy(),
                dq=r.dq.copy(),
                tau=r.tau.copy(),
                gripper=self._gripper_from_motor(r.gripper_q),
                q_setpoint=sp[arm_slice(side)].copy(),
                gripper_setpoint=float(sp[gripper_index(side)]),
                motor_errors=tuple(r.error_codes),
                gripper_tau=self._close_sign * r.gripper_tau,
                gripper_torque_cmd=self._grip_tau_cmd[side],
                active=side in self.active_sides,
            )
        return RobotState(timestamp=time.monotonic(), arms=arms, fault=fault, stats=dict(stats or {}))
