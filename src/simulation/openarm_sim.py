"""
MuJoCo stand-in for the real OpenArm driver, using the official v1 model.

Implements the same :class:`~src.control.robot_interface.RobotBackend` API as
:class:`~src.hardware.openarm_bimanual.OpenArmBimanual`, with the same
:class:`~src.control.safety.JointSafetyFilter` in the loop, so any controller,
scripted skill or policy can be dry-run here before touching the robot.

Two modes:

- ``realtime=True``: a physics thread steps the model against the wall clock
  (like the hardware thread); the optional passive viewer is synced from it.
- ``realtime=False``: nothing runs in the background; :meth:`advance` steps
  physics for exactly ``dt``. Deterministic and as fast as the CPU allows,
  for tests and data collection.

The 16 position actuators of ``openarm_bimanual.xml`` are ordered exactly like
the 16-dim command vector, so a setpoint maps 1:1 onto ``data.ctrl``.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import mujoco
import numpy as np
from numpy.typing import NDArray

from ..control.robot_interface import (
    ARM_DOF,
    CMD_DIM,
    SIDES,
    ArmState,
    JointLimits,
    RobotBackend,
    RobotState,
    Side,
    arm_slice,
    gripper_index,
    pack_command,
)
from ..control.safety import JointSafetyFilter
from ..planning.openarm_kinematics import load_openarm_spec

logger = logging.getLogger("RoboTokens.OpenArmSim")

DEFAULT_SCENE = "third_party/openarm/openarm_mujoco/v1/scene.xml"


class OpenArmSim(RobotBackend):
    """Official OpenArm v1 bimanual model behind the robot backend API.

    Args:
        mjcf_path: Scene to load (must include the official ``openarm_bimanual.xml``).
        realtime: Background physics thread against the wall clock (see module doc).
        render: Open the interactive passive viewer (``realtime=True`` only).
        control_rate_hz: Rate of the safety filter / setpoint update, as on hardware.
        max_joint_velocity / max_gripper_velocity / joint_limit_margin: Safety filter params.
        initial_qpos: Per-side arm joints at start (default: all zeros, i.e. how the
            real arms hang after power-on).
        gravity_compensation_scale: Gravity feedforward from the official model at the
            setpoint, applied as joint torque (same option as the hardware driver).
    """

    def __init__(
        self,
        mjcf_path: str = DEFAULT_SCENE,
        realtime: bool = False,
        render: bool = False,
        control_rate_hz: float = 250.0,
        max_joint_velocity: Union[float, Sequence[float]] = 1.5,
        max_gripper_velocity: float = 0.022,
        joint_limit_margin: float = 0.02,
        initial_qpos: Optional[Mapping[Side, Sequence[float]]] = None,
        gravity_compensation_scale: float = 1.0,
    ) -> None:
        self.model = load_openarm_spec(mjcf_path).compile()
        self.data = mujoco.MjData(self.model)
        self.realtime = realtime
        self.render = render
        self._substeps = max(1, int(round(1.0 / (control_rate_hz * self.model.opt.timestep))))
        self._ctrl_dt = self._substeps * self.model.opt.timestep

        self._qadr: Dict[Side, NDArray[np.int64]] = {}
        self._vadr: Dict[Side, NDArray[np.int64]] = {}
        self._gadr: Dict[Side, int] = {}
        self._act: Dict[Side, NDArray[np.int64]] = {}
        lower = np.zeros(CMD_DIM)
        upper = np.zeros(CMD_DIM)
        for side in SIDES:
            jids = [self.model.joint(f"openarm_{side}_joint{i}").id for i in range(1, ARM_DOF + 1)]
            self._qadr[side] = self.model.jnt_qposadr[jids]
            self._vadr[side] = self.model.jnt_dofadr[jids]
            self._gadr[side] = int(self.model.jnt_qposadr[self.model.joint(f"openarm_{side}_finger_joint1").id])
            self._act[side] = np.array([self.model.actuator(f"{side}_joint{i}_ctrl").id for i in range(1, ARM_DOF + 1)])
            lower[arm_slice(side)] = self.model.jnt_range[jids, 0]
            upper[arm_slice(side)] = self.model.jnt_range[jids, 1]
            grange = self.model.jnt_range[self.model.joint(f"openarm_{side}_finger_joint1").id]
            lower[gripper_index(side)], upper[gripper_index(side)] = grange
        self.limits = JointLimits(lower=lower, upper=upper)
        self.filter = JointSafetyFilter(self.limits, max_joint_velocity, max_gripper_velocity, joint_limit_margin)
        self.gravity_scale = gravity_compensation_scale
        self._gravity_kin = None
        if gravity_compensation_scale > 0:
            from ..planning.openarm_kinematics import OpenArmKinematics

            self._gravity_kin = OpenArmKinematics()
        self._initial = {s: np.asarray((initial_qpos or {}).get(s, np.zeros(ARM_DOF)), dtype=np.float64) for s in SIDES}

        self._lock = threading.RLock()
        self._target = np.zeros(CMD_DIM)
        self._setpoint = np.zeros(CMD_DIM)
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._viewer = None
        self._started = False

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], realtime: bool = False, render: bool = False) -> "OpenArmSim":
        """Same safety parameters as the hardware driver (``configs/task_config.yaml``)."""
        hw = cfg["hardware"]
        return cls(
            mjcf_path=cfg["robot"]["sim_scene_path"],
            realtime=realtime or render,
            render=render,
            control_rate_hz=hw["control_rate_hz"],
            max_joint_velocity=hw["max_joint_velocity_rad_s"],
            max_gripper_velocity=hw["max_gripper_velocity_m_s"],
            joint_limit_margin=hw["joint_limit_margin_rad"],
            initial_qpos=hw["rest_qpos"],
            gravity_compensation_scale=cfg["robot"].get("sim_gravity_compensation_scale", 1.0),
        )

    # ---------------------------------------------------------- lifecycle

    def reset(self) -> None:
        """Resets physics to ``initial_qpos`` with closed grippers and re-anchors the filter."""
        with self._lock:
            mujoco.mj_resetData(self.model, self.data)
            for side in SIDES:
                self.data.qpos[self._qadr[side]] = self._initial[side]
            # Grippers closed: the calibrated real robot rests with closed grippers.
            self.data.qpos[[self._gadr[s] for s in SIDES]] = 0.0
            mujoco.mj_forward(self.model, self.data)
            measured = self._measured_vector()
            self._target = self.filter.reset(measured)
            self._setpoint = self._target.copy()
            self.data.ctrl[:] = self._setpoint

    def start(self) -> None:
        if self._started:
            return
        self.reset()
        self._started = True
        if self.realtime:
            if self.render:
                import mujoco.viewer

                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._stop_evt.clear()
            self._thread = threading.Thread(target=self._loop, name="openarm_sim_loop", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        if self._thread is not None:
            self._stop_evt.set()
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        self._started = False

    # ---------------------------------------------------------------- I/O

    def get_state(self) -> RobotState:
        with self._lock:
            arms = {}
            for side in SIDES:
                arms[side] = ArmState(
                    q=self.data.qpos[self._qadr[side]].copy(),
                    dq=self.data.qvel[self._vadr[side]].copy(),
                    tau=self.data.actuator_force[self._act[side]].copy(),
                    gripper=float(self.data.qpos[self._gadr[side]]),
                    q_setpoint=self._setpoint[arm_slice(side)].copy(),
                    gripper_setpoint=float(self._setpoint[gripper_index(side)]),
                )
            return RobotState(timestamp=float(self.data.time), arms=arms, fault=None, stats={"sim_time": float(self.data.time)})

    def set_joint_targets(
        self,
        q_arm: Optional[Mapping[Side, NDArray[np.float64]]] = None,
        gripper: Optional[Mapping[Side, float]] = None,
    ) -> None:
        with self._lock:
            for side, q in (q_arm or {}).items():
                self._target[arm_slice(side)] = q
            for side, g in (gripper or {}).items():
                self._target[gripper_index(side)] = g

    def get_joint_targets(self) -> NDArray[np.float64]:
        with self._lock:
            return self._target.copy()

    def advance(self, dt: float) -> None:
        if self.realtime:
            super().advance(dt)
            return
        for _ in range(max(1, int(round(dt / self._ctrl_dt)))):
            self._control_step()

    # ------------------------------------------------------------ physics

    def _control_step(self) -> None:
        with self._lock:
            self._setpoint = self.filter.step(self._target, self._ctrl_dt)
            self.data.ctrl[:] = self._setpoint
            if self._gravity_kin is not None:
                g = self._gravity_kin.gravity_torques({s: self._setpoint[arm_slice(s)] for s in SIDES})
                for side in SIDES:
                    self.data.qfrc_applied[self._vadr[side]] = self.gravity_scale * g[side]
            for _ in range(self._substeps):
                mujoco.mj_step(self.model, self.data)

    def _loop(self) -> None:
        next_tick = time.monotonic()
        last_sync = 0.0
        while not self._stop_evt.is_set():
            self._control_step()
            now = time.monotonic()
            if self._viewer is not None and now - last_sync > 1.0 / 60.0:
                with self._lock:
                    self._viewer.sync()
                last_sync = now
            next_tick += self._ctrl_dt
            remaining = next_tick - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            else:
                next_tick = time.monotonic()

    def _measured_vector(self) -> NDArray[np.float64]:
        return pack_command(
            {s: self.data.qpos[self._qadr[s]] for s in SIDES},
            {s: float(self.data.qpos[self._gadr[s]]) for s in SIDES},
        )
