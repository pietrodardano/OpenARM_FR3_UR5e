"""
Read-only MuJoCo digital twin that follows a running robot. The robot is the leader.

The twin shows the demo scene (``demo.sim.scene_path``: measured workcell, 72.5 cm table,
bins, cans), posed at ``mirror.rate_hz`` from the *measured* state of any ``RobotBackend``
(normally the real ``OpenArmBimanual``). Every frame:

1. ``robot.get_state()`` gives the measured arm joints and finger openings (a locked
   snapshot the driver thread already publishes).
2. They are written into the twin's ``qpos`` (``finger_joint2 = finger_joint1``).
3. Cans: they start at the object positions (vision / ``demo.manual_objects``). While the
   gripper squeezes in torque mode, the can closest to the TCP is drawn attached to the hand,
   at the offset it had when the squeeze started. It stays where it was released.
4. ``mj_kinematics`` runs, with no physics, no contacts and no actuators.

The twin never sends a command and never calls a method that changes the robot. If it crashes
or its window is closed, the robot does not notice.

Overlays: commanded TCP (green) vs measured TCP (yellow), and a red marker while the driver has
a latched fault. On macOS the passive viewer needs the ``mjpython`` launcher.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import mujoco
import numpy as np
from numpy.typing import NDArray

from ..control.robot_interface import ARM_DOF, SIDES, RobotBackend, RobotState, Side

logger = logging.getLogger(__name__)

GREEN = (0.1, 0.9, 0.2, 0.9)
YELLOW = (1.0, 0.85, 0.1, 0.9)
RED = (1.0, 0.1, 0.1, 0.9)
ATTACH_RADIUS_M = 0.07


@dataclass
class Marker:
    """One overlay geom in the twin's world frame."""

    type: int  # mujoco.mjtGeom
    size: Sequence[float]
    pos: Sequence[float]
    rgba: Sequence[float]


@dataclass
class MirrorConfig:
    """``mirror`` section of task_config.yaml."""

    scene_path: str
    rate_hz: float = 30.0
    show_targets: bool = True

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "MirrorConfig":
        """Builds the config; the scene defaults to the demo scene (``demo.sim.scene_path``)."""
        m = cfg.get("mirror", {})
        return cls(
            scene_path=str(m.get("scene_path") or cfg["demo"]["sim"]["scene_path"]),
            rate_hz=float(m.get("rate_hz", 30.0)),
            show_targets=bool(m.get("show_targets", True)),
        )


class RealRobotMirror:
    """Kinematic twin of ``robot`` in a MuJoCo passive viewer (or headless, for tests)."""

    def __init__(self, robot: RobotBackend, config: MirrorConfig,
                 objects: Optional[Sequence[Mapping[str, Any]]] = None) -> None:
        """Args:
            robot: Leader backend. Only ``get_state()`` is ever called on it.
            config: Scene, rate and overlays.
            objects: Object positions (base frame, ``x``/``y``) to place the scene's cans at.
        """
        self.robot = robot
        self.config = config
        self.model = mujoco.MjModel.from_xml_path(config.scene_path)
        self.data = mujoco.MjData(self.model)
        self._target_data = mujoco.MjData(self.model)  # FK of the commanded setpoints
        m = self.model
        self._qadr: Dict[Side, NDArray[np.int64]] = {}
        self._gadr: Dict[Side, List[int]] = {}
        for side in SIDES:
            jids = [m.joint(f"openarm_{side}_joint{i}").id for i in range(1, ARM_DOF + 1)]
            self._qadr[side] = m.jnt_qposadr[jids]
            self._gadr[side] = [int(m.jnt_qposadr[m.joint(f"openarm_{side}_finger_joint{k}").id]) for k in (1, 2)]
        # TCP = openarm_<side>_hand_tcp body (between the fingertips), as in OpenArmKinematics.
        self._tcp_body = {s: m.body(f"openarm_{s}_hand_tcp").id for s in SIDES}
        self.base_z = float(m.body("openarm_body_link0").pos[2])  # base frame height in the scene
        self._cans: Dict[str, int] = {}
        i = 1
        while mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"can_{i}") >= 0:
            bid = m.body(f"can_{i}").id
            self._cans[f"can_{i}"] = int(m.jnt_qposadr[m.body_jntadr[bid]])
            i += 1
        for name, obj in zip(self._cans, objects or []):
            q = self._cans[name]
            self.data.qpos[q:q + 2] = [float(obj["x"]), float(obj["y"])]
        self._attached: Dict[Side, Optional[tuple]] = {s: None for s in SIDES}
        self._viewer: Any = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.last_state: Optional[RobotState] = None
        self.frames = 0

    # ------------------------------------------------------------ mirroring
    def _tcp_pose(self, side: Side) -> tuple:
        b = self._tcp_body[side]
        return self.data.xpos[b].copy(), self.data.xmat[b].reshape(3, 3).copy()

    def _update_held(self, st: RobotState) -> None:
        """Attaches the nearest can while a gripper squeezes in torque mode; detaches on release."""
        for side in SIDES:
            arm = st.arms[side]
            squeezing = arm.active and arm.gripper_torque_cmd is not None and arm.gripper > 0.005
            p_tcp, R_tcp = self._tcp_pose(side)
            if squeezing and self._attached[side] is None and self._cans:
                dist = {n: float(np.linalg.norm(self.data.qpos[q:q + 3] - p_tcp)) for n, q in self._cans.items()}
                name = min(dist, key=dist.get)
                if dist[name] < ATTACH_RADIUS_M:
                    q = self._cans[name]
                    R_can = np.zeros(9)
                    mujoco.mju_quat2Mat(R_can, self.data.qpos[q + 3:q + 7])
                    self._attached[side] = (name, R_tcp.T @ (self.data.qpos[q:q + 3] - p_tcp),
                                            R_tcp.T @ R_can.reshape(3, 3))
            elif not squeezing:
                self._attached[side] = None
            if self._attached[side] is not None:
                name, rel_p, rel_R = self._attached[side]
                q = self._cans[name]
                self.data.qpos[q:q + 3] = p_tcp + R_tcp @ rel_p
                quat = np.zeros(4)
                mujoco.mju_mat2Quat(quat, (R_tcp @ rel_R).flatten())
                self.data.qpos[q + 3:q + 7] = quat

    def update(self) -> bool:
        """Poses the twin from the robot's measured state once. False if no state yet."""
        try:
            st = self.robot.get_state()
        except RuntimeError:  # driver not started yet
            return False
        for side in SIDES:
            arm = st.arms[side]
            self.data.qpos[self._qadr[side]] = arm.q
            self.data.qpos[self._gadr[side]] = arm.gripper
            self._target_data.qpos[self._qadr[side]] = arm.q_setpoint
        mujoco.mj_kinematics(self.model, self.data)
        self._update_held(st)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_kinematics(self.model, self._target_data)
        self.last_state = st
        self.frames += 1
        return True

    def tcp(self, side: Side, commanded: bool = False) -> NDArray[np.float64]:
        """Measured (default) or commanded TCP position of ``side``, robot base frame, m."""
        d = self._target_data if commanded else self.data
        return d.xpos[self._tcp_body[side]].copy() - np.array([0.0, 0.0, self.base_z])

    def can_position(self, name: str) -> NDArray[np.float64]:
        """World position of a can in the twin."""
        q = self._cans[name]
        return self.data.qpos[q:q + 3].copy()

    def overlay(self) -> List[Marker]:
        """Overlay geoms for the current frame (world frame)."""
        markers: List[Marker] = []
        st = self.last_state
        if st is None:
            return markers
        off = np.array([0.0, 0.0, self.base_z])
        if self.config.show_targets:
            for side in SIDES:
                if st.arms[side].active:
                    markers.append(Marker(mujoco.mjtGeom.mjGEOM_SPHERE, (0.012,), self.tcp(side, True) + off, GREEN))
                    markers.append(Marker(mujoco.mjtGeom.mjGEOM_SPHERE, (0.009,), self.tcp(side) + off, YELLOW))
        if st.fault:
            markers.append(Marker(mujoco.mjtGeom.mjGEOM_SPHERE, (0.04,), (0.0, 0.0, self.base_z + 1.0), RED))
        return markers

    def _draw(self, markers: List[Marker]) -> None:
        scn = self._viewer.user_scn
        n = min(len(markers), scn.maxgeom)
        for i, mk in enumerate(markers[:n]):
            size = np.zeros(3)
            size[: len(mk.size)] = mk.size
            mujoco.mjv_initGeom(scn.geoms[i], int(mk.type), size, np.asarray(mk.pos, dtype=np.float64),
                                np.eye(3).flatten(), np.asarray(mk.rgba, dtype=np.float32))
        scn.ngeom = n

    # ------------------------------------------------------------- viewer
    def start(self) -> None:
        """Opens the passive viewer and starts following the robot in a background thread."""
        import mujoco.viewer

        self._viewer = mujoco.viewer.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False)
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="real_robot_mirror", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        period = 1.0 / self.config.rate_hz
        while not self._stop.is_set() and self._viewer.is_running():
            t0 = time.monotonic()
            try:
                with self._viewer.lock():
                    if self.update():
                        self._draw(self.overlay())
                self._viewer.sync()
            except Exception as exc:  # noqa: BLE001 - the twin must never disturb the run
                logger.warning("mirror frame skipped: %s", exc)
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    def stop(self) -> None:
        """Stops following and closes the viewer. Never touches the robot."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
