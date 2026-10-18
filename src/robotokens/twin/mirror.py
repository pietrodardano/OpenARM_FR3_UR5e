"""Read-only MuJoCo twin that follows the robot (FR7). The robot is the leader.

At ``twin.rate_hz`` the twin reads ``arm.state()`` (a snapshot the servo thread already
publishes), writes the measured joints and finger opening into its own copy of the workcell
model and runs kinematics only: no physics, no contacts, no actuators. It never calls anything
else on the arm, so if the viewer crashes or is closed the robot does not notice.

Objects (mocap bodies) stand at the believed positions. While the gripper squeezes in torque
mode, the object nearest the TCP is drawn attached to the hand at the offset it had when the
squeeze started; it stays where it is released. Overlays: commanded TCP (green), measured TCP
(yellow), latched fault (red).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, List, Optional, Sequence, Tuple

import mujoco
import numpy as np
from numpy.typing import NDArray

from ..config import Config
from ..robot.model import OBJECT_HALF_HEIGHT_M, build_scene

logger = logging.getLogger(__name__)

ATTACH_RADIUS_M = 0.07
GREEN, YELLOW, RED = (0.1, 0.9, 0.2, 0.9), (1.0, 0.85, 0.1, 0.9), (1.0, 0.1, 0.1, 0.9)


class Twin:
    def __init__(self, cfg: Config, arm: Any, objects: Sequence[Tuple[float, float]] = ()) -> None:
        self.cfg = cfg
        self.arm = arm
        self.model = build_scene(cfg, n_objects=2).compile()
        self.data = mujoco.MjData(self.model)
        self._cmd = mujoco.MjData(self.model)  # FK of the commanded setpoints
        m, side = self.model, cfg.robot.side
        self._qadr = m.jnt_qposadr[[m.joint(f"openarm_{side}_joint{i}").id for i in range(1, 8)]]
        self._gadr = [int(m.jnt_qposadr[m.joint(f"openarm_{side}_finger_joint{k}").id]) for k in (1, 2)]
        self._site = m.site(f"{side}_tcp").id
        self._mocap = [int(m.body_mocapid[m.body(f"object_{i + 1}").id]) for i in range(2)]
        self._held: Optional[Tuple[int, NDArray, NDArray]] = None  # (object, offset, rotation in TCP frame)
        self.set_objects(objects)
        self._viewer: Any = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.last_state: Any = None

    def set_objects(self, objects: Sequence[Tuple[float, float]]) -> None:
        """Places the objects upright on the table at the believed (x, y); unused ones go under the floor."""
        for i, mid in enumerate(self._mocap):
            if i < len(objects):
                x, y = objects[i]
                self.data.mocap_pos[mid] = [x, y, self.cfg.table_z + OBJECT_HALF_HEIGHT_M]
            else:
                self.data.mocap_pos[mid] = [0.0, 0.0, -10.0]
            self.data.mocap_quat[mid] = [1, 0, 0, 0]

    # ---------------------------------------------------------------- mirroring
    def update(self) -> bool:
        """Poses the twin from the measured state once. False before the driver has started."""
        try:
            st = self.arm.state()
        except RuntimeError:
            return False
        self.data.qpos[self._qadr] = st.q
        self.data.qpos[self._gadr] = st.gripper
        self._cmd.qpos[self._qadr] = st.q_setpoint
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_kinematics(self.model, self._cmd)
        self._update_held(st)
        self.last_state = st
        return True

    def _update_held(self, st: Any) -> None:
        p_tcp = self.data.site_xpos[self._site]
        R_tcp = self.data.site_xmat[self._site].reshape(3, 3)
        squeezing = st.gripper_torque_cmd is not None and st.gripper > 0.005
        if not squeezing:
            self._held = None
        elif self._held is None:
            dist = [np.linalg.norm(self.data.mocap_pos[mid] - p_tcp) for mid in self._mocap]
            i = int(np.argmin(dist))
            if dist[i] < ATTACH_RADIUS_M:
                mid = self._mocap[i]
                R_obj = np.zeros(9)
                mujoco.mju_quat2Mat(R_obj, self.data.mocap_quat[mid])
                self._held = (mid, R_tcp.T @ (self.data.mocap_pos[mid] - p_tcp), R_tcp.T @ R_obj.reshape(3, 3))
        if self._held is not None:
            mid, rel_p, rel_R = self._held
            self.data.mocap_pos[mid] = p_tcp + R_tcp @ rel_p
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, (R_tcp @ rel_R).flatten())
            self.data.mocap_quat[mid] = quat

    def tcp(self, commanded: bool = False) -> NDArray:
        """Measured (default) or commanded TCP position, base frame."""
        return (self._cmd if commanded else self.data).site_xpos[self._site].copy()

    def object_position(self, i: int) -> NDArray:
        return self.data.mocap_pos[self._mocap[i]].copy()

    # ------------------------------------------------------------------ viewer
    def _draw(self) -> None:
        markers: List[Tuple[float, NDArray, tuple]] = [(0.012, self.tcp(True), GREEN), (0.009, self.tcp(), YELLOW)]
        if self.last_state is not None and self.last_state.fault:
            markers.append((0.04, np.array([0.0, 0.0, 1.0]), RED))
        scn = self._viewer.user_scn
        for i, (r, pos, rgba) in enumerate(markers[: scn.maxgeom]):
            mujoco.mjv_initGeom(scn.geoms[i], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, 0, 0]), pos,
                                np.eye(3).flatten(), np.asarray(rgba, np.float32))
        scn.ngeom = len(markers)

    def start(self) -> None:
        """Opens the passive viewer and follows the arm in a background thread."""
        import mujoco.viewer

        self._viewer = mujoco.viewer.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False)
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="twin", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        period = 1.0 / self.cfg.twin.rate_hz
        while not self._stop.is_set() and self._viewer.is_running():
            t0 = time.monotonic()
            try:
                with self._viewer.lock():
                    if self.update():
                        self._draw()
                self._viewer.sync()
            except Exception as exc:  # noqa: BLE001 - the twin must never disturb the robot
                logger.warning("twin frame skipped: %s", exc)
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
