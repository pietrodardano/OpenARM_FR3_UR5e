"""The official OpenArm v1 model plus the measured workcell, built at runtime in the base frame.

:func:`build_scene` loads ``openarm_mujoco/v1/openarm_bimanual.xml`` unmodified (meshes keep
resolving) and adds, from the config: a ``<side>_tcp`` site between the fingertips, the 60 mm
torso profile (replacing the model's torso mesh, as validated on ``main``), the table, the two
bins, mocap objects for the twin and a handover marker. Everything is in the robot base frame
(origin at ``openarm_body_link0``), so there is no world/base offset anywhere in the code.

:class:`ArmModel` is the single source of joint limits, forward kinematics, the vertical TCP
Jacobian (payload), gravity torques (feedforward, payload) and the workcell clearance check (SR4).
The arm that is not driven stays at its rest pose (all zeros) in the model.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import mujoco
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from ..config import Config, resolve

Pose = Tuple[NDArray, NDArray]  # position (3,), rotation matrix (3, 3)

OBJECT_RADIUS_M = 0.033  # twin-only stand-in for a soda can
OBJECT_HALF_HEIGHT_M = 0.06
BIN_COLORS = {"left": (0.85, 0.25, 0.20, 1.0), "right": (0.20, 0.45, 0.85, 1.0)}
# Bodies kept table_margin_m above the tabletop (SR4 table guard).
GUARD_BODIES = ("link3", "link4", "link5", "link6", "link7", "hand", "hand_tcp", "left_finger", "right_finger")


def _box(body, name: str, pos, half, rgba, collide: bool = True) -> None:
    body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=list(pos), size=list(half), rgba=list(rgba),
                  contype=int(collide), conaffinity=int(collide))


def build_scene(cfg: Config, n_objects: int = 2) -> mujoco.MjSpec:
    """Official bimanual model + workcell, as an editable spec (robot base frame)."""
    path = resolve(cfg.robot.model_path)
    if not path.exists():
        raise FileNotFoundError(f"official OpenArm model not found at {path}: run ./setup.sh")
    spec = mujoco.MjSpec.from_file(str(path))
    for side in ("left", "right"):
        spec.body(f"openarm_{side}_hand_tcp").add_site(name=f"{side}_tcp", size=[0.01, 0, 0],
                                                         rgba=[1, 0, 0, 1], group=4)
    for g in list(spec.body("openarm_body_link0").geoms):  # replaced by the measured torso profile
        spec.delete(g)

    w, wc = spec.worldbody, cfg.workcell
    floor_z = -cfg.base_z_in_world
    table_z = cfg.table_z
    t = wc.table
    w.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, pos=[0, 0, floor_z], size=[3, 3, 0.1],
               rgba=[0.32, 0.34, 0.38, 1], contype=0, conaffinity=0)
    _box(w, "torso", (0, 0, floor_z + wc.torso.height_m / 2),
         (wc.torso.width_m / 2, wc.torso.width_m / 2, wc.torso.height_m / 2), (0.65, 0.68, 0.72, 1))
    _box(w, "table", (t.center_x_m, 0, table_z - t.thickness_m / 2),
         (t.length_m / 2, t.width_m / 2, t.thickness_m / 2), (0.46, 0.30, 0.18, 1))
    for name, b in wc.bins.items():
        body = w.add_body(name=f"bin_{name}", pos=[b.x, b.y, table_z])
        half, th, wh, rgba = b.inner_size_m / 2, 0.006, b.wall_height_m, BIN_COLORS[name]
        _box(body, f"bin_{name}_floor", (0, 0, 0.0015), (half + th, half + th, 0.0015), rgba)
        for i, (dx, dy, sx, sy) in enumerate([(half + th / 2, 0, th / 2, half + th), (-half - th / 2, 0, th / 2, half + th),
                                              (0, half + th / 2, half, th / 2), (0, -half - th / 2, half, th / 2)]):
            _box(body, f"bin_{name}_wall{i}", (dx, dy, wh / 2), (sx, sy, wh / 2), rgba)
    for i in range(n_objects):
        body = w.add_body(name=f"object_{i + 1}", mocap=True, pos=[0.0, 0.0, floor_z])
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[OBJECT_RADIUS_M, OBJECT_HALF_HEIGHT_M, 0],
                      rgba=[0.78, 0.80, 0.84, 1], contype=0, conaffinity=0)
    h = cfg.task.handover
    w.add_site(name="handover", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.03, 0, 0],
               pos=[h.x, h.y, table_z + h.above_table_m], rgba=[0.6, 0.3, 0.9, 0.35])
    w.add_light(pos=[-1.5, -1.5, 2.5], dir=[0.5, 0.5, -1.0], diffuse=[0.8, 0.8, 0.8])
    return spec


class ArmModel:
    """Kinematics, gravity and clearance of the configured arm in the workcell model.

    Not thread-safe: the driver's servo thread gets its own instance for gravity.
    """

    def __init__(self, cfg: Config, n_objects: int = 2) -> None:
        self.cfg = cfg
        self.side = cfg.robot.side
        self.spec = build_scene(cfg, n_objects)
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        m = self.model
        jids = [m.joint(f"openarm_{self.side}_joint{i}").id for i in range(1, 8)]
        self.qadr = m.jnt_qposadr[jids].copy()
        self.dofadr = m.jnt_dofadr[jids].copy()
        self.finger_qadr = [int(m.jnt_qposadr[m.joint(f"openarm_{self.side}_finger_joint{k}").id]) for k in (1, 2)]
        self.lower = m.jnt_range[jids, 0].copy()
        self.upper = m.jnt_range[jids, 1].copy()
        self.site_id = m.site(f"{self.side}_tcp").id
        self.home = np.asarray(cfg.robot.home_qpos, dtype=np.float64)
        self.R_forward = self.fk(self.home)[1]
        self._guard = [m.body(f"openarm_{self.side}_{b}").id for b in GUARD_BODIES]
        self._arm_geoms, self._obstacles = self._collision_sets()
        for g in self._arm_geoms:  # contacts within the clearance get reported
            m.geom_margin[g] = max(m.geom_margin[g], cfg.motion.collision_clearance_m)

    def _collision_sets(self) -> Tuple[set, set]:
        m, other = self.model, "right" if self.side == "left" else "left"
        arm, obstacles = set(), set()
        for g in range(m.ngeom):
            if not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue  # visual only (meshes, objects)
            body = m.geom_bodyid[g]
            name = m.body(body).name
            if name.startswith(f"openarm_{self.side}_"):
                arm.add(g)
            elif name.startswith(f"openarm_{other}_") or self._is_static(body):
                obstacles.add(g)
        return arm, obstacles

    def _is_static(self, body: int) -> bool:
        m = self.model
        while body != 0:
            if m.body_dofnum[body] > 0 or m.body_mocapid[body] >= 0:
                return False
            body = m.body_parentid[body]
        return True

    # ---------------------------------------------------------------- kinematics
    def _set(self, q: NDArray, gripper_m: Optional[float] = None) -> None:
        self.data.qpos[self.qadr] = q
        if gripper_m is not None:
            self.data.qpos[self.finger_qadr] = gripper_m

    def fk(self, q: NDArray) -> Pose:
        """TCP pose (base frame) at joints ``q``."""
        self._set(q)
        mujoco.mj_kinematics(self.model, self.data)
        return self.data.site_xpos[self.site_id].copy(), self.data.site_xmat[self.site_id].reshape(3, 3).copy()

    def jz(self, q: NDArray) -> NDArray:
        """Vertical row of the TCP position Jacobian (7,), m/rad."""
        self._set(q)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        jp = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jp, None, self.site_id)
        return jp[2, self.dofadr].copy()

    def gravity(self, q: NDArray) -> NDArray:
        """Joint torques (Nm) that hold the arm (no payload) at ``q``; + follows the joint axis."""
        self._set(q)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self.data.qfrc_bias[self.dofadr].copy()

    def hand_rotation(self, yaw: float, pitch: float) -> NDArray:
        """Hand-forward orientation (home pose) pitched down by ``pitch`` about Y, then turned by ``yaw`` about Z.
        The closing axis stays horizontal: a lateral squeeze of an upright object."""
        return Rotation.from_euler("z", yaw).as_matrix() @ Rotation.from_euler("y", pitch).as_matrix() @ self.R_forward

    # ----------------------------------------------------------------- clearance
    def blocked(self, q: NDArray, gripper_m: Optional[float] = None) -> Optional[str]:
        """None if the arm at ``q`` is clear of the workcell, else what it hits (SR4).

        Two checks: the guarded bodies stay ``table_margin_m`` above the table footprint, and no
        arm collision geom comes within ``collision_clearance_m`` of the torso, table, bins or
        the other arm.
        """
        mc, t = self.cfg.motion, self.cfg.workcell.table
        self._set(q, self.cfg.gripper.open_m if gripper_m is None else gripper_m)
        mujoco.mj_fwdPosition(self.model, self.data)
        x0, x1 = t.center_x_m - t.length_m / 2 - mc.table_margin_m, t.center_x_m + t.length_m / 2 + mc.table_margin_m
        for bid in self._guard:
            x, y, z = self.data.xpos[bid]
            if x0 <= x <= x1 and abs(y) <= t.width_m / 2 + mc.table_margin_m and z < self.cfg.table_z + mc.table_margin_m:
                return f"{self.model.body(bid).name} at {np.round(self.data.xpos[bid], 3).tolist()} (table guard)"
        hits: List[str] = []
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            arm, obs = (g1, g2) if g1 in self._arm_geoms else (g2, g1)
            if arm in self._arm_geoms and obs in self._obstacles and c.dist < mc.collision_clearance_m:
                hits.append(f"{self.model.body(self.model.geom_bodyid[arm]).name} - "
                            f"{self.model.geom(obs).name or self.model.body(self.model.geom_bodyid[obs]).name} "
                            f"({c.dist * 1e3:.0f} mm)")
        return ", ".join(hits[:3]) if hits else None

    def object_names(self) -> Sequence[str]:
        return [self.model.body(b).name for b in range(self.model.nbody) if self.model.body_mocapid[b] >= 0]
