"""Kinematic MuJoCo wiping scene on the OpenArm workcell (sim only, never touches hardware).

The project's own ArmModel + mink IK drive the left arm. A wiping board sits on a riser on the
table (the bare tabletop is at the edge of the arm's reach), with a grid of dirt cells on it. A
sponge follows the TCP; a cell loses dirt each step the sponge passes over it. Dirt clearing is
a stand-in: proximity, not friction physics.
"""

import dataclasses as dc

import mujoco
import numpy as np

from robotokens.config import load_config
from robotokens.robot.ik import IK
from robotokens.robot.model import ArmModel

TOOL_Z = 0.20  # TCP height above the tabletop
BOARD_TOP = 0.155  # board surface above the tabletop (sponge 4.5 cm)
PITCH = np.radians(60)
BOARD_X, BOARD_Y = (0.22, 0.40), (0.03, 0.35)
CELL = 0.02
SPONGE_R = 0.03
WIPE_RATE = 0.35  # dirt removed per step under the sponge
RATE_HZ = 12.0


class WipeSim:
    def __init__(self, width: int = 480, height: int = 360) -> None:
        cfg = load_config()
        self.cfg = cfg = dc.replace(cfg, workcell=dc.replace(cfg.workcell, bins={}))
        self.arm = ArmModel(cfg, n_objects=0)
        self.ik = IK(self.arm)
        self.R = self.arm.hand_rotation(0.0, PITCH)
        tz = cfg.table_z
        self.z_tool, self.z_board = tz + TOOL_Z, tz + BOARD_TOP
        xs = np.arange(BOARD_X[0] + CELL / 2, BOARD_X[1], CELL)
        ys = np.arange(BOARD_Y[0] + CELL / 2, BOARD_Y[1], CELL)
        self.cells = np.array([(x, y) for x in xs for y in ys])
        self.grid_shape = (len(xs), len(ys))

        spec = self.arm.spec.copy() if hasattr(self.arm.spec, "copy") else self.arm.spec
        w = spec.worldbody
        cx, cy = np.mean(BOARD_X), np.mean(BOARD_Y)
        hx, hy = (BOARD_X[1] - BOARD_X[0]) / 2 + 0.02, (BOARD_Y[1] - BOARD_Y[0]) / 2 + 0.02
        riser = (self.z_board - 0.012 + tz) / 2
        w.add_geom(name="riser", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[cx, cy, riser],
                   size=[hx - 0.03, hy - 0.03, (self.z_board - 0.012 - tz) / 2], rgba=[0.35, 0.35, 0.38, 1],
                   contype=0, conaffinity=0)
        w.add_geom(name="board", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[cx, cy, self.z_board - 0.006],
                   size=[hx, hy, 0.006], rgba=[0.92, 0.92, 0.90, 1], contype=0, conaffinity=0)
        for i, (x, y) in enumerate(self.cells):
            w.add_geom(name=f"dirt{i}", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[x, y, self.z_board + 0.0008],
                       size=[CELL / 2 * 0.96, CELL / 2 * 0.96, 0.0008], rgba=[0.45, 0.30, 0.12, 0],
                       contype=0, conaffinity=0)
        sp = w.add_body(name="sponge", mocap=True, pos=[cx, cy, self.z_board + 0.012])
        sp.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[SPONGE_R, 0.012, 0], rgba=[0.95, 0.80, 0.15, 1],
                    contype=0, conaffinity=0)
        rod = w.add_body(name="rod", mocap=True, pos=[cx, cy, self.z_board + 0.03])
        rod.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.006, 0.02, 0], rgba=[0.2, 0.2, 0.2, 1],
                     contype=0, conaffinity=0)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.dirt_gid = np.array([m.geom(f"dirt{i}").id for i in range(len(self.cells))])
        self.sponge_mid = int(m.body_mocapid[m.body("sponge").id])
        self.rod_mid = int(m.body_mocapid[m.body("rod").id])
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = [cx - 0.02, cy - 0.02, self.z_board + 0.12]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 1.25, 205.0, -42.0
        self.renderer = mujoco.Renderer(m, height, width)
        self.width, self.height = width, height

    def close(self) -> None:
        self.renderer.close()

    # --------------------------------------------------------------- episode
    def reset(self, dirt: np.ndarray, start_xy=(0.31, 0.19)) -> np.ndarray:
        self.dirt = dirt.astype(np.float64).copy()
        self.dirt0 = self.dirt.sum()
        self.q = self.arm.home.copy()
        self.q = self.ik.solve(np.array([*start_xy, self.z_tool]), self.R, self.q, 300).q
        self.max_qd = np.zeros(7)
        self.ik_fail = 0
        self.xy = self._place(np.asarray(start_xy, float))
        return self.xy

    def clip(self, xy: np.ndarray) -> np.ndarray:
        return np.clip(xy, [BOARD_X[0], BOARD_Y[0]], [BOARD_X[1], BOARD_Y[1]])

    def _place(self, xy: np.ndarray) -> np.ndarray:
        r = self.ik.solve(np.array([*xy, self.z_tool]), self.R, self.q, 40)
        if r.pos_error > 0.005:
            self.ik_fail += 1
        self.max_qd = np.maximum(self.max_qd, np.abs(r.q - self.q) * RATE_HZ)
        self.q = r.q
        self.data.qpos[self.arm.qadr] = self.q
        self.data.qpos[self.arm.finger_qadr] = 0.012
        mujoco.mj_kinematics(self.model, self.data)
        tcp = self.data.site_xpos[self.arm.site_id].copy()
        self.data.mocap_pos[self.sponge_mid] = [tcp[0], tcp[1], self.z_board + 0.012]
        self.data.mocap_pos[self.rod_mid] = [tcp[0], tcp[1], self.z_board + 0.035]
        near = np.linalg.norm(self.cells - tcp[:2], axis=1) < SPONGE_R
        if abs(tcp[2] - self.z_tool) < 0.005:
            self.dirt[near] = np.maximum(0.0, self.dirt[near] - WIPE_RATE)
        return tcp[:2]

    def step(self, target_xy: np.ndarray) -> np.ndarray:
        self.xy = self._place(self.clip(target_xy))
        return self.xy

    def cleaned(self) -> float:
        return 1.0 - self.dirt.sum() / self.dirt0

    def render(self) -> np.ndarray:
        self.model.geom_rgba[self.dirt_gid, 3] = np.clip(self.dirt, 0, 1) * 0.95
        self.renderer.update_scene(self.data, camera=self.cam)
        return self.renderer.render()


def dirt_pattern(sim: WipeSim, seed: int) -> np.ndarray:
    """2-3 random spills (soft blobs) on the board."""
    rng = np.random.default_rng(seed)
    d = np.zeros(len(sim.cells))
    for _ in range(rng.integers(2, 4)):
        c = np.array([rng.uniform(*BOARD_X), rng.uniform(*BOARD_Y)])
        r = rng.uniform(0.035, 0.07)
        d += np.exp(-np.sum((sim.cells - c) ** 2, 1) / (2 * r * r)) * 1.5
    return np.where(d > 0.3, np.minimum(d, 1.0), 0.0)
