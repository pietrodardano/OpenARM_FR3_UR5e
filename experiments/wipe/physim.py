"""Physical wiping: real grasp of a sponge, force-controlled contact, dirt from contact (sim only).

Full MuJoCo dynamics on the official OpenArm model. The left arm's position servos are switched
off and the arm is torque-controlled (like MIT mode with kp=0 + feedforward):

    tau = J^T F_task + N (posture PD towards the mink IK solution) + gravity/Coriolis

F_task: a Cartesian spring-damper in x/y (follows the wipe path), an orientation spring, and in
z either a spring (free motion) or a force loop regulating the measured sponge-board normal force.
The gripper is the model's own finger actuator closing on the sponge handle: friction holds it.

The hand works pitched 45 deg (pointing it straight down puts joints 1/5/7 on their limits at this
height). The sponge has a vertical handle that the parallel jaws grip diagonally, so the pad sits
straight below the grasp: pressing down does not twist the handle between the fingers.

Dirt removal follows Archard's wear law: removal ~ normal force x sliding distance, applied to
the cells under the pad footprint, only while the pad is in contact with the board.
"""

import dataclasses as dc

import mujoco
import numpy as np

from robotokens.config import load_config
from robotokens.robot.ik import IK
from robotokens.robot.model import ArmModel

PITCH = np.radians(45)
TOOL_Z = 0.20  # nominal TCP height above the tabletop while wiping
PAD = np.array([0.03, 0.02, 0.0125])  # pad half sizes (6 x 4 x 2.5 cm)
HANDLE = np.array([0.015, 0.010, 0.035])  # vertical handle half sizes: 3 cm, 2 cm across the jaws, 7 cm tall
GRIP_ABOVE_PAD = 0.035  # TCP this far above the pad top (middle of the handle)
CELL = 0.015
F_DES = 10.0  # N, pad pressed onto the board
WEAR_PER_NM = 1.0 / (F_DES * 0.12)  # full dirt cell removed by ~2 passes of 6 cm at F_DES
DT = 0.001
TAU_MAX = np.array([40, 40, 27, 27, 7, 7, 7], float)  # DM8009 x2, DM4340 x2, DM4310 x3 (Nm)
TCP_PATCH = (0.30, 0.22, 0.075)  # centre x, y and half size of the comfortable TCP patch (searched)


class PhysWipe:
    def __init__(self, width: int = 640, height: int = 480) -> None:
        cfg = load_config()
        self.cfg = cfg = dc.replace(cfg, workcell=dc.replace(cfg.workcell, bins={}))
        self.arm = ArmModel(cfg, n_objects=0)  # kinematics + mink IK (separate model)
        self.ik = IK(self.arm)
        self.R_tool = self.arm.hand_rotation(0.0, PITCH)
        tz = cfg.table_z

        self.u = -self.R_tool[:, 2]  # back along the approach axis (pre-grasp direction)
        self.pad_to_tcp = np.array([0, 0, PAD[2] + GRIP_ABOVE_PAD])  # pad centre -> TCP
        self.z_board = tz + TOOL_Z - (PAD[2] + self.pad_to_tcp[2])
        self.pad_offset_xy = -self.pad_to_tcp[:2]  # TCP -> pad centre, horizontal
        cx, cy, hs = TCP_PATCH
        px, py = cx + self.pad_offset_xy[0], cy + self.pad_offset_xy[1]
        self.dirt_x, self.dirt_y = (px - hs, px + hs), (py - hs, py + hs)
        xs = np.arange(self.dirt_x[0] + CELL / 2, self.dirt_x[1], CELL)
        ys = np.arange(self.dirt_y[0] + CELL / 2, self.dirt_y[1], CELL)
        self.cells = np.array([(x, y) for x in xs for y in ys])
        self.grid_shape = (len(xs), len(ys))
        self.sponge_home = np.array([self.dirt_x[0] + 0.035, self.dirt_y[1] + 0.035])  # just off the dirt

        spec = self.arm.spec
        w = spec.worldbody
        bx0, bx1 = self.dirt_x[0] - 0.03, self.dirt_x[1] + 0.03
        by0, by1 = self.dirt_y[0] - 0.03, self.dirt_y[1] + 0.07
        bcx, bcy, bhx, bhy = (bx0 + bx1) / 2, (by0 + by1) / 2, (bx1 - bx0) / 2, (by1 - by0) / 2
        rz = (self.z_board - 0.012 - tz) / 2
        w.add_geom(name="riser", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[bcx, bcy, tz + rz],
                   size=[bhx - 0.03, bhy - 0.03, rz], rgba=[0.35, 0.35, 0.38, 1])
        w.add_geom(name="board", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[bcx, bcy, self.z_board - 0.006],
                   size=[bhx, bhy, 0.006], rgba=[0.92, 0.92, 0.90, 1], friction=[0.5, 0.005, 0.0001])
        for i, (x, y) in enumerate(self.cells):
            w.add_geom(name=f"dirt{i}", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[x, y, self.z_board + 0.0006],
                       size=[CELL / 2 * 0.96, CELL / 2 * 0.96, 0.0006], rgba=[0.45, 0.30, 0.12, 0],
                       contype=0, conaffinity=0)
        sp = w.add_body(name="sponge", pos=[*self.sponge_home, self.z_board + PAD[2] + 0.001])
        sp.add_freejoint(name="sponge_free")
        sp.add_geom(name="sponge_pad", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(PAD), mass=0.04,
                    rgba=[0.95, 0.80, 0.15, 1], friction=[0.5, 0.005, 0.0001], condim=4)
        sp.add_geom(name="sponge_handle", type=mujoco.mjtGeom.mjGEOM_BOX,
                    pos=[0, 0, PAD[2] + HANDLE[2]], size=list(HANDLE),
                    mass=0.03, rgba=[0.15, 0.45, 0.85, 1], friction=[1.5, 0.02, 0.001], condim=6)
        self.model = m = spec.compile()
        m.opt.timestep = DT
        self.data = mujoco.MjData(m)
        side = self.arm.side
        self.qadr, self.dofadr = self.arm.qadr.copy(), self.arm.dofadr.copy()
        self.site = m.site(f"{side}_tcp").id
        self.pad = m.geom("sponge_pad").id
        self.board = m.geom("board").id
        self.sponge_body = m.body("sponge").id
        self.dirt_gid = np.array([m.geom(f"dirt{i}").id for i in range(len(self.cells))])
        self.finger_act = m.actuator(f"{side}_finger1_ctrl").id
        for i in range(1, 8):  # torque control: position servos of the driven arm off
            a = m.actuator(f"{side}_joint{i}_ctrl").id
            m.actuator_gainprm[a, 0] = 0.0
            m.actuator_biasprm[a, :3] = 0.0
        m.site_rgba[m.site("handover").id, 3] = 0.0
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = [bcx, bcy - 0.02, self.z_board + 0.08]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 0.85, 200.0, -38.0
        self.renderer = mujoco.Renderer(m, height, width)
        self.width, self.height = width, height
        self._jp, self._jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        # posture: the most joint-limit-clear IK solution at the patch centre (searched offline)
        lo, hi = self.arm.lower, self.arm.upper
        rng, best = np.random.default_rng(0), None
        for _ in range(30):
            r = self.ik.solve(np.array([cx, cy, tz + TOOL_Z]), self.R_tool, lo + (hi - lo) * rng.uniform(0.15, 0.85, 7), 150)
            if r.pos_error < 2e-3 and r.rot_error < 0.02:
                mg = np.min(np.minimum(r.q - lo, hi - r.q))
                if best is None or mg > best[0]:
                    best = (mg, r.q)
        self.q_work = best[1]

    def close(self) -> None:
        self.renderer.close()

    # ------------------------------------------------------------- state
    def reset(self, dirt: np.ndarray) -> None:
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        self.dirt = dirt.astype(float).copy()
        self.dirt0 = self.dirt.sum()
        above = self.tcp_for_pad(self.sponge_home, 0.06)
        q = self.ik.solve(above, self.R_tool, self.q_work, 300).q
        d.qpos[self.qadr] = q
        self.q_post = q.copy()
        d.qpos[self.arm.finger_qadr] = self.cfg.gripper.open_m
        self.gripper(open_=True)
        mujoco.mj_forward(m, d)
        self.p_des, self.R_des = above.copy(), self.R_tool.copy()
        self.z_mode, self.f_int = "pos", 0.0
        self.t, self.wiped_path = 0.0, 0.0
        self.tau_peak, self.sat_steps, self.steps = np.zeros(7), 0, 0

    def tcp_for_pad(self, pad_xy, lift: float = 0.0) -> np.ndarray:
        """TCP position that puts the (grasped, flat) pad centre at pad_xy on the board, raised by lift."""
        return np.array([pad_xy[0], pad_xy[1], self.z_board + PAD[2]]) + self.pad_to_tcp + [0, 0, lift]

    def tcp(self):
        return self.data.site_xpos[self.site].copy(), self.data.site_xmat[self.site].reshape(3, 3).copy()

    def contact_force(self) -> float:
        """Normal force between the pad and the board (what a wrist F/T sensor would show)."""
        f, c6 = 0.0, np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if {int(c.geom1), int(c.geom2)} == {self.pad, self.board}:
                mujoco.mj_contactForce(self.model, self.data, i, c6)
                f += c6[0]
        return f

    def sponge_pose(self):
        return self.data.xpos[self.sponge_body].copy(), self.data.xmat[self.sponge_body].reshape(3, 3).copy()

    def gripper(self, open_: bool, squeeze_m: float = 0.0) -> None:
        self.data.ctrl[self.finger_act] = self.cfg.gripper.open_m if open_ else squeeze_m

    # ----------------------------------------------------------- control
    def _torque(self, f_des: float) -> np.ndarray:
        m, d = self.model, self.data
        mujoco.mj_jacSite(m, d, self._jp, self._jr, self.site)
        J = np.vstack([self._jp[:, self.dofadr], self._jr[:, self.dofadr]])
        qd = d.qvel[self.dofadr]
        p, Rm = self.tcp()
        v = J @ qd
        F = np.zeros(6)
        kp, kd = np.array([1200.0, 1200.0, 900.0]), np.array([70.0, 70.0, 60.0])
        F[:3] = kp * (self.p_des - p) - kd * v[:3]
        if self.z_mode == "force":
            fm = self.contact_force()
            self.f_int = np.clip(self.f_int + (f_des - fm) * DT, -2.0, 2.0)
            F[2] = -(f_des + 3.0 * self.f_int) - 40.0 * v[2]
        e_R = Rm.T @ self.R_des
        rv = np.array([e_R[2, 1] - e_R[1, 2], e_R[0, 2] - e_R[2, 0], e_R[1, 0] - e_R[0, 1]]) / 2
        F[3:] = 40.0 * (Rm @ rv) - 2.5 * v[3:]
        Jinv = J.T @ np.linalg.inv(J @ J.T + 1e-4 * np.eye(6))
        N = np.eye(7) - Jinv @ J
        q = d.qpos[self.qadr]
        tau = J.T @ F + N @ (8.0 * (self.q_post - q) - 0.8 * qd) + d.qfrc_bias[self.dofadr]
        self.steps += 1
        if np.any(np.abs(tau) > TAU_MAX):
            self.sat_steps += 1
        return np.clip(tau, -TAU_MAX, TAU_MAX)

    def step(self, n: int = 1, f_des: float = F_DES) -> None:
        m, d = self.model, self.data
        for _ in range(n):
            tau = self._torque(f_des)
            self.tau_peak = np.maximum(self.tau_peak, np.abs(tau))
            d.qfrc_applied[:] = 0.0
            d.qfrc_applied[self.dofadr] = tau
            sp_prev = d.xpos[self.sponge_body].copy()
            mujoco.mj_step(m, d)
            self.t += DT
            self._wear(sp_prev)

    def _wear(self, sp_prev: np.ndarray) -> None:
        fn = self.contact_force()
        if fn < 1.0:
            return
        pos, Rs = self.sponge_pose()
        if Rs[2, 2] < np.cos(np.radians(10)):  # pad not flat on the board
            return
        slide = np.linalg.norm(pos[:2] - sp_prev[:2])
        self.wiped_path += slide
        rel = (self.cells - pos[:2]) @ Rs[:2, :2]  # cell centres in the pad frame
        under = (np.abs(rel[:, 0]) < PAD[0]) & (np.abs(rel[:, 1]) < PAD[1])
        if under.any():
            self.dirt[under] = np.maximum(0.0, self.dirt[under] - WEAR_PER_NM * fn * slide)

    def cleaned(self) -> float:
        return 1.0 - self.dirt.sum() / self.dirt0

    def set_target(self, p: np.ndarray, update_posture: bool = True) -> None:
        self.p_des = np.asarray(p, float).copy()
        if update_posture:
            self.q_post = self.ik.solve(self.p_des, self.R_des, self.q_post, 30).q

    def render(self) -> np.ndarray:
        self.model.geom_rgba[self.dirt_gid, 3] = np.clip(self.dirt, 0, 1) * 0.95
        self.renderer.update_scene(self.data, camera=self.cam)
        return self.renderer.render()


def dirt_pattern(sim: PhysWipe, seed: int, stubborn: bool = False) -> np.ndarray:
    """2-3 random spills (soft blobs) inside the dirt area. stubborn: 1-2 smaller stains that need
    3x the wear (3 units of dirt per cell instead of 1)."""
    rng = np.random.default_rng(seed)
    d = np.zeros(len(sim.cells))
    n, (r0, r1), level = ((1, 3), (0.015, 0.025), 3.0) if stubborn else ((2, 4), (0.02, 0.04), 1.0)
    for _ in range(rng.integers(*n)):
        c = np.array([rng.uniform(sim.dirt_x[0] + 0.02, sim.dirt_x[1] - 0.02),
                      rng.uniform(sim.dirt_y[0] + 0.02, sim.dirt_y[1] - 0.02)])
        r = rng.uniform(r0, r1)
        d += np.exp(-np.sum((sim.cells - c) ** 2, 1) / (2 * r * r)) * 1.5
    return np.where(d > 0.3, np.minimum(d, 1.0), 0.0) * level
