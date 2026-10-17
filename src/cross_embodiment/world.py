"""Shared MuJoCo world for the cross-embodiment demo: simulated FR3/UR5e arms plus a kinematic OpenArm.

* ``SimArmRobot``: one simulated arm on joint-torque control (PD + feedforward) with a Robotiq
  2F-85 gripper. Joint torque is applied through ``qfrc_applied``, so the arm feels gravity and the
  weight of a grasped can, and the applied torque is what a real motor would report.
* The real OpenArm is a follower: its joints are overwritten each step from the real arm's state
  (see ``DemoWorld.set_openarm``). It never moves on its own and never touches the sim objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import mujoco
import numpy as np

from src.cross_embodiment.layout import SCENE_PATH, DemoLayout, load_layout
from src.kinematics.arm_kinematics import ArmModel

UR5E_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")

# Joint-torque control gains and limits per robot (rough catalogue values; tuned only enough to be
# stable and to hold the arm against gravity). Joint names are templates over the arm side.
ROBOT_SPECS = {
    "ur5e": dict(
        joints=[f"ur5e_{{side}}_{n}" for n in UR5E_JOINTS],
        kp=[2000.0, 2000.0, 1500.0, 500.0, 500.0, 300.0],
        kd=[200.0, 200.0, 150.0, 50.0, 50.0, 30.0],
        tau_limit=[150.0, 150.0, 150.0, 28.0, 28.0, 28.0],
    ),
    "fr3": dict(
        joints=[f"fr3_{{side}}_fr3_joint{i}" for i in range(1, 8)],
        kp=[600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0],
        kd=[50.0, 50.0, 50.0, 20.0, 20.0, 20.0, 10.0],
        tau_limit=[87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0],
    ),
}


def joint_names(robot: str, side: str) -> list[str]:
    return [n.format(side=side) for n in ROBOT_SPECS[robot]["joints"]]
GRIPPER_OPEN, GRIPPER_CLOSED = 0.0, 255.0  # 2F-85 actuator control range


def set_weld(model: "mujoco.MjModel", data: "mujoco.MjData", eq_id: int, body1: int, body2: int) -> None:
    """Activate a weld between two bodies, freezing their current relative pose."""
    r1 = data.xmat[body1].reshape(3, 3)
    rel_pos = r1.T @ (data.xpos[body2] - data.xpos[body1])
    rel_quat = np.zeros(4)
    mujoco.mju_mat2Quat(rel_quat, (r1.T @ data.xmat[body2].reshape(3, 3)).reshape(9))
    model.eq_data[eq_id, 0:3] = 0.0  # anchor at body2's origin
    model.eq_data[eq_id, 3:6] = rel_pos
    model.eq_data[eq_id, 6:10] = rel_quat
    data.eq_active[eq_id] = 1


@dataclass
class SimArmRobot:
    """The working arm of a simulated robot with its gripper, driven by joint-torque PD control.

    Its home pose is wherever the scene's ``home`` keyframe puts it (the data must be at that keyframe
    when the robot is created).
    """

    name: str
    model: "mujoco.MjModel"
    data: "mujoco.MjData"
    side: str = "left"
    tau_noise_std: float = 0.02  # Nm, measurement noise on the reported torque
    seed: int = 0
    arm: ArmModel = field(init=False)
    kp: np.ndarray = field(init=False)
    kd: np.ndarray = field(init=False)
    tau_limit: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        spec = ROBOT_SPECS[self.name]
        m = self.model
        self.arm = ArmModel(m, joint_names(self.name, self.side), f"gripper_{self.name}_{self.side}_pinch", "site")
        self.kp, self.kd = np.array(spec["kp"]), np.array(spec["kd"])
        self.tau_limit = np.array(spec["tau_limit"])
        self.home_q = self.data.qpos[self.arm.qpos_idx].copy()
        self.n = self.arm.n
        self._ctrl = m.actuator(f"gripper_{self.name}_{self.side}_fingers_actuator").id
        self._base_body = m.body(f"gripper_{self.name}_{self.side}_base").id
        self._pinch = self.arm.tcp_id
        self.q_des = self.home_q.copy()
        self.dq_des = np.zeros(self.n)
        self.tau_ff = np.zeros(self.n)
        self.tau = np.zeros(self.n)  # last applied torque = what a real motor would report
        self._rng = np.random.default_rng(self.seed)
        self.attached: Optional[int] = None
        self._weld_ids = {
            i: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"{self.name}_grasp_can_{i}") for i in (1, 2)
        }

    # ------------------------------------------------------------------ state
    @property
    def q(self) -> np.ndarray:
        return self.data.qpos[self.arm.qpos_idx].copy()

    @property
    def dq(self) -> np.ndarray:
        return self.data.qvel[self.arm.dof_idx].copy()

    def pinch_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """World position and rotation of the gripper pinch point (the TCP)."""
        return self.data.site_xpos[self._pinch].copy(), self.data.site_xmat[self._pinch].reshape(3, 3).copy()

    def measured_torque(self) -> np.ndarray:
        """Joint torque as a motor would report it (applied torque plus sensor noise)."""
        return self.tau + self._rng.normal(0.0, self.tau_noise_std, self.n)

    def jacobian_at_pinch(self) -> np.ndarray:
        """3 x n position Jacobian of the pinch point at the current state."""
        jp = np.zeros((3, self.model.nv))
        jr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jp, jr, self._pinch)
        return jp[:, self.arm.dof_idx]

    def gravity_torque(self) -> np.ndarray:
        """Gravity (plus Coriolis) torque of the arm and gripper only; a welded can is not included."""
        return self.data.qfrc_bias[self.arm.dof_idx].copy()

    # --------------------------------------------------------------- commands
    def command(self, q_des: np.ndarray, dq_des: Optional[np.ndarray] = None, tau_ff: Optional[np.ndarray] = None) -> None:
        self.q_des = np.asarray(q_des, dtype=float)
        self.dq_des = np.zeros(self.n) if dq_des is None else np.asarray(dq_des, dtype=float)
        self.tau_ff = np.zeros(self.n) if tau_ff is None else np.asarray(tau_ff, dtype=float)

    def apply_torque(self) -> None:
        """Write the PD + feedforward torque into ``qfrc_applied`` (call every physics substep)."""
        tau = self.kp * (self.q_des - self.q) + self.kd * (self.dq_des - self.dq) + self.tau_ff
        self.tau = np.clip(tau, -self.tau_limit, self.tau_limit)
        self.data.qfrc_applied[self.arm.dof_idx] = self.tau

    def gripper(self, closed: bool) -> None:
        self.data.ctrl[self._ctrl] = GRIPPER_CLOSED if closed else GRIPPER_OPEN

    # ------------------------------------------------------------------ grasp
    def attach(self, can_index: int) -> None:
        """Weld a can to the gripper at its current relative pose (the sim cheat that holds the object)."""
        can_body = self.model.body(f"{self.name}_can_{can_index}").id
        set_weld(self.model, self.data, self._weld_ids[can_index], self._base_body, can_body)
        # The weld alone holds the can, so switch its collisions off while attached: pad contact on top of
        # the weld would over-constrain it and make the arm's torque oscillate.
        geom = self.model.body_geomadr[can_body]
        self._saved_collision = (int(geom), int(self.model.geom_contype[geom]), int(self.model.geom_conaffinity[geom]))
        self.model.geom_contype[geom] = 0
        self.model.geom_conaffinity[geom] = 0
        self.attached = can_index

    def detach(self) -> None:
        if self.attached is not None:
            self.data.eq_active[self._weld_ids[self.attached]] = 0
            geom, contype, conaffinity = self._saved_collision
            self.model.geom_contype[geom], self.model.geom_conaffinity[geom] = contype, conaffinity
            self.attached = None


class ParkedArm:
    """The idle arm of a sim robot: held at its home pose with PD plus gravity feedforward.

    It is decoration only: its collisions are switched off so it can never load the working arm.
    """

    def __init__(self, robot_name: str, side: str, model: "mujoco.MjModel", data: "mujoco.MjData") -> None:
        spec = ROBOT_SPECS[robot_name]
        names = joint_names(robot_name, side)
        ids = [model.joint(n).id for n in names]
        self.qadr, self.dadr = model.jnt_qposadr[ids], model.jnt_dofadr[ids]
        self.kp, self.kd = np.array(spec["kp"]), np.array(spec["kd"])
        self.tau_limit = np.array(spec["tau_limit"])
        self.q_hold = data.qpos[self.qadr].copy()
        self.data = data
        prefixes = (f"{robot_name}_{side}_", f"gripper_{robot_name}_{side}_")
        for b in range(model.nbody):
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(prefixes):
                for g in range(model.ngeom):
                    if model.geom_bodyid[g] == b:
                        model.geom_contype[g] = 0
                        model.geom_conaffinity[g] = 0

    def apply_torque(self) -> None:
        d = self.data
        tau = self.kp * (self.q_hold - d.qpos[self.qadr]) - self.kd * d.qvel[self.dadr] + d.qfrc_bias[self.dadr]
        d.qfrc_applied[self.dadr] = np.clip(tau, -self.tau_limit, self.tau_limit)


class DemoWorld:
    """The whole scene: sim robots plus the kinematic OpenArm follower."""

    def __init__(self, layout: Optional[DemoLayout] = None, scene_xml: str = str(SCENE_PATH), tau_noise_std: float = 0.02, seed: int = 0) -> None:
        self.layout = layout or load_layout()
        self.model = mujoco.MjModel.from_xml_path(scene_xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self._apply_can_masses()
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)  # mj_setConst left the data at qpos0
        self.robots = {
            name: SimArmRobot(name, self.model, self.data, side=self.layout.stations[name].arm, tau_noise_std=tau_noise_std, seed=seed + i)
            for i, name in enumerate(self.layout.sim_robots)
        }
        other = {"left": "right", "right": "left"}
        self.parked = {name: ParkedArm(name, other[self.layout.stations[name].arm], self.model, self.data) for name in self.layout.sim_robots}
        m = self.model
        self._oa_q_idx = np.array([m.jnt_qposadr[m.joint(f"openarm_left_joint{i}").id] for i in range(1, 8)])
        self._oa_finger_idx = np.array([m.jnt_qposadr[m.joint(f"openarm_left_finger_joint{i}").id] for i in (1, 2)])
        # Every OpenArm joint (both arms, fingers) is held kinematically; the right arm stays parked.
        self._oa_all = np.array(
            [m.jnt_qposadr[j] for j in range(m.njnt) if (m.joint(j).name or "").startswith("openarm_")]
        )
        self._oa_dofs = np.array(
            [m.jnt_dofadr[j] for j in range(m.njnt) if (m.joint(j).name or "").startswith("openarm_")]
        )
        self._oa_pose = self.data.qpos[self._oa_all].copy()
        self.time = 0.0
        mujoco.mj_forward(m, self.data)

    def _apply_can_masses(self) -> None:
        """Make the layout the single source of truth for can masses, whatever the generated scene file says.

        The scene XML bakes masses in at generation time. Overriding only ``body_mass`` would leave the
        inertia and derived constants stale (and can destabilise the weld), so set a consistent solid
        cylinder and recompute the model constants.
        """
        m, can = self.model, self.layout.can
        for robot in self.layout.sim_robots:
            station = self.layout.stations[robot]
            for index in (1, 2):
                body = m.body(f"{robot}_can_{index}").id
                mass = station.true_mass_kg(index, can)
                height = 2.0 * can.half_height_m
                lateral = mass * (3.0 * can.radius_m**2 + height**2) / 12.0
                m.body_mass[body] = mass
                m.body_inertia[body] = [lateral, lateral, 0.5 * mass * can.radius_m**2]
        mujoco.mj_setConst(m, self.data)

    # ------------------------------------------------------------ OpenArm follower
    def set_openarm(self, q_left: Sequence[float], gripper_opening_m: Optional[float] = None) -> None:
        """Mirror the real left OpenArm: set its joints (and fingers) from the real arm's measured state."""
        pose = self._oa_pose.copy()
        lookup = {int(a): i for i, a in enumerate(self._oa_all)}
        for adr, value in zip(self._oa_q_idx, q_left):
            pose[lookup[int(adr)]] = float(value)
        if gripper_opening_m is not None:
            for adr in self._oa_finger_idx:
                pose[lookup[int(adr)]] = float(np.clip(gripper_opening_m, 0.0, 0.044))
        self._oa_pose = pose

    # ------------------------------------------------------------------- stepping
    def step(self, duration_s: float) -> None:
        """Advance physics by ``duration_s`` with zero-order-hold commands."""
        m, d = self.model, self.data
        for _ in range(max(1, int(round(duration_s / m.opt.timestep)))):
            d.qfrc_applied[:] = 0.0
            for robot in self.robots.values():
                robot.apply_torque()
            for parked in self.parked.values():
                parked.apply_torque()
            mujoco.mj_step(m, d)
            d.qpos[self._oa_all] = self._oa_pose
            d.qvel[self._oa_dofs] = 0.0
            self.time += m.opt.timestep
        mujoco.mj_forward(m, d)  # refresh derived quantities after the kinematic overwrite

    # ------------------------------------------------------- simulated user (hand-over)
    def set_user_hand(self, robot: str, position: Sequence[float]) -> None:
        """Move the simulated user's hand (a mocap body, visual only until it takes the can)."""
        self.data.mocap_pos[self.model.body_mocapid[self.model.body(f"{robot}_user_hand").id]] = position

    def user_hand_pos(self, robot: str) -> np.ndarray:
        return self.data.mocap_pos[self.model.body_mocapid[self.model.body(f"{robot}_user_hand").id]].copy()

    def user_takes_can(self, robot: str, can_index: int) -> None:
        """The user's hand grips the can: weld it to the hand body at the current relative pose."""
        eq = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, f"{robot}_hand_can_{can_index}")
        set_weld(self.model, self.data, eq, self.model.body(f"{robot}_user_hand").id, self.model.body(f"{robot}_can_{can_index}").id)

    def push_can(self, robot: str, can_index: int, force_xyz: Sequence[float]) -> None:
        """Apply an external force (N) to a can: the simulated user pulling on the held object."""
        self.data.xfrc_applied[self.model.body(f"{robot}_can_{can_index}").id, :3] = force_xyz

    def body_pos(self, name: str) -> np.ndarray:
        return self.data.xpos[self.model.body(name).id].copy()

    def can_pos(self, robot: str, index: int) -> np.ndarray:
        return self.body_pos(f"{robot}_can_{index}")
