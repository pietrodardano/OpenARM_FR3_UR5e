"""
OpenArm implementation of the task contracts, on the proven real-robot controller (from ``deploy/openarm-real``).

- :class:`OpenArmSkills` (``ManipulationSkills``): Cartesian moves via
  ``IKController.move_l``, gripper torque mode via ``scripts.grip_torque_test.grip`` /
  ``release``, a fixed weigh pose (``demo.weigh.pose``, can kept upright), joint torques
  from ``RobotState``.
- :class:`OpenArmPayloadEstimator` (``PayloadEstimator``): ``src.estimation.payload.measure_load``.

Every motion is checked before the arm moves; nothing here changes a command of the driver:

- **Reach:** a Cartesian target must lie inside ``controllers.ik.workspace`` (the controller
  would silently clip it otherwise), and the IK must follow every sampled point of the straight
  line within ``demo.motion`` tolerances.
- **Collision guard:** at every sampled configuration of a Cartesian line or a joint move, the
  arm bodies (link 3 to the fingertips) must stay ``table_margin_m`` above the measured table
  (:class:`~src.task.workcell.WorkcellGeometry`), and the arm's collision geoms must stay
  ``collision_clearance_m`` away from the torso, table, bins and the other arm
  (:class:`~src.planning.workcell_collision.WorkcellCollision` on the demo scene).
- **Side route:** with the 72.5 cm table close to the torso, the hanging zero pose is below the
  tabletop right behind its edge, so a direct joint move between zero and home sweeps the hand
  through the table. :meth:`OpenArmSkills.go_home` and :meth:`OpenArmSkills.park` route through
  ``demo.motion.side_route_qpos`` (arm swung out beside the table) whenever the direct move
  fails the guard. The driver's ``stop()`` and ``go_to_zero`` move straight to zero, so they may
  only be called after :meth:`OpenArmSkills.park`.

Any failed check raises :class:`UnsafeMotion` before the arm moves.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from ..control.ik_controller import IKController
from ..control.robot_interface import GRIPPER_OPEN_M, SIDES, RobotBackend, Side, arm_slice
from ..estimation.payload import measure_load
from ..planning.openarm_ik import BimanualIK
from ..planning.openarm_kinematics import OpenArmKinematics
from ..planning.workcell_collision import WorkcellCollision
from .interfaces import MassEstimate, TcpTarget
from .workcell import WorkcellGeometry

logger = logging.getLogger(__name__)

MIN_JERK_PEAK_FACTOR = 1.875  # peak / average speed of a minimum-jerk profile
GUARD_BODIES = ("link3", "link4", "link5", "link6", "link7", "hand", "hand_tcp", "left_finger", "right_finger")


@dataclass
class MotionConfig:
    """``demo.motion``, ``demo.grasp`` and ``demo.weigh`` settings used by the adapter."""

    peak_speed_m_s: float = 0.08
    peak_rot_speed_rad_s: float = 0.6
    min_move_s: float = 1.5
    joint_speed_deg_s: float = 15.0
    max_ik_pos_error_m: float = 0.005
    max_ik_rot_error_rad: float = 0.035
    check_points: int = 8
    table_margin_m: float = 0.035
    collision_clearance_m: float = 0.01
    side_route_qpos: List[List[float]] = field(default_factory=lambda: [[0.0, -1.05, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                                         [0.0, -1.05, 0.0, 1.3, 0.0, 0.0, 0.0]])
    grasp_torque_nm: float = 1.0
    open_m: float = GRIPPER_OPEN_M
    min_held_opening_m: float = 0.001
    settle_s: float = 1.0
    dither_deg: float = 1.0
    dither_period_s: float = 2.0
    dither_cycles: int = 3

    @classmethod
    def from_config(cls, demo: Mapping[str, Any]) -> "MotionConfig":
        """Builds the settings from the ``demo`` section."""
        m, g, w = demo.get("motion", {}), demo.get("grasp", {}), demo.get("weigh", {})
        d = cls()
        return cls(
            peak_speed_m_s=float(m.get("peak_speed_m_s", d.peak_speed_m_s)),
            peak_rot_speed_rad_s=float(m.get("peak_rot_speed_rad_s", d.peak_rot_speed_rad_s)),
            min_move_s=float(m.get("min_move_s", d.min_move_s)),
            joint_speed_deg_s=float(m.get("joint_speed_deg_s", d.joint_speed_deg_s)),
            max_ik_pos_error_m=float(m.get("max_ik_pos_error_m", d.max_ik_pos_error_m)),
            max_ik_rot_error_rad=float(m.get("max_ik_rot_error_rad", d.max_ik_rot_error_rad)),
            check_points=int(m.get("check_points", d.check_points)),
            table_margin_m=float(m.get("table_margin_m", d.table_margin_m)),
            collision_clearance_m=float(m.get("collision_clearance_m", d.collision_clearance_m)),
            side_route_qpos=[[float(v) for v in q] for q in m.get("side_route_qpos", d.side_route_qpos)],
            grasp_torque_nm=float(g.get("torque_nm", d.grasp_torque_nm)),
            open_m=float(np.clip(g.get("open_m", d.open_m), 0.0, GRIPPER_OPEN_M)),
            min_held_opening_m=float(g.get("min_held_opening_m", d.min_held_opening_m)),
            settle_s=float(w.get("settle_s", d.settle_s)),
            dither_deg=float(w.get("dither_deg", d.dither_deg)),
            dither_period_s=float(w.get("dither_period_s", d.dither_period_s)),
            dither_cycles=int(w.get("dither_cycles", d.dither_cycles)),
        )


class UnsafeMotion(RuntimeError):
    """A motion would leave the workspace, be unreachable for the IK, or hit the table."""


UnreachableTarget = UnsafeMotion  # earlier name


class OpenArmSkills:
    """``ManipulationSkills`` for one OpenArm arm on any ``RobotBackend``.

    Target orientation: the hand-forward orientation of ``robot.home_qpos`` (fingers along +X,
    closing axis along Y), pitched down by ``target.pitch`` about Y, turned by ``target.yaw``
    about +Z. The closing axis stays horizontal: a lateral squeeze of an upright can.
    """

    def __init__(self, robot: RobotBackend, cfg: Dict[str, Any], side: Side,
                 kinematics: OpenArmKinematics, require_object: bool = True,
                 collision_scene: Optional[str] = None) -> None:
        """Args:
            robot: Backend (started before the first skill call).
            cfg: Full task_config.yaml.
            side: Arm to use (must be in ``robot.active_sides``).
            kinematics: Official model wrapper, used for orientation, reach and table checks.
            require_object: If False, :meth:`grasp` reports success even when the fingers
                close on nothing (motion rehearsal without objects).
            collision_scene: Workcell MuJoCo scene for the geometric collision check (torso,
                table, bins, other arm); None = table guard only.
        """
        if side not in robot.active_sides:
            raise ValueError(f"{side} arm is not active (hardware.active_sides = {list(robot.active_sides)})")
        self.robot = robot
        self.cfg = cfg
        self.side: Side = side
        self.kin = kinematics
        self.require_object = require_object
        self.motion = MotionConfig.from_config(cfg["demo"])
        self.geo = WorkcellGeometry.from_config(cfg)
        self.ik = IKController.from_config(cfg, robot, kinematics)
        self._checker = BimanualIK(kinematics)
        self.home = {s: np.asarray(cfg["robot"]["home_qpos"][s], dtype=np.float64) for s in SIDES}
        self.rest = {s: np.asarray(cfg["hardware"]["rest_qpos"][s], dtype=np.float64) for s in SIDES}
        self.side_route = [np.asarray(q, dtype=np.float64) for q in self.motion.side_route_qpos]  # from zero outward
        self.R_forward = self.kin.fk(self.home)[side][1].copy()
        self._guard_ids = [self.kin.model.body(f"openarm_{side}_{b}").id for b in GUARD_BODIES]
        self.collision = (WorkcellCollision(collision_scene, side, self.motion.collision_clearance_m)
                          if collision_scene else None)

    # ----------------------------------------------------------- checks
    def rotation(self, yaw: float, pitch: float = 0.0) -> NDArray[np.float64]:
        """Hand-forward orientation pitched down by ``pitch`` (about Y), then turned by ``yaw`` (about Z)."""
        return Rotation.from_euler("z", yaw).as_matrix() @ Rotation.from_euler("y", pitch).as_matrix() @ self.R_forward

    def _arms(self, q_side: NDArray[np.float64]) -> Dict[Side, NDArray[np.float64]]:
        """Current joint targets of both arms, with ``q_side`` for the demo arm."""
        targets = self.robot.get_joint_targets()
        q = {s: targets[arm_slice(s)].copy() for s in SIDES}
        q[self.side] = np.asarray(q_side, dtype=np.float64)
        return q

    def table_clear(self, q_side: NDArray[np.float64]) -> Optional[str]:
        """None if ``q_side`` is clear of the workcell, else what it hits.

        Two checks: the guarded body origins stay ``table_margin_m`` above the table, and (with a
        collision scene) no arm geom comes closer than ``collision_clearance_m`` to the torso,
        table, bins or the other arm.
        """
        q = self._arms(q_side)
        self.kin.fk(q)
        for bid in self._guard_ids:
            p = self.kin.data.xpos[bid]
            if self.geo.below_table_surface(p, self.motion.table_margin_m):
                return f"{self.kin.model.body(bid).name} at {np.round(p, 3).tolist()} (table guard)"
        if self.collision is not None:
            hits = self.collision.contacts(q)
            if hits:
                return ", ".join(hits[:3])
        return None

    def check_joint_path(self, q_from: NDArray[np.float64], q_to: NDArray[np.float64], n: int = 24) -> None:
        """Raises :class:`UnsafeMotion` if the straight joint-space move hits the table guard."""
        for a in np.linspace(0.0, 1.0, n + 1):
            hit = self.table_clear(q_from + a * (q_to - q_from))
            if hit:
                raise UnsafeMotion(f"joint move collides: {hit} (at {a:.0%})")

    def check_reachable(self, target: TcpTarget) -> None:
        """Raises :class:`UnsafeMotion` unless the straight line to ``target`` is reachable and table-safe."""
        p1 = target.position
        box = self.ik.cfg.workspace[self.side]
        if not np.allclose(box.clip(p1), p1):
            raise UnsafeMotion(f"{np.round(p1, 3).tolist()} is outside controllers.ik.workspace.{self.side} "
                               f"(low {list(box.low)}, high {list(box.high)})")
        self.ik.reset()
        p0, R0 = self.ik.pose_cmd[self.side]
        r0, r1 = Rotation.from_matrix(R0), Rotation.from_matrix(self.rotation(target.yaw, target.pitch))
        delta = (r0.inv() * r1).as_rotvec()
        q_seed = {s: v.copy() for s, v in self.ik.q_ref.items()}
        n = max(1, self.motion.check_points)
        for k in range(1, n + 1):
            a = k / n
            R = (r0 * Rotation.from_rotvec(a * delta)).as_matrix()
            res = self._checker.solve({self.side: (p0 + a * (p1 - p0), R)}, q_seed, max_iters=100)
            pe, re = res.pos_error[self.side], res.rot_error[self.side]
            if pe > self.motion.max_ik_pos_error_m or re > self.motion.max_ik_rot_error_rad:
                raise UnsafeMotion(
                    f"IK cannot follow the line to {np.round(p1, 3).tolist()} (at {a:.0%}: "
                    f"{pe * 1e3:.0f} mm / {np.degrees(re):.1f} deg error)")
            hit = self.table_clear(res.q_arm[self.side])
            if hit:
                raise UnsafeMotion(f"line to {np.round(p1, 3).tolist()} collides: {hit} (at {a:.0%})")
            q_seed = res.q_arm

    # ----------------------------------------------------------- joint routes
    def _joint_move(self, q: NDArray[np.float64]) -> None:
        """Minimum-jerk joint move of the arm at ``motion.joint_speed_deg_s`` peak."""
        from scripts.grip_torque_test import move_to_pose

        move_to_pose(self.robot, self.side, q, self.motion.joint_speed_deg_s)
        self.ik.reset()

    def _check_chain(self, chain: List[NDArray[np.float64]]) -> None:
        for a, b in zip(chain[:-1], chain[1:]):
            self.check_joint_path(a, b)

    def _route(self, goal: NDArray[np.float64]) -> None:
        """Joint move to ``goal``: direct if table-safe, else along the side route.

        Joins the side route at its waypoint closest to the start, so a start beside the table
        and a start at zero both work, and leaves it at the waypoint closest to the goal.
        """
        q0 = self.robot.get_joint_targets()[arm_slice(self.side)].copy()
        try:
            self.check_joint_path(q0, goal)
            chain = [q0, goal]
        except UnsafeMotion as direct:
            route = self.side_route
            i_in = int(np.argmin([np.max(np.abs(q0 - w)) for w in route]))
            i_out = int(np.argmin([np.max(np.abs(goal - w)) for w in route]))
            step = 1 if i_out >= i_in else -1
            chain = [q0] + [route[i] for i in range(i_in, i_out + step, step)] + [goal]
            try:
                self._check_chain(chain)
            except UnsafeMotion as via:
                raise UnsafeMotion(f"no table-safe joint route. Direct: {direct}. Side route: {via}") from via
            logger.info("routing the %s arm beside the table (side route)", self.side)
        for q in chain[1:]:
            if np.max(np.abs(q - self.robot.get_joint_targets()[arm_slice(self.side)])) > 1e-6:
                self._joint_move(q)

    def park(self) -> None:
        """Moves to the first side-route waypoint (straight arm beside the table), from which the
        straight move to zero (``go_to_zero`` / ``robot.stop()``) stays clear of the table.
        Call before handing over to them."""
        self.check_joint_path(self.side_route[0], self.rest[self.side])
        self._route(self.side_route[0])

    # ------------------------------------------------- ManipulationSkills
    def go_home(self) -> None:
        """Table-safe joint route to ``robot.home_qpos`` (hand forward, elbow 90°)."""
        self._route(self.home[self.side])

    def move_to(self, target: TcpTarget) -> None:
        """Checked straight-line TCP motion (``IKController.move_l``)."""
        self.check_reachable(target)
        p0, R0 = self.ik.pose_cmd[self.side]
        R1 = self.rotation(target.yaw, target.pitch)
        dist = float(np.linalg.norm(target.position - p0))
        angle = float(Rotation.from_matrix(R0.T @ R1).magnitude())
        duration = max(self.motion.min_move_s,
                       MIN_JERK_PEAK_FACTOR * dist / self.motion.peak_speed_m_s,
                       MIN_JERK_PEAK_FACTOR * angle / self.motion.peak_rot_speed_rad_s)
        logger.info("move_l %s -> %s pitch %.0f deg (%.2f m, %.0f deg, %.1f s)", self.side,
                    np.round(target.position, 3).tolist(), np.degrees(target.pitch), dist, np.degrees(angle), duration)
        self.ik.move_l({self.side: (target.position, R1)}, duration=duration)

    def grasp(self) -> bool:
        """Torque-mode grip (approach, then ``demo.grasp.torque_nm``). False if nothing is held."""
        from scripts.grip_torque_test import grip

        opening = grip(self.robot, self.side, self.motion.grasp_torque_nm, self.cfg)
        if opening < self.motion.min_held_opening_m and self.require_object:
            self.release()
            return False
        return True

    def release(self) -> None:
        """Leaves torque mode and opens to ``demo.grasp.open_m``."""
        from scripts.grip_torque_test import release

        release(self.robot, self.side, self.motion.open_m)
        self.ik.reset()

    def lift_to_weigh_pose(self) -> None:
        """Checked Cartesian move to the fixed weigh pose (``demo.weigh.pose``), can kept upright."""
        w = self.cfg["demo"]["weigh"]["pose"]
        self.move_to(TcpTarget(float(w["x"]), float(w["y"]), self.geo.table_top_z_m + float(w["above_table_m"]),
                               0.0, float(np.radians(w.get("pitch_deg", 0.0)))))

    def read_joint_torques(self) -> NDArray[np.float64]:
        """Measured joint torques of the arm (7,), Nm."""
        state = self.robot.get_state()
        if state.fault:
            raise RuntimeError(f"robot fault: {state.fault}")
        return state.arms[self.side].tau.copy()

    def wait(self, dt: float) -> None:
        """Holds the current targets for ``dt`` (``robot.advance``)."""
        self.robot.advance(dt)
        if self.robot.fault:
            raise RuntimeError(f"robot fault: {self.robot.fault}")

    def abort(self) -> None:
        """Best effort, never raises: lift 10 cm if low over the table, then park beside it.

        A held object stays gripped. If a step fails, the arm holds its pose and the runner
        does not move to zero (``scripts/run_weigh_demo.py``).
        """
        try:
            if self.robot.fault:
                return
            self.ik.reset()
            p, R = self.ik.pose_cmd[self.side]
            if p[2] < self.geo.table_top_z_m + 0.20:
                finger = R[:, 2]  # tool z = finger direction
                pitch = float(np.arcsin(np.clip(-finger[2], -1.0, 1.0)))
                yaw = float(np.arctan2(finger[1], finger[0]))
                self.move_to(TcpTarget(float(p[0]), float(p[1]), float(p[2]) + 0.10, yaw, pitch))
        except Exception as exc:  # noqa: BLE001 - abort must never raise
            logger.error("abort: lift failed: %s", exc)
        try:
            if not self.robot.fault:
                self.park()
        except Exception as exc:  # noqa: BLE001
            logger.error("abort: park failed: %s", exc)


class OpenArmPayloadEstimator:
    """``PayloadEstimator`` backed by ``src.estimation.payload.measure_load``."""

    def __init__(self, robot: RobotBackend, kinematics: OpenArmKinematics, side: Side, motion: MotionConfig) -> None:
        """Args:
            robot: Started backend.
            kinematics: Official model wrapper (gravity model, TCP Jacobian).
            side: Arm holding the object.
            motion: Settle time and dither parameters.
        """
        self.robot = robot
        self.kin = kinematics
        self.side: Side = side
        self.motion = motion

    def measure(self, label: str) -> MassEstimate:
        """Settles, then runs the dithered load measurement at the current pose."""
        self.robot.advance(self.motion.settle_s)
        s = measure_load(self.robot, self.kin, self.side, label, amplitude_deg=self.motion.dither_deg,
                         period_s=self.motion.dither_period_s, cycles=self.motion.dither_cycles)
        logger.info("%s: %.0f +- %.0f g equivalent (joints %s)", label, s.mass * 1e3, s.mass_se * 1e3,
                    [j + 1 for j in s.joints])
        return MassEstimate(label=label, mass_kg=s.mass, mass_se_kg=s.mass_se, raw=s)
