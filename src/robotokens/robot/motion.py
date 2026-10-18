"""Checked arm motion: straight-line TCP moves (moveL) and table-safe joint routes (SR4).

OpenArm has no Cartesian controller, so :meth:`Motion.move_to` streams IK solutions of a
minimum-jerk straight line (position lerp + orientation slerp) at ``motion.rate_hz`` into the
driver. Before the arm moves, every line is checked at ``check_points``: the target must be
inside ``motion.workspace``, the IK must follow within tolerance and the arm must stay clear of
the table, torso and bins (:meth:`ArmModel.blocked`). Joint moves are checked the same way and,
when the direct move would sweep through the table (rest <-> home), go via the side route.

Any failed check raises :class:`UnsafeMotion` before the arm moves.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation, Slerp

from ..config import Config
from .driver import Arm
from .ik import IK
from .model import ArmModel, Pose
from .safety import MIN_JERK_PEAK, minimum_jerk

logger = logging.getLogger(__name__)

NEAR_REST_RAD = 0.2  # a start within this of rest may settle onto rest unchecked (see plan_route)


class UnsafeMotion(RuntimeError):
    """A motion would leave the workspace, be unreachable for the IK, or hit the workcell."""


@dataclass(frozen=True)
class Target:
    """TCP target in the base frame: position (m), hand yaw about +Z and pitch down (rad)."""

    x: float
    y: float
    z: float
    yaw: float = 0.0
    pitch: float = 0.0

    @property
    def position(self) -> NDArray:
        return np.array([self.x, self.y, self.z])


class Motion:
    def __init__(self, arm: Arm, model: ArmModel, cfg: Config) -> None:
        self.arm = arm
        self.model = model
        self.cfg = cfg
        self.mc = cfg.motion
        self.ik = IK(model)
        self.rest = np.asarray(cfg.robot.rest_qpos, dtype=np.float64)
        self.side_route = [np.asarray(q, dtype=np.float64) for q in cfg.robot.side_route_qpos]
        self.q_ref: NDArray = np.zeros(7)
        self.pose_cmd: Pose = (np.zeros(3), np.eye(3))

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        """Re-anchors at the driver's current joint targets."""
        self.q_ref = self.arm.targets()[0]
        self.pose_cmd = self.model.fk(self.q_ref)

    def tcp_command(self) -> Pose:
        """TCP pose of the current joint targets."""
        self.reset()
        return self.pose_cmd

    def tcp_measured(self) -> Pose:
        return self.model.fk(self.arm.state().q)

    # ----------------------------------------------------------------- checks
    def check_joint_path(self, q0: NDArray, q1: NDArray, n: int = 24) -> None:
        g = self.arm.targets()[1]
        for a in np.linspace(0.0, 1.0, n + 1):
            hit = self.model.blocked(q0 + a * (q1 - q0), g)
            if hit:
                raise UnsafeMotion(f"joint move collides: {hit} (at {a:.0%})")

    def _line(self, target: Target) -> Tuple[NDArray, Rotation, NDArray]:
        p0, R0 = self.pose_cmd
        r0, r1 = Rotation.from_matrix(R0), Rotation.from_matrix(self.model.hand_rotation(target.yaw, target.pitch))
        return p0, r0, (r0.inv() * r1).as_rotvec()

    def check_line(self, target: Target) -> None:
        """Raises :class:`UnsafeMotion` unless the straight line to ``target`` is reachable and clear."""
        p1 = target.position
        ws = self.mc.workspace
        if np.any(p1 < ws.low) or np.any(p1 > ws.high):
            raise UnsafeMotion(f"{np.round(p1, 3).tolist()} is outside motion.workspace "
                               f"(low {list(ws.low)}, high {list(ws.high)})")
        self.reset()
        p0, r0, delta = self._line(target)
        q_seed, g = self.q_ref, self.arm.targets()[1]
        n = max(1, self.mc.check_points)
        for k in range(1, n + 1):
            a = k / n
            res = self.ik.solve(p0 + a * (p1 - p0), (r0 * Rotation.from_rotvec(a * delta)).as_matrix(), q_seed, 100)
            if res.pos_error > self.mc.max_ik_pos_error_m or res.rot_error > self.mc.max_ik_rot_error_rad:
                raise UnsafeMotion(f"IK cannot follow the line to {np.round(p1, 3).tolist()} (at {a:.0%}: "
                                   f"{res.pos_error * 1e3:.0f} mm / {np.degrees(res.rot_error):.1f} deg error)")
            hit = self.model.blocked(res.q, g)
            if hit:
                raise UnsafeMotion(f"line to {np.round(p1, 3).tolist()} collides: {hit} (at {a:.0%})")
            q_seed = res.q

    # ----------------------------------------------------------------- motion
    def move_to(self, target: Target) -> None:
        """Checked straight-line TCP move (minimum-jerk timing at the configured peak speeds)."""
        self.check_line(target)
        p0, r0, delta = self._line(target)
        p1 = target.position
        dist, angle = float(np.linalg.norm(p1 - p0)), float(np.linalg.norm(delta))
        duration = max(self.mc.min_move_s, MIN_JERK_PEAK * dist / self.mc.peak_speed_m_s,
                       MIN_JERK_PEAK * angle / self.mc.peak_rot_speed_rad_s)
        logger.info("moveL -> %s pitch %.0f deg (%.2f m, %.0f deg, %.1f s)", np.round(p1, 3).tolist(),
                    np.degrees(target.pitch), dist, np.degrees(angle), duration)
        slerp = Slerp([0.0, 1.0], Rotation.concatenate([r0, r0 * Rotation.from_rotvec(delta)]))
        steps = max(1, int(duration * self.mc.rate_hz))
        for k in range(1, steps + 1):
            a = minimum_jerk(k / steps)
            res = self.ik.solve(p0 + a * (p1 - p0), slerp(a).as_matrix(), self.q_ref, self.mc.ik_iters)
            self.q_ref = res.q
            self.arm.set_targets(q=res.q)
            self.arm.advance(1.0 / self.mc.rate_hz)
            if self.arm.fault:
                raise RuntimeError(f"move aborted: {self.arm.fault}")
        self.pose_cmd = self.model.fk(self.q_ref)

    def move_joints(self, q: NDArray, gripper: float | None = None) -> None:
        """Minimum-jerk joint move at ``joint_speed_deg_s`` peak (not path-checked: use :meth:`route`)."""
        q0 = self.arm.targets()[0]
        dist_deg = float(np.degrees(np.max(np.abs(np.asarray(q) - q0))))
        duration = max(1.5, MIN_JERK_PEAK * dist_deg / self.mc.joint_speed_deg_s)
        self.arm.move_joints(q, gripper=gripper, duration=duration)
        self.arm.advance(0.2)
        self.reset()

    def plan_route(self, goal: NDArray, q0: NDArray | None = None) -> List[NDArray]:
        """Checked joint waypoints from the current targets to ``goal``: direct, else via the side route.

        Joins the side route at its waypoint closest to the start and leaves it at the one closest
        to the goal, so a start at rest and a start beside the table both work. A start that hangs
        near rest but already inside the (conservative) table guard, e.g. after the arm was moved
        by hand, first settles straight onto the exact rest pose: a short move behind the table edge.
        """
        q0 = self.arm.targets()[0] if q0 is None else np.asarray(q0, dtype=np.float64)
        goal = np.asarray(goal, dtype=np.float64)
        if (self.model.blocked(q0, self.arm.targets()[1]) and not np.allclose(q0, self.rest)
                and np.max(np.abs(q0 - self.rest)) <= NEAR_REST_RAD):
            logger.info("start pose is inside the table guard near rest: settling onto the rest pose first")
            return [q0] + self.plan_route(goal, self.rest)
        try:
            self.check_joint_path(q0, goal)
            return [q0, goal]
        except UnsafeMotion as direct:
            route = self.side_route
            i_in = int(np.argmin([np.max(np.abs(q0 - w)) for w in route]))
            i_out = int(np.argmin([np.max(np.abs(goal - w)) for w in route]))
            step = 1 if i_out >= i_in else -1
            chain = [q0] + [route[i] for i in range(i_in, i_out + step, step)] + [goal]
            try:
                for a, b in zip(chain[:-1], chain[1:]):
                    self.check_joint_path(a, b)
            except UnsafeMotion as via:
                raise UnsafeMotion(f"no table-safe joint route. Direct: {direct}. Side route: {via}") from via
            return chain

    def route(self, goal: NDArray) -> None:
        chain = self.plan_route(goal)
        if len(chain) > 2:
            logger.info("routing beside the table (side route)")
        for q in chain[1:]:
            if np.max(np.abs(q - self.arm.targets()[0])) > 1e-6:
                self.move_joints(q)

    def go_home(self) -> None:
        """Table-safe joint route to the home pose (hand forward, elbow 90 deg)."""
        self.route(self.model.home)

    def park(self) -> None:
        """Moves beside the table (first side-route waypoint), from where the straight move to rest
        (``Arm.stop()``) is clear of the table."""
        self.check_joint_path(self.side_route[0], self.rest)
        self.route(self.side_route[0])
