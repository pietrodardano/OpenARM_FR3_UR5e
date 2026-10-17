"""Motion for the simulated arms: lateral-grasp orientation, straight Cartesian paths, joint timing.

Paths are straight lines in Cartesian space with the tool orientation held fixed, solved by
warm-started IK (so consecutive points stay on one joint branch), then timed with a minimum-jerk
profile along the joint-space arc length. The sim robots may cheat on object positions, but the
motion itself still has to be reachable and free of table contact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import mujoco
import numpy as np

from src.cross_embodiment.world import SimArmRobot

# Lateral grasp: tool z (approach) = world +x, finger-separation axis (tool y) = world y, so the
# fingers close sideways on an upright can.
GRASP_ROTATION = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])


class MotionError(RuntimeError):
    """A target is unreachable, a path leaves the reachable set, or it would touch the table."""


def min_jerk(s: np.ndarray) -> np.ndarray:
    """Minimum-jerk position profile for normalized time ``s`` in [0, 1]."""
    s = np.clip(s, 0.0, 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def min_jerk_rate(s: np.ndarray) -> np.ndarray:
    """Derivative of ``min_jerk`` with respect to normalized time."""
    s = np.clip(s, 0.0, 1.0)
    return 30 * s**2 - 60 * s**3 + 30 * s**4


@dataclass
class JointPath:
    """A polyline in joint space with a minimum-jerk time law over its arc length."""

    points: np.ndarray  # (k, n)
    duration_s: float

    def __post_init__(self) -> None:
        seg = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        self._arc = np.concatenate([[0.0], np.cumsum(seg)])
        self.length = float(self._arc[-1])

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Joint position and velocity at time ``t`` seconds after the start of the move."""
        if self.length < 1e-9 or self.duration_s <= 0:
            return self.points[-1].copy(), np.zeros(self.points.shape[1])
        tau = float(np.clip(t / self.duration_s, 0.0, 1.0))
        arc = float(min_jerk(np.array(tau))) * self.length
        k = int(np.clip(np.searchsorted(self._arc, arc, side="right") - 1, 0, len(self.points) - 2))
        span = self._arc[k + 1] - self._arc[k]
        frac = 0.0 if span < 1e-12 else (arc - self._arc[k]) / span
        q = self.points[k] + frac * (self.points[k + 1] - self.points[k])
        direction = (self.points[k + 1] - self.points[k]) / max(span, 1e-12)
        speed = float(min_jerk_rate(np.array(tau))) * self.length / self.duration_s
        return q, direction * speed

    def ticks(self, dt: float) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        steps = max(1, int(np.ceil(self.duration_s / dt)))
        for k in range(1, steps + 1):
            yield self.sample(k * dt)


def duration_for(length_rad: float, max_speed_rad_s: float, minimum_s: float = 0.4) -> float:
    """Move time so the peak joint speed (1.875 x mean for min-jerk) stays at the speed cap."""
    return max(minimum_s, 1.875 * length_rad / max_speed_rad_s)


def solve_pose(robot: SimArmRobot, position: np.ndarray, seed: np.ndarray, rotation: np.ndarray = GRASP_ROTATION) -> np.ndarray:
    """IK for one pose with the tool orientation held; raises ``MotionError`` if it does not converge."""
    result = robot.arm.solve_ik(np.asarray(position, dtype=float), seed, target_rot=rotation, restarts=8, pos_tol_m=1e-3, ori_tol_rad=0.02)
    if not result.converged:
        raise MotionError(
            f"{robot.name}: no IK solution at {np.round(position, 3).tolist()} "
            f"(position error {result.position_error_m * 1e3:.1f} mm, orientation error {result.axis_error_rad:.3f} rad)"
        )
    return result.q


def line_path(robot: SimArmRobot, start: np.ndarray, end: np.ndarray, seed: np.ndarray, step_m: float = 0.01, rotation: np.ndarray = GRASP_ROTATION) -> np.ndarray:
    """Joint waypoints along the straight line ``start`` to ``end`` with the orientation fixed."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    count = max(2, int(np.ceil(np.linalg.norm(end - start) / step_m)) + 1)
    qs = []
    q = seed
    for alpha in np.linspace(0.0, 1.0, count):
        q = solve_pose(robot, start + alpha * (end - start), q, rotation)
        qs.append(q)
    return np.array(qs)


def table_contacts(
    robot: SimArmRobot, model: "mujoco.MjModel", data: "mujoco.MjData", qs: np.ndarray, margin_m: float = 0.0, ignore_can: Optional[int] = None
) -> list[int]:
    """Indices of waypoints where the arm or gripper touches the table, a bin or a can (except ``ignore_can``)."""
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    env = {
        g
        for g in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith((f"{robot.name}_table", f"{robot.name}_bin"))
        or (
            (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(f"{robot.name}_can_")
            and (ignore_can is None or not (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(f"{robot.name}_can_{ignore_can}_"))
        )
    }
    mine = set()
    for b in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if name.startswith((f"{robot.name}_{robot.side}_", f"gripper_{robot.name}_{robot.side}_")):
            mine.update(g for g in range(model.ngeom) if model.geom_bodyid[g] == b and (model.geom_contype[g] or model.geom_conaffinity[g]))
    bad = []
    for k, q in enumerate(qs):
        scratch.qpos[robot.arm.qpos_idx] = q
        mujoco.mj_forward(model, scratch)
        for c in scratch.contact[: scratch.ncon]:
            if (c.geom1 in mine and c.geom2 in env) or (c.geom2 in mine and c.geom1 in env):
                if c.dist < margin_m:
                    bad.append(k)
                    break
    return bad


def plan_line(robot: SimArmRobot, model, data, start, end, seed, max_speed: float, check_table: bool = True, ignore_can: Optional[int] = None) -> JointPath:
    qs = line_path(robot, start, end, seed)
    if check_table:
        bad = table_contacts(robot, model, data, qs, ignore_can=ignore_can)
        if bad:
            raise MotionError(f"{robot.name}: straight path touches the table at waypoint(s) {bad[:5]}")
    length = float(np.sum(np.linalg.norm(np.diff(qs, axis=0), axis=1)))
    return JointPath(qs, duration_for(length, max_speed))


def plan_joint_move(robot: SimArmRobot, q_from: np.ndarray, q_to: np.ndarray, max_speed: float, model=None, data=None, check_table: bool = False, ignore_can: Optional[int] = None) -> JointPath:
    """Straight joint-space move between two configurations."""
    qs = np.linspace(q_from, q_to, 40)
    if check_table and model is not None:
        bad = table_contacts(robot, model, data, qs, ignore_can=ignore_can)
        if bad:
            raise MotionError(f"{robot.name}: joint move touches the table at waypoint(s) {bad[:5]}")
    length = float(np.linalg.norm(q_to - q_from))
    return JointPath(qs, duration_for(length, max_speed))
