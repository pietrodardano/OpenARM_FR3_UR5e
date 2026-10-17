"""Scripted pick-and-place skills for the OpenArm tabletop scene.

Object poses are observed once (:meth:`SkillRunner.perceive`) and remembered;
skills plan from that memory, not from continuous perception. In simulation
the observation is the true pose, standing in for a head-camera estimate.

Motion primitives (both execute joint-space paths on the arm position servos):
  - ``move_joint``: smooth joint move to a configuration (transit).
  - ``move_linear``: straight TCP line, planned by continuation IK (each waypoint
    seeded from the previous, so the whole line stays on one IK branch).
Skills compose them: ``pick(side, obj)`` and ``place(side, xy, surface_z)``.
Approach paths are planned backwards from the grasp/place pose, so approach,
grasp and lift always share an IK branch.

Top-down poses near the robot's midline are only reachable around the arm's
natural wrist yaw, so grasps specify yaw only up to the object's symmetry
(``Grasp.yaw_period_deg``) and placing leaves yaw free.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation, Slerp

from src.control.ik import BimanualIK, Pose
from src.simulation.tabletop_scene import HOME_ARM_QPOS, OBJECT_NAMES, SIDES, TABLE_HEIGHT, Side, TabletopScene

logger = logging.getLogger(__name__)

JointPath = List[NDArray[np.float64]]


@dataclass(frozen=True)
class Grasp:
    """How to grasp an object, relative to its remembered pose."""

    # TCP target = object origin + this world-frame offset (m).
    offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Approach axis tilted from straight down towards +x (deg); 0 = top-down.
    tilt_deg: float = 0.0
    # Finger closing axis yaw about world z (deg; 0 = along world y), relative to the
    # object's yaw. Any multiple of ``yaw_period_deg`` is equivalent; None = any yaw.
    yaw_deg: float = 0.0
    yaw_period_deg: Optional[float] = None
    # Pre-grasp stand-off along the approach axis (m).
    pregrasp: float = 0.10


# Fingertips extend ~2 cm past the TCP, so a top-down TCP must stay >= ~2.5 cm
# above the table.
GRASPS: Dict[str, Grasp] = {
    "cube": Grasp(offset=(0.0, 0.0, 0.008), yaw_period_deg=90.0),  # centre is 2.5 cm above the table
    "apple": Grasp(offset=(0.0, 0.0, 0.0)),  # centre is 3.0 cm above the table
}

LIFT_HEIGHT = 0.10  # m above the grasp / place pose
CONTROL_DT = 0.01  # s; servo targets update at 100 Hz
LINE_STEP = 0.01  # m between continuation-IK waypoints on straight lines
TRACKING_TOL = 0.01  # m; warn if the simulated TCP ends a move further than this from its target


def grasp_rotation(tilt_deg: float = 0.0, yaw_deg: float = 0.0) -> NDArray[np.float64]:
    """TCP orientation: z = approach direction, y = finger closing axis."""
    t = np.radians(tilt_deg)
    z = np.array([np.sin(t), 0.0, -np.cos(t)])
    y = np.array([0.0, -1.0, 0.0])
    R = np.column_stack([np.cross(y, z), y, z])
    return Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix() @ R


def _yaw_of(R: NDArray[np.float64]) -> float:
    """Yaw (deg) of a TCP rotation's finger closing axis, as in :func:`grasp_rotation`."""
    y = R[:, 1]
    return float(np.degrees(np.arctan2(-y[0], -y[1])))


def _min_jerk(s: float) -> float:
    return 10 * s**3 - 15 * s**4 + 6 * s**5


class SkillRunner:
    """Plans and executes motion primitives and skills on a :class:`TabletopScene`.

    Args:
        scene: The simulation scene, assumed reset with arms at home.
        on_tick: Called after every control tick, e.g. to record video frames.
    """

    def __init__(self, scene: TabletopScene, on_tick: Optional[Callable[[TabletopScene], None]] = None) -> None:
        self.scene = scene
        self.on_tick = on_tick
        self.ik = BimanualIK()
        self.q: Dict[Side, NDArray[np.float64]] = {side: scene.arm_qpos(side) for side in SIDES}  # commanded
        self.objects: Dict[str, NDArray[np.float64]] = {}
        self.object_yaw: Dict[str, float] = {}
        self.held: Dict[Side, Optional[Tuple[str, NDArray[np.float64]]]] = {side: None for side in SIDES}
        self._substeps = round(CONTROL_DT / scene.model.opt.timestep)

    # ------------------------------------------------------------ perception

    def perceive(self) -> Dict[str, NDArray[np.float64]]:
        """Observes and remembers all object positions and yaws (once, not continuously)."""
        for name in OBJECT_NAMES:
            pos, quat = self.scene.object_pose(name)
            self.objects[name] = pos
            self.object_yaw[name] = float(Rotation.from_quat(quat, scalar_first=True).as_euler("xyz", degrees=True)[2])
        return self.objects

    # -------------------------------------------------------------- planning

    def tcp_pose(self, side: Side, q: Optional[NDArray[np.float64]] = None) -> Pose:
        """TCP pose (IK model) for ``side`` at joint config ``q`` (default: commanded)."""
        self._load(side, self.q[side] if q is None else q)
        return self.ik.tcp_pose(side)

    def _load(self, side: Side, q_side: NDArray[np.float64]) -> None:
        other = SIDES[1 - SIDES.index(side)]
        self.ik.set_arm_q(other, self.q[other])
        self.ik.set_arm_q(side, q_side)

    def _solve(
        self,
        side: Side,
        pose: Pose,
        seed: NDArray[np.float64],
        free_yaw: bool = False,
        multistart: bool = True,
    ) -> NDArray[np.float64]:
        """IK for one arm (other arm held at its commanded config); returns the arm's joints."""
        other = SIDES[1 - SIDES.index(side)]
        self._load(side, seed)
        self.ik.set_free_yaw(side, free_yaw)
        targets = {side: pose, other: self.ik.tcp_pose(other)}
        if multistart:
            converged, err = self.ik.solve_multistart(side, targets)
        else:
            converged, err = self.ik.solve(targets)
        self.ik.set_free_yaw(side, False)
        if not converged:
            raise RuntimeError(f"{side} arm cannot reach {np.round(pose[0], 3)} (residual {err * 1000:.1f} mm)")
        return self.ik.arm_q(side)

    def _plan_line(self, side: Side, q_start: NDArray[np.float64], goal: Pose, free_yaw: bool = False) -> JointPath:
        """Joint path for a straight TCP line from the pose at ``q_start`` to ``goal``.

        With ``free_yaw`` the TCP may spin about its approach axis along the way."""
        p0, R0 = self.tcp_pose(side, q_start)
        p1, R1 = goal
        slerp = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([R0, R1])))
        n = max(2, int(np.ceil(np.linalg.norm(p1 - p0) / LINE_STEP)))
        path, q = [q_start], q_start
        for k in range(1, n + 1):
            s = k / n
            q = self._solve(side, (p0 + s * (p1 - p0), slerp(s).as_matrix()), q, free_yaw, multistart=False)
            path.append(q)
        return path

    # ------------------------------------------------------------- execution

    def _tick(self) -> None:
        for side in SIDES:
            self.scene.set_arm_target(side, self.q[side])
        self.scene.step(self._substeps)
        if self.on_tick is not None:
            self.on_tick(self.scene)

    def hold(self, duration: float) -> None:
        for _ in range(max(1, round(duration / CONTROL_DT))):
            self._tick()

    def set_gripper(self, side: Side, open_fraction: float, duration: float = 0.6) -> None:
        self.scene.set_gripper(side, open_fraction)
        self.hold(duration)

    def follow(self, paths: Dict[Side, JointPath], duration: float) -> None:
        """Executes joint paths (piecewise linear, min-jerk timing along the path)."""
        n = max(1, round(duration / CONTROL_DT))
        for k in range(1, n + 1):
            s = _min_jerk(k / n)
            for side, path in paths.items():
                x = s * (len(path) - 1)
                i = min(int(x), len(path) - 2)
                self.q[side] = path[i] + (x - i) * (path[i + 1] - path[i])
            self._tick()
        for side, path in paths.items():
            self._check_tracking(side, self.tcp_pose(side, path[-1])[0])

    def _check_tracking(self, side: Side, p: NDArray[np.float64]) -> None:
        err = np.linalg.norm(self.scene.tcp_pose(side)[0] - p)
        if err > TRACKING_TOL:
            logger.warning("%s TCP is %.1f mm off target after move", side, err * 1e3)

    # ------------------------------------------------------------ primitives

    def move_joint(self, side: Side, q_goal: NDArray[np.float64], duration: float) -> None:
        self.follow({side: [self.q[side], q_goal]}, duration)

    def move_linear(self, side: Side, goal: Pose, duration: float) -> None:
        self.follow({side: self._plan_line(side, self.q[side], goal)}, duration)

    def go_home(self, sides: Tuple[Side, ...] = SIDES, duration: float = 2.0) -> None:
        self.follow({side: [self.q[side], HOME_ARM_QPOS[side]] for side in sides}, duration)

    # ---------------------------------------------------------------- skills

    def _plan_grasp(self, side: Side, name: str) -> Tuple[Pose, NDArray[np.float64]]:
        """Grasp TCP pose for ``name`` and an arm configuration reaching it.

        Solves with yaw free to find the arm's natural yaw, then tries the yaws the
        object's symmetry allows, nearest to the natural one first.
        """
        g = GRASPS[name]
        p = self.objects[name] + np.array(g.offset)
        q_free = self._solve(side, (p, grasp_rotation(g.tilt_deg)), self.q[side], free_yaw=True)
        R_free = self.tcp_pose(side, q_free)[1]
        if g.yaw_period_deg is None:
            return (p, R_free), q_free
        natural = _yaw_of(R_free)
        base = self.object_yaw[name] + g.yaw_deg
        k0 = round((natural - base) / g.yaw_period_deg)
        half = round(180.0 / g.yaw_period_deg)
        yaws = sorted((base + g.yaw_period_deg * k for k in range(k0 - half, k0 + half + 1)), key=lambda y: abs(y - natural))
        for yaw in yaws:
            R = grasp_rotation(g.tilt_deg, yaw)
            try:
                q = self._solve(side, (p, R), q_free)
            except RuntimeError:
                continue
            logger.info("%s grasp yaw %.0f deg (natural %.0f)", name, yaw, natural)
            return (p, R), q
        raise RuntimeError(f"{side} arm has no reachable grasp yaw for {name}")

    def pick(self, side: Side, name: str) -> bool:
        """Grasps ``name`` at its remembered pose and lifts it. Returns True if held."""
        (p, R), q_grasp = self._plan_grasp(side, name)
        approach = R[:, 2]
        # Plan backwards from the grasp so approach, grasp and lift share an IK branch.
        retreat = self._plan_line(side, q_grasp, (p - GRASPS[name].pregrasp * approach, R))
        lift = self._plan_line(side, q_grasp, (p + [0.0, 0.0, LIFT_HEIGHT], R))

        self.set_gripper(side, 1.0, duration=0.2)
        self.move_joint(side, retreat[-1], duration=2.0)
        self.follow({side: retreat[::-1]}, duration=1.2)
        self.set_gripper(side, 0.0, duration=0.6)
        self.follow({side: lift}, duration=1.2)

        # Fingers stopped by the object, not fully closed.
        held = self.scene.gripper_opening(side) > 0.002
        if held:
            # Remember where the object sits relative to the TCP for placing.
            self.held[side] = (name, self.objects[name] - p)
        return held

    def place(self, side: Side, xy: NDArray[np.float64], surface_z: float) -> None:
        """Puts the held object down with its origin above ``xy`` on a surface at ``surface_z``.

        The TCP keeps pointing down but may yaw freely, so the object's final yaw is
        whatever the arm reaches most naturally.
        """
        if self.held[side] is None:
            raise RuntimeError(f"{side} arm is not holding anything")
        name, obj_from_tcp = self.held[side]
        R = self.tcp_pose(side)[1]
        rest_height = self.objects[name][2] - TABLE_HEIGHT  # object origin height when resting
        obj_target = np.array([xy[0], xy[1], surface_z + rest_height + 0.005])
        p = obj_target - obj_from_tcp

        q_place = self._solve(side, (p, R), self.q[side], free_yaw=True)
        R = self.tcp_pose(side, q_place)[1]
        up = self._plan_line(side, q_place, (p + [0.0, 0.0, LIFT_HEIGHT], R), free_yaw=True)

        self.move_joint(side, up[-1], duration=1.8)
        self.follow({side: up[::-1]}, duration=1.2)
        self.set_gripper(side, 1.0, duration=0.5)
        self.follow({side: up}, duration=1.0)
        self.held[side] = None
        self.objects[name] = obj_target  # memory update: we know where we put it
