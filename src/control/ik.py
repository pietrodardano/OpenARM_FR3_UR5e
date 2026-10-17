"""Differential IK for both OpenArm TCPs (mink, robot-only model).

The IK model is the upstream OpenArm v1 bimanual MJCF without table or objects,
with the same ``<side>_tcp`` sites as the simulation scene. Joint names match
the scene, so solutions map one-to-one onto the scene's arm actuators.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import mink
import numpy as np
from numpy.typing import NDArray

from src.simulation.tabletop_scene import HOME_ARM_QPOS, MIRROR, SIDES, Side, build_robot_model

Pose = Tuple[NDArray[np.float64], NDArray[np.float64]]  # (position (3,), rotation matrix (3, 3))

# Forearms-forward pose: a good generic IK seed for reaching over the table.
_READY_LEFT = np.array([0.0, 0.0, 0.0, np.pi / 2, 0.0, 0.0, 0.0])
READY_ARM_QPOS: Dict[str, NDArray[np.float64]] = {"left": _READY_LEFT, "right": MIRROR * _READY_LEFT}


class BimanualIK:
    """Tracks a TCP pose target per arm; both arm tasks are always active, so an
    arm whose target does not change holds its pose.

    With ``set_free_yaw(side, True)`` rotation about that TCP's approach axis
    (local z) is unconstrained, which greatly enlarges the reachable set for
    top-down poses; the arm settles at its natural yaw.
    """

    def __init__(self, max_joint_vel: float = 2.0, posture_cost: float = 1e-3) -> None:
        self.model = build_robot_model()
        self.config = mink.Configuration(self.model)
        self._arm_qpos_adr = {
            side: np.array([self.model.jnt_qposadr[self.model.joint(f"openarm_{side}_joint{i}").id] for i in range(1, 8)])
            for side in SIDES
        }
        self.tcp_tasks = {
            side: mink.FrameTask(f"{side}_tcp", "site", position_cost=1.0, orientation_cost=0.5, lm_damping=1e-2)
            for side in SIDES
        }
        self._free_yaw = {side: False for side in SIDES}
        # Weak pull towards the mid-range ready pose resolves the 7-DOF redundancy
        # consistently and keeps solutions away from joint limits.
        self.posture = mink.PostureTask(self.model, cost=posture_cost)
        q_ready = self.config.q.copy()
        for side in SIDES:
            q_ready[self._arm_qpos_adr[side]] = READY_ARM_QPOS[side]
        self.posture.set_target(q_ready)
        for side in SIDES:
            self.set_arm_q(side, HOME_ARM_QPOS[side])
        self.limits = [
            mink.ConfigurationLimit(self.model),
            mink.VelocityLimit(
                self.model, {f"openarm_{side}_joint{i}": max_joint_vel for side in SIDES for i in range(1, 8)}
            ),
        ]

    # ------------------------------------------------------------------ state

    def arm_q(self, side: Side) -> NDArray[np.float64]:
        return self.config.q[self._arm_qpos_adr[side]].copy()

    def set_arm_q(self, side: Side, q: NDArray[np.float64]) -> None:
        full = self.config.q.copy()
        full[self._arm_qpos_adr[side]] = q
        self.config.update(full)

    def tcp_pose(self, side: Side) -> Pose:
        T = self.config.get_transform_frame_to_world(f"{side}_tcp", "site")
        return T.translation().copy(), T.rotation().as_matrix().copy()

    def set_free_yaw(self, side: Side, free: bool) -> None:
        self._free_yaw[side] = free
        self.tcp_tasks[side].set_orientation_cost([0.5, 0.5, 0.0] if free else 0.5)

    def tcp_error(self, side: Side) -> Tuple[float, float]:
        """(position error m, orientation error rad) of ``side``'s task, ignoring yaw if free."""
        e = self.tcp_tasks[side].compute_error(self.config)
        rot = e[3:5] if self._free_yaw[side] else e[3:]
        return float(np.linalg.norm(e[:3])), float(np.linalg.norm(rot))

    # ---------------------------------------------------------------- solving

    def step(self, targets: Dict[Side, Pose], dt: float) -> None:
        """One differential-IK step of length ``dt`` towards ``targets``."""
        for side, (p, R) in targets.items():
            self.tcp_tasks[side].set_target(mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(R), p))
        tasks = [*self.tcp_tasks.values(), self.posture]
        v = mink.solve_ik(self.config, tasks, dt, solver="daqp", damping=1e-3, limits=self.limits)
        self.config.integrate_inplace(v, dt)

    def solve_multistart(
        self,
        side: Side,
        targets: Dict[Side, Pose],
        seeds: Optional[List[NDArray[np.float64]]] = None,
        n_random: int = 8,
    ) -> Tuple[bool, float]:
        """Like :meth:`solve`, but if starting from the current configuration fails,
        retries ``side`` from ``seeds``, the ready pose and ``n_random`` fixed random
        seeds within joint limits. Leaves the best solution in the configuration."""
        q_start = self.config.q.copy()
        jids = [self.model.joint(f"openarm_{side}_joint{i}").id for i in range(1, 8)]
        lo, hi = self.model.jnt_range[jids].T
        rng = np.random.default_rng(0)
        candidates = [None, *(seeds or []), READY_ARM_QPOS[side], *rng.uniform(lo, hi, size=(n_random, 7))]
        best: Tuple[float, Optional[NDArray[np.float64]]] = (np.inf, None)
        for seed in candidates:
            self.config.update(q_start)
            if seed is not None:
                self.set_arm_q(side, seed)
            converged, err = self.solve(targets)
            if converged:
                return True, err
            if err < best[0]:
                best = (err, self.config.q.copy())
        self.config.update(best[1])
        return False, best[0]

    def solve(
        self, targets: Dict[Side, Pose], dt: float = 0.02, max_iters: int = 500, pos_tol: float = 1e-3
    ) -> Tuple[bool, float]:
        """Iterates :meth:`step` until every TCP is within ``pos_tol`` (m) and ~1 deg.

        Returns (converged, worst position error in m).
        """
        err = np.inf
        for _ in range(max_iters):
            self.step(targets, dt)
            errs = [self.tcp_error(side) for side in targets]
            err = max(e[0] for e in errs)
            if err < pos_tol and max(e[1] for e in errs) < np.radians(1.0):
                return True, err
        return False, err
