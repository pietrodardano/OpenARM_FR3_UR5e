"""
Differential inverse kinematics for both OpenArm TCPs (mink on the official model).

Each :meth:`BimanualIK.solve` call starts from a seed configuration (normally the
previous solution) and runs a few QP-based differential-IK iterations with the
official joint limits as hard constraints. A weak posture task towards the
official ``home`` keyframe resolves the 7-DOF redundancy consistently, and the
solver damping keeps consecutive solutions close to each other, which is what a
policy running at 10-30 Hz needs: small, continuous joint-space steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import mink
import numpy as np
from numpy.typing import NDArray

from ..control.robot_interface import SIDES, Side
from .openarm_kinematics import OpenArmKinematics, Pose, tcp_site_name


@dataclass
class IKResult:
    q_arm: Dict[Side, NDArray[np.float64]]  # (7,) per side
    pos_error: Dict[Side, float]  # m
    rot_error: Dict[Side, float]  # rad
    converged: bool


class BimanualIK:
    """TCP pose IK for one or both arms.

    Args:
        kinematics: Official model wrapper (shared with the controller).
        position_cost / orientation_cost: Task weights of the TCP frame tasks.
        posture_cost: Weight of the pull towards the home posture (redundancy).
        damping: Levenberg-Marquardt style damping (keeps steps small/continuous).
        dt: Integration step of one differential-IK iteration (s).
        max_iters: Iteration cap per solve.
        pos_tol / rot_tol: Convergence tolerances (m / rad).
    """

    def __init__(
        self,
        kinematics: OpenArmKinematics,
        position_cost: float = 1.0,
        orientation_cost: float = 0.5,
        posture_cost: float = 1e-3,
        damping: float = 1e-3,
        dt: float = 0.05,
        max_iters: int = 50,
        pos_tol: float = 1e-3,
        rot_tol: float = np.radians(1.0),
        solver: str = "daqp",
    ) -> None:
        self.kin = kinematics
        self.model = kinematics.model
        self.config = mink.Configuration(self.model)
        self.damping = damping
        self.dt = dt
        self.max_iters = max_iters
        self.pos_tol = pos_tol
        self.rot_tol = rot_tol
        self.solver = solver
        self.orientation_cost = orientation_cost
        self._position_only = {side: False for side in SIDES}
        self.tcp_tasks = {
            side: mink.FrameTask(
                tcp_site_name(side),
                "site",
                position_cost=position_cost,
                orientation_cost=orientation_cost,
                lm_damping=1e-2,
            )
            for side in SIDES
        }
        q_home = self.config.q.copy()
        for side in SIDES:
            q_home[self.kin.qpos_adr[side]] = self.kin.home_qpos[side]
        self.posture = mink.PostureTask(self.model, cost=posture_cost)
        self.posture.set_target(q_home)
        self.config.update(q_home)
        self.limits = [mink.ConfigurationLimit(self.model)]

    def set_position_only(self, side: Side, position_only: bool) -> None:
        """Ignore TCP orientation for ``side`` (e.g. for coarse reaching)."""
        self._position_only[side] = position_only
        self.tcp_tasks[side].set_orientation_cost(0.0 if position_only else self.orientation_cost)

    def solve(
        self,
        targets: Mapping[Side, Pose],
        q_seed: Mapping[Side, NDArray[np.float64]],
        max_iters: Optional[int] = None,
    ) -> IKResult:
        """Solves for the arms in ``targets``; other arms stay at their seed."""
        q = self.config.q.copy()
        for side, q_side in q_seed.items():
            q[self.kin.qpos_adr[side]] = q_side
        self.config.update(q)
        for side, (p, R) in targets.items():
            self.tcp_tasks[side].set_target(
                mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(np.asarray(R)), np.asarray(p))
            )
        tasks = [self.tcp_tasks[s] for s in targets] + [self.posture]

        pos_err: Dict[Side, float] = {}
        rot_err: Dict[Side, float] = {}
        converged = False
        for _ in range(max_iters or self.max_iters):
            v = mink.solve_ik(self.config, tasks, self.dt, solver=self.solver, damping=self.damping, limits=self.limits)
            self.config.integrate_inplace(v, self.dt)
            for side in targets:
                e = self.tcp_tasks[side].compute_error(self.config)
                pos_err[side] = float(np.linalg.norm(e[:3]))
                rot_err[side] = 0.0 if self._position_only[side] else float(np.linalg.norm(e[3:]))
            if all(pos_err[s] < self.pos_tol and rot_err[s] < self.rot_tol for s in targets):
                converged = True
                break
        q_out = {side: self.config.q[self.kin.qpos_adr[side]].copy() for side in SIDES}
        return IKResult(q_arm=q_out, pos_error=pos_err, rot_error=rot_err, converged=converged)
