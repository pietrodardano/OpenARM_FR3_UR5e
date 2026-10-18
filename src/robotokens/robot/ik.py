"""Differential IK of the TCP (mink on the official model, daqp QP solver).

Each :meth:`IK.solve` starts from a seed (normally the previous solution) and runs QP-based
differential-IK iterations with the official joint limits as hard constraints. A weak posture
task towards the home pose resolves the 7-DOF redundancy consistently; damping keeps
consecutive solutions close, which is what straight-line moves need.
"""

from __future__ import annotations

from dataclasses import dataclass

import mink
import numpy as np
from numpy.typing import NDArray

from .model import ArmModel


@dataclass
class IKResult:
    q: NDArray  # (7,)
    pos_error: float  # m
    rot_error: float  # rad
    converged: bool


class IK:
    def __init__(self, model: ArmModel, position_cost: float = 1.0, orientation_cost: float = 0.5,
                 posture_cost: float = 1e-3, damping: float = 1e-3, dt: float = 0.05,
                 pos_tol: float = 1e-3, rot_tol: float = np.radians(1.0)) -> None:
        self.m = model
        self.config = mink.Configuration(model.model)
        self.damping, self.dt, self.pos_tol, self.rot_tol = damping, dt, pos_tol, rot_tol
        self.task = mink.FrameTask(f"{model.side}_tcp", "site", position_cost=position_cost,
                                   orientation_cost=orientation_cost, lm_damping=1e-2)
        q_home = self.config.q.copy()
        q_home[model.qadr] = model.home
        self.posture = mink.PostureTask(model.model, cost=posture_cost)
        self.posture.set_target(q_home)
        self.config.update(q_home)
        self.limits = [mink.ConfigurationLimit(model.model)]

    def solve(self, p: NDArray, R: NDArray, q_seed: NDArray, max_iters: int = 50) -> IKResult:
        q = self.config.q.copy()
        q[self.m.qadr] = q_seed
        self.config.update(q)
        self.task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(np.asarray(R)), np.asarray(p)))
        pos_err = rot_err = float("inf")
        converged = False
        for _ in range(max_iters):
            v = mink.solve_ik(self.config, [self.task, self.posture], self.dt, solver="daqp",
                              damping=self.damping, limits=self.limits)
            self.config.integrate_inplace(v, self.dt)
            e = self.task.compute_error(self.config)
            pos_err, rot_err = float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:]))
            if pos_err < self.pos_tol and rot_err < self.rot_tol:
                converged = True
                break
        return IKResult(q=self.config.q[self.m.qadr].copy(), pos_error=pos_err, rot_error=rot_err, converged=converged)
