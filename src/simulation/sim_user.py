"""
A simulated person for the handover in ``OpenArmSim``, plus the backend wrapper that drives it.

At the handover pose the robot waits for an external contact it can *feel* in its joint
torques. In simulation nobody is there, so :class:`SimulatedUser` plays the person:

1. A can that stays within ``radius_m`` of the handover pose for ``grasp_after_s`` (sim time)
   gets the user's pull: a constant world force ``pull_force_n`` on the can body
   (``xfrc_applied``). The robot only sees the resulting change in its joint torques.
2. Once the gripper has opened, the user "takes" the can: it is moved to ``taken_xyz_world``
   on the table and stops being pulled.

:class:`SimUserBackend` wraps any ``RobotBackend`` and runs :meth:`SimulatedUser.update`
before each ``advance()``, so the same code works with the real-time and the stepped simulator.
It forwards everything else unchanged, and it never touches the robot's commands.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import mujoco
import numpy as np
from numpy.typing import NDArray


class SimulatedUser:
    """Takes the object offered at the handover pose, by pulling on it."""

    def __init__(
        self,
        sim: Any,
        handover_xyz_world: Sequence[float],
        can_bodies: Sequence[str],
        grasp_after_s: Optional[float],
        pull_force_n: Sequence[float],
        taken_xyz_world: Sequence[float],
        gripper_side: str = "left",
        radius_m: float = 0.08,
    ) -> None:
        """Args:
            sim: ``OpenArmSim`` (its ``model``, ``data`` and lock are used).
            handover_xyz_world: Handover TCP position in the scene's world frame.
            can_bodies: Names of the objects that can be taken.
            grasp_after_s: Delay before the pull (sim time); None = nobody ever takes it.
            pull_force_n: World force on the can while pulling, N.
            taken_xyz_world: Where a taken can is put (on the table, out of the way).
            gripper_side: Arm whose finger opening tells that the gripper has released.
            radius_m: A can within this distance of the handover pose counts as offered.
        """
        self.sim = sim
        self.handover = np.asarray(handover_xyz_world, dtype=np.float64)
        self.bodies = {name: sim.model.body(name).id for name in can_bodies}
        self.qadr = {name: int(sim.model.jnt_qposadr[sim.model.body_jntadr[bid]]) for name, bid in self.bodies.items()}
        self.vadr = {name: int(sim.model.jnt_dofadr[sim.model.body_jntadr[bid]]) for name, bid in self.bodies.items()}
        self.grasp_after_s = grasp_after_s
        self.force = np.asarray(pull_force_n, dtype=np.float64)
        self.taken_xyz = np.asarray(taken_xyz_world, dtype=np.float64)
        self.finger_qadr = int(sim.model.jnt_qposadr[sim.model.joint(f"openarm_{gripper_side}_finger_joint1").id])
        self.radius = radius_m
        self._near_since: Optional[float] = None
        self._pulling: Optional[str] = None
        self._grip_at_pull = 0.0
        self.taken: list = []

    def _lock(self) -> Any:
        return getattr(self.sim, "_lock")

    def update(self) -> None:
        """One check: start pulling, keep pulling, or take the can once released."""
        if self.grasp_after_s is None:
            return
        with self._lock():
            d = self.sim.data
            t = float(d.time)
            if self._pulling is None:
                near = [n for n, b in self.bodies.items()
                        if n not in self.taken and np.linalg.norm(d.xpos[b] - self.handover) < self.radius]
                if not near:
                    self._near_since = None
                    return
                self._near_since = t if self._near_since is None else self._near_since
                if t - self._near_since >= self.grasp_after_s:
                    self._pulling = near[0]
                    self._grip_at_pull = float(d.qpos[self.finger_qadr])
            if self._pulling is not None:
                name = self._pulling
                if float(d.qpos[self.finger_qadr]) > self._grip_at_pull + 0.005:  # fingers opened: take it
                    d.xfrc_applied[self.bodies[name]] = 0.0
                    q = self.qadr[name]
                    d.qpos[q:q + 3] = self.taken_xyz
                    d.qpos[q + 3:q + 7] = [1.0, 0.0, 0.0, 0.0]
                    d.qvel[self.vadr[name]:self.vadr[name] + 6] = 0.0
                    mujoco.mj_forward(self.sim.model, d)
                    self.taken.append(name)
                    self._pulling = None
                    self._near_since = None
                else:
                    d.xfrc_applied[self.bodies[name], :3] = self.force


class SimUserBackend:
    """``RobotBackend`` wrapper: runs the simulated user before every ``advance()``."""

    def __init__(self, robot: Any, user: SimulatedUser) -> None:
        """Args:
            robot: The simulated backend.
            user: The simulated person.
        """
        self._robot = robot
        self.user = user

    def advance(self, dt: float) -> None:
        """Lets the user act, then advances the robot."""
        self.user.update()
        self._robot.advance(dt)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._robot, name)


class SimObjectDetector:
    """Stand-in for the vision pipeline in simulation: reports where the cans really are.

    Like the vision contract, it returns base-frame table positions of the objects standing
    on the table (not held, not in a bin, not taken). The sim "cheats" on perception only;
    which can is heavier is still unknown to the robot.
    """

    def __init__(self, sim: Any, names: Sequence[str], base_z_in_world_m: float, table_top_world_m: float,
                 label: str = "soda can") -> None:
        """Args:
            sim: ``OpenArmSim`` with the demo scene.
            names: Can bodies.
            base_z_in_world_m: Height of the robot base frame in the scene.
            table_top_world_m: Tabletop height in the scene.
            label: Label of the returned detections.
        """
        self.sim, self.names, self.label = sim, list(names), label
        self.base_z, self.table_z = base_z_in_world_m, table_top_world_m

    def detect(self, prompt: str) -> list:
        """Cans standing on the tabletop, as base-frame ``ObjectDetection``s (order = scene order)."""
        from src.task.interfaces import ObjectDetection

        with getattr(self.sim, "_lock"):
            out = []
            for n in self.names:
                bid = self.sim.model.body(n).id
                p = self.sim.data.xpos[bid]
                upright = self.sim.data.xmat[bid][8] > 0.95
                on_table = abs(p[2] - self.table_z - 0.06) < 0.01
                if upright and on_table:
                    out.append(ObjectDetection(label=self.label, x=float(p[0]), y=float(p[1]), confidence=0.99))
            return out


def can_positions_world(sim: Any, names: Sequence[str]) -> NDArray[np.float64]:
    """(n, 3) world positions of the named bodies."""
    return np.array([sim.data.xpos[sim.model.body(n).id] for n in names])
