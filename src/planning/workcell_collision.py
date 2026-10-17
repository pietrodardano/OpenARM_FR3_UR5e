"""
Geometric collision check of one OpenArm arm against the static workcell.

Loads a MuJoCo workcell scene (the demo scene: official OpenArm v1 arms, measured torso,
72.5 cm table, bins) and, for a given arm configuration, reports any contact closer than
``clearance_m`` between the moving arm's collision geoms and:

- static geometry: table, torso profile, camera bracket, bins (bodies without degrees of freedom);
- the other arm (parked or moving), including its gripper.

Free bodies (the cans) are ignored: the arm is meant to touch them. Finger-finger contacts of
the same gripper are ignored too. The check is kinematic: it runs ``mj_kinematics`` and
``mj_collision`` on a private ``MjData`` and never touches a running simulator or robot.
"""

from __future__ import annotations

from typing import List, Mapping, Optional, Set

import mujoco
import numpy as np
from numpy.typing import NDArray

ARM_DOF = 7


class WorkcellCollision:
    """Clearance check of ``side``'s arm geoms against static geometry and the other arm."""

    def __init__(self, scene_path: str, side: str, clearance_m: float = 0.01) -> None:
        """Args:
            scene_path: MuJoCo scene containing the official arms and the workcell.
            side: Moving arm, ``"left"`` or ``"right"``.
            clearance_m: Minimum allowed distance between the arm and an obstacle, m.
        """
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        self.side = side
        self.clearance = clearance_m
        m = self.model
        self._qadr = {s: [int(m.jnt_qposadr[m.joint(f"openarm_{s}_joint{i}").id]) for i in range(1, ARM_DOF + 1)]
                      for s in ("left", "right")}
        self._gadr = {s: [int(m.jnt_qposadr[m.joint(f"openarm_{s}_finger_joint{k}").id]) for k in (1, 2)]
                      for s in ("left", "right")}
        other = "right" if side == "left" else "left"
        self.arm_geoms: Set[int] = set()
        self.obstacle_geoms: Set[int] = set()
        self._finger_bodies = {m.body(f"openarm_{side}_{f}").id for f in ("left_finger", "right_finger")}
        for g in range(m.ngeom):
            if not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue  # visual only
            body = m.geom_bodyid[g]
            name = m.body(body).name
            if name.startswith(f"openarm_{side}_"):
                self.arm_geoms.add(g)
            elif name.startswith(f"openarm_{other}_"):
                self.obstacle_geoms.add(g)
            elif self._is_static(body):
                self.obstacle_geoms.add(g)
        # Inflate the arm geoms so contacts within the clearance are reported.
        for g in self.arm_geoms:
            m.geom_margin[g] = max(m.geom_margin[g], clearance_m)

    def _is_static(self, body: int) -> bool:
        """True for bodies welded to the world (no degrees of freedom up the tree)."""
        m = self.model
        while body != 0:
            if m.body_dofnum[body] > 0:
                return False
            body = m.body_parentid[body]
        return True

    def contacts(self, q_arms: Mapping[str, NDArray[np.float64]], grippers_m: Optional[Mapping[str, float]] = None
                 ) -> List[str]:
        """Collisions at ``q_arms`` (per-side (7,) joints), as ``"arm_body - obstacle_body (dist mm)"``."""
        m, d = self.model, self.data
        for s, q in q_arms.items():
            d.qpos[self._qadr[s]] = q
        for s in ("left", "right"):
            g = (grippers_m or {}).get(s, 0.044 if s == self.side else 0.0)
            d.qpos[self._gadr[s]] = g
        mujoco.mj_fwdPosition(m, d)  # kinematics + collision detection
        hits = []
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            arm, obs = (g1, g2) if g1 in self.arm_geoms else (g2, g1)
            if arm not in self.arm_geoms or obs not in self.obstacle_geoms or c.dist >= self.clearance:
                continue
            hits.append(f"{m.body(m.geom_bodyid[arm]).name} - {m.body(m.geom_bodyid[obs]).name} "
                        f"({c.dist * 1e3:.0f} mm)")
        return hits
