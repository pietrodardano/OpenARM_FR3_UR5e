"""
Measured workcell geometry, in the robot base frame used by the controller.

Two frames meet here:

- **World** (MuJoCo workcell scenes, physical_workcell.md): origin on the floor under the
  torso. The arm root (``openarm_body_link0``) sits at ``z = arm_mount_height - 0.697``,
  because the official model's ``arm_origin`` (shoulder axis) is 0.697 m above that root.
- **Base** (official model, ``IKController``, every ``demo:`` pose): origin at
  ``openarm_body_link0``, so ``p_world = p_base + (0, 0, base_z_in_world)``.

Everything is derived from the ``workcell`` config section (the measured table height and
arm mount height), so the demo, the table guard, the scene generator and the twin agree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

OPENARM_ARM_ORIGIN_ABOVE_ROOT_M = 0.697


@dataclass(frozen=True)
class WorkcellGeometry:
    """Table box and frame offset, base frame unless stated otherwise."""

    base_z_in_world_m: float
    table_top_z_m: float  # base frame
    table_rear_x_m: float
    table_front_x_m: float
    table_half_width_m: float
    table_top_world_m: float

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "WorkcellGeometry":
        """Builds the geometry from the ``workcell`` section of task_config.yaml."""
        w = cfg["workcell"]
        t = w["table"]
        base_z = float(w["arm_mount_height_m"]) - OPENARM_ARM_ORIGIN_ABOVE_ROOT_M
        top_world = float(t["top_height_m"])
        cx, length = float(t["center_x_m"]), float(t["length_m"])
        return cls(
            base_z_in_world_m=base_z,
            table_top_z_m=top_world - base_z,
            table_rear_x_m=cx - length / 2.0,
            table_front_x_m=cx + length / 2.0,
            table_half_width_m=float(t["width_m"]) / 2.0,
            table_top_world_m=top_world,
        )

    def to_world(self, p_base: NDArray[np.float64]) -> NDArray[np.float64]:
        """Base-frame point -> world (scene) point."""
        return np.asarray(p_base, dtype=np.float64) + np.array([0.0, 0.0, self.base_z_in_world_m])

    def to_base(self, p_world: NDArray[np.float64]) -> NDArray[np.float64]:
        """World (scene) point -> base-frame point."""
        return np.asarray(p_world, dtype=np.float64) - np.array([0.0, 0.0, self.base_z_in_world_m])

    def below_table_surface(self, p_base: NDArray[np.float64], margin_m: float) -> bool:
        """True if ``p_base`` is over the table footprint and less than ``margin_m`` above its top."""
        x, y, z = (float(v) for v in p_base)
        over = self.table_rear_x_m - margin_m <= x <= self.table_front_x_m + margin_m \
            and abs(y) <= self.table_half_width_m + margin_m
        return over and z < self.table_top_z_m + margin_m
