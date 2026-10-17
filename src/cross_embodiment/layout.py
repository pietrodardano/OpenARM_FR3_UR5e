"""Per-station layout of the simulated cross-embodiment demo, read from ``configs/task_config.yaml``.

Both the scene generator and the controllers read the same ``cross_embodiment_demo`` block, so the
object positions, bins and hand-over poses in the rendered scene are the ones the robots drive to.
The sim robots may "cheat": object positions are given directly, there is no vision.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "task_config.yaml"
SCENE_PATH = PROJECT_ROOT / "third_party" / "mujoco_menagerie" / "cross_embodiment_openarm_ur5e_fr3.xml"


@dataclass(frozen=True)
class CanSpec:
    radius_m: float
    half_height_m: float
    empty_kg: float
    full_kg: float
    friction: float


@dataclass(frozen=True)
class TableSpec:
    center_x_m: float
    half_x_m: float
    half_y_m: float
    top_z_m: float


@dataclass(frozen=True)
class GraspSpec:
    approach_clearance_m: float
    lift_height_m: float
    max_joint_speed_rad_s: float
    settle_s: float
    weigh_window_s: float


@dataclass(frozen=True)
class StationSpec:
    name: str
    arm: str  # "left" or "right": the working arm; the other one is parked
    y0_m: float
    cans_xy_m: tuple[tuple[float, float], tuple[float, float]]
    full_can: int  # 1 or 2
    bin_heavy_xy_m: tuple[float, float]
    bin_light_xy_m: tuple[float, float]
    handover_xyz_m: tuple[float, float, float]
    ending: str  # "sort" or "handover"

    def true_mass_kg(self, can_index: int, can: CanSpec) -> float:
        """Ground-truth mass of can 1 or 2 (the sim knows it; the robot must estimate it)."""
        return can.full_kg if can_index == self.full_can else can.empty_kg


@dataclass(frozen=True)
class DemoLayout:
    sim_robots: tuple[str, ...]
    control_dt_s: float
    can: CanSpec
    table: TableSpec
    grasp: GraspSpec
    stations: Mapping[str, StationSpec]

    def can_rest_z_m(self) -> float:
        """World z of a can's centre while it stands on the table."""
        return self.table.top_z_m + self.can.half_height_m


def load_layout(config_path: Path = CONFIG_PATH) -> DemoLayout:
    """Parse the ``cross_embodiment_demo`` block; raise ``KeyError`` with the missing key if absent."""
    with open(config_path) as handle:
        block: Mapping[str, Any] = yaml.safe_load(handle)["cross_embodiment_demo"]
    stations = {}
    for name, s in block["stations"].items():
        if s["ending"] not in {"sort", "handover"}:
            raise ValueError(f"{name}: ending must be 'sort' or 'handover', got {s['ending']!r}")
        if s.get("arm", "left") not in {"left", "right"}:
            raise ValueError(f"{name}: arm must be 'left' or 'right'")
        if int(s["full_can"]) not in (1, 2):
            raise ValueError(f"{name}: full_can must be 1 or 2")
        stations[name] = StationSpec(
            name=name,
            arm=s.get("arm", "left"),
            y0_m=float(s["y0_m"]),
            cans_xy_m=tuple(tuple(float(v) for v in xy) for xy in s["cans_xy_m"]),  # type: ignore[arg-type]
            full_can=int(s["full_can"]),
            bin_heavy_xy_m=tuple(float(v) for v in s["bins_xy_m"]["heavy"]),  # type: ignore[arg-type]
            bin_light_xy_m=tuple(float(v) for v in s["bins_xy_m"]["light"]),  # type: ignore[arg-type]
            handover_xyz_m=tuple(float(v) for v in s["handover_xyz_m"]),  # type: ignore[arg-type]
            ending=s["ending"],
        )
    return DemoLayout(
        sim_robots=tuple(block["sim_robots"]),
        control_dt_s=float(block["control_dt_s"]),
        can=CanSpec(**{k: float(v) for k, v in block["can"].items()}),
        table=TableSpec(**{k: float(v) for k, v in block["table"].items()}),
        grasp=GraspSpec(**{k: float(v) for k, v in block["grasp"].items()}),
        stations=stations,
    )
