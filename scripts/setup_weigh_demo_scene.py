#!/usr/bin/env python3
"""Generate the MuJoCo scene of the "vision cannot weigh" demo.

Starts from the measured physical workcell (``setup_openarm_v1_scene.py``: official OpenArm v1
parallel-link model, 60 mm aluminium torso, shoulder at 1.330 m, table top at 0.725 m) and
replaces the old pick block with the demo props, all placed from ``configs/task_config.yaml``:

- two identical-looking cans (``can_1``, ``can_2``): free bodies with real mass and friction,
  so the gripper really lifts them and the joint torques really carry their weight. Only the
  sim knows which one is full (``demo.sim.masses_kg``).
- two walled bins on the table (``bin_left``, ``bin_right``, robot's left = +Y).
- a translucent marker at the up-forward handover pose.

Re-run after changing ``demo.manual_objects``, ``demo.bins``, ``demo.handover`` or ``demo.sim``:

    uv run python scripts/setup_weigh_demo_scene.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any, Dict
from xml.etree import ElementTree as ET

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.setup_openarm_v1_scene import OFFICIAL_MODEL, generate_scene  # noqa: E402
from src.task.workcell import WorkcellGeometry  # noqa: E402

BIN_COLORS = {"left": "0.85 0.25 0.20 1", "right": "0.20 0.45 0.85 1"}


def _fmt(*values: float) -> str:
    return " ".join(f"{v:.6f}" for v in values)


def _add_bin(worldbody: ET.Element, name: str, x: float, y: float, z_table: float, b: Dict[str, Any]) -> None:
    """Static open box: floor plate and four walls, standing on the table."""
    half = float(b.get("inner_size_m", 0.11)) / 2.0
    wall_h = float(b.get("wall_height_m", 0.03))
    t = 0.006
    body = ET.SubElement(worldbody, "body", name=f"bin_{name}", pos=_fmt(x, y, z_table))
    rgba = BIN_COLORS.get(name, "0.5 0.5 0.5 1")
    ET.SubElement(body, "geom", name=f"bin_{name}_floor", type="box", pos=_fmt(0, 0, 0.0015),
                  size=_fmt(half + t, half + t, 0.0015), rgba=rgba)
    for i, (dx, dy, sx, sy) in enumerate([(half + t / 2, 0, t / 2, half + t), (-half - t / 2, 0, t / 2, half + t),
                                          (0, half + t / 2, half, t / 2), (0, -half - t / 2, half, t / 2)]):
        ET.SubElement(body, "geom", name=f"bin_{name}_wall{i}", type="box", pos=_fmt(dx, dy, wall_h / 2),
                      size=_fmt(sx, sy, wall_h / 2), rgba=rgba)


def generate_demo_scene(cfg: Dict[str, Any], output: Path) -> Path:
    """Writes the demo scene next to the official model (mesh paths keep resolving)."""
    geo = WorkcellGeometry.from_config(cfg)
    demo = cfg["demo"]
    can = demo["sim"]["can"]
    masses = [float(m) for m in demo["sim"]["masses_kg"]]
    objects = demo["manual_objects"]
    if len(objects) != len(masses):
        raise ValueError("demo.sim.masses_kg needs one mass per demo.manual_objects entry")

    with tempfile.TemporaryDirectory() as tmp:
        base_scene = generate_scene(OFFICIAL_MODEL, Path(tmp) / "workcell.xml")
        tree = ET.parse(base_scene)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    assert worldbody is not None
    # Drop the old pick-and-place props (block, weld, place zone).
    for el in list(worldbody):
        if el.get("name") in ("pick_cube", "place_zone"):
            worldbody.remove(el)
    equality = root.find("equality")
    if equality is not None:
        for el in list(equality):
            if el.get("name") == "left_pick_cube_grasp":
                equality.remove(el)

    # Grasp fidelity: elliptic friction cones, higher impratio and noslip iterations stop the
    # slow creep of a gripped object through soft contacts (MuJoCo's recommended grasp settings).
    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        root.insert(0, option)
    option.set("cone", "elliptic")
    option.set("impratio", "10")
    option.set("noslip_iterations", "3")

    asset = root.find("asset")
    assert asset is not None
    ET.SubElement(asset, "material", name="can_material", rgba="0.78 0.80 0.84 1", reflectance="0.3")

    z_table = geo.table_top_world_m
    r, hh = float(can["radius_m"]), float(can["half_height_m"])
    for i, (obj, mass) in enumerate(zip(objects, masses), start=1):
        body = ET.SubElement(worldbody, "body", name=f"can_{i}", pos=_fmt(float(obj["x"]), float(obj["y"]), z_table + hh))
        ET.SubElement(body, "freejoint", name=f"can_{i}_free")
        # Identical look; only the mass differs (the sim's ground truth).
        ET.SubElement(body, "geom", name=f"can_{i}_geom", type="cylinder", size=_fmt(r, hh), mass=f"{mass:.4f}",
                      material="can_material", friction=f"{float(can['friction']):.3f} {float(can.get('torsional_friction', 0.05)):.3f} 0.002", condim="4",
                      solref="0.004 1", solimp="0.95 0.99 0.001")

    for name, b in demo["bins"].items():
        _add_bin(worldbody, name, float(b["x"]), float(b["y"]), z_table, b)

    h = demo["handover"]
    ET.SubElement(worldbody, "site", name="handover_pose", type="sphere", size="0.03",
                  pos=_fmt(float(h["x"]), float(h["y"]), z_table + float(h["above_table_m"])), rgba="0.6 0.3 0.9 0.35")

    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return output


def main() -> int:
    """Reads the config and writes the scene to ``demo.sim.scene_path``."""
    with open(PROJECT_ROOT / "configs" / "task_config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out = PROJECT_ROOT / cfg["demo"]["sim"]["scene_path"]
    print(generate_demo_scene(cfg, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
