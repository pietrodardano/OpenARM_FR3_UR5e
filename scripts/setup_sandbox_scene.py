"""Generate the OpenArm manipulation sandbox: the workcell with a table at elbow height and props.

    python scripts/setup_sandbox_scene.py                       # jar dispenser + cups
    python scripts/setup_sandbox_scene.py --props dispenser,cups,blocks --table-height 1.0
    python scripts/setup_sandbox_scene.py --props all
    python scripts/setup_sandbox_scene.py --list

It starts from the measured physical workcell (``setup_openarm_v1_scene.py``: floor,
60 mm torso profile, camera), drops its low pick-and-place table / cube, and adds a table
whose top is just below the robot's elbow joints (z = 1.11 m, shoulders at 1.33 m) so the
forearms hover over the surface at the home pose.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from setup_openarm_v1_scene import OFFICIAL_MODEL, generate_scene  # noqa: E402
from src.simulation import sandbox_props as sp  # noqa: E402

SANDBOX_MODEL = PROJECT_ROOT / "third_party/openarm/openarm_mujoco/v1/openarm_bimanual_sandbox.xml"
ELBOW_HEIGHT_M = 1.11  # openarm_*_joint4 anchor height in the physical workcell scene
DEFAULT_TABLE_TOP_M = 1.05  # 6 cm under the elbows: the forward-pointing gripper clears it at home

PROPS = {
    "dispenser": "16x16x25 cm jug, spigot + spring-return push-down lever (faces the robot)",
    "cups": "tumbler under the nozzle, tall cup, mug",
    "blocks": "wood / rubber / metal / foam blocks (friction & density presets)",
    "lidbox": "box with a hinged lid",
    "cloth": "deformable cloth (flexcomp; slower)",
    "sack": "deformable soft sack (flexcomp; slower)",
}
DEFAULT_PROPS = ["dispenser", "cups"]


def _strip_pick_place(root: ET.Element) -> None:
    worldbody = root.find("worldbody")
    for el in list(worldbody):
        if el.get("name") in ("work_table_top", "pick_cube", "place_zone"):
            worldbody.remove(el)
    equality = root.find("equality")
    if equality is not None:
        for el in list(equality):
            if el.get("name") == "left_pick_cube_grasp":
                equality.remove(el)


def build_sandbox(props: list[str], table_top: float = DEFAULT_TABLE_TOP_M, output: Path = SANDBOX_MODEL) -> Path:
    """Writes the sandbox MJCF to ``output`` (next to the official model so mesh paths resolve)."""
    generate_scene(OFFICIAL_MODEL, output)
    tree = ET.parse(output)
    root = tree.getroot()
    worldbody, asset = root.find("worldbody"), root.find("asset")
    _strip_pick_place(root)

    T = table_top
    sp.add_table(worldbody, asset, T)

    # Layout keeps clear of the grippers' home-pose footprint (x 0.30-0.52, |y| 0.10-0.20, z ~1.06-1.16).
    nozzle_xy = (0.47, 0.0)
    if "dispenser" in props:
        nozzle_xy = sp.add_jar_dispenser(worldbody, asset, "dispenser", (0.62, 0.0, T))["nozzle_xy"]
    if "cups" in props:
        sp.add_cup(worldbody, asset, "cup_tumbler", (nozzle_xy[0], nozzle_xy[1], T), "tumbler")  # under the nozzle
        sp.add_cup(worldbody, asset, "cup_tall", (0.45, 0.28, T), "tall", rgba=(0.95, 0.7, 0.7, 1))
        sp.add_cup(worldbody, asset, "cup_mug", (0.45, -0.28, T), "mug", rgba=(0.4, 0.6, 0.9, 1))
    if "blocks" in props:
        for i, preset in enumerate(("wood", "rubber", "metal", "foam")):
            sp.add_block(worldbody, f"block_{preset}", (0.62, 0.16 + 0.08 * i, T + 0.025), preset)
    if "lidbox" in props:
        sp.add_lidded_box(worldbody, asset, "lidbox", (0.64, -0.22, T))
    if "cloth" in props:
        sp.add_cloth(worldbody, "cloth", (0.36, -0.40, T + 0.012))
    if "sack" in props:
        sp.add_soft_sack(worldbody, "sack", (0.35, 0.40, T + 0.05))

    cam_pos, cam_target = (0.32, -0.60, T + 0.30), (0.52, 0.0, T + 0.12)  # side view: spigot + lever
    ET.SubElement(worldbody, "camera", name="sandbox_overview", pos=" ".join(map(str, cam_pos)),
                  xyaxes=" ".join(f"{v:.6f}" for v in sp.camera_xyaxes(cam_pos, cam_target)), fovy="55")

    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    mujoco.MjModel.from_xml_path(str(output))  # validation
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--props", default=",".join(DEFAULT_PROPS), help="comma list, or 'all' (see --list)")
    parser.add_argument("--table-height", type=float, default=DEFAULT_TABLE_TOP_M, help="tabletop z in metres")
    parser.add_argument("--out", type=Path, default=SANDBOX_MODEL)
    parser.add_argument("--list", action="store_true", help="list available props")
    args = parser.parse_args()
    if args.list:
        for k, v in PROPS.items():
            print(f"{k:10s} {v}")
        return
    props = list(PROPS) if args.props == "all" else [p.strip() for p in args.props.split(",") if p.strip()]
    unknown = set(props) - set(PROPS)
    if unknown:
        parser.error(f"unknown props {sorted(unknown)}; choose from {list(PROPS)}")
    print(build_sandbox(props, args.table_height, args.out))


if __name__ == "__main__":
    main()
