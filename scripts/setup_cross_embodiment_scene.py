"""Generate a visual OpenArm-to-UR5e/FR3 cross-embodiment MuJoCo scene.

The scene is intentionally render-only: it combines CAD-backed OpenArm v1,
UR5e, and FR3 assets but does not claim a shared controller, IK solver, or
collision model. Demonstrations transfer through named task-space TCP targets.
"""

from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MENAGERIE_ROOT = PROJECT_ROOT / "third_party/mujoco_menagerie"
OPENARM_SCENE = PROJECT_ROOT / "third_party/openarm/openarm_mujoco/v1/openarm_bimanual_physical_scene.xml"
GRIPPER_MODEL = MENAGERIE_ROOT / "robotiq_2f85/2f85.xml"
OUTPUT_SCENE = MENAGERIE_ROOT / "cross_embodiment_openarm_ur5e_fr3.xml"

# The OpenArm shoulders are at z=1.330 m in the physical workcell. The root
# plane of each comparison model is chosen so the first actuated shoulder joint
# is at that same height. Stations sit to the OpenArm's world-left (+y) and
# world-right (-y), respectively, which is also left/right in the frontal view.
SHOULDER_HEIGHT_M = 1.330
TORSO_HEIGHT_M = 1.420
TORSO_PROFILE_HALF_WIDTH_M = 0.030
# Like OpenArm's link0, each arm base sits flush on a torso side face and is
# rotated +-90 deg about X so its first joint axis points sideways.
BASE_YAW_DEG = {"ur5e": 0.0, "fr3": 0.0}

# Side-mounted home poses (IK-solved offline for the gripper pinch point): pinch
# ~0.45 m ahead of the torso at z=0.95, 0.30 m (UR5e) / 0.20 m (FR3) off the
# station centre, tool z-axis along +x, inside joint limits and free of
# self/pair collisions.
HOME_QPOS = {
    "ur5e_left_": [-1.7238, -4.1067, 2.1065, -1.1414, 0.153, -0.0494],
    "ur5e_right_": [-1.4178, -1.3934, 1.9628, -0.5694, 0.153, 0.0471],
    "fr3_left_": [0.9473, 0.405, 0.1364, -2.4167, -2.0608, 1.6781, -0.6587],
    "fr3_right_": [-0.8082, 0.4155, -0.2734, -2.4213, 2.0892, 1.6339, -1.0493],
}


def _quat_mul(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, ...]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _axis_quat(axis: str, degrees: float) -> tuple[float, ...]:
    half = math.radians(degrees) / 2.0
    vector = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return (math.cos(half), *(math.sin(half) * v for v in vector))


def _side_mount_quat(side: str, yaw_deg: float) -> str:
    """Base orientation: Rx(-+90) as in OpenArm link0, after an optional yaw about the base z."""
    roll = _axis_quat("x", -90.0 if side == "left" else 90.0)
    return " ".join(f"{v:.6f}" for v in _quat_mul(roll, _axis_quat("z", yaw_deg)))


def _prefixed_robot_body(
    model_path: Path,
    prefix: str,
    position: str,
    quaternion: str,
) -> tuple[list[ET.Element], list[ET.Element], ET.Element, dict[str, list[ET.Element]]]:
    """Return renamed assets/defaults, a fixed-base CAD robot body and its constraint sections.

    Menagerie models are standalone MuJoCo documents. Their named entities are
    prefixed here so four independently mounted instances can coexist.
    """
    root = ET.parse(model_path).getroot()
    asset = root.find("asset")
    defaults = root.find("default")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        raise ValueError(f"Unexpected Menagerie model structure: {model_path}")

    assets = [deepcopy(element) for element in asset]
    default_elements = [deepcopy(element) for element in defaults] if defaults is not None else []
    source_body = next((deepcopy(element) for element in worldbody if element.tag == "body"), None)
    if source_body is None:
        raise ValueError(f"No root body found in {model_path}")

    # Closed-chain grippers (2F-85) need their contact excludes, tendon,
    # equality constraints and actuator carried over with the same renaming.
    sections = {
        tag: [deepcopy(child) for child in root.find(tag)]
        for tag in ("contact", "tendon", "equality", "actuator")
        if root.find(tag) is not None
    }
    named_elements = [*assets, *default_elements, source_body, *(e for items in sections.values() for e in items)]
    rename_map: dict[str, str] = {}
    for element in named_elements:
        for descendant in element.iter():
            name = descendant.get("name")
            if name:
                rename_map[name] = f"{prefix}{name}"
            element_class = descendant.get("class")
            if element_class:
                rename_map[element_class] = f"{prefix}{element_class}"

    # Menagerie mesh names are implicit when omitted. Give every mesh an
    # explicit prefixed name and resolve its asset path to an absolute path.
    asset_directory = model_path.parent / "assets"
    for mesh in (element for element in assets if element.tag == "mesh"):
        mesh_file = mesh.get("file")
        if mesh_file is None:
            continue
        source_name = mesh.get("name")
        implicit_name = Path(mesh_file).stem
        prefixed_name = f"{prefix}{source_name or implicit_name}"
        if source_name:
            rename_map[source_name] = prefixed_name
        else:
            rename_map[implicit_name] = prefixed_name
        mesh.set("name", prefixed_name)
        mesh.set("file", str((asset_directory / mesh_file).resolve()))

    for element in named_elements:
        for descendant in element.iter():
            for attribute, value in list(descendant.attrib.items()):
                if value in rename_map:
                    descendant.set(attribute, rename_map[value])

    source_body.attrib.pop("quat", None)
    wrapper = ET.Element("body", name=f"{prefix}station", pos=position, quat=quaternion)
    wrapper.append(source_body)
    return assets, default_elements, wrapper, sections


def _attach_gripper(root: ET.Element, robot_body: ET.Element, arm_prefix: str) -> None:
    """Mount a Robotiq 2F-85 on the arm's ``attachment_site`` (flange)."""
    site = next(e for e in robot_body.iter("site") if e.get("name") == f"{arm_prefix}attachment_site")
    parent = next(b for b in robot_body.iter("body") if site in list(b))
    prefix = f"gripper_{arm_prefix}"
    assets, defaults, wrapper, sections = _prefixed_robot_body(
        GRIPPER_MODEL, prefix, site.get("pos", "0 0 0"), site.get("quat", "1 0 0 0")
    )
    wrapper.set("name", f"{prefix}mount")
    parent.append(wrapper)
    root.find("default").extend(defaults)
    root.find("asset").extend(assets)
    for tag, items in sections.items():
        container = root.find(tag)
        if container is None:
            container = ET.SubElement(root, tag)
        container.extend(items)


def _add_bimanual_station(worldbody: ET.Element, name: str, center_y: float) -> None:
    """Add the 60 mm aluminum torso that both side-mounted arms bolt onto."""
    ET.SubElement(
        worldbody,
        "geom",
        name=f"{name}_torso_profile",
        type="box",
        pos=f"0 {center_y:.6f} {TORSO_HEIGHT_M / 2.0:.6f}",
        size=f"{TORSO_PROFILE_HALF_WIDTH_M:.6f} {TORSO_PROFILE_HALF_WIDTH_M:.6f} {TORSO_HEIGHT_M / 2.0:.6f}",
        material="cross_embodiment_aluminum",
        contype="0",
        conaffinity="0",
    )


def generate_scene() -> Path:
    """Generate the visual-only OpenArm, dual-UR5e, dual-FR3 comparison scene."""
    if not OPENARM_SCENE.is_file():
        raise FileNotFoundError("Generate the physical OpenArm scene before this cross-embodiment scene.")
    ur5e_model = MENAGERIE_ROOT / "universal_robots_ur5e/ur5e.xml"
    fr3_model = MENAGERIE_ROOT / "franka_fr3/fr3.xml"
    if not ur5e_model.is_file() or not fr3_model.is_file():
        raise FileNotFoundError("MuJoCo Menagerie UR5e/FR3 sources are missing under third_party/mujoco_menagerie.")

    tree = ET.parse(OPENARM_SCENE)
    root = tree.getroot()
    asset = root.find("asset")
    worldbody = root.find("worldbody")
    compiler = root.find("compiler")
    if asset is None or worldbody is None or compiler is None:
        raise ValueError("Unexpected generated OpenArm scene structure.")
    compiler.set("meshdir", str((OPENARM_SCENE.parent / "meshes").resolve()))
    root_default = root.find("default")
    if root_default is None:
        raise ValueError("Generated OpenArm scene has no top-level default container.")

    ET.SubElement(asset, "material", name="cross_embodiment_aluminum", rgba="0.65 0.68 0.72 1", reflectance="0.35")

    # Two independent bimanual stations flank OpenArm. From a frontal view,
    # UR5e is on OpenArm's left (+y) and FR3 is on its right (-y). The two
    # arms in each pair are mounted symmetrically on a shared torso rail.
    ur5e_station_y = 1.350
    fr3_station_y = -1.350
    _add_bimanual_station(worldbody, "ur5e", ur5e_station_y)
    _add_bimanual_station(worldbody, "fr3", fr3_station_y)
    placements = []
    for model_path, model_name, station_y in ((ur5e_model, "ur5e", ur5e_station_y), (fr3_model, "fr3", fr3_station_y)):
        for side, sign in (("left", 1.0), ("right", -1.0)):
            position = f"0 {station_y + sign * TORSO_PROFILE_HALF_WIDTH_M:.6f} {SHOULDER_HEIGHT_M:.6f}"
            placements.append(
                (model_path, f"{model_name}_{side}_", position, _side_mount_quat(side, BASE_YAW_DEG[model_name]))
            )
    for model_path, prefix, position, quaternion in placements:
        assets, defaults, robot_body, _ = _prefixed_robot_body(model_path, prefix, position, quaternion)
        for default in defaults:
            root_default.append(default)
        for item in assets:
            asset.append(item)
        worldbody.append(robot_body)
        _attach_gripper(root, robot_body, prefix)

    ET.SubElement(asset, "material", name="cross_embodiment_marker", rgba="0.05 0.95 0.95 0.75")
    # Canonical task-frame markers. A future adapter maps OpenArm TCP goals to
    # each robot's base frame and solves that robot's own IK.
    for name, position in (
        ("task_frame_left_tcp_target", "0.45 1.00 0.95"),
        ("task_frame_right_tcp_target", "0.45 0.80 0.95"),
    ):
        ET.SubElement(
            worldbody,
            "site",
            name=name,
            pos=position,
            type="sphere",
            size="0.035",
            material="cross_embodiment_marker",
            group="4",
        )

    ET.SubElement(
        worldbody,
        "camera",
        name="cross_embodiment_overview",
        pos="3.2 0 1.6",
        xyaxes="0 1 0 -0.35 0 0.94",
        fovy="60",
    )

    ET.indent(tree, space="  ")
    OUTPUT_SCENE.parent.mkdir(parents=True, exist_ok=True)
    tree.write(OUTPUT_SCENE, encoding="utf-8", xml_declaration=True)
    _write_home_keyframe(tree)
    return OUTPUT_SCENE


def _write_home_keyframe(tree: ET.ElementTree) -> None:
    """Extend the OpenArm ``home`` keyframe with UR5e/FR3 home poses.

    Needs the compiled model to map joint names to qpos addresses, so the scene
    is written first, compiled, and the keyframe qpos is rewritten.
    """
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(OUTPUT_SCENE))
    qpos = model.qpos0.copy()
    key = model.key("home")
    qpos[:] = key.qpos
    for prefix, values in HOME_QPOS.items():
        joints = [model.joint(i) for i in range(model.njnt) if model.joint(i).name.startswith(prefix)]
        if len(joints) != len(values):
            raise ValueError(f"Expected {len(values)} joints for {prefix}, found {len(joints)}")
        for joint, value in zip(joints, values):
            qpos[joint.qposadr[0]] = value
    root = tree.getroot()
    key_element = root.find("keyframe/key[@name='home']")
    if key_element is None:
        raise ValueError("Scene has no 'home' keyframe.")
    key_element.set("qpos", " ".join(f"{v:.6f}" for v in qpos))
    tree.write(OUTPUT_SCENE, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    print(generate_scene())
