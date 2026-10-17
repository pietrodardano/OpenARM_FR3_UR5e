"""Generate the local physical-workcell wrapper for the official OpenArm v1 model.

The upstream model remains unmodified. This script writes a generated MJCF next
to that ignored source checkout, so its mesh paths continue to resolve exactly
as supplied by OpenArm.
"""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_MODEL = (
    PROJECT_ROOT / "third_party/openarm/openarm_mujoco/v1/openarm_bimanual.xml"
)
GENERATED_MODEL = (
    PROJECT_ROOT
    / "third_party/openarm/openarm_mujoco/v1/openarm_bimanual_physical_scene.xml"
)

# World convention: +x forward, +y left, +z up; the ground plane is z = 0.
ARM_MOUNT_HEIGHT_M = 1.330
TORSO_HEIGHT_M = 1.420
TORSO_PROFILE_WIDTH_M = 0.060
CAMERA_HEIGHT_ABOVE_TORSO_M = 0.180
CAMERA_FORWARD_OFFSET_M = 0.070
CAMERA_DOWNWARD_PITCH_DEG = 20.0
TABLE_TOP_HEIGHT_M = 0.750
TABLE_TOP_THICKNESS_M = 0.040
# Provisional footprint: replace with measured dimensions when available.
TABLE_LENGTH_M = 1.200
TABLE_WIDTH_M = 0.800
# Rear edge is at x=0.050 m: 20 mm clear of the 60 mm torso profile's
# forward face at x=0.030 m, while keeping the tabletop close to the base.
TABLE_CENTER_X_M = 0.650
PICK_BLOCK_SIZE_M = (0.040, 0.050, 0.100)
# Centered between the official v1 left parallel-link finger collision meshes
# at the verified reachable lateral grasp posture.
PICK_CUBE_CENTER_M = (0.259, 0.194, TABLE_TOP_HEIGHT_M + PICK_BLOCK_SIZE_M[2] / 2.0)
PLACE_ZONE_CENTER_M = (0.200, 0.250, TABLE_TOP_HEIGHT_M)

# The official model's arm_origin site is 0.697 m above openarm_body_link0.
OPENARM_ARM_ORIGIN_ABOVE_ROOT_M = 0.697


def _add_static_workcell(worldbody: ET.Element) -> None:
    """Add the measured floor, aluminum torso, camera bracket, and RGB camera."""
    torso_center_z = TORSO_HEIGHT_M / 2.0
    camera_z = TORSO_HEIGHT_M + CAMERA_HEIGHT_ABOVE_TORSO_M
    table_center_z = TABLE_TOP_HEIGHT_M - TABLE_TOP_THICKNESS_M / 2.0

    ET.SubElement(
        worldbody,
        "geom",
        name="ground",
        type="plane",
        size="4 4 0.1",
        material="physical_ground",
        contype="1",
        conaffinity="1",
    )
    pickup_body = ET.SubElement(
        worldbody,
        "body",
        name="pick_cube",
        pos="{:.6f} {:.6f} {:.6f}".format(*PICK_CUBE_CENTER_M),
    )
    ET.SubElement(pickup_body, "freejoint", name="pick_cube_freejoint")
    ET.SubElement(
        pickup_body,
        "geom",
        name="pick_cube_geom",
        type="box",
        size="{:.6f} {:.6f} {:.6f}".format(*(dimension / 2.0 for dimension in PICK_BLOCK_SIZE_M)),
        mass="0.050",
        material="pick_cube_material",
        contype="1",
        conaffinity="1",
    )
    ET.SubElement(
        worldbody,
        "site",
        name="place_zone",
        pos="{:.6f} {:.6f} {:.6f}".format(*PLACE_ZONE_CENTER_M),
        size="0.075 0.075 0.002",
        type="box",
        rgba="0.15 0.85 0.25 0.35",
        group="4",
    )
    ET.SubElement(
        worldbody,
        "light",
        name="workcell_key_light",
        pos="-1.5 -1.5 3.0",
        dir="0.5 0.5 -1.0",
        directional="true",
        castshadow="true",
        diffuse="0.85 0.85 0.85",
        specular="0.20 0.20 0.20",
    )
    ET.SubElement(
        worldbody,
        "light",
        name="workcell_fill_light",
        pos="1.5 1.5 2.5",
        dir="-0.4 -0.4 -1.0",
        directional="true",
        diffuse="0.35 0.40 0.48",
        specular="0.05 0.05 0.05",
    )
    ET.SubElement(
        worldbody,
        "geom",
        name="work_table_top",
        type="box",
        pos=f"{TABLE_CENTER_X_M:.6f} 0 {table_center_z:.6f}",
        size=(
            f"{TABLE_LENGTH_M / 2.0:.6f} {TABLE_WIDTH_M / 2.0:.6f} "
            f"{TABLE_TOP_THICKNESS_M / 2.0:.6f}"
        ),
        material="table_surface",
        contype="1",
        conaffinity="1",
    )
    ET.SubElement(
        worldbody,
        "geom",
        name="aluminum_torso_profile",
        type="box",
        pos=f"0 0 {torso_center_z:.6f}",
        size=f"{TORSO_PROFILE_WIDTH_M / 2.0:.6f} {TORSO_PROFILE_WIDTH_M / 2.0:.6f} {torso_center_z:.6f}",
        material="aluminum_profile",
        contype="1",
        conaffinity="1",
    )
    ET.SubElement(
        worldbody,
        "geom",
        name="camera_mount_bracket",
        type="cylinder",
        fromto=(
            f"0 0 {TORSO_HEIGHT_M:.6f} "
            f"{CAMERA_FORWARD_OFFSET_M:.6f} 0 {camera_z:.6f}"
        ),
        size="0.012",
        material="aluminum_profile",
        contype="1",
        conaffinity="1",
    )
    ET.SubElement(
        worldbody,
        "site",
        name="physical_camera_mount",
        pos=f"{CAMERA_FORWARD_OFFSET_M:.6f} 0 {camera_z:.6f}",
        size="0.015",
        rgba="0.1 0.8 0.2 1",
        group="4",
    )
    # MuJoCo cameras look along local -Z. local +X is robot right (-world Y),
    # local +Y completes a right-handed frame, and local -Z points forward and
    # 20 degrees downward. The FOV is provisional until camera calibration.
    ET.SubElement(
        worldbody,
        "camera",
        name="physical_hbv_rgb_camera",
        pos=f"{CAMERA_FORWARD_OFFSET_M:.6f} 0 {camera_z:.6f}",
        xyaxes="0 -1 0 0.3420201433 0 0.9396926208",
        fovy="70",
    )


def generate_scene(
    source_model: Path = OFFICIAL_MODEL,
    output_model: Path = GENERATED_MODEL,
) -> Path:
    """Write the official v1 model augmented with the measured static workcell.

    Args:
        source_model: Unmodified OpenArm v1 bimanual MJCF.
        output_model: Location for the generated, local simulation scene.

    Returns:
        The generated MJCF path.

    Raises:
        FileNotFoundError: If the official OpenArm source checkout is absent.
        ValueError: If the expected OpenArm root body cannot be found.
    """
    if not source_model.is_file():
        raise FileNotFoundError(
            "Official OpenArm v1 model not found. Clone the sources described in "
            "docs/setup/openarm_v1_environment.md before generating this scene: "
            f"{source_model}"
        )

    tree = ET.parse(source_model)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    openarm_root = root.find("./worldbody/body[@name='openarm_body_link0']")
    asset = root.find("asset")
    if worldbody is None or openarm_root is None or asset is None:
        raise ValueError("Unexpected official OpenArm v1 MJCF structure.")

    # The upstream root body includes the original OpenArm chassis/torso meshes.
    # Retain the body solely as the fixed kinematic parent of both official arm
    # chains, but remove its direct visual and collision geometry so the measured
    # 60 mm square aluminum profile is the only torso in this workcell scene.
    for geom in list(openarm_root.findall("geom")):
        openarm_root.remove(geom)

    visual = root.find("visual")
    if visual is None:
        visual = ET.Element("visual")
        root.insert(list(root).index(worldbody), visual)
    ET.SubElement(
        visual,
        "headlight",
        ambient="0.45 0.45 0.45",
        diffuse="0.55 0.55 0.55",
        specular="0.10 0.10 0.10",
    )
    ET.SubElement(visual, "rgba", haze="0.10 0.12 0.15 1", fog="0.92 0.94 0.98 1")

    openarm_root.set("pos", f"0 0 {ARM_MOUNT_HEIGHT_M - OPENARM_ARM_ORIGIN_ABOVE_ROOT_M:.6f}")
    ET.SubElement(asset, "material", name="physical_ground", rgba="0.32 0.34 0.38 1", reflectance="0.12")
    ET.SubElement(asset, "material", name="aluminum_profile", rgba="0.65 0.68 0.72 1", reflectance="0.35")
    ET.SubElement(asset, "material", name="table_surface", rgba="0.46 0.30 0.18 1", reflectance="0.08")
    ET.SubElement(asset, "material", name="pick_cube_material", rgba="0.85 0.18 0.12 1", reflectance="0.15")
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    ET.SubElement(
        equality,
        "weld",
        name="left_pick_cube_grasp",
        body1="openarm_left_hand_tcp",
        body2="pick_cube",
        active="false",
        solref="0.002 1",
    )
    _add_static_workcell(worldbody)

    ET.indent(tree, space="  ")
    output_model.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_model, encoding="utf-8", xml_declaration=True)
    return output_model


if __name__ == "__main__":
    print(generate_scene())