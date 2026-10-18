#!/usr/bin/env python3
"""
Camera pose from the CAD mount geometry, not from touching the robot or taping the table.

Writes configs/camera_extrinsics.yaml using only:
  * configs/camera_intrinsics.yaml (measured lens: fx, fy, cx, cy, distortion) -- a must, this
    script refuses to run without it (pass --allow-estimated-intrinsics to override anyway).
  * the nominal mount dimensions in task_config.yaml's workcell.rgb_camera section (forward
    offset, height above the torso, downward pitch) plus workcell.table.top_height_m.

No photo is taken and the robot is not touched -- this is pure geometry from numbers already in
the config. That is also its weakness: those numbers are tape-measure design estimates of the
mount, not a calibration. Expect the camera position to be off by however far the real mount is
from its CAD position (could be several cm), and the pitch by however many degrees the bracket
actually sits at versus the drawing.

Use it to get the vision pipeline running end to end quickly (object detection, demo wiring,
workspace checks). Before trusting it for an unattended real grasp:
  1. Run scripts/detect_objects.py and check the printed base-frame positions against a ruler.
  2. If they are off by more than you can live with, replace this with
     scripts/calibrate_camera_touch.py (recommended) or scripts/calibrate_table_tape.py --
     both solve for the *actual* pose instead of assuming the drawing is correct.

Usage:
    python scripts/calibrate_camera_nominal.py
    python scripts/calibrate_camera_nominal.py --config configs/task_config.yaml \\
        --intrinsics configs/camera_intrinsics.yaml --out configs/camera_extrinsics.yaml
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys

import cv2
import numpy as np
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from scripts.calibrate_table_tape import load_intrinsics, plane_homography, rotation_to_quat  # noqa: E402

# The official OpenArm v1 model's arm_origin site, above openarm_body_link0 (see
# scripts/setup_openarm_v1_scene.py). The physical workcell positions that root at
# workcell.arm_mount_height_m - this, so the robot base frame (controllers/IK, origin at
# openarm_body_link0) sits this far below the workcell world frame (origin on the floor).
OPENARM_ARM_ORIGIN_ABOVE_ROOT_M = 0.697


def camera_pose_from_mount(offset_forward_m: float, height_above_torso_m: float, torso_height_m: float,
                           pitch_deg: float, root_z_in_world: float) -> tuple[np.ndarray, np.ndarray]:
    """CAD mount geometry -> (camera_position_base, R_cam_in_base), both in the robot base frame.

    The camera faces +x (forward), no roll, pitched ``pitch_deg`` down (about the world y axis).
    OpenCV camera axes: +x right, +y down, +z forward (into the scene). With the robot's +y left,
    "right in the image" for a forward-looking, unrolled camera is -y, which pins the x axis
    regardless of pitch; the y and z axes then follow from the right-handed frame.
    """
    pitch = np.radians(pitch_deg)
    cos_p, sin_p = np.cos(pitch), np.sin(pitch)
    x_cam = np.array([0.0, -1.0, 0.0])             # image right
    z_cam = np.array([cos_p, 0.0, -sin_p])          # optical axis: forward, pitched down
    y_cam = np.cross(z_cam, x_cam)                  # image down (right-handed: y = z x x)
    r_cam_in_base = np.stack([x_cam, y_cam, z_cam], axis=1)  # columns = camera axes in the base frame

    camera_pos_world = np.array([offset_forward_m, 0.0, torso_height_m + height_above_torso_m])
    camera_pos_base = camera_pos_world - np.array([0.0, 0.0, root_z_in_world])
    return camera_pos_base, r_cam_in_base


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/task_config.yaml")
    parser.add_argument("--intrinsics", default="configs/camera_intrinsics.yaml")
    parser.add_argument("--out", default="configs/camera_extrinsics.yaml")
    parser.add_argument("--table-z", type=float, default=None,
                        help="tabletop height in the base frame, m. Default: derived from "
                             "workcell.table.top_height_m and workcell.arm_mount_height_m.")
    parser.add_argument("--allow-estimated-intrinsics", action="store_true",
                        help="proceed even without configs/camera_intrinsics.yaml (not recommended: "
                             "without it, the pose and FOV can be badly wrong on this wide-angle lens).")
    args = parser.parse_args()

    with open(args.config) as handle:
        cfg = yaml.safe_load(handle)
    wc, vc = cfg["workcell"], cfg["vision"]
    cam = wc["rgb_camera"]
    image_size = (int(vc["width"]), int(vc["height"]))

    root_z_in_world = float(wc["arm_mount_height_m"]) - OPENARM_ARM_ORIGIN_ABOVE_ROOT_M
    table_z = args.table_z if args.table_z is not None else float(wc["table"]["top_height_m"]) - root_z_in_world
    cam_pos, r_cam_in_base = camera_pose_from_mount(
        offset_forward_m=float(cam["offset_forward_from_torso_m"]),
        height_above_torso_m=float(cam["height_above_torso_m"]),
        torso_height_m=float(wc["torso"]["height_m"]),
        pitch_deg=float(cam["downward_pitch_deg"]),
        root_z_in_world=root_z_in_world,
    )

    camera_matrix, dist_coeffs, source = load_intrinsics(args.intrinsics, image_size)
    estimated_intrinsics = camera_matrix is None
    if estimated_intrinsics:
        if not args.allow_estimated_intrinsics:
            parser.error(
                f"{args.intrinsics} not found or unusable ({source}). Run scripts/calibrate_intrinsics.py "
                "first -- intrinsics are not optional here, this script has no rectangle to estimate the "
                "focal length from. Pass --allow-estimated-intrinsics to proceed anyway (rough, ~90 deg guess).")
        print(f"WARNING: no measured intrinsics ({source}); using a rough guess.")
        focal = 0.9 * max(image_size)
        camera_matrix = np.array([[focal, 0.0, image_size[0] / 2.0],
                                  [0.0, focal, image_size[1] / 2.0],
                                  [0.0, 0.0, 1.0]])
        dist_coeffs = np.zeros((5, 1))
        source = "estimated (no calibration board)"
    else:
        print(f"Using intrinsics: {source}")

    r_world_to_cam = r_cam_in_base.T
    rvec, _ = cv2.Rodrigues(r_world_to_cam)
    tvec = (-r_world_to_cam @ cam_pos).reshape(3, 1)
    h_plane_from_pixel = plane_homography(rvec, tvec, camera_matrix, table_z)

    optical_axis = r_cam_in_base @ np.array([0.0, 0.0, 1.0])
    pitch_down = float(np.degrees(np.arcsin(np.clip(-optical_axis[2], -1.0, 1.0))))
    yaw = float(np.degrees(np.arctan2(optical_axis[1], optical_axis[0])))
    r_mujoco = r_cam_in_base @ np.diag([1.0, -1.0, -1.0])  # MuJoCo: -z forward, +y up; OpenCV: +z forward, +y down
    quat = rotation_to_quat(r_mujoco)
    fovy = float(np.degrees(2.0 * np.arctan(image_size[1] / (2.0 * camera_matrix[1, 1]))))

    warnings = [
        "Pose (camera_in_base_frame / mujoco_camera / table_plane) is NOT measured: it comes from the "
        "CAD mount dimensions in task_config.yaml's workcell.rgb_camera, not from touching the robot or "
        "taping the table. Expect several cm of position error and a few degrees of pitch error. Check "
        "scripts/detect_objects.py against a ruler before trusting this for an unattended grasp; replace "
        "with scripts/calibrate_camera_touch.py (recommended) when precision matters.",
        "Valid only for the tabletop plane at table_plane.z_m, and only while neither the camera nor the "
        "table moves, or the mount geometry in task_config.yaml changes.",
    ]
    if estimated_intrinsics:
        warnings.insert(0, "No measured intrinsics either: fx/fy/fovy are a rough 90 deg guess. Run "
                           "scripts/calibrate_intrinsics.py.")

    payload = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "calibration_method": "nominal_cad",
        "frame": "robot_base (+x forward, +y left, +z up)",
        "image_width": image_size[0],
        "image_height": image_size[1],
        "rotate_180_applied": bool(vc.get("rotate_180", False)),
        "intrinsics": {
            "source": source,
            "estimated": bool(estimated_intrinsics),
            "fx": float(camera_matrix[0, 0]),
            "fy": float(camera_matrix[1, 1]),
            "cx": float(camera_matrix[0, 2]),
            "cy": float(camera_matrix[1, 2]),
            "fovy_deg": fovy,
            "dist_coeffs": [float(v) for v in np.asarray(dist_coeffs).ravel()],
        },
        "camera_in_base_frame": {
            "position_m": [float(v) for v in cam_pos],
            "rotation_opencv": [[float(v) for v in row] for row in r_cam_in_base],
            "optical_axis_in_base": [float(v) for v in optical_axis],
            "downward_pitch_deg": pitch_down,
            "yaw_deg": yaw,
        },
        "mujoco_camera": {
            "pos": [float(v) for v in cam_pos],
            "quat_wxyz": [float(v) for v in quat],
            "fovy_deg": fovy,
        },
        "table_plane": {
            "z_m": float(table_z),
            "homography_plane_from_pixel": [[float(v) for v in row] for row in h_plane_from_pixel],
            "note": "Undistort the pixel with the intrinsics above before applying the homography.",
        },
        "warnings": warnings,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    print(f"\nWrote {args.out} (nominal CAD pose, NOT a calibration)")
    print(f"  camera position (base frame): {np.round(cam_pos, 3)}")
    print(f"  pitch down: {pitch_down:.1f} deg, yaw: {yaw:.1f} deg, fovy (from measured lens): {fovy:.1f} deg")
    print(f"  table z (base frame): {table_z:.3f} m")
    for w in warnings:
        print(f"  WARNING: {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
