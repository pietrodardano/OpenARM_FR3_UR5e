#!/usr/bin/env python3
"""
Camera calibration against the robot itself: touch known points, click them.

This is the reliable one. You move the gripper to a spread of points, the script
reads the robot's own forward kinematics for each, you click the tip in the
captured frame, and it solves for the camera pose. No tape measure, no printed
target, no assumption about where the table edge sits relative to the torso.

Why that matters: with a taped or printed target, the dominant error is not the
camera maths, it is knowing where the target physically sits in the robot base
frame. A tape measure good to +-5 mm puts ~7 mm of error into the tabletop
mapping, and +-10 mm puts ~13 mm in -- worse than everything else combined.
Here the reference points come from the robot's own kinematics, so that term
disappears, and whatever bias the FK has is shared with the controller that will
later execute the grasp: the camera and the arm agree even where both are
slightly wrong, which is what actually lands the gripper on the object.

Measured accuracy of this method, from a noise study (1-2 px clicks, 2 mm of
touch/FK error, measured intrinsics): ~2 mm mean tabletop error and ~1 cm camera
position error with 16 points. Needs 6 minimum; 12-20 is the sweet spot.

THIS SCRIPT NEVER COMMANDS MOTION. It only reads joint positions. Move the arm
yourself -- by hand with the motors disabled, in a gravity-compensated hold, or
with scripts/jog_joints.py in another terminal.

Usage:
    # intrinsics first (scripts/calibrate_intrinsics.py); then:
    ../.venv/bin/python scripts/calibrate_camera_touch.py --side right

    # re-click / re-solve a saved session later, with no robot attached
    ../.venv/bin/python scripts/calibrate_camera_touch.py --load-session outputs/touch_session

Procedure:
  1. Close the gripper fully. The TCP is a point on the gripper axis, 90 mm past
     the hand base -- with the fingers closed, essentially where the fingertips
     meet. That is the point to click, every time.
  2. Move the tip to a point the camera can see and press Enter. Repeat 12-20
     times, spreading the points over the whole area you will pick from, and
     include points at two or three HEIGHTS (on the table, ~100 mm up, ~200 mm
     up). Height spread is what conditions the pose: all-coplanar points give a
     noticeably worse fit, and the script will tell you if you drifted into one.
  3. Click the tip in each captured frame when prompted.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

import cv2
import numpy as np
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from calibrate_table_tape import (  # noqa: E402
    WINDOW, draw_table_grid, draw_zoom_inset, load_config, load_intrinsics, pixel_to_table,
    plane_homography, probe_devices, rotation_to_quat, verify_interactive, write_extrinsics,
)

MIN_POINTS = 6


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------
def read_tcp_position(cfg: dict, side: str) -> np.ndarray:
    """TCP position in the base frame, from the robot's measured joints. Reads only."""
    from src.control.robot_interface import SIDES  # noqa: PLC0415
    from src.hardware import OpenArmBimanual  # noqa: PLC0415
    from src.planning.openarm_kinematics import OpenArmKinematics  # noqa: PLC0415

    robot = OpenArmBimanual.from_config(cfg)
    state = robot.read_passive()
    kinematics = OpenArmKinematics(cfg.get("robot", {}).get("mjcf_path"))
    poses = kinematics.fk({s: state.arms[s].q for s in SIDES})
    return poses[side][0].copy()


def capture_frame(device: int, width: int, height: int, rotate_180: bool) -> np.ndarray:
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        working = probe_devices()
        raise SystemExit(f"Could not open camera index {device}. "
                         + (f"These opened: {working}; re-run with --device {working[0]}."
                            if working else "Check `ls /dev/video*` and permissions."))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    frame = None
    try:
        for _ in range(5):
            ok, grabbed = cap.read()
            if ok:
                frame = grabbed
    finally:
        cap.release()
    if frame is None:
        raise SystemExit(f"Camera {device} opened but returned no frames.")
    return cv2.rotate(frame, cv2.ROTATE_180) if rotate_180 else frame


def spread_report(points: np.ndarray) -> str:
    extent = points.max(axis=0) - points.min(axis=0)
    return (f"spread x {extent[0] * 1000:.0f} mm, y {extent[1] * 1000:.0f} mm, "
            f"z {extent[2] * 1000:.0f} mm")


def planarity_mm(points: np.ndarray) -> float:
    """Thickness of the point cloud along its thinnest direction, in mm."""
    if len(points) < 4:
        return 0.0
    centered = points - points.mean(axis=0)
    return float(np.linalg.svd(centered, compute_uv=False)[-1] / np.sqrt(len(points)) * 1000.0)


def collect_samples(args: argparse.Namespace, cfg: dict) -> tuple[np.ndarray, list[np.ndarray]]:
    vision = cfg.get("vision", {}) or {}
    device = args.device if args.device is not None else int(vision.get("device_index", 0))
    width, height = int(vision.get("width", 1280)), int(vision.get("height", 720))
    rotate = vision.get("rotate_180", False) if args.rotate_180 is None else args.rotate_180

    print("=" * 70)
    print("  Robot-touch camera calibration -- sampling")
    print("=" * 70)
    print("This script never commands motion; move the arm yourself.")
    print("Close the gripper fully. Click the point where the fingertips meet, every time.")
    print(f"Sampling the {args.side} arm's TCP. Spread the points over the whole pick area and")
    print("use two or three heights (table, ~100 mm up, ~200 mm up).")
    print("\n  Enter = sample here,  u = undo the last one,  f = finished,  q = abort\n")

    points: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    while True:
        answer = input(f"[{len(points)} samples] > ").strip().lower()
        if answer == "q":
            raise SystemExit("Aborted.")
        if answer == "u":
            if points:
                points.pop()
                frames.pop()
                print("  dropped the last sample")
            continue
        if answer == "f":
            if len(points) < MIN_POINTS:
                print(f"  need at least {MIN_POINTS} samples, have {len(points)}")
                continue
            break
        if answer != "":
            print("  Enter to sample, u to undo, f to finish, q to abort")
            continue

        tcp = read_tcp_position(cfg, args.side)
        frame = capture_frame(device, width, height, rotate)
        points.append(tcp)
        frames.append(frame)
        current = np.array(points)
        print(f"  sampled TCP  x {tcp[0]:+.4f}  y {tcp[1]:+.4f}  z {tcp[2]:+.4f}  m   "
              f"({spread_report(current)})")

    return np.array(points), frames


# ---------------------------------------------------------------------------
# sessions (touching the robot is the expensive part -- keep the raw data)
# ---------------------------------------------------------------------------
def save_session(directory: str, points: np.ndarray, frames: list[np.ndarray], side: str,
                 rotate_180: bool) -> None:
    os.makedirs(directory, exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(os.path.join(directory, f"frame_{i:02d}.png"), frame)
    with open(os.path.join(directory, "session.json"), "w") as handle:
        json.dump({"side": side, "rotate_180": bool(rotate_180),
                   "tcp_points_base_m": points.tolist(),
                   "saved": datetime.datetime.now().isoformat(timespec="seconds")}, handle, indent=2)
    print(f"\nSaved the session to {directory} -- re-click or re-solve it later with "
          f"--load-session {directory}")


def load_session(directory: str) -> tuple[np.ndarray, list[np.ndarray], dict]:
    path = os.path.join(directory, "session.json")
    if not os.path.exists(path):
        raise SystemExit(f"No session.json in {directory}")
    with open(path) as handle:
        meta = json.load(handle)
    points = np.array(meta["tcp_points_base_m"], dtype=np.float64)
    frames = []
    for i in range(len(points)):
        frame_path = os.path.join(directory, f"frame_{i:02d}.png")
        frame = cv2.imread(frame_path)
        if frame is None:
            raise SystemExit(f"Missing frame {frame_path} for sample {i}")
        frames.append(frame)
    print(f"Loaded {len(points)} samples from {directory} ({meta.get('side', '?')} arm)")
    return points, frames, meta


# ---------------------------------------------------------------------------
# clicking
# ---------------------------------------------------------------------------
def click_tips(frames: list[np.ndarray], points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Click the gripper tip in each frame. Returns the kept (points, pixels)."""
    print("\nClick the gripper tip in each frame -- the same physical point every time.")
    print("  click = place,  u = undo,  Enter = accept and go to the next frame,")
    print("  s = skip this sample (bad frame, tip hidden),  q = abort\n")

    kept_points: list[np.ndarray] = []
    kept_pixels: list[tuple[float, float]] = []
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)

    for index, frame in enumerate(frames):
        picked: list[tuple[float, float]] = []
        hover = [0, 0]

        def on_mouse(event, x, y, _flags, _param):
            hover[0], hover[1] = x, y
            if event == cv2.EVENT_LBUTTONDOWN:
                picked.clear()
                picked.append((float(x), float(y)))

        cv2.setMouseCallback(WINDOW, on_mouse)
        target = points[index]
        while True:
            canvas = frame.copy()
            if picked:
                cv2.drawMarker(canvas, (int(picked[0][0]), int(picked[0][1])), (0, 255, 0),
                               cv2.MARKER_CROSS, 18, 2)
            draw_zoom_inset(canvas, frame, hover)
            label = (f"sample {index + 1}/{len(frames)}  TCP x {target[0]:+.3f} y {target[1]:+.3f} "
                     f"z {target[2]:+.3f}" + ("  [Enter to accept]" if picked else "  [click the tip]"))
            cv2.putText(canvas, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(canvas, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(WINDOW, canvas)

            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                cv2.destroyAllWindows()
                raise SystemExit("Aborted.")
            if key == ord("u"):
                picked.clear()
            if key == ord("s"):
                print(f"  sample {index + 1}: skipped")
                picked.clear()
                break
            if key in (13, 10) and picked:
                kept_points.append(target)
                kept_pixels.append(picked[0])
                break

    cv2.destroyAllWindows()
    if len(kept_points) < MIN_POINTS:
        raise SystemExit(f"Only {len(kept_points)} usable samples, need {MIN_POINTS}.")
    return np.array(kept_points), np.array(kept_pixels)


# ---------------------------------------------------------------------------
# solve
# ---------------------------------------------------------------------------
def solve_touch_pose(points: np.ndarray, pixels: np.ndarray, camera_matrix: np.ndarray,
                     dist_coeffs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Camera pose from N>=6 touched points. Returns (rvec, tvec, per-point px residuals).

    SQPNP handles the general (non-coplanar) case and, unlike the planar solvers,
    does not need the points shifted into a z = 0 plane.
    """
    ok, rvec, tvec = cv2.solvePnP(points, pixels, camera_matrix, dist_coeffs,
                                  flags=cv2.SOLVEPNP_SQPNP)
    if not ok or not np.all(np.isfinite(rvec)) or not np.all(np.isfinite(tvec)):
        ok, rvec, tvec = cv2.solvePnP(points, pixels, camera_matrix, dist_coeffs,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or not np.all(np.isfinite(rvec)):
        raise SystemExit("solvePnP failed -- check that the clicked tips match the sampled points, "
                         "in order, and that the intrinsics belong to this camera.")
    rvec, tvec = cv2.solvePnPRefineLM(points, pixels, camera_matrix, dist_coeffs, rvec, tvec)
    projected, _ = cv2.projectPoints(points, rvec, tvec, camera_matrix, dist_coeffs)
    residuals = np.linalg.norm(projected.reshape(-1, 2) - pixels, axis=1)
    return rvec, tvec.reshape(3, 1), residuals


def solve_with_outlier_rejection(points: np.ndarray, pixels: np.ndarray, camera_matrix: np.ndarray,
                                 dist_coeffs: np.ndarray, max_residual_px: float):
    """Solves, drops points that clearly do not belong, re-solves. Returns the surviving set."""
    rvec, tvec, residuals = solve_touch_pose(points, pixels, camera_matrix, dist_coeffs)
    dropped: list[tuple[int, float]] = []
    while True:
        worst = int(np.argmax(residuals))
        if residuals[worst] <= max_residual_px or len(points) - 1 < MIN_POINTS:
            break
        dropped.append((worst, float(residuals[worst])))
        keep = np.ones(len(points), bool)
        keep[worst] = False
        points, pixels = points[keep], pixels[keep]
        rvec, tvec, residuals = solve_touch_pose(points, pixels, camera_matrix, dist_coeffs)
    return rvec, tvec, residuals, points, pixels, dropped


# ---------------------------------------------------------------------------
# report / output
# ---------------------------------------------------------------------------
def tip_offset_check(points: np.ndarray, table_z: float) -> str:
    """
    The lowest touches should be sitting on the tabletop. If FK says they are
    consistently above or below it, that difference is the offset between the TCP
    site and the physical point you touched with (plus any table-height error).
    """
    lowest = points[:, 2].min()
    on_table = points[np.abs(points[:, 2] - lowest) < 0.01]
    delta_mm = (float(np.mean(on_table[:, 2])) - table_z) * 1000.0
    verdict = "consistent with the configured tabletop" if abs(delta_mm) < 5 else "CHECK THIS"
    return (f"Lowest touches sit {delta_mm:+.1f} mm from the configured tabletop "
            f"(z = {table_z:.3f} m) -- {verdict}.\n"
            f"  If you touched the table with those, that offset is the TCP-to-fingertip "
            f"distance plus any\n  error in workcell.table.top_height_m. It does not affect "
            f"the camera pose below, but it does\n  tell you how much to trust the table height.")


def report(payload: dict, residuals: np.ndarray, dropped: list, points: np.ndarray,
           table_z: float) -> None:
    cam = payload["camera_in_base_frame"]
    pos = cam["position_m"]
    print("\n" + "=" * 70)
    print("  Calibration result (robot-touch)")
    print("=" * 70)
    print(f"Points used         : {len(points)}  ({spread_report(points)})")
    thickness = planarity_mm(points)
    print(f"Height spread       : point cloud is {thickness:.0f} mm thick along its thinnest axis"
          + ("" if thickness > 20 else "  <-- nearly coplanar, the pose is poorly conditioned;"
                                       " re-sample with points at 2-3 heights"))
    if dropped:
        print(f"Dropped as outliers : {len(dropped)} "
              f"({', '.join(f'#{i + 1} at {r:.1f} px' for i, r in dropped)})")
    print(f"Residuals           : {residuals.mean():.2f} px mean, {residuals.max():.2f} px worst")
    print(f"Intrinsics          : {payload['intrinsics']['source']}")
    print(f"  vertical FOV      : {payload['intrinsics']['fovy_deg']:.2f} deg")
    print(f"Camera position     : x {pos[0]:+.3f}  y {pos[1]:+.3f}  z {pos[2]:+.3f}  m")
    print(f"  downward pitch    : {cam['downward_pitch_deg']:.2f} deg below horizontal")
    print(f"  yaw off midline   : {cam['yaw_deg']:+.2f} deg (+ = toward the robot's left)")
    mj = payload["mujoco_camera"]
    print("\nMuJoCo camera (paste into the scene's <camera> element):")
    print(f'  pos="{mj["pos"][0]:.4f} {mj["pos"][1]:.4f} {mj["pos"][2]:.4f}" '
          f'quat="{mj["quat_wxyz"][0]:.5f} {mj["quat_wxyz"][1]:.5f} '
          f'{mj["quat_wxyz"][2]:.5f} {mj["quat_wxyz"][3]:.5f}" fovy="{mj["fovy_deg"]:.2f}"')
    print(f"\n{tip_offset_check(points, table_z)}")
    if residuals.mean() > 3.0:
        print("\nWARNING: residuals are high. Either a tip was clicked in the wrong place, the arm")
        print("moved between the FK read and the frame grab, or the intrinsics are not this camera's.")
        print("Re-run with --load-session to re-click without touching the robot again.")


def build_payload(points, pixels, residuals, camera_matrix, dist_coeffs, source, estimated,
                  rvec, tvec, image_size, table_z, rotate_180, side) -> dict:
    R_world_to_cam, _ = cv2.Rodrigues(rvec)
    R_cam_in_base = R_world_to_cam.T
    cam_pos = (-R_cam_in_base @ tvec).ravel()
    optical_axis = R_cam_in_base @ np.array([0.0, 0.0, 1.0])
    quat = rotation_to_quat(R_cam_in_base @ np.diag([1.0, -1.0, -1.0]))
    fovy = float(np.degrees(2.0 * np.arctan(image_size[1] / (2.0 * camera_matrix[1, 1]))))
    homography = plane_homography(rvec, tvec, camera_matrix, table_z)

    warnings: list[str] = []
    if estimated:
        warnings.append("No measured intrinsics: run scripts/calibrate_intrinsics.py and re-solve "
                        "this session -- the pose and fovy are unreliable without it.")
    if planarity_mm(points) <= 20.0:
        warnings.append("Touched points are nearly coplanar; the camera pose is poorly "
                        "conditioned. Re-sample with points at 2-3 heights.")
    warnings.append("Valid while neither the camera nor the robot base moves. The tabletop "
                    "homography additionally assumes the plane at table_plane.z_m.")

    return {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "method": "robot_touch",
        "frame": "robot_base (+x forward, +y left, +z up)",
        "image_width": image_size[0],
        "image_height": image_size[1],
        "rotate_180_applied": bool(rotate_180),
        "intrinsics": {
            "source": source,
            "estimated": bool(estimated),
            "fx": float(camera_matrix[0, 0]),
            "fy": float(camera_matrix[1, 1]),
            "cx": float(camera_matrix[0, 2]),
            "cy": float(camera_matrix[1, 2]),
            "fovy_deg": fovy,
            "dist_coeffs": [float(v) for v in np.asarray(dist_coeffs).ravel()],
        },
        "camera_in_base_frame": {
            "position_m": [float(v) for v in cam_pos],
            "rotation_opencv": [[float(v) for v in row] for row in R_cam_in_base],
            "optical_axis_in_base": [float(v) for v in optical_axis],
            "downward_pitch_deg": float(np.degrees(np.arcsin(np.clip(-optical_axis[2], -1, 1)))),
            "yaw_deg": float(np.degrees(np.arctan2(optical_axis[1], optical_axis[0]))),
        },
        "mujoco_camera": {
            "pos": [float(v) for v in cam_pos],
            "quat_wxyz": [float(v) for v in quat],
            "fovy_deg": fovy,
        },
        "table_plane": {
            "z_m": float(table_z),
            "homography_plane_from_pixel": [[float(v) for v in row] for row in homography],
            "note": "Undistort the pixel with the intrinsics above before applying the homography.",
        },
        "calibration_target": {
            "method": "robot_touch",
            "arm": side,
            "points_used": len(points),
            "tcp_points_base_m": [[float(v) for v in row] for row in points],
            "clicked_pixels": [[float(v) for v in row] for row in pixels],
            "point_cloud_thickness_mm": planarity_mm(points),
        },
        "warnings": warnings,
        "residuals": {
            "reprojection_error_px": float(np.sqrt(np.mean(residuals ** 2))),
            "mean_residual_px": float(residuals.mean()),
            "max_residual_px": float(residuals.max()),
            "per_point_px": [float(v) for v in residuals],
        },
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/task_config.yaml")
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument("--intrinsics", default="configs/camera_intrinsics.yaml")
    parser.add_argument("--out", default="configs/camera_extrinsics.yaml")
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--rotate-180", dest="rotate_180", action="store_true", default=None)
    parser.add_argument("--no-rotate-180", dest="rotate_180", action="store_false", default=None)
    parser.add_argument("--table-z", type=float, default=None,
                        help="Tabletop height, m. Default: workcell.table.top_height_m.")
    parser.add_argument("--max-residual-px", type=float, default=4.0,
                        help="Drop a touched point whose reprojection residual exceeds this, then "
                             "re-solve (default 4 px). A mis-clicked tip is the usual cause.")
    parser.add_argument("--session", default=None,
                        help="Directory to save the raw samples into (frames + FK points).")
    parser.add_argument("--load-session", default=None,
                        help="Re-click and re-solve a saved session; needs no robot or camera.")
    parser.add_argument("--allow-estimated-intrinsics", action="store_true",
                        help="Proceed without measured intrinsics. The pose will not be reliable.")
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    table_cfg = (config.get("workcell", {}) or {}).get("table", {}) or {}
    table_z = args.table_z if args.table_z is not None else float(table_cfg.get("top_height_m", 0.75))

    if args.load_session:
        points, frames, meta = load_session(args.load_session)
        side = meta.get("side", args.side)
        rotate_180 = bool(meta.get("rotate_180", False))
    else:
        points, frames = collect_samples(args, config)
        side = args.side
        vision = config.get("vision", {}) or {}
        rotate_180 = bool(vision.get("rotate_180", False) if args.rotate_180 is None
                          else args.rotate_180)
        if args.session:
            save_session(args.session, points, frames, side, rotate_180)

    image_size = (frames[0].shape[1], frames[0].shape[0])
    camera_matrix, dist_coeffs, source = load_intrinsics(args.intrinsics, image_size)
    estimated = camera_matrix is None
    if estimated:
        if not args.allow_estimated_intrinsics:
            raise SystemExit(
                f"No measured intrinsics ({source}).\n"
                f"This method needs them -- without the lens model the pose is not trustworthy on a\n"
                f"wide-angle camera. Print configs/charuco_5x7_square28mm_marker20mm.pdf (or run\n"
                f"scripts/charuco_for_screen.py to show the board on a tablet), take 20-30 photos, then:\n"
                f"  ../.venv/bin/python scripts/calibrate_intrinsics.py --images <dir> "
                f"--square-size 28 --marker-size 20\n"
                f"Pass --allow-estimated-intrinsics to proceed anyway with a guessed focal length.")
        focal = 0.9 * max(image_size)  # crude stand-in; the warning in the output says so
        camera_matrix = np.array([[focal, 0.0, image_size[0] / 2.0],
                                  [0.0, focal, image_size[1] / 2.0], [0.0, 0.0, 1.0]])
        dist_coeffs = np.zeros((5, 1))
        source = "GUESSED focal length, no distortion model -- not trustworthy"
    else:
        print(f"Using intrinsics: {source}")

    kept_points, pixels = click_tips(frames, points)
    rvec, tvec, residuals, kept_points, pixels, dropped = solve_with_outlier_rejection(
        kept_points, pixels, camera_matrix, dist_coeffs, args.max_residual_px)

    payload = build_payload(kept_points, pixels, residuals, camera_matrix, dist_coeffs, source,
                            estimated, rvec, tvec, image_size, table_z, rotate_180, side)
    write_extrinsics(args.out, payload)
    report(payload, residuals, dropped, kept_points, table_z)
    print(f"\nWrote {args.out}")

    if not args.no_verify:
        overlay = draw_table_grid(frames[0], rvec, tvec, camera_matrix, dist_coeffs, table_cfg, table_z)
        homography = plane_homography(rvec, tvec, camera_matrix, table_z)
        verify_interactive(overlay, homography, camera_matrix, dist_coeffs)


if __name__ == "__main__":
    main()
