"""
Tape-rectangle camera calibration: does it recover a known camera pose, and do
the frame conventions it writes out actually mean what they claim?

Every test renders a synthetic view from a ground-truth camera, so the expected
answer is known exactly.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from calibrate_table_tape import (  # noqa: E402
    corners_from_edges, estimate_focal, fit_line, plane_homography, pixel_to_table,
    rotation_to_quat, solve_pose, world_corners,
)
from src.vision.table_calibration import TableCalibration  # noqa: E402

WIDTH, HEIGHT = 1280, 720
TABLE_Z = 0.750
CAM_POS = np.array([0.070, 0.0, 1.600])
FOVY = 85.0


def ground_truth(pitch_deg=32.0, yaw_deg=4.0, roll_deg=3.0, fovy=FOVY, cam_pos=CAM_POS):
    """Camera looking down at the table: returns (rvec, tvec, K, R_cam_in_base)."""
    pitch, yaw, roll = np.radians([pitch_deg, yaw_deg, roll_deg])
    forward = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), -np.sin(pitch)])
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    cos_r, sin_r = np.cos(roll), np.sin(roll)
    R_cam_in_base = np.column_stack([cos_r * right + sin_r * down,
                                     -sin_r * right + cos_r * down,
                                     forward])
    R = R_cam_in_base.T
    focal = HEIGHT / (2.0 * np.tan(np.radians(fovy) / 2.0))
    K = np.array([[focal, 0.0, WIDTH / 2.0], [0.0, focal, HEIGHT / 2.0], [0.0, 0.0, 1.0]])
    return cv2.Rodrigues(R)[0], (-R @ cam_pos).reshape(3, 1), K, R_cam_in_base


def project(points, rvec, tvec, K, dist=None):
    dist = np.zeros((5, 1)) if dist is None else dist
    return cv2.projectPoints(np.asarray(points, float), rvec, tvec, K, dist)[0].reshape(-1, 2)


@pytest.fixture
def target():
    return world_corners(near_x=0.300, center_y=0.0, side=0.500, depth=0.400, table_z=TABLE_Z)


def test_world_corners_follow_the_base_frame(target):
    near_left, near_right, far_right, far_left = target
    assert near_left[1] > near_right[1], "+y must be the robot's left"
    assert far_left[0] > near_left[0], "+x must point away from the robot"
    assert np.allclose(target[:, 2], TABLE_Z)


def test_recovers_a_known_pose_without_distortion(target):
    rvec, tvec, K, R_cam_in_base = ground_truth()
    pixels = project(target, rvec, tvec, K)

    focal = estimate_focal(target, pixels, (WIDTH, HEIGHT))
    assert focal == pytest.approx(K[1, 1], rel=1e-3), "focal from the rectangle's own perspective"

    K_est = np.array([[focal, 0, WIDTH / 2], [0, focal, HEIGHT / 2], [0, 0, 1.0]])
    rvec_fit, tvec_fit, error_px = solve_pose(target, pixels, K_est, np.zeros((5, 1)))
    assert error_px < 0.1

    recovered = (-cv2.Rodrigues(rvec_fit)[0].T @ tvec_fit).ravel()
    assert recovered == pytest.approx(CAM_POS, abs=2e-3)
    assert cv2.Rodrigues(rvec_fit)[0].T == pytest.approx(R_cam_in_base, abs=2e-3)


def test_recovers_a_known_pose_through_lens_distortion(target):
    """The measured-intrinsics path has to be exact even on a wide-angle lens."""
    rvec, tvec, K, _ = ground_truth()
    dist = np.array([[-0.32, 0.12, 0.0, 0.0, -0.02]]).T
    pixels = project(target, rvec, tvec, K, dist)

    rvec_fit, tvec_fit, error_px = solve_pose(target, pixels, K, dist)
    assert error_px < 0.1
    recovered = (-cv2.Rodrigues(rvec_fit)[0].T @ tvec_fit).ravel()
    assert recovered == pytest.approx(CAM_POS, abs=2e-3)


def test_focal_estimate_survives_a_roll_free_view(target):
    """
    The orthogonal-vanishing-point constraint alone degenerates when the camera
    has no roll relative to the target -- the usual mounting. The other
    constraint has to carry it.
    """
    rvec, tvec, K, _ = ground_truth(yaw_deg=0.0, roll_deg=0.0)
    pixels = project(target, rvec, tvec, K)
    assert estimate_focal(target, pixels, (WIDTH, HEIGHT)) == pytest.approx(K[1, 1], rel=1e-3)


def test_homography_maps_pixels_back_onto_the_tabletop(target):
    rvec, tvec, K, _ = ground_truth()
    probes = np.array([[0.40, 0.10, TABLE_Z], [0.65, -0.15, TABLE_Z], [0.50, 0.0, TABLE_Z]])
    pixels = project(probes, rvec, tvec, K)

    rvec_fit, tvec_fit, _ = solve_pose(target, project(target, rvec, tvec, K), K, np.zeros((5, 1)))
    homography = plane_homography(rvec_fit, tvec_fit, K, TABLE_Z)
    mapped = pixel_to_table(pixels, homography, K, np.zeros((5, 1)))
    assert mapped == pytest.approx(probes[:, :2], abs=1e-3)


def test_mujoco_quaternion_points_the_camera_the_same_way(target):
    """MuJoCo cameras look down -z with +y up; OpenCV looks down +z with +y down."""
    rvec, tvec, K, R_cam_in_base = ground_truth()
    quat = rotation_to_quat(R_cam_in_base @ np.diag([1.0, -1.0, -1.0]))
    w, x, y, z = quat
    R_mujoco = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    assert R_mujoco.T @ R_mujoco == pytest.approx(np.eye(3), abs=1e-9)
    assert -R_mujoco[:, 2] == pytest.approx(R_cam_in_base[:, 2], abs=1e-9), "view direction"
    assert R_mujoco[2, 1] > 0, "camera up must point up in the base frame"


def test_edge_clicks_and_corner_clicks_agree_when_noise_free(target):
    rvec, tvec, K, _ = ground_truth()
    corners = project(target, rvec, tvec, K)
    # EDGE_ORDER is near, left, far, right; corners are near-left, near-right, far-right, far-left.
    lines = []
    for a, b in ((0, 1), (0, 3), (3, 2), (1, 2)):
        ts = np.linspace(0.05, 0.95, 5)[:, None]
        lines.append(fit_line(corners[a] + (corners[b] - corners[a]) * ts))
    assert corners_from_edges(lines) == pytest.approx(corners, abs=0.5)


def test_calibration_file_round_trips_through_the_loader(target, tmp_path, monkeypatch):
    """End to end: run the script headless, then read it back with TableCalibration."""
    import calibrate_table_tape as script

    rvec, tvec, K, _ = ground_truth()
    dist = np.array([[-0.32, 0.12, 0.0, 0.0, -0.02]]).T
    pixels = project(target, rvec, tvec, K, dist)

    intrinsics_path = tmp_path / "intrinsics.yaml"
    fs = cv2.FileStorage(str(intrinsics_path), cv2.FILE_STORAGE_WRITE)
    fs.write("image_width", WIDTH)
    fs.write("image_height", HEIGHT)
    fs.write("camera_matrix", K)
    fs.write("dist_coeffs", dist)
    fs.release()

    frame_path = tmp_path / "frame.png"
    cv2.imwrite(str(frame_path), np.zeros((HEIGHT, WIDTH, 3), np.uint8))
    out_path = tmp_path / "extrinsics.yaml"

    monkeypatch.setattr(sys, "argv", [
        "calibrate_table_tape.py",
        "--config", str(ROOT / "configs/task_config.yaml"),
        "--image", str(frame_path),
        "--intrinsics", str(intrinsics_path),
        "--out", str(out_path),
        "--side", "500", "--depth", "400",
        "--near-edge-x", "0.300", "--table-z", str(TABLE_Z),
        "--points", ";".join(f"{u},{v}" for u, v in pixels),
    ])
    script.main()

    calib = TableCalibration.load(str(out_path))
    assert calib.intrinsics_estimated is False
    assert calib.table_z == pytest.approx(TABLE_Z)
    assert calib.image_size == (WIDTH, HEIGHT)
    assert calib.pixels_to_table(pixels) == pytest.approx(target[:, :2], abs=2e-3)

    probe = np.array([[0.55, 0.12, TABLE_Z]])
    probe_px = project(probe, rvec, tvec, K, dist)[0]
    assert calib.pixel_to_base(*probe_px) == pytest.approx(probe[0], abs=3e-3)
    assert calib.table_to_pixel(0.55, 0.12) == pytest.approx(tuple(probe_px), abs=1.0)


def test_rejects_a_fronto_parallel_view(target):
    """Looking straight down carries no focal-length information; say so instead of guessing."""
    rvec, tvec, K, _ = ground_truth(pitch_deg=90.0, yaw_deg=0.0, roll_deg=0.0,
                                    cam_pos=np.array([0.50, 0.0, 1.60]))
    pixels = project(target, rvec, tvec, K)
    with pytest.raises(ValueError, match="fronto-parallel|non-physical|focal"):
        estimate_focal(target, pixels, (WIDTH, HEIGHT))


def test_pose_solver_handles_a_target_off_the_object_frame_origin(target):
    """
    Regression: the tabletop sits at z = 0.75, not z = 0. OpenCV 5's planar solver
    returns success-with-NaN for a target that is not in the z = 0 plane, which
    would otherwise be written straight into the calibration file.
    """
    rvec, tvec, K, _ = ground_truth()
    pixels = project(target, rvec, tvec, K)
    rvec_fit, tvec_fit, error_px = solve_pose(target, pixels, K, np.zeros((5, 1)))

    assert np.all(np.isfinite(rvec_fit)) and np.all(np.isfinite(tvec_fit))
    assert error_px < 0.1
    # Returned in the base frame, so it must reproject base-frame points directly.
    reprojected = project(target, rvec_fit, tvec_fit, K)
    assert reprojected == pytest.approx(pixels, abs=0.1)
    assert (-cv2.Rodrigues(rvec_fit)[0].T @ tvec_fit).ravel() == pytest.approx(CAM_POS, abs=2e-3)
