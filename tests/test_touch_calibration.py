"""
Robot-touch camera calibration: does it recover a known camera pose from touched
points, drop a mis-clicked one, and refuse to pretend a coplanar set is fine?
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibrate_camera_touch import (  # noqa: E402
    build_payload, load_session, planarity_mm, save_session, solve_touch_pose,
    solve_with_outlier_rejection, tip_offset_check,
)
from src.vision.table_calibration import TableCalibration  # noqa: E402
from test_table_calibration import CAM_POS, TABLE_Z, ground_truth, project  # noqa: E402

WIDTH, HEIGHT = 1280, 720
DIST = np.array([[-0.32, 0.12, 0.0, 0.0, -0.02]]).T


def touched_points(heights=(0.0, 0.10, 0.20)) -> np.ndarray:
    """A spread of gripper-tip positions over the table, at several heights."""
    grid = [(x, y) for x in np.linspace(0.34, 0.70, 4) for y in np.linspace(-0.26, 0.26, 3)]
    points = [[x, y, TABLE_Z] for x, y in grid]
    for height in heights[1:]:
        points += [[x, y, TABLE_Z + height] for x, y in grid[::3]]
    return np.array(points)


def test_recovers_a_known_camera_pose_from_touched_points():
    points = touched_points()
    rvec, tvec, K, R_cam_in_base = ground_truth()
    pixels = project(points, rvec, tvec, K, DIST)

    rvec_fit, tvec_fit, residuals = solve_touch_pose(points, pixels, K, DIST)
    assert residuals.max() < 0.1
    assert (-cv2.Rodrigues(rvec_fit)[0].T @ tvec_fit).ravel() == pytest.approx(CAM_POS, abs=1e-3)
    assert cv2.Rodrigues(rvec_fit)[0].T == pytest.approx(R_cam_in_base, abs=1e-3)


def test_survives_realistic_click_and_fk_noise():
    """~1.5 px clicks and 2 mm of touch error should still land within a centimetre."""
    points = touched_points()
    rvec, tvec, K, _ = ground_truth()
    rng = np.random.default_rng(3)
    errors = []
    for _ in range(30):
        actual = points + rng.normal(0, 0.002, points.shape)
        pixels = project(actual, rvec, tvec, K, DIST) + rng.normal(0, 1.5, (len(points), 2))
        rvec_fit, tvec_fit, _ = solve_touch_pose(points, pixels, K, DIST)
        recovered = (-cv2.Rodrigues(rvec_fit)[0].T @ tvec_fit).ravel()
        errors.append(np.linalg.norm(recovered - CAM_POS))
    assert np.median(errors) < 0.03, f"median camera position error {np.median(errors) * 1000:.0f} mm"


def test_drops_a_misclicked_point_and_recovers():
    points = touched_points()
    rvec, tvec, K, _ = ground_truth()
    pixels = project(points, rvec, tvec, K, DIST)
    pixels[5] += np.array([60.0, -45.0])  # one tip clicked on the wrong thing

    rvec_fit, tvec_fit, residuals, kept, _, dropped = solve_with_outlier_rejection(
        points, pixels, K, DIST, max_residual_px=4.0)

    assert [index for index, _ in dropped] == [5]
    assert len(kept) == len(points) - 1
    assert residuals.max() < 0.1
    assert (-cv2.Rodrigues(rvec_fit)[0].T @ tvec_fit).ravel() == pytest.approx(CAM_POS, abs=1e-3)


def test_flags_a_coplanar_point_set():
    flat = touched_points(heights=(0.0,))
    spread = touched_points()
    assert planarity_mm(flat) < 1.0
    assert planarity_mm(spread) > 20.0

    rvec, tvec, K, _ = ground_truth()
    payload = build_payload(flat, project(flat, rvec, tvec, K, DIST), np.zeros(len(flat)), K, DIST,
                            "measured", False, rvec, tvec, (WIDTH, HEIGHT), TABLE_Z, False, "right")
    assert any("coplanar" in w for w in payload["warnings"])


def test_tip_offset_check_reports_the_gap_to_the_table():
    points = touched_points()
    points[:, 2] += 0.004  # as if the fingertips sat 4 mm above the TCP site
    message = tip_offset_check(points, TABLE_Z)
    assert "+4.0 mm" in message


def test_session_round_trips_and_output_loads_as_a_calibration(tmp_path):
    points = touched_points()
    rvec, tvec, K, _ = ground_truth()
    pixels = project(points, rvec, tvec, K, DIST)
    frames = [np.zeros((HEIGHT, WIDTH, 3), np.uint8) for _ in points]

    save_session(str(tmp_path / "s"), points, frames, "right", True)
    loaded_points, loaded_frames, meta = load_session(str(tmp_path / "s"))
    assert loaded_points == pytest.approx(points)
    assert len(loaded_frames) == len(points)
    assert meta["side"] == "right" and meta["rotate_180"] is True

    payload = build_payload(points, pixels, np.zeros(len(points)), K, DIST, "measured (test)",
                            False, rvec, tvec, (WIDTH, HEIGHT), TABLE_Z, True, "right")
    out = tmp_path / "extrinsics.yaml"
    from calibrate_table_tape import write_extrinsics
    write_extrinsics(str(out), payload)

    calib = TableCalibration.load(str(out))
    assert calib.rotate_180 is True
    assert calib.intrinsics_estimated is False
    probe = np.array([[0.52, 0.08, TABLE_Z]])
    probe_px = project(probe, rvec, tvec, K, DIST)[0]
    assert calib.pixel_to_base(*probe_px) == pytest.approx(probe[0], abs=2e-3)
