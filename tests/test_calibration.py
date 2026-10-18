"""FR8: camera model from intrinsics + nominal mount, or + touch calibration."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from robotokens.vision.calib_tools import solve_pose
from robotokens.vision.calibration import CameraModel, load_camera_model, nominal_pose, save_extrinsics


def test_nominal_pose_from_mount(cfg):
    pos, R = nominal_pose(cfg)
    # Camera 7 cm in front of the torso, 18 cm above it (1.42 m), base 0.633 m above the floor.
    np.testing.assert_allclose(pos, [0.07, 0.0, 1.60 - 0.633], atol=1e-9)
    optical_axis = R[:, 2]
    assert np.degrees(np.arcsin(-optical_axis[2])) == pytest.approx(cfg.workcell.camera.pitch_down_deg)
    np.testing.assert_allclose(R[:, 0], [0, -1, 0], atol=1e-12)  # image right = robot -Y


def test_load_without_extrinsics_uses_nominal(cfg, tmp_path):
    cam = load_camera_model(cfg, extrinsics_path=tmp_path / "missing.yaml")
    assert cam.source == "nominal"
    np.testing.assert_allclose(cam.position, nominal_pose(cfg)[0])


def synthetic_camera():
    K = np.array([[700.0, 0, 640], [0, 700.0, 360], [0, 0, 1]])
    dist = np.array([-0.3, 0.1, 0.001, -0.001, -0.02])
    pos = np.array([0.08, 0.02, 0.95])
    pitch = np.radians(68)
    x = np.array([0.0, -1.0, 0.0])
    z = np.array([np.cos(pitch), 0.0, -np.sin(pitch)])
    R = np.stack([x, np.cross(z, x), z], axis=1)
    return CameraModel(K=K, dist=dist, R=R, position=pos, image_size=(1280, 720), source="test")


def test_pixel_to_plane_round_trip():
    cam = synthetic_camera()
    pts = np.array([[0.25, 0.1, 0.092], [0.35, 0.3, 0.212], [0.18, -0.02, 0.092]])
    uv = cam.project(pts)
    for (u, v), p in zip(uv, pts):
        x, y = cam.pixel_to_plane(u, v, p[2])
        assert (x, y) == pytest.approx((p[0], p[1]), abs=1e-6)


def test_touch_rotation_fit_recovers_tilt_with_click_noise():
    """Points on one tilted plane + ~1 cm click noise + a mis-click: the rotation (position fixed
    to the measured mount) is still recovered; the mis-click is flagged, not fitted."""
    cam = synthetic_camera()
    rng = np.random.default_rng(0)
    x = rng.uniform(0.22, 0.47, 34)
    pts = np.column_stack([x, rng.uniform(0.02, 0.36, 34), 0.16 + 0.8 * (x - 0.22)])  # nearly planar, like the arm's reach
    px = cam.project(pts) + rng.normal(0, 8.0, (34, 2))
    px[5] += [90.0, -60.0]  # one mis-click
    tilt = np.radians(6)  # start 6 deg off in pitch (eyeballed mount)
    R_init = cam.R @ np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]])
    R, residuals, inliers = solve_pose(pts, px, cam.K, cam.dist, cam.position, R_init)
    angle = np.degrees(np.arccos(np.clip((np.trace(cam.R.T @ R) - 1) / 2, -1, 1)))
    assert angle < 0.5
    assert not inliers[5] and inliers.sum() >= 30
    assert np.median(residuals) < 15


def test_saved_touch_extrinsics_override_nominal(cfg, tmp_path):
    path = tmp_path / "ext.yaml"
    cam = synthetic_camera()
    save_extrinsics(path, cam.R, cam.position, {"method": "robot_touch"})
    loaded = load_camera_model(cfg, extrinsics_path=path)
    assert loaded.source == "touch"
    np.testing.assert_allclose(loaded.position, cam.position)
    assert yaml.safe_load(path.read_text())["method"] == "robot_touch"


def test_plausibility_gate():  # FR8: a bad touch fit never replaces the camera model
    from robotokens.vision.calib_tools import implausible

    R_nom = synthetic_camera().R
    ok = implausible(np.array([0.08, 0.0, 0.96]), R_nom, np.array([0.07, 0.0, 0.967]), R_nom, kept=30, total=34)
    assert ok is None
    assert "points" in implausible(np.array([0.07, 0, 0.967]), R_nom, np.array([0.07, 0, 0.967]), R_nom, 11, 34)
    assert "position" in implausible(np.array([0.30, -0.04, 0.92]), R_nom, np.array([0.07, 0, 0.967]), R_nom, 30, 34)
    yawed = np.array([[np.cos(0.7), -np.sin(0.7), 0], [np.sin(0.7), np.cos(0.7), 0], [0, 0, 1]]) @ R_nom
    assert "rotation" in implausible(np.array([0.07, 0, 0.967]), yawed, np.array([0.07, 0, 0.967]), R_nom, 30, 34)
