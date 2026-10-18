"""
Tests for the demo's vision detector (src/vision/vision_detector.py) without torch:
a synthetic calibrated camera and a stub localizer stand in for the camera and GroundingDINO+SAM2.
"""

import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

TABLE_Z = 0.092  # base frame, 0.725 m table


def synthetic_calibration():
    """Camera on the torso mount, looking forward and 35 deg down, as a TableCalibration."""
    from src.vision.table_calibration import TableCalibration

    K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
    pitch = np.radians(35.0)
    # OpenCV camera axes in the base frame: x right (-Y), y down, z forward + down.
    z_c = np.array([np.cos(pitch), 0.0, -np.sin(pitch)])
    x_c = np.array([0.0, -1.0, 0.0])
    y_c = np.cross(z_c, x_c)
    R_cam_in_base = np.column_stack([x_c, y_c, z_c])
    cam_pos = np.array([0.07, 0.0, 0.97])  # 1.60 m above the floor
    rvec, _ = cv2.Rodrigues(R_cam_in_base.T)
    tvec = -R_cam_in_base.T @ cam_pos
    # homography pixel -> table (x, y): from 4 projected table points
    pts = np.array([[0.2, -0.1, TABLE_Z], [0.5, -0.1, TABLE_Z], [0.5, 0.4, TABLE_Z], [0.2, 0.4, TABLE_Z]])
    pix, _ = cv2.projectPoints(pts, rvec, tvec, K, np.zeros(5))
    H, _ = cv2.findHomography(pix.reshape(-1, 2), pts[:, :2])
    calib = TableCalibration(camera_matrix=K, dist_coeffs=np.zeros((5, 1)), homography_plane_from_pixel=H,
                             table_z=TABLE_Z, image_size=(640, 480), rotate_180=False, intrinsics_estimated=False,
                             camera_position=cam_pos, camera_rotation=R_cam_in_base)
    return calib, rvec, tvec, K


def project(p, rvec, tvec, K):
    uv, _ = cv2.projectPoints(np.asarray([p], dtype=np.float64), rvec, tvec, K, np.zeros(5))
    return uv.reshape(2)


class StubLocalizer:
    """Returns one detection per can: box + top-face centroid, as ObjectLocalizer would."""

    def __init__(self, cans, rvec, tvec, K, top=0.12, jitter_px=0.0, extra=None):
        self.cans, self.rvec, self.tvec, self.K, self.top = cans, rvec, tvec, K, top
        self.rng = np.random.default_rng(0)
        self.jitter = jitter_px
        self.extra = extra or []
        self.calls = 0

    def locate(self, frame, prompt):
        self.calls += 1
        out = []
        for x, y in self.cans:
            c = project([x, y, TABLE_Z + self.top], self.rvec, self.tvec, self.K) + self.rng.normal(0, self.jitter, 2)
            b = project([x, y, TABLE_Z], self.rvec, self.tvec, self.K)
            out.append(SimpleNamespace(label="soda can", score=0.8, face=SimpleNamespace(centroid=c),
                                       box_xyxy=np.array([b[0] - 20, c[1] - 10, b[0] + 20, b[1]])))
        out.extend(self.extra[self.calls - 1] if self.calls - 1 < len(self.extra) else [])
        return out


@unittest.skipIf(cv2 is None, "OpenCV not installed (uv sync --extra calib)")
class TestVisionDetector(unittest.TestCase):
    def setUp(self):
        from src.vision.vision_detector import VisionDetectorConfig

        self.calib, self.rvec, self.tvec, self.K = synthetic_calibration()
        self.cfg = VisionDetectorConfig(frames=3, workspace=((0.12, 0.45), (-0.05, 0.45)))
        self.cans = [(0.21, 0.13), (0.21, 0.27)]

    def _detector(self, localizer):
        from src.vision.vision_detector import VisionObjectDetector

        return VisionObjectDetector(self.cfg, self.calib, lambda: np.zeros((480, 640, 3), np.uint8), localizer)

    def test_pixel_to_plane_inverts_projection(self):
        p = [0.3, 0.2, TABLE_Z + 0.12]
        u, v = project(p, self.rvec, self.tvec, self.K)
        np.testing.assert_allclose(self.calib.pixel_to_plane(u, v, p[2]), p[:2], atol=1e-6)

    def test_top_face_centroid_maps_to_can_position(self):
        found = self._detector(StubLocalizer(self.cans, self.rvec, self.tvec, self.K, jitter_px=0.5)).detect("soda can")
        self.assertEqual(len(found), 2)
        for d, (x, y) in zip(found, self.cans):
            self.assertLess(np.hypot(d.x - x, d.y - y), 0.005)
            self.assertEqual(d.frame, "base")

    def test_table_homography_on_top_face_would_be_wrong(self):
        """Why the ray-plane mapping is needed: the top face is 12 cm above the table."""
        u, v = project([0.21, 0.13, TABLE_Z + 0.12], self.rvec, self.tvec, self.K)
        x, y = self.calib.pixel_to_table(u, v)
        self.assertGreater(np.hypot(x - 0.21, y - 0.13), 0.02)  # ~3 cm here: more than the layout gaps

    def test_transient_and_out_of_workspace_detections_are_dropped(self):
        ghost = SimpleNamespace(label="soda can", score=0.9, face=SimpleNamespace(
            centroid=project([0.33, 0.05, TABLE_Z + 0.12], self.rvec, self.tvec, self.K)), box_xyxy=np.zeros(4))
        far = SimpleNamespace(label="soda can", score=0.9, face=SimpleNamespace(
            centroid=project([0.80, 0.20, TABLE_Z + 0.12], self.rvec, self.tvec, self.K)), box_xyxy=np.zeros(4))
        loc = StubLocalizer(self.cans, self.rvec, self.tvec, self.K, extra=[[ghost, far], [far], [far]])
        found = self._detector(loc).detect("soda can")
        self.assertEqual(len(found), 2)  # ghost seen in 1/3 frames, far object outside the workspace
        self.assertEqual(loc.calls, 3)

    def test_feeds_the_task_state_machine(self):
        import yaml

        from src.task import DemoConfig, Phase, WeighAndSortTask, parse_prompt
        from src.task.mocks import MockManipulationSkills, MockPayloadEstimator

        with open(os.path.join(os.path.dirname(__file__), "..", "configs", "task_config.yaml")) as f:
            cfg = yaml.safe_load(f)
        det = self._detector(StubLocalizer(self.cans, self.rvec, self.tvec, self.K))
        task = WeighAndSortTask(det, MockManipulationSkills(), MockPayloadEstimator({"object_1": 0.35, "object_2": 0.02}),
                                DemoConfig.from_config(cfg), parse_prompt("put the lighter can in the left bin"))
        report = task.run()
        self.assertEqual(report.outcome, Phase.DONE, report.error)
        self.assertEqual(report.placements, {"object_2": "left", "object_1": "right"})


if __name__ == "__main__":
    unittest.main()
