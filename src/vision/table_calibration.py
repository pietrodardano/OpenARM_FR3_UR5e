"""
Consumer for the tape-rectangle calibration written by
scripts/calibrate_table_tape.py.

Turns a pixel on the tabletop into a grasp target in the robot base frame
(+x forward, +y left, +z up), and back:

    from src.vision.table_calibration import TableCalibration

    calib = TableCalibration.load("configs/camera_extrinsics.yaml")
    x, y = calib.pixel_to_table(cx, cy)          # metres, base frame
    target = calib.pixel_to_base(cx, cy)         # (x, y, table_z)

The mapping is only valid for points lying on the tabletop plane, and only
while neither the camera nor the table moves. Re-run the calibration after
either is touched.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np
import yaml


@dataclass(frozen=True)
class TableCalibration:
    camera_matrix: np.ndarray          # 3x3, pixels
    dist_coeffs: np.ndarray            # (n, 1)
    homography_plane_from_pixel: np.ndarray  # 3x3, undistorted pixel -> (x, y) on the table
    table_z: float                     # tabletop height in the base frame, metres
    image_size: tuple[int, int]        # (width, height) the calibration was taken at
    rotate_180: bool                   # whether the calibrated frame was rotated 180 deg
    intrinsics_estimated: bool         # True => camera pose / fovy are not trustworthy
    warnings: tuple[str, ...] = ()
    camera_position: np.ndarray | None = None   # (3,) camera centre in the base frame, m
    camera_rotation: np.ndarray | None = None   # 3x3, OpenCV camera axes expressed in the base frame

    @classmethod
    def load(cls, path: str = "configs/camera_extrinsics.yaml") -> "TableCalibration":
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Run: uv run python scripts/calibrate_table_tape.py")
        with open(path) as handle:
            data = yaml.safe_load(handle)
        intrinsics = data["intrinsics"]
        camera_matrix = np.array([[intrinsics["fx"], 0.0, intrinsics["cx"]],
                                  [0.0, intrinsics["fy"], intrinsics["cy"]],
                                  [0.0, 0.0, 1.0]])
        return cls(
            camera_matrix=camera_matrix,
            dist_coeffs=np.array(intrinsics["dist_coeffs"], dtype=np.float64).reshape(-1, 1),
            homography_plane_from_pixel=np.array(
                data["table_plane"]["homography_plane_from_pixel"], dtype=np.float64),
            table_z=float(data["table_plane"]["z_m"]),
            image_size=(int(data["image_width"]), int(data["image_height"])),
            rotate_180=bool(data.get("rotate_180_applied", False)),
            intrinsics_estimated=bool(intrinsics.get("estimated", False)),
            warnings=tuple(data.get("warnings", ())),
            camera_position=(np.array(data["camera_in_base_frame"]["position_m"], dtype=np.float64)
                             if "camera_in_base_frame" in data else None),
            camera_rotation=(np.array(data["camera_in_base_frame"]["rotation_opencv"], dtype=np.float64)
                             if "camera_in_base_frame" in data else None),
        )

    def pixel_to_table(self, u: float, v: float) -> tuple[float, float]:
        """Pixel on the tabletop -> (x, y) in the robot base frame, metres."""
        return tuple(self.pixels_to_table(np.array([[u, v]], dtype=np.float64))[0])

    def pixels_to_table(self, pixels: np.ndarray) -> np.ndarray:
        """Vectorised pixel_to_table. Takes an (n, 2) array, returns (n, 2) in metres."""
        pts = np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
        undistorted = cv2.undistortPoints(pts, self.camera_matrix, self.dist_coeffs,
                                          P=self.camera_matrix).reshape(-1, 2)
        homogeneous = np.hstack([undistorted, np.ones((len(undistorted), 1))])
        mapped = (self.homography_plane_from_pixel @ homogeneous.T).T
        return mapped[:, :2] / mapped[:, 2:3]

    def pixel_to_plane(self, u: float, v: float, z: float) -> tuple[float, float]:
        """Pixel -> (x, y) where its camera ray meets the horizontal plane at base-frame height ``z``.

        Use it for points that are not on the tabletop, e.g. the centroid of an object's top face
        at ``table_z + object_height``: the table homography would place it too far from the
        camera (parallax). Needs the camera pose from the calibration file.
        """
        if self.camera_position is None or self.camera_rotation is None:
            raise ValueError("calibration file has no camera_in_base_frame pose; re-run the calibration")
        undistorted = cv2.undistortPoints(np.array([[[u, v]]], dtype=np.float64), self.camera_matrix,
                                          self.dist_coeffs).reshape(2)
        ray = self.camera_rotation @ np.array([undistorted[0], undistorted[1], 1.0])
        if abs(ray[2]) < 1e-9:
            raise ValueError("camera ray is parallel to the plane")
        lam = (z - self.camera_position[2]) / ray[2]
        if lam <= 0:
            raise ValueError("the plane is behind the camera along this ray")
        p = self.camera_position + lam * ray
        return float(p[0]), float(p[1])

    def pixel_to_base(self, u: float, v: float) -> np.ndarray:
        """Pixel on the tabletop -> (x, y, z) in the robot base frame, metres."""
        x, y = self.pixel_to_table(u, v)
        return np.array([x, y, self.table_z])

    def table_to_pixel(self, x: float, y: float) -> tuple[float, float]:
        """Inverse mapping: a point on the tabletop -> its pixel (distortion included)."""
        h_pixel_from_plane = np.linalg.inv(self.homography_plane_from_pixel)
        undistorted = h_pixel_from_plane @ np.array([x, y, 1.0])
        undistorted = undistorted[:2] / undistorted[2]
        # Re-apply distortion: undistorted pixel -> normalised ray -> projectPoints.
        inv_k = np.linalg.inv(self.camera_matrix)
        ray = inv_k @ np.array([undistorted[0], undistorted[1], 1.0])
        projected, _ = cv2.projectPoints(ray.reshape(1, 3), np.zeros(3), np.zeros(3),
                                         self.camera_matrix, self.dist_coeffs)
        return tuple(projected.reshape(2))
