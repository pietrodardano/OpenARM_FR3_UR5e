"""Camera model in the robot base frame (FR8): intrinsics + pose, pixel <-> base frame.

Two modes:

- **intrinsics only**: ChArUco intrinsics (``camera.intrinsics_path``) + the nominal mount pose
  from ``workcell.camera`` (CAD numbers; expect a few cm of error).
- **intrinsics + touch**: ``robotokens calib touch`` solves the pose from robot fingertip touches
  and writes ``camera.extrinsics_path``; whenever that file exists it overrides the nominal pose.

The RGB camera has no depth: a pixel becomes a ray, intersected with a horizontal plane of known
height (the table, or the top face of an object on it).
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import yaml
from numpy.typing import NDArray

from ..config import Config, resolve

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CameraModel:
    K: NDArray  # 3x3
    dist: NDArray  # OpenCV distortion coefficients
    R: NDArray  # 3x3, columns = OpenCV camera axes (x right, y down, z forward) in the base frame
    position: NDArray  # (3,) camera centre, base frame, m
    image_size: Tuple[int, int]  # (width, height)
    source: str  # "nominal" | "touch"

    def pixel_to_plane(self, u: float, v: float, z: float) -> Tuple[float, float]:
        """(x, y) where the pixel's ray meets the horizontal plane at base-frame height ``z``."""
        n = cv2.undistortPoints(np.array([[[u, v]]], dtype=np.float64), self.K, self.dist).reshape(2)
        ray = self.R @ np.array([n[0], n[1], 1.0])
        if abs(ray[2]) < 1e-9:
            raise ValueError("camera ray is parallel to the plane")
        lam = (z - self.position[2]) / ray[2]
        if lam <= 0:
            raise ValueError("the plane is behind the camera along this ray")
        p = self.position + lam * ray
        return float(p[0]), float(p[1])

    def project(self, points: NDArray) -> NDArray:
        """Base-frame points (n, 3) -> pixels (n, 2), distortion included."""
        rvec, _ = cv2.Rodrigues(self.R.T)
        tvec = -self.R.T @ self.position
        px, _ = cv2.projectPoints(np.asarray(points, dtype=np.float64).reshape(-1, 3), rvec, tvec, self.K, self.dist)
        return px.reshape(-1, 2)


def load_intrinsics(path: str | Path, image_size: Tuple[int, int]) -> Tuple[NDArray, NDArray]:
    """Reads the ChArUco calibration (OpenCV FileStorage); rescales K to ``image_size`` (same aspect)."""
    p = resolve(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found: run `robotokens calib intrinsics` first")
    fs = cv2.FileStorage(str(p), cv2.FILE_STORAGE_READ)
    try:
        K, dist = fs.getNode("camera_matrix").mat(), fs.getNode("dist_coeffs").mat()
        w, h = fs.getNode("image_width").real(), fs.getNode("image_height").real()
    finally:
        fs.release()
    if K is None:
        raise ValueError(f"{p} has no camera_matrix")
    if (int(w), int(h)) != tuple(image_size):
        if abs(w / h - image_size[0] / image_size[1]) > 1e-3:
            raise ValueError(f"{p} was calibrated at {int(w)}x{int(h)}, a different aspect ratio than {image_size}")
        K = K.copy()
        K[:2, :] *= image_size[0] / w
    return K, dist.ravel()


def nominal_pose(cfg: Config) -> Tuple[NDArray, NDArray]:
    """Camera position and rotation from the CAD mount: facing +X, no roll, pitched down."""
    cam, wc = cfg.workcell.camera, cfg.workcell
    pitch = np.radians(cam.pitch_down_deg)
    x = np.array([0.0, -1.0, 0.0])  # image right = robot right (-Y)
    z = np.array([np.cos(pitch), 0.0, -np.sin(pitch)])  # optical axis
    R = np.stack([x, np.cross(z, x), z], axis=1)
    pos = np.array([cam.forward_offset_m, 0.0, wc.torso.height_m + cam.height_above_torso_m - cfg.base_z_in_world])
    return pos, R


def save_extrinsics(path: str | Path, R: NDArray, position: NDArray, meta: Dict[str, Any]) -> None:
    """Writes the touch-calibrated pose (plus metadata) as YAML."""
    data = {"generated": datetime.datetime.now().isoformat(timespec="seconds"), **meta,
            "frame": "robot base (+x forward, +y left, +z up)",
            "position_m": [float(v) for v in position],
            "rotation_opencv": [[float(v) for v in row] for row in R]}
    resolve(path).write_text(yaml.safe_dump(data, sort_keys=False))


def load_camera_model(cfg: Config, extrinsics_path: Optional[str | Path] = None) -> CameraModel:
    """Intrinsics + the touch pose if ``extrinsics_path`` exists, else the nominal mount pose."""
    size = (cfg.camera.width, cfg.camera.height)
    K, dist = load_intrinsics(cfg.camera.intrinsics_path, size)
    ext = resolve(extrinsics_path or cfg.camera.extrinsics_path)
    if ext.exists():
        data = yaml.safe_load(ext.read_text())
        pos, R, source = np.array(data["position_m"], float), np.array(data["rotation_opencv"], float), "touch"
    else:
        pos, R = nominal_pose(cfg)
        source = "nominal"
        logger.warning("no %s: using the nominal camera mount (expect cm errors; run `robotokens calib touch`)",
                       ext.name)
    logger.info("camera model: intrinsics + %s pose", source)
    return CameraModel(K=K, dist=np.asarray(dist, float), R=R, position=pos, image_size=size, source=source)
