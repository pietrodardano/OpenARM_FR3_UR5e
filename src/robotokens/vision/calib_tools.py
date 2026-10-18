"""Calibration solvers (FR8): ChArUco intrinsics and the robot-touch camera pose.

Touch calibration: the gripper tip (= TCP with closed fingers) is moved to 12-20 points spread
over the pick area at 2-3 heights; each point's base-frame position comes from the robot's own
forward kinematics, its pixel from a click; the camera rotation is fitted with the measured mount
position fixed (see solve_pose). Errors in the robot's FK
are shared with the controller that later grasps, so camera and arm agree where it matters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

MIN_TOUCH_POINTS = 6


def charuco_board(square_mm: float, marker_mm: float, squares: Tuple[int, int] = (5, 7),
                  dictionary: str = "6X6_250") -> "cv2.aruco.CharucoBoard":
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, f"DICT_{dictionary}"))
    return cv2.aruco.CharucoBoard(squares, square_mm / 1000.0, marker_mm / 1000.0, d)


def board_corners(gray: NDArray, board: "cv2.aruco.CharucoBoard") -> int:
    """Number of ChArUco corners found (0 = board not seen)."""
    corners, _ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(gray)
    return 0 if corners is None else len(corners)


def calibrate_intrinsics(images: Iterable[NDArray], board: "cv2.aruco.CharucoBoard"
                         ) -> Tuple[NDArray, NDArray, float, Tuple[int, int], int]:
    """(K, dist, rms px, (width, height), images used) from BGR photos of the board."""
    detector = cv2.aruco.CharucoDetector(board)
    obj, img, size = [], [], None
    for frame in images:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        corners, ids, _, _ = detector.detectBoard(gray)
        if corners is None or len(corners) < 6:
            continue
        o, i = board.matchImagePoints(corners, ids)
        if o is not None and len(o) >= 6:
            obj.append(o)
            img.append(i)
    if len(obj) < 10:
        raise ValueError(f"only {len(obj)} usable board views; take 20-30 sharp photos at varied angles")
    rms, K, dist, _, _ = cv2.calibrateCamera(obj, img, size, None, None)
    return K, dist.ravel(), float(rms), size, len(obj)


def write_intrinsics(path: str | Path, K: NDArray, dist: NDArray, size: Tuple[int, int], rms: float) -> None:
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    fs.write("image_width", int(size[0]))
    fs.write("image_height", int(size[1]))
    fs.write("camera_matrix", K)
    fs.write("dist_coeffs", np.asarray(dist).reshape(1, -1))
    fs.write("reprojection_error", rms)
    fs.release()


def solve_pose(points: NDArray, pixels: NDArray, K: NDArray, dist: NDArray, position: NDArray, R_init: NDArray,
               click_noise_px: float = 15.0) -> Tuple[NDArray, NDArray, NDArray]:
    """Camera rotation from touched base-frame points and their clicked pixels.

    The camera position is fixed to the measured mount: the arm's reachable points lie close to
    one tilted plane, where a free 6-DOF fit trades camera distance for tilt and drifts by
    decimetres. The rotation (the eyeballed part of the mount) is well determined. A robust loss
    absorbs fingertip click noise (~10 px); clicks beyond 3 x ``click_noise_px`` are flagged.

    Returns (R camera axes in base, residuals px of every point, inlier mask).
    """
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    points = np.asarray(points, dtype=np.float64)
    pixels = np.asarray(pixels, dtype=np.float64)
    if len(points) < MIN_TOUCH_POINTS:
        raise ValueError(f"need at least {MIN_TOUCH_POINTS} touched points, have {len(points)}")
    position = np.asarray(position, dtype=np.float64)

    def rotation(v: NDArray) -> NDArray:
        return Rotation.from_rotvec(v).as_matrix() @ R_init

    def project(v: NDArray) -> NDArray:
        R = rotation(v)
        rvec, _ = cv2.Rodrigues(R.T)
        return cv2.projectPoints(points, rvec, -R.T @ position, K, dist)[0].reshape(-1, 2)

    fit = least_squares(lambda v: (project(v) - pixels).ravel(), np.zeros(3), loss="soft_l1", f_scale=click_noise_px)
    residuals = np.linalg.norm(project(fit.x) - pixels, axis=1)
    return rotation(fit.x), residuals, residuals < 3 * click_noise_px


def planarity_mm(points: NDArray) -> float:
    """Thickness of the touched point cloud along its thinnest axis (< 20 mm: badly conditioned)."""
    c = np.asarray(points) - np.mean(points, axis=0)
    return float(np.linalg.svd(c, compute_uv=False)[-1] / np.sqrt(len(c)) * 1000.0)


def implausible(pos: NDArray, R: NDArray, pos_nominal: NDArray, R_nominal: NDArray, kept: int, total: int,
                min_kept: float = 0.7, max_offset_m: float = 0.15, max_angle_deg: float = 15.0) -> str | None:
    """Why a touch-calibrated pose cannot be right (None = plausible).

    The camera is bolted to the torso mount: the solved pose must lie near the nominal one, and
    most touched points must agree with it. A fit on a small consistent subset (mis-clicks) can
    have a low residual and still be far off.
    """
    if kept < min_kept * total:
        return f"only {kept}/{total} points agree (clicks inconsistent?)"
    off = float(np.linalg.norm(np.asarray(pos) - pos_nominal))
    if off > max_offset_m:
        return f"camera position {off * 100:.0f} cm from the nominal mount"
    angle = float(np.degrees(np.arccos(np.clip((np.trace(R_nominal.T @ R) - 1) / 2, -1.0, 1.0))))
    if angle > max_angle_deg:
        return f"camera rotation {angle:.0f} deg from the nominal mount"
    return None
