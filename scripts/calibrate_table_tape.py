#!/usr/bin/env python3
"""
Guided camera calibration from a tape rectangle on the table.

You tape a rectangle of measured size onto the tabletop, tell the script where
that rectangle sits in the robot base frame, click its four corners in the
captured frame, and it solves for the camera pose and the tabletop mapping.

Writes configs/camera_extrinsics.yaml:
  * pixel -> tabletop (x, y) homography, so a pixel on the table turns straight
    into a base-frame grasp target -- this is the trustworthy output,
  * camera position + rotation in the robot base frame (+x forward, +y left,
    +z up, origin on the floor under the torso -- the MJCF world frame),
  * the MuJoCo camera pos/quat/fovy to paste into the scene.

Intrinsics decide how much of that you can trust:
  * configs/camera_intrinsics.yaml present (from scripts/calibrate_intrinsics.py):
    distortion is handled properly. Tabletop mapping lands within ~2-5 mm inside
    the taped area, camera position within a few cm. Use this path.
  * no intrinsics file: the focal length is estimated from the rectangle's own
    perspective, with the principal point assumed centred and distortion ignored.
    The tabletop mapping still comes out around 1-2 cm, which is enough for coarse
    picking, but the camera pose and FOV can be wildly wrong on a wide-angle lens
    like this HBV module -- and they look fine in the residuals while being wrong.
    Do not put them in the MJCF.

Accuracy degrades as you extrapolate outside the taped rectangle (roughly 1-3 cm
at 300 mm beyond an edge), so tape around the area you actually pick from rather
than a small square in the middle.

Usage:
    # live camera: 500 x 400 mm rectangle, near edge 150 mm forward of the table's rear edge
    ../.venv/bin/python scripts/calibrate_table_tape.py --side 500 --depth 400 \
        --near-edge-from-table-rear 150

    # from a photo, rectangle not centred on the midline
    ../.venv/bin/python scripts/calibrate_table_tape.py --image snapshot.jpg \
        --side 500 --depth 400 --near-edge-x 0.300 --center-y 0.050

    # click along the tape edges instead of the corners (5+ clicks per edge)
    ../.venv/bin/python scripts/calibrate_table_tape.py --mode edges --side 500

    # headless: pass the four corner pixels yourself
    ../.venv/bin/python scripts/calibrate_table_tape.py --image snapshot.jpg --side 500 \
        --points "412,585;889,591;812,402;489,399"
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

# Clicked in this order. "near" = edge closest to the robot, "left" = robot's
# +y side, i.e. the left half of the image for a forward-looking camera.
CORNER_ORDER = ("near-left", "near-right", "far-right", "far-left")
EDGE_ORDER = ("near", "left", "far", "right")
EDGE_HELP = {
    "near": "the edge closest to the robot",
    "left": "the side edge on the robot's left (+y), running away from the robot",
    "far": "the edge furthest from the robot",
    "right": "the side edge on the robot's right (-y)",
}
WINDOW = "Tape rectangle calibration"
# Set when grab_frame() actually rotates the captured frame.
ROTATION_APPLIED = [False]


# ---------------------------------------------------------------------------
# config / intrinsics
# ---------------------------------------------------------------------------
def load_config(path: str) -> dict:
    with open(path) as handle:
        return yaml.safe_load(handle)


def load_intrinsics(path: str, image_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, str]:
    """Reads the ChArUco calibration, or returns (None, None, reason) if unusable."""
    if not os.path.exists(path):
        return None, None, f"{path} not found"
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    try:
        camera_matrix = fs.getNode("camera_matrix").mat()
        dist_coeffs = fs.getNode("dist_coeffs").mat()
        width = fs.getNode("image_width").real()
        height = fs.getNode("image_height").real()
    finally:
        fs.release()
    if camera_matrix is None:
        return None, None, f"{path} has no camera_matrix"
    if int(width) != image_size[0] or int(height) != image_size[1]:
        # Intrinsics scale with resolution as long as the aspect ratio matches.
        if abs(width / height - image_size[0] / image_size[1]) > 1e-3:
            return None, None, (f"{path} was calibrated at {int(width)}x{int(height)}, which has a "
                                f"different aspect ratio than this {image_size[0]}x{image_size[1]} frame")
        scale = image_size[0] / width
        camera_matrix = camera_matrix.copy()
        camera_matrix[:2, :] *= scale
        print(f"  scaled intrinsics from {int(width)}x{int(height)} to "
              f"{image_size[0]}x{image_size[1]} (factor {scale:.4f})")
    return camera_matrix, dist_coeffs, f"measured ({os.path.basename(path)})"


def estimate_focal(obj_points: np.ndarray, img_points: np.ndarray,
                   image_size: tuple[int, int]) -> float:
    """
    Focal length in pixels from one metric planar target, assuming square pixels,
    no skew, no distortion, and the principal point at the image centre.

    With the principal point subtracted, the homography from the target plane
    (metres) to pixels is H = K [r1 | r2 | t] with K = diag(f, f, 1). Because r1
    and r2 are orthonormal, h1' w h1 = h2' w h2 and h1' w h2 = 0, where
    w = (K K')^-1 = diag(1/f^2, 1/f^2, 1). Both are linear in 1/f^2, so they are
    solved together in least squares: either one alone degenerates for some views
    (the second one, the orthogonal-vanishing-point constraint, vanishes whenever
    the camera has no roll relative to the target -- the usual case here), but not
    both at once.
    """
    center = np.array([image_size[0] / 2.0, image_size[1] / 2.0])
    plane = obj_points[:, :2].astype(np.float64)          # metres, on the table
    pixels = (img_points - center).astype(np.float64)      # principal point at the origin
    H, _ = cv2.findHomography(plane.reshape(-1, 1, 2), pixels.reshape(-1, 1, 2), 0)
    if H is None:
        raise ValueError("Could not fit a homography to the four corners -- check the corner order.")
    scale = np.sqrt(np.linalg.norm(H[:, 0]) * np.linalg.norm(H[:, 1]))
    if scale < 1e-12:
        raise ValueError("Degenerate homography -- check the corner order and the measured geometry.")
    H = H / scale
    h1, h2 = H[:, 0], H[:, 1]

    # coeff * (1/f^2) + offset = 0, one row per constraint.
    coeffs = np.array([h1[0] ** 2 + h1[1] ** 2 - h2[0] ** 2 - h2[1] ** 2,
                       h1[0] * h2[0] + h1[1] * h2[1]])
    offsets = np.array([h1[2] ** 2 - h2[2] ** 2,
                        h1[2] * h2[2]])
    denom = float(coeffs @ coeffs)
    if denom < 1e-18:
        raise ValueError(
            "This view carries no information about the focal length (the target is seen almost "
            "perfectly fronto-parallel). Tilt matters: the camera must look at the table at an angle. "
            "Otherwise run scripts/calibrate_intrinsics.py for measured intrinsics.")
    inv_f_squared = float(-(coeffs @ offsets) / denom)
    if inv_f_squared <= 1e-12:
        raise ValueError(
            f"Focal-length estimate came out non-physical (1/f^2 = {inv_f_squared:.3e}). The clicks, "
            "the measured square size, or its measured position in the base frame disagree with each "
            "other. Re-measure and re-click, or run scripts/calibrate_intrinsics.py.")
    focal = float(1.0 / np.sqrt(inv_f_squared))
    residual = float(np.max(np.abs(coeffs * inv_f_squared + offsets)))
    print(f"  focal estimated from the target: {focal:.1f} px (constraint residual {residual:.2e})")
    return focal


# ---------------------------------------------------------------------------
# capture
# ---------------------------------------------------------------------------
def camera_names() -> dict[int, str]:
    """Linux: /dev/videoN -> the driver's name for it, so cameras can be found by name."""
    names: dict[int, str] = {}
    base = "/sys/class/video4linux"
    if not os.path.isdir(base):
        return names
    for entry in sorted(os.listdir(base)):
        if not entry.startswith("video"):
            continue
        try:
            index = int(entry[len("video"):])
            with open(os.path.join(base, entry, "name")) as handle:
                names[index] = handle.read().strip()
        except (ValueError, OSError):
            continue
    return names


def video_capture(index: int) -> cv2.VideoCapture:
    """
    Opens a camera index, pinned to V4L2 on Linux.

    Without pinning, OpenCV falls back to the FFMPEG backend when a V4L2 node
    cannot stream -- and FFMPEG numbers only the capture devices, so the same
    integer means a different camera in each backend. That is how "index 1"
    silently becomes someone else's webcam. With CAP_V4L2, index N is
    /dev/videoN and nothing else.
    """
    if os.path.isdir("/sys/class/video4linux"):
        return cv2.VideoCapture(index, cv2.CAP_V4L2)
    return cv2.VideoCapture(index)


def captures(index: int) -> bool:
    """True if this index opens and hands back a real frame."""
    cap = video_capture(index)
    try:
        if not cap.isOpened():
            return False
        for _ in range(3):
            ok, frame = cap.read()
            if ok and frame is not None and frame.size:
                return True
        return False
    except cv2.error:
        return False
    finally:
        cap.release()


def probe_devices(limit: int = 8) -> list[int]:
    """
    Camera indices that actually open and deliver a frame.

    Only indices the kernel actually exposes are tried; probing past them makes
    OpenCV print a wall of backend errors for devices that were never there.
    """
    known = sorted(camera_names())
    return [index for index in (known or range(limit)) if captures(index)]


def describe_cameras() -> str:
    names = camera_names()
    if not names:
        return "no /dev/video* devices found"
    working = set(probe_devices())
    return "\n".join(
        f"    --device {index:<2} {names[index]!r}"
        + ("" if index in working else "   (does not capture -- usually a metadata node)")
        for index in sorted(names))


def find_camera_index(name_hint: str | None) -> int | None:
    """
    Lowest capturing /dev/videoN whose driver name matches the hint.

    Index is not a stable way to name a camera: a laptop's built-in webcam
    usually takes 0, UVC devices expose a second metadata-only node next to the
    real one, and the order changes when things are replugged. The name is
    stable, so vision.camera_device is what we match on.
    """
    if not name_hint:
        return None
    hint = name_hint.strip().lower()
    for index, name in sorted(camera_names().items()):
        if hint not in name.lower():
            continue
        if captures(index):
            return index
    return None


def open_camera(device: int | None, width: int, height: int, name_hint: str | None = None,
                fallback_index: int | None = None) -> tuple[cv2.VideoCapture, int, int, int]:
    """
    Opens the camera and puts it in the requested mode.

    MJPG is requested first on purpose: this HBV module (and most cheap UVC
    cameras) only offers the higher resolutions in MJPG, and silently serves
    640x480 YUYV instead if you just ask for the size. Calibrating in one mode
    and running in another invalidates the intrinsics -- the field of view is
    not even the same between its 4:3 and 16:9 modes -- so the actual mode is
    verified and reported, never assumed.
    """
    if device is None:
        # Name first, index only as a fallback: a configured index goes stale the
        # moment the machine, the OS or the USB order changes, and silently opening
        # the wrong camera is worse than failing.
        device = find_camera_index(name_hint)
        if device is not None:
            print(f"  found {name_hint!r} at index {device}")
        elif fallback_index is not None:
            device = int(fallback_index)
            print(f"  {name_hint!r} not found by name; falling back to configured index {device}")
        else:
            raise SystemExit(f"Could not find a camera matching {name_hint!r}. Available:\n"
                             f"{describe_cameras()}")

    cap = video_capture(device)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {device}. Available:\n{describe_cameras()}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    frame = None
    for _ in range(6):
        ok, grabbed = cap.read()
        if ok:
            frame = grabbed
    if frame is None:
        cap.release()
        raise SystemExit(f"Camera index {device} opened but delivered no frames. Available:\n"
                         f"{describe_cameras()}")

    actual_w, actual_h = frame.shape[1], frame.shape[0]
    if (actual_w, actual_h) != (width, height):
        print(f"  WARNING: asked for {width}x{height}, got {actual_w}x{actual_h}. The camera does "
              f"not offer that mode.\n           Calibrate in the SAME mode the pipeline will run "
              f"in, or the intrinsics will not apply.")
    else:
        print(f"  {actual_w}x{actual_h} MJPG")
    return cap, device, actual_w, actual_h


def grab_frame(args: argparse.Namespace, vision_cfg: dict) -> np.ndarray:
    """Captures one frame; --image is taken as-is, on the assumption it is already upright."""
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"Could not read --image {args.image}")
        return frame

    width = int(vision_cfg.get("width", 1280))
    height = int(vision_cfg.get("height", 720))
    cap, _index, _w, _h = open_camera(args.device, width, height,
                                      vision_cfg.get("camera_device"),
                                      vision_cfg.get("device_index"))
    frame = None
    try:
        for _ in range(10):  # let auto-exposure settle
            ok, grabbed = cap.read()
            if ok:
                frame = grabbed
    finally:
        cap.release()
    if frame is None:
        raise SystemExit("Camera stopped delivering frames.")

    rotate = vision_cfg.get("rotate_180", False) if args.rotate_180 is None else args.rotate_180
    if rotate:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        print("  applied 180 deg rotation (vision.rotate_180)")
        ROTATION_APPLIED[0] = True
    return frame


# ---------------------------------------------------------------------------
# marking the rectangle in the image
# ---------------------------------------------------------------------------
def pick_corners(frame: np.ndarray) -> np.ndarray:
    """Opens a window and collects the four taped corners, in CORNER_ORDER."""
    picked: list[tuple[float, float]] = []
    hover = [0, 0]

    def on_mouse(event, x, y, _flags, _param):
        hover[0], hover[1] = x, y
        if event == cv2.EVENT_LBUTTONDOWN and len(picked) < 4:
            picked.append((float(x), float(y)))

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_mouse)
    print("\nClick the four corners of the taped rectangle, in this order:")
    for i, name in enumerate(CORNER_ORDER, start=1):
        print(f"  {i}. {name}")
    print("  'near' = the edge closest to the robot, 'left' = the robot's left (+y).")
    print("  u = undo last click,  r = reset,  Enter = accept,  q = abort")
    print("  A zoomed inset follows the cursor -- click the outer corner of the tape, consistently.\n")

    while True:
        canvas = frame.copy()
        for i, (px, py) in enumerate(picked):
            cv2.drawMarker(canvas, (int(px), int(py)), (0, 255, 0), cv2.MARKER_CROSS, 18, 2)
            cv2.putText(canvas, str(i + 1), (int(px) + 10, int(py) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if len(picked) == 4:
            cv2.polylines(canvas, [np.int32(picked)], True, (0, 255, 255), 2)
        draw_zoom_inset(canvas, frame, hover)

        label = ("all 4 picked -- Enter to accept, u to undo"
                 if len(picked) == 4 else f"click: {CORNER_ORDER[len(picked)]}")
        cv2.putText(canvas, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(canvas, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow(WINDOW, canvas)

        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            cv2.destroyAllWindows()
            raise SystemExit("Aborted.")
        if key == ord("u") and picked:
            picked.pop()
        if key == ord("r"):
            picked.clear()
        if key in (13, 10) and len(picked) == 4:
            break

    return np.array(picked, dtype=np.float64)


def pick_edges(frame: np.ndarray) -> np.ndarray:
    """
    Collects clicks along each of the four tape edges and intersects the fitted
    lines to get the corners. More accurate than clicking corners directly: a
    line fitted through several clicks averages out click noise, and tape
    corners are rounded or overlapped where the strips meet anyway.
    """
    clicks: list[list[tuple[float, float]]] = [[], [], [], []]
    current = [0]
    hover = [0, 0]

    def on_mouse(event, x, y, _flags, _param):
        hover[0], hover[1] = x, y
        if event == cv2.EVENT_LBUTTONDOWN and current[0] < 4:
            clicks[current[0]].append((float(x), float(y)))

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_mouse)
    print("\nFor each tape edge in turn, click 5 or more points spread along it -- click the OUTER")
    print("boundary of the strip, the same boundary you measured between. Order:")
    for i, name in enumerate(EDGE_ORDER, start=1):
        print(f"  {i}. {name:>5} edge -- {EDGE_HELP[name]}")
    print("  n = next edge (needs 2+ clicks),  u = undo,  r = reset edge,  Enter = accept,  q = abort")
    print("  Click right out to both ends of each edge, not just the middle -- the corners come from")
    print("  extrapolating these lines, so clicks bunched in the middle are worse than clicking the")
    print("  corners directly. Five or more per edge, spread end to end, is the point where this wins.")
    print("  The zoom inset follows the cursor.\n")

    colors = [(0, 255, 0), (0, 200, 255), (255, 160, 0), (255, 0, 255)]
    while True:
        canvas = frame.copy()
        for i, pts in enumerate(clicks):
            for px, py in pts:
                cv2.drawMarker(canvas, (int(px), int(py)), colors[i], cv2.MARKER_CROSS, 14, 2)
            if len(pts) >= 2:
                line = fit_line(np.array(pts))
                draw_infinite_line(canvas, line, colors[i], frame.shape)
        done = current[0] >= 4
        label = ("all 4 edges fitted -- Enter to accept"
                 if done else f"edge {current[0] + 1}/4: {EDGE_ORDER[current[0]]} "
                              f"({len(clicks[current[0]])} clicks)")
        draw_zoom_inset(canvas, frame, hover)
        cv2.putText(canvas, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(canvas, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow(WINDOW, canvas)

        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            cv2.destroyAllWindows()
            raise SystemExit("Aborted.")
        if key == ord("u"):
            if not done and clicks[current[0]]:
                clicks[current[0]].pop()
            elif current[0] > 0:
                current[0] -= 1
        if key == ord("r") and not done:
            clicks[current[0]].clear()
        if key == ord("n") and not done and len(clicks[current[0]]) >= 2:
            current[0] += 1
        if key in (13, 10):
            if not done and len(clicks[current[0]]) >= 2:
                current[0] += 1
            if current[0] >= 4:
                break

    return corners_from_edges([fit_line(np.array(pts)) for pts in clicks])


def fit_line(points: np.ndarray) -> np.ndarray:
    """Total-least-squares line through 2+ points, as homogeneous coefficients."""
    if len(points) == 2:
        return np.cross(np.append(points[0], 1.0), np.append(points[1], 1.0))
    vx, vy, x0, y0 = cv2.fitLine(points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    return np.cross(np.array([x0, y0, 1.0]), np.array([x0 + vx, y0 + vy, 1.0]))


def corners_from_edges(lines: list[np.ndarray]) -> np.ndarray:
    """Intersects the near/left/far/right edge lines into corners, in CORNER_ORDER."""
    edges = dict(zip(EDGE_ORDER, lines))
    pairs = {"near-left": ("near", "left"), "near-right": ("near", "right"),
             "far-right": ("far", "right"), "far-left": ("far", "left")}
    corners = []
    for name in CORNER_ORDER:
        a, b = pairs[name]
        point = np.cross(edges[a], edges[b])
        if abs(point[2]) < 1e-9:
            raise SystemExit(f"The {a} and {b} edges came out parallel, so the {name} corner is "
                             f"undefined. Re-run and click points spread along each edge.")
        corners.append(point[:2] / point[2])
    return np.array(corners, dtype=np.float64)


def draw_infinite_line(canvas: np.ndarray, line: np.ndarray, color, shape) -> None:
    h, w = shape[:2]
    a, b, c = line
    points = []
    if abs(b) > 1e-9:
        points = [(0, -c / b), (w - 1, -(a * (w - 1) + c) / b)]
    elif abs(a) > 1e-9:
        points = [(-c / a, 0), (-c / a, h - 1)]
    if len(points) == 2 and all(abs(p[1]) < 1e6 for p in points):
        cv2.line(canvas, tuple(np.int32(points[0])), tuple(np.int32(points[1])), color, 1, cv2.LINE_AA)


def draw_zoom_inset(canvas: np.ndarray, frame: np.ndarray, hover: list[int],
                    half: int = 40, factor: int = 5) -> None:
    """Magnifies the region under the cursor into the top-right of the canvas."""
    h, w = frame.shape[:2]
    cx = int(np.clip(hover[0], half, w - half - 1))
    cy = int(np.clip(hover[1], half, h - half - 1))
    patch = frame[cy - half:cy + half, cx - half:cx + half]
    if patch.size == 0:
        return
    zoom = cv2.resize(patch, (half * 2 * factor, half * 2 * factor), interpolation=cv2.INTER_NEAREST)
    side = zoom.shape[0]
    cv2.drawMarker(zoom, (side // 2, side // 2), (0, 0, 255), cv2.MARKER_CROSS, 24, 1)
    x0 = canvas.shape[1] - side - 10
    if x0 < 0 or side + 10 > canvas.shape[0]:
        return
    canvas[10:10 + side, x0:x0 + side] = zoom
    cv2.rectangle(canvas, (x0, 10), (x0 + side, 10 + side), (255, 255, 255), 2)


# ---------------------------------------------------------------------------
# solve
# ---------------------------------------------------------------------------
def world_corners(near_x: float, center_y: float, side: float, depth: float,
                  table_z: float) -> np.ndarray:
    """Base-frame coordinates of the tape corners, in CORNER_ORDER (+y = left)."""
    half = side / 2.0
    return np.array([
        [near_x,         center_y + half, table_z],  # near-left
        [near_x,         center_y - half, table_z],  # near-right
        [near_x + depth, center_y - half, table_z],  # far-right
        [near_x + depth, center_y + half, table_z],  # far-left
    ], dtype=np.float64)


def solve_pose(obj_points: np.ndarray, img_points: np.ndarray, camera_matrix: np.ndarray,
               dist_coeffs: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Camera pose from the four corners, returned in the base frame as (rvec, tvec)
    mapping base-frame points into the camera.

    The planar solvers (IPPE) require the target to lie in the object frame's
    z = 0 plane, and our corners sit on the tabletop at z = table_z. Passed as-is,
    OpenCV 5 returns success with NaN, so the plane is shifted to z = 0 for the
    solve and the pose is shifted back afterwards.
    """
    z_offset = float(np.mean(obj_points[:, 2]))
    shifted = obj_points.copy()
    shifted[:, 2] -= z_offset

    rvec = tvec = None
    for flag in (cv2.SOLVEPNP_IPPE, cv2.SOLVEPNP_SQPNP, cv2.SOLVEPNP_ITERATIVE):
        ok, candidate_r, candidate_t = cv2.solvePnP(shifted, img_points, camera_matrix,
                                                    dist_coeffs, flags=flag)
        # "ok" is not enough: some solvers report success while returning NaN.
        if ok and np.all(np.isfinite(candidate_r)) and np.all(np.isfinite(candidate_t)):
            rvec, tvec = candidate_r, candidate_t
            break
    if rvec is None:
        raise SystemExit("solvePnP failed on all solvers -- check the corner order, the measured "
                         "geometry and the intrinsics.")

    refined_r, refined_t = cv2.solvePnPRefineLM(shifted, img_points, camera_matrix, dist_coeffs,
                                                rvec, tvec)
    if np.all(np.isfinite(refined_r)) and np.all(np.isfinite(refined_t)):
        rvec, tvec = refined_r, refined_t

    projected, _ = cv2.projectPoints(shifted, rvec, tvec, camera_matrix, dist_coeffs)
    error = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - img_points) ** 2, axis=1))))

    # Back to the base frame: X_shifted = X_base - (0, 0, z_offset), so
    # X_cam = R X_base + (tvec - R (0, 0, z_offset)).
    R, _ = cv2.Rodrigues(rvec)
    tvec = tvec.reshape(3, 1) - R @ np.array([[0.0], [0.0], [z_offset]])
    return rvec, tvec, error


def rotation_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> (w, x, y, z), the MuJoCo quaternion order."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    return q / np.linalg.norm(q)


def plane_homography(rvec: np.ndarray, tvec: np.ndarray, camera_matrix: np.ndarray,
                     table_z: float) -> np.ndarray:
    """
    Homography mapping an undistorted pixel to tabletop (x, y) in the base frame.

    For a world point (x, y, table_z): X_cam = [r1 r2] (x, y)^T + (r3 * table_z + t),
    so the pixel is K [r1 | r2 | r3 * table_z + t] (x, y, 1)^T.
    """
    R, _ = cv2.Rodrigues(rvec)
    third = (R[:, 2:3] * table_z) + tvec.reshape(3, 1)
    h_pixel_from_plane = camera_matrix @ np.hstack([R[:, 0:1], R[:, 1:2], third])
    return np.linalg.inv(h_pixel_from_plane)


def pixel_to_table(pixels: np.ndarray, h_plane_from_pixel: np.ndarray, camera_matrix: np.ndarray,
                   dist_coeffs: np.ndarray) -> np.ndarray:
    """Undistorts pixels, then maps them onto the tabletop plane (base-frame x, y)."""
    pts = np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
    undistorted = cv2.undistortPoints(pts, camera_matrix, dist_coeffs, P=camera_matrix)
    homogeneous = np.hstack([undistorted.reshape(-1, 2), np.ones((len(pts), 1))])
    mapped = (h_plane_from_pixel @ homogeneous.T).T
    return mapped[:, :2] / mapped[:, 2:3]


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------
def draw_table_grid(frame: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, camera_matrix: np.ndarray,
                    dist_coeffs: np.ndarray, table_cfg: dict, table_z: float, step: float = 0.1) -> np.ndarray:
    """Overlays a base-frame grid on the tabletop plane, so you can eyeball the fit."""
    canvas = frame.copy()
    length = float(table_cfg.get("length_m", 1.2))
    width = float(table_cfg.get("width_m", 0.8))
    center_x = float(table_cfg.get("center_x_m", 0.65))
    x_min, x_max = center_x - length / 2.0, center_x + length / 2.0
    y_min, y_max = -width / 2.0, width / 2.0

    def project(points: np.ndarray) -> np.ndarray:
        projected, _ = cv2.projectPoints(points, rvec, tvec, camera_matrix, dist_coeffs)
        return projected.reshape(-1, 2)

    segments: list[tuple[np.ndarray, tuple[int, int, int]]] = []
    for x in np.arange(x_min, x_max + 1e-9, step):
        segments.append((np.array([[x, y_min, table_z], [x, y_max, table_z]]), (90, 160, 90)))
    for y in np.arange(y_min, y_max + 1e-9, step):
        segments.append((np.array([[x_min, y, table_z], [x_max, y, table_z]]), (90, 160, 90)))
    # Base-frame axes at the table's rear edge on the midline.
    segments.append((np.array([[x_min, 0.0, table_z], [x_min + 0.2, 0.0, table_z]]), (0, 0, 255)))   # +x
    segments.append((np.array([[x_min, 0.0, table_z], [x_min, 0.2, table_z]]), (0, 255, 0)))         # +y

    for points, color in segments:
        # Subdivide so lens distortion shows as curvature instead of being straightened.
        samples = np.linspace(points[0], points[1], 40)
        pts = project(samples.astype(np.float64))
        for a, b in zip(pts[:-1], pts[1:]):
            if np.all(np.isfinite([a, b])) and np.max(np.abs([a, b])) < 1e5:
                cv2.line(canvas, tuple(np.int32(a)), tuple(np.int32(b)), color, 1, cv2.LINE_AA)
    return canvas


def verify_interactive(frame: np.ndarray, h_plane_from_pixel: np.ndarray, camera_matrix: np.ndarray,
                       dist_coeffs: np.ndarray) -> None:
    """Click anywhere on the table; prints the base-frame (x, y) that pixel maps to."""
    marks: list[tuple[tuple[int, int], np.ndarray]] = []

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            xy = pixel_to_table(np.array([[x, y]], dtype=np.float64),
                                h_plane_from_pixel, camera_matrix, dist_coeffs)[0]
            marks.append(((x, y), xy))
            print(f"  pixel ({x:4d}, {y:4d})  ->  base frame x = {xy[0]:+.3f} m, y = {xy[1]:+.3f} m")

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_mouse)
    print("\nVerification: the green grid is a 100 mm base-frame grid on the tabletop, red = +x, "
          "green = +y.\n  Click any point on the table to read its base-frame coordinates -- put a "
          "ruler on the table\n  and check a couple against reality. q or Enter to finish.\n")

    while True:
        canvas = frame.copy()
        for (px, py), xy in marks:
            cv2.drawMarker(canvas, (px, py), (0, 200, 255), cv2.MARKER_TILTED_CROSS, 14, 2)
            text = f"{xy[0]:+.3f}, {xy[1]:+.3f}"
            cv2.putText(canvas, text, (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(canvas, text, (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        cv2.imshow(WINDOW, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27, 13, 10):
            break
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------
def write_extrinsics(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    header = (
        "# Camera extrinsics in the robot base frame (+x forward, +y left, +z up,\n"
        "# origin on the floor under the torso -- the MJCF world frame).\n"
        "# Generated by scripts/calibrate_table_tape.py. Do not hand-edit; re-run the script.\n"
    )
    with open(path, "w") as handle:
        handle.write(header)
        yaml.safe_dump(payload, handle, sort_keys=False, default_flow_style=None)


def print_report(payload: dict) -> None:
    cam = payload["camera_in_base_frame"]
    suffix = "   <-- UNRELIABLE, see the note below" if payload["intrinsics"]["estimated"] else ""
    pos = cam["position_m"]
    print("\n" + "=" * 68)
    print("  Calibration result")
    print("=" * 68)
    print(f"Intrinsics source   : {payload['intrinsics']['source']}")
    print(f"  fx, fy            : {payload['intrinsics']['fx']:.1f}, {payload['intrinsics']['fy']:.1f} px")
    print(f"  vertical FOV      : {payload['intrinsics']['fovy_deg']:.2f} deg "
          f"(configs/task_config.yaml has workcell.rgb_camera.fovy_deg){suffix}")
    print(f"Camera position     : x {pos[0]:+.3f}  y {pos[1]:+.3f}  z {pos[2]:+.3f}  m{suffix}")
    print(f"Optical axis        : forward {cam['optical_axis_in_base'][0]:+.3f}, "
          f"left {cam['optical_axis_in_base'][1]:+.3f}, up {cam['optical_axis_in_base'][2]:+.3f}")
    print(f"  downward pitch    : {cam['downward_pitch_deg']:.2f} deg below horizontal")
    print(f"  yaw off midline   : {cam['yaw_deg']:+.2f} deg (+ = toward the robot's left)")
    print(f"Reprojection error  : {payload['residuals']['reprojection_error_px']:.2f} px over the "
          f"4 tape corners")
    print(f"Corner position err : {payload['residuals']['max_corner_error_mm']:.1f} mm max, "
          f"{payload['residuals']['mean_corner_error_mm']:.1f} mm mean (tape corner vs. its "
          f"back-projection)")
    print("\nMuJoCo camera (paste into the scene's <camera> element):"
          + ("  [UNRELIABLE without measured intrinsics]" if payload["intrinsics"]["estimated"] else ""))
    mj = payload["mujoco_camera"]
    print(f'  pos="{mj["pos"][0]:.4f} {mj["pos"][1]:.4f} {mj["pos"][2]:.4f}" '
          f'quat="{mj["quat_wxyz"][0]:.5f} {mj["quat_wxyz"][1]:.5f} '
          f'{mj["quat_wxyz"][2]:.5f} {mj["quat_wxyz"][3]:.5f}" fovy="{mj["fovy_deg"]:.2f}"')


def warn_on_quality(payload: dict, estimated_focal: bool) -> None:
    error_px = payload["residuals"]["reprojection_error_px"]
    max_mm = payload["residuals"]["max_corner_error_mm"]
    print()
    if error_px > 4.0 or max_mm > 15.0:
        print("WARNING: the fit is loose. Usual causes, in order of likelihood:")
        print("  * the clicked corners are not the corners you measured to (click the same boundary")
        print("    of the tape -- the outer edge -- every time, zoomed in with the inset),")
        print("  * the measured side lengths or the measured position in the base frame are off,")
        print("  * the taped shape is not actually square (its diagonals must match),")
        print("  * corners clicked in the wrong order (near-left, near-right, far-right, far-left),")
        print("  * the tabletop height in the config does not match the real table.")
        print("Re-run before trusting these numbers. Two things help most: a larger taped rectangle,")
        print("and --mode edges with 5+ clicks spread end to end along each edge.")
    else:
        print("Fit looks healthy. Note that a 4-point fit always reprojects its own corners well, so a")
        print("small error here is necessary but not sufficient -- the verification step below, with a")
        print("ruler on the table, is what actually confirms the calibration.")
    print("\nExpected accuracy, from a noise study of this estimator (1-2 px of click error):")
    if estimated_focal:
        print("  Tabletop mapping (pixel -> base-frame x, y): ~1-2 cm, dominated by the uncorrected")
        print("    lens distortion rather than by your clicking. Usable for coarse grasping.")
        print("  Camera pose and FOV: NOT usable. On a wide-angle lens, ignoring distortion drags the")
        print("    focal estimate far off (a simulated k1 = -0.32 gave 62 deg instead of 85 deg, and")
        print("    the camera position off by 0.6 m) while still reprojecting the four corners almost")
        print("    perfectly -- so a clean residual above does NOT vouch for the pose. Do not paste")
        print("    the MJCF pose/fovy anywhere until you have measured intrinsics.")
    else:
        print("  Tabletop mapping: ~2-5 mm inside the taped rectangle; ~1-3 cm about 300 mm outside")
        print("    it, growing with distance -- so enclose the workspace with the tape.")
        print("  Camera position: ~2-6 cm, improving with a larger rectangle. The tabletop homography")
        print("    is much better conditioned than the pose, so prefer it for grasp targets and treat")
        print("    the MJCF pose as a good visual approximation.")
    if estimated_focal:
        print("\nTo fix that, measure the intrinsics once -- it takes about ten minutes. Print")
        print("configs/charuco_5x7_square28mm_marker20mm.pdf, take 20-30 photos of it at varied")
        print("angles, distances and frame positions, then:")
        print("  ../.venv/bin/python scripts/calibrate_intrinsics.py --images <dir> --square-size 28 "
              "--marker-size 20")
        print("and re-run this script -- it picks the measured intrinsics up automatically, and the")
        print("camera pose, the FOV and the tabletop mapping all become trustworthy.")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def print_setup_guide(side_mm: float, depth_mm: float, table_rear_x: float, table_z: float,
                      mode: str) -> None:
    print("=" * 68)
    print("  Tape-rectangle camera calibration")
    print("=" * 68)
    print("Setup, before anything else:")
    print(f"  1. Tape a rectangle onto the tabletop -- currently told {side_mm:.0f} mm left-right by")
    print(f"     {depth_mm:.0f} mm forward. Make it as LARGE as the camera's view of the table allows,")
    print("     spanning the area you will actually pick from. Accuracy inside the taped area is a")
    print("     few mm; it degrades roughly linearly as you extrapolate outside it (~1-3 cm at")
    print("     300 mm beyond the edge), so enclose the workspace instead of taping a small square")
    print("     in the middle. Thin masking or electrical tape; crisp corners.")
    print("  2. Square it up: measure both diagonals and make them equal. Then measure the two side")
    print("     lengths you actually taped, outer edge to outer edge, and pass them as --side /")
    print("     --depth (mm). The numbers you measure beat the numbers you intended.")
    print("  3. Keep the near edge parallel to the table's rear edge (the edge nearest the robot).")
    print("  4. Measure how far the rectangle's near edge sits forward of the table's rear edge and")
    print("     pass it as --near-edge-from-table-rear (mm). The configured rear edge is at")
    print(f"     base-frame x = {table_rear_x:.3f} m.")
    print("  5. If the rectangle is not centred on the robot's midline, measure the offset of its")
    print("     centre and pass --center-y (m, + = toward the robot's left).")
    print(f"  6. The tabletop height from configs/task_config.yaml is z = {table_z:.3f} m. Every")
    print("     result below is referred to that plane, so check it with a tape measure first.")
    print("  7. The camera must see the table at an angle. A dead-on fronto-parallel view carries no")
    print("     focal-length information and makes the pose ill-conditioned.")
    print(f"Marking mode: {mode}. Do not move the camera or the tape between capture and use.\n")


def parse_points(text: str) -> np.ndarray:
    pairs = [chunk for chunk in text.replace(" ", "").split(";") if chunk]
    if len(pairs) != 4:
        raise SystemExit(f'--points needs 4 "x,y" pairs separated by ";", got {len(pairs)}')
    return np.array([[float(v) for v in pair.split(",")] for pair in pairs], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/task_config.yaml")
    parser.add_argument("--intrinsics", default="configs/camera_intrinsics.yaml",
                        help="ChArUco intrinsics; if missing, the focal length is estimated from the square.")
    parser.add_argument("--out", default="configs/camera_extrinsics.yaml")
    parser.add_argument("--side", type=float, default=500.0,
                        help="Measured left-right side of the taped rectangle, mm (default 500).")
    parser.add_argument("--depth", type=float, default=None,
                        help="Measured forward side, mm (default: same as --side).")
    parser.add_argument("--near-edge-from-table-rear", type=float, default=150.0,
                        help="Distance from the table's rear edge to the square's near edge, mm.")
    parser.add_argument("--near-edge-x", type=float, default=None,
                        help="Base-frame x of the square's near edge, m. Overrides the measurement above.")
    parser.add_argument("--center-y", type=float, default=0.0,
                        help="Base-frame y of the square's centre, m (+ = robot's left). Default 0 (midline).")
    parser.add_argument("--table-z", type=float, default=None,
                        help="Tabletop height, m. Default: workcell.table.top_height_m.")
    parser.add_argument("--image", default=None, help="Calibrate from this photo instead of the live camera.")
    parser.add_argument("--device", type=int, default=None, help="Camera index (default: vision.device_index).")
    parser.add_argument("--rotate-180", dest="rotate_180", action="store_true", default=None)
    parser.add_argument("--no-rotate-180", dest="rotate_180", action="store_false", default=None)
    parser.add_argument("--mode", choices=("corners", "edges"), default="corners",
                        help="How to mark the rectangle. 'corners' (default): click the four corners. "
                             "'edges': click points along each of the four tape edges and intersect the "
                             "fitted lines -- worth it at 5+ clicks per edge (below that it is slightly "
                             "worse than clicking corners), and the way out when the tape corners are "
                             "rounded, overlapped or partly occluded.")
    parser.add_argument("--points", default=None,
                        help='Headless: the four corner pixels as "x,y;x,y;x,y;x,y" in '
                             "near-left, near-right, far-right, far-left order.")
    parser.add_argument("--save-frame", default=None, help="Also write the captured frame here.")
    parser.add_argument("--no-verify", action="store_true", help="Skip the interactive verification window.")
    args = parser.parse_args()

    config = load_config(args.config)
    vision_cfg = config.get("vision", {}) or {}
    table_cfg = (config.get("workcell", {}) or {}).get("table", {}) or {}
    table_z = args.table_z if args.table_z is not None else float(table_cfg.get("top_height_m", 0.75))
    table_rear_x = float(table_cfg.get("center_x_m", 0.65)) - float(table_cfg.get("length_m", 1.2)) / 2.0
    side = args.side / 1000.0
    depth = (args.depth if args.depth is not None else args.side) / 1000.0
    near_x = (args.near_edge_x if args.near_edge_x is not None
              else table_rear_x + args.near_edge_from_table_rear / 1000.0)

    print_setup_guide(args.side, args.depth if args.depth is not None else args.side,
                      table_rear_x, table_z, "given pixels" if args.points else args.mode)
    print(f"Square in the base frame: near edge x = {near_x:.3f} m, centre y = {args.center_y:+.3f} m, "
          f"z = {table_z:.3f} m")

    if args.points is None and not args.image:
        input("\nPress Enter to capture a frame from the camera...")
    print("\nCapturing...")
    frame = grab_frame(args, vision_cfg)
    image_size = (frame.shape[1], frame.shape[0])
    print(f"  frame: {image_size[0]}x{image_size[1]}")
    if args.save_frame:
        cv2.imwrite(args.save_frame, frame)
        print(f"  saved {args.save_frame}")

    if args.points:
        img_points = parse_points(args.points)
    elif args.mode == "edges":
        img_points = pick_edges(frame)
    else:
        img_points = pick_corners(frame)
    print("Corners:")
    for name, (px, py) in zip(CORNER_ORDER, img_points):
        print(f"  {name:>10}: ({px:.1f}, {py:.1f})")

    obj_points = world_corners(near_x, args.center_y, side, depth, table_z)
    camera_matrix, dist_coeffs, source = load_intrinsics(args.intrinsics, image_size)
    estimated_focal = camera_matrix is None
    if estimated_focal:
        print(f"\nNo measured intrinsics ({source}); estimating the focal length from the rectangle.")
        focal = estimate_focal(obj_points, img_points, image_size)
        camera_matrix = np.array([[focal, 0.0, image_size[0] / 2.0],
                                  [0.0, focal, image_size[1] / 2.0],
                                  [0.0, 0.0, 1.0]])
        dist_coeffs = np.zeros((5, 1))
        source = "estimated from the tape rectangle (principal point assumed centred, no distortion)"
    else:
        print(f"\nUsing intrinsics: {source}")

    rvec, tvec, error_px = solve_pose(obj_points, img_points, camera_matrix, dist_coeffs)

    R_world_to_cam, _ = cv2.Rodrigues(rvec)
    R_cam_in_base = R_world_to_cam.T
    cam_pos = (-R_cam_in_base @ tvec).ravel()
    optical_axis = R_cam_in_base @ np.array([0.0, 0.0, 1.0])  # OpenCV camera looks along +z
    pitch_down = float(np.degrees(np.arcsin(np.clip(-optical_axis[2], -1.0, 1.0))))
    yaw = float(np.degrees(np.arctan2(optical_axis[1], optical_axis[0])))
    # MuJoCo cameras look along -z with +y up; OpenCV is +z forward, +y down.
    R_mujoco = R_cam_in_base @ np.diag([1.0, -1.0, -1.0])
    quat = rotation_to_quat(R_mujoco)
    fovy = float(np.degrees(2.0 * np.arctan(image_size[1] / (2.0 * camera_matrix[1, 1]))))

    h_plane_from_pixel = plane_homography(rvec, tvec, camera_matrix, table_z)
    recovered = pixel_to_table(img_points, h_plane_from_pixel, camera_matrix, dist_coeffs)
    corner_errors_mm = np.linalg.norm(recovered - obj_points[:, :2], axis=1) * 1000.0

    warnings: list[str] = []
    if estimated_focal:
        warnings.append(
            "No measured intrinsics: fx/fy/fovy and camera_in_base_frame/mujoco_camera are rough and "
            "can be badly wrong on a wide-angle lens (distortion biases the estimate). Only "
            "table_plane.homography_plane_from_pixel is usable (~1-2 cm). Run "
            "scripts/calibrate_intrinsics.py, then re-run this script.")
    if error_px > 4.0 or float(np.max(corner_errors_mm)) > 15.0:
        warnings.append("Loose fit -- re-measure the target and re-click the corners.")
    warnings.append(
        "Valid only for the tabletop plane at table_plane.z_m, and only while neither the camera nor "
        "the table moves.")

    payload = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "frame": "robot_base (+x forward, +y left, +z up)",
        "image_width": image_size[0],
        "image_height": image_size[1],
        # Whatever consumes this must feed the camera through the same rotation.
        "rotate_180_applied": bool(ROTATION_APPLIED[0]),
        "intrinsics": {
            "source": source,
            "estimated": bool(estimated_focal),
            "fx": float(camera_matrix[0, 0]),
            "fy": float(camera_matrix[1, 1]),
            "cx": float(camera_matrix[0, 2]),
            "cy": float(camera_matrix[1, 2]),
            "fovy_deg": fovy,
            "dist_coeffs": [float(v) for v in np.asarray(dist_coeffs).ravel()],
        },
        "camera_in_base_frame": {
            "position_m": [float(v) for v in cam_pos],
            # Columns are the camera's OpenCV axes (+x right, +y down, +z forward) in the base frame.
            "rotation_opencv": [[float(v) for v in row] for row in R_cam_in_base],
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
            # (x_base, y_base, 1) ~ H @ (u_undistorted, v_undistorted, 1)
            "homography_plane_from_pixel": [[float(v) for v in row] for row in h_plane_from_pixel],
            "note": "Undistort the pixel with the intrinsics above before applying the homography.",
        },
        "calibration_target": {
            "marking_mode": "given pixels" if args.points else args.mode,
            "side_mm": float(args.side),
            "depth_mm": float(args.depth if args.depth is not None else args.side),
            "near_edge_x_m": float(near_x),
            "center_y_m": float(args.center_y),
            "corner_order": list(CORNER_ORDER),
            "corners_base_frame_m": [[float(v) for v in row] for row in obj_points],
            "corners_pixels": [[float(v) for v in row] for row in img_points],
        },
        "warnings": warnings,
        "residuals": {
            "reprojection_error_px": error_px,
            "max_corner_error_mm": float(np.max(corner_errors_mm)),
            "mean_corner_error_mm": float(np.mean(corner_errors_mm)),
        },
    }

    write_extrinsics(args.out, payload)
    print_report(payload)
    print(f"\nWrote {args.out}")
    warn_on_quality(payload, estimated_focal)

    if not args.no_verify and args.points is None:
        overlay = draw_table_grid(frame, rvec, tvec, camera_matrix, dist_coeffs, table_cfg, table_z)
        verify_interactive(overlay, h_plane_from_pixel, camera_matrix, dist_coeffs)


if __name__ == "__main__":
    main()
