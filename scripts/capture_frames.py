#!/usr/bin/env python3
"""
Shoot calibration photos from the robot camera, with live feedback.

Live preview with, for each frame: whether the ChArUco board is detected, how
sharp the frame is, and which parts of the image you have covered so far. Press
space to keep a frame, and it refuses the blurry ones -- motion blur is the usual
reason a photo gets silently skipped later by calibrate_intrinsics.py.

Usage:
    # intrinsics photos of the board, with detection + coverage feedback
    python scripts/capture_frames.py --out outputs/intrinsics_photos \\
        --device 0 --charuco --square-size 19.242 --marker-size 13.758

    # just grab frames, no board
    python scripts/capture_frames.py --out outputs/frames --device 0

Keys:  space = keep this frame,  b = keep even if blurry,  d = delete the last,
       q = done
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from calibrate_table_tape import load_config, probe_devices  # noqa: E402

WINDOW = "Capture calibration frames"
GRID = 3  # coverage is tracked over a 3x3 split of the frame


def sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian: low means blurred or out of focus."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def make_detector(args: argparse.Namespace):
    if not args.charuco:
        return None, None
    if not hasattr(cv2.aruco, "CharucoDetector"):
        raise SystemExit("This OpenCV has no cv2.aruco.CharucoDetector (needs >= 4.7). "
                         "Run through the project venv: python scripts/capture_frames.py ...")
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard((args.squares_x, args.squares_y),
                                   args.square_size / 1000.0, args.marker_size / 1000.0, aruco_dict)
    return cv2.aruco.CharucoDetector(board), board


def coverage_cell(center: np.ndarray, size: tuple[int, int]) -> tuple[int, int]:
    col = min(GRID - 1, int(center[0] / size[0] * GRID))
    row = min(GRID - 1, int(center[1] / size[1] * GRID))
    return row, col


def draw_overlay(canvas: np.ndarray, kept: int, sharp: float, min_sharp: float,
                 corners: int, expected: int, coverage: np.ndarray, charuco: bool) -> None:
    height, width = canvas.shape[:2]
    blurry = sharp < min_sharp
    lines = [f"kept {kept}   sharpness {sharp:5.0f} {'BLURRY -- hold still' if blurry else 'ok'}"]
    if charuco:
        lines.append(f"board corners {corners}/{expected}"
                     + ("   <-- not detected, check glare / angle" if corners == 0 else ""))
    lines.append("space = keep,  b = keep anyway,  d = delete last,  q = done")
    for i, text in enumerate(lines):
        y = 26 + i * 26
        cv2.putText(canvas, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(canvas, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (120, 120, 255) if blurry and i == 0 else (255, 255, 255), 1)

    # Coverage map: cells you have not yet photographed the board in, hollow.
    box = 26
    x0, y0 = width - GRID * box - 12, height - GRID * box - 12
    for row in range(GRID):
        for col in range(GRID):
            top_left = (x0 + col * box, y0 + row * box)
            bottom_right = (top_left[0] + box - 3, top_left[1] + box - 3)
            hits = int(coverage[row, col])
            cv2.rectangle(canvas, top_left, bottom_right, (255, 255, 255), 1)
            if hits:
                cv2.rectangle(canvas, top_left, bottom_right, (0, 200 if hits < 3 else 255, 0), -1)
                cv2.putText(canvas, str(hits), (top_left[0] + 6, top_left[1] + box - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.putText(canvas, "frame coverage", (x0 - 2, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="Directory to write the kept frames into.")
    parser.add_argument("--config", default="configs/task_config.yaml")
    parser.add_argument("--device", type=int, default=None,
                        help="Camera index (default: vision.device_index from the config).")
    parser.add_argument("--rotate-180", dest="rotate_180", action="store_true", default=None)
    parser.add_argument("--no-rotate-180", dest="rotate_180", action="store_false", default=None)
    parser.add_argument("--charuco", action="store_true",
                        help="Detect the ChArUco board live and track frame coverage.")
    parser.add_argument("--square-size", type=float, default=28.0, help="Board square, mm.")
    parser.add_argument("--marker-size", type=float, default=20.0, help="Board marker, mm.")
    parser.add_argument("--squares-x", type=int, default=5)
    parser.add_argument("--squares-y", type=int, default=7)
    parser.add_argument("--min-sharpness", type=float, default=60.0,
                        help="Reject frames below this Laplacian variance (default 60).")
    args = parser.parse_args()

    config = load_config(args.config)
    vision = config.get("vision", {}) or {}
    device = args.device if args.device is not None else int(vision.get("device_index", 0))
    width, height = int(vision.get("width", 1280)), int(vision.get("height", 720))
    rotate = vision.get("rotate_180", False) if args.rotate_180 is None else args.rotate_180

    detector, _board = make_detector(args)
    expected = (args.squares_x - 1) * (args.squares_y - 1)

    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        working = probe_devices()
        raise SystemExit(f"Could not open camera index {device}. "
                         + (f"These opened: {working}; re-run with --device {working[0]}."
                            if working else "Check `ls /dev/video*` and permissions."))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    os.makedirs(args.out, exist_ok=True)
    coverage = np.zeros((GRID, GRID), int)
    saved: list[str] = []
    print(f"Writing to {args.out}. Aim for 20-30 frames with the board at varied angles,")
    print("distances and positions -- fill every cell of the coverage map, corners included.\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("camera returned no frame")
                break
            if rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharp = sharpness(gray)

            canvas = frame.copy()
            corners_found = 0
            center = None
            if detector is not None:
                charuco_corners, charuco_ids, _mc, _mi = detector.detectBoard(gray)
                if charuco_corners is not None and len(charuco_corners) > 0:
                    corners_found = len(charuco_corners)
                    points = charuco_corners.reshape(-1, 2)
                    center = points.mean(axis=0)
                    for point in points:
                        cv2.circle(canvas, tuple(np.int32(point)), 3, (0, 255, 0), -1)
            draw_overlay(canvas, len(saved), sharp, args.min_sharpness, corners_found, expected,
                         coverage, args.charuco)
            cv2.imshow(WINDOW, canvas)

            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("d") and saved:
                os.remove(saved.pop())
                print(f"  deleted, {len(saved)} kept")
            if key in (ord(" "), ord("b")):
                if sharp < args.min_sharpness and key != ord("b"):
                    print(f"  too blurry ({sharp:.0f} < {args.min_sharpness:.0f}) -- hold still, "
                          f"or press b to keep it anyway")
                    continue
                if args.charuco and corners_found == 0 and key != ord("b"):
                    print("  board not detected -- move/tilt it, cut the glare, or press b to keep")
                    continue
                path = os.path.join(args.out, f"frame_{len(saved):03d}.png")
                cv2.imwrite(path, frame)
                saved.append(path)
                if center is not None:
                    row, col = coverage_cell(center, (frame.shape[1], frame.shape[0]))
                    coverage[row, col] += 1
                print(f"  kept {os.path.basename(path)}  (sharpness {sharp:.0f}, "
                      f"corners {corners_found})")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"\n{len(saved)} frames in {args.out}")
    if args.charuco:
        empty = int((coverage == 0).sum())
        print(f"Coverage: {GRID * GRID - empty}/{GRID * GRID} cells of the frame"
              + (f" -- {empty} never had the board in them; distortion is worst at the edges, so "
                 f"go back and cover them" if empty else " -- good"))
    if len(saved) < 15:
        print("Fewer than 15 frames: calibrate_intrinsics.py wants 20-30 for a usable fit.")


if __name__ == "__main__":
    main()
