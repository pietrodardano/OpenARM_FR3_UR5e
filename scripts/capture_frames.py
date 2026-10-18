#!/usr/bin/env python3
"""
Shoot calibration photos from the robot camera, with live feedback.

Live preview with, for each frame: whether the ChArUco board is detected, how
sharp the frame is, and which parts of the image you have covered so far. Press
space to keep a frame, and it refuses the blurry ones -- motion blur is the usual
reason a photo gets silently skipped later by calibrate_intrinsics.py.

Usage:
    # intrinsics photos of the board, with detection + coverage feedback
    ../.venv/bin/python scripts/capture_frames.py --out outputs/intrinsics_photos \\
        --charuco --square-size 26.968 --marker-size 19.233

    # just grab frames, no board
    ../.venv/bin/python scripts/capture_frames.py --out outputs/frames

The camera is found by the `vision.camera_device` name from the config, so you
normally do not pass --device at all; it prints which index it opened.

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

from calibrate_table_tape import load_config, open_camera  # noqa: E402

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
                         "Run through the project venv: ../.venv/bin/python scripts/capture_frames.py ...")
    attribute = f"DICT_{args.dictionary.upper().replace('DICT_', '')}"
    if not hasattr(cv2.aruco, attribute):
        raise SystemExit(f"Unknown --dictionary {args.dictionary!r}")
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, attribute))
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


def board_signature(points: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Centroid, apparent area and principal-axis angle of a detected board."""
    centroid = points.mean(axis=0)
    hull = cv2.convexHull(points.astype(np.float32))
    area = float(cv2.contourArea(hull))
    centered = points - centroid
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    angle = float(np.degrees(np.arctan2(vt[0, 1], vt[0, 0])))
    return centroid, area, angle


def is_novel(signature, kept_signatures, min_shift_px=70.0, min_area_ratio=0.25,
             min_angle_deg=12.0) -> bool:
    """
    True if this view adds information the kept set does not already have.

    Calibration needs varied views, not many copies of one: the focal length and
    the distortion coefficients are only separable when the board is seen at
    different distances, tilts and places in the frame. Near-duplicates inflate
    the photo count while leaving the fit just as under-determined, so a frame
    has to differ in position, apparent size or rotation to earn its place.
    """
    centroid, area, angle = signature
    for other_centroid, other_area, other_angle in kept_signatures:
        shifted = np.linalg.norm(centroid - other_centroid) > min_shift_px
        resized = abs(area - other_area) / max(area, other_area, 1.0) > min_area_ratio
        turned = abs((angle - other_angle + 90) % 180 - 90) > min_angle_deg
        if not (shifted or resized or turned):
            return False
    return True


def guidance(coverage: np.ndarray, areas: list[float], frame_area: float) -> str:
    """One short instruction for whatever the kept set is still missing."""
    empty = [(r, c) for r in range(GRID) for c in range(GRID) if coverage[r, c] == 0]
    if empty:
        row, col = min(empty, key=lambda rc: (coverage[rc[0], rc[1]], rc))
        vertical = ("top", "middle", "bottom")[row]
        horizontal = ("left", "centre", "right")[col]
        return f"move the board to the {vertical} {horizontal} of the view"
    if areas:
        fractions = np.array(areas) / frame_area
        if fractions.max() < 0.25:
            return "bring the board CLOSER so it fills more of the view"
        if fractions.min() > 0.12:
            return "take the board FURTHER away for a few frames"
    return "keep tilting it to new angles"


def run_auto(cap, detector, args, rotate: bool, expected: int, out_dir: str) -> list[str]:
    """Keeps diverse, sharp, detected frames on its own -- no keypresses."""
    import time  # noqa: PLC0415

    saved: list[str] = []
    signatures: list[tuple] = []
    areas: list[float] = []
    coverage = np.zeros((GRID, GRID), int)
    started = time.time()
    last_keep = 0.0
    last_message = ""
    show = not args.no_window

    print(f"AUTO mode: hold the tablet in front of the camera and move it SLOWLY -- vary the")
    print(f"angle (tilt 20-45 deg each way), the distance, and where it sits in the view.")
    print(f"Frames are kept automatically. Target {args.target} good views; Ctrl+C to stop early.\n")

    try:
        while time.time() - started < args.timeout:
            covered = int((coverage > 0).sum())
            if len(saved) >= args.target and covered >= args.min_cells:
                break
            ok, frame = cap.read()
            if not ok:
                continue
            if rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharp = sharpness(gray)
            corners, ids, _mc, _mi = detector.detectBoard(gray)
            found = 0 if corners is None else len(corners)

            keep_reason = None
            too_small = False
            saturated = False
            if found >= max(6, expected // 3) and sharp >= args.min_sharpness:
                points = corners.reshape(-1, 2)
                span = float(points[:, 0].max() - points[:, 0].min())
                too_small = span < args.min_board_width * frame.shape[1]
                signature = board_signature(points)
                row, col = coverage_cell(signature[0], (frame.shape[1], frame.shape[0]))
                # One spot must not eat the whole quota: a low reprojection error over
                # 25 near-identical views is not a good calibration, it is an overfit
                # one, and the distortion stays unconstrained everywhere else.
                saturated = coverage[row, col] >= args.per_cell_cap
                if too_small or saturated:
                    pass
                elif is_novel(signature, signatures) and time.time() - last_keep > 0.4:
                    path = os.path.join(out_dir, f"frame_{len(saved):03d}.png")
                    cv2.imwrite(path, frame)
                    saved.append(path)
                    signatures.append(signature)
                    areas.append(signature[1])
                    coverage[row, col] += 1
                    last_keep = time.time()
                    keep_reason = (f"kept {len(saved)}/{args.target}  "
                                   f"corners {found}/{expected}  sharpness {sharp:.0f}")

            message = ("bring the board CLOSER -- it is too small to decode" if too_small
                       else "this part of the view has enough -- move the board elsewhere" if saturated
                       else guidance(coverage, areas, float(frame.shape[0] * frame.shape[1])))
            if keep_reason:
                print(f"  {keep_reason}   -> next: {message}")
                last_message = message
            elif message != last_message and time.time() - last_keep > 1.5:
                last_message = message
                print(f"  {message}" if (found or too_small) else
                      "  no board detected -- bring it closer, and check for glare on the screen")
                last_keep = time.time() - 0.4

            if show:
                canvas = frame.copy()
                if found:
                    for point in corners.reshape(-1, 2):
                        cv2.circle(canvas, tuple(np.int32(point)), 3, (0, 255, 0), -1)
                draw_overlay(canvas, len(saved), sharp, args.min_sharpness, found, expected,
                             coverage, True)
                cv2.putText(canvas, message, (12, canvas.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                cv2.putText(canvas, message, (12, canvas.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow(WINDOW, canvas)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        print("\n  stopped early")

    empty = int((coverage == 0).sum())
    covered = GRID * GRID - empty
    print(f"\nKept {len(saved)} views; {covered}/{GRID * GRID} areas of the frame covered.")
    if covered < args.min_cells:
        print(f"  Only {covered} areas covered. The lens model will be poorly constrained in the")
        print(f"  rest of the frame -- a low reprojection error does not fix that. Consider")
        print(f"  re-running and walking the board into every part of the view.")
    return saved


def run_calibration(out_dir: str, args: argparse.Namespace) -> int:
    """Hands the captured frames straight to the intrinsics solver."""
    import subprocess  # noqa: PLC0415

    command = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "calibrate_intrinsics.py"),
               "--images", out_dir,
               "--square-size", str(args.square_size),
               "--marker-size", str(args.marker_size),
               "--squares-x", str(args.squares_x),
               "--squares-y", str(args.squares_y),
               "--out", args.intrinsics_out]
    print("\n" + "=" * 70)
    print("Running: " + " ".join(command[1:]))
    print("=" * 70)
    return subprocess.call(command)


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
    parser.add_argument("--auto", action="store_true",
                        help="Keep good, varied frames automatically -- no keypresses. You just "
                             "move the board around.")
    parser.add_argument("--target", type=int, default=25,
                        help="How many distinct views to keep in --auto (default 25).")
    parser.add_argument("--timeout", type=float, default=240.0,
                        help="Give up after this many seconds in --auto (default 240).")
    parser.add_argument("--min-cells", type=int, default=7,
                        help="Do not finish until the board has been seen in this many of the 9 "
                             "areas of the frame (default 7). Coverage is what constrains the "
                             "distortion; frame count alone does not.")
    parser.add_argument("--per-cell-cap", type=int, default=5,
                        help="Most frames to keep with the board in any one area of the frame.")
    parser.add_argument("--calibrate", action="store_true",
                        help="Run calibrate_intrinsics.py on the captured frames when done.")
    parser.add_argument("--intrinsics-out", default="configs/camera_intrinsics.yaml")
    parser.add_argument("--no-window", action="store_true",
                        help="Do not open a preview window (terminal feedback only).")
    parser.add_argument("--dictionary", default="6X6_250",
                        help="ArUco dictionary of the board (default 6X6_250).")
    parser.add_argument("--min-board-width", type=float, default=0.20,
                        help="Reject views where the board spans less than this fraction of the "
                             "frame width (default 0.20). Markers seen small cannot be decoded, "
                             "and a board that barely spans the frame contributes almost nothing.")
    parser.add_argument("--min-sharpness", type=float, default=60.0,
                        help="Reject frames below this Laplacian variance (default 60).")
    args = parser.parse_args()

    config = load_config(args.config)
    vision = config.get("vision", {}) or {}

    width, height = int(vision.get("width", 1280)), int(vision.get("height", 720))
    rotate = vision.get("rotate_180", False) if args.rotate_180 is None else args.rotate_180

    detector, _board = make_detector(args)
    expected = (args.squares_x - 1) * (args.squares_y - 1)

    cap, device, width, height = open_camera(args.device, width, height,
                                             vision.get("camera_device"),
                                             vision.get("device_index"))

    os.makedirs(args.out, exist_ok=True)

    if args.auto:
        if detector is None:
            raise SystemExit("--auto needs the board: add --charuco (and the square/marker sizes).")
        try:
            saved = run_auto(cap, detector, args, rotate, expected, args.out)
        finally:
            cap.release()
            cv2.destroyAllWindows()
        if len(saved) < 10:
            raise SystemExit(f"Only {len(saved)} usable views -- too few to calibrate. The board has "
                             f"to be detected (green dots) and moved around; check glare and that "
                             f"the whole board stays in view.")
        if args.calibrate:
            raise SystemExit(run_calibration(args.out, args))
        return

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
