#!/usr/bin/env python3
"""
Render a ChArUco board sized for a tablet or phone screen, at the screen's exact
native resolution.

Why a screen beats paper: it is dead flat (no curl, no tape, no printer scaling),
self-lit, and you can get the square size right to a fraction of a percent from
the display's pixel pitch instead of squinting at a ruler.

The trick is to generate the image at exactly the screen's pixel resolution. Any
viewer that shows it fullscreen without cropping then displays it 1:1, so one
image pixel is one screen pixel and the square size follows from the PPI:

    square_mm = square_px * 25.4 / ppi

For the INTRINSICS step the physical size does not matter at all -- fx, fy, cx,
cy and the distortion coefficients come out identical whether the squares are
14 mm or 60 mm (the size only scales the per-photo distance to the board, which
scripts/calibrate_intrinsics.py discards). A board smaller than A4 is therefore
perfectly fine, and so is a few percent of error in the size you pass it.

The size is only there to be consistent: every photo in one calibration set must
share one scale, so do not change the zoom, rotation or viewer part-way through.
The ruler bar drawn into the image tells you the true square size whatever the
viewer does, which is why showing it 1:1 is convenient but not required.

Usage:
    # iPad 10.9": 2360x1640 at 264 ppi
    ../.venv/bin/python scripts/charuco_for_screen.py --screen 2360x1640 --ppi 264

    # PPI unknown: give the diagonal instead and it works it out
    ../.venv/bin/python scripts/charuco_for_screen.py --screen 2360x1640 --diagonal-in 10.86

Then, on the tablet:
  * copy the PNG over and open it in the photo/gallery viewer, fullscreen,
  * do NOT pinch-zoom, crop or rotate it, and lock screen rotation,
  * turn OFF auto-brightness, night mode / warm tint, and any ambient dimming,
    and set brightness to roughly 70%: a blown-out screen loses marker contrast,
  * turn off auto-lock so the screen never dims mid-session,
  * verify the scale with a ruler against the number this script prints.
"""

from __future__ import annotations

import argparse
import math
import os

import cv2
import numpy as np


def resolve_dictionary(name: str) -> int:
    """'4X4_50' -> cv2.aruco.DICT_4X4_50, with a readable error for a typo."""
    attribute = f"DICT_{name.upper().replace('DICT_', '')}"
    if not hasattr(cv2.aruco, attribute):
        available = sorted(a[5:] for a in dir(cv2.aruco)
                           if a.startswith("DICT_") and "APRIL" not in a)
        raise SystemExit(f"Unknown dictionary {name!r}. Available: {', '.join(available)}")
    return getattr(cv2.aruco, attribute)


def parse_screen(text: str) -> tuple[int, int]:
    try:
        width, height = (int(v) for v in text.lower().replace(" ", "").split("x"))
    except ValueError:
        raise SystemExit(f'--screen must look like 2360x1640, got "{text}"')
    if width < 200 or height < 200:
        raise SystemExit("--screen looks too small to be a screen resolution")
    return width, height


def draw_ruler_and_labels(canvas: np.ndarray, board_rect: tuple[int, int, int, int],
                          mm_per_px: float, squares_x: int, squares_y: int,
                          square_mm: float, dictionary_label: str = "DICT_6X6_250") -> None:
    """
    Draws a ruler bar and the expected grid dimensions into the quiet zone.

    Without this you have to remember which side is which and what it should
    measure; with it, the image states its own ground truth, and any scaling by
    the viewer shows up directly as a ruler bar that does not read 100 mm.
    Everything is kept clear of the board itself so marker detection is untouched.
    """
    x0, y0, board_w, board_h = board_rect
    height, width = canvas.shape[:2]
    margin_x = (width - board_w) // 2
    margin_y = (height - board_h) // 2
    px_per_mm = 1.0 / mm_per_px
    font = cv2.FONT_HERSHEY_SIMPLEX

    def text(where, label, scale=0.9, thickness=2):
        cv2.putText(canvas, label, where, font, scale, 0, thickness, cv2.LINE_AA)

    # Side labels, just outside the board, saying what each side should measure.
    across = f"{squares_x} squares across = {squares_x * square_mm:.1f} mm"
    down = f"{squares_y} squares down = {squares_y * square_mm:.1f} mm"
    if margin_y >= 60:
        text((x0, max(30, y0 - 20)), across)
        text((x0, min(height - 12, y0 + board_h + 44)), down)
    else:
        text((max(8, x0 - margin_x + 8), max(30, y0 - 12)), across, 0.7, 2)
        text((max(8, x0 - margin_x + 8), min(height - 10, y0 + board_h + 30)), down, 0.7, 2)

    # Ruler bar, in whichever margin has room, with 10 mm ticks.
    horizontal = margin_y >= margin_x
    bar_mm = 100.0
    span_px = int(round(bar_mm * px_per_mm))
    if horizontal:
        if span_px > width - 40:
            bar_mm, span_px = 50.0, int(round(50.0 * px_per_mm))
        bar_y = min(height - 60, y0 + board_h + margin_y // 2 + 40)
        bar_x = (width - span_px) // 2
        cv2.line(canvas, (bar_x, bar_y), (bar_x + span_px, bar_y), 0, 4)
        for mm in range(0, int(bar_mm) + 1, 10):
            x = bar_x + int(round(mm * px_per_mm))
            long_tick = mm % 50 == 0
            cv2.line(canvas, (x, bar_y - (22 if long_tick else 12)),
                     (x, bar_y + (22 if long_tick else 12)), 0, 4 if long_tick else 2)
        text((bar_x, bar_y + 60), f"this bar is {bar_mm:.0f} mm -- if your ruler disagrees, the "
                                  f"viewer is scaling the image", 0.7, 2)
    else:
        if span_px > height - 40:
            bar_mm, span_px = 50.0, int(round(50.0 * px_per_mm))
        bar_x = min(width - 60, x0 + board_w + margin_x // 2 + 20)
        bar_y = (height - span_px) // 2
        cv2.line(canvas, (bar_x, bar_y), (bar_x, bar_y + span_px), 0, 4)
        for mm in range(0, int(bar_mm) + 1, 10):
            y = bar_y + int(round(mm * px_per_mm))
            long_tick = mm % 50 == 0
            cv2.line(canvas, (bar_x - (22 if long_tick else 12), y),
                     (bar_x + (22 if long_tick else 12), y), 0, 4 if long_tick else 2)
        text((max(8, bar_x - 300), bar_y - 24), f"bar = {bar_mm:.0f} mm", 0.7, 2)

    text((12, height - 16), f"square {square_mm:.3f} mm  |  {squares_x}x{squares_y}  |  "
                            f"{dictionary_label}  |  show fullscreen, do not zoom", 0.6, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--screen", required=True,
                        help="Screen resolution in PHYSICAL pixels, e.g. 2360x1640. Use the panel's "
                             "native resolution, not a browser's CSS pixel size.")
    parser.add_argument("--ppi", type=float, default=None, help="Display pixel density, px per inch.")
    parser.add_argument("--diagonal-in", type=float, default=None,
                        help="Screen diagonal in inches; used to work out the PPI if --ppi is absent.")
    parser.add_argument("--squares-x", type=int, default=5)
    parser.add_argument("--squares-y", type=int, default=7)
    parser.add_argument("--marker-ratio", type=float, default=20.0 / 28.0,
                        help="Marker size as a fraction of the square (default matches the repo's "
                             "printable boards: 20/28).")
    parser.add_argument("--dictionary", default="6X6_250",
                        help="ArUco dictionary, e.g. 6X6_250 (default) or 4X4_50. Fewer bits per "
                             "marker means each bit gets more pixels, so a low-bit dictionary "
                             "survives being seen small or soft. Must match what the calibration "
                             "scripts are told.")
    parser.add_argument("--margin-squares", type=float, default=0.6,
                        help="White quiet zone around the board, in squares (default 0.6). ArUco "
                             "needs white around the outer markers to detect them.")
    parser.add_argument("--landscape", action="store_true",
                        help="Lay the board out for a landscape screen (swaps which screen axis the "
                             "board's long side follows).")
    parser.add_argument("--measured-across", type=float, default=None,
                        help="Measured width of the printed grid, in mm, across the short side "
                             "(--squares-x squares). Pass what your ruler actually says and the "
                             "corrected square/marker sizes are printed. Use this whenever the "
                             "viewer does not show the image 1:1 -- it does not need to.")
    parser.add_argument("--no-labels", action="store_true",
                        help="Omit the ruler bar and side labels drawn in the white margin.")
    parser.add_argument("--out", default="configs/charuco_screen.png")
    args = parser.parse_args()

    screen_w, screen_h = parse_screen(args.screen)
    if args.ppi:
        ppi = args.ppi
    elif args.diagonal_in:
        ppi = math.hypot(screen_w, screen_h) / args.diagonal_in
        print(f"PPI from the diagonal: {ppi:.2f} px/in")
    else:
        raise SystemExit("Pass --ppi, or --diagonal-in so the PPI can be worked out.")

    squares_x, squares_y = args.squares_x, args.squares_y
    if args.landscape:
        squares_x, squares_y = squares_y, squares_x

    # Largest whole-pixel square that fits with the quiet zone. Whole pixels matter:
    # a fractional square size would be rounded by the renderer, silently changing
    # the geometry away from what we report.
    usable_w = screen_w / (squares_x + 2 * args.margin_squares)
    usable_h = screen_h / (squares_y + 2 * args.margin_squares)
    square_px = int(math.floor(min(usable_w, usable_h)))
    if square_px < 20:
        raise SystemExit(f"Only {square_px} px per square fits -- too small to detect reliably. "
                         f"Use fewer squares (--squares-x/--squares-y) or a bigger screen.")
    marker_px = int(round(square_px * args.marker_ratio))

    mm_per_px = 25.4 / ppi
    square_mm = square_px * mm_per_px
    marker_mm = marker_px * mm_per_px
    board_w_px, board_h_px = squares_x * square_px, squares_y * square_px

    aruco_dict = cv2.aruco.getPredefinedDictionary(resolve_dictionary(args.dictionary))
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_mm / 1000.0,
                                   marker_mm / 1000.0, aruco_dict)
    board_img = board.generateImage((board_w_px, board_h_px), marginSize=0, borderBits=1)

    canvas = np.full((screen_h, screen_w), 255, np.uint8)
    x0 = (screen_w - board_w_px) // 2
    y0 = (screen_h - board_h_px) // 2
    canvas[y0:y0 + board_h_px, x0:x0 + board_w_px] = board_img
    if not args.no_labels:
        draw_ruler_and_labels(canvas, (x0, y0, board_w_px, board_h_px), mm_per_px,
                              squares_x, squares_y, square_mm,
                              f"DICT_{args.dictionary.upper()}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cv2.imwrite(args.out, canvas)

    print(f"\nWrote {args.out}  ({screen_w}x{screen_h} px, exactly your screen)")
    print("=" * 70)
    print(f"Board          : {squares_x}x{squares_y} squares, DICT_{args.dictionary.upper()}")
    print(f"Square         : {square_px} px  ->  {square_mm:.4f} mm   at {ppi:.2f} ppi")
    print(f"Marker         : {marker_px} px  ->  {marker_mm:.4f} mm")
    print(f"Printed grid   : {board_w_px} x {board_h_px} px  ->  "
          f"{board_w_px * mm_per_px:.2f} x {board_h_px * mm_per_px:.2f} mm")
    print("=" * 70)
    print("\nRULER CHECK (do this once, it is the only thing that can catch a scaling viewer):")
    print(f"  Measure the whole black-and-white grid across its {squares_x} squares. It should be")
    print(f"  {board_w_px * mm_per_px:.1f} mm wide and {board_h_px * mm_per_px:.1f} mm tall on screen.")
    print("  If it differs, the viewer is scaling the image, or the PPI is wrong. Either fix the")
    print("  viewer, or just use the measurement: square_size = measured_grid_width / "
          f"{squares_x}.")
    print("  Measuring across all the squares and dividing is the accurate way to do it -- it")
    print(f"  divides your ruler error by {squares_x}.")
    if args.measured_across:
        actual_square = args.measured_across / squares_x
        actual_marker = actual_square * marker_px / square_px
        scale = actual_square / square_mm
        print("\n" + "=" * 70)
        print(f"MEASURED: grid {args.measured_across:.1f} mm across {squares_x} squares")
        print(f"  -> square {actual_square:.3f} mm, marker {actual_marker:.3f} mm "
              f"(viewer scale {scale:.3f}x vs 1:1)")
        if abs(scale - 1.0) > 0.02:
            print("  The viewer is not showing the image 1:1. That is fine -- use the numbers above,")
            print("  and do not change the zoom, rotation or viewer until you have taken all the")
            print("  photos, because every photo has to share one scale.")
        print("=" * 70)
        square_mm, marker_mm = actual_square, actual_marker

    print("\nNEXT -- intrinsics (the size below barely matters; coverage and sharpness do):")
    print(f"  ../.venv/bin/python scripts/calibrate_intrinsics.py --images <dir> \\")
    print(f"      --square-size {square_mm:.3f} --marker-size {marker_mm:.3f} \\")
    print(f"      --squares-x {squares_x} --squares-y {squares_y}")
    print("  Take 20-30 photos: tilt the tablet 20-45 deg in different directions, vary the")
    print("  distance, and put the board in the CORNERS of the frame as well as the middle --")
    print("  that is what pins down the distortion. Keep it sharp (no motion blur) and keep")
    print("  room lights out of the glass; glare kills marker detection.")
    print("\n  Shoot them with live detection + frame-coverage feedback:")
    print(f"    ../.venv/bin/python scripts/capture_frames.py --out outputs/intrinsics_photos \\")
    print(f"        --charuco --square-size {square_mm:.3f} "
          f"--marker-size {marker_mm:.3f} \\")
    print(f"        --squares-x {squares_x} --squares-y {squares_y}")
    print("\nNEXT -- camera pose. The board is NOT used for this; put the tablet away.")
    print("  With the arm powered (most accurate -- no tape measure in the chain):")
    print("    ../.venv/bin/python scripts/calibrate_camera_touch.py --side right \\")
    print("        --session outputs/touch_session")
    print("  Without moving the robot, from a measured tape rectangle on the table:")
    print("    ../.venv/bin/python scripts/calibrate_table_tape.py \\")
    print("        --side 500 --depth 400 --near-edge-from-table-rear 150")


if __name__ == "__main__":
    main()
