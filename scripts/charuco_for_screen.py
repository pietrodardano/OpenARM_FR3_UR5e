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

Size only matters when the same board is used to measure the camera's position
(scripts/calibrate_table_tape.py --mode board), because that is where metric
scale enters. Hence the PPI route, plus a printed check you can confirm with a
ruler.

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


def parse_screen(text: str) -> tuple[int, int]:
    try:
        width, height = (int(v) for v in text.lower().replace(" ", "").split("x"))
    except ValueError:
        raise SystemExit(f'--screen must look like 2360x1640, got "{text}"')
    if width < 200 or height < 200:
        raise SystemExit("--screen looks too small to be a screen resolution")
    return width, height


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
    parser.add_argument("--margin-squares", type=float, default=0.6,
                        help="White quiet zone around the board, in squares (default 0.6). ArUco "
                             "needs white around the outer markers to detect them.")
    parser.add_argument("--landscape", action="store_true",
                        help="Lay the board out for a landscape screen (swaps which screen axis the "
                             "board's long side follows).")
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

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_mm / 1000.0,
                                   marker_mm / 1000.0, aruco_dict)
    board_img = board.generateImage((board_w_px, board_h_px), marginSize=0, borderBits=1)

    canvas = np.full((screen_h, screen_w), 255, np.uint8)
    x0 = (screen_w - board_w_px) // 2
    y0 = (screen_h - board_h_px) // 2
    canvas[y0:y0 + board_h_px, x0:x0 + board_w_px] = board_img

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cv2.imwrite(args.out, canvas)

    print(f"\nWrote {args.out}  ({screen_w}x{screen_h} px, exactly your screen)")
    print("=" * 70)
    print(f"Board          : {squares_x}x{squares_y} squares, DICT_6X6_250")
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
    print("\nNEXT -- intrinsics (the size below barely matters; coverage and sharpness do):")
    print(f"  ../.venv/bin/python scripts/calibrate_intrinsics.py --images <dir> \\")
    print(f"      --square-size {square_mm:.3f} --marker-size {marker_mm:.3f} \\")
    print(f"      --squares-x {squares_x} --squares-y {squares_y}")
    print("  Take 20-30 photos: tilt the tablet 20-45 deg in different directions, vary the")
    print("  distance, and put the board in the CORNERS of the frame as well as the middle --")
    print("  that is what pins down the distortion. Keep it sharp (no motion blur) and keep")
    print("  room lights out of the glass; glare kills marker detection.")
    print("\nNEXT -- camera pose, with the same board lying flat on the table:")
    print(f"  ../.venv/bin/python scripts/calibrate_table_tape.py --mode board \\")
    print(f"      --square-size {square_mm:.3f} --marker-size {marker_mm:.3f} \\")
    print(f"      --squares-x {squares_x} --squares-y {squares_y} --board-near-left 0.300,0.100")
    print("  Here the size DOES matter, so do the ruler check first.")


if __name__ == "__main__":
    main()
