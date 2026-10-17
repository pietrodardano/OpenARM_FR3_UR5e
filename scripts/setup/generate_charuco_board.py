#!/usr/bin/env python3
"""
Generate printable A4 ChArUco Boards for OpenCV Camera & Arm Calibration.

Board 1: 5x7 grid, square_length = 38 mm, marker_length = 28 mm (Full A4 coverage)
Board 2: 5x7 grid, square_length = 28 mm, marker_length = 20 mm (Compact A4 with mounting margins)
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw


def generate_charuco_a4(
    squares_x: int,
    squares_y: int,
    square_len_mm: float,
    marker_len_mm: float,
    output_prefix: str,
    dpi: int = 300,
):
    # A4 Dimensions: 210 x 297 mm
    mm_to_px = dpi / 25.4
    a4_w_px = int(round(210.0 * mm_to_px))  # 2480 px
    a4_h_px = int(round(297.0 * mm_to_px))  # 3508 px

    board_w_mm = squares_x * square_len_mm
    board_h_mm = squares_y * square_len_mm

    board_w_px = int(round(board_w_mm * mm_to_px))
    board_h_px = int(round(board_h_mm * mm_to_px))

    # ArUco dictionary: DICT_6X6_250 (standard, highly distinct IDs)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y),
        square_len_mm / 1000.0,
        marker_len_mm / 1000.0,
        aruco_dict,
    )

    # Generate raw board image
    board_img = board.generateImage((board_w_px, board_h_px), marginSize=0, borderBits=1)

    # A4 Canvas
    a4_canvas = np.ones((a4_h_px, a4_w_px), dtype=np.uint8) * 255

    # Center placement
    x_offset = (a4_w_px - board_w_px) // 2
    y_offset = (a4_h_px - board_h_px) // 2

    a4_canvas[y_offset : y_offset + board_h_px, x_offset : x_offset + board_w_px] = board_img

    # Annotate with PIL
    pil_img = Image.fromarray(a4_canvas).convert("RGB")
    draw = ImageDraw.Draw(pil_img)

    header = (
        f"ChArUco Calibration Board ({squares_x}x{squares_y}) | "
        f"Square: {square_len_mm:.1f} mm | Marker: {marker_len_mm:.1f} mm | Dict: DICT_6X6_250"
    )
    footer = "PRINT AT 100% SCALE (NO 'FIT TO PAGE'). Verify dimensions with a physical ruler before calibration."

    draw.text((x_offset, max(40, y_offset - 100)), header, fill=(0, 0, 0))
    draw.text((x_offset, min(a4_h_px - 100, y_offset + board_h_px + 30)), footer, fill=(50, 50, 50))

    # Draw 50mm verification ruler bar
    ruler_50mm_px = int(round(50.0 * mm_to_px))
    ruler_y = min(a4_h_px - 40, y_offset + board_h_px + 80)
    draw.line([(x_offset, ruler_y), (x_offset + ruler_50mm_px, ruler_y)], fill=(0, 0, 0), width=4)
    draw.line([(x_offset, ruler_y - 12), (x_offset, ruler_y + 12)], fill=(0, 0, 0), width=4)
    draw.line([(x_offset + ruler_50mm_px, ruler_y - 12), (x_offset + ruler_50mm_px, ruler_y + 12)], fill=(0, 0, 0), width=4)
    draw.text((x_offset + ruler_50mm_px + 15, ruler_y - 8), "50 mm Verification Ruler Bar", fill=(0, 0, 0))

    png_path = f"{output_prefix}.png"
    pdf_path = f"{output_prefix}.pdf"

    pil_img.save(png_path, "PNG", dpi=(dpi, dpi))
    pil_img.save(pdf_path, "PDF", resolution=dpi)

    print(f"Generated: {png_path} ({board_w_mm:.1f}x{board_h_mm:.1f} mm on A4)")
    print(f"Generated: {pdf_path}")
    return png_path, pdf_path


if __name__ == "__main__":
    # Boards are written next to the other configs, relative to the repository root.
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "configs"))
    os.makedirs(out_dir, exist_ok=True)

    # 1. 5x7 with 38 mm squares, 28 mm markers (Full A4 coverage)
    generate_charuco_a4(
        squares_x=5,
        squares_y=7,
        square_len_mm=38.0,
        marker_len_mm=28.0,
        output_prefix=os.path.join(out_dir, "charuco_5x7_square38mm_marker28mm"),
    )

    # 2. 5x7 with 28 mm squares, 20 mm markers (Extra margins)
    generate_charuco_a4(
        squares_x=5,
        squares_y=7,
        square_len_mm=28.0,
        marker_len_mm=20.0,
        output_prefix=os.path.join(out_dir, "charuco_5x7_square28mm_marker20mm"),
    )
