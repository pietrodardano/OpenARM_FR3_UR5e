#!/usr/bin/env python3
"""Poll the demo's vision detector once: no robot involved.

Grabs frames from the HBV camera, runs GroundingDINO + SAM2 (src/vision/vision_detector.py),
prints every object's position in the robot base frame, and saves an annotated frame (boxes,
top-face centroids, base-frame x/y) to check against a ruler on the table:

    python scripts/detect_objects.py --prompt "soda can"
    python scripts/detect_objects.py --prompt "cardboard box" --out outputs/detections.png

Needs OpenCV, torch, GroundingDINO, SAM2, the weights in weights/ (requirements-vision.txt)
and configs/camera_extrinsics.yaml (docs/camera_calibration.md).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    """Runs one detection poll and writes the annotated frame."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt", default=None, help="detector text prompt (default: demo.object_prompt)")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs" / "detections.png")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "task_config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    import cv2

    from src.vision.vision_detector import build_vision_detector

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    prompt = args.prompt or cfg["demo"]["object_prompt"]
    detector = build_vision_detector(cfg, str(PROJECT_ROOT))
    found = detector.detect(prompt)
    print(f"{len(found)} x '{prompt}' (robot base frame, m):")
    for d in found:
        print(f"  ({d.x:+.3f}, {d.y:+.3f})  conf {d.confidence:.2f}  '{d.label}'")

    frame = detector.last_frame.copy()
    for det in detector.last_raw:
        x0, y0, x1, y1 = (int(round(v)) for v in det.box_xyxy)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 200, 255), 2)
        face = getattr(det, "face", None)
        if face is not None:
            u, v = (int(round(c)) for c in face.centroid)
            cv2.circle(frame, (u, v), 5, (0, 0, 255), -1)
            xy = detector._to_base(det, detector.config.top_height(prompt))
            if xy is not None:
                cv2.putText(frame, f"({xy[0]:.3f}, {xy[1]:.3f})", (x0, max(15, y0 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    detector.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), frame)
    print(f"annotated frame: {args.out}")
    return 0 if found else 1


if __name__ == "__main__":
    sys.exit(main())
