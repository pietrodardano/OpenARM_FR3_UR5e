#!/usr/bin/env python3
"""Track rectangular top faces: GroundingDINO -> SAM2 -> PCA.

For each detected object this reports the top face's centroid, which side is
longer, the two side lengths in pixels, and the face's orientation -- then
tracks the centroid across frames with a stable id.

    python3 scripts/track_rect_faces.py --webcam 2 --prompt "box . book . phone"
    python3 scripts/track_rect_faces.py --image photo.jpg --prompt "box"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vision.camera import LatestFrameCamera, correct_colors, orient  # noqa: E402
from src.vision.object_localizer import Detection, ObjectLocalizer  # noqa: E402
from src.vision.rect_face_pca import CentroidTracker  # noqa: E402

PALETTE = [(0, 255, 0), (255, 128, 0), (0, 180, 255), (255, 0, 200), (255, 255, 0)]


def draw(frame_bgr: np.ndarray, tracked: dict[int, object], detections: list[Detection]) -> np.ndarray:
    out = frame_bgr.copy()

    for det in detections:
        if det.mask is None:
            continue
        colour = PALETTE[hash(det.label) % len(PALETTE)]
        tint = np.zeros_like(out)
        tint[det.mask] = colour
        out = cv2.addWeighted(out, 1.0, tint, 0.35, 0)

    for tid, face in tracked.items():
        colour = PALETTE[tid % len(PALETTE)]
        cx, cy = face.centroid.astype(int)

        corners = face.corners().astype(np.int32)
        cv2.polylines(out, [corners], isClosed=True, color=colour, thickness=2)

        # Long axis solid, short axis dashed-looking (thinner), from the centroid.
        hl = (0.5 * face.long_length * face.long_axis).astype(int)
        hs = (0.5 * face.short_length * face.short_axis).astype(int)
        cv2.arrowedLine(out, (cx - hl[0], cy - hl[1]), (cx + hl[0], cy + hl[1]), colour, 3, tipLength=0.08)
        cv2.line(out, (cx - hs[0], cy - hs[1]), (cx + hs[0], cy + hs[1]), colour, 1)

        cv2.circle(out, (cx, cy), 5, (255, 255, 255), -1)
        cv2.circle(out, (cx, cy), 5, colour, 2)

        shape = "SQUARE" if face.is_square else f"L={face.long_length:.0f} S={face.short_length:.0f}"
        text = f"#{tid} ({cx},{cy}) {shape} {face.angle_deg:+.0f}deg"
        # Anchor above the rectangle's topmost corner so labels of overlapping
        # faces do not land on top of each other.
        tx, ty = corners[corners[:, 1].argmin()]
        ty = max(14, ty - 6)
        cv2.putText(out, text, (int(tx), int(ty)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, (int(tx), int(ty)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
    return out


def report(tracked: dict[int, object], detections: list[Detection]) -> None:
    labels = [d.label for d in detections]
    for (tid, face), label in zip(tracked.items(), labels + [""] * len(tracked)):
        if face.is_square:
            which = "square (orientation ambiguous)"
        else:
            which = f"long side {face.long_length:.0f}px along {face.angle_deg:+.0f}deg, short side {face.short_length:.0f}px"
        print(f"  #{tid} {label}: centroid=({face.centroid[0]:.0f},{face.centroid[1]:.0f})  "
              f"{which}  aspect={face.aspect_ratio:.2f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--webcam", type=int, help="Camera index (robot HBV camera = 2).")
    p.add_argument("--image", help="Run on a single image file instead of the camera.")
    p.add_argument("--prompt", default="box", help="Phrases separated by ' . '.")
    p.add_argument("--sam2", choices=["small", "large"], default="small")
    p.add_argument("--infer-size", type=int, default=480)
    p.add_argument("--tilt-deg", type=float, default=20.0,
                   help="Camera tilt off vertical. Geometry is rectified to a straight-down "
                        "view before PCA, so lengths/angles describe the real top face. 0 disables.")
    p.add_argument("--focal-px", type=float, default=None,
                   help="Focal length in px for the rectification. Defaults to a ~90deg diagonal FOV; "
                        "run scripts/calibrate_intrinsics.py for the real value.")
    p.add_argument("--whole-object", action="store_true",
                   help="Segment the whole object instead of isolating just its top face.")
    p.add_argument("--detect-every", type=int, default=1,
                   help="Run GroundingDINO every N frames (SAM2 still runs every frame). "
                        "N>1 trades detection latency for speed.")
    p.add_argument("--flip", choices=["none", "180", "h", "v"], default="180")
    p.add_argument("--no-color-fix", action="store_true")
    p.add_argument("--saturation", type=float, default=1.35)
    p.add_argument("--out", default="outputs/rect_faces.png", help="Where to write the annotated image.")
    p.add_argument("--headless-frames", type=int, default=0,
                   help="Run N frames printing geometry instead of opening a window.")
    args = p.parse_args()

    if not args.image and args.webcam is None:
        p.error("Pass either --image <path> or --webcam <index>.")

    loc = ObjectLocalizer(sam2_size=args.sam2, infer_size=args.infer_size,
                          tilt_deg=args.tilt_deg, focal_px=args.focal_px,
                          detect_every=args.detect_every)
    tracker = CentroidTracker()

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            p.error(f"Could not read {args.image}")
        dets = loc.locate(frame, args.prompt, top_face_only=not args.whole_object)
        tracked = tracker.update([d.face for d in dets if d.face])
        print(f"{len(dets)} detection(s):")
        report(tracked, dets)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.out, draw(frame, tracked, dets))
        print(f"Wrote {args.out}")
        return

    cam = LatestFrameCamera(args.webcam)
    n = 0
    try:
        while True:
            frame, stamp = cam.read()
            if frame is None:
                time.sleep(0.01)
                continue

            t0 = time.perf_counter()
            frame = orient(frame, args.flip)
            if not args.no_color_fix:
                frame = correct_colors(frame, args.saturation)
            dets = loc.locate(frame, args.prompt, top_face_only=not args.whole_object)
            tracked = tracker.update([d.face for d in dets if d.face])
            dt = time.perf_counter() - t0
            n += 1

            if args.headless_frames:
                print(f"frame {n}: {dt*1000:.0f}ms  age={(time.time()-stamp)*1000:.0f}ms  {len(dets)} det(s)")
                report(tracked, dets)
                if n >= args.headless_frames:
                    break
                continue

            out = draw(frame, tracked, dets)
            cv2.putText(out, f"{1/max(dt,1e-6):.1f} FPS", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("rect face PCA", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.release()
        if not args.headless_frames:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
