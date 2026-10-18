#!/usr/bin/env python3
"""
GroundingDINO text-prompted object detection. No camera calibration needed --
this only produces 2D pixel boxes.

Usage (single image file):
    python3 scripts/test_grounding_dino.py --image path/to/photo.jpg --prompt "cube . apple . plate"

Usage (live camera, press 'q' to quit):
    python3 scripts/test_grounding_dino.py --webcam 2 --prompt "cube . apple . plate"

The robot's HBV HD CAMERA is /dev/video2 -> --webcam 2. The laptop's built-in
ASUS webcam is /dev/video0.

The prompt format is GroundingDINO's convention: lowercase phrases separated
by " . ".
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time

import cv2
import numpy as np
import torch
from PIL import Image

from groundingdino.util.inference import load_model, load_image, predict, annotate
import groundingdino.datasets.transforms as T

# GroundingDINO's deformable-attention op always calls its compiled CUDA
# extension when tensors are on GPU, even if that extension failed to build
# (no CUDA toolkit/nvcc here to compile it) -- it never falls back on its
# own. Force it onto the pure-PyTorch implementation that already ships in
# the same module but is otherwise unreachable on a CUDA tensor.
import groundingdino.models.GroundingDINO.ms_deform_attn as _msda


class _PureTorchDeformAttn:
    @staticmethod
    def apply(value, spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step):
        del level_start_index, im2col_step
        return _msda.multi_scale_deformable_attn_pytorch(value, spatial_shapes, sampling_locations, attention_weights)


_msda.MultiScaleDeformableAttnFunction = _PureTorchDeformAttn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(REPO_ROOT, "weights", "GroundingDINO_SwinT_OGC.py")
WEIGHTS_PATH = os.path.join(REPO_ROOT, "weights", "groundingdino_swint_ogc.pth")

BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25


class LatestFrameCamera:
    """Camera reader that always serves the newest frame.

    V4L2 queues frames at the camera's rate; a slower consumer reading them
    FIFO falls progressively further behind (seconds of lag). This drains in a
    background thread and keeps only the most recent frame, so latency stays
    bounded by inference time instead of growing with the queue.
    """

    def __init__(self, index: int, width: int = 640, height: int = 480) -> None:
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {index}")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._stamp = 0.0
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self._lock:
                self._frame = frame
                self._stamp = time.time()

    def read(self) -> tuple[np.ndarray | None, float]:
        with self._lock:
            if self._frame is None:
                return None, 0.0
            return self._frame.copy(), self._stamp

    def release(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()


def correct_colors(frame_bgr: np.ndarray, saturation: float) -> np.ndarray:
    """Gray-world white balance plus a saturation lift.

    The HBV sensor's auto-WB leaves a blue cast (B channel ~18 levels hot) and
    washed-out colour (mean saturation ~47/255), which makes reds read as
    purple. Equalising the channel means neutralises the cast.
    """
    means = frame_bgr.reshape(-1, 3).mean(axis=0)
    gain = means.mean() / np.maximum(means, 1e-6)
    out = np.clip(frame_bgr.astype(np.float32) * gain, 0, 255).astype(np.uint8)

    if saturation != 1.0:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out


def orient(frame_bgr: np.ndarray, flip: str) -> np.ndarray:
    if flip == "180":
        return cv2.rotate(frame_bgr, cv2.ROTATE_180)
    if flip == "h":
        return cv2.flip(frame_bgr, 1)
    if flip == "v":
        return cv2.flip(frame_bgr, 0)
    return frame_bgr


def build_transform(infer_size: int) -> T.Compose:
    return T.Compose(
        [
            T.RandomResize([infer_size], max_size=int(infer_size * 4 / 3)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def run_predict(model, image_tensor, prompt: str, half: bool):
    if half:
        with torch.autocast("cuda", dtype=torch.float16):
            return predict(
                model=model, image=image_tensor, caption=prompt,
                box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD,
            )
    return predict(
        model=model, image=image_tensor, caption=prompt,
        box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD,
    )


def run_on_image(model, image_path: str, prompt: str, out_path: str, half: bool) -> None:
    image_source, image = load_image(image_path)
    boxes, logits, phrases = run_predict(model, image, prompt, half)
    print(f"Found {len(phrases)} detections:")
    for phrase, score in zip(phrases, logits):
        print(f"  {phrase}: {float(score):.2f}")

    annotated = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    cv2.imwrite(out_path, annotated)
    print(f"Wrote annotated image to {out_path}")


def run_on_webcam(model, cam_index: int, prompt: str, infer_size: int, half: bool,
                  flip: str, saturation: float, fix_colors: bool, headless_frames: int = 0) -> None:
    transform = build_transform(infer_size)
    cam = LatestFrameCamera(cam_index)
    print("Press 'q' to quit." if not headless_frames else f"Headless timing run: {headless_frames} frames")

    fps_avg, lag_avg, n = 0.0, 0.0, 0
    try:
        while True:
            frame_bgr, stamp = cam.read()
            if frame_bgr is None:
                time.sleep(0.01)
                continue

            t0 = time.perf_counter()
            frame_bgr = orient(frame_bgr, flip)
            if fix_colors:
                frame_bgr = correct_colors(frame_bgr, saturation)
            image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image_transformed, _ = transform(Image.fromarray(image_rgb), None)
            boxes, logits, phrases = run_predict(model, image_transformed, prompt, half)
            infer_s = time.perf_counter() - t0

            # End-to-end age of what we're about to display: capture -> now.
            lag_s = time.time() - stamp
            n += 1
            fps_avg += (1.0 / max(infer_s, 1e-6) - fps_avg) / n
            lag_avg += (lag_s - lag_avg) / n

            if headless_frames:
                print(f"  frame {n}: infer={infer_s*1000:.0f}ms  frame_age={lag_s*1000:.0f}ms  dets={len(phrases)}")
                if n >= headless_frames:
                    break
                continue

            # annotate() already returns BGR (it converts internally), so it
            # goes straight to imshow -- converting again swaps red and blue.
            annotated = annotate(image_source=image_rgb, boxes=boxes, logits=logits, phrases=phrases)
            cv2.putText(
                annotated, f"{1.0/max(infer_s,1e-6):.1f} FPS  lag {lag_s*1000:.0f}ms",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )
            cv2.imshow("GroundingDINO", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.release()
        if not headless_frames:
            cv2.destroyAllWindows()
        if n:
            print(f"\nmean inference FPS: {fps_avg:.1f}   mean displayed-frame age: {lag_avg*1000:.0f}ms")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", help="Path to a single image file to run detection on.")
    parser.add_argument("--webcam", type=int, help="Camera device index (robot HBV camera = 2).")
    parser.add_argument("--prompt", default="cube . apple . plate", help="Phrases separated by ' . '.")
    parser.add_argument("--out", default="outputs/grounding_dino_test.jpg")
    parser.add_argument("--infer-size", type=int, default=480,
                        help="Short-side resize for inference. 480 matches the camera's native height; "
                             "400 is faster, 800 is the upstream default and upscales 480p input for no gain.")
    parser.add_argument("--no-half", action="store_true", help="Disable fp16 autocast (slower, marginally more precise).")
    parser.add_argument("--flip", choices=["none", "180", "h", "v"], default="180",
                        help="Camera orientation fix. The HBV cam is mounted inverted, so 180 is the default.")
    parser.add_argument("--no-color-fix", action="store_true",
                        help="Disable gray-world white balance (the sensor has a blue cast that makes reds look purple).")
    parser.add_argument("--saturation", type=float, default=1.35,
                        help="Saturation multiplier applied after white balance. 1.0 = unchanged.")
    parser.add_argument("--headless-frames", type=int, default=0,
                        help="Run N frames printing timings instead of opening a window (for benchmarking over SSH).")
    args = parser.parse_args()

    if not args.image and args.webcam is None:
        parser.error("Pass either --image <path> or --webcam <index>.")

    for path, what in ((CONFIG_PATH, "config"), (WEIGHTS_PATH, "weights")):
        if not os.path.exists(path):
            print(f"Missing {what}: {path}")
            sys.exit(1)

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    half = torch.cuda.is_available() and not args.no_half

    model = load_model(CONFIG_PATH, WEIGHTS_PATH)
    # load_model() leaves the model on CPU; predict() would otherwise move all
    # 172M params across on its first call mid-loop.
    if torch.cuda.is_available():
        model = model.to("cuda")

    if args.image:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        run_on_image(model, args.image, args.prompt, args.out, half)
    else:
        run_on_webcam(model, args.webcam, args.prompt, args.infer_size, half,
                      args.flip, args.saturation, not args.no_color_fix, args.headless_frames)


if __name__ == "__main__":
    main()
