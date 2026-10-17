"""Capture and colour correction for the HBV egocentric USB camera.

The robot's HBV HD CAMERA enumerates as /dev/video2 (index 2); /dev/video0 is
the laptop's built-in webcam.
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np
from numpy.typing import NDArray

HBV_CAMERA_INDEX = 2


class LatestFrameCamera:
    """Camera reader that always serves the newest frame.

    V4L2 queues frames at the camera's rate; a slower consumer reading them
    FIFO falls progressively further behind (seconds of lag). This drains the
    queue in a background thread and keeps only the most recent frame, so
    latency stays bounded by processing time instead of growing with the queue.
    """

    def __init__(self, index: int = HBV_CAMERA_INDEX, width: int = 640, height: int = 480) -> None:
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {index}")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._lock = threading.Lock()
        self._frame: NDArray[np.uint8] | None = None
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

    def read(self) -> tuple[NDArray[np.uint8] | None, float]:
        """Returns (newest frame, capture timestamp), or (None, 0.0) before the first frame."""
        with self._lock:
            if self._frame is None:
                return None, 0.0
            return self._frame.copy(), self._stamp

    def release(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()

    def __enter__(self) -> "LatestFrameCamera":
        return self

    def __exit__(self, *_exc) -> None:
        self.release()


def correct_colors(frame_bgr: NDArray[np.uint8], saturation: float = 1.35) -> NDArray[np.uint8]:
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


def orient(frame_bgr: NDArray[np.uint8], flip: str = "180") -> NDArray[np.uint8]:
    """Apply the camera's mounting correction. The HBV cam is mounted inverted."""
    if flip == "180":
        return cv2.rotate(frame_bgr, cv2.ROTATE_180)
    if flip == "h":
        return cv2.flip(frame_bgr, 1)
    if flip == "v":
        return cv2.flip(frame_bgr, 0)
    return frame_bgr
