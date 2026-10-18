"""HBV USB camera: newest-frame reader, mounting rotation, colour correction."""

from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np
from numpy.typing import NDArray

from ..config import CameraConfig


class Camera:
    """Serves the newest frame, already rotated and colour-corrected like the calibration.

    V4L2 queues frames; a slow consumer reading them FIFO falls seconds behind. A background
    thread drains the queue and keeps only the latest frame. ``CAP_V4L2`` is forced: without it
    OpenCV may fall back to another backend and silently open a different camera. MJPG is
    requested because the HBV module only offers 1280x720 in MJPG.
    """

    def __init__(self, cfg: CameraConfig) -> None:
        self.cfg = cfg
        self.cap = cv2.VideoCapture(cfg.device_index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera /dev/video{cfg.device_index} (v4l2-ctl --list-devices)")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._lock = threading.Lock()
        self._frame: Optional[NDArray] = None
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self._lock:
                self._frame = frame

    def read(self, timeout_s: float = 2.0) -> NDArray:
        """Newest frame (BGR), rotated / colour-corrected per config."""
        t_end = time.monotonic() + timeout_s
        while True:
            with self._lock:
                frame = None if self._frame is None else self._frame.copy()
            if frame is not None:
                break
            if time.monotonic() > t_end:
                raise RuntimeError(f"camera {self.cfg.device_index} delivers no frames")
            time.sleep(0.01)
        if (frame.shape[1], frame.shape[0]) != (self.cfg.width, self.cfg.height):
            raise RuntimeError(f"camera delivers {frame.shape[1]}x{frame.shape[0]}, config says "
                               f"{self.cfg.width}x{self.cfg.height} (the intrinsics depend on the mode)")
        if self.cfg.rotate_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        return correct_colors(frame) if self.cfg.color_correct else frame

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()


def correct_colors(frame_bgr: NDArray, saturation: float = 1.35) -> NDArray:
    """Gray-world white balance + saturation lift (the HBV sensor has a blue cast)."""
    means = frame_bgr.reshape(-1, 3).mean(axis=0)
    out = np.clip(frame_bgr.astype(np.float32) * (means.mean() / np.maximum(means, 1e-6)), 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
