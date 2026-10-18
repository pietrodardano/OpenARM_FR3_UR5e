"""
The demo's ``ObjectDetector``: RGB camera -> GroundingDINO + SAM2 -> object positions on the table.

Implements :class:`src.task.interfaces.ObjectDetector` on the vision pipeline of
``src/vision/object_localizer.py`` and the camera calibration of
``src/vision/table_calibration.py``:

1. Grab ``frames`` camera frames (rotated like the calibration, colour-corrected).
2. :meth:`ObjectLocalizer.locate` per frame: GroundingDINO boxes from the text prompt, SAM2
   masks, PCA fit of each object's top face.
3. Pixel -> robot base frame:
   - The top-face centroid is intersected with the plane at ``table_z + top height`` of the object
     (:meth:`TableCalibration.pixel_to_plane`). The table homography would place a point 12 cm
     above the table too far from the camera.
   - Without a face fit, the bottom-centre of the box goes through the table homography.
4. Detections of all frames are merged within ``merge_radius_m``. Objects seen in fewer than
   ``min_frame_fraction`` of the frames are dropped, and so is anything outside ``workspace``.

Vision says *where* the objects are, never which one is heavier. torch, GroundingDINO and SAM2
are only imported when the real localizer is built (:func:`build_vision_detector`), so the
geometry and tests run without them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from ..task.interfaces import ObjectDetection
from .table_calibration import TableCalibration

logger = logging.getLogger(__name__)


@dataclass
class VisionDetectorConfig:
    """``vision.object_detection`` section of task_config.yaml."""

    calibration_path: str = "configs/camera_extrinsics.yaml"
    camera_index: int = 0
    frames: int = 3
    merge_radius_m: float = 0.04
    min_frame_fraction: float = 0.5
    color_correct: bool = True
    object_top_height_m: Dict[str, float] = field(default_factory=lambda: {"can": 0.12})
    default_top_height_m: float = 0.12
    workspace: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None  # ((x_min, x_max), (y_min, y_max))
    localizer: Dict[str, Any] = field(default_factory=dict)  # ObjectLocalizer kwargs

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "VisionDetectorConfig":
        """Builds the config from the full task_config.yaml."""
        v = cfg.get("vision", {}) or {}
        od = v.get("object_detection", {}) or {}
        ws = od.get("workspace")
        d = cls()
        return cls(
            calibration_path=str(od.get("calibration_path", d.calibration_path)),
            camera_index=int(v.get("device_index", d.camera_index)),  # same camera the calibration used
            frames=int(od.get("frames", d.frames)),
            merge_radius_m=float(od.get("merge_radius_m", d.merge_radius_m)),
            min_frame_fraction=float(od.get("min_frame_fraction", d.min_frame_fraction)),
            color_correct=bool(od.get("color_correct", d.color_correct)),
            object_top_height_m={str(k): float(h) for k, h in (od.get("object_top_height_m") or d.object_top_height_m).items()},
            default_top_height_m=float(od.get("default_top_height_m", d.default_top_height_m)),
            workspace=None if ws is None else ((float(ws["x"][0]), float(ws["x"][1])), (float(ws["y"][0]), float(ws["y"][1]))),
            localizer=dict(od.get("localizer", {}) or {}),
        )

    def top_height(self, prompt: str) -> float:
        """Top-face height above the table for the object named in ``prompt``."""
        low = prompt.lower()
        for key, h in self.object_top_height_m.items():
            if key in low:
                return h
        return self.default_top_height_m


@dataclass
class _Hit:
    x: float
    y: float
    score: float
    label: str


class VisionObjectDetector:
    """``ObjectDetector`` on the RGB camera with GroundingDINO + SAM2."""

    def __init__(self, config: VisionDetectorConfig, calibration: TableCalibration,
                 frame_source: Callable[[], NDArray[np.uint8]], localizer: Any,
                 close: Optional[Callable[[], None]] = None) -> None:
        """Args:
            config: Detection settings.
            calibration: Camera intrinsics, pose and table homography (robot base frame).
            frame_source: Returns the newest BGR frame, already rotated like the calibration.
            localizer: Object with ``locate(frame_bgr, prompt) -> [Detection]`` (``ObjectLocalizer``).
            close: Releases the camera (called by :meth:`close`).
        """
        self.config = config
        self.calib = calibration
        self.frame_source = frame_source
        self.localizer = localizer
        self.last_frame: Optional[NDArray[np.uint8]] = None
        self.last_raw: List[Any] = []
        self._close = close

    def close(self) -> None:
        """Releases the camera. Safe to call twice."""
        if self._close is not None:
            self._close()
            self._close = None

    def _to_base(self, det: Any, top_height: float) -> Optional[Tuple[float, float]]:
        """Base-frame (x, y) of one localizer detection, or None if it cannot be mapped."""
        face = getattr(det, "face", None)
        try:
            if face is not None and self.calib.camera_position is not None:
                u, v = (float(c) for c in face.centroid)
                return self.calib.pixel_to_plane(u, v, self.calib.table_z + top_height)
            x0, _y0, x1, y1 = (float(c) for c in det.box_xyxy)  # bottom-centre of the box touches the table
            return tuple(self.calib.pixel_to_table(0.5 * (x0 + x1), y1))  # type: ignore[return-value]
        except ValueError as exc:
            logger.warning("detection dropped: %s", exc)
            return None

    def _in_workspace(self, x: float, y: float) -> bool:
        if self.config.workspace is None:
            return True
        (x0, x1), (y0, y1) = self.config.workspace
        return x0 <= x <= x1 and y0 <= y <= y1

    def detect(self, prompt: str) -> List[ObjectDetection]:
        """Objects matching ``prompt`` on the table, base frame, fused over ``frames`` frames."""
        h_top = self.config.top_height(prompt)
        hits: List[List[_Hit]] = []
        n_frames = max(1, self.config.frames)
        for _ in range(n_frames):
            frame = self.frame_source()
            self.last_frame = frame
            raw = self.localizer.locate(frame, prompt)
            self.last_raw = raw
            frame_hits = []
            for det in raw:
                xy = self._to_base(det, h_top)
                if xy is not None and self._in_workspace(*xy):
                    frame_hits.append(_Hit(xy[0], xy[1], float(det.score), str(det.label)))
            hits.append(frame_hits)
        return self._fuse(hits, n_frames)

    def _fuse(self, hits: Sequence[Sequence[_Hit]], n_frames: int) -> List[ObjectDetection]:
        """Greedy clustering of per-frame hits; one detection per cluster seen often enough."""
        clusters: List[List[_Hit]] = []
        for frame_hits in hits:
            for h in sorted(frame_hits, key=lambda h: -h.score):
                best = None
                for c in clusters:
                    cx, cy = np.mean([m.x for m in c]), np.mean([m.y for m in c])
                    if np.hypot(h.x - cx, h.y - cy) <= self.config.merge_radius_m:
                        best = c
                        break
                if best is None:
                    clusters.append([h])
                else:
                    best.append(h)
        need = max(1, int(np.ceil(self.config.min_frame_fraction * n_frames)))
        out = []
        for c in clusters:
            if len(c) < need:
                continue
            out.append(ObjectDetection(label=c[0].label, x=float(np.median([m.x for m in c])),
                                       y=float(np.median([m.y for m in c])),
                                       confidence=float(np.mean([m.score for m in c]))))
        return sorted(out, key=lambda d: d.y)


def build_vision_detector(cfg: Mapping[str, Any], repo_root: str = ".") -> VisionObjectDetector:
    """The real detector: HBV camera + GroundingDINO + SAM2 (imports torch). Raises if not set up."""
    import os

    vc = VisionDetectorConfig.from_config(cfg)
    path = vc.calibration_path if os.path.isabs(vc.calibration_path) else os.path.join(repo_root, vc.calibration_path)
    calib = TableCalibration.load(path)  # FileNotFoundError names the calibration script to run
    from .camera import LatestFrameCamera, correct_colors, orient  # needs OpenCV

    try:
        from .object_localizer import ObjectLocalizer  # needs torch, GroundingDINO, SAM2, weights/
    except ImportError as exc:
        raise RuntimeError(f"the vision pipeline needs torch, GroundingDINO and SAM2 ({exc}). Install them on "
                           "the vision machine: see requirements-vision.txt. Or pass --objects / --detector manual.") from exc
    for w in calib.warnings:
        logger.warning("camera calibration: %s", w)
    if calib.intrinsics_estimated:
        logger.warning("camera calibration has no measured intrinsics: object positions are rough (~1-2 cm)")
    width, height = calib.image_size
    camera = LatestFrameCamera(index=vc.camera_index, width=width, height=height)

    def frame_source() -> NDArray[np.uint8]:
        import time

        for _ in range(200):
            frame, _stamp = camera.read()
            if frame is not None:
                break
            time.sleep(0.01)
        else:
            raise RuntimeError(f"camera {vc.camera_index} delivers no frames")
        if (frame.shape[1], frame.shape[0]) != (width, height):
            raise RuntimeError(f"camera frame {frame.shape[1]}x{frame.shape[0]} differs from the calibration "
                               f"{width}x{height}: re-calibrate or fix the capture size")
        frame = orient(frame, "180") if calib.rotate_180 else frame
        return correct_colors(frame) if vc.color_correct else frame

    kwargs = dict(vc.localizer)
    kwargs.setdefault("focal_px", float(calib.camera_matrix[0, 0]))  # measured lens, not the 90 deg guess
    try:
        localizer = ObjectLocalizer(**kwargs)
    except Exception:
        camera.release()
        raise
    return VisionObjectDetector(vc, calib, frame_source, localizer, close=camera.release)
