"""The demo's :class:`Detector` on the camera: localizer hits -> base-frame objects (FR2).

Per frame, each hit's top-face centroid is intersected with the plane at
``table_z + top height`` of that hit's class, read from its label (a top face is not on the table:
the table plane would put it too far from the camera). Hits of ``frames`` frames are clustered
within ``merge_radius_m``; clusters seen in fewer than ``min_frame_fraction`` of the frames and
anything outside ``workspace`` are dropped. Vision says *where*, never which object is heavier.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ..config import Config, VisionConfig
from ..task.skills import Detection
from .calibration import CameraModel

logger = logging.getLogger(__name__)


class VisionDetector:
    def __init__(self, cfg: VisionConfig, camera_model: CameraModel, table_z: float,
                 frame_source: Callable[[], NDArray], localizer: Any, close: Optional[Callable[[], None]] = None) -> None:
        self.cfg = cfg
        self.cam = camera_model
        self.table_z = table_z
        self.frame_source = frame_source
        self.localizer = localizer
        self.last_frame: Optional[NDArray] = None
        self.last_hits: List[Any] = []
        self._close = close

    def close(self) -> None:
        if self._close is not None:
            self._close()
            self._close = None

    def _in_workspace(self, x: float, y: float) -> bool:
        ws = self.cfg.workspace
        return ws.x[0] <= x <= ws.x[1] and ws.y[0] <= y <= ws.y[1]

    def detect(self, prompt: str) -> List[Detection]:
        """Objects matching ``prompt`` (several classes: "soda can . cardboard box")."""
        hits: List[List[Tuple[float, float, float, str]]] = []
        for _ in range(max(1, self.cfg.frames)):
            self.last_frame = self.frame_source()
            self.last_hits = self.localizer.locate(self.last_frame, prompt)
            frame_hits = []
            for h in self.last_hits:
                if h.centroid is None:
                    continue
                z_top = self.table_z + self.cfg.top_height(str(h.label) or prompt)  # this hit's class
                try:
                    x, y = self.cam.pixel_to_plane(*h.centroid, z_top)
                except ValueError as exc:
                    logger.warning("hit dropped: %s", exc)
                    continue
                if self._in_workspace(x, y):
                    frame_hits.append((x, y, float(h.score), str(h.label)))
            hits.append(frame_hits)
        return self._fuse(hits)

    def _fuse(self, hits: List[List[Tuple[float, float, float, str]]]) -> List[Detection]:
        clusters: List[list] = []
        for frame_hits in hits:
            for h in sorted(frame_hits, key=lambda h: -h[2]):
                for c in clusters:
                    if np.hypot(h[0] - np.mean([m[0] for m in c]), h[1] - np.mean([m[1] for m in c])) <= self.cfg.merge_radius_m:
                        c.append(h)
                        break
                else:
                    clusters.append([h])
        need = max(1, int(np.ceil(self.cfg.min_frame_fraction * len(hits))))
        out = [Detection(x=float(np.median([m[0] for m in c])), y=float(np.median([m[1] for m in c])),
                         confidence=float(np.mean([m[2] for m in c])), label=c[0][3])
               for c in clusters if len(c) >= need]
        return sorted(out, key=lambda d: d.y)


def build_vision_detector(cfg: Config) -> VisionDetector:
    """Camera + GroundingDINO + SAM2 (imports torch; raises if the vision stack is missing)."""
    from .calibration import load_camera_model
    from .camera import Camera
    from .localizer import Localizer

    cam_model = load_camera_model(cfg)
    camera = Camera(cfg.camera)
    try:
        localizer = Localizer(cfg.vision, focal_px=float(cam_model.K[0, 0]))
    except Exception:
        camera.close()
        raise
    return VisionDetector(cfg.vision, cam_model, cfg.table_z, camera.read, localizer, close=camera.close)
