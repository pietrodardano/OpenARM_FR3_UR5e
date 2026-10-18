"""FR2: localizer hits -> base-frame objects (fusion, frame fraction, workspace), no torch."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from robotokens.vision.detector import VisionDetector
from test_calibration import synthetic_camera


class StubLocalizer:
    """Returns the projected top-face centroids of the given base-frame points, per frame."""

    def __init__(self, cam, frames_points):
        self.cam = cam
        self.frames = list(frames_points)

    def locate(self, _frame, _prompt):
        pts = self.frames.pop(0)
        if not pts:
            return []
        uv = self.cam.project(np.asarray(pts))
        return [SimpleNamespace(label="soda can", score=0.8, centroid=tuple(c)) for c in uv]


def detector(cfg, frames_points):
    cam = synthetic_camera()
    return VisionDetector(cfg.vision, cam, cfg.table_z, lambda: np.zeros((720, 1280, 3), np.uint8),
                          StubLocalizer(cam, frames_points))


def top(x, y, cfg):
    return [x, y, cfg.table_z + cfg.vision.object_top_height_m["can"]]


def test_two_cans_fused_and_sorted_by_y(cfg):
    a, b = top(0.21, 0.27, cfg), top(0.22, 0.13, cfg)
    dets = detector(cfg, [[a, b], [a, b], [b]]).detect("soda can")
    assert len(dets) == 2
    np.testing.assert_allclose([dets[0].x, dets[0].y], [0.22, 0.13], atol=1e-6)  # sorted by y
    np.testing.assert_allclose([dets[1].x, dets[1].y], [0.21, 0.27], atol=1e-6)
    assert dets[0].confidence == pytest.approx(0.8)


def test_rare_detection_dropped(cfg):
    a, ghost = top(0.21, 0.27, cfg), top(0.35, 0.05, cfg)
    dets = detector(cfg, [[a, ghost], [a], [a]]).detect("soda can")  # ghost in 1 of 3 frames
    assert len(dets) == 1


def test_outside_workspace_dropped(cfg):
    far = top(0.60, 0.20, cfg)  # workspace x <= 0.45
    assert detector(cfg, [[far]] * 3).detect("soda can") == []


class LabelledStub:
    """Hits with their own labels: a can and a box top projected at their own heights."""

    def __init__(self, cam, items):
        self.cam, self.items = cam, items

    def locate(self, _frame, _prompt):
        uv = self.cam.project(np.array([p for p, _ in self.items]))
        return [SimpleNamespace(label=lab, score=0.7, centroid=tuple(c)) for c, (_, lab) in zip(uv, self.items)]


def test_top_height_follows_each_detection_label(cfg):
    """FR2: a two-class prompt ("aluminum can . box") maps each hit at its own top height."""
    cam = synthetic_camera()
    h = cfg.vision.object_top_height_m
    items = [([0.22, 0.13, cfg.table_z + h["box"]], "box"),
             ([0.21, 0.27, cfg.table_z + h["can"]], "aluminum")]
    det = VisionDetector(cfg.vision, cam, cfg.table_z, lambda: None, LabelledStub(cam, items))
    out = det.detect("aluminum can . box")
    np.testing.assert_allclose([[o.x, o.y] for o in out], [[0.22, 0.13], [0.21, 0.27]], atol=1e-6)
    assert [o.label for o in out] == ["box", "aluminum"]
