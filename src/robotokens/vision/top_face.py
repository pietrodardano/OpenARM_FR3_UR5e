"""Top-face centroid of an upright object from its whole-object SAM2 mask (ported from main).

The camera looks ~20 deg off vertical, so the silhouette is the top face swept down the image by
the object's projected height. That height is read from the pair of hull edges aligned with the
image vertical, and the top face is recovered as ``hull ∩ (hull shifted up by the height)``.
A rectifying homography K Rx(tilt) K^-1 maps the image to a virtual straight-down view, where
the face centroid is the plain mean (a perspective-correct centre).
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

_MAX_POINTS = 20_000


def rectify_homography(tilt_deg: float, width: int, height: int, focal_px: float) -> NDArray:
    K = np.array([[focal_px, 0.0, width / 2.0], [0.0, focal_px, height / 2.0], [0.0, 0.0, 1.0]])
    th = np.radians(tilt_deg)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(th), -np.sin(th)], [0.0, np.sin(th), np.cos(th)]])
    return K @ rot_x @ np.linalg.inv(K)


def _apply_h(H: NDArray, pts: NDArray) -> NDArray:
    out = np.column_stack((pts, np.ones(len(pts)))) @ H.T
    return out[:, :2] / out[:, 2:3]


def rectangularity(mask: NDArray, H: Optional[NDArray]) -> float:
    """How well the (rectified) outline fills its min-area rectangle, in [0, 1]."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    outline = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    if len(outline) < 4:
        return 0.0
    if H is not None:
        outline = _apply_h(H, outline)
    o32 = outline.astype(np.float32)
    (_, (w, h), _) = cv2.minAreaRect(o32)
    return 0.0 if w * h <= 0 else min(1.0, float(cv2.contourArea(o32)) / float(w * h))


def _silhouette(mask: NDArray) -> Optional[NDArray]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    eps = max(2.0, 0.004 * cv2.arcLength(hull, True))  # tight: keep the short height edges
    return cv2.approxPolyDP(hull, eps, True).reshape(-1, 2).astype(np.float32)


def extrusion_height(polygon: NDArray, tol_deg: float = 15.0) -> float:
    """Projected object height (px): the aligned pair of vertical hull edges with the closest lengths."""
    min_cos = np.cos(np.radians(tol_deg))
    aligned = []
    for i in range(len(polygon)):
        edge = polygon[(i + 1) % len(polygon)] - polygon[i]
        length = float(np.linalg.norm(edge))
        if length >= 4.0 and abs(edge[1] / length) >= min_cos:
            aligned.append(length)
    if len(aligned) < 2:
        return aligned[0] if aligned else 0.0
    aligned.sort()
    return min(((b - a, 0.5 * (a + b)) for a, b in zip(aligned, aligned[1:])),
               key=lambda p: p[0] / max(p[1], 1e-6))[1]


def extract_top_face(mask: NDArray, H: Optional[NDArray], min_height_px: float = 6.0) -> NDArray:
    """Top-face mask from the whole-object mask (the input mask when no side face is visible)."""
    poly = _silhouette(mask)
    if poly is None or len(poly) < 4:
        return mask
    h = extrusion_height(poly)
    if h < min_height_px:
        return mask
    shifted = (poly - np.array([0.0, h], np.float32)).astype(np.float32)
    area, inter = cv2.intersectConvexConvex(poly, shifted)
    if inter is None or area <= 0:
        return mask
    top = np.zeros(mask.shape, np.uint8)
    cv2.fillPoly(top, [np.int32(inter.reshape(-1, 2))], 1)
    top = top.astype(bool)
    if top.sum() < 0.15 * mask.sum() or rectangularity(top, H) < rectangularity(mask, H):
        return mask  # degenerate cut (e.g. bad height on a non-prism)
    return top


def face_centroid(mask: NDArray, H: Optional[NDArray]) -> Optional[Tuple[float, float]]:
    """Perspective-correct centre of a face mask in image pixels (None for a degenerate mask)."""
    ys, xs = np.nonzero(mask)
    if xs.size < 10:
        return None
    if xs.size > _MAX_POINTS:
        idx = np.linspace(0, xs.size - 1, _MAX_POINTS).astype(np.intp)
        xs, ys = xs[idx], ys[idx]
    pts = np.column_stack((xs, ys)).astype(np.float64)
    if H is None:
        c = pts.mean(axis=0)
    else:
        c = _apply_h(np.linalg.inv(H), _apply_h(H, pts).mean(axis=0, keepdims=True))[0]
    return float(c[0]), float(c[1])
